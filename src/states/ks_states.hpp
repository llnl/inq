/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__STATES__KS_STATES
#define INQ__STATES__KS_STATES

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/complex.hpp>
#include <gpu/array.hpp>
#include <operations/sum.hpp>
#include <parallel/partition.hpp>
#include <states/spin_config.hpp>

namespace inq {
namespace states {

class ks_states {

	spin_config spin_;
	double num_electrons_;
	int nstates_;
	int num_spin_indices_;
	int num_density_components_;
	int spinor_dim_;
	double temperature_;
	double max_occ_;
	int nkpoints_;

public:

	typedef complex coeff_type;
    
	ks_states(const spin_config spin, const double nelectrons, const int extra_states = 0, double temperature = 0.0, int nkpoints = 1):
		spin_(spin),
		temperature_(temperature),
		nkpoints_(nkpoints)
	{
		
		max_occ_ = 1.0;
		if(spin == spin_config::UNPOLARIZED) max_occ_ = 2.0;
			
		if(spin == spin_config::NON_COLLINEAR){
			nstates_ = ceil(nelectrons);
		} else {
			nstates_ = ceil(0.5*nelectrons);
		}

		nstates_ += extra_states;
			
		num_spin_indices_ = 1;
		if(spin == spin_config::POLARIZED) num_spin_indices_ = 2;

		num_density_components_ = 1;
		if(spin == spin_config::POLARIZED) num_density_components_ = 2;
		if(spin == spin_config::NON_COLLINEAR) num_density_components_ = 4;		

		spinor_dim_ = 1;
		if(spin == spin_config::NON_COLLINEAR) spinor_dim_ = 2;
		
		num_electrons_ = nelectrons;
	}

	int num_states() const {
		return nstates_;
	}

	int num_spin_indices() const {
		return num_spin_indices_;
	}

	int num_density_components() const {
		return num_density_components_;
	}

	auto spinor_dim() const {
		return spinor_dim_;
	}
	
	template<class OStream>
	friend OStream& operator<<(OStream& out, ks_states const& self){
		out << "Kohn-Sham representation:" << std::endl;
		out << "  Number of electrons      = " << self.num_electrons_ << std::endl;
		out << "  Spin                     = " << self.spin_ << std::endl;
		out << "  States per spin/kpoint   = " << self.num_states() << std::endl;
		out << std::endl;
		return out;
	}
	
	auto num_electrons() const {
		return num_electrons_;
	}

	static auto smear_function(double xx)  {

		double const maxarg = 200.0;
		
		if(xx > maxarg){
			return 1.0;
		} else if(xx > -maxarg) {
			return 1.0/(1.0 + exp(-xx));
		} else {
			return 0.0;
		}
		
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <class CommType, typename EigType, typename KwType, typename FunctionType>
	auto get_efermi(CommType & comm, double nelec, EigType const & eig, KwType const & kw, FunctionType function){
			
		int const nitmax = 200;
		double const tol = 1e-10;
		auto drange = sqrt(-log(tol*0.01));

		double emin = std::numeric_limits<double>::max();
		double emax = std::numeric_limits<double>::lowest();
		
		for(long ie = 0; ie < eig.size(); ie++){
			emin = std::min(emin, eig[ie]);
			emax = std::max(emax, eig[ie]);
		}
		
		emin = comm.all_reduce_value(emin, boost::mpi3::min<>{}) - drange;
		emax = comm.all_reduce_value(emax, boost::mpi3::max<>{}) + drange;
		
		double efermi;
		
		for(int iter = 0; iter < nitmax; iter++){
			efermi = 0.5*(emin + emax);

			double sumq = 0.0;
			for(long ie = 0; ie < eig.size(); ie++){
				sumq = sumq + max_occ_*kw[ie]*function(efermi, real(eig[ie]));
			}
			comm.all_reduce_in_place_n(&sumq, 1, std::plus<>{});
			
			if(fabs(sumq - nelec) <= tol) break;
			if(sumq <= nelec) emin = efermi;
			if(sumq >= nelec) emax = efermi;
		}
		return efermi;		
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <class CommType, typename EigenvalType, typename KpinWeightsType, typename OccsType>
	double update_occupations(CommType & comm, EigenvalType const & eigenval, KpinWeightsType const & kpin_weights, OccsType & occs) {

		assert(sizes(eigenval) == sizes(occs));
		
		auto feig = eigenval.flatted();
		auto focc = occs.flatted();

		auto kweights = gpu::array<double, 2>(occs.extensions());
		gpu::run((~kweights).size(), kweights.size(),
						 [kw = begin(kweights), kp = begin(kpin_weights)] GPU_LAMBDA (auto ii, auto jj){
							 kw[jj][ii] = kp[jj];
						 });
		auto fkw = kweights.flatted();

		return get_occupations(num_electrons_, comm, feig, fkw, focc);
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <class CommType, typename EigenvalType, typename KpinWeightsType, typename OccsType>
	double get_occupations(double const electrons_available, CommType & comm, EigenvalType const & feig, KpinWeightsType const & fkw, OccsType & focc){

		double efermi;

		if(temperature_ == 0.0){

			auto func = [] (auto efermi, auto eig){
				return (eig <= efermi) ? 1.0 :  0.0;
			};

			auto nelec = ceil(electrons_available/max_occ_)*max_occ_; 

			efermi = get_efermi(comm, nelec, feig, fkw, func);

			double homo = std::numeric_limits<double>::lowest();
			int homoloc = 0;
			for(long ie = 0; ie < feig.size(); ie++){
				focc[ie] = max_occ_*fkw[ie]*func(efermi, feig[ie]);
				if(func(efermi, feig[ie]) > 0.0 and feig[ie] >= homo){
					homo = feig[ie];
					homoloc = ie;
				}
			}

			struct { 
				double value; 
				int index; 
			} in, out;

			in.value = homo;
			in.index = -comm.rank();
			
			MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm.get());

			if(comm.rank() == -out.index) {
				assert(out.value == homo);
				focc[homoloc] -= nelec - electrons_available;
				assert(focc[homoloc] >= 0.0);
			} else {
				assert(out.value >= homo);
			}
			
		} else {

			auto dsmear = std::max(1e-14, temperature_);

			auto func = [dsmear] (auto efermi, auto eig){
				return smear_function((efermi - eig)/dsmear);
			};
			
			efermi = get_efermi(comm, electrons_available, feig, fkw, func);
			
			for(long ie = 0; ie < feig.size(); ie++){
				focc[ie] = max_occ_*fkw[ie]*func(efermi, feig[ie]);
				assert(focc[ie] >= 0.0);
			}

		}
		
		assert(fabs(comm.all_reduce_value(operations::sum(focc)) - electrons_available) <= 1e-10);
		
		return efermi;
	}
	
};

}
}
#endif

#ifdef INQ_STATES_KS_STATES_UNIT_TEST
#undef INQ_STATES_KS_STATES_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace Catch::literals;
	using namespace inq;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

  SECTION("Spin unpolarized even"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 12.0);

		CHECK(st.num_electrons() == 12.0);
    CHECK(st.num_states() == 6);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 3.4032608849_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 2.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 2.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 2.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 2.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 2.0);

  }
	
  SECTION("Spin unpolarized odd"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 11.0);

		CHECK(st.num_electrons() == 11.0);
    CHECK(st.num_states() == 6);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);		
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 3.4032608849_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 2.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 2.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 2.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 2.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 1.0);

  }
	
  SECTION("Spin unpolarized even extra"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 12.0, 2);

		CHECK(st.num_electrons() == 12.0);
    CHECK(st.num_states() == 8);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		if(part.contains(6)) eigenvalues[0][part.global_to_local(parallel::global_index(6))] = 1.1;
		if(part.contains(7)) eigenvalues[0][part.global_to_local(parallel::global_index(7))] = 1.3;		
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 1.0660326106_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 2.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 2.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 2.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 2.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 2.0);
		if(part.contains(6)) CHECK(occupations[0][part.global_to_local(parallel::global_index(6))] == 0.0);
		if(part.contains(7)) CHECK(occupations[0][part.global_to_local(parallel::global_index(7))] == 0.0);		

  }
		
  SECTION("Spin unpolarized odd extra"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 11.235, 2);

		CHECK(st.num_electrons() == 11.235);
    CHECK(st.num_states() == 8);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);		
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);		
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 1.0;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.2;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		if(part.contains(6)) eigenvalues[0][part.global_to_local(parallel::global_index(6))] = 1.1;
		if(part.contains(7)) eigenvalues[0][part.global_to_local(parallel::global_index(7))] = 1.3;		
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 1.0660326106_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 2.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 2.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 2.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 2.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 1.235_a);
		if(part.contains(6)) CHECK(occupations[0][part.global_to_local(parallel::global_index(6))] == 0.0);
		if(part.contains(7)) CHECK(occupations[0][part.global_to_local(parallel::global_index(7))] == 0.0);	

  }
	
  SECTION("Spin unpolarized with temperature"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 4.0, 4, 0.01);

		CHECK(st.num_electrons() == 4.0);    
    CHECK(st.num_states() == 6);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);		
		CHECK(st.spinor_dim() == 1);
		
		comm.barrier();
		
		parallel::partition part(st.num_states(), comm);
		
		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.0;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 0.246510327_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0_a);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 1.9810772793_a);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 0.0094611446_a);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 0.0094611446_a);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 4.315768121820668e-07_a);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 3.779107816290222e-33_a);
		
  }
	
	SECTION("Spin polarized"){
    
    states::ks_states st(states::spin_config::POLARIZED, 11.0);

		CHECK(st.num_electrons() == 11.0);    
    CHECK(st.num_states() == 6);
    CHECK(st.num_spin_indices() == 2);
		CHECK(st.num_density_components() == 2);
		CHECK(st.spinor_dim() == 1);		

  }

  SECTION("Non-collinear spin"){
    
    states::ks_states st(states::spin_config::NON_COLLINEAR, 11.0);

		CHECK(st.num_electrons() == 11.0);
    CHECK(st.num_states() == 11);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 4);		
		CHECK(st.spinor_dim() == 2);
				
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);		
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0))  eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1))  eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2))  eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3))  eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4))  eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5))  eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		if(part.contains(6))  eigenvalues[0][part.global_to_local(parallel::global_index(6))] = 1.1;
		if(part.contains(7))  eigenvalues[0][part.global_to_local(parallel::global_index(7))] = 1.3;
		if(part.contains(8))  eigenvalues[0][part.global_to_local(parallel::global_index(8))] = 1.3;
		if(part.contains(9))  eigenvalues[0][part.global_to_local(parallel::global_index(9))] = 1.6;
		if(part.contains(10)) eigenvalues[0][part.global_to_local(parallel::global_index(10))] = 1.8;
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 4.0032608849_a);
		
		if(part.contains(0))  CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 1.0);
		if(part.contains(1))  CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 1.0);
		if(part.contains(2))  CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 1.0);
		if(part.contains(3))  CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 1.0);
		if(part.contains(4))  CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 1.0);
		if(part.contains(5))  CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 1.0);
		if(part.contains(6))  CHECK(occupations[0][part.global_to_local(parallel::global_index(6))] == 1.0);
		if(part.contains(7))  CHECK(occupations[0][part.global_to_local(parallel::global_index(7))] == 1.0);
		if(part.contains(8))  CHECK(occupations[0][part.global_to_local(parallel::global_index(8))] == 1.0);
		if(part.contains(9))  CHECK(occupations[0][part.global_to_local(parallel::global_index(9))] == 1.0);
		if(part.contains(10)) CHECK(occupations[0][part.global_to_local(parallel::global_index(10))] == 1.0);

  }

  SECTION("Spin unpolarized even extra"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 12.0, 2);

		CHECK(st.num_electrons() == 12.0);
    CHECK(st.num_states() == 8);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);		
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		if(part.contains(6)) eigenvalues[0][part.global_to_local(parallel::global_index(6))] = 1.1;
		if(part.contains(7)) eigenvalues[0][part.global_to_local(parallel::global_index(7))] = 1.3;		
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 1.0660326106_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 2.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 2.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 2.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 2.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 2.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 2.0);
		if(part.contains(6)) CHECK(occupations[0][part.global_to_local(parallel::global_index(6))] == 0.0);
		if(part.contains(7)) CHECK(occupations[0][part.global_to_local(parallel::global_index(7))] == 0.0);		

  }
	
	SECTION("Non-collinear spin with extra states"){
    
    states::ks_states st(states::spin_config::NON_COLLINEAR, 8.0, 2);

		CHECK(st.num_electrons() == 8.0);
    CHECK(st.num_states() == 10);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 4);		
		CHECK(st.spinor_dim() == 2);
				
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({1, part.local_size()});
		gpu::array<double, 1> kweights(1, 1.0);		
		gpu::array<double, 2> occupations({1, part.local_size()});

		if(part.contains(0)) eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.1;
		if(part.contains(1)) eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.2;
		if(part.contains(2)) eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.3;
		if(part.contains(3)) eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.3;
		if(part.contains(4)) eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.4;
		if(part.contains(5)) eigenvalues[0][part.global_to_local(parallel::global_index(5))] = 1.0;
		if(part.contains(6)) eigenvalues[0][part.global_to_local(parallel::global_index(6))] = 1.1;
		if(part.contains(7)) eigenvalues[0][part.global_to_local(parallel::global_index(7))] = 1.3;
		if(part.contains(8)) eigenvalues[0][part.global_to_local(parallel::global_index(8))] = 1.5;
		if(part.contains(9)) eigenvalues[0][part.global_to_local(parallel::global_index(9))] = 1.6;
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 1.4131114159_a);
		
		if(part.contains(0)) CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 1.0);
		if(part.contains(1)) CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 1.0);
		if(part.contains(2)) CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 1.0);
		if(part.contains(3)) CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 1.0);
		if(part.contains(4)) CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 1.0);
		if(part.contains(5)) CHECK(occupations[0][part.global_to_local(parallel::global_index(5))] == 1.0);
		if(part.contains(6)) CHECK(occupations[0][part.global_to_local(parallel::global_index(6))] == 1.0);
		if(part.contains(7)) CHECK(occupations[0][part.global_to_local(parallel::global_index(7))] == 1.0);
		if(part.contains(8)) CHECK(occupations[0][part.global_to_local(parallel::global_index(8))] == 0.0);
		if(part.contains(9)) CHECK(occupations[0][part.global_to_local(parallel::global_index(9))] == 0.0);

  }
			
	SECTION("Spin unpolarized kpoints"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 6.0, 2, 0.0, 3);

		CHECK(st.num_electrons() == 6.0);
    CHECK(st.num_states() == 5);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);		
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({3, part.local_size()});
		gpu::array<double, 1> kweights(3, 1.0/3.0);		
		gpu::array<double, 2> occupations({3, part.local_size()});

		if(part.contains(0)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.11;
			eigenvalues[1][part.global_to_local(parallel::global_index(0))] = 0.12;
			eigenvalues[2][part.global_to_local(parallel::global_index(0))] = 0.13;
		}
		if(part.contains(1)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.21;
			eigenvalues[1][part.global_to_local(parallel::global_index(1))] = 0.22;
			eigenvalues[2][part.global_to_local(parallel::global_index(1))] = 0.23;
		}
		if(part.contains(2)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.31;
			eigenvalues[1][part.global_to_local(parallel::global_index(2))] = 0.35;
			eigenvalues[2][part.global_to_local(parallel::global_index(2))] = 0.33;
		}
		if(part.contains(3)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.34;
			eigenvalues[1][part.global_to_local(parallel::global_index(3))] = 0.42;
			eigenvalues[2][part.global_to_local(parallel::global_index(3))] = 0.43;
		}
		if(part.contains(4)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.51;
			eigenvalues[1][part.global_to_local(parallel::global_index(4))] = 0.52;
			eigenvalues[2][part.global_to_local(parallel::global_index(4))] = 0.53;
		}
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 0.3413536007_a);
		
		if(part.contains(0)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
		}
		if(part.contains(1)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
		}			
		if(part.contains(2)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(2))] == 0.0_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(2))] == 0.6666666667_a);
		}
		if(part.contains(3)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(3))] == 0.0_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(3))] == 0.0_a);
		}
		if(part.contains(4)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 0.0);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(4))] == 0.0);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(4))] == 0.0);
		}

  }

	SECTION("Spin unpolarized partial kpoints"){
    
    states::ks_states st(states::spin_config::UNPOLARIZED, 5.5, 2, 0.0, 3);

		CHECK(st.num_electrons() == 5.5);
    CHECK(st.num_states() == 5);
    CHECK(st.num_spin_indices() == 1);
		CHECK(st.num_density_components() == 1);
		CHECK(st.spinor_dim() == 1);
		
		parallel::partition part(st.num_states(), comm);

		gpu::array<double, 2> eigenvalues({3, part.local_size()});
		gpu::array<double, 1> kweights(3, 1.0/3.0);		
		gpu::array<double, 2> occupations({3, part.local_size()});

		if(part.contains(0)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(0))] = 0.11;
			eigenvalues[1][part.global_to_local(parallel::global_index(0))] = 0.12;
			eigenvalues[2][part.global_to_local(parallel::global_index(0))] = 0.13;
		}
		if(part.contains(1)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(1))] = 0.21;
			eigenvalues[1][part.global_to_local(parallel::global_index(1))] = 0.22;
			eigenvalues[2][part.global_to_local(parallel::global_index(1))] = 0.23;
		}
		if(part.contains(2)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(2))] = 0.31;
			eigenvalues[1][part.global_to_local(parallel::global_index(2))] = 0.35;
			eigenvalues[2][part.global_to_local(parallel::global_index(2))] = 0.33;
		}
		if(part.contains(3)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(3))] = 0.34;
			eigenvalues[1][part.global_to_local(parallel::global_index(3))] = 0.42;
			eigenvalues[2][part.global_to_local(parallel::global_index(3))] = 0.43;
		}
		if(part.contains(4)) {
			eigenvalues[0][part.global_to_local(parallel::global_index(4))] = 0.51;
			eigenvalues[1][part.global_to_local(parallel::global_index(4))] = 0.52;
			eigenvalues[2][part.global_to_local(parallel::global_index(4))] = 0.53;
		}
		
		auto efermi = st.update_occupations(comm, eigenvalues, kweights, occupations);

		CHECK(efermi == 0.3413536007_a);
		
		if(part.contains(0)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(0))] == 0.6666666667_a);
		}
		if(part.contains(1)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(1))] == 0.6666666667_a);
		}			
		if(part.contains(2)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(2))] == 0.6666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(2))] == 0.0_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(2))] == 0.6666666667_a);
		}
		if(part.contains(3)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(3))] == 0.1666666667_a);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(3))] == 0.0_a);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(3))] == 0.0_a);
		}
		if(part.contains(4)) {
			CHECK(occupations[0][part.global_to_local(parallel::global_index(4))] == 0.0);
			CHECK(occupations[1][part.global_to_local(parallel::global_index(4))] == 0.0);
			CHECK(occupations[2][part.global_to_local(parallel::global_index(4))] == 0.0);
		}

  }
 
}
#endif
