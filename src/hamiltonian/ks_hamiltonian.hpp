/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__KS_HAMILTONIAN
#define INQ__HAMILTONIAN__KS_HAMILTONIAN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <multi/adaptors/fftw.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/exchange_operator.hpp>
#include <hamiltonian/projector.hpp>
#include <hamiltonian/projector_all.hpp>
#include <hamiltonian/relativistic_projector.hpp>
#include <hamiltonian/scalar_potential.hpp>
#include <input/environment.hpp>
#include <operations/transform.hpp>
#include <operations/laplacian.hpp>
#include <operations/gradient.hpp>
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>

#include <utils/profiling.hpp>

#include <list>
#include <unordered_map>

namespace inq {
namespace hamiltonian {

template <typename PotentialType>
class ks_hamiltonian {
		
public:

	using potential_type = PotentialType;

private:

	exchange_operator exchange_;
	basis::field_set<basis::real_space, double> vxc_;
	basis::field_set<basis::real_space, PotentialType> scalar_potential_;
	vector3<double, covariant> uniform_vector_potential_;
	projector_all projectors_all_;		
	std::list<relativistic_projector> projectors_rel_;
	states::ks_states states_;

#ifdef ENABLE_CUDA
public:
#endif
		
		template <typename OccType, typename ArrayType>
		static double occ_sum(OccType const & occupations, ArrayType const & array) {
			CALI_CXX_MARK_FUNCTION;
			
			assert(occupations.size() == array.size());
			return gpu::run(gpu::reduce(array.size()), 0.0, [occ = begin(occupations), arr = begin(array)] GPU_LAMBDA (auto ip) {
				return occ[ip]*real(arr[ip]);
			});
		}

public:
	
	void update_projectors(const basis::real_space & basis, const atomic_potential & pot, systems::ions const & ions){
			
		CALI_CXX_MARK_FUNCTION;

		std::list<projector> projectors;
			
		for(int iatom = 0; iatom < ions.size(); iatom++){
			auto && ps = pot.pseudo_for_element(ions.species(iatom));

			if(ps.has_total_angular_momentum()){
				projectors_rel_.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors_rel_.back().empty()) projectors_rel_.pop_back();
			} else {
				projectors.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors.back().empty()) projectors.pop_back();
			}
		}

		projectors_all_ = projector_all(projectors);
	}
		

	////////////////////////////////////////////////////////////////////////////////////////////
		
	ks_hamiltonian(const basis::real_space & basis, ionic::brillouin const & bzone, states::ks_states const & states, atomic_potential const & pot, systems::ions const & ions,
								 const double exchange_coefficient, bool use_ace = false):
		exchange_(basis.cell(), bzone, exchange_coefficient, use_ace),
		vxc_(basis, states.num_density_components()),
		scalar_potential_(basis, states.num_density_components()),
		uniform_vector_potential_({0.0, 0.0, 0.0}),
		states_(states)
	{
		scalar_potential_.fill(0.0);
		update_projectors(basis, pot, ions);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto non_local(const states::orbital_set<basis::real_space, complex> & phi) const {

		CALI_CXX_MARK_FUNCTION;
 
		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);
		
		states::orbital_set<basis::real_space, complex> vnlphi(phi.skeleton());
		vnlphi.fill(0.0);
		
		projectors_all_.apply(proj, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		for(auto & pr : projectors_rel_) pr.apply(phi, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		return vnlphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Occupations>
	auto non_local_energy(states::orbital_set<basis::real_space, complex> const & phi, Occupations const & occupations, bool const reduce_states = true) const {

		CALI_CXX_MARK_FUNCTION;

		auto en = projectors_all_.energy(phi, phi.kpoint() + uniform_vector_potential_, occupations, reduce_states);
		for(auto & pr : projectors_rel_) en += pr.energy(phi, occupations, phi.kpoint() + uniform_vector_potential_);
		return en;
		
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("hamiltonian_real");

		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);
			
		auto phi_fs = operations::transform::to_fourier(phi);
		
		auto hphi_fs = operations::laplacian(phi_fs, -0.5, -2.0*phi.basis().cell().metric().to_contravariant(phi.kpoint() + uniform_vector_potential_));
			
		auto hphi = operations::transform::to_real(hphi_fs);

		hamiltonian::scalar_potential_add(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().metric().norm(phi.kpoint() + uniform_vector_potential_), phi, hphi);
		exchange_(phi, hphi);

		for(auto & pr : projectors_rel_) pr.apply(phi, hphi, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi, phi.kpoint() + uniform_vector_potential_);

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::fourier_space, complex> & phi) const{
			
		CALI_CXX_MARK_SCOPE("hamiltonian_fourier");

		auto phi_rs = operations::transform::to_real(phi);

		auto proj = projectors_all_.project(phi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi_rs = hamiltonian::scalar_potential(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().metric().norm(phi.kpoint() + uniform_vector_potential_), phi_rs);
		
		exchange_(phi_rs, hphi_rs);

		for(auto & pr : projectors_rel_) pr.apply(phi_rs, hphi_rs, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi = operations::transform::to_fourier(hphi_rs);

		operations::laplacian_add(phi, hphi, -0.5, -2.0*phi.basis().cell().metric().to_contravariant(phi.kpoint() + uniform_vector_potential_));

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto momentum(const states::orbital_set<basis::real_space, complex> & phi) const{
		CALI_CXX_MARK_FUNCTION;

		return operations::gradient(phi, /* factor = */ 1.0, /*shift = */ phi.kpoint() + uniform_vector_potential_);
	}
		
	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto & projectors_all() const {
		return projectors_all_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto & projectors_rel() const {
		return projectors_rel_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <class output_stream>
	void info(output_stream & out) const {
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & scalar_potential() {
		return scalar_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & exchange() {
		return exchange_;
	}
	
	auto & exchange() const {
		return exchange_;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto & uniform_vector_potential() const {
		return uniform_vector_potential_;
	}
	auto & uniform_vector_potential() {
		return uniform_vector_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & vxc() const {
		return vxc_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Perturbation>
	friend class self_consistency;
	
};
}
}
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
#endif

#ifdef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST
#undef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
  
	parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});

	auto set_comm = basis::set_subcomm(cart_comm);
	auto basis_comm = basis::basis_subcomm(cart_comm);	

	auto ions = systems::ions(systems::cell::cubic(10.0_b));

	basis::real_space rs(ions.cell(), /*spacing = */ 0.49672941, basis_comm);

	SECTION("Basis"){
		
		CHECK(rs.size() == 8000);
		CHECK(rs.rspacing()[0] == 0.5_a);
		CHECK(rs.rspacing()[1] == 0.5_a);	
		CHECK(rs.rspacing()[2] == 0.5_a);
		CHECK(rs.volume_element() == 0.125_a);
	}
	
	hamiltonian::atomic_potential pot(ions.species_list(), rs.gcutoff());
	
	states::ks_states st(states::spin_config::UNPOLARIZED, 11.0);

  states::orbital_set<basis::real_space, complex> phi(rs, st.num_states(), 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);

	auto bzone = ionic::brillouin(ions, input::kpoints::gamma());
	
	hamiltonian::ks_hamiltonian<double> ham(rs, bzone, st, pot, ions, 0.0);

	SECTION("Constant function"){
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){

					ham.scalar_potential().hypercubic()[ix][iy][iz][0] = 0.0;
					
					for(int ist = 0; ist < phi.local_set_size(); ist++){
						phi.hypercubic()[ix][iy][iz][ist] = 1.0;
					}
				}
			}
		}
		
		auto hphi = ham(phi);
		
		double diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.local_set_size(); ist++){
						diff += fabs(hphi.hypercubic()[ix][iy][iz][ist] - 0.0);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		diff /= hphi.set_size()*hphi.basis().size();
		
		CHECK(diff < 1e-14);
		
	}
	
	SECTION("Plane wave"){
		
		double kk = 2.0*M_PI/rs.rlength()[0];
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){

					ham.scalar_potential().hypercubic()[ix][iy][iz][0] = 0.0;
					
					for(int ist = 0; ist < phi.local_set_size(); ist++){

						auto ixg = rs.cubic_part(0).local_to_global(ix);
						auto iyg = rs.cubic_part(1).local_to_global(iy);
						auto izg = rs.cubic_part(2).local_to_global(iz);	
						auto istg = phi.set_part().local_to_global(ist);
						
						double xx = rs.point_op().rvector_cartesian(ixg, iyg, izg)[0];
						phi.hypercubic()[ix][iy][iz][ist] = complex(cos(istg.value()*kk*xx), sin(istg.value()*kk*xx));
					}
				}
			}
		}

		auto hphi = ham(phi);
		
		double diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.local_set_size(); ist++){
						auto istg = phi.set_part().local_to_global(ist);
						diff += fabs(hphi.hypercubic()[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*phi.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}
		
		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		diff /= hphi.set_size()*hphi.basis().size();

		CHECK(diff < 1e-14);
		
	}

	SECTION("Harmonic oscillator"){

		double ww = 2.0;

		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){

					auto ixg = rs.cubic_part(0).local_to_global(ix);
					auto iyg = rs.cubic_part(1).local_to_global(iy);
					auto izg = rs.cubic_part(2).local_to_global(iz);	
					
					double r2 = rs.point_op().r2(ixg, iyg, izg);
					ham.scalar_potential().hypercubic()[ix][iy][iz][0] = 0.5*ww*ww*r2;

					for(int ist = 0; ist < phi.local_set_size(); ist++){
						phi.hypercubic()[ix][iy][iz][ist] = exp(-ww*r2);
					}
					
				}
			}
		}

		auto hphi = ham(phi);
		
		double diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.local_set_size(); ist++){
						diff += fabs(hphi.hypercubic()[ix][iy][iz][ist] - 1.5*ww*phi.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		diff /= hphi.set_size()*hphi.basis().size();

		CHECK(diff == 0.0051420503_a);
		
	}


	SECTION("Plane wave - fourier"){
		
		double kk = 2.0*M_PI/rs.rlength()[0];
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){

					ham.scalar_potential().hypercubic()[ix][iy][iz][0] = 0.0;
					
					for(int ist = 0; ist < phi.local_set_size(); ist++){

						auto ixg = rs.cubic_part(0).local_to_global(ix);
						auto iyg = rs.cubic_part(1).local_to_global(iy);
						auto izg = rs.cubic_part(2).local_to_global(iz);	
						auto istg = phi.set_part().local_to_global(ist);
						
						double xx = rs.point_op().rvector_cartesian(ixg, iyg, izg)[0];
						phi.hypercubic()[ix][iy][iz][ist] = complex(cos(istg.value()*kk*xx), sin(istg.value()*kk*xx));
					}
				}
			}
		}

		auto hphi = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
		double diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.local_set_size(); ist++){

						auto istg = phi.set_part().local_to_global(ist);

						diff += fabs(hphi.hypercubic()[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*phi.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		diff /= hphi.set_size()*hphi.basis().size();

		CHECK(diff < 1e-14);
		
	}

	SECTION("Harmonic oscillator - fourier"){

		double ww = 2.0;

		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){

					auto ixg = rs.cubic_part(0).local_to_global(ix);
					auto iyg = rs.cubic_part(1).local_to_global(iy);
					auto izg = rs.cubic_part(2).local_to_global(iz);	
					
					double r2 = rs.point_op().r2(ixg, iyg, izg);
					ham.scalar_potential().hypercubic()[ix][iy][iz][0] = 0.5*ww*ww*r2;

					for(int ist = 0; ist < phi.local_set_size(); ist++){
						phi.hypercubic()[ix][iy][iz][ist] = exp(-ww*r2);
					}
					
				}
			}
		}

		auto hphi = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
		double diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.local_set_size(); ist++){
						diff += fabs(hphi.hypercubic()[ix][iy][iz][ist] - 1.5*ww*phi.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		diff /= hphi.set_size()*hphi.basis().size();

		CHECK(diff == 0.0051420503_a);
		
	}
	
}
#endif
