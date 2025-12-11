/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__DENSITY
#define INQ__OBSERVABLES__DENSITY

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <basis/field_set.hpp>
#include <operations/integral.hpp>
#include <operations/transfer.hpp>
#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>
#include <observables/magnetization.hpp>

namespace inq {
namespace observables {
namespace density {

template<class occupations_array_type, class field_set_type>
void calculate_add(const occupations_array_type & occupations, field_set_type & phi, basis::field_set<typename field_set_type::basis_type, double> & density){

	assert(phi.basis() == density.basis());
	
	if(not phi.spinors()){

		assert(phi.spin_index() < density.set_size());

		gpu::run(phi.basis().part().local_size(),
						 [nst = phi.set_part().local_size(), occ = begin(occupations), ph = begin(phi.matrix()), den = begin(density.matrix()), ispin = phi.spin_index()] GPU_LAMBDA (auto ipoint){
							 for(int ist = 0; ist < nst; ist++) den[ipoint][ispin] += occ[ist]*norm(ph[ipoint][ist]);
						 });
	} else {
		
		assert(density.set_size() == 4);

		assert(get<1>(sizes(phi.spinor_array())) == phi.spinor_dim());
		assert(get<2>(sizes(phi.spinor_array())) == phi.local_spinor_set_size());
		
		gpu::run(phi.basis().part().local_size(),
						 [nst = phi.local_spinor_set_size(), occ = begin(occupations), ph = begin(phi.spinor_array()), den = begin(density.matrix())] GPU_LAMBDA (auto ipoint){
							 for(int ist = 0; ist < nst; ist++) {
								 den[ipoint][0] += occ[ist]*norm(ph[ipoint][0][ist]);
								 den[ipoint][1] += occ[ist]*norm(ph[ipoint][1][ist]);
								 auto crossterm = occ[ist]*ph[ipoint][0][ist]*conj(ph[ipoint][1][ist]);
								 den[ipoint][2] += real(crossterm);
								 den[ipoint][3] += imag(crossterm);
							 }
						 });
	}
		
}

///////////////////////////////////////////////////////////////

template<class occupations_array_type, class field_set_type, class vector_field_set_type, typename VectorSpace>
void calculate_gradient_add(const occupations_array_type & occupations, field_set_type const & phi, vector_field_set_type const & gphi, basis::field<typename vector_field_set_type::basis_type, vector3<double, VectorSpace>> & gdensity){

	CALI_CXX_MARK_SCOPE("density::calculate_gradient");

	gpu::run(phi.basis().part().local_size(),
					 [nst = phi.local_spinor_set_size(), occs = begin(occupations),	phip = begin(phi.spinor_array()), gphip = begin(gphi.spinor_array()), gdensityp = begin(gdensity.linear()), nspinor = phi.spinor_dim()]
					 GPU_LAMBDA (auto ip){
						 for(int ispinor = 0; ispinor < nspinor; ispinor++){
							 for(int ist = 0; ist < nst; ist++) {
								 gdensityp[ip] += occs[ist]*real(conj(gphip[ip][ispinor][ist])*phip[ip][ispinor][ist] + conj(phip[ip][ispinor][ist])*gphip[ip][ispinor][ist]);
							 }
						 }
					 });
}

///////////////////////////////////////////////////////////////

template <typename ElecType>
basis::field_set<basis::real_space, double> calculate(ElecType & elec){
	
	CALI_CXX_MARK_SCOPE("density::calculate");

	basis::field_set<basis::real_space, double> density(elec.density_basis(), elec.states().num_density_components());

	density.fill(0.0);

	int iphi = 0;
	for(auto & phi : elec.kpin()) {
		if(phi.basis() == density.basis()) {
			density::calculate_add(elec.occupations()[iphi], phi, density);
		} else {
			auto fine_phi = operations::transfer::refine(phi, density.basis());
			density::calculate_add(elec.occupations()[iphi], fine_phi, density);
		}
		iphi++;
	}

	density.all_reduce(elec.kpin_states_comm());

	return density;
}

///////////////////////////////////////////////////////////////

template <class FieldType>
auto normalize(FieldType & density, const double & total_charge) -> typename FieldType::element_type {

	CALI_CXX_MARK_FUNCTION;
	
	auto max_index = std::min(2, density.set_size());
	auto qq = operations::integral_partial_sum(density, max_index);
	assert(fabs(qq) > 1e-16);

	gpu::run(density.local_set_size(), density.basis().local_size(),
					 [den = begin(density.matrix()), factor = total_charge/qq] GPU_LAMBDA (auto ist, auto ip){ 
						 den[ip][ist] *= factor;
					 });
	return qq;
}

///////////////////////////////////////////////////////////////

template <class BasisType, class ElementType>
basis::field<BasisType, ElementType> total(basis::field_set<BasisType, ElementType> const & spin_density){

	CALI_CXX_MARK_FUNCTION;

	assert(spin_density.set_size() == 1 or spin_density.set_size() == 2 or spin_density.set_size() == 4);
	assert(spin_density.set_size() == spin_density.local_set_size());
	
	basis::field<BasisType, ElementType> total_density(spin_density.basis());

	gpu::run(spin_density.basis().local_size(),
					 [spi = begin(spin_density.matrix()), tot = begin(total_density.linear()), nspin = spin_density.set_size()] GPU_LAMBDA (auto ip){
						 if(nspin == 1) tot[ip] = spi[ip][0];
						 else tot[ip] = spi[ip][0] + spi[ip][1];
					 });

	return total_density;
}

///////////////////////////////////////////////////////////////

template <class FieldType>
void rotate_total_magnetization(FieldType & density, vector3<double> const & magnet_dir) {

	CALI_CXX_MARK_FUNCTION

	vector3 e_v = magnet_dir/sqrt(norm(magnet_dir));

	if (density.set_size() == 2){
		gpu::run(density.basis().local_size(),
			[den = begin(density.matrix()), mv = e_v[2]] GPU_LAMBDA (auto ip){
				auto n0 = den[ip][0] + den[ip][1];
				auto m0 = den[ip][0] - den[ip][1];
				den[ip][0] = 0.5*(n0 + m0*mv);
				den[ip][1] = 0.5*(n0 - m0*mv);
			});
	}
	else {
		assert(density.set_size() == 4);
		gpu::run(density.basis().local_size(),
				[den = begin(density.matrix()), mv = e_v] GPU_LAMBDA (auto ip){
					auto mag = observables::local_magnetization(den[ip], 4);
					auto m0 = sqrt(norm(mag));
					auto n0 = den[ip][0] + den[ip][1];
					den[ip][0] = 0.5*(n0 + m0*mv[2]);
					den[ip][1] = 0.5*(n0 - m0*mv[2]);
					den[ip][2] = m0*mv[0]/2.0;
					den[ip][3] = -m0*mv[1]/2.0;
				});
	}
}

}
}
}
#endif

#ifdef INQ_OBSERVABLES_DENSITY_UNIT_TEST
#undef INQ_OBSERVABLES_DENSITY_UNIT_TEST

#include <basis/trivial.hpp>
#include <math/complex.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	const int npoint = 100;
	const int nvec = 12;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	parallel::cartesian_communicator<2> cart_comm(comm, {});
	
	auto basis_comm = basis::basis_subcomm(cart_comm);
	
	basis::trivial bas(npoint, basis_comm);
	
	SECTION("double"){
		
		states::orbital_set<basis::trivial, double> aa(bas, nvec, 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);

		gpu::array<double, 1> occ(aa.set_part().local_size());
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++){
			for(int jj = 0; jj < aa.set_part().local_size(); jj++){
				aa.matrix()[ii][jj] = sqrt(bas.part().local_to_global(ii).value())*(aa.set_part().local_to_global(jj).value() + 1);
			}
		}

		for(int jj = 0; jj < aa.set_part().local_size(); jj++) occ[jj] = 1.0/(aa.set_part().local_to_global(jj).value() + 1);

		basis::field_set<basis::trivial, double> dd(bas, 1);
		dd.fill(0.0);
		
		observables::density::calculate_add(occ, aa, dd);

		dd.all_reduce(aa.set_comm());
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) CHECK(dd.matrix()[ii][0] == Approx(0.5*bas.part().local_to_global(ii).value()*nvec*(nvec + 1)));

		auto tdd = observables::density::total(dd);
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) CHECK(tdd.linear()[ii] == Approx(0.5*bas.part().local_to_global(ii).value()*nvec*(nvec + 1)));
		
	}
	
	SECTION("complex"){
		
		states::orbital_set<basis::trivial, complex> aa(bas, nvec, 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);

		gpu::array<double, 1> occ(nvec);
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++){
			for(int jj = 0; jj < aa.set_part().local_size(); jj++){
				aa.matrix()[ii][jj] = sqrt(bas.part().local_to_global(ii).value())*(aa.set_part().local_to_global(jj).value() + 1)*exp(complex(0.0, M_PI/65.0*bas.part().local_to_global(ii).value()));
			}
		}

		for(int jj = 0; jj < aa.set_part().local_size(); jj++) occ[jj] = 1.0/(aa.set_part().local_to_global(jj).value() + 1);

		basis::field_set<basis::trivial, double> dd(bas, 1);
		dd.fill(0.0);
		
		observables::density::calculate_add(occ, aa, dd);

		dd.all_reduce(aa.set_comm());
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) CHECK(dd.matrix()[ii][0] == Approx(0.5*bas.part().local_to_global(ii).value()*nvec*(nvec + 1)));

		auto tdd = observables::density::total(dd);

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) CHECK(tdd.linear()[ii] == Approx(0.5*bas.part().local_to_global(ii).value()*nvec*(nvec + 1)));
		
	}
	
	SECTION("spinor"){
		
		states::orbital_set<basis::trivial, complex> aa(bas, nvec, 2, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);

		CHECK(get<1>(sizes(aa.spinor_array())) == 2);
		
		gpu::array<double, 1> occ(nvec);
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++){
			for(int jj = 0; jj < aa.local_spinor_set_size(); jj++){
				auto iig = bas.part().local_to_global(ii).value();
				auto jjg = aa.spinor_set_part().local_to_global(jj).value();
				aa.spinor_array()[ii][0][jj] = sqrt(iig)*(jjg + 1)*exp(complex(0.0, M_PI/65.0*iig));
				aa.spinor_array()[ii][1][jj] = sqrt(iig)*(jjg + 1)*exp(complex(0.0, M_PI/65.0*iig));
			}
		}

		for(int jj = 0; jj < aa.local_spinor_set_size(); jj++) {
			auto jjg = aa.spinor_set_part().local_to_global(jj).value();      
			occ[jj] = 1.0/(jjg + 1);
		}

		basis::field_set<basis::trivial, double> dd(bas, 4);
		dd.fill(0.0);
		
		observables::density::calculate_add(occ, aa, dd);

		dd.all_reduce(aa.set_comm());

		for(int ii = 0; ii < dd.basis().part().local_size(); ii++) {
			auto iig = bas.part().local_to_global(ii).value();      
			CHECK(dd.matrix()[ii][0] == Approx(0.5*iig*nvec*(nvec + 1)));
			CHECK(dd.matrix()[ii][1] == Approx(0.5*iig*nvec*(nvec + 1)));
			CHECK(dd.matrix()[ii][2] == Approx(0.5*iig*nvec*(nvec + 1)));
			CHECK(fabs(dd.matrix()[ii][3]) < 1e-12);
		}

		
		auto tdd = observables::density::total(dd);

		for(int ii = 0; ii < dd.basis().part().local_size(); ii++) {
			auto iig = bas.part().local_to_global(ii).value();
			CHECK(tdd.linear()[ii] == Approx(2.0*0.5*iig*nvec*(nvec + 1)));
		}
		
	}

	SECTION("normalize double"){
		
		basis::field_set<basis::trivial, double> aa(bas, 1);

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) aa.matrix()[ii][0] = sqrt(bas.part().local_to_global(ii).value());

		observables::density::normalize(aa, 33.3);

		CHECK(operations::integral_sum(aa) == 33.3_a);
		
	}
	
	SECTION("normalize complex"){
		
		basis::field_set<basis::trivial, complex> aa(bas, 1);

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++){
			aa.matrix()[ii][0] = sqrt(bas.part().local_to_global(ii).value())*exp(complex(0.0, M_PI/65.0*bas.part().local_to_global(ii).value()));
		}

		observables::density::normalize(aa, 19.2354);

		CHECK(real(operations::integral_sum(aa)) == 19.2354_a);
		
	}
	
}
#endif
