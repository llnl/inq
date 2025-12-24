/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPERATIONS__ORTHOGONALIZE
#define INQ__OPERATIONS__ORTHOGONALIZE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <basis/field_set.hpp>
#include <matrix/cholesky.hpp>
#include <math/complex.hpp>
#include <operations/overlap.hpp>
#include <operations/rotate.hpp>

#include <cstdlib>
#include <utils/profiling.hpp>

namespace inq {
namespace operations {

template <class field_set_type>
void orthogonalize(field_set_type & phi, bool nocheck = false){
	CALI_CXX_MARK_FUNCTION;

	auto olap = overlap(phi);
	matrix::cholesky(olap);
	operations::rotate_trs(olap, phi);
}

template <class field_set_type, typename Operator>
void orthogonalize(field_set_type & phi, Operator op){
	CALI_CXX_MARK_FUNCTION;

	auto olap = overlap(phi,op(phi));
	matrix::cholesky(olap);
	operations::rotate_trs(olap, phi);
}

template <class field_set_type>
void orthogonalize(field_set_type & phi1, field_set_type const & phi2){
	CALI_CXX_MARK_FUNCTION;

	assert(not phi1.set_part().parallel());

	auto olap = overlap(phi2, phi1);
	auto olap_array = matrix::all_gather(olap);

	gpu::run(phi1.set_part().local_size(), phi1.basis().local_size(),
					 [ph1 = begin(phi1.matrix()), ph2 = begin(phi2.matrix()), ol = begin(olap_array), nst2 = phi2.set_part().local_size()] GPU_LAMBDA (auto ist, auto ip){
						 for(int ist2 = 0; ist2 < nst2; ist2++){
							 ph1[ip][ist] -= ol[ist2][ist]*ph2[ip][ist2];
						 }
					 });
	
}

}
}
#endif

#ifdef INQ_OPERATIONS_ORTHOGONALIZE_UNIT_TEST
#undef INQ_OPERATIONS_ORTHOGONALIZE_UNIT_TEST

#include <basis/trivial.hpp>
#include <catch2/catch_all.hpp>
#include <operations/randomize.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	auto parstates = comm.size();
	if(comm.size() == 4) parstates = 2;
	if(comm.size() >= 5) parstates = 1;
	
	parallel::cartesian_communicator<2> cart_comm(comm, {boost::mpi3::fill, parstates});
	auto basis_comm = basis::basis_subcomm(cart_comm);

	basis::real_space pw(systems::cell::cubic(6.3_b), /*spacing =*/ 0.44428829, basis_comm);
	
	SECTION("Dimension 3"){
		basis::field_set<basis::real_space, complex> phi(pw, 3);
		
		operations::randomize(phi);
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		auto olap_array = matrix::all_gather(olap);
		
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-14);
			} else {
					CHECK(fabs(olap_array[ii][jj]) < 1e-14);
				}
			}
		}
	}
	
	SECTION("Dimension 2, warning test"){
		basis::field_set<basis::real_space, complex> phi(pw, 2);

		operations::randomize(phi);

		int nbasis = get<0>(sizes(phi.matrix()));
		
		for(int ii = 0 ; ii < phi.set_size(); ii++){
		  for(int jj = 0; jj < nbasis; jj++){
		    phi.matrix().transposed()[ii][jj]=0.0;
		    //if(ii == jj) phi.matrix().transposed()[ii][jj] = 1.0;
		  }
		}
		phi.matrix().transposed()[0][0] = 1.0;
		phi.matrix().transposed()[1][0] = 1.0;
		
		operations::orthogonalize(phi, /* ncheck = */ true);
		
		//auto olap = operations::overlap(phi);
		
	}

	SECTION("Dimension 100"){
		basis::field_set<basis::real_space, complex> phi(pw, 100, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		auto olap_array = matrix::all_gather(olap);
				
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-14);
				} else {
					CHECK(fabs(olap_array[ii][jj]) < 1e-13);
				}
			}
		}
	}

	SECTION("Dimension 37 - double orthogonalize"){
		basis::field_set<basis::real_space, complex> phi(pw, 37, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		auto olap_array = matrix::all_gather(olap);
		
		for(int ii = 0; ii < phi.set_size(); ii++){
			for(int jj = 0; jj < phi.set_size(); jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-16);
				} else {
					CHECK(fabs(olap_array[ii][jj]) < 5e-16);
				}
			}
		}
	}

	SECTION("Dimension 100 spinor"){
		states::orbital_set<basis::real_space, complex> phi(pw, 100, /*spinor_dim = */ 2, /*kpoint = */ vector3<double, covariant>{0.0, 0.0, 0.0}, /*spin_index = */ 0, cart_comm);
		
		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		auto olap_array = matrix::all_gather(olap);

		assert(get<0>(olap_array.sizes()) == phi.spinor_set_size());
		assert(get<1>(olap_array.sizes()) == phi.spinor_set_size());
		
		for(int ii = 0; ii < phi.spinor_set_size(); ii++){
			for(int jj = 0; jj < phi.spinor_set_size(); jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-14);
				} else {
					CHECK(fabs(olap_array[ii][jj]) < 1e-13);
				}
			}
		}
	}

	SECTION("Dimension 37 spinor - double orthogonalize"){
		states::orbital_set<basis::real_space, complex> phi(pw, 37, /*spinor_dim = */ 2, /*kpoint = */ vector3<double, covariant>{0.0, 0.0, 0.0}, /*spin_index = */ 0, cart_comm);

		operations::randomize(phi);
		
		operations::orthogonalize(phi);
		operations::orthogonalize(phi);
		
		auto olap = operations::overlap(phi);
		auto olap_array = matrix::all_gather(olap);

		assert(get<0>(olap_array.sizes()) == phi.spinor_set_size());
		assert(get<1>(olap_array.sizes()) == phi.spinor_set_size());

		for(int ii = 0; ii < phi.spinor_set_size(); ii++){
			for(int jj = 0; jj < phi.spinor_set_size(); jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-16);
				} else {
					CHECK(fabs(olap_array[ii][jj]) < 5e-16);
				}
			}
		}
	}
	
	SECTION("Two arguments"){
		basis::field_set<basis::real_space, complex> phi1(pw, 100);
		basis::field_set<basis::real_space, complex> phi2(pw, 100);   

		operations::randomize(phi1);
		operations::orthogonalize(phi1); 

		gpu::run(phi1.set_part().local_size(), phi1.basis().local_sizes()[2], phi1.basis().local_sizes()[1], phi1.basis().local_sizes()[0],
						 [point_op = phi1.basis().point_op(), ph1 = begin(phi1.hypercubic()), ph2 = begin(phi2.hypercubic())] GPU_LAMBDA (auto ist, auto iz, auto iy, auto ix){
							 ph2[ix][iy][iz][ist] = ph1[ix][iy][iz][ist]*point_op.rvector_cartesian(ix, iy, iz)[0];
						 });

		operations::orthogonalize(phi2, phi1);
		
		auto olap = operations::overlap(phi2, phi1);

		auto olap_array = matrix::all_gather(olap);
				
		for(int ii = 0; ii < phi1.set_size(); ii++){
			for(int jj = 0; jj < phi1.set_size(); jj++){
				CHECK(fabs(olap_array[ii][jj]) < 1e-13);
			}
		}
		
	}

	SECTION("Orthogonalize w.r.t. non diagonal matrix complex"){
  
		const int npoint = 100;
		const int nvec = 12;


		gpu::array<complex, 2> matrix({npoint, npoint});
		
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				matrix[ip][jp] = 1e-3*ip*jp;
				if(ip == jp) matrix[ip][jp] = ip + 1.0;
			}
		}
		
		operations::matrix_operator<complex> op(std::move(matrix));
		
		basis::trivial bas(npoint, parallel::communicator{boost::mpi3::environment::get_self_instance()});
		basis::field_set<basis::trivial, complex> phi(bas, nvec);

		for(int ip = 0; ip < npoint; ip++){
			for(int ivec = 0; ivec < nvec; ivec++){
				phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
			}
		}
		auto matrix_op = [& op](auto const & phi ){
			return op(phi);
		};
		operations::orthogonalize(phi,matrix_op);
		auto olap = operations::overlap(phi, matrix_op(phi));
		auto olap_array = matrix::all_gather(olap);

		for(int ii = 0; ii < nvec; ii++){
			for(int jj = 0; jj < nvec; jj++){
				if(ii == jj) {
					CHECK(real(olap_array[ii][ii]) == 1.0_a);
					CHECK(fabs(imag(olap_array[ii][ii])) < 1e-16);
				} else {
					CHECK(fabs(olap_array[ii][jj]) < 5e-16);
				}
			}
		}
	}
	
}
#endif
