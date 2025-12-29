/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__EIGENSOLVERS__STEEPEST_DESCENT
#define INQ__EIGENSOLVERS__STEEPEST_DESCENT

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/complex.hpp>
#include <math/vector3.hpp>
#include <gpu/array.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <operations/shift.hpp>
#include <operations/orthogonalize.hpp>
#include <operations/overlap_diagonal.hpp>

namespace inq {
namespace eigensolvers {

template <class operator_type, class preconditioner_type, class field_set_type>
void steepest_descent(const operator_type & ham, const preconditioner_type & prec, field_set_type & phi){

	CALI_CXX_MARK_FUNCTION;
	
	const int num_steps = 5;

	auto hphi = ham(phi);
	
	for(int istep = 0; istep < num_steps; istep++){

		auto residual = hphi;
		auto eigenvalues = operations::overlap_diagonal(residual, phi);
		auto norm =	operations::overlap_diagonal(phi);
			
		auto evnorm = eigenvalues;
		
		gpu::run(eigenvalues.size(),
						 [evn = begin(evnorm), nor = begin(norm)] GPU_LAMBDA (auto ist){
							 evn[ist] /= real(nor[ist]);
						 });
		
		operations::shift(-1.0, evnorm, phi, residual);

		prec(residual);

		auto mm = gpu::array<typename field_set_type::element_type, 2>({6, eigenvalues.size()});

		auto hresidual = ham(residual);
		mm[0] = operations::overlap_diagonal(residual, residual);
		mm[1] = operations::overlap_diagonal(phi, residual);
		mm[2] = operations::overlap_diagonal(residual, hresidual);
		mm[3] = operations::overlap_diagonal(phi, hresidual);
		mm[4] = eigenvalues;
		mm[5] = norm;

		auto lambda = gpu::array<double, 1>(eigenvalues.size());
		
		gpu::run(phi.local_spinor_set_size(),
						 [m = begin(mm), lam = begin(lambda)]
						 GPU_LAMBDA (auto ist){
							 auto ca = real(m[0][ist]*m[3][ist] - m[2][ist]*m[1][ist]);
							 auto cb = real(m[5][ist]*m[2][ist] - m[4][ist]*m[0][ist]);
							 auto cc = real(m[4][ist]*m[1][ist] - m[3][ist]*m[5][ist]);
							 auto den = cb + sqrt(cb*cb - 4.0*ca*cc);

							 if(fabs(den) < 1e-15) { //this happens if we are perfectly converged
								 lam[ist] = 0.0;
							 } else {
								 lam[ist] = 2.0*cc/den;
							 }
						 });

		gpu::run(phi.local_spinor_set_size(), phi.spinor_matrix().size(),
						 [res = begin(residual.spinor_matrix()), ph = begin(phi.spinor_matrix()),
							hres = begin(hresidual.spinor_matrix()), hph = begin(hphi.spinor_matrix()), lam = begin(lambda), istep] GPU_LAMBDA (auto ist, auto ip){
							 ph[ip][ist] += lam[ist]*res[ip][ist];
							 if(istep !=  num_steps - 1) hph[ip][ist] += lam[ist]*hres[ip][ist];
						 });
	}

	operations::orthogonalize(phi);
		
}

}
}
#endif

#ifdef INQ_EIGENSOLVERS_STEEPEST_DESCENT_UNIT_TEST
#undef INQ_EIGENSOLVERS_STEEPEST_DESCENT_UNIT_TEST

#include <basis/trivial.hpp>
#include <operations/matrix_operator.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	
  const int npoint = 100;
  const int nvec = 12;
  
  basis::trivial bas(npoint, parallel::communicator{boost::mpi3::environment::get_self_instance()});
	
	gpu::array<complex, 2> identity_matrix({npoint, npoint});
  
	for(int ip = 0; ip < npoint; ip++){
		for(int jp = 0; jp < npoint; jp++){
			identity_matrix[ip][jp] = 0.0;
			if(ip == jp) identity_matrix[ip][jp] = 1.0;
		}
	}
  
	operations::matrix_operator<complex> identity(std::move(identity_matrix));

	SECTION("Diagonal matrix complex"){
  
    gpu::array<complex, 2> diagonal_matrix({npoint, npoint});
    
    for(int ip = 0; ip < npoint; ip++){
      for(int jp = 0; jp < npoint; jp++){
        diagonal_matrix[ip][jp] = 0.0;
        if(ip == jp) diagonal_matrix[ip][jp] = ip + 1.0;
      }
    }
    
    operations::matrix_operator<complex> diagonal_op(std::move(diagonal_matrix));
    
    basis::field_set<basis::trivial, complex> phi(bas, nvec);

		for(int ip = 0; ip < npoint; ip++){
      for(int ivec = 0; ivec < nvec; ivec++){
        phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
      }
    }

		operations::orthogonalize(phi);

		const int num_iter = 100;
		
		for(int iter = 0; iter < num_iter; iter++){

			eigensolvers::steepest_descent(diagonal_op, identity, phi);
			
			auto residual = diagonal_op(phi);
			auto eigenvalues = operations::overlap_diagonal(phi, residual);
			operations::shift(-1.0, eigenvalues, phi, residual);
			auto normres = operations::overlap_diagonal(residual);
			
			/*
				fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
				
				for(int ivec = 0; ivec < phi.local_set_size(); ivec++){
				fmt::print(std::cout, "    state {:4d}  evalue = {:18.12f}  res = {:15.10e}\n", ivec + 1, real(eigenvalues[ivec]), real(normres[ivec]));
				}
			*/

			if(num_iter - 1 == iter){

				CHECK(fabs(eigenvalues[0]) == 1.000000001634_a);
				CHECK(fabs(eigenvalues[1]) == 2.000000003689_a);
				CHECK(fabs(eigenvalues[2]) == 3.000000001955_a);
				CHECK(fabs(eigenvalues[3]) == 4.000000001760_a);
				CHECK(fabs(eigenvalues[4]) == 5.000000002561_a);
				CHECK(fabs(eigenvalues[5]) == 6.000000003127_a);
				CHECK(fabs(eigenvalues[6]) == 7.000000002312_a);
				CHECK(fabs(eigenvalues[7]) == 8.000000000292_a);
				CHECK(fabs(eigenvalues[8]) == 8.999999999033_a);
				CHECK(fabs(eigenvalues[9]) == 9.999999998497_a);
				CHECK(fabs(eigenvalues[10]) == 10.999999998768_a);
				CHECK(fabs(eigenvalues[11]) == 11.999999998422_a);

			}
			
		}
 	
 }

#if 0
	SECTION("Periodic Laplacian matrix complex"){
		
    gpu::array<complex, 2> laplacian_matrix({npoint, npoint});
    
    for(int ip = 0; ip < npoint; ip++){
      for(int jp = 0; jp < npoint; jp++){
        laplacian_matrix[ip][jp] = 0.0;
				identity_matrix[ip][jp] = 0.0;
        if(ip == jp) laplacian_matrix[ip][jp] = -1.0;
        if(ip == jp + 1 or ip == jp - 1) laplacian_matrix[ip][jp] = 2.0;
      }
    }
		
    operations::matrix_operator<complex> laplacian(std::move(laplacian_matrix));
    
    basis::field_set<basis::trivial, complex> phi(bas, nvec);

		phi = 0.0;
		for(int ivec = 0; ivec < nvec; ivec++) phi.matrix()[ivec][ivec] = 1.0;

		for(int iter = 0; iter < 100; iter++){
			
			eigensolvers::steepest_descent(laplacian, identity, phi);
			
			auto residual = laplacian(phi);
			auto eigenvalues = operations::overlap_diagonal(phi, residual);
			operations::shift(eigenvalues, phi, residual, -1.0);
			auto normres = operations::overlap_diagonal(residual);
			
			for(int ivec = 0; ivec < phi.local_set_size(); ivec++){
				fmt::print(std::cout, " state {:4d}  evalue = {:18.12f}  res = {:5.0e}\n", ivec + 1, real(eigenvalues[ivec]), real(normres[ivec]));
			}
		}
	}
#endif 	

}
#endif
