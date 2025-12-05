/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SOLVERS__STEEPEST_DESCENT
#define INQ__SOLVERS__STEEPEST_DESCENT

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

//required for debugging
//#include <mpi3/ostream.hpp>

namespace inq {
namespace solvers {

template <class operator_type, class preconditioner_type, class field_set_type>
double steepest_descent(const operator_type & ham, const preconditioner_type & prec, field_set_type & rhs, field_set_type & phi){
	CALI_CXX_MARK_SCOPE("solver::steepest_descent");

	const int num_steps = 5;

	auto mm = gpu::array<typename field_set_type::element_type, 2>({3, phi.local_set_size()});
	auto lambda = gpu::array<typename field_set_type::element_type, 1>(phi.local_set_size());
	auto normres = gpu::array<double, 1>(phi.local_set_size());

	auto residual = ham(phi);
	operations::shift(-1.0, rhs, residual);

	for(int istep = 0; istep < num_steps; istep++){
    
		//calculate the residual
			
    auto sd = residual;
		prec(sd);
		auto hsd = ham(sd);
      
		mm[0] = operations::overlap_diagonal(hsd, hsd);
		mm[1] = operations::overlap_diagonal(residual, hsd);
		mm[2] = operations::overlap_diagonal(residual, residual);

		gpu::run(phi.local_set_size(),
						 [m = begin(mm), lam = begin(lambda), nor = begin(normres)] GPU_LAMBDA (auto ist){
							 auto ca = m[0][ist];
							 auto cb = 4.0*real(m[1][ist]);
							 auto cc = m[2][ist];

							 nor[ist] = fabs(cc);

							 auto sqarg = cb*cb - 4.0*ca*cc;
							 auto signb = (real(conj(cb)*sqarg) >= 0)?1.0:-1.0;

							 auto qq = -0.5*(cb + signb*sqrt(sqarg));
							 lam[ist] = cc/qq;

						 });
		
		operations::shift(1.0, lambda, sd, phi);
		operations::shift(1.0, lambda, hsd, residual);
	}

	auto maxloc =
#ifdef HAVE_CUDA
		thrust::max_element(normres.begin(), normres.end())
#else
		std   ::max_element(normres.begin(), normres.end())
#endif
	;
	if(maxloc == normres.end()) return 0.0;
	return *maxloc;
}

}
}
#endif

#ifdef INQ_SOLVERS_STEEPEST_DESCENT_UNIT_TEST
#undef INQ_SOLVERS_STEEPEST_DESCENT_UNIT_TEST

#include <basis/trivial.hpp>
#include <operations/matrix_operator.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	
  const int npoint = 100;
  const int nvec = 12;
  
  basis::trivial bas(npoint, boost::mpi3::environment::get_self_instance());
	
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
    basis::field_set<basis::trivial, complex> rhs(bas, nvec);
 
		for(int ip = 0; ip < npoint; ip++){
      for(int ivec = 0; ivec < nvec; ivec++){
        phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
        rhs.matrix()[ip][ivec] = cos(ip*(ivec + 1.0)/2.0);
      }
    }

		const int num_iter = 50;
		
		for(int iter = 0; iter < num_iter; iter++){

			solvers::steepest_descent(diagonal_op, identity, rhs, phi);
			
			auto residual = diagonal_op(phi);
			operations::shift(-1.0, rhs, residual);
			auto normres = operations::overlap_diagonal(residual);
			
      
      tfm::format(std::cout, "  Iteration %4d:\n", iter);
			
      for(int ivec = 0; ivec < phi.set_size(); ivec++){
        if(num_iter - 1 == iter) CHECK(fabs(normres[ivec]) < 1e-8);
				tfm::format(std::cout, "    state %4d  res = %15.10e\n", ivec + 1, real(normres[ivec]));
      }

		}	
 }


}
#endif

