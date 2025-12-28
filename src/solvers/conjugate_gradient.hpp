/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SOLVERS__CONJUGATE_GRADIENT
#define INQ__SOLVERS__CONJUGATE_GRADIENT

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
double conjugate_gradient(const operator_type & op, const preconditioner_type &, field_set_type & bb, field_set_type & xx){
	CALI_CXX_MARK_SCOPE("solver::conjugate_gradient");

	auto const num_steps = 5;
	auto const nst = xx.local_set_size();
 
	auto residual = op(xx);
	gpu::run(nst, xx.basis().local_size(), [bbp = begin(bb.matrix()), res = begin(residual.matrix())] GPU_LAMBDA (auto ist, auto ip){
		res[ip][ist] = bbp[ip][ist] - res[ip][ist];
	});
	
	auto search = residual;
	
	for(int istep = 0; istep < num_steps; istep++){

		auto gamma = operations::overlap_diagonal(residual);

		auto opsearch = op(search);

		auto alpha = operations::overlap_diagonal(search, opsearch);
    gpu::run(nst, [al = begin(alpha), ga = begin(gamma)] GPU_LAMBDA (auto ii) { al[ii] = ga[ii]/al[ii]; });

		operations::shift(-1.0, alpha, opsearch, residual);
		operations::shift(1.0, alpha, search, xx);

		auto beta = operations::overlap_diagonal(residual);
		gpu::run(nst, [be = begin(beta), ga = begin(gamma)] GPU_LAMBDA (auto ii) { be[ii] /= ga[ii]; });

		gpu::run(nst, xx.basis().local_size(), [sea = begin(search.matrix()), res = begin(residual.matrix()), be = begin(beta)] GPU_LAMBDA (auto ist, auto ip){
      sea[ip][ist] = res[ip][ist] + be[ist]*sea[ip][ist];
    });
		
	}

	auto normres = operations::overlap_diagonal(residual);

  auto maxloc =
#ifdef HAVE_CUDA
		thrust::
#else
		std   ::
#endif
    max_element(normres.begin(), normres.end(), [] GPU_LAMBDA (auto aa, auto bb) {return fabs(aa) < fabs(bb); });
  
  if(maxloc == normres.end()) return 0.0;
  return fabs(*maxloc);
}

}
}
#endif

#ifdef INQ_SOLVERS_CONJUGATE_GRADIENT_UNIT_TEST
#undef INQ_SOLVERS_CONJUGATE_GRADIENT_UNIT_TEST

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
    basis::field_set<basis::trivial, complex> rhs(bas, nvec);
 
		for(int ip = 0; ip < npoint; ip++){
      for(int ivec = 0; ivec < nvec; ivec++){
        phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
        rhs.matrix()[ip][ivec] = cos(ip*(ivec + 1.0)/2.0);
      }
    }

		const int num_iter = 50;
		
		for(int iter = 0; iter < num_iter; iter++){

			solvers::conjugate_gradient(diagonal_op, identity, rhs, phi);
			
			auto residual = diagonal_op(phi);
			operations::shift(-1.0, rhs, residual);
			auto normres = operations::overlap_diagonal(residual);
			
      
      fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
			
      for(int ivec = 0; ivec < phi.set_size(); ivec++){
        if(num_iter - 1 == iter) CHECK(fabs(normres[ivec]) < 1e-8);
				fmt::print(std::cout, "    state {:4d}  res = {:15.10e}\n", ivec + 1, real(normres[ivec]));
      }

		}	
 }


}
#endif

