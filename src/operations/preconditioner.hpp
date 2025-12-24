/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPERATIONS__PRECONDITIONER
#define INQ__OPERATIONS__PRECONDITIONER

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <basis/fourier_space.hpp>
#include <operations/transform.hpp>
#include <states/orbital_set.hpp>

#include <cstdlib>

#include <utils/profiling.hpp>

#include <operations/overlap_diagonal.hpp>

namespace inq {
namespace operations {

class preconditioner {
	
	// Implements the preconditioner of Teter, Payne and Allan, Phys. Rev. B, 40 12255 (1989)
public:
	
	GPU_FUNCTION static auto k_function(double x) {

		// The original function of the TPA paper
		auto num = ((8.0*x + 12.0)*x + 18.0)*x + 27.0;
		auto den = (((16.0*x + 8.0)*x + 12.0)*x + 18.0)*x + 27.0;

		return num/den;
	}
		
	template <class type>
	void operator()(states::orbital_set<basis::fourier_space, type> & phi) const {

		CALI_MARK_BEGIN("preconditioner_reduction");
		auto kinetic = operations::overlap_diagonal_normalized(laplacian(phi, -0.5), phi, operations::real_part{});
		CALI_MARK_END("preconditioner_reduction");

		{ CALI_CXX_MARK_SCOPE("preconditioner_apply");
			gpu::run(phi.local_set_size(), phi.basis().local_sizes()[2], phi.basis().local_sizes()[1], phi.basis().local_sizes()[0], 
							 [kine = begin(kinetic), phcub = begin(phi.hypercubic()), point_op = phi.basis().point_op(), spinor_dim = phi.spinor_dim()]
							 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
								 auto lapl = -0.5*(-point_op.g2(i0, i1, i2));
								 phcub[i0][i1][i2][ist] = k_function(lapl/kine[ist/spinor_dim])*phcub[i0][i1][i2][ist];
							 });
		}
	}

	template <class type>
	void operator()(states::orbital_set<basis::real_space, type> & phi) const {
			
		auto fphi = operations::transform::to_fourier(phi);
		operator()(fphi);
		phi = operations::transform::to_real(fphi);
	}

};

class no_preconditioner {
public:
	template <class Type>
	void operator()(Type &) const {
	}
};

}
}
#endif

#ifdef INQ_OPERATIONS_PRECONDITIONER_UNIT_TEST
#undef INQ_OPERATIONS_PRECONDITIONER_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
