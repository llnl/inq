/* -*- indent-tabs-mode: t -*- */

#ifndef GPU__ATOMIC
#define GPU__ATOMIC

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <gpu/run.hpp>
#include <math/complex.hpp>
#include <math/vector3.hpp>

namespace gpu {

namespace atomic_op {

template <typename Type1, typename Type2>
GPU_FUNCTION inline Type1 add(Type1 * val, Type2 const & incr){
#ifdef ENABLE_GPU
#ifdef __CUDA_ARCH__
	static_assert(__CUDA_ARCH__ >= 600, "gpu library needs target gpu architecture >= 6.0");
#endif
  return atomicAdd(val, incr);
#else
  auto old_val = *val;
  *val += incr;
  return old_val;
#endif
}

GPU_FUNCTION inline inq::complex add(inq::complex * val, inq::complex const & incr){
	auto re = add((double *) val, inq::real(incr));
	auto im = add(((double *) val) + 1, inq::imag(incr));
	return inq::complex(re, im);
}

template <typename Type, typename Space>
GPU_FUNCTION inline auto add(inq::vector3<Type, Space> * val, inq::vector3<Type, Space> const & incr){
	auto v0 = add((Type *) val + 0, incr[0]) ;
	auto v1 = add((Type *) val + 1, incr[1]) ;
	auto v2 = add((Type *) val + 2, incr[2]) ;
	return inq::vector3<Type, Space>{v0, v1, v2};
}

#ifdef ENABLE_GPU
template <typename Type2>
GPU_FUNCTION inline long add(long * val, Type2 const & incr){
  return add((unsigned long long int *) val, incr);
}

template <typename Type2>
GPU_FUNCTION inline long add(size_t * val, Type2 const & incr){
	static_assert(sizeof(size_t) == sizeof(unsigned long long int));
  return add((unsigned long long int *) val, incr);
}
#endif

}

template <typename Type>
class atomic {
	Type & val_;

public:
	
	GPU_FUNCTION explicit atomic(Type & val):
		val_(val){
	}

	GPU_FUNCTION void operator+=(Type const & incr) {
		gpu::atomic_op::add(&val_, incr);
	}

	GPU_FUNCTION void operator++(int) {
		gpu::atomic_op::add(&val_, 1);
	}

	GPU_FUNCTION void operator++() {
		gpu::atomic_op::add(&val_, 1);
	}
	
};


}
#endif

#ifdef GPURUN__ATOMIC__UNIT_TEST
#undef GPURUN__ATOMIC__UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(GPURUN_TEST_FILE, GPURUN_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
}

#endif

