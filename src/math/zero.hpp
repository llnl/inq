/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATH__ZERO
#define INQ__MATH__ZERO

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/vector3.hpp>

#include <cassert>
#include <array>
#include <cmath>

#include <tinyformat/tinyformat.h>

namespace inq {

template <typename Type>
GPU_FUNCTION constexpr auto zero(){
  return Type{};
}

struct zero_t {
};

auto zero() {
	return zero_t{};
}

auto operator==(complex const & aa, zero_t) {
	return aa == zero<complex>();
}

template <typename Type>
auto operator==(Type const & aa, zero_t) {
	return aa == zero<Type>();
}

auto operator==(zero_t z, complex const & aa) {
	return aa == z;
}

template <typename Type>
auto operator==(zero_t z, Type const & aa) {
	return aa == z;
}

}
#endif

#ifdef INQ_MATH_ZERO_UNIT_TEST
#undef INQ_MATH_ZERO_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	
	CHECK(zero<double>() == 0.0);
	CHECK(zero<complex>() == complex{0.0, 0.0});
	CHECK(zero<vector3<double>>() == vector3<double>{0.0, 0.0, 0.0});
	CHECK(zero<vector3<complex>>() == vector3<complex>{complex{0.0, 0.0}, complex{0.0, 0.0}, complex{0.0, 0.0}});
	CHECK(zero<vector3<vector3<double>>>() == vector3<vector3<double>>{vector3<double>{0.0, 0.0, 0.0}, vector3<double>{0.0, 0.0, 0.0}, vector3<double>{0.0, 0.0, 0.0}});
	
	CHECK(zero() == 0.0);
	CHECK(zero() == complex{0.0, 0.0});
	CHECK(zero() == vector3<double>{0.0, 0.0, 0.0});
	CHECK(zero() == vector3<complex>{complex{0.0, 0.0}, complex{0.0, 0.0}, complex{0.0, 0.0}});

	//	double aa = zero();
	//	CHECK(aa == 0.0);
	
}

#endif
