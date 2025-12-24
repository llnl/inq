/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__UTILS__NUM_STR
#define INQ__UTILS__NUM_STR

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstdlib>
#include <utils/calculator.hpp>

#include <tinyformat/tinyformat.h>

namespace inq {
namespace utils {

static inline auto num_to_str(long num){
	char numcstr[12];
	snprintf(numcstr, 11, "%010ld", num);
	return std::string(numcstr);
}

template <typename NumType>
static inline auto num_to_str(std::string const & fmt, NumType const & num){
	std::stringstream ss;
	tfm::format(ss, fmt.c_str(), num);
	return ss.str();
}

template <typename Type>
static inline auto str_to(std::string const &);

template <>
auto str_to<double>(std::string const & str) {
	return utils::calculator::eval(str);
}

template <>
auto str_to<long>(std::string const & str) {
  return std::lround(str_to<double>(str));
}

template <>
[[maybe_unused]] auto str_to<int>(std::string const & str) {
  return int(str_to<long>(str));
}

static inline auto str_to_index(std::string const & str) {
	if(str == "x") return 0;
	if(str == "y") return 1;
	if(str == "z") return 2;
  return -1;
}

}
}
#endif

#ifdef INQ_UTILS_NUM_STR_UNIT_TEST
#undef INQ_UTILS_NUM_STR_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <math/complex.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

  SECTION("number to string"){
    
    CHECK(utils::num_to_str(0)    == "0000000000");
    CHECK(utils::num_to_str(1024) == "0000001024");
    CHECK(utils::num_to_str(-333) == "-000000333");

		CHECK(utils::num_to_str("%20.12f", 1.23456789) == "      1.234567890000");
		CHECK(utils::num_to_str("%.12f", 1.23456789)   == "1.234567890000");		

  }

  SECTION("string to number"){

    CHECK(utils::str_to<double>("0.3") == 0.3_a);
    CHECK(utils::str_to<double>("-56750.25456") == -56750.25456_a);
		CHECK(utils::str_to<double>("1/2") == 0.5_a);
		CHECK(utils::str_to<double>("sqrt(2)") == 1.4142135624_a);

    CHECK(utils::str_to<int>("333") == 333);
    CHECK(utils::str_to<int>("-45") == -45);
		CHECK(utils::str_to<int>("28 - 84") == -56);

    CHECK(utils::str_to<long>("467854545460193") == 467854545460193l);
    CHECK(utils::str_to<long>("-467489690221") == -467489690221l);
		CHECK(utils::str_to<long>("2^32") == 4294967296l);
  }
	
}

#endif

