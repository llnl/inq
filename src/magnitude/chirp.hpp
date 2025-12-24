/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MAGNITUDE__CHIRP
#define INQ__MAGNITUDE__CHIRP

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/quantity.hpp>
#include <utils/lowercase.hpp>
#include <magnitude/time.hpp>

namespace inq {
namespace magnitude {

struct chirp;

auto operator ""_auchirp(long double val){
	return inq::quantity<chirp>::from_atomic_units(val);
}

auto operator ""_invfs2(long double val){
	return inq::quantity<chirp>::from_atomic_units(val/ ((1.0_fs).in_atomic_units() * (1.0_fs).in_atomic_units() ) );
}

auto operator ""_invps2(long double val){
	return inq::quantity<chirp>::from_atomic_units(val/ ((1.0_ps).in_atomic_units() * (1.0_ps).in_atomic_units() ) );
}

struct chirp {
	static auto parse(double value, std::string units){

		units = utils::lowercase(units);
		
		if(units == "auchirp" ) {
			return value*1.0_auchirp;
		} else if (units == "invfs2" ){
			return value*1.0_invfs2;
		} else if (units == "invps2" ){
			return value*1.0_invps2;
		} else {
			throw std::runtime_error("inq error: unknown chirp units '" + units + "'.");
		}
	}
	
};

}

}

#endif

#ifdef INQ_MAGNITUDE_CHIRP_UNIT_TEST
#undef INQ_MAGNITUDE_CHIRP_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;

  {
    auto ch = 100.0_auchirp;
    CHECK(ch.in_atomic_units() == 100.0_a);
  }

  {
    auto ch = 1709.109149248447_invfs2;
    CHECK(ch.in_atomic_units() == 1.0_a);
  }

  {
    auto ch = 1709109149.248447_invps2;
    CHECK(ch.in_atomic_units() == 1.0_a);
  }


	CHECK(chirp::parse(1.0, "auchirp") == 1.0_auchirp);	
	CHECK(chirp::parse(12.0, "invfs2") == 12.0_invfs2);
	CHECK(chirp::parse(1.0, "invps2") == 1.0_invps2);
	CHECK_THROWS(chirp::parse(1.0, "not_a_unit") );	
	
}
#endif
