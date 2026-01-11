/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MAGNITUDE__TIME
#define INQ__MAGNITUDE__TIME

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/quantity.hpp>
#include <magnitude/energy.hpp>

namespace inq {
namespace magnitude {

struct time;

auto operator ""_atomictime(long double val){
	return inq::quantity<time>::from_atomic_units(val);
}

auto operator ""_atomictimeunits(long double val){
	return inq::quantity<time>::from_atomic_units(val);
}

auto operator ""_atu(long double val){
	return inq::quantity<time>::from_atomic_units(val);
}

auto operator ""_attosecond(long double val){
	return inq::quantity<time>::from_atomic_units(0.0413413733352975*val);
}

auto operator ""_as(long double val){
	return val*1.0_attosecond;
}

auto operator ""_femtosecond(long double val){
	return val*1000.0_attosecond;
}

auto operator ""_fs(long double val){
	return val*1.0_femtosecond;
}

auto operator ""_picosecond(long double val){
	return val*1000.0_femtosecond;
}

auto operator ""_ps(long double val){
	return val*1.0_picosecond;
}

auto operator ""_nanosecond(long double val){
	return val*1000.0_picosecond;
}

auto operator ""_ns(long double val){
	return val*1.0_nanosecond;
}

auto operator/(double num, quantity<energy> den){
  return quantity<time>::from_atomic_units(num/den.in_atomic_units());
}

struct time {
	static auto parse(double value, std::string units){

		units = utils::lowercase(units);
		
		if(units == "atomictime" or units == "atomictimeunits"  or units == "atomictimeunit" or units == "atu") {
			return value*1.0_atu;
		} else if (units == "attosecond" or units == "attoseconds" or units == "as"){
			return value*1.0_as;
		} else if (units == "femtosecond" or units == "femtoseconds" or units == "fs"){
			return value*1.0_fs;
		} else if (units == "picosecond" or units == "picoseconds" or units == "ps"){
			return value*1.0_ps;
		} else if (units == "nanosecond" or units == "nanoseconds" or units == "ns"){
			return value*1.0_ns;

		} else {
			throw std::runtime_error("inq error: unknown time units '" + units + "'.");
		}
	}
	
};

}
}
#endif

#ifdef INQ_MAGNITUDE_TIME_UNIT_TEST
#undef INQ_MAGNITUDE_TIME_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;

  {
    auto ti = 100.0_atomictime;
    CHECK(ti.in_atomic_units() == 100.0_a);
  }

  {
    auto ti = 0.9562_attosecond;
    CHECK(ti.in_atomic_units() == 0.0395306211832115_a);
  }
  
  {
    auto ti = 0.9562_as;
    CHECK(ti.in_atomic_units() == 0.0395306211832115_a);
  }

  {
    auto ti = 43.27_femtosecond;
    CHECK(ti.in_atomic_units() == 1788.84122421832_a);
  }
   
  {
    auto ti = 43.27_fs;
    CHECK(ti.in_atomic_units() == 1788.84122421832_a);
  }
  
  {
    auto ti = 17.77_picosecond;
    CHECK(ti.in_atomic_units() == 734636.204168237_a);
  }
  
  {
    auto ti = 17.77_ps;
    CHECK(ti.in_atomic_units() == 734636.204168237_a);
  }

  {
    auto ti = 0.03974_nanosecond;
    CHECK(ti.in_atomic_units() == 1642906.17634472_a);
  }

  {
    auto ti = 0.03974_ns;
    CHECK(ti.in_atomic_units() == 1642906.17634472_a);
  }
  
  {
    auto ti = 1.0/1.0_Ha;
    CHECK(ti.in_atomic_units() == 1.0_a);
  }

  {
    auto ti = 1.0/100.0_eV;
    CHECK(ti.in_atomic_units() == 0.272113862460642_a);
  }

	CHECK(time::parse(1.0, "atu") == 1.0_atu);
	CHECK(time::parse(-1.0, "atomictimeunits") == -1.0_atu);
	CHECK(time::parse(1.0, "ATOMICTIME") == 1.0_atu);
	CHECK(time::parse(10.0, "as") == 10.0_as);
	CHECK(time::parse(1.0, "nanoseconds") == 1.0_ns);
	CHECK(time::parse(0.1, "femtoSECONDS") == 0.1_fs);
	CHECK(time::parse(1.0, "ps") == 1.0_ps);
	CHECK(time::parse(3.0, "Picosecond") == 3.0_ps);
	CHECK(time::parse(1.0, "nanoseconds") == 1.0_ns);
	CHECK(time::parse(-12.0, "nS") == -12.0_ns);
	
	CHECK_THROWS(time::parse(1.0, "not_a_unit"));
	
}
#endif

