/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MAGNITUDE__LENGTH
#define INQ__MAGNITUDE__LENGTH

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/quantity.hpp>
#include <utils/lowercase.hpp>

namespace inq {
namespace magnitude {

struct length;

auto operator ""_b(long double val){
	return inq::quantity<length>::from_atomic_units(val);
}

auto operator ""_bohr(long double val){
	return inq::quantity<length>::from_atomic_units(val);
}

auto operator ""_angstrom(long double val){
	return inq::quantity<length>::from_atomic_units(1.88972612462938*val);
}

auto operator ""_A(long double val){
	return val*1.0_angstrom;
}

auto operator "" _nanometer(long double val){
	return val*10.0*1.0_angstrom;
}

auto operator "" _nm(long double val){
	return val*1.0_nanometer;
}

auto operator "" _picometer(long double val){
	return val/1000.0*1.0_nanometer;
}

auto operator "" _pm(long double val){
	return val*1.0_picometer;
}

static auto const bohr     = inq::magnitude::operator""_bohr(1);

static auto const angstrom = inq::magnitude::operator""_angstrom(1); // Å, AA, Angstrom
static auto const A        = inq::magnitude::operator""_A(1);        // Å, AA, Angstrom

struct length {
	static auto parse(double value, std::string units){

		units = utils::lowercase(units);
		
		if(units == "bohr" or units == "bohrs" or units == "b") {
			return value*1.0_b;
		} else if (units == "angstrom" or units == "angstroms" or units == "a"){
			return value*1.0_A;
		} else if (units == "nanometer" or units == "nanometers" or units == "nm"){
			return value*1.0_nm;
		} else if (units == "picometer" or units == "picometers" or units == "pm"){
			return value*1.0_pm;
		} else {
			throw std::runtime_error("inq error: unknown length units '" + units + "'.");
		}
	}
	
};

}

}

#endif

#ifdef INQ_MAGNITUDE_LENGTH_UNIT_TEST
#undef INQ_MAGNITUDE_LENGTH_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;

  {
    auto le = 100.0_b;
    CHECK(le.in_atomic_units() == 100.0);
  }

  {
    auto le = 100.0_bohr;
    CHECK(le.in_atomic_units() == 100.0);
  }
  
  {
    auto le = 100.0_A;
    CHECK(le.in_atomic_units() == 188.9726124629_a);
  }
  
  {
    auto le = 100.0_angstrom;
    CHECK(le.in_atomic_units() == 188.9726124629_a);
  }

  {
    auto le = 874.21_nanometer;
    CHECK(le.in_atomic_units() == 16520.1747541225_a);
  }

  {
    auto le = 874.21_nm;
    CHECK(le.in_atomic_units() == 16520.1747541225_a);
  }
   
  {
    auto le = 23.333_picometer;
    CHECK(le.in_atomic_units() == 0.440929796659773_a);
  }
  
  {
    auto le = 23.333_pm;
    CHECK(le.in_atomic_units() == 0.440929796659773_a);
  }

	CHECK(length::parse(1.0, "bohr") == 1.0_b);	
	CHECK(length::parse(12.0, "Bohr") == 12.0_b);
	CHECK(length::parse(1.0, "BOHR") == 1.0_b);
	CHECK(length::parse(0.34, "a") == 0.34_A);
	CHECK(length::parse(1.0, "angstrom") == 1.0_A);
	CHECK(length::parse(1.0, "angSTROMS") == 1.0_A);		
	CHECK(length::parse(1.0, "PICOMETER") == 1.0_pm);	
	CHECK(length::parse(3.0, "picometers") == 3.0_pm);
	CHECK(length::parse(1.0, "pm") == 1.0_pm);	
	CHECK(length::parse(-11.0, "NANOmeter") == -11.0_nm);	
	
	CHECK_THROWS(length::parse(1.0, "not_a_unit") == 1.0_b);	
	
}
#endif
