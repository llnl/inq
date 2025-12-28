/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__UTILS__MATCH
#define INQ__UTILS__MATCH

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <array>
#include <cmath>

#include <spdlog/spdlog.h> //for fmt
#include <spdlog/fmt/ostr.h>

namespace inq {
namespace utils {

  class match {

  public:
		
		match(double arg_tol)
      :tol_(arg_tol),
       ok_(true)
		{
		}

    template <class Type>
    auto check(const std::string & match_name, const Type & value, const Type & reference, double tol = 0.0){

			if(tol == 0.0) tol = tol_;
			
      auto diff = fabs(reference - value);
      
      if(diff > tol){
				
        fmt::print(std::cout, "\nMatch '{}': FAILED\n", match_name);
        fmt::print(std::cout, "  calculated value = {:.12f}\n", value);
        fmt::print(std::cout, "  reference value  = {:.12f}\n", reference);
        fmt::print(std::cout, "  difference       = {:.1e}\n", diff);
				fmt::print(std::cout, "  tolerance        = {:.1e}\n\n", tol);
        ok_ = false;
        return false;
      } else {
        fmt::print(std::cout, "Match '{}': SUCCESS (value = {:.12f} , diff = {:.1e})\n", match_name, value, diff);
        return true;
      }
    }

    auto ok() const {
      return ok_;
    }

    auto fail() const {
      return not ok();
    }

		void operator&=(bool value) {
			ok_ = ok_ and value;
		}
    
	protected:

    double tol_;
    bool ok_;   
    
  };
}
}
#endif

#ifdef INQ_UTILS_MATCH_UNIT_TEST
#undef INQ_UTILS_MATCH_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	inq::utils::match mtc(1e-7);

  CHECK(mtc.ok());
  CHECK(not mtc.fail());  
  
  CHECK(mtc.check("test true", 10.0, 10.0 + 1e-8));

  CHECK(mtc.ok());
  CHECK(not mtc.fail());  

  CHECK(not mtc.check("test false", 3.0, 4.0));

  CHECK(not mtc.ok());
  CHECK(mtc.fail());
  
}
#endif
