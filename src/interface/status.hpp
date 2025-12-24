/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__STATUS
#define INQ__INTERFACE__STATUS

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/cell.hpp>
#include <interface/ions.hpp>
#include <interface/electrons.hpp>
#include <interface/kpoints.hpp>
#include <interface/ground_state.hpp>
#include <interface/results_ground_state.hpp>
#include <interface/real_time.hpp>
#include <interface/results_real_time.hpp>
#include <interface/species.hpp>
#include <interface/theory.hpp>

namespace inq {
namespace interface {

struct {
	
	constexpr auto name() const {
		return "status";
	}

	constexpr auto one_line() const {
		return "Shows the status of the currently simulation";
	}

	constexpr auto help() const {
		return R""""(

The 'status' command
==================

Prints all the currently defined parameters and available results.

)"""";
	}

	static void status() {
		interface::cell.status();
		interface::ions.status();
		interface::species.status();
		interface::kpoints.status();
		interface::electrons.status();
		interface::theory.status();
		interface::ground_state.status();
		interface::results_ground_state.status();
		interface::real_time.status();
		interface::results_real_time.status();
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		using utils::str_to;
		
		if(args.size() == 0) {
			status();
			actions::normal_exit();
		}
		
		if(input::environment::global().comm().root()) actions::error(input::environment::global().comm(), "Invalid syntax in the status command");
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		module.def("status", &status);
	}
#endif
	
} const status;

}
}
#endif

#ifdef INQ_INTERFACE_STATUS_UNIT_TEST
#undef INQ_INTERFACE_STATUS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
