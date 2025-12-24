/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__GROUND_STATE
#define INQ__INTERFACE__GROUND_STATE

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/actions.hpp>
#include <interface/runtime_options.hpp>
#include <systems/electrons.hpp>

namespace inq {
namespace interface {

struct {
		
	constexpr auto name() const {
		return "ground-state";
	}

	constexpr auto one_line() const {
		return "Defines how the ground state is calculated";
	}
	
	constexpr auto help() const {
		return R""""(

The 'ground-state' command
==================

This command defines the options for ground-state self-consistency
calculations. These are the available options:

- Shell:  `ground-state`
  Python: `ground_state.status()`

  When no arguments are given, `ground-state` will just print the
  currently defined options (including default values).

  Shell example:  `inq ground-state`
  Python example: `pinq.ground_state.status()`


- Shell:  `ground-state max-steps <value>`
  Python: `ground_state.max_steps(value)`

  Sets the maximum number of self-consistency steps that will be
  done. The default value is 200.

  Shell example:  `inq ground-state max-steps 100`
  Python example: `pinq.ground_state.max_steps(100)`


- Shell:  `ground-state tolerance <value>`
  Python: `ground_state.tolerance(value)`

  The tolerance used to consider that the self-consistency iteration
  is converged. The default value is 1e-6.

  Shell example:  `inq ground-state tolerance 1e-9`
  Python example: `pinq.ground_state.tolerance(1e-9)`


- Shell:  `ground-state mixing <value>`
  Python: `ground_state.mixing(value)`

  Set the mixing factor for the self-consistency. The default value is 0.3.

  Shell example:  `inq ground-state mixing 0.1`
  Python example: `inq.ground_state.mixing(0.1)`


)"""";
	}

	static void status() {
		auto gs_opts = options::ground_state::load(".inq/default_ground_state_options");
		if(input::environment::global().comm().root()) std::cout << gs_opts;
	}

	void operator()() const {
		status();
	}

	static void max_steps(int nsteps) {
		using namespace magnitude;

		auto gs_opts = options::ground_state::load(".inq/default_ground_state_options").max_steps(nsteps);
		gs_opts.save(input::environment::global().comm(), ".inq/default_ground_state_options");
	}
	
	static void tolerance(double tol) {
		using namespace magnitude;

		auto gs_opts = options::ground_state::load(".inq/default_ground_state_options").energy_tolerance(tol*1.0_Ha);
		gs_opts.save(input::environment::global().comm(), ".inq/default_ground_state_options");
	}
	
	static void mixing(double factor) {
		using namespace magnitude;

		auto gs_opts = options::ground_state::load(".inq/default_ground_state_options").mixing(factor);
		gs_opts.save(input::environment::global().comm(), ".inq/default_ground_state_options");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {
		using utils::str_to;
		
		if(args.size() == 0) {
			operator()();
			actions::normal_exit();
		}

		if(args.size() == 2 and (args[0] == "max-steps")){
			max_steps(str_to<int>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		if(args.size() == 2 and (args[0] == "tolerance")){
			tolerance(str_to<double>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args.size() == 2 and (args[0] == "mixing")){
			mixing(str_to<double>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		actions::error(input::environment::global().comm(), "Invalid syntax in 'ground-state' command");
	}
	
#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule("ground_state", help());
		sub.def("status",      &status);
		sub.def("max_steps", &max_steps);
		sub.def("tolerance", &tolerance);
		sub.def("mixing",    &mixing);
		
	}
#endif
	
} const ground_state;

}
}
#endif

#ifdef INQ_INTERFACE_GROUND_STATE_UNIT_TEST
#undef INQ_INTERFACE_GROUND_STATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
