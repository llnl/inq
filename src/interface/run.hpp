/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RUN
#define INQ__INTERFACE__RUN

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/initial_guess.hpp>
#include <ground_state/calculator.hpp>
#include <interface/runtime_options.hpp>
#include <real_time/propagate.hpp>
#include <real_time/results.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "run";
	}

	constexpr auto one_line() const {
		return "Runs the simulation";
	}
	
	constexpr auto help() const {
		return R""""(

The 'run' command
==================

This command runs the actual simulation. It requires all the
simulation parameters to be set before running.

Note that this is the expensive part in the execution on
inq. Depending on the system you are using you might want to execute
this in parallel or through a queuing system.

These are the options available:

-  Shell:  `run ground-state`
   Python: `run.ground_state()`

   Runs a ground-state calculation with fixed ions.

   Shell example:  `inq run ground-state`
   Python example: `pinq.run.ground_state()`


-  Shell:  `run real-time`
   Python: `run.real_time()`

   Runs a real-time simulation.

   Shell example:  `inq run real-time`
   Python example: `pinq.run.real_time()`


-  Shell:  `run resume`
   Python: `run.resume()`

   Restarts the run of a stopped simulation from a checkpoint. For the
   moment this only works for real-time simulations.

   Shell example:  `inq run resume`
   Python example: `pinq.run.resume()`


)"""";
	}

	static void ground_state(runtime_options const & run_opts = {}) {
		auto ions = systems::ions::load(".inq/default_ions");

		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::gamma());

		try { bz = ionic::brillouin::load(".inq/default_brillouin"); }
		catch(...) {
			bz.save(input::environment::global().comm(), ".inq/default_brillouin");
		}
		
		systems::electrons electrons(ions, options::electrons::load(".inq/default_electrons_options"), bz);
 
		if(not electrons.try_load(".inq/default_orbitals")){
			ground_state::initial_guess(ions, electrons);
		}
		auto result = ground_state::calculate(ions, electrons, options::theory::load(".inq/default_theory"), options::ground_state::load(".inq/default_ground_state_options").calculate_forces());

		result.save(input::environment::global().comm(), ".inq/default_results_ground_state");
		electrons.save(".inq/default_orbitals");
	}

	static void real_time(runtime_options const & run_opts = {}) {
		auto ions = systems::ions::load(".inq/default_ions");

		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::gamma());

		try { bz = ionic::brillouin::load(".inq/default_brillouin"); }
		catch(...) {
			bz.save(input::environment::global().comm(), ".inq/default_brillouin");
		}
		
		auto par = input::environment::global().par().kpoints(run_opts.par_kpoints).states(run_opts.par_states).domains(run_opts.par_domains);
		systems::electrons electrons(par, ions, options::electrons::load(".inq/default_electrons_options"), bz);
 
		if(not electrons.try_load(".inq/default_orbitals")) actions::error(input::environment::global().comm(), "Cannot load a ground-state electron configuration for a real-time run.\n Please run a ground-state first.");

		auto opts = options::real_time::load(".inq/default_real_time_options");	
		auto res = real_time::results();
		res.obs = opts.observables_container();
		
		real_time::propagate(ions, electrons, [&res](auto obs){ res(obs); }, options::theory::load(".inq/default_theory"), opts, perturbations::blend::load(".inq/default_perturbations"));
		res.save(input::environment::global().comm(), ".inq/default_results_real_time");

	}

	static void resume(runtime_options const & run_opts = {}) {

		auto dirname = std::string(".inq/default_checkpoint/");
		
		std::optional<std::string> run_type;
		utils::load_optional(dirname + "/type", run_type);

		if(not run_type.has_value()) {
			actions::error(input::environment::global().comm(), "Cannot resume run, a checkpoint was not found.");
		}
		
		if(*run_type != "real-time") {
			actions::error(input::environment::global().comm(), "Unknown checkpoint type '" + *run_type + "'.");
		}

		auto ions = systems::ions::load(".inq/default_ions");
		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::gamma());

		try { bz = ionic::brillouin::load(".inq/default_brillouin"); }
		catch(...) {
			bz.save(input::environment::global().comm(), ".inq/default_brillouin");
		}
		
		systems::electrons electrons(input::environment::global().par().states(), ions, options::electrons::load(".inq/default_electrons_options"), bz);
 
		if(not electrons.try_load(dirname + "/real-time/orbitals")) {
			actions::error(input::environment::global().comm(), "Cannot load the restart electron orbitals.\n The checkpoint must be corrupt");
		}

		auto opts = options::real_time::load(".inq/default_real_time_options");	

		auto res = real_time::results::load(dirname + "/real-time/observables");

		res.obs = opts.observables_container();
		
		real_time::propagate(ions, electrons, [&res](auto obs){ res(obs); },
												 options::theory::load(".inq/default_theory"), opts, perturbations::blend::load(".inq/default_perturbations"),
												 /* start_step = */ res.total_steps);
		res.save(input::environment::global().comm(), ".inq/default_results_real_time");

	}
	
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {
		
		if(args.size() == 0) actions::error(input::environment::global().comm(), "Missing argument to the 'run' command");
		
		if(args.size() == 1 and args[0] == "ground-state") {
			ground_state(run_opts);
			actions::normal_exit();
		}
				
		if(args.size() == 1 and args[0] == "real-time") {
			real_time(run_opts);
			actions::normal_exit();
		}
						
		if(args.size() == 1 and args[0] == "resume") {
			resume(run_opts);
			actions::normal_exit();
		}
		
		actions::error(input::environment::global().comm(), "Invalid syntax in the 'run' command");
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule(name(), help());
		sub.def("ground_state", [](){ground_state();});
		sub.def("real_time",    [](){real_time();});
		sub.def("resume",       [](){resume();});
		
	}
#endif

} const run;

}
}
#endif

#ifdef INQ_INTERFACE_RUN_UNIT_TEST
#undef INQ_INTERFACE_RUN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
