/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__ELECTRONS
#define INQ__INTERFACE__ELECTRONS

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
		return "electrons";
	}

	constexpr auto one_line() const {
		return "Defines the electrons in the simulation and how they are represented.";
	}

	constexpr auto help() const {
		
		return R""""(

The 'electrons' command
==================

This command defines the electrons that are in the system and how they
are going to be represented through several values that can be set by
the user.

- Shell:  `electrons`
  Python: `electrons.status()`

  When no arguments are given (or the `status` function in Python),
  `electrons` will just print the currently defined options (including
  default values).

  Shell example:  `inq electrons`
  Python example: `pinq.electrons.status()`

- Shell:  `electrons cutoff <value> <units>`
  Python: `electrons.cutoff(value, units)`

  Sets the energy cutoff for the simulation grid. A higher cutoff
  implies a more precise, but more costly, simulation. The value must
  be followed by its units, check `inq help units` for details on what
  units are available.

  Shell example:  `inq electrons cutoff 30.0 Ry`
  Python example: `pinq.electrons.cutoff(30.0, "Ry)`


- Shell:  `electrons spacing <value> <units>`
  Python: `electrons.spacing(value, units)`

  As an alternative to the cutoff, you can sets the spacing for the
  simulation grid. A lower spacing implies a more precise, but more
  costly, simulation. The value must be followed by its length units,
  check `inq help units` for details on what units are available.

  Shell example:  `inq electrons spacing 0.23 A`
  Python example: `pinq.electrons.spacing(0.23, "A")`


- Shell:  `electrons spin <value>`
  Python: `electrons.spin_unpolarized()`
          `electrons.spin_polarized()`
          `electrons.spin_non_collinear()`

  Sets the spin configuration used in the simulation. In the command
  line interface this is selected by an argument whose values can be
  'unpolarized' (the default), 'polarized' and 'non-collinear'. For
  Python there are different functions for each value.

  Shell example:  `inq electrons spin polarized`
  Python example: `pinq.electrons.spin_polarized()`


- Shell:  `electrons extra-electrons <value>`
  Python: `electrons.extra_electrons(value)`

  Inq determines the number of electrons from the ions present in the
  system. Using this variable you can add or remove electrons from the
  simulation. Positive values add electrons and negative values remove
  them. The number of electrons does not have to be an integer, you
  can add fractions of an electrons (given as a decimal number).

  Note that in first principles simulations the electrons are not
  associated a priory to a certain atom. So when you add an electron
  there is no concept of 'where' you put it. This will be determined
  by the ground-state optimization.

  Shell example:  `inq electrons extra-electrons -0.5`
  Python example: `pinq.electrons.extra_electrons(-0.5)`

- Shell example:  `electrons extra-states <value>`
  Python example: `electrons.extra_states(value)`

  Inq automatically selects a number of states (orbitals, bands) that is
  enough to represent all the electrons in the system. In many cases
  you want additional states, and you do that using this command. The
  value must be a positive integer.

  Extra-states are necessary when setting an electronic temperature
  and to improve ground-state convergence.

  Shell example:  `inq electrons extra-states 2`.
  Python example: `pinq.electrons.extra_states(2)`.


- Shell:  `electrons temperature <value> <units>`
  Python: `electrons.temperature(value, units)`

  This command sets the temperature of the electrons in the
  ground-state optimization. The value must be positive and the units
  must be given. Check `inq help units` for details on what units are
  available. Most likely you want to use 'eV' or 'K'.

  Note that when you add a temperature you also need to specify
  extra-states.

  Shell example:  `inq electrons temperature 273.15 Kelvin`
  Pyhton example: `pinq.electrons.temperature(273.15, "Kelvin")`


- Shell:  `electrons density-cutoff <factor>`
  Python: `electrons.density_cutoff(factor)`

  Sets the energy cutoff for the grid to represent the density. It
  must be given as a factor over the main cutoff. Currently the only
  accepted values are 1 and 4. The default is 1.

  Shell example:  `inq electrons density-cutoff 4`
  Python example: `pinq.electrons.density_cutoff(4)`


)"""";
	}

	static void status() {
		auto el_opts = options::electrons::load(".inq/default_electrons_options");
		if(input::environment::global().comm().root()) std::cout << el_opts;
	}
	
	void operator()() const {
		status();
	}

	static void extra_states(int nstates) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").extra_states(nstates);
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}

	static void extra_electrons(double nelectrons) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").extra_electrons(nelectrons);
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}
	
	static void cutoff(quantity<magnitude::energy> ecut) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").cutoff(ecut);
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}

	static void spacing(quantity<magnitude::length> const & val) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").spacing(val);
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}
	
	static void spin_unpolarized() {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").spin_unpolarized();
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}

	static void spin_polarized() {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").spin_polarized();
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}

	static void spin_non_collinear() {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").spin_non_collinear();
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}

	static void temperature(quantity<magnitude::energy> temp) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").temperature(temp);
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}
	
	static void density_cutoff(double factor) {
		auto el_opts = options::electrons::load(".inq/default_electrons_options").density_factor(sqrt(factor));
		el_opts.save(input::environment::global().comm(), ".inq/default_electrons_options");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		using utils::str_to;
		
		if(args.size() == 0) {
			operator()();
			actions::normal_exit();
		}
		
		if(args[0] == "extra-states"){

			if(args.size() == 1) actions::error(input::environment::global().comm(), "Missing extra_states argument");
			if(args.size() >= 3) actions::error(input::environment::global().comm(), "Too many arguments to extra_states argument");

			extra_states(str_to<int>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		if(args[0] == "extra-electrons"){

			if(args.size() == 1) actions::error(input::environment::global().comm(), "Missing extra_electrons argument");
			if(args.size() >= 3) actions::error(input::environment::global().comm(), "Too many arguments to extra_electrons argument");

			extra_electrons(str_to<double>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "cutoff"){

			if(args.size() < 3) actions::error(input::environment::global().comm(), "Missing cutoff arguments. Use 'cutoff <value> <units>'");
			if(args.size() > 3) actions::error(input::environment::global().comm(), "Too many arguments to cutoff argument");

			cutoff(magnitude::energy::parse(str_to<double>(args[1]), args[2]));
			
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "spacing"){

			if(args.size() < 3) actions::error(input::environment::global().comm(), "Missing spacing arguments. Use 'spacing <value> <units>'");
			if(args.size() > 3) actions::error(input::environment::global().comm(), "Too many arguments to spacing argument");

			spacing(magnitude::length::parse(str_to<double>(args[1]), args[2]));
			
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		if(args.size() == 2 and args[0] == "spin" and args[1] == "unpolarized"){
			spin_unpolarized();
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "spin" and args[1] == "polarized"){
			spin_polarized();
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "spin" and args[1] == "non-collinear") {
			spin_non_collinear();
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "temperature"){

			if(args.size() < 3) actions::error(input::environment::global().comm(), "Missing temperature arguments. Use 'temperature <value> <units>'.");
			if(args.size() > 3) actions::error(input::environment::global().comm(), "Too many arguments to temperature argument.  Use 'temperature <value> <units>'.");

			temperature(magnitude::energy::parse(str_to<double>(args[1]), args[2]));
			
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "density-cutoff"){

			if(args.size() == 1) actions::error(input::environment::global().comm(), "Missing density-cutoff argument");
			if(args.size() >= 3) actions::error(input::environment::global().comm(), "Too many arguments to density-cutoff argument");

			density_cutoff(str_to<double>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		actions::error(input::environment::global().comm(), "Invalid syntax in the 'electrons' command");
	}
	
#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;

		auto sub = module.def_submodule(name(), help());
		
		sub.def("status", &status);
		sub.def("extra_states", &extra_states, "num_extra_states"_a);
		sub.def("extra_electrons", &extra_electrons, "num_extra_electrons"_a);
		sub.def("spin_unpolarized", &spin_unpolarized);
		sub.def("spin_polarized", &spin_polarized);
		sub.def("spin_non_collinear", &spin_non_collinear);
		sub.def("density_cutoff", &density_cutoff);
		
		sub.def("cutoff", [](double ecut, std::string const & units) {
			cutoff(magnitude::energy::parse(ecut, units));
		}, "cutoff_energy"_a, "units"_a);

		sub.def("spacing", [](double spac, std::string const & units) {
			spacing(magnitude::length::parse(spac, units));
		}, "grid_spacing"_a, "units"_a);

		sub.def("temperature", [](double temp, std::string const & units) {
			temperature(magnitude::energy::parse(temp, units));
		}, "electronic_temperature"_a, "units"_a);
		
	}
#endif
	
} const electrons;

}
}
#endif

#ifdef INQ_INTERFACE_ELECTRONS_UNIT_TEST
#undef INQ_INTERFACE_ELECTRONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
