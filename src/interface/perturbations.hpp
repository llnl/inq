/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__PERTURBATIONS
#define INQ__INTERFACE__PERTURBATIONS

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/runtime_options.hpp>
#include <perturbations/blend.hpp>

namespace inq {
namespace interface {

struct {
	
	constexpr auto name() const {
		return "perturbations";
	}

	constexpr auto one_line() const {
		return "Defines the perturbations in a real-time simulation";
	}
	constexpr auto help() const {
		
		return R""""(

The 'perturbations' command
==================

This command defines the perturbations applied to the system in a real-time simulations.

Note that the perturbations can be applied in different ways depending
on the periodicity of the system. The gauge selection is done
automatically by the code.

These are the uses for the command:

- Shell:  `perturbations`
  Python: `perturbations.status()`

  Without any arguments (or with the status() function in Python),
  `perturbations` prints a the perturbations that are currently
  defined.

  Shell example:  `inq perturbations`.
  Python example: `pinq.perturbations.status()`


- Shell:  `perturbations clear`

  Removes any perturbations currently defined in the system.

  Shell example:  `inq perturbations clear`
  Python example: `pinq.perturbations.clear()`


- Shell:  `perturbations kick <px> <py> <pz>`
  Python: `perturbations.kick([px, py, pz])`

  Specifies a kick perturbation. This corresponds of a uniform
  time-dependent electric field. The time profile is a delta function
  at time 0. The direction and intensity of the field is given by the
  values _px_, _py_ and _pz_ (these values are in cartesian units and
  always considered in atomic units).

  A kick has a flat profile in frequency space, so it will excite all
  frequencies equally. This is useful to obtain linear response
  properties.

  For finite systems this kick is applied in the length gauge as a
  phase to the orbitals (this avoid time-discretization errors). For
  periodic systems, the kick is applied in the velocity gauge as a
  uniform vector potential.

  Shell example:  `inq perturbations kick 0.0 0.0 0.1`
  Python example: `pinq.perturbations.kick([0.0, 0.0, 0.1])`


- Shell:  `perturbations laser <px> <py> <pz> <frequency|wavelength> <f> <units>`
  Python: currently not implemented

  Adds a laser perturbation. The laser is approximated by a
  monochromatic electric field (the magnetic part is ignored). The
  direction and intensity of the field is given by the values _px_,
  _py_ and _pz_ (in cartesian coordinated and always in atomic units).

  You also have to specify the frequency of the laser using the
  keyword 'frequency' followed by its value and its units (you can use
  'eV' or 'THz' for example, check `inq help units` for a full list of
  the energy units available).

  Alternative you can use the keyword 'wavelength' followed by the
  wavelength of the laser and its units (typically 'nm').

  For finite systems the laser is applied in the length gauge as a
  potential. For periodic systems, the laser is applied in the
  velocity gauge as a uniform vector potential.

  Shell examples: `inq perturbations laser  0    0    0.5 frequency  300 THz`
                  `inq perturbations laser -0.01 0.01 0   wavelength 880 nm`


)"""";
	}

	static void status() {
		auto perturbations = perturbations::blend::load(".inq/default_perturbations");
		if(input::environment::global().comm().root()) std::cout << perturbations;
	}
	
	void operator()() const {
		status();
	}

	static void clear() {
		auto perturbations = perturbations::blend::load(".inq/default_perturbations");
		perturbations.clear();
		perturbations.save(input::environment::global().comm(), ".inq/default_perturbations");
	}

	static void kick(vector3<double, cartesian> const & polarization) {
		auto perturbations = perturbations::blend::load(".inq/default_perturbations");
		perturbations.clear();
		auto cell = systems::ions::load(".inq/default_ions").cell();
		perturbations.add(perturbations::kick(cell, polarization));
		perturbations.save(input::environment::global().comm(), ".inq/default_perturbations");
	}

	static void laser(vector3<double, cartesian> const & polarization, quantity<magnitude::energy> freq) {
		auto perturbations = perturbations::blend::load(".inq/default_perturbations");
		perturbations.clear();
		auto cell = systems::ions::load(".inq/default_ions").cell();
		if(cell.periodicity() == 0){
			perturbations.add(perturbations::laser(polarization, freq, perturbations::gauge::length));
		} else {
			perturbations.add(perturbations::laser(polarization, freq, perturbations::gauge::velocity));
		}
		perturbations.save(input::environment::global().comm(), ".inq/default_perturbations");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {
		using namespace magnitude;
		using utils::str_to;
		
		if(args.size() == 0) {
			operator()();
			actions::normal_exit();
		}

		if(args[0] == "clear"){
			if(args.size() != 1) actions::error(input::environment::global().comm(), "The 'perturbations clear' command doesn't take arguments");
			clear();
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "kick"){
			if(args.size() != 4) actions::error(input::environment::global().comm(), "Invalid arguments for 'perturbations kick'");
			kick({str_to<double>(args[1]), str_to<double>(args[2]), str_to<double>(args[3])});
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "laser"){
			if(args.size() != 7) actions::error(input::environment::global().comm(), "Invalid syntax for 'perturbations laser'.\nUse 'inq perturbations laser <px> <py> <pz> frequency <f> <units>'");

			quantity<energy> freq;
			
			if(args[4] == "frequency") {
				freq = energy::parse(str_to<double>(args[5]), args[6]);
			} else if(args[4] == "wavelength") {
				freq = energy::from_wavelength(length::parse(str_to<double>(args[5]), args[6]));
			} else {
				actions::error(input::environment::global().comm(), "Invalid argument " + args[4] + " for 'perturbations laser'.\nUse 'inq perturbations laser <px> <py> <pz> <frequency|wavelength> <f> <units>'");
			}

			laser({str_to<double>(args[1]), str_to<double>(args[2]), str_to<double>(args[3])}, freq);
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		actions::error(input::environment::global().comm(), "Invalid syntax in the perturbations command");
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;

		auto sub = module.def_submodule(name(), help());

		sub.def("status", &status);
		sub.def("clear",  &clear);

		sub.def("kick", [](std::vector<double> const & pol) {
			kick({pol[0], pol[1], pol[2]});
		}, "polarization"_a);

	}
#endif
	
} const perturbations;

}
}
#endif

#ifdef INQ_INTERFACE_PERTURBATIONS_UNIT_TEST
#undef INQ_INTERFACE_PERTURBATIONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
