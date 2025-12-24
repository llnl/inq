/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__KPOINTS
#define INQ__INTERFACE__KPOINTS

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/runtime_options.hpp>
#include <ionic/brillouin.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "kpoints";
	}

	constexpr auto one_line() const {
		return "Specifies the kpoints used to sample the Brillouin zone in the simulation";
	}
	
	constexpr auto help() const {
		
		return R""""(
The 'kpoints' command
==================

This command defines the kpoints to be used in the simulation. Kpoints
have to be given after the ions are defined.

Note that right now inq doesn't use symmetries to reduce the kpoint
grid.

These are the options available:

-  Shell:  `kpoints`
   Python: `kpoints.status()`

   Prints a list of the current kpoints in the simulation.

   Shell example:  `inq kpoints`
   Python example: `pinq.kpoints.status()`


-  Shell:  `kpoints gamma`
   Python: `kpoints.gamma()`

   Tells the system to only use the 0 0 0 point. This is the default.

   Shell example:  `inq kpoints gamma`
   Python example: `pinq.kpoints.gamma()`


-  Shell:  `kpoints grid <nx> <ny> <nz>`
   Python: `kpoints.grid(nx, ny, nz)`

   Define a uniform grid to sample the Brillouin zone with sizes _nx_,
   _ny_ and _nz_. This generates a non-shifted grid that will include
   gamma.

   Shell example:  `inq kpoints grid 8 8 8`
   Pyhton example: `pinq.kpoints.grid(8, 8, 8)`


-  Shell:  `kpoints shifted grid <nx> <ny> <nz>`
   Python: `kpoints.shifted_grid(nx, ny, nz)`

   Define a shifted uniform grid to sample the Brillouin zone with sizes _nx_,
   _ny_ and _nz_. This grid doesn't include gamma.

   Shell example:  `inq kpoints shifted grid 4 4 4`
   Python example: `pinq.kpoints.shifted_grid(4, 4, 4)`


-  Shell:  `kpoints insert <kx> <ky> <kz> <w>`
   Python: `kpoints.insert(kx, ky, kz, w)`

   Add a kpoint to the grid with coordinates _kx_, _ky_, _kz_, and
   weight _w_. The kpoint must be given in covariant coordinates in the
   range [-0.5, 0.5).

   Shell example:  `inq kpoints insert 0.25 0.25 0.25 1.0`
   Python example: `pinq.kpoints.insert(0.25, 0.25, 0.25, 1.0)`


-  Shell:  `kpoints clear`
   Python: `kpoints.clear()`

   Removes all the kpoints defined in the grid.

   Shell example:  `inq kpoints clear`
   Python example: `pinq.kpoints.clear()`


)"""";
	}

  static void status() {
		auto bz = ionic::brillouin::load(".inq/default_brillouin");
		if(input::environment::global().comm().root()) std::cout << bz;
	}

	void operator()() const {
		status();
	}

  static void gamma() {
		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::gamma());
		bz.save(input::environment::global().comm(), ".inq/default_brillouin");
	}

  static void grid(int nx, int ny, int nz) {
		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::grid({nx, ny, nz}));
		bz.save(input::environment::global().comm(), ".inq/default_brillouin");
	}

  static void shifted_grid(int nx, int ny, int nz) {
		auto bz = ionic::brillouin(systems::ions::load(".inq/default_ions"), input::kpoints::grid({nx, ny, nz}, true));
		bz.save(input::environment::global().comm(), ".inq/default_brillouin");
	}

  static void insert(double const & kx, double const & ky, double const & kz, double const & weight) {
    auto bz = ionic::brillouin{};
    try { bz = ionic::brillouin::load(".inq/default_brillouin"); }
    catch(...){  }
    bz.insert({kx, ky, kz}, weight);
		bz.save(input::environment::global().comm(), ".inq/default_brillouin");
  }

  static void clear() {
    auto bz = ionic::brillouin{};
    bz.save(input::environment::global().comm(), ".inq/default_brillouin");
  }
  
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		using utils::str_to;
		
    if(args.size() == 0) {
			operator()();
			actions::normal_exit();
		}

    if(args.size() == 1 and args[0] == "gamma") {
			gamma();
      if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

    if(args.size() == 4 and args[0] == "grid") {
      
      grid(str_to<int>(args[1]), str_to<int>(args[2]), str_to<int>(args[3]));
      if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

    if(args.size() == 4 and args[0] == "shifted-grid") {
      
      shifted_grid(str_to<int>(args[1]), str_to<int>(args[2]), str_to<int>(args[3]));
      if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
    
    if(args.size() == 5 and args[0] == "insert") {
      
      insert(str_to<double>(args[1]), str_to<double>(args[2]), str_to<double>(args[3]), str_to<double>(args[4]));
      if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
    
    if(args.size() == 1 and args[0] == "clear") {
			clear();
      if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
    
		actions::error(input::environment::global().comm(), "Invalid syntax in the 'kpoints' command");
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule(name(), help());
		sub.def("status",         &status);
		sub.def("gamma",        &gamma);
		sub.def("grid",         &grid, "nx"_a, "ny"_a, "nz"_a);
		sub.def("shifted_grid", &shifted_grid, "nx"_a, "ny"_a, "nz"_a);
		sub.def("insert",       &insert, "kx"_a, "ky"_a, "kz"_a, "weight"_a);
		sub.def("clear",        &clear);
		
	}
#endif	
	
} const kpoints;

}
}
#endif

#ifdef INQ_INTERFACE_KPOINTS_UNIT_TEST
#undef INQ_INTERFACE_KPOINTS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
