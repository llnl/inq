/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__IONS
#define INQ__INTERFACE__IONS

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/actions.hpp>
#include <interface/cell.hpp>
#include <interface/runtime_options.hpp>
#include <systems/ions.hpp>

namespace inq {
namespace interface {

struct {
	
	constexpr auto name() const {
		return "ions";
	}

	constexpr auto one_line() const {
		return "Defines the ions in the simulation";
	}

	constexpr auto help() const {
		return R""""(

The 'ions' command
==================

This command defines the ions present in the system and its
coordinates. These can be given one by one using `insert` or from a
file using `file`.

If you are adding ions one by one or reading an XYZ file, you must
first declare the cell for system (see `inq help cell` or
`help(pinq.cell)`).

These are the uses for the command:

- Shell:  `ions`
  Python: `ions.status()`

  Without any arguments (or status() in python), `ions` prints a list of
  the ions currently in the system.

  Shell example:  `inq ions`
  Python example: `pinq.ions.status()`


- Shell:  `ions clear`
  Python: `ions.clear()`

  Removes any ions present in the system. Does not change the cell.

  Shell example:  `inq ions clear`
  Python example: `pinq.ions.clear()`


- Shell:  `ions insert <symbol> <x> <y> <z> <units>`
  Python: `ions.insert(symbol, [x, y, z], units)`

  Add an ion of type _symbol_ at coordinates _x_, _y_, and _z_. The
  units for the coordinates must be specified (check `inq help units`
  for details).

  Shell example:  `inq ions insert He  0.0 0. 0.0 2.0 angstrom'
  Python example: `pinq.ions.insert("He", [0.0, 0.0, 2.0], "Angstrom")`


- Shell:  `ions insert fractional <symbol> <x> <y> <z>`
  Python: `ions.insert_fractional(symbol, [x, y, z])`

  Insert an ion of type _symbol_ at fractional coordinates _x_, _y_,
  and _z_.

  Shell example:  `inq ions insert fractional Si 0.25 0.25 0.25`
  Python example: `pinq.ions.insert_fractional("Si", [0.25, 0.25, 0.25])`


- Shell:  `ions remove <index>`
  Python: `ions.remove(index)`

  Removes the ion given by _index_. Note that the ions indices start
  from 0, not 1.

  Shell example:  `inq ions remove 3`
  Python example: `pinq.ions.remove(3)`


- Shell:  `ions file <filename>`
  Python: `ions.file(filename)`

  Read a coordinate file given by <filename>. The supported formats
  are POSCAR, CIF and XYZ. For POSCAR and CIF both the ions and the
  cell information will be read. For XYZ only the atomic positions are
  read, so the cell must be defined before reading the file.

  Shell example:  `inq ions file diamond.cif`
  Python example: `pinq.ions.file("diamond.cif")`


- Shell:  `ions file <filename> radius <r> <units>`
  Python: `ions.file(filename, radius, units)`

  Reads a coordinate file from <filename> and define a cell around the ions. The cell
  is orthorhombic and finite. The cell has the smallest possible size
  so that walls are at least distance 'r' from any atom. This is
  useful for molecular systems where you need to converge the size of
  the cell.  The units for the radius value must be specified (check
  `inq help units` for details).

  Note: right now this functionality is only implemented for XYZ files.

  Shell example:  `inq ions file glucose.xyz radius 2.0 A`
  Python example: `pinq.ions.file("glucose.xyz", 2.0, "A")`

- Python: `ions.from_ase(atoms)`
- Shell:  (not available)

  Imports the cell and ions from an Atomic Simulation Environment
  (ASE) object. This is only available for the Python interface.

  Python example: `pinq.ions.from_ase(atoms)`


)"""";
	}

	static void status() {
		auto ions = systems::ions::load(".inq/default_ions");		
		if(input::environment::global().comm().root()) std::cout << ions;
	}

	void operator()() const {
		status();
	}

	static void insert(ionic::species const & sp, vector3<quantity<magnitude::length>> const & pos) {
		auto ions = systems::ions::load(".inq/default_ions");
		ions.insert(sp, pos);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void insert_fractional(ionic::species const & sp, vector3<double, contravariant> const & pos) {
		auto ions = systems::ions::load(".inq/default_ions");
		ions.insert_fractional(sp, pos);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void remove(long index) {
		auto ions = systems::ions::load(".inq/default_ions");
		if(index < 0 or index >= ions.size()) actions::error(input::environment::global().comm(), "Invalid ion index in the 'ions remove' command.");
		ions.remove(index);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void clear() {
		auto ions = systems::ions::load(".inq/default_ions");
		ions.clear();
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void file(std::string const & filename) {
		std::string extension = utils::lowercase(filename.substr(filename.find_last_of(".") + 1));

		if(extension == "xyz") {
			auto cell = systems::ions::load(".inq/default_ions").cell();
			auto ions = systems::ions::parse(filename, cell);
			ions.save(input::environment::global().comm(), ".inq/default_ions");
		} else {
			auto ions = systems::ions::parse(filename);
			ions.save(input::environment::global().comm(), ".inq/default_ions");
		}
	}
	
	static void file(std::string const & filename, quantity<magnitude::length> const & radius) {
		auto ions = systems::ions::parse(filename, radius);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		using utils::str_to;
		
		if(args.size() == 0) {
			operator()();
			actions::normal_exit();
		}

		if(args[0] == "clear"){

			if(args.size() != 1) actions::error(input::environment::global().comm(), "The 'ions clear' command doesn't take arguments.");
			clear();
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

 		if(args.size() >= 2 and args[0] == "insert" and args[1] == "fractional"){

			if(args.size() != 6) actions::error(input::environment::global().comm(), "Wrong arguments for ions insert.\nUse: inq ions insert fractional <symbol> <x> <y> <z>");
			
			auto symbol = args[2];
			auto xx = str_to<double>(args[3]);
			auto yy = str_to<double>(args[4]);
			auto zz = str_to<double>(args[5]);
			
			insert_fractional(symbol, {xx, yy, zz});
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "insert"){

			if(args.size() != 6) actions::error(input::environment::global().comm(), "Wrong arguments for ions insert.\nUse: inq ions insert <symbol> <x> <y> <z> <units>");

			auto symbol = args[1];
			auto units = args[5];
			auto xx = magnitude::length::parse(str_to<double>(args[2]), units);
			auto yy = magnitude::length::parse(str_to<double>(args[3]), units);
			auto zz = magnitude::length::parse(str_to<double>(args[4]), units);
			
			insert(symbol, {xx, yy, zz});
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}

		if(args[0] == "remove"){

			if(args.size() != 2) actions::error(input::environment::global().comm(), "Wrong arguments for ions remove.\nUse: inq ions remove <index>");
			
			remove(str_to<long>(args[1]));
			if(not run_opts.quiet) operator()();
			actions::normal_exit();
		}
		
		if(args.size() == 2 and args[0] == "file"){
			file(args[1]);
			if(not run_opts.quiet) {
				interface::cell();
				operator()();
			}
			actions::normal_exit();
		}
		
		if(args.size() == 5 and args[0] == "file" and args[2] == "radius"){
			auto radius = magnitude::length::parse(str_to<double>(args[3]), args[4]);
			
			file(args[1], radius);
			if(not run_opts.quiet) {
				interface::cell();
				operator()();
			}
			actions::normal_exit();
		}
		
		if(input::environment::global().comm().root()) actions::error(input::environment::global().comm(), "Invalid syntax in the ions command");
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;

		auto sub = module.def_submodule(name(), help());

		sub.def("status", &status);
		sub.def("clear",  &clear);
		sub.def("remove", &remove);

		sub.def("insert", [](std::string const & symbol, std::vector<double> const & coords, std::string const & units) {
			auto c0 = magnitude::length::parse(coords[0], units);
			auto c1 = magnitude::length::parse(coords[1], units);
			auto c2 = magnitude::length::parse(coords[2], units);

			insert(symbol, {c0, c1, c2});
		}, "species"_a,  "coordinates"_a,  "units"_a);

		sub.def("insert_fractional", [](std::string const & symbol, std::vector<double> const & coords) {
			insert_fractional(symbol, {coords[0], coords[1], coords[2]});
		}, "species"_a,  "coordinates"_a);

		sub.def("file", [](std::string const & filename) {
			file(filename);
		}, "filename"_a);
		
		sub.def("file", [](std::string const & filename, double radius, std::string const & units) {
			file(filename, magnitude::length::parse(radius, units));
		}, "filename"_a, "radius"_a, "units"_a);
		
		sub.def("from_ase", [](pybind11::object const & atoms) {
			auto ions = systems::ions::import_ase(atoms);
			ions.save(input::environment::global().comm(), ".inq/default_ions");
		}, "atoms"_a);
		
	}
#endif
	
} const ions;

}
}
#endif

#ifdef INQ_INTERFACE_IONS_UNIT_TEST
#undef INQ_INTERFACE_IONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
