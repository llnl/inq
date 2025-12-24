/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__SPECIES
#define INQ__INTERFACE__SPECIES

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <ground_state/initial_guess.hpp>
#include <real_time/propagate.hpp>
#include <real_time/results.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "species";
	}

	constexpr auto one_line() const {
		return "Define the species used in the simulation";
	}
	
	constexpr auto help() const {
		return R""""(

The 'species' command
==================

This command defines speciess the actual simulation.

These are the options available:

-  Shell:  `species`
   Python: `species.status()`

   Shows the species currently defined in the simulation.

   Shell example:  `inq species`
   Python example: `pinq.species.status()`


-  Shell:  `species set`
   Python: `species.pseudo_set()`

   Returns the current pseudopotential set that will be used by
   default for all species in the simulation.

   Shell example:  `inq species set`
   Python example: `pinq.species.pseudo_set()`


-  Shell:  `species list-sets`
   Python: `species.list_sets()`

   Prints a list of the available pseudopotentials.

   Shell example:  `inq species list-sets`
   Python example: `pinq.species.list_sets()`


-  Shell:  `species set <set name>`
   Python: `species.pseudo_set("set_name")`

   Selects the pseudopotential set that will be used by default for
   all species in the simulation. The name must be one those returned
   by the `list-sets` command.

   Shell example:  `inq species set sg15`
   Python example: `pinq.species.set("sg15")`


-  Shell:  `species add <element>`
   Python: `species.add("element")`

   Adds a new species of given by <element>. The element must be a
   IUPAC recognized element chemical symbol.

   Shell example:  `inq species add Tc`
   Python example: `pinq.species.add("Tc")`


-  Shell:  `species add <symbol> element <element>`
   Python: `species.add_by_element("symbol", "element")`

   Adds a new species given by its <symbol> as an alias to
   <element>. Note that element must be a IUPAC recognized element
   symbol.

   Shell example:  `inq species add proton element H`
   Python example: `pinq.species.add_by_element("proton", "H")`


-  Shell:  `species add <symbol> atomic_number <z>`
   Python: `species.add_by_atomic_number("symbol", z)`

   Adds a new species given by its <symbol> as an alias to
   the chemical element with atomic number <z>.

   Shell example:  `inq species add C14 atomic-number 6`
   Python example: `pinq.species.add_by_atomic_number("C14", 6)`


-  Shell:  `species <symbol> set <set name>`
   Python: `species.pseudo_set("symbol", "set_name")`

   Sets the pseudopotential set that will be used for the species
   given by <symbol>.

   Shell example:  `inq species He set sg15`
   Python example: `pinq.species.set("He", "sg15")`


-  Shell:  `species <symbol> file <filename>`
   Python: `species.file("symbol", "filename")`

   Sets the pseudopotential file that will be used for the species
   given by <symbol>.

   Shell example:  `inq species Xe file Xe.upf`
   Python example: `pinq.species.file("Xe", "Xe.upf")`


-  Shell:  `species <symbol> mass <value>`
   Python: `species.mass("symbol", value)`

   Sets the ionic mass for the species given by <symbol>. The mass
   must be given in atomic mass units (amu). They should not be
   confused with atomic units of mass. In amu the mass of a carbon
   atom is 12.

   Shell example:  `inq species U mass 235`
   Python example: `pinq.species.mass("U", 235)`


)"""";
	}

	static auto string_to_pseudo_set(std::string set_name){
		try { 
			std::replace(set_name.begin(), set_name.end(), '-', '_');
			pseudo::set_id set(set_name);
			return set;
		}
		catch(...) {
			actions::error(input::environment::global().comm(), "Unknown set '" + set_name + "'.");
			return pseudo::set_id{};
		}
	}

	static auto string_to_species(std::string element){
		element[0] = std::toupper(element[0]);
		auto new_species = ionic::species(element);
		if(not new_species.valid()) actions::error(input::environment::global().comm(), "Unknown element '" + element + "' in `species` command.");
		return new_species;
	}
	
	static void status() {
		if(input::environment::global().comm().root()) std::cout << systems::ions::load(".inq/default_ions").species_list();
	}
	
	void operator()() const {
		status();
	}

	static void pseudo_set() {
		if(input::environment::global().comm().root()) std::cout << systems::ions::load(".inq/default_ions").species_list().pseudopotentials() << std::endl;
	}

	static void pseudo_set(std::string const & set_name) {
		auto ions = systems::ions::load(".inq/default_ions");

		ions.species_list().pseudopotentials() = string_to_pseudo_set(set_name);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void list_sets() {
		auto list = pseudo::set_id::list();
		std::cout << "Available pseudopotential sets:" << std::endl;
		for(auto const & item : list) {
			std::cout << "  " << item << std::endl;
		}
	}

	static void add_species(std::string const & element) {

		auto ions = systems::ions::load(".inq/default_ions");
		ions.species_list().insert(string_to_species(element));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}
	
	static void add_species_by_element(std::string symbol, std::string const & element) {

		symbol[0] = std::toupper(symbol[0]);
		auto ions = systems::ions::load(".inq/default_ions");
		ions.species_list().insert(string_to_species(element).symbol(symbol));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void add_species_by_atomic_number(std::string symbol, int atomic_number) {

		symbol[0] = std::toupper(symbol[0]);
		auto ions = systems::ions::load(".inq/default_ions");
		auto new_species = ionic::species(atomic_number).symbol(symbol);
		if(not new_species.valid()) actions::error(input::environment::global().comm(), "Invalid atomic number ", atomic_number, " in `species add`.");
		ions.species_list().insert(new_species);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void pseudo_set(std::string symbol, std::string const & set_name) {

		symbol[0] = std::toupper(symbol[0]);
		auto ions = systems::ions::load(".inq/default_ions");
		if(not ions.species_list().contains(symbol)) ions.species_list().insert(string_to_species(symbol));
		ions.species_list()[symbol].pseudo_set(string_to_pseudo_set(set_name));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void file(std::string symbol, std::string const & file) {

		symbol[0] = std::toupper(symbol[0]);
		auto ions = systems::ions::load(".inq/default_ions");
		if(not ions.species_list().contains(symbol)) ions.species_list().insert(string_to_species(symbol));
		ions.species_list()[symbol].pseudo_file(file);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void mass(std::string symbol, double const & mass_value) {

		symbol[0] = std::toupper(symbol[0]);
		auto ions = systems::ions::load(".inq/default_ions");
		if(not ions.species_list().contains(symbol)) ions.species_list().insert(string_to_species(symbol));
		ions.species_list()[symbol].mass(mass_value);
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {
		
		using utils::str_to;
		
		if(args.size() == 0) {
			status();
			actions::normal_exit();
		}

		if(args.size() == 1 and args[0] == "set") {
			pseudo_set();
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "set") {
			pseudo_set(args[1]);
			status();
      actions::normal_exit();
    }

		if(args.size() == 1 and args[0] == "list-sets") {
			list_sets();
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "add") {
			add_species(args[1]);
			status();
			actions::normal_exit();
		}
		
		if(args.size() == 4 and args[0] == "add" and args[2] == "element") {
			add_species_by_element(args[1], args[3]);
			status();
			actions::normal_exit();
		}
		
		if(args.size() == 4 and args[0] == "add" and args[2] == "atomic-number") {
			add_species_by_atomic_number(args[1], str_to<int>(args[3]));
			status();
			actions::normal_exit();
		}
		
		if(args.size() == 3 and args[1] == "set") {
			pseudo_set(args[0], args[2]);
			status();
			actions::normal_exit();
		}

		if(args.size() == 3 and args[1] == "file") {
			file(args[0], args[2]);
			status();
			actions::normal_exit();
		}

		if(args.size() == 3 and args[1] == "mass") {
			mass(args[0], str_to<double>(args[2]));
			status();
			actions::normal_exit();
		}

		if(input::environment::global().comm().root()) actions::error(input::environment::global().comm(), "Invalid syntax in the species command");
	}
	
#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule(name(), help());
		sub.def("status",    &status);
		sub.def("list_sets", &list_sets);
		sub.def("file",      &file);
		
	}
#endif

} const species;

}
}
#endif

#ifdef INQ_INTERFACE_SPECIES_UNIT_TEST
#undef INQ_INTERFACE_SPECIES_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
