/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__CELL
#define INQ__INTERFACE__CELL

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/actions.hpp>
#include <interface/runtime_options.hpp>
#include <systems/ions.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "cell";
	}

	constexpr auto one_line() const {
		return "Defines the simulation cell";
	}
	
	constexpr auto help() const {
		
		return R""""(

The 'cell' command
==================

The 'cell' command defines the inq simulation cell. There are
different variations of the command depending on the type of cell you
want to define.

In all cases you must include the units for the cell. For the moment
note you can use 'bohr' (or 'b') and 'angstrom' (or 'A').  See `inq
help units` for other supported units.

An optional last argument defines the periodicity of the cell. The
periodicity can be specified as a number from 0 (or '0d') to 3 ('3d'),
or as 'finite', 'wire', 'slab' and 'periodic'. If the periodicity is
not given, the system is considered periodic. For Python either a
integer or a string can be given.

Note that defining the cell is not necessary if you are going to read
it from a file (using the `inq ion file` command).

The following are the accepted forms of the cell command:

- Shell:  cell
  Python: pinq.cell.status()

  Without any arguments (or the status function in python), `cell`
  prints the cell currently defined in the system.

  Shell example:  `inq cell`
  Python example: `pinq.cell.status()`


- Shell:  inq cell cubic <a> <units> [periodicity]
  Python: pinq.cell.cubic(a, units, periodicity = 3)

  Defines a cubic cell of side <a>.

  Shell example:   `inq cell cubic 5.0 A finite`
  Python example:  `pinq.cell.cubic(5.0, "A", "finite")`


- Shell:  inq cell orthorhombic <a> <b> <c> <units> [periodicity]
  Python: pinq.cell.orthorhombic(a, b, c, units, periodicity = 3)

  Defines a orthorhombic cell of sides a, b and c.

  Shell example:  `inq cell orthorhombic 10.0 10.0 12.0 bohr`
  Python example: `pinq.cell.orthorhombic(10.0, 10.0, 12.0, "bohr")`


- Shell:  inq cell  <a1> <a2> <a3>  <b1> <b2> <b3>  <c1> <c2> <c3>  <units> [periodicity]
  Python: inq.cell.lattice([a1, a2, a3], [b1, b2, b3], [c1, c2, c3], units, periodicity)

  Creates a general cell defined by the lattice vectors a, b, and c
  (given by components). The units are applied to all the vector
  components.

  Shell example:   `inq cell  4.6478 0 0  -2.3239 4.02512 0  0 0 10.0 b 2d`
  Python example:  `pinq.cell.lattice([4.6478, 0, 0], [-2.3239, 4.02512, 0], [0, 0, 10.0], "b", "2d")`


- Shell:  inq cell  <a1> <a2> <a3>  <b1> <b2> <b3>  <c1> <c2> <c3>  scale <s> <units> [periodicity]
  Python: inq.cell.lattice([a1, a2, a3], [b1, b2, b3], [c1, c2, c3], scale, units, periodicity)

  Like the previous case, creates a general cell defined by the lattice vectors a, b, and c
  (given by components). However in this case a general scale factor is applied to all the vectors.

  Shell example:  `inq cell  0.0 0.5 0.5  0.5 0.0 0.5  0.5 0.5 0.0  scale 3.57 angstrom`
  Python example: `pinq.cell.lattice([0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], 3.57, "angstrom")`
  

)"""";
	}
	
	static void status() {
		auto cell = systems::ions::load(".inq/default_ions").cell();
		if(input::environment::global().comm().root()) std::cout << cell;
	}

	void operator()() const {
		status();
	}
	
	static void cubic(quantity<magnitude::length> const aa, int periodicity = 3) {
		systems::ions ions(systems::cell::cubic(aa).periodicity(periodicity));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void orthorhombic(quantity<magnitude::length> const aa, quantity<magnitude::length> const bb, quantity<magnitude::length> const cc, int periodicity = 3) {
		systems::ions ions(systems::cell::orthorhombic(aa, bb, cc).periodicity(periodicity));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	static void lattice(quantity<magnitude::length> const aa0, quantity<magnitude::length> const aa1, quantity<magnitude::length> const aa2,
											quantity<magnitude::length> const bb0, quantity<magnitude::length> const bb1, quantity<magnitude::length> const bb2,
											quantity<magnitude::length> const cc0, quantity<magnitude::length> const cc1, quantity<magnitude::length> const cc2,
											int periodicity = 3) {
		systems::ions ions(systems::cell::lattice({aa0, aa1, aa2}, {bb0, bb1, bb2}, {cc0, cc1, cc2}).periodicity(periodicity));
		ions.save(input::environment::global().comm(), ".inq/default_ions");
	}

	void operator()(quantity<magnitude::length> const aa0, quantity<magnitude::length> const aa1, quantity<magnitude::length> const aa2,
									quantity<magnitude::length> const bb0, quantity<magnitude::length> const bb1, quantity<magnitude::length> const bb2,
									quantity<magnitude::length> const cc0, quantity<magnitude::length> const cc1, quantity<magnitude::length> const cc2,
									int periodicity = 3) const {
		lattice(aa0, aa1, aa2, bb0, bb1, bb2, cc0, cc1, cc2, periodicity);
	}
	
private:

	static auto parse_periodicity(std::string per_string){
		
		if(per_string == "finite" or per_string == "0" or per_string == "0d") {
			return 0;
		} else if (per_string == "wire" or per_string == "1" or per_string == "1d"){
			return 1;
		} else if (per_string == "slab" or per_string == "2" or per_string == "2d"){
			return 2;
		} else if (per_string == "periodic" or per_string == "3" or per_string == "3d"){
			return 3;
		} else {
			throw std::runtime_error("Error: unknown periodicity '" + per_string + "'.");
		}
	}

public:
	
	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const & run_opts) const {

		using utils::str_to;
		
		if(args.size() == 0) {
			status();
			actions::normal_exit();
		}
		
		if(args[0] == "cubic"){
			if(args.size() != 3 and args.size() != 4) actions::error(input::environment::global().comm(), "Wrong arguments for a cubic cell definition.\nUse: inq cell cubic <lattice_parameter> <units> [periodicity]");
			
			auto aa = magnitude::length::parse(str_to<double>(args[1]), args[2]);

			int per = 3;
			if(args.size() == 4) per = parse_periodicity(args[3]);
			
			cubic(aa, per);
			if(not run_opts.quiet) status();
			actions::normal_exit();
		}
		
		if(args[0] == "orthorhombic"){
			if(args.size() != 5 and args.size() != 6) {
				actions::error(input::environment::global().comm(), "Wrong arguments for an orthorhombic cell definition.\nUse: inq cell orthorhombic <a> <b> <c> <units> [periodicity]");
			}
			
			auto units = args[4];
			auto aa = magnitude::length::parse(str_to<double>(args[1]), units);
			auto bb = magnitude::length::parse(str_to<double>(args[2]), units);
			auto cc = magnitude::length::parse(str_to<double>(args[3]), units);

			int per = 3;
			if(args.size() == 6) per = parse_periodicity(args[5]);
			
			orthorhombic(aa, bb, cc, per);
			if(not run_opts.quiet) status();
			actions::normal_exit();
		}
		
		
		if(args.size() >= 10 and args.size() <= 13) {

			auto scale = 1.0;
			auto units_name = args[9];
			auto per_location = 10;
			
			if(units_name == "scale"){
				scale = str_to<double>(args[10]);
				units_name = args[11];
				per_location = 12;
			}

			auto unit = magnitude::length::parse(scale, units_name);
				
			auto aa0 = str_to<double>(args[0])*unit;
			auto aa1 = str_to<double>(args[1])*unit;
			auto aa2 = str_to<double>(args[2])*unit;
			
			auto bb0 = str_to<double>(args[3])*unit;
			auto bb1 = str_to<double>(args[4])*unit;
			auto bb2 = str_to<double>(args[5])*unit;

			auto cc0 = str_to<double>(args[6])*unit;
			auto cc1 = str_to<double>(args[7])*unit;
			auto cc2 = str_to<double>(args[8])*unit;

			int per = 3;
			if(long(args.size()) > per_location) per = parse_periodicity(args[per_location]);
			
			operator()(aa0, aa1, aa2, bb0, bb1, bb2, cc0, cc1, cc2, per);
			
			if(not run_opts.quiet) status();
			actions::normal_exit();
		}

		actions::error(input::environment::global().comm(), "Invalid syntax in the 'cell' command");
	}

#ifdef INQ_PYTHON_INTERFACE
	
	static auto parse_periodicity(pybind11::object const & per){
		if(pybind11::isinstance<pybind11::int_>(per)) return pybind11::cast<int>(per);
		return parse_periodicity(pybind11::cast<std::string>(per));
	}
	
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;
 
		auto sub = module.def_submodule(name(), help());
		sub.def("status", &status);

		sub.def("cubic", [](double const & lattice_parameter, std::string const & units, pybind11::object const & per) {
			cubic(magnitude::length::parse(lattice_parameter, units), parse_periodicity(per));
		}, "lattice_parameter"_a, "units"_a, "periodicity"_a = 3);

		sub.def("orthorhombic", [](double const & lattice_parameter_a, double const & lattice_parameter_b, double const & lattice_parameter_c, std::string const & units, pybind11::object const & per) {
			auto aa = magnitude::length::parse(lattice_parameter_a, units);
			auto bb = magnitude::length::parse(lattice_parameter_b, units);
			auto cc = magnitude::length::parse(lattice_parameter_c, units);
			orthorhombic(aa, bb, cc, parse_periodicity(per));
		}, "lattice_parameter_a"_a,  "lattice_parameter_b"_a,  "lattice_parameter_c"_a, "units"_a, "periodicity"_a = 3);

		sub.def("lattice", [](std::vector<double> const & lat1, std::vector<double> const & lat2, std::vector<double> const & lat3, std::string const & units, pybind11::object const & per) {
			auto a0 = magnitude::length::parse(lat1[0], units);
			auto a1 = magnitude::length::parse(lat1[1], units);
			auto a2 = magnitude::length::parse(lat1[2], units);

			auto b0 = magnitude::length::parse(lat2[0], units);
			auto b1 = magnitude::length::parse(lat2[1], units);
			auto b2 = magnitude::length::parse(lat2[2], units);

			auto c0 = magnitude::length::parse(lat3[0], units);
			auto c1 = magnitude::length::parse(lat3[1], units);
			auto c2 = magnitude::length::parse(lat3[2], units);

			lattice(a0, a1, a2, b0, b1, b2, c0, c1, c2, parse_periodicity(per));
		}, "lattice_vector_1"_a,  "lattice_vector_2"_a,  "lattice_vector_3"_a, "units"_a, "periodicity"_a = 3);

		sub.def("lattice", [](std::vector<double> const & lat1, std::vector<double> const & lat2, std::vector<double> const & lat3, double const & scale, std::string const & units, pybind11::object const & per) {
			auto su = magnitude::length::parse(scale, units);
			lattice(su*lat1[0], su*lat1[1], su*lat1[2],  su*lat2[0], su*lat2[1], su*lat2[2],  su*lat3[0], su*lat3[1], su*lat3[2], parse_periodicity(per));
		}, "lattice_vector_1"_a,  "lattice_vector_2"_a,  "lattice_vector_3"_a, "scale"_a, "units"_a, "periodicity"_a = 3);
		
	}
#endif

} const cell ;
		
}
}
#endif

#ifdef INQ_INTERFACE_CELL_UNIT_TEST
#undef INQ_INTERFACE_CELL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
