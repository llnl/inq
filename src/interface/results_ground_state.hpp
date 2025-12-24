/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RESULTS_GROUND_STATE
#define INQ__INTERFACE__RESULTS_GROUND_STATE

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/runtime_options.hpp>
#include <ground_state/results.hpp>

namespace inq {
namespace interface {

struct {		

	constexpr auto name() const {
		return "results ground-state";
	}

	constexpr auto one_line() const {
		return "Get information about the results obtained from a ground-state calculation";
	}

	constexpr auto help() const {
		
		return R""""(
The 'results ground-state' command
==================

This command queries the results obtained from a ground-state
calculation. Without arguments, it prints the values calculated.

The options allows you to query a specific value. In this case only
the value will be printed without any other text, so it is suitable
for easy parsing in scripting. The values are returned in atomic
units.

These are the available subcommands:

- Shell:  `results ground-state`
  Python: `results.ground_state.status()`

  When no arguments are given (or the status function is used), print
  the values calculated.

  Shell example:  `inq results ground-state`.
  Python example: `pinq.results.ground_state.status()`


- Shell:  `results ground-state iterations`
  Python: `results.ground_state.iterations()`

  Print the number of self-consistency iterations done.

  Shell example:  `inq results ground-state iterations`
  Python example: `pinq.results.ground_state.iterations()`

- Shell:  `results ground-state magnetization [index]`
  Python: `results.ground_state.magnetization()`

  Query the magnetization. In Python or if no additional arguments are
  given, print the whole 3D vector. For the shell, optionally you can
  add an index argument to print a specific component of the
  vector. The index can be given as the letters _x_, _y_ or _z_.

  Note that for spin unpolarized systems the magnetization is always
  zero. For spin polarized the magnetization is assumed on the _z_
  direction.

  Shell example:  `inq results ground-state magnetization z`
  Python example: `pinq.results.ground_state.magnetization()`


- Shell:  `results ground-state dipole [index]`
  Python: `results.ground_state.dipole()`

  Return the dipole of the system in atomic units. The whole vector is
  given if no additional arguments are included. In the CLI, you can
  also add an index argument to print a specific component. The index
  can be given as the letters _x_, _y_ or _z_.

  Note that the dipole is only calculated for the non-periodic
  directions. For the periodic directions is set to zero since the
  dipole is not properly defined.

  Shell example:  `inq results ground-state magnetization z`
  Python example: `pinq.results.ground_state.magnetization()`

- Shell:  `results forces [iatom] [index]`
  Python: `results.forces()`

  Return the forces over the atoms in the system in atomic units. In
  Python this is an array. In the CLI, if no arguments are given, all
  the forces are printed. Alternatively is possible to pass an atom
  index (that starts from 0), or both an atom and direction index
  (_x_, _y_, or _z_).

  Shell example:  `inq results forces 7 z`
  Python example: `pinq.results.forces()`


- Shell:  `results ground-state energy`
  Python: `results.ground_state.energy.status()`

  When no arguments are given (or status() in Python), `energy` will
  print all the energy values available.

  Shell example:  `inq results ground-state energy`
  Python example: `pinq.results.ground_state.energy.status()`


- Shell:  `results ground-state energy total`
  Python: `results.ground_state.energy.total()`

  Returns the total energy of the calculation. This includes the ionic
  contribution.

  Shell example:  `inq results ground-state energy total`.
  Python example: `pinq.results.ground_state.energy.total()`

- Shell:  `results ground-state energy kinetic`
  Python: `results.ground_state.energy.kinetic()`

  The electronic kinetic energy.

  Shell example:  `inq results ground-state energy kinetic`
  Python example: `pinq.results.ground_state.energy.kinetic()`


- Shell:  `results ground-state energy eigenvalues`
  Python: `results.ground_state.energy.eigenvalues()`

  The sum of the eigenvalues, weighed by the occupations.

  Shell example:  `inq results ground-state energy eigenvalues`
  Python example: `pinq.results.ground_state.energy.eigenvalues()`

- Shell:  `results ground-state energy Hartree`
  Python: `results.ground_state.energy.hartree()`

  The classical electrostatic interaction energy between electrons.

  Shell example:  `inq results ground-state energy Hartree`
  Python example: `pinq.results.ground_state.energy.hartree()`


- Shell:  `results ground-state energy external`
  Python: `results.ground_state.energy.external()`

  The energy of the interaction of the electrons with the local
  potential generated by the ions. This doesn't include the non-local
  pseudopotential part.

  Shell example:  `inq results ground-state energy external`.
  Python example: `pinq.results.ground_state.energy.external()`


- Shell:  `results ground-state energy non-local`
  Python: `results.ground_state.energy.non_local()`

  The energy of the interaction of the electrons with the non-local
  part of the ionic pseudo-potentials.

  Shell example:  `inq results ground-state energy non-local`
  Python example: `pinq.results.ground_state.energy.non_local()`


- Shell:  `results ground-state energy xc`
  Python: `results.ground_state.energy.xc()`

  The exchange and correlation energy from DFT semi-local
  functionals. It doesn't include the contribution from Hartree-Fock
  exchange (see `energy exact_exchange`).

  Shell example:  `inq results ground-state energy xc`
  Python example: `pinq.results.ground_state.energy.xc()`


- Shell:  `results ground-state energy nvxc`
  Python: `results.ground_state.energy.nvxc()`

  The energy of the interaction of the exchange and correlation
  potential and the density. This is different from the exchange and
  correlation energy.

  Shell example:  `inq results ground-state energy nvxc`
  Python example: `pinq.results.ground_state.energy.nvxc()`


- Shell:  `results ground-state energy exact-exchange`
  Python: `results.ground_state.energy.exact_exchange()`

  The Hartree-Fock exact-exchange energy. This is calculated for
  Hartree-Fock and hybrid functionals.

  Shell example:  `inq results ground-state energy exact-exchange`
  Python example: `pinq.results.ground_state.energy.exact_exchange()`


- Shell:  `results ground-state energy ion`
  Python: `results.ground_state.energy.ion()`

  The ion-ion interaction energy. This value is calculated taking into
  account the periodicity of the system.

  Shell example:  `inq results ground-state energy ion`.
  Python example: `pinq.results.ground_state.energy.ion()`


)"""";
	}
	static void status() {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		if(input::environment::global().comm().root()) std::cout << res;
	}
	
	void operator()() const {
		status();
	}

	static auto iterations() {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.total_iter;
	}

	static auto magnetization() {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.magnetization;
	}

	static auto dipole() {
		auto res = ground_state::results::load(".inq/default_results_ground_state");
		return res.dipole;
	}
	
	static void energy() {
		auto ener = ground_state::results::load(".inq/default_results_ground_state").energy;
		if(input::environment::global().comm().root()) std::cout << ener;
	}
	
  static double energy_total() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.total();
  }
	
  static double energy_kinetic() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.kinetic();
  }

  static double energy_eigenvalues() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.eigenvalues();
  }

  static double energy_external() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.external();
  }
  
  static double energy_non_local() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.non_local();
  }
  
  static double energy_hartree() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.hartree();
  }
  
  static double energy_xc() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.xc();
  }

  static double energy_nvxc() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.nvxc();
  }

  static double energy_exact_exchange() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.exact_exchange();
  }
  
  static double energy_ion() {
    return ground_state::results::load(".inq/default_results_ground_state").energy.ion();
  }

	static auto forces() {
    return ground_state::results::load(".inq/default_results_ground_state").forces;
	}

	template <typename ArgsType>
	void command(ArgsType args, runtime_options const & run_opts) const {

		if(args.size() == 0){
			operator()();
			actions::normal_exit();
		}
		
		if(args.size() == 1 and args[0] == "iterations"){
			std::cout << iterations() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 1 and args[0] == "magnetization"){
			std::cout << magnetization() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "magnetization"){
			auto idir = utils::str_to_index(args[1]);
			if(idir == -1) actions::error(input::environment::global().comm(), "Invalid index in the 'results ground-state magnetization' command");
			if(input::environment::global().comm().root())  printf("%.6f\n", magnetization()[idir]);
			actions::normal_exit();
		}
		
		if(args.size() == 1 and args[0] == "dipole"){
			std::cout << dipole() << std::endl;
			actions::normal_exit();
		}

		if(args.size() == 2 and args[0] == "dipole"){
			auto idir = utils::str_to_index(args[1]);
			if(idir == -1) actions::error(input::environment::global().comm(), "Invalid index in the 'results ground-state dipole' command");
			if(input::environment::global().comm().root())  printf("%.6f\n", dipole()[idir]);
			actions::normal_exit();
		}
		
		if(args[0] == "energy"){

			args.erase(args.begin());

			if(args.size() == 0) {
				energy();
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "total"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_total());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "kinetic"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_kinetic());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "eigenvalues"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_eigenvalues());
				actions::normal_exit();
			}
    
			if(args.size() == 1 and args[0] == "external"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_external());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "non-local"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_non_local());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "hartree"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_hartree());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "xc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_xc());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "nvxc"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_nvxc());
				actions::normal_exit();
			}

			if(args.size() == 1 and args[0] == "exact-exchange"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_exact_exchange());
				actions::normal_exit();
			}
        
			if(args.size() == 1 and args[0] == "ion"){
				if(input::environment::global().comm().root()) printf("%.20e\n", energy_ion());
				actions::normal_exit();
			}
		}
		
		if(args[0] == "forces"){
			auto forces_array = forces();

			if(args.size() == 1) {

				if(input::environment::global().comm().root()) {
					for(auto & force : forces_array) printf("%.20e\t%.20e\t%.20e\n", force[0], force[1], force[2]);
				}
				actions::normal_exit();
					
			} else if (args.size() == 2 or args.size() == 3) {
				auto index = utils::str_to<long>(args[1]);
				if(index < 0 or index >= forces_array.size()) actions::error(input::environment::global().comm(), "Invalid index ", index, " in the 'results ground-state forces' command");

				if(args.size() == 2) {
					if(input::environment::global().comm().root()) printf("%.20e\t%.20e\t%.20e\n", forces_array[index][0], forces_array[index][1], forces_array[index][2]);
					actions::normal_exit();
				}
					
				auto idir = utils::str_to_index(args[2]);
				if(idir == -1) actions::error(input::environment::global().comm(), "Invalid coordinate index in the 'results ground-state forces' command");
				if(input::environment::global().comm().root()) printf("%.20e\n", forces_array[index][idir]);
				actions::normal_exit();
			}
		}
		
		actions::error(input::environment::global().comm(), "Invalid syntax in the 'results ground-state' command");    
	}
	
#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		namespace py = pybind11;
		using namespace pybind11::literals;

		auto sub = module.def_submodule("ground_state", help());

		sub.def("status",          &status);
		sub.def("iterations",    &iterations);
		sub.def("magnetization", []() {
			return static_cast<std::vector<double>>(magnetization());			
		});
		sub.def("dipole", []() {
			return static_cast<std::vector<double>>(dipole());			
		});
		
		sub.def("forces", []() {

			auto forces_multi = forces();
			
			py::array_t<double, py::array::c_style> forces_array({forces_multi.size(), 3l});
			
			auto arr = forces_array.mutable_unchecked();
			
			for (py::ssize_t iatom = 0; iatom < arr.shape(0); iatom++) {
				for (py::ssize_t idir = 0; idir < arr.shape(1); idir++) {
					arr(iatom, idir) = forces_multi[iatom][idir];
				}
			}
		
			return forces_array;
		});

		auto sub_en = sub.def_submodule("energy");
		sub_en.def("status",         &energy);
		sub_en.def("total",          &energy_total);
		sub_en.def("kinetic",        &energy_kinetic);
		sub_en.def("eigenvalues",    &energy_eigenvalues);
		sub_en.def("hartree",        &energy_hartree);
		sub_en.def("external",       &energy_external);
		sub_en.def("non_local",      &energy_non_local);
		sub_en.def("xc",             &energy_xc);
		sub_en.def("nvxc",           &energy_nvxc);
		sub_en.def("exact_exchange", &energy_exact_exchange);
		sub_en.def("ion",            &energy_ion);
		
	}
#endif
	
} const results_ground_state;

}
}
#endif

#ifdef INQ_INTERFACE_RESULTS_GROUND_STATE_UNIT_TEST
#undef INQ_INTERFACE_RESULTS_GROUND_STATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
