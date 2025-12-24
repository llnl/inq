/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__THEORY
#define INQ__INTERFACE__THEORY

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/runtime_options.hpp>
#include <options/theory.hpp>

namespace inq {
namespace interface {

struct {		

	constexpr auto name() const {
		return "theory";
	}

	constexpr auto one_line() const {
		return "Defines the theory used to represent the electrons-electron interaction";
	}
	
	constexpr auto help() const {
		
		return R""""(

The 'theory' command
==================

This command specified the theory level used to model the
electron-electron interaction. Most of the time you will want to use
some form of density functional theory (DFT) but some other options
are available. This command will allow you to pick up the DFT
functional you want to use, the most popular ones have a specific
option or you can use the `functional` command to select any
functional from libxc.

These are the options available:

- `theory`

   Without any argument, `theory` will just print the current theory level that is set.

   Example: `inq theory`.

- `theory dft` (default)

   This is the default, DFT in the PBE approximation is used to model
   the electron-electron interaction.

   Example: `inq theory dft`
`
- `theory non-interacting`

   There is no electron-electron interaction, particles are assumed to
   be independent.

   Example: `inq theory non-interacting`

- `theory Hartree`

   Particles only interact through classical electrostatics. Note that
   as implemented in inq, hartree does not include a self-interaction
   correction term.

   Example: `inq theory Hartree`

- `theory Hartree-Fock`

   Exchange is modeled by the Hartree-Fock method. Note that this
   method is much more expensive than pure DFT.

   Example: `inq theory Hartree-Fock`

- `theory lda`

   The local density approximation in DFT.

   Example: `inq theory lda`

- `theory pbe`

   The PBE GGA approximation in DFT.

   Example: `inq theory pbe`

- `theory pbe0`

   The PBE0 (also known as PBEH) hybrid functional. Note that this
   functional includes Hartree-Fock exact exchange, so it is much more
   computationally expensive than GGA functionals like pbe.

   Example: `inq theory pbe0`

- `theory b3lyp`

   The B3LYP hybrid functional. Note that this functional includes
   Hartree-Fock exact exchange, so it is much more computationally
   expensive than GGA functionals like pbe.

   Example: `inq theory b3lyp`

- `theory functional <exchange_name> [correlation_name]`

   This option allows you to select any functional combination from
   the libxc library using the functional names (functional id numbers
   are not supported). Note that the correlation functional is
   optional, it is okay to pass just one functional. You can find a
   list of libxc functionals here [1].

   [1] https://www.tddft.org/programs/libxc/functionals/
   
   Examples: `inq theory functional XC_GGA_X_RPBE XC_GGA_C_PBE`
             `inq theory functional LDA_XC_TETER93`

)"""";
	}	

	static void status() {
		auto theo = options::theory::load(".inq/default_theory");
		if(input::environment::global().comm().root()) std::cout << theo;
	}

	void operator()() const {
		status();
	}
	
	void non_interacting() const{
		auto theo = options::theory::load(".inq/default_theory").non_interacting();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}

	void hartree() const{
		auto theo = options::theory::load(".inq/default_theory").hartree();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}

	void hartree_fock() const{
		auto theo = options::theory::load(".inq/default_theory").hartree_fock();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}

	void dft() const{
		auto theo = options::theory::load(".inq/default_theory").dft();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}
	
	void lda() const{
		auto theo = options::theory::load(".inq/default_theory").lda();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}
	
	void pbe() const{
		auto theo = options::theory::load(".inq/default_theory").pbe();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}
	
	void pbe0() const{
		auto theo = options::theory::load(".inq/default_theory").pbe0();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}
	
	void b3lyp() const{
		auto theo = options::theory::load(".inq/default_theory").b3lyp();
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}

	void functional(int exchange, int correlation = XC_NONE) const{
		auto theo = options::theory::load(".inq/default_theory").functional(exchange, correlation);
		theo.save(input::environment::global().comm(), ".inq/default_theory");
	}
	
	template <typename ArgsType>
	void command(ArgsType args, runtime_options const & run_opts) const {

		if(args.size() == 0){
			operator()();
			actions::normal_exit();
			
		} else if(args.size() == 1 and args[0] == "non-interacting") {
			non_interacting();
		} else if(args.size() == 1 and args[0] == "hartree") {
			hartree();
		} else if(args.size() == 1 and args[0] == "hartree-fock") {
			hartree_fock();
		} else if(args.size() == 1 and args[0] == "dft") {
			dft();
		} else if(args.size() == 1 and args[0] == "lda") {
			lda();
		} else if(args.size() == 1 and args[0] == "pbe") {
			pbe();
		} else if(args.size() == 1 and args[0] == "pbe0") {
			pbe0();
		} else if(args.size() == 1 and args[0] == "b3lyp") {
			b3lyp();

		} else if(args[0] == "functional") {

			if(args.size() == 1) actions::error(input::environment::global().comm(), "Missing arguments for the 'theory functional' command");			
			if(args.size() > 3)  actions::error(input::environment::global().comm(), "Too many arguments for the 'theory functional' command");

			std::replace(args[1].begin(), args[1].end(), '-', '_'); //functional names use underscores
			auto exchange_id = xc_functional_get_number(args[1].c_str());
			if(exchange_id == -1) actions::error(input::environment::global().comm(), "Unknown exchange functional '" + args[1] + "'in 'theory' command");
			
			auto correlation_id = XC_NONE;
			if(args.size() == 3) {
				std::replace(args[2].begin(), args[2].end(), '-', '_'); //functional names use underscores
				correlation_id = xc_functional_get_number(args[2].c_str());
				if(correlation_id == -1) actions::error(input::environment::global().comm(), "Unknown correlation functional '" + args[2] + "' in 'theory' command");
			}
			
			functional(exchange_id, correlation_id);
			
		} else {				
			actions::error(input::environment::global().comm(), "Invalid syntax in 'theory' command");
		}

		if(not run_opts.quiet) operator()();
		actions::normal_exit();
	}
	
} const theory;

}
}
#endif

#ifdef INQ_INTERFACE_THEORY_UNIT_TEST
#undef INQ_INTERFACE_THEORY_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
