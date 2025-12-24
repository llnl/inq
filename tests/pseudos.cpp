/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;
	
	utils::match energy_match(2.0e-5);

	auto cell = systems::cell::cubic(15.0_b).finite();

	{
		systems::ions ions(cell);
		
		ions.insert(ionic::species("C").pseudo_file(inq::config::path::unit_tests_data() + "C_ONCV_PBE-1.2.xml"), {0.0_b, 0.0_b, 0.0_b});

		systems::electrons electrons(ions, options::electrons{}.cutoff(25.0_Ha).extra_states(4).temperature(300.0_K));
		
		ground_state::initial_guess(ions, electrons);
		
		auto result = ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    ,           -5.365890535419);
		energy_match.check("kinetic energy",      result.energy.kinetic()   ,           3.193381219632);
		energy_match.check("eigenvalues",         result.energy.eigenvalues(),         -1.403103443279);
		energy_match.check("Hartree energy",      result.energy.hartree(),              4.375093556147);
		energy_match.check("external energy",     result.energy.external(),           -12.007376353904);
		energy_match.check("non-local energy",    result.energy.non_local(),            0.486692942407);
		energy_match.check("XC energy",           result.energy.xc(),                  -1.413681899701);
		energy_match.check("XC density integral", result.energy.nvxc(),                -1.825988363707);
		energy_match.check("HF exchange energy",  result.energy.exact_exchange(),       0.000000000000);
		energy_match.check("ion-ion energy",      result.energy.ion(),                  0.000000000000);

	}

	{
		systems::ions ions(cell);
		ions.species_list().pseudopotentials() = pseudo::set_id::ccecp();
		ions.insert(ionic::species("C"), {0.0_b, 0.0_b, 0.0_b});

		systems::electrons electrons(ions, options::electrons{}.cutoff(25.0_Ha).extra_states(4).temperature(300.0_K));
		
		ground_state::initial_guess(ions, electrons);
		
		auto result = ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    ,           -5.387198836790);
		energy_match.check("kinetic energy",      result.energy.kinetic()  ,            3.454393581500);
		energy_match.check("eigenvalues",         result.energy.eigenvalues(),         -1.409228514997);
		energy_match.check("Hartree energy",      result.energy.hartree(),              4.391970222592);
		energy_match.check("external energy",     result.energy.external(),           -12.416057447122);
		energy_match.check("non-local energy",    result.energy.non_local(),            0.603238619052);
		energy_match.check("XC energy",           result.energy.xc(),                  -1.420743812813);
		energy_match.check("XC density integral", result.energy.nvxc(),                -1.834743713612);
		energy_match.check("HF exchange energy",  result.energy.exact_exchange(),       0.000000000000);
		energy_match.check("ion-ion energy",      result.energy.ion(),                  0.000000000000);

	}

	{
		systems::ions ions(cell);
		ions.insert(ionic::species("C").pseudo_set(pseudo::set_id::sg15()), {0.0_b, 0.0_b, 0.0_b});
		
		systems::electrons electrons(ions, options::electrons{}.cutoff(25.0_Ha).extra_states(4).temperature(300.0_K));
		
		ground_state::initial_guess(ions, electrons);
		
		auto result = ground_state::calculate(ions, electrons, options::theory{}.non_interacting(), inq::options::ground_state{}.energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total()    ,           -8.849545883572);
		energy_match.check("kinetic energy",      result.energy.kinetic()  ,            5.537183342491);
		energy_match.check("eigenvalues",         result.energy.eigenvalues(),         -8.849545883572);
		energy_match.check("Hartree energy",      result.energy.hartree(),              0.000000000000);
		energy_match.check("external energy",     result.energy.external(),           -15.074544413501);
		energy_match.check("non-local energy",    result.energy.non_local(),            0.687815187437);
		energy_match.check("XC energy",           result.energy.xc(),                   0.000000000000);
		energy_match.check("XC density integral", result.energy.nvxc(),                 0.000000000000);
		energy_match.check("HF exchange energy",  result.energy.exact_exchange(),       0.000000000000);
		energy_match.check("ion-ion energy",      result.energy.ion(),                  0.000000000000);
	}
		
	return energy_match.fail();
}

