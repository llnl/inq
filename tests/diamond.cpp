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
	
	utils::match energy_match(3.0e-5);

	auto a = 3.567095_A;
	systems::ions ions(systems::cell::lattice({0.0_b, a/2.0, a/2.0}, {a/2, 0.0_b, a/2.0}, {a/2.0, a/2.0, 0.0_b}));
	
	ions.insert_fractional("C", {0.0,  0.0,  0.0 });
	ions.insert_fractional("C", {0.25, 0.25, 0.25});

	{
		systems::electrons electrons(ions, options::electrons{}.cutoff(35.0_Ha).extra_states(3), input::kpoints::gamma() + input::kpoints::point({-0.5, -0.5, -0.5}, 0.0));

		energy_match.check("size x", electrons.states_basis().sizes()[0], 14);
		energy_match.check("size y", electrons.states_basis().sizes()[1], 14);
		energy_match.check("size z", electrons.states_basis().sizes()[2], 14);
		
		ground_state::initial_guess(ions, electrons);
		
		auto result = ground_state::calculate(ions, electrons, options::theory{}.pbe(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1e-8_Ha));
		
		energy_match.check("total energy",        result.energy.total(),          -10.949243966036);
		energy_match.check("kinetic energy",      result.energy.kinetic(),         11.423986855221);
		energy_match.check("eigenvalues",         result.energy.eigenvalues(),      0.796448318173);
		energy_match.check("Hartree energy",      result.energy.hartree(),          1.474429968732);
		energy_match.check("external energy",     result.energy.external(),        -7.038773784808);
		energy_match.check("non-local energy",    result.energy.non_local(),       -1.505554093563);
		energy_match.check("XC energy",           result.energy.xc(),              -4.568608783014);
		energy_match.check("XC density integral", result.energy.nvxc(),            -5.032070596140);
		energy_match.check("HF exchange energy",  result.energy.exact_exchange(),   0.000000000000);
		energy_match.check("ion-ion energy",      result.energy.ion(),            -10.734724128603);

		auto all_eigenvalues = parallel::gather(+electrons.eigenvalues().flatted(), electrons.kpin_states_comm(), 0);

		if(electrons.kpin_states_comm().root()){
			energy_match.check("gamma homo",          all_eigenvalues[3],      0.303919090581);
			energy_match.check("gamma lumo",          all_eigenvalues[4],      0.496760372099);
			energy_match.check("0.5,0.5,0.5 homo",    all_eigenvalues[7 + 3],  0.195846631157);
			energy_match.check("0.5,0.5,0.5 lumo",    all_eigenvalues[7 + 4],  0.597360038004);
		}

		auto ked = observables::kinetic_energy_density(electrons);
		
		energy_match.check("kinetic energy", operations::integral(ked), 11.423986855221);
	}
	
	fftw_cleanup();
	return energy_match.fail();
	
}
