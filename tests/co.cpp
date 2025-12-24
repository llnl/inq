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
	using inq::vector3;
	
	inq::utils::match match(3.0e-4);

	{
		auto ions = systems::ions{systems::cell::orthorhombic(6.0_b, 6.0_b, 10.0_b).finite()};

		auto distance = 1.13_A;
		ions.insert("C", {0.0_b, 0.0_b, -distance/2});
		ions.insert("O", {0.0_b, 0.0_b,  distance/2});

		auto & env = input::environment::global();
		systems::electrons electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha));
		
		ground_state::initial_guess(ions, electrons);
		
		auto scf_options = options::ground_state{}.energy_tolerance(1.0e-9_Ha);
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), scf_options);
		
		match.check("total energy",        result.energy.total(),             -22.422491657617);
		match.check("kinetic energy",      result.energy.kinetic(),            14.383830601080);
		match.check("eigenvalues",         result.energy.eigenvalues(),        -5.886442006103);
		match.check("Hartree energy",      result.energy.hartree(),            28.443566178578);
		match.check("external energy",     result.energy.external(),          -67.925218068703);
		match.check("non-local energy",    result.energy.non_local(),          -2.607996478012);
		match.check("XC energy",           result.energy.xc(),                 -5.955835892018);
		match.check("XC density integral", result.energy.nvxc(),               -6.624190417623);
		match.check("HF exchange energy",  result.energy.exact_exchange(),      0.000000000000);
		match.check("ion-ion energy",      result.energy.ion(),                11.239162001458);

		match.check("dipole x", result.dipole[0],    3.53046e-10);
		match.check("dipole y", result.dipole[1],    3.57393e-10);
		match.check("dipole z", result.dipole[2], 	-0.02476);
		
		auto amp = 0.001;
		
		auto result_pol_x = ground_state::calculate(ions, electrons, options::theory{}.lda(), scf_options, perturbations::simple_electric_field{{amp, 0.0, 0.0}});

		match.check("total energy",        result_pol_x.energy.total(),             -22.422495920238);
		match.check("kinetic energy",      result_pol_x.energy.kinetic(),            14.383818737102);
		match.check("eigenvalues",         result_pol_x.energy.eigenvalues(),        -5.886453829940);
		match.check("Hartree energy",      result_pol_x.energy.hartree(),            28.443558304032);
		match.check("external energy",     result_pol_x.energy.external(),          -67.925204482184);
		match.check("non-local energy",    result_pol_x.energy.non_local(),          -2.607995750966);
		match.check("XC energy",           result_pol_x.energy.xc(),                 -5.955834729679);
		match.check("XC density integral", result_pol_x.energy.nvxc(),               -6.624188941956);
		match.check("HF exchange energy",  result_pol_x.energy.exact_exchange(),      0.000000000000);
		match.check("ion-ion energy",      result_pol_x.energy.ion(),                11.239162001458);

		match.check("dipole x", result_pol_x.dipole[0],  	 0.008525240751);
		match.check("dipole y", result_pol_x.dipole[1],   -0.000000000102);
		match.check("dipole z", result_pol_x.dipole[2], 	-0.024759853102);

		auto result_pol_z = ground_state::calculate(ions, electrons, options::theory{}.lda(), scf_options, perturbations::simple_electric_field{{0.0, 0.0, amp}});

		match.check("total energy",        result_pol_z.energy.total(),             -22.424609857701);
		match.check("kinetic energy",      result_pol_z.energy.kinetic(),            14.384278182234);
		match.check("eigenvalues",         result_pol_z.energy.eigenvalues(),        -5.885829308879);
		match.check("Hartree energy",      result_pol_z.energy.hartree(),            28.446375748301);
		match.check("external energy",     result_pol_z.energy.external(),          -67.930073499256);
		match.check("non-local energy",    result_pol_z.energy.non_local(),          -2.608348211361);
		match.check("XC energy",           result_pol_z.energy.xc(),                 -5.956004079078);
		match.check("XC density integral", result_pol_z.energy.nvxc(),               -6.624437277099);
		match.check("HF exchange energy",  result_pol_z.energy.exact_exchange(),      0.000000000000);
		match.check("ion-ion energy",      result_pol_z.energy.ion(),                11.239162001458);

		match.check("dipole x", result_pol_z.dipole[0],  	 1.55026e-13);
		match.check("dipole y", result_pol_z.dipole[1],   -7.84581e-14);
		match.check("dipole z", result_pol_z.dipole[2], 	-0.00961886);
		
		auto pol_xx = (result_pol_x.dipole[0] - result.dipole[0])/amp;
		auto pol_zz = (result_pol_z.dipole[2] - result.dipole[2])/amp;

		match.check("polarizability xx", pol_xx,   8.525240626912, 1e-3);
		match.check("polarizability zz", pol_zz,  15.141008374656, 1e-3);

	}
	
	fftw_cleanup(); //required for valgrind
		
	return match.fail();

}

