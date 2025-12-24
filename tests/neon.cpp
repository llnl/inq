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

	auto comm = input::environment::global().comm();

	utils::match energy_match(2.0e-5);

	interface::clear();
	
	//inq cell cubic 15.0 bohr finite
	interface::cell.cubic(15.0_b, 0);

	//inq cell
	interface::cell();

	//inq ions add Ne 0.0 0.0 0.0 bohr
	interface::ions.insert("Ne", {0.0_b, 0.0_b, 0.0_b});

  //inq ions
	interface::ions();
	
	//inq electrons extra_states 3
	interface::electrons.extra_states(3);
	
	//inq electrons cutoff 30.0 Ha
	interface::electrons.cutoff(30.0_Ha);

	//inq theory non_interacting
	interface::theory.non_interacting();

	//REAL SPACE PSEUDO
	//inq run ground_state
	interface::run.ground_state();
	
	energy_match &= interface::util.match(interface::results_ground_state.energy_total()      , -61.821394985540, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_kinetic()    ,  35.822176514976, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_eigenvalues(), -61.821394985540, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_hartree()    ,   0.000000000000, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_external()   , -79.551722060593, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_non_local()  , -18.091849439924, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_xc()         ,   0.000000000000, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_nvxc()       ,   0.000000000000, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_nvxc()       ,   0.000000000000, 2.0e-5);
	energy_match &= interface::util.match(interface::results_ground_state.energy_ion()        ,   0.000000000000, 2.0e-5);
	
	return energy_match.fail();
}
