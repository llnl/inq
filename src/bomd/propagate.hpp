/* -*- indent-tabs-mode: t -*- */
#ifndef INQ__BOMD__PROPAGATE
#define INQ__BOMD__PROPAGATE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <ground_state/calculator.hpp>
#include <ionic/propagator.hpp>
#include <options/real_time.hpp>
#include <real_time/viewables.hpp>
#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <utils/profiling.hpp>

#include <chrono>

namespace inq {
namespace bomd {

template <typename ProcessFunction, typename IonSubPropagator = ionic::propagator::molecular_dynamics>
void propagate(systems::ions & ions, systems::electrons & electrons,  ProcessFunction func, const options::theory & inter, const options::real_time & opts, IonSubPropagator const& ion_propagator = {}){
		CALI_CXX_MARK_FUNCTION;

		using namespace inq::magnitude;
		
		auto console = electrons.logger();

		const double dt = opts.dt();
		const int numsteps = opts.num_steps();

		if(console) console->trace(std::string("initializing real-time propagation:\n") +
					   std::string("  time step        = {} atomictime ({:.2f} fs)\n") + 
					   std::string("  number of steps  = {}\n") + 
					   std::string("  propagation time = {} atomictime ({:.2f} fs)"), dt, dt*1.0_atomictime/1.0_fs, numsteps, numsteps*dt, numsteps*dt*1.0_atomictime/1.0_fs);

		auto calculate_gs = ground_state::calculator(ions, electrons, inter, options::ground_state{}.calculate_forces().silent());

		ground_state::results::energy_type energy;
		ground_state::results::forces_type forces;
		
		if(console) console->trace("starting Born-Oppenheimer propagation");

		auto iter_start_time = std::chrono::high_resolution_clock::now();
		for(int istep = 0; istep < numsteps + 1; istep++){
			CALI_CXX_MARK_SCOPE("time_step");

			auto time = istep*dt;
			
			if(istep > 0) ion_propagator.propagate_positions(dt, ions, forces);
			
			auto res = calculate_gs(electrons);
			energy = res.energy;
			forces = res.forces;

			if(istep > 0) ion_propagator.propagate_velocities(dt, ions, forces);

			auto new_time = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;

			func(real_time::viewables{istep == numsteps, istep, time, ions, electrons, energy, forces, calculate_gs.hamiltonian(), perturbations::none{}});
			
			if(console) console->info("step {:9d} :  t =  {:9.3f}  e = {:.12f}  scf_iter = {:4d}  wtime = {:9.3f}", istep, time, energy.total(), res.total_iter, elapsed_seconds.count());

			iter_start_time = new_time;
		}

		if(console) console->trace("Born-Oppenheimer propagation ended normally");
	}

}
}
#endif

#ifdef INQ_BOMD_PROPAGATE_UNIT_TEST
#undef INQ_BOMD_PROPAGATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
