/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <utils/match.hpp>
#include <ground_state/initial_guess.hpp>

#include <input/environment.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;

  int const reps = 10;
  
	auto & env = inq::input::environment::global();	

  parallel::cartesian_communicator<2> cart_comm(env.comm(), {1, boost::mpi3::fill});

	auto basis_comm = cart_comm.axis(1);

  for(auto ecut = 25.0_Ha; ecut <= 300.0_Ha; ecut += 25.0_Ha){

    auto spacing = M_PI*sqrt(0.5/ecut.in_atomic_units());
    basis::real_space rs(systems::cell::cubic(6.66_b), spacing, basis_comm);
    
    for(int ist = 1; ist <= 256; ist *= 2){
      
      basis::field_set<basis::real_space, complex> phi(rs, ist, cart_comm);
      
      auto iter_start_time = std::chrono::high_resolution_clock::now();
      
      for(int irep = 0; irep < reps; irep++){
        auto fphi = operations::transform::to_fourier(phi);
      }
      
      auto new_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
    
      std::cout << ecut.in_atomic_units() << '\t' << ist << '\t' << elapsed_seconds.count()/reps*1000.0 << "\tms" << std::endl;

    }
  }
  
}
