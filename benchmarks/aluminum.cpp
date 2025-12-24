/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/inq.hpp>

#include <sstream>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;

	auto & env = inq::input::environment::global();

	int pardomains = 1;
	bool groundstate_only = false;
	vector3<int, contravariant> reps{2, 2, 2};
	vector3<int> kpoints{1, 1, 1};
	int niter = 10;
	
	auto functional = options::theory{}.pbe();
	
	{
		int opt;
		while ((opt = getopt(argc, argv, "p:?gs:k:yi:")) != EOF){
			switch(opt){
			case 'p':
				pardomains = atoi(optarg);
				break;
			case 'g':
				groundstate_only = true;
				break;
			case 's':
				sscanf(optarg, "%d%*c%d%*c%d", &reps[0], &reps[1], &reps[2]);
				break;
			case 'k':
				sscanf(optarg, "%d%*c%d%*c%d", &kpoints[0], &kpoints[1], &kpoints[2]);
				break;
			case 'y':
				functional = options::theory{}.pbe0();
				break;
 		  case 'i':
				niter = atoi(optarg);
				break;
			case '?':
				std::cerr << "usage is " << std::endl;
				std::cerr << "-p N to set the number of processors in the domain partition (1 by default)." << std::endl;
				std::cerr << "-g only calculate the ground state." << std::endl;
				std::cerr << "-s NxMxK the supercell size in each direction (2x2x2 by default)." << std::endl;
				std::cerr << "-s NxMxK the size of the kpoint grid (1x1x1 by default)." << std::endl;
				std::cerr << "-y use the PBE0 hybrid functional (the default is PBE)." << std::endl;
				std::cerr << "-i N to set the number of SCF iterations to run (10 by default)." << std::endl;	
				exit(1);
			default:
				abort();
			}
		}
	}
	
	auto alat = 7.6524459_bohr;

	using frac_coord = vector3<double, contravariant>;
	
	std::vector<frac_coord> cell;

	cell.emplace_back(frac_coord{0.0, 0.0, 0.0});
	cell.emplace_back(frac_coord{0.0, 0.5, 0.5});
	cell.emplace_back(frac_coord{0.5, 0.0, 0.5});
	cell.emplace_back(frac_coord{0.5, 0.5, 0.0});
	
	systems::ions ions(systems::cell::orthorhombic(reps[0]*alat, reps[1]*alat, reps[2]*alat));
	
	for(int ix = 0; ix < reps[0]; ix++){
		for(int iy = 0; iy < reps[1]; iy++){
			for(int iz = 0; iz < reps[2]; iz++){
				auto base = frac_coord{double(ix), double(iy), double(iz)};
				for(unsigned iatom = 0; iatom < cell.size(); iatom++){
					ions.insert_fractional("Al", (base + cell[iatom])/reps);
				}
			}
		}
	}
	
	assert(int(ions.size()) == int(cell.size()*product(reps)));
				 
	systems::electrons electrons(env.par().states().domains(pardomains), ions, options::electrons{}.spacing(alat/20).extra_states(2*product(reps)).temperature(1000.0_K), input::kpoints::grid(kpoints));
	
	auto restart_dir = "aluminum_" + std::to_string(reps[0]) + "_" + std::to_string(reps[1]) + "_" + std::to_string(reps[2]);

	if(kpoints[0] != 1 and kpoints[1] != 1 and kpoints[2] != 1) {
		restart_dir += 	"_k" + std::to_string(kpoints[0]) + "_" + std::to_string(kpoints[1]) + "_" + std::to_string(kpoints[2]);
	}
	
	auto not_found_gs = groundstate_only or not electrons.try_load(restart_dir);

	if(not_found_gs){
		ground_state::initial_guess(ions, electrons);
		try {	ground_state::calculate(ions, electrons, functional, inq::options::ground_state{}.steepest_descent().max_steps(niter).mixing(0.1)); }
		catch (...) { }
		if(not groundstate_only) electrons.save(restart_dir);
	}

	if(not groundstate_only){
		real_time::propagate(ions, electrons, [](auto){}, functional, options::real_time{}.num_steps(100).dt(0.05_atomictime));
	}
	
}

