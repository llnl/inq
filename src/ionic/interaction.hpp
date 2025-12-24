/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__IONIC__INTERACTION
#define INQ__IONIC__INTERACTION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <math/vector3.hpp>
#include <cmath>
#include <ionic/periodic_replicas.hpp>
#include <multi/array.hpp>
#include <limits>
#include <complex>
#include <hamiltonian/atomic_potential.hpp>

#include <utils/profiling.hpp>

namespace inq {
namespace ionic {

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class array_charge, class array_positions, class array_forces>
void interaction_energy_finite(const int natoms, const cell_type & cell, const array_charge & charge, const array_positions & positions, pseudo::math::erf_range_separation const & sep,
															 double & energy, array_forces & forces){
	
	energy = 0.0;
	for(int iatom = 0; iatom < natoms; iatom++) forces[iatom] = vector3<double>{0.0, 0.0, 0.0};
	
	for(int iatom = 0; iatom < natoms; iatom++){
		double zi = charge[iatom];

		for(int jatom = iatom + 1; jatom < natoms; jatom++){
			double zj = charge[jatom];

			vector3<double> rij =  cell.position_in_cell(positions[iatom]) - cell.position_in_cell(positions[jatom]);
			double invr = 1.0/sqrt(norm(rij));

			auto de = zi*zj*invr;
			auto df = de*invr*(rij*invr);

			energy += de;
			forces[iatom] += df;
			forces[jatom] -= df;
		}
	}
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class array_charge, class array_positions, class array_forces>
void ewald_fourier_3d(const int natoms, const cell_type & cell, const array_charge & charge, double total_charge, const array_positions & positions, double alpha,
											double & efs, array_forces & forces){
	 
	// G = 0 energy
	efs = -M_PI*total_charge*total_charge/(2.0*alpha*alpha*cell.volume());

	double gcut = std::numeric_limits<double>::max();
	for(int idir = 0; idir < 3; idir++) gcut = std::min(gcut, norm(cell.reciprocal(idir)));
	gcut = sqrt(gcut);
      
	const int isph = ceil(9.5*alpha/gcut);

	std::vector<std::complex<double> > phase(natoms);
	
	for(int ix = -isph; ix <= isph; ix++){
		for(int iy = -isph; iy <= isph; iy++){
			for(int iz = -isph; iz <= isph; iz++){
					
				const int ss = ix*ix + iy*iy + iz*iz;
					
				if(ss == 0 || ss > isph*isph) continue;
					
				vector3<double> gg = ix*cell.reciprocal(0) + iy*cell.reciprocal(1) + iz*cell.reciprocal(2);
				double gg2 = norm(gg);
					
				double exparg = -0.25*gg2/(alpha*alpha);
					
				if(exparg < -36.0) continue;
					
				double factor = 2.0*M_PI/cell.volume()*exp(exparg)/gg2;
					
				std::complex<double> sumatoms = 0.0;
				for(int iatom = 0; iatom < natoms; iatom++){
					double gx = dot(gg, positions[iatom]);
					auto aa = charge[iatom]*std::complex<double>(cos(gx), sin(gx));
					phase[iatom] = aa;
					sumatoms += aa;
				}
					
				efs += factor*std::real(sumatoms*std::conj(sumatoms));
					
				for(int iatom = 0; iatom < natoms; iatom++){
					for(int idir = 0; idir < 3; idir++){
						auto tmp = std::complex<double>(0.0, 1.0)*gg[idir]*phase[iatom];
						forces[iatom][idir] -= factor*std::real(std::conj(tmp)*sumatoms + tmp*std::conj(sumatoms));
					}
				}
					
			}
		}
	}
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class array_charge, class array_positions, class array_forces>
void ewald_fourier_2d(const int natoms, const cell_type & cell, const array_charge & charge, double total_charge, const array_positions & positions, double alpha,
											double & efs, array_forces & forces){
	
	//	In-Chul Yeh and Max L. Berkowitz, J. Chem. Phys. 111, 3155 (1999).

	auto dz_max = 0.0;
	for(int iatom = 0; iatom < natoms; iatom++){
		for(int jatom = 0; jatom < natoms; jatom++){
			dz_max = std::max(dz_max, fabs(positions[iatom][2] - positions[jatom][2]));
		}
	}

	auto rcut = 2.0*alpha*4.6 + 2.0*alpha*alpha*dz_max;

	while(true){
		if(rcut*dz_max >= 718) break;
		auto erfc1 = 1.0 - erf(alpha*dz_max + 0.5*rcut/alpha);
		if(erfc1*exp(rcut*dz_max) < 1e-10) break;
		rcut *= 1.414;
	}

	auto ix_max = ceil(rcut/length(cell.reciprocal(0)));
	auto iy_max = ceil(rcut/length(cell.reciprocal(1)));

	auto area_cell = fabs(cell.lattice(0)[0]*cell.lattice(1)[1] - cell.lattice(0)[1]*cell.reciprocal(1)[0]);

	efs = 0.0;

	for(int iatom = 0; iatom < natoms; iatom++){
		for(int jatom = 0; jatom < natoms; jatom++){

			auto factor = M_PI/area_cell;

			auto dz_ij = positions[iatom][2] - positions[jatom][2];
			auto tmp_erf = erf(alpha*dz_ij);
			auto factor1 = dz_ij*tmp_erf;
			auto factor2 = exp(-pow(alpha*dz_ij, 2))/(sqrt(M_PI)*alpha);

			efs -= factor*charge[iatom]*charge[jatom]*(factor1 + factor2);

			if(iatom == jatom) continue;
			if(fabs(tmp_erf) < 1e-16) continue;

			forces[iatom][2] -= -2.0*factor*charge[iatom]*charge[jatom]*tmp_erf;
		}
	}

	for(int ix = -ix_max; ix <= ix_max; ix++){
		for(int iy = -iy_max; iy <= iy_max; iy++){		

			auto ss = ix*ix + iy*iy;
			if(ss == 0) continue;

			auto gg = ix*cell.reciprocal(0) + iy*cell.reciprocal(1);
			auto gg2 = norm(gg);

			if(gg2 < 1e-16) continue;
			auto gg_abs = sqrt(gg2);
			auto factor = 0.5*M_PI/(area_cell*gg_abs);

			for(int iatom = 0; iatom < natoms; iatom++){
				for(int jatom = iatom; jatom < natoms; jatom++){

					auto gx = gg[0]*(positions[iatom][0] - positions[jatom][0]) + gg[1]*(positions[iatom][1] - positions[jatom][1]);
					auto dz_ij = positions[iatom][2] - positions[jatom][2];

					auto erfc1 = 1.0 - erf(alpha*dz_ij + 0.5*gg_abs/alpha);
					auto factor1 = exp(gg_abs*dz_ij)*erfc1;
					if(fabs(erfc1) <= 1e-16) factor1 = 0.0;

					auto erfc2 = 1.0 - erf(-alpha*dz_ij + 0.5*gg_abs/alpha);					
					auto factor2 = exp(-gg_abs*dz_ij)*erfc2;
					if(fabs(erfc2) <= 1e-16) factor2 = 0.0;
					
					auto coeff = 2.0;
					if(iatom == jatom) coeff = 1.0;

					efs += factor*coeff*charge[iatom]*charge[jatom]*cos(gx)*(factor1 + factor2);

					if(iatom == jatom) continue;

					forces[iatom][0] -= -2.0*factor*gg[0]*charge[iatom]*charge[jatom]*sin(gx)*(factor1 + factor2);
					forces[iatom][1] -= -2.0*factor*gg[1]*charge[iatom]*charge[jatom]*sin(gx)*(factor1 + factor2);					
					forces[jatom][0] += -2.0*factor*gg[0]*charge[iatom]*charge[jatom]*sin(gx)*(factor1 + factor2);
					forces[jatom][1] += -2.0*factor*gg[1]*charge[iatom]*charge[jatom]*sin(gx)*(factor1 + factor2);					

					factor1 = gg_abs*erfc1 - 2.0*alpha/sqrt(M_PI)*exp(-pow(alpha*dz_ij + 0.5*gg_abs/alpha, 2));
					if(fabs(factor1) > 1e-16){
						factor1 *= exp(gg_abs*dz_ij);
					} else {
						factor1 = 0.0;
					}

					factor2 = gg_abs*erfc2 - 2.0*alpha/sqrt(M_PI)*exp(-pow(-alpha*dz_ij + 0.5*gg_abs/alpha, 2));
					if(fabs(factor2) > 1e-16){
						factor2 *= exp(-gg_abs*dz_ij);
					} else {
						factor2 = 0.0;
					}

					forces[iatom][2] -= 2.0*factor*charge[iatom]*charge[jatom]*cos(gx)*(factor1 - factor2);
					forces[jatom][2] += 2.0*factor*charge[iatom]*charge[jatom]*cos(gx)*(factor1 - factor2);					
					
				}
			}
			
		}
	}
	
	
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class array_charge, class array_positions, class array_forces>
void interaction_energy_periodic(int periodicity, const int natoms, const cell_type & cell, const array_charge & charge, const array_positions & positions, pseudo::math::erf_range_separation const & sep,
																 double & energy, array_forces & forces){

	assert(periodicity == 2 or periodicity == 3);

	const double alpha = 0.21;

	double ers = 0.0;
	for(int iatom = 0; iatom < natoms; iatom++) forces[iatom] = vector3<double>(0.0, 0.0, 0.0);

	double rcut = 6.0/alpha;

	for(int iatom = 0; iatom < natoms; iatom++){
		double zi = charge[iatom];

		double maxdist = 0.0;
		for(int jatom = 0; jatom < natoms; jatom++){
			maxdist = std::max(maxdist, norm(cell.position_in_cell(positions[jatom]) - cell.position_in_cell(positions[iatom])));
		}
		
		periodic_replicas rep(cell, cell.position_in_cell(positions[iatom]), rcut + sqrt(maxdist));

		for(unsigned irep = 0; irep < rep.size(); irep++){
			vector3<double> xi = rep[irep];
				
			for(int jatom = 0; jatom < natoms; jatom++){
				double zj = charge[jatom];
					
				vector3<double> rij = xi - cell.position_in_cell(positions[jatom]);
				double rr = length(rij);
					
				if(rr < 1.0e-5) continue;
				
				auto eor = erfc(alpha*rr)/rr;
					
				ers += 0.5*zi*zj*eor;

				forces[jatom] -= zi*zj*rij*(eor + 2.0*alpha/sqrt(M_PI)*exp(-alpha*alpha*rr*rr))/(rr*rr);
			}
		}
	}
	
	double eself = 0.0;

	// self-interaction
	double total_charge = 0.0;
	for(int iatom = 0; iatom < natoms; iatom++){
		double zi = charge[iatom];

		total_charge += zi;
		eself -= alpha*zi*zi/sqrt(M_PI);
	}

	double efs;
	if(cell.periodicity() == 3){
		ewald_fourier_3d(natoms, cell, charge, total_charge, positions, alpha, efs, forces);
	} else 	if(cell.periodicity() == 2){
		ewald_fourier_2d(natoms, cell, charge, total_charge, positions, alpha, efs, forces);
	} else {
		efs = 0.0;
		assert(false);
	}

	double epseudo = 0.0;
	if(cell.periodicity() == 3){
		// Previously unaccounted G = 0 term from pseudopotentials. 
		// See J. Ihm, A. Zunger, M.L. Cohen, J. Phys. C 12, 4409 (1979)
		for(int iatom = 0; iatom < natoms; iatom++){
			epseudo += M_PI*charge[iatom]*pow(sep.sigma()*sqrt(2.0), 2)/cell.volume()*total_charge;
		}
	}

	energy = ers + eself + efs + epseudo;
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class array_charge, class array_positions, class array_forces>
void interaction_energy(const int natoms, const cell_type & cell, const array_charge & charge, const array_positions & positions, pseudo::math::erf_range_separation const & sep,
												double & energy, array_forces & forces){

	if(cell.periodicity() == 0) {
		interaction_energy_finite(natoms, cell, charge, positions, sep, energy, forces);
	} else if(cell.periodicity() == 2 or cell.periodicity() == 3) {
		interaction_energy_periodic(cell.periodicity(), natoms, cell, charge, positions, sep, energy, forces);
	} else {
		throw std::runtime_error("inq internal error: ionic interaction not implemented for periodicity " + std::to_string(cell.periodicity()));
	}
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class geometry_type>
auto interaction_energy(const cell_type & cell, const geometry_type & geo, const hamiltonian::atomic_potential & atomic_pot){

	CALI_CXX_MARK_FUNCTION;

	double energy;
	boost::multi::array<vector3<double>, 1> forces(geo.size());
	boost::multi::array<double, 1> charges(geo.size());

	for(int ii = 0; ii < geo.size(); ii++) charges[ii] = atomic_pot.pseudo_for_element(geo.species(ii)).valence_charge();

	interaction_energy(geo.size(), cell, charges, geo.positions(), atomic_pot.range_separation(), energy, forces);

	return energy;
}

///////////////////////////////////////////////////////////////////////////

template <class cell_type, class geometry_type>
auto interaction_forces(const cell_type & cell, const geometry_type & geo, const hamiltonian::atomic_potential & atomic_pot){

	CALI_CXX_MARK_FUNCTION;

	double energy;
	boost::multi::array<vector3<double>, 1> forces(geo.size());
	boost::multi::array<double, 1> charges(geo.size());

	for(int ii = 0; ii < geo.size(); ii++) charges[ii] = atomic_pot.pseudo_for_element(geo.species(ii)).valence_charge();

	interaction_energy(geo.size(), cell, charges, geo.positions(), atomic_pot.range_separation(), energy, forces);

	return forces;
}


}
}
#endif

#ifdef INQ_IONIC_INTERACTION_UNIT_TEST
#undef INQ_IONIC_INTERACTION_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <vector>
#include <valarray>
#include <gpu/array.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
  	const pseudo::math::erf_range_separation sep(0.625);
 
  SECTION("Aluminum cubic cell"){
  
    double aa = 7.653;
    
    systems::cell cell(vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa));
    
    std::valarray<double> charge(4);
    charge = 3.0;
    
    std::vector<vector3<double>> positions(4);
    positions[0] = vector3<double>(0.0,    0.0,    0.0);
    positions[1] = vector3<double>(aa/2.0, aa/2.0, 0.0);
    positions[2] = vector3<double>(aa/2.0, 0.0,    aa/2.0);
    positions[3] = vector3<double>(0.0,    aa/2.0, aa/2.0);
    
    double energy;
    std::vector<vector3<double>> forces(4);
    
    ionic::interaction_energy(4, cell, charge, positions, sep, energy, forces);
    
    CHECK(energy == -9.99517178_a); //this number comes from Octopus
    
  }

  SECTION("Diamond"){

    double aa = 6.74065308785213;
    
    systems::cell cell(vector3<double>(0.0, aa/2.0, aa/2.0), vector3<double>(aa/2.0, 0.0, aa/2.0), vector3<double>(aa/2.0, aa/2.0, 0.0));

    const double charge[2] = {4.0, 4.0};
    
    std::vector<vector3<double>> positions(2);
    positions[0] = cell.to_cartesian(vector3<double, contravariant>(0.0,  0.0,  0.0 ));
    positions[1] = cell.to_cartesian(vector3<double, contravariant>(0.25, 0.25, 0.25));
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

    CHECK(energy == -10.73490075_a); //this number comes from Octopus

  }

  SECTION("Iron"){
    
    double aa = 5.3970578;
    
    systems::cell cell(vector3<double>(-aa/2.0, aa/2.0, aa/2.0), vector3<double>(aa/2.0, -aa/2.0, aa/2.0), vector3<double>(aa/2.0, aa/2.0, -aa/2.0));

    const double charge = 16.0;
    
    const vector3<double> position(0.0, 0.0, 0.0);
    
    double energy;
    std::vector<vector3<double>> forces(1);

    ionic::interaction_energy(1, cell, &charge, &position, sep, energy, forces);

    CHECK(energy == -78.31680646_a); //this number comes from Octopus
    
  }

	SECTION("N2 finite"){
    
    double aa = 20.0;
    
    systems::cell cell(vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa), /* periodicity = */ 0);

		const double charge[2] = {5.0, 5.0};

    double distance = 2.0739744;

    std::vector<vector3<double>> positions(2);
    positions[0] = vector3<double>(0.0, 0.0, -distance/2.0);
    positions[1] = vector3<double>(0.0, 0.0,  distance/2.0);
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == 12.05415072_a); 

		CHECK(fabs(forces[0][0]) < 1.0e-16);
		CHECK(fabs(forces[0][1]) < 1.0e-16);
		CHECK(forces[0][2] == -5.81210198_a);
		CHECK(fabs(forces[1][0]) < 1.0e-16);
		CHECK(fabs(forces[1][1]) < 1.0e-16);
		CHECK(forces[1][2] == 5.81210198_a);
    
  }
		
  SECTION("N2 supercell"){
    
    double aa = 20.0;
    
    systems::cell cell(vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa));

		const double charge[2] = {5.0, 5.0};

    double distance = 2.0739744;

    std::vector<vector3<double>> positions(2);
    positions[0] = vector3<double>(0.0, 0.0, -distance/2.0);
    positions[1] = vector3<double>(0.0, 0.0,  distance/2.0);
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == 5.02018926_a); 

		CHECK(fabs(forces[0][0]) < 1.0e-16);
		CHECK(fabs(forces[0][1]) < 1.0e-16);
		CHECK(forces[0][2] == -5.7840844208_a);
		CHECK(fabs(forces[1][0]) < 1.0e-16);
		CHECK(fabs(forces[1][1]) < 1.0e-16);
		CHECK(forces[1][2] == 5.7840844208_a);
    
  }
	
	SECTION("N2 supercell shifted"){
    
    double aa = 20.0;
    
    systems::cell cell(vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa));

		const double charge[2] = {5.0, 5.0};

    double distance = 2.0739744;

    std::vector<vector3<double>> positions(2);
    positions[0] = vector3<double>(0.0, 0.0, 293.9 -distance/2.0);
    positions[1] = vector3<double>(0.0, 0.0, 293.9 + distance/2.0);
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == 5.02018926_a); 

		CHECK(fabs(forces[0][0]) < 1.0e-16);
		CHECK(fabs(forces[0][1]) < 1.0e-16);
		CHECK(forces[0][2] == -5.7840844208_a);
		CHECK(fabs(forces[1][0]) < 1.0e-16);
		CHECK(fabs(forces[1][1]) < 1.0e-16);
		CHECK(forces[1][2] == 5.7840844208_a);
    
  }
		
	SECTION("Si64"){
    
    double alat = 20.25;
    
    systems::cell cell(vector3<double>(alat, 0.0, 0.0), vector3<double>(0.0, alat, 0.0), vector3<double>(0.0, 0.0, alat));

		gpu::array<double, 1> charge(64, 4.0);

    std::vector<vector3<double>> positions;
		
		positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.000000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.000000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.500000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.500000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.000000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.000000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.500000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.500000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.250000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.250000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.750000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.000000000,   alat*0.750000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.250000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.250000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.750000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.500000000,   alat*0.750000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.250000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.250000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.750000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.750000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.250000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.250000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.750000000,    alat*0.000000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.750000000,    alat*0.500000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.000000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.000000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.500000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.250000000,   alat*0.500000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.000000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.000000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.500000000,    alat*0.250000000));
	  positions.push_back(vector3<double>(  alat*0.750000000,   alat*0.500000000,    alat*0.750000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.125000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.125000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.625000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.625000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.125000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.125000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.625000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.625000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.125000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.125000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.625000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.625000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.125000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.125000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.625000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.625000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.375000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.375000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.875000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.125000000,   alat*0.875000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.375000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.375000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.875000000,    alat*0.375000000));
	  positions.push_back(vector3<double>(  alat*0.625000000,   alat*0.875000000,    alat*0.875000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.375000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.375000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.875000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.375000000,   alat*0.875000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.375000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.375000000,    alat*0.625000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.875000000,    alat*0.125000000));
	  positions.push_back(vector3<double>(  alat*0.875000000,   alat*0.875000000,    alat*0.625000000));
		
    double energy;
    std::vector<vector3<double>> forces(64);

    ionic::interaction_energy(64, cell, charge, positions, sep, energy, forces);

    CHECK(energy == -253.0283966274_a); 

		for(int iatom = 0; iatom < 64; iatom++){
			CHECK(length(forces[iatom]) < 1e-12);
		}
  }

	SECTION("H2O finite"){
    
    double aa = 20.0;
    
    systems::cell cell(vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa), /* periodicity = */ 0);

		const double charge[3] = {6.0, 1.0, 1.0};

    std::vector<vector3<double>> positions(3);
    positions[0] = vector3<double>( 0.0,      -0.553586, 0.0);
    positions[1] = vector3<double>( 1.429937,  0.553586, 0.0);
    positions[2] = vector3<double>(-1.429937,  0.553586, 0.0);

    double energy;
    std::vector<vector3<double>> forces(3);

    ionic::interaction_energy(3, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == 6.98512326_a); 

		CHECK(fabs(forces[0][0]) < 1.0e-12);
		CHECK(forces[0][1] == -2.2462868673626861_a);
		CHECK(fabs(forces[0][2]) < 1.0e-12);

		CHECK(forces[1][0] == 1.5728305979055519_a);
		CHECK(forces[1][1] == 1.1231434336813431_a);		
		CHECK(fabs(forces[1][2]) < 1.0e-12);

		CHECK(forces[2][0] == -1.5728305979055519_a);
		CHECK(forces[2][1] == 1.1231434336813431_a);
		CHECK(fabs(forces[2][2]) < 1.0e-12);
		
  }

	SECTION("BN"){
    
    auto aa = 2.7401029;
		auto lx = 3*aa;
		auto ly = sqrt(3.0)*aa;
		auto lz = 7.5589045;
    
    systems::cell cell(vector3<double>(lx, 0.0, 0.0), vector3<double>(0.0, ly, 0.0), vector3<double>(0.0, 0.0, lz), 2);

		const double charge[4] = {3.0, 5.0, 3.0, 5.0};

    std::vector<vector3<double>> positions(4);
    positions[0] = vector3<double>(0.0,        0.0,    0.0);
    positions[1] = vector3<double>(2.0/3.0*lx, 0.0,    0.0);
		positions[2] = vector3<double>(0.5*lx,     0.5*ly, 0.0);
		positions[3] = vector3<double>(1.0/6.0*lx, 0.5*ly, 0.0);
    
    double energy;
    std::vector<vector3<double>> forces(4);

    ionic::interaction_energy(4, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == -39.9332202862_a); 

		CHECK(fabs(forces[0][0]) < 1.0e-12);
		CHECK(fabs(forces[0][1]) < 1.0e-12);
		CHECK(fabs(forces[0][2]) < 1.0e-12);
		CHECK(fabs(forces[1][0]) < 1.0e-12);
		CHECK(fabs(forces[1][1]) < 1.0e-12);	
		CHECK(fabs(forces[1][2]) < 1.0e-12);
		CHECK(fabs(forces[2][0]) < 1.0e-12);
		CHECK(fabs(forces[2][1]) < 1.0e-12);	
		CHECK(fabs(forces[2][2]) < 1.0e-12);
		CHECK(fabs(forces[3][0]) < 1.0e-12);
		CHECK(fabs(forces[3][1]) < 1.0e-12);	
		CHECK(fabs(forces[3][2]) < 1.0e-12);
    
  }
	
	SECTION("BN displaced"){
    
    auto aa = 2.7401029;
		auto lx = 3*aa;
		auto ly = sqrt(3.0)*aa;
		auto lz = 7.5589045;
    
    systems::cell cell(vector3<double>(lx, 0.0, 0.0), vector3<double>(0.0, ly, 0.0), vector3<double>(0.0, 0.0, lz), 2);

		const double charge[4] = {3.0, 5.0, 3.0, 5.0};

    std::vector<vector3<double>> positions(4);
    positions[0] = vector3<double>(0.0,        0.0,    0.0);
    positions[1] = vector3<double>(2.0/3.0*lx, 0.0,    -1.0);
		positions[2] = vector3<double>(0.5*lx,     0.5*ly, 1.0);
		positions[3] = vector3<double>(1.0/6.0*lx, 0.5*ly, 0.0);
    
    double energy;
    std::vector<vector3<double>> forces(4);

    ionic::interaction_energy(4, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == -45.1682138928_a); 

		CHECK(forces[0][0] == -0.33195958936549447_a);
		CHECK(fabs(forces[0][1]) < 1.0e-12);
		CHECK(forces[0][2] == 0.69323391694246583_a);
		CHECK(forces[1][0] == -0.57710256739291033_a);
		CHECK(fabs(forces[1][1]) < 1.0e-12);	
		CHECK(forces[1][2] == -5.0157974583153031_a);
		CHECK(forces[2][0] == 0.57710256739291232_a);
		CHECK(fabs(forces[2][1]) < 1.0e-12);	
		CHECK(forces[2][2] == 4.0937958343994687_a);
		CHECK(forces[3][0] == 0.33195958936549214_a);
		CHECK(fabs(forces[3][1]) < 1.0e-12);	
		CHECK(forces[3][2] == 0.22876770697336701_a);
		
  }

	SECTION("Graphene 3D periodicity"){
    
		auto dcc = 2.6834111;
		auto aa = sqrt(3)*dcc;
		auto lz = 10.0;
    
    systems::cell cell(aa*vector3<double>(1.0, 0.0, 0.0), aa*vector3<double>(-1.0/2.0, sqrt(3.0)/2.0, 0.0), vector3<double>(0.0, 0.0, lz), 3);

		const double charge[2] = {4.0, 4.0};

    std::vector<vector3<double>> positions(2);
    positions[0] = vector3<double>(0.0, 0.0, 0.0);
    positions[1] = vector3<double>(0.0, dcc, 0.0);
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == -1.0617337162_a);
		
		CHECK(fabs(forces[0][0]) < 1.0e-12);
		CHECK(fabs(forces[0][1]) < 1.0e-12);
		CHECK(fabs(forces[0][2]) < 1.0e-12);
		CHECK(fabs(forces[1][0]) < 1.0e-12);
		CHECK(fabs(forces[1][1]) < 1.0e-12);
  }
	
	SECTION("Graphene 2D periodicity"){
    
		auto dcc = 2.6834111;
		auto aa = sqrt(3)*dcc;
		auto lz = 10.0;
    
    systems::cell cell(aa*vector3<double>(1.0, 0.0, 0.0), aa*vector3<double>(-1.0/2.0, sqrt(3.0)/2.0, 0.0), vector3<double>(0.0, 0.0, lz), 2);

		const double charge[2] = {4.0, 4.0};

    std::vector<vector3<double>> positions(2);
    positions[0] = vector3<double>(0.0, 0.0, 0.0);
    positions[1] = vector3<double>(0.0, dcc, 0.0);
    
    double energy;
    std::vector<vector3<double>> forces(2);

    ionic::interaction_energy(2, cell, charge, positions, sep, energy, forces);

		//these numbers come from Octopus
    CHECK(energy == -19.8137164427_a);
		
		CHECK(fabs(forces[0][0]) < 1.0e-12);
		CHECK(fabs(forces[0][1]) < 1.0e-12);
		CHECK(fabs(forces[0][2]) < 1.0e-12);
		CHECK(fabs(forces[1][0]) < 1.0e-12);
		CHECK(fabs(forces[1][1]) < 1.0e-12);
  }	
	
}
#endif
