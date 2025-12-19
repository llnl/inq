/* -*- indent-tabs-mode: t -*- */

/*
 Copyright (C) 2019 Xavier Andrade

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <ground_state/calculator.hpp>
#include <ground_state/initial_guess.hpp>
#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <utils/match.hpp>

#include <input/environment.hpp>

template <class LatticeParameters>
auto total_energy(inq::input::environment &env, LatticeParameters lat) {
  using namespace inq;
  using namespace inq::magnitude;

  auto const box = systems::cell::cubic(lat);

  systems::ions const ions(box);

  systems::electrons electrons(env.par(), ions,
                               options::electrons{}
                                   .cutoff(30.0_Ha)
                                   .temperature(300.0_K)
                                   .extra_electrons(14.0)
                                   .extra_states(2),
                               input::kpoints::grid({1, 1, 1}, false));

  ground_state::initial_guess(ions, electrons);

  return ground_state::calculate(ions, electrons, options::theory{}.lda(),
                                 inq::options::ground_state{}
                                     .energy_tolerance(1e-9_Ha)
                                     .calculate_forces());
}

int main(int argc, char **argv) {

  using namespace inq;
  using namespace inq::magnitude;

  input::environment env;

  utils::match match(1.0e-5);

  auto const lat1_in_bohr = 10.0;
  auto const vol1_in_bohr3 = std::pow(lat1_in_bohr, 3);

  auto const result1 = total_energy(env, lat1_in_bohr*1.0_b);

  auto const lat2_in_bohr = 10.01;
  auto const vol2_in_bohr3 = std::pow(lat2_in_bohr, 3);

  auto const result2 = total_energy(env, lat2_in_bohr*1.0_b);
							
  auto const stress = result1.stress;
  std::cout << "stress" << std::endl;
  std::cout << stress[0][0] << " " << stress[0][1] << " " << stress[0][2]
            << '\n';
  std::cout << stress[1][0] << " " << stress[1][1] << " " << stress[1][2]
            << '\n';
  std::cout << stress[2][0] << " " << stress[2][1] << " " << stress[2][2]
            << '\n';

  std::cout << "energies (1, 2) = " << result1.energy.total() << " "
            << result2.energy.total() << std::endl;
  std::cout << "volumes (1, 2) = " << vol1_in_bohr3 << " " << vol2_in_bohr3
            << std::endl;

  std::cout << "(energy(2) - energy(1) ) / (vo2 - vol1) = "
            << (result2.energy.total() - result1.energy.total())/(vol2_in_bohr3 - vol1_in_bohr3)
            << std::endl;


  match.check("total energy 1",   result1.energy.total(),  -0.684940553454);
	match.check("total energy 2",   result2.energy.total(),  -0.686911961588);
	match.check("finite diff pressure", (result2.energy.total() - result1.energy.total())/(vol2_in_bohr3 - vol1_in_bohr3), -0.000656479347, 1e-7);

	match.check("stress xx", stress[0][0], -0.000658598015, 1e-8);	
	match.check("stress yy", stress[1][1], -0.000658598015, 1e-8);
	match.check("stress zz", stress[2][2], -0.000658598015, 1e-8);

	match.check("stress xy", stress[0][1], 0.0, 1e-8);	
	match.check("stress xz", stress[0][2], 0.0, 1e-8);
	match.check("stress yz", stress[1][2], 0.0, 1e-8);

	match.check("stress yx", stress[1][0], 0.0, 1e-8);	
	match.check("stress zx", stress[2][0], 0.0, 1e-8);
	match.check("stress zy", stress[2][1], 0.0, 1e-8);
	
	return match.fail();
}
