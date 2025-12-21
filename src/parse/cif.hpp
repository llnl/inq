/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PARSE__CIF
#define INQ__PARSE__CIF

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cmath>
#include <gemmi/cif.hpp>
#include <gemmi/symmetry.hpp>
#include <vector>

#include <ionic/species.hpp>
#include <magnitude/length.hpp>
#include <math/vector3.hpp>

namespace inq {
namespace parse {

class cif {

	std::vector<vector3<double>> lattice_vectors_;
	std::vector<ionic::species> atoms_;
	std::vector<vector3<double, contravariant>> positions_;

  static auto to_double(std::string const * strptr, std::string const & cif_file_name){
    if(strptr == NULL) throw std::runtime_error("Error: cannot read cell parameters from CIF file '" + cif_file_name + "'.");
    return stod(*strptr);
  }
  
public:

	static auto detect(std::string const & filename){
		std::string extension = utils::lowercase(filename.substr(filename.find_last_of(".") + 1));

		return extension == "cif";
	}
	
	cif(const std::string & cif_file_name):
		lattice_vectors_(3)
	{
    namespace cif = gemmi::cif;
    cif::Document doc = cif::read_file(cif_file_name);
		
		//THE CELL
		auto length_a = to_double(doc.sole_block().find_value("_cell_length_a"), cif_file_name);
    auto length_b = to_double(doc.sole_block().find_value("_cell_length_b"), cif_file_name);
    auto length_c = to_double(doc.sole_block().find_value("_cell_length_c"), cif_file_name);
    auto angle_alpha = to_double(doc.sole_block().find_value("_cell_angle_alpha"), cif_file_name);
    auto angle_beta = to_double(doc.sole_block().find_value("_cell_angle_beta"), cif_file_name);
    auto angle_gamma = to_double(doc.sole_block().find_value("_cell_angle_gamma"), cif_file_name);

    angle_alpha *= M_PI/180.0;
    angle_beta  *= M_PI/180.0;
    angle_gamma *= M_PI/180.0;

    using namespace inq::magnitude;
    auto angstrom = in_atomic_units(1.0_A);
    
    lattice_vectors_[0] = angstrom*length_a*vector3<double>({1.0, 0.0, 0.0});
    lattice_vectors_[1] = angstrom*length_b*vector3<double>({cos(angle_gamma), sin(angle_gamma), 0.0});

    auto cosbeta = cos(angle_beta);
    auto c1 = (cos(angle_alpha) - cos(angle_beta)*cos(angle_gamma))/sin(angle_gamma);
    auto c2 = sqrt(1.0 - cosbeta*cosbeta - c1*c1);
    lattice_vectors_[2] = angstrom*length_c*vector3<double>({cosbeta, c1, c2});

		auto cell = systems::cell(lattice_vectors_[0], lattice_vectors_[1], lattice_vectors_[2]);
		
		//SYMMETRIES
		auto symmetries = doc.sole_block().find_loop("_space_group_symop_operation_xyz");
		if(symmetries.length() == 0) symmetries = doc.sole_block().find_loop("_symmetry_equiv_pos_as_xyz");

		
		//THE ATOMS
		auto atom_symbols = doc.sole_block().find_loop("_atom_site_label");
		if(atom_symbols.length() == 0) throw std::runtime_error("Error: cannot find the atoms in CIF file '" + cif_file_name + "'.");

		auto natoms = atom_symbols.length();
		
		auto atom_x = doc.sole_block().find_loop("_atom_site_fract_x");
		auto atom_y = doc.sole_block().find_loop("_atom_site_fract_y");
		auto atom_z = doc.sole_block().find_loop("_atom_site_fract_z");

		if(natoms != atom_x.length() or natoms != atom_y.length() or natoms != atom_z.length()){
			throw std::runtime_error("Error: cannot read the atomic coordinates from CIF file '" + cif_file_name + "'.");
		}

		std::vector<std::string> symbols;
		
		for(int iatom = 0; iatom < natoms; iatom++){
			auto symbol = atom_symbols[iatom];
			symbol.erase(std::remove_if(symbol.begin(), symbol.end(), ::isdigit), symbol.end()); //remove the numbers
			auto pos = vector3<double, contravariant>{stod(atom_x[iatom]), stod(atom_y[iatom]), stod(atom_z[iatom])};
			pos = cell.position_in_cell(pos);
			
			symbols.emplace_back(symbol);
			positions_.emplace_back(pos);

			for(auto & symm: symmetries){
				symm.erase(std::remove(symm.begin(), symm.end(), '\''), symm.end());
				auto op = gemmi::parse_triplet(symm);
				auto symm_pos = vector3<double, contravariant>{op.apply_to_xyz(std::array<double, 3>{pos[0], pos[1], pos[2]})};
				symm_pos = cell.position_in_cell(symm_pos);
				
				auto duplicated = false;
				
				for(int jatom = 0; jatom < (int) symbols.size(); jatom++){
					if(cell.norm(symm_pos - positions_[jatom]) > 1e-8) continue;
					
					if(symbol != symbols[jatom]) throw std::runtime_error("Error: the file '" + cif_file_name + "' contains two different atoms in the same position.");
					duplicated = true;
					break;
				}

				if(duplicated) continue;

				symbols.emplace_back(symbol);
				positions_.emplace_back(symm_pos);
			}
		}

		assert(symbols.size() == positions_.size());
		
		for(auto iatom = 0; iatom < (int) symbols.size(); iatom++) atoms_.emplace_back(symbols[iatom]);
	}
	
	auto size() const {
		return long(positions_.size());
	}	
	
	auto & lattice() const {
		return lattice_vectors_;
	}

	auto cell() const {
		return systems::cell(lattice_vectors_[0], lattice_vectors_[1], lattice_vectors_[2]);		
	}

	auto & atoms() const {
		return atoms_;
	}

	auto & positions() const {
		return positions_;
	}	
	
};


}
}
#endif

#ifdef INQ_PARSE_CIF_UNIT_TEST
#undef INQ_PARSE_CIF_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <config/path.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Al"){
		
		parse::cif cif_file(config::path::unit_tests_data() + "Al.cif");

		CHECK(cif_file.lattice()[0][0] == 7.634890386_a);
		CHECK(cif_file.lattice()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[0][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][1] == 7.634890386_a);
		CHECK(cif_file.lattice()[1][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][2] == 7.634890386_a);		
		
		CHECK(cif_file.size() == 4);

		CHECK(cif_file.atoms()[0] == "Al");
		CHECK(cif_file.atoms()[1] == "Al");
		CHECK(cif_file.atoms()[2] == "Al");		
		CHECK(cif_file.atoms()[3] == "Al");

		CHECK(cif_file.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][2] == Approx(0.0).margin(1e-12));
		
		CHECK(cif_file.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][1] == -0.5_a);
		CHECK(cif_file.positions()[1][2] == -0.5_a);

		CHECK(cif_file.positions()[2][0] == -0.5_a);
		CHECK(cif_file.positions()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[2][2] == -0.5_a);

		CHECK(cif_file.positions()[3][0] == -0.5_a);
		CHECK(cif_file.positions()[3][1] == -0.5_a);
		CHECK(cif_file.positions()[3][2] == Approx(0.0).margin(1e-12));
	}
	
	SECTION("SrTiO3"){
		
		parse::cif cif_file(config::path::unit_tests_data() + "9002806.cif");

		CHECK(cif_file.lattice()[0][0] == 10.4316661532_a);
		CHECK(cif_file.lattice()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[0][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][1] == 10.4316661532_a);
		CHECK(cif_file.lattice()[1][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][2] == 14.7525249371_a);		
		
		CHECK(cif_file.size() == 20);

		//Conversions for units and cell convention to compare with obabel
		//generated positions
		//
		//  1.38005 ->  0.25
		//  1.95168 ->  0.25
		//  2.76010 -> -0.5
		//  3.90335 -> -0.5
		//  4.14015 -> -0.25
		//  5.85503 -> -0.25
		
		//Sr         0.00000        0.00000        1.95168
		CHECK(cif_file.atoms()[0] == "Sr");
		CHECK(cif_file.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][2] == 0.25_a);

		//Sr         0.00000        0.00000        5.85503
		CHECK(cif_file.atoms()[1] == "Sr");
		CHECK(cif_file.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][2] == -0.25_a);

		//Sr         2.76010        2.76010        5.85503
		CHECK(cif_file.atoms()[2] == "Sr");
		CHECK(cif_file.positions()[2][0] == -0.5_a);
		CHECK(cif_file.positions()[2][1] == -0.5_a);
		CHECK(cif_file.positions()[2][2] == -0.25_a);

		//Sr         2.76010        2.76010        1.95167
		CHECK(cif_file.atoms()[3] == "Sr");
		CHECK(cif_file.positions()[3][0] == -0.5_a);
		CHECK(cif_file.positions()[3][1] == -0.5_a);
		CHECK(cif_file.positions()[3][2] ==  0.25_a);

		//Ti         2.76010        0.00000        0.00000
		CHECK(cif_file.atoms()[4] == "Ti");
		CHECK(cif_file.positions()[4][0] == -0.5_a);
		CHECK(cif_file.positions()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[4][2] == Approx(0.0).margin(1e-12));

		//Ti         2.76010        0.00000        3.90335
		CHECK(cif_file.atoms()[5] == "Ti");
		CHECK(cif_file.positions()[5][0] == -0.5_a);
		CHECK(cif_file.positions()[5][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[5][2] == -0.5_a);

		//Ti         0.00000        2.76010        3.90335
		CHECK(cif_file.atoms()[6] == "Ti");
		CHECK(cif_file.positions()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[6][1] == -0.5_a);
		CHECK(cif_file.positions()[6][2] == -0.5_a);

		//Ti         0.00000        2.76010        0.00000
		CHECK(cif_file.atoms()[7] == "Ti");
		CHECK(cif_file.positions()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[7][1] == -0.5_a);
		CHECK(cif_file.positions()[7][2] == Approx(0.0).margin(1e-12));

		//O          0.00000        2.76010        1.95168
		CHECK(cif_file.atoms()[8] == "O");
		CHECK(cif_file.positions()[8][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[8][1] == -0.5_a);
		CHECK(cif_file.positions()[8][2] == 0.25_a);

		//O          0.00000        2.76010        5.85503
		CHECK(cif_file.atoms()[9] == "O");
		CHECK(cif_file.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[9][1] == -0.5_a);
		CHECK(cif_file.positions()[9][2] == -0.25_a);

		//O          2.76010        0.00000        5.85503
		CHECK(cif_file.atoms()[10] == "O");
		CHECK(cif_file.positions()[10][0] == -0.5_a);
		CHECK(cif_file.positions()[10][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[10][2] == -0.25_a);

		//O          2.76010        0.00000        1.95167
		CHECK(cif_file.atoms()[11] == "O");
		CHECK(cif_file.positions()[11][0] == -0.5_a);
		CHECK(cif_file.positions()[11][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[11][2] ==  0.25_a);

		//O          4.14015        1.38005        0.00000
		CHECK(cif_file.atoms()[12] == "O");
		CHECK(cif_file.positions()[12][0] == -0.25_a);
		CHECK(cif_file.positions()[12][1] ==  0.25_a);
		CHECK(cif_file.positions()[12][2] == Approx(0.0).margin(1e-12));

		//4.14015        1.38005        3.90335
		CHECK(cif_file.atoms()[13] == "O");
		CHECK(cif_file.positions()[13][0] == -0.25_a);
		CHECK(cif_file.positions()[13][1] ==  0.25_a);
		CHECK(cif_file.positions()[13][2] == -0.5_a);

		//O          1.38005        4.14015        3.90335
		CHECK(cif_file.atoms()[14] == "O");
		CHECK(cif_file.positions()[14][0] ==  0.25_a);
		CHECK(cif_file.positions()[14][1] == -0.25_a);
		CHECK(cif_file.positions()[14][2] == -0.5_a);

		//O          1.38005        1.38005        3.90335
		CHECK(cif_file.atoms()[15] == "O");
		CHECK(cif_file.positions()[15][0] ==  0.25_a);
		CHECK(cif_file.positions()[15][1] ==  0.25_a);
		CHECK(cif_file.positions()[15][2] == -0.5_a);

		//O          4.14015        4.14015        3.90335
		CHECK(cif_file.atoms()[16] == "O");
		CHECK(cif_file.positions()[16][0] == -0.25_a);
		CHECK(cif_file.positions()[16][1] == -0.25_a);
		CHECK(cif_file.positions()[16][2] == -0.5_a);

		//O          4.14015        4.14015        0.00000
		CHECK(cif_file.atoms()[17] == "O");
		CHECK(cif_file.positions()[17][0] == -0.25_a);
		CHECK(cif_file.positions()[17][1] == -0.25_a);
		CHECK(cif_file.positions()[17][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        1.38005        0.00000
		CHECK(cif_file.atoms()[18] == "O");
		CHECK(cif_file.positions()[18][0] ==  0.25_a);
		CHECK(cif_file.positions()[18][1] ==  0.25_a);
		CHECK(cif_file.positions()[18][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        4.14015        0.00000
		CHECK(cif_file.atoms()[19] == "O");
		CHECK(cif_file.positions()[19][0] ==  0.25_a);
		CHECK(cif_file.positions()[19][1] == -0.25_a);
		CHECK(cif_file.positions()[19][2] == Approx(0.0).margin(1e-12));
		
	}

	SECTION("Ca2PI symmetrized"){
		
		parse::cif cif_file(config::path::unit_tests_data() + "Ca2PI_symm.cif");

		CHECK(cif_file.lattice()[0][0] == 8.1469916149_a);
		CHECK(cif_file.lattice()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[0][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][0] == -4.0734958074_a);
		CHECK(cif_file.lattice()[1][1] ==  7.0555017029_a);
		CHECK(cif_file.lattice()[1][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][2] == 42.0773092856_a);		
		
		CHECK(cif_file.size() == 12);
		
		CHECK(cif_file.atoms()[0] == "Ca");
		CHECK(cif_file.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][2] == 0.22988153_a);
		
		CHECK(cif_file.atoms()[1] == "Ca");
		CHECK(cif_file.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][2] == -0.22988153_a);

		CHECK(cif_file.atoms()[2] == "Ca");
		CHECK(cif_file.positions()[2][0] ==  -0.3333333333_a);
		CHECK(cif_file.positions()[2][1] ==   0.3333333333_a);
		CHECK(cif_file.positions()[2][2] ==  -0.4367851367_a);
		
		CHECK(cif_file.atoms()[3] == "Ca");
		CHECK(cif_file.positions()[3][0] ==  -0.3333333333_a);
		CHECK(cif_file.positions()[3][1] ==   0.3333333333_a);
		CHECK(cif_file.positions()[3][2] ==   0.10345180330_a);
		
		CHECK(cif_file.atoms()[4] == "Ca");
		CHECK(cif_file.positions()[4][0] ==   0.3333333333_a);
		CHECK(cif_file.positions()[4][1] ==  -0.3333333333_a);
		CHECK(cif_file.positions()[4][2] ==  -0.1034518033_a);
		
		CHECK(cif_file.atoms()[5] == "Ca");
		CHECK(cif_file.positions()[5][0] ==   0.3333333333_a);
		CHECK(cif_file.positions()[5][1] ==  -0.3333333333_a);
		CHECK(cif_file.positions()[5][2] ==   0.4367851367_a);

		CHECK(cif_file.atoms()[6] == "P");
		CHECK(cif_file.positions()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[6][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[6][2] == -0.5_a);

		CHECK(cif_file.atoms()[7] == "P");
		CHECK(cif_file.positions()[7][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[7][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[7][2] == -0.1666666667_a);

		CHECK(cif_file.atoms()[8] == "P");
		CHECK(cif_file.positions()[8][0] ==  0.3333333333_a);
		CHECK(cif_file.positions()[8][1] == -0.3333333333_a);
		CHECK(cif_file.positions()[8][2] ==  0.1666666667_a);

		CHECK(cif_file.atoms()[9] == "I");
		CHECK(cif_file.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(cif_file.atoms()[10] == "I");
		CHECK(cif_file.positions()[10][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[10][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[10][2] ==  0.3333333333_a);

		CHECK(cif_file.atoms()[11] == "I");
		CHECK(cif_file.positions()[11][0] ==   0.3333333333_a);
		CHECK(cif_file.positions()[11][1] ==  -0.3333333333_a);
		CHECK(cif_file.positions()[11][2] ==  -0.3333333333_a);
		
	}
	
	SECTION("Ca2PI not symmetrized"){
		
		parse::cif cif_file(config::path::unit_tests_data() + "Ca2PI.cif");

		CHECK(cif_file.lattice()[0][0] == 8.1469916149_a);
		CHECK(cif_file.lattice()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[0][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][0] == -4.0734958074_a);
		CHECK(cif_file.lattice()[1][1] ==  7.0555017029_a);
		CHECK(cif_file.lattice()[1][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][2] == 42.0773092856_a);		
		
		CHECK(cif_file.size() == 12);

		//the order her is to match the symmetrized test above
		CHECK(cif_file.atoms()[1] == "Ca");
		CHECK(cif_file.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[1][2] == 0.22988153_a);

		CHECK(cif_file.atoms()[4] == "Ca");
		CHECK(cif_file.positions()[4][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[4][2] == -0.22988153_a);

		CHECK(cif_file.atoms()[3] == "Ca");
		CHECK(cif_file.positions()[3][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[3][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[3][2] == -0.4367851367_a);

		CHECK(cif_file.atoms()[0] == "Ca");
		CHECK(cif_file.positions()[0][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[0][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[0][2] ==  0.1034518033_a);

		CHECK(cif_file.atoms()[5] == "Ca");
		CHECK(cif_file.positions()[5][0] ==  0.3333333333_a);
		CHECK(cif_file.positions()[5][1] == -0.3333333333_a);
		CHECK(cif_file.positions()[5][2] == -0.1034518033_a);

		CHECK(cif_file.atoms()[2] == "Ca");
		CHECK(cif_file.positions()[2][0] ==  0.3333333333_a);
		CHECK(cif_file.positions()[2][1] == -0.3333333333_a);
		CHECK(cif_file.positions()[2][2] ==  0.4367851367_a);

		CHECK(cif_file.atoms()[7] == "P");
		CHECK(cif_file.positions()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[7][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[7][2] == -0.5_a);

		CHECK(cif_file.atoms()[8] == "P");
		CHECK(cif_file.positions()[8][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[8][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[8][2] == -0.1666666667_a);
		
		CHECK(cif_file.atoms()[6] == "P");
		CHECK(cif_file.positions()[6][0] ==  0.3333333333_a);
		CHECK(cif_file.positions()[6][1] == -0.3333333333_a);
		CHECK(cif_file.positions()[6][2] ==  0.1666666667_a);

		CHECK(cif_file.atoms()[9] == "I");
		CHECK(cif_file.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(cif_file.atoms()[10] == "I");
		CHECK(cif_file.positions()[10][0] == -0.3333333333_a);
		CHECK(cif_file.positions()[10][1] ==  0.3333333333_a);
		CHECK(cif_file.positions()[10][2] ==  0.3333333333_a);

		CHECK(cif_file.atoms()[11] == "I");
		CHECK(cif_file.positions()[11][0] ==  0.3333333333_a);
		CHECK(cif_file.positions()[11][1] == -0.3333333333_a);
		CHECK(cif_file.positions()[11][2] == -0.3333333333_a);
		
	}
	
	SECTION("Na"){
		
		parse::cif cif_file(config::path::unit_tests_data() + "Na.cif");

		//These lattice vectors match openbabel
		CHECK(cif_file.lattice()[0][0] == 17.7976863062_a);
		CHECK(cif_file.lattice()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[0][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[1][0] == 16.3923048366_a);
		CHECK(cif_file.lattice()[1][1] == 6.9318092872_a);
		CHECK(cif_file.lattice()[1][2] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.lattice()[2][0] == 16.3923048366_a);
		CHECK(cif_file.lattice()[2][1] ==  3.3234384423_a);
		CHECK(cif_file.lattice()[2][2] ==  6.0831518898_a);		
		
		CHECK(cif_file.size() == 3);
		
		CHECK(cif_file.atoms()[0] == "Na");
		CHECK(cif_file.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(cif_file.positions()[0][2] == Approx(0.0).margin(1e-12));

		CHECK(cif_file.atoms()[1] == "Na");
		CHECK(cif_file.positions()[1][0] == 0.22222_a);
		CHECK(cif_file.positions()[1][1] == 0.22222_a);
		CHECK(cif_file.positions()[1][2] == 0.22222_a);

		CHECK(cif_file.atoms()[2] == "Na");
		CHECK(cif_file.positions()[2][0] == -0.22222_a);
		CHECK(cif_file.positions()[2][1] == -0.22222_a);
		CHECK(cif_file.positions()[2][2] == -0.22222_a);
	}
	
}
#endif
