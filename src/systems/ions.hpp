/* -*- indent-tabs-mode: t -*- */

//  Copyright (C) 2019-2020 Xavier Andrade, Alfredo A. Correa

#ifndef INQ__SYSTEMS__IONS
#define INQ__SYSTEMS__IONS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <spglib.h>

#include <systems/cell.hpp>
#include <ionic/species.hpp>
#include <ionic/species_set.hpp>
#include <gpu/array.hpp>
#include <parse/cif.hpp>
#include <parse/poscar.hpp>
#include <parse/xyz.hpp>
#include <utils/load_save.hpp>
#include <utils/lowercase.hpp>

namespace inq {
namespace systems {

class ions {

public:
	
	using positions_type  =	std::vector<vector3<double>>;
	using velocities_type = std::vector<vector3<double>>;

private:

	systems::cell cell_;
	std::vector<std::string> atoms_;
	positions_type positions_;
	velocities_type velocities_;
	ionic::species_set species_list_;

	template <typename PositionType>
	void add_atom(ionic::species const & element, PositionType const & position, vector3<double> const & vel = vector3<double>(0.0, 0.0, 0.0)){
		atoms_.push_back(element.symbol());
		positions_.push_back(in_atomic_units(position));
		velocities_.push_back(vel);
		if(not species_list_.contains(element.symbol())) species_list_.insert(element);
	}

public:

	ions(inq::systems::cell arg_cell_input):
		cell_(std::move(arg_cell_input)){
	}

	void clear() {
		atoms_.clear();
		positions_.clear();
		velocities_.clear();
	}
	
	static ions parse(std::string filename, inq::systems::cell const & cell) {
		return parse(filename, std::optional<quantity<magnitude::length>>{}, cell);
	}
	
	static ions parse(std::string filename, std::optional<quantity<magnitude::length>> radius = {}, std::optional<inq::systems::cell> cell = {}) {

		using namespace inq::magnitude;

		assert(not (cell.has_value() and radius.has_value()));
		if(radius.has_value() and radius->in_atomic_units() <= 0.0) throw std::runtime_error("error: a non-positive radius was given when parsing file '" + filename + "'.");
		
		if(parse::cif::detect(filename)) {
			if(cell.has_value() or radius.has_value()) throw std::runtime_error("error: the radius or cell arguments cannot be given for parsing CIF file '" + filename + "'.");
			
			parse::cif file(filename);
			ions parsed(file.cell());
			for(int ii = 0; ii < file.size(); ii++) parsed.insert_fractional(file.atoms()[ii], file.positions()[ii]);
			return parsed;
		}

		if(parse::poscar::detect(filename)) {
			if(cell.has_value() or radius.has_value()) throw std::runtime_error("error: the radius or cell arguments cannot be given for parsing POSCAR file '" + filename + "'.");			
			
			parse::poscar file(filename);
			ions parsed(file.cell());
			for(int ii = 0; ii < file.size(); ii++) parsed.add_atom(file.atoms()[ii], file.positions()[ii], file.velocities()[ii]);
			return parsed;
		}

		if(parse::xyz::detect(filename)) {
			if(not cell.has_value() and not radius.has_value()) throw std::runtime_error("error: the radius or cell argument needs to be provided for parsing XYZ file '" + filename + "'.");

			auto file = parse::xyz(filename);

			// find the size of the containing box
			auto maxl = vector3<double>{0.0, 0.0, 0.0};
			if(radius.has_value()){
				for(int ii = 0; ii < file.size(); ii++) {
					for(int idir = 0; idir < 3; idir++) maxl[idir] = std::max(maxl[idir], fabs(file.positions()[ii][idir]) + radius->in_atomic_units());
				}

				cell = systems::cell::orthorhombic(2.0_b*maxl[0], 2.0_b*maxl[1], 2.0_b*maxl[2]).finite();
			}
	
			ions parsed(*cell);
			for(int ii = 0; ii < file.size(); ii++) parsed.add_atom(file.atoms()[ii], file.positions()[ii]);
			return parsed;
		}
		
		throw std::runtime_error("error: unsupported or unknown format for file '" + filename + "'.");
		return ions(*cell); //dummy return value to keep the compiler happy, this should never be reached
	}

#ifdef INQ_PYTHON_INTERFACE
	static auto import_ase(pybind11::object atoms){
		namespace py = pybind11;
		using namespace pybind11::literals;
		using namespace inq::magnitude;

		auto pypbc = atoms.attr("get_pbc")().cast<py::array_t<bool>>();
		auto pbc = static_cast<bool *>(pypbc.request().ptr);
			
		auto per = -1;
		if(pbc[0] == false and pbc[1] == false and pbc[2] == false) per = 0;
		if(pbc[0] == true  and pbc[1] == false and pbc[2] == false) per = 1;
		if(pbc[0] == true  and pbc[1] == true  and pbc[2] == false) per = 2;
		if(pbc[0] == true  and pbc[1] == true  and pbc[2] == true ) per = 3;
		if(per == -1) throw std::runtime_error("INQ: does not support some partial periodicities from ASE");
		
		auto lattice = atoms.attr("get_cell")().attr("__array__")().cast<py::array_t<double>>();
		auto lat = static_cast<double *>(lattice.request().ptr);
		
		auto lat0 = vector3(1.0_A*lat[0], 1.0_A*lat[1], 1.0_A*lat[2]);
		auto lat1 = vector3(1.0_A*lat[3], 1.0_A*lat[4], 1.0_A*lat[5]);
		auto lat2 = vector3(1.0_A*lat[6], 1.0_A*lat[7], 1.0_A*lat[8]);
		
		systems::ions ions(systems::cell(in_atomic_units(lat0), in_atomic_units(lat1), in_atomic_units(lat2), per));
		
		auto atomic_numbers = atoms.attr("get_atomic_numbers")().cast<py::array_t<int>>();
		auto num = static_cast<int *>(atomic_numbers.request().ptr);
		auto positions = atoms.attr("get_positions")().cast<py::array_t<double>>();
		auto pos = static_cast<double *>(positions.request().ptr);
		
		for(int ii = 0; ii < atomic_numbers.size(); ii++){
			ions.insert(num[ii], 1.0_A*vector3{pos[3*ii + 0], pos[3*ii + 1], pos[3*ii + 2]});
		}
		
		return ions;
	}
#endif
	
	auto & symbols() const {
		return atoms_;
	}

	auto & symbol(int const & iatom) const {
		return atoms_[iatom];
	}
	
	auto & species(int const & iatom) const {
		return species_list_[atoms_[iatom]];
	}
	
	auto & positions() const {
		return positions_;
	}
	
	auto & positions() {
		return positions_;
	}

	auto & velocities() const {
		return velocities_;
	}
    
	auto & velocities() {
		return velocities_;
	}

	auto & species_list() const {
		return species_list_;
	}

	auto & species_list() {
		return species_list_;
	}
	
	auto symmetry_string() const {

		assert(size() > 0);
		
		char symbol[11];
		
		std::vector<int> types(size());
		std::vector<double> lin_pos(3*size());
		
		for(int iatom = 0; iatom < size(); iatom++){
			types[iatom] = species(iatom).atomic_number();
			auto pos = cell_.to_contravariant(cell_.position_in_cell(positions()[iatom]));
			lin_pos[3*iatom + 0] = pos[0];
			lin_pos[3*iatom + 1] = pos[1];
			lin_pos[3*iatom + 2] = pos[2];
		}

		double amat[9];
		amat[0] = cell_.lattice(0)[0];
		amat[1] = cell_.lattice(0)[1];
		amat[2] = cell_.lattice(0)[2];
		amat[3] = cell_.lattice(1)[0];
		amat[4] = cell_.lattice(1)[1];
		amat[5] = cell_.lattice(1)[2];
		amat[6] = cell_.lattice(2)[0];
		amat[7] = cell_.lattice(2)[1];
		amat[8] = cell_.lattice(2)[2];
		
		auto symnum = spg_get_international(symbol, reinterpret_cast<double (*)[3]>(amat), reinterpret_cast<double (*)[3]>(lin_pos.data()), types.data(), size(), 1e-4);
		return symbol + std::string(" (number ") + std::to_string(symnum) + std::string(")");
	}
	
	auto & cell() const {
		return cell_;
	}

	void insert(ionic::species const & sp, vector3<quantity<magnitude::length>> const & pos){
		add_atom(sp, pos);
	}
	
	void insert_fractional(ionic::species const & sp, vector3<double, contravariant> const & pos){
		add_atom(sp, cell_.to_cartesian(pos));
	}

	void remove(long index) {
		if(index < 0 or index >= size()) throw std::runtime_error("Error: invalid index in systems::ions::remove");
		atoms_.erase(atoms_.begin() + index);
		positions_.erase(positions_.begin() + index);
		velocities_.erase(velocities_.begin() + index);
	}
	
	int size() const {
		return (long) positions_.size();
	}
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, ions const & self){

		auto print_vel = false;
		for(int iatom = 0; iatom < self.size(); iatom++){
			if(norm(self.velocities_[iatom]) > 1e-9) {
				print_vel = true;
				break;
			}
		}
		
		out << "Ions (" << self.size() << " total):" << std::endl;
		for(int iatom = 0; iatom < self.size(); iatom++){
			out << "  " << iatom << "\t-\t" << self.atoms_[iatom] << '\t' << self.positions_[iatom];
			if(print_vel) out << "\t\t-\t" << self.velocities_[iatom];
			out << '\n';
		}
		out << std::endl;
		return out;
	}

	auto kinetic_energy() {
		auto energy = 0.0;
		for(int iatom = 0; iatom < size(); iatom++){
			energy += 0.5*species(iatom).mass()*norm(velocities()[iatom]);
		}
		return energy;
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the ions to directory '" + dirname + "'.";

		cell_.save(comm, dirname + "/cell");
		species_list_.save(comm, dirname + "/species_list");
		utils::save_value(comm, dirname + "/num_ions",   size(),      error_message);
		utils::save_container(comm, dirname + "/atoms",      atoms_,      error_message);
		utils::save_container(comm, dirname + "/positions",  positions_,  error_message);
		utils::save_container(comm, dirname + "/velocities", velocities_, error_message);
	}
	
	static auto load(std::string const & dirname) {
		auto cell = systems::cell::load(dirname + "/cell");

		auto error_message = "INQ error: Cannot load the ions from directory '" + dirname + "'.";

		auto read_ions = ions(cell);
		read_ions.species_list() = ionic::species_set::load(dirname + "/species_list");
		
		int num;
		utils::load_value(dirname + "/num_ions", num, error_message);

		if(num == 0) return read_ions;
		
		auto atoms_file = std::ifstream(dirname + "/atoms");
		if(not atoms_file) throw std::runtime_error(error_message);

		auto positions_file = std::ifstream(dirname + "/positions");
		if(not positions_file) throw std::runtime_error(error_message);

		auto velocities_file = std::ifstream(dirname + "/velocities");
		if(not velocities_file) throw std::runtime_error(error_message);

		for(int iatom = 0; iatom < num; iatom++){
			std::string symbol;
			vector3<double> pos, vel;
			atoms_file >> symbol;
			positions_file >> pos;
			velocities_file >> vel;
			
			read_ions.add_atom(symbol, pos, vel);
		}
		
		return read_ions;
	}
	
};

}
}
#endif

#ifdef INQ_SYSTEMS_IONS_UNIT_TEST
#undef INQ_SYSTEMS_IONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	SECTION("Create empty and add an atom"){
		
		auto dcc = 1.42_A;
		auto aa = sqrt(3)*dcc;
		auto lz = 10.0_b;
		systems::ions ions(systems::cell::lattice(aa*vector3{1.0, 0.0, 0.0}, aa*vector3{-1.0/2.0, sqrt(3.0)/2.0, 0.0}, {0.0_b, 0.0_b, lz}).periodicity(2));
		
		CHECK(ions.cell().periodicity() == 2);
		
    CHECK(ions.size() == 0);

    ions.insert("Xe", {1000.0_b, -200.0_b, 6.0_b});

    CHECK(ions.size() == 1);
    CHECK(ions.species(0).atomic_number() == 54);
    CHECK(ions.species(0) == ionic::species(54));
    CHECK(ions.species(0).charge() == -54.0_a);
    CHECK(ions.species(0).mass() == 239333.5935636_a);
    CHECK(ions.positions()[0][0] == 1000.0_a);
    CHECK(ions.positions()[0][1] == -200.0_a);
    CHECK(ions.positions()[0][2] == 6.0_a);
		CHECK(ions.velocities()[0][0] == 0.0_a);
    CHECK(ions.velocities()[0][1] == 0.0_a);
    CHECK(ions.velocities()[0][2] == 0.0_a);
		CHECK(ions.species_list().size() == 1);
		
    ions.positions()[0][0] += 8;
    
    CHECK(ions.positions()[0][0] == 1008.0_a);
		CHECK(ions.velocities().size() == ions.positions().size());
		
  }
	
	SECTION("Read an xyz file with cell"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "benzene.xyz", systems::cell::cubic(66.6_A).finite());

		CHECK(ions.cell().lattice(0)[0] == 125.8557599003_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 125.8557599003_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 125.8557599003_a);
		CHECK(ions.cell().periodicity() == 0);
		
    CHECK(ions.size() == 12);
		CHECK(ions.species_list().size() == 2);
		
    CHECK(ions.species(2) == "C");
    CHECK(ions.species(2).charge() == -6.0_a);
    CHECK(ions.species(2).mass() == 21892.1617296_a);
    CHECK(ions.positions()[2][0] == 2.2846788549_a);
    CHECK(ions.positions()[2][1] == -1.3190288178_a);
    CHECK(ions.positions()[2][2] == 0.0_a);

    CHECK(ions.species(11) == "H");
    CHECK(ions.species(11).charge() == -1.0_a);
    CHECK(ions.species(11).mass() == 1837.17994584_a);
    CHECK(ions.positions()[11][0] == -4.0572419367_a);
    CHECK(ions.positions()[11][1] == 2.343260364_a);
    CHECK(ions.positions()[11][2] == 0.0_a);
		CHECK(ions.velocities()[11][0] == 0.0_a);
    CHECK(ions.velocities()[11][1] == 0.0_a);
    CHECK(ions.velocities()[11][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.positions().size());
		
    ions.insert("Cl", {-3.0_b, 4.0_b, 5.0_b});

    CHECK(ions.size() == 13);
		CHECK(ions.species_list().size() == 3);
    CHECK(ions.species(12).atomic_number() == 17);
    CHECK(ions.species(12) == ionic::species(17));
    CHECK(ions.species(12).charge() == -17.0_a);
    CHECK(ions.species(12).mass() == 64614.105771_a);
    CHECK(ions.positions()[12][0] == -3.0_a);
    CHECK(ions.positions()[12][1] == 4.0_a);
    CHECK(ions.positions()[12][2] == 5.0_a);
		CHECK(ions.velocities()[12][0] == 0.0_a);
    CHECK(ions.velocities()[12][1] == 0.0_a);
    CHECK(ions.velocities()[12][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.positions().size());

		ions.save(comm, "ions_save_benzene");
		auto read_ions = systems::ions::load("ions_save_benzene");

		CHECK(read_ions.cell().lattice(0)[0] == 125.8557599003_a);
		CHECK(read_ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(1)[1] == 125.8557599003_a);
		CHECK(read_ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(read_ions.cell().lattice(2)[2] == 125.8557599003_a);
		CHECK(read_ions.cell().periodicity() == 0);

    CHECK(read_ions.size() == 13);
		CHECK(read_ions.species_list().size() == 3);
    CHECK(read_ions.species(2) == "C");
    CHECK(read_ions.species(2).charge() == -6.0_a);
    CHECK(read_ions.species(2).mass() == 21892.1617296_a);
    CHECK(read_ions.positions()[2][0] == 2.2846788549_a);
    CHECK(read_ions.positions()[2][1] == -1.3190288178_a);
    CHECK(read_ions.positions()[2][2] == 0.0_a);

    CHECK(read_ions.species(11) == "H");
    CHECK(read_ions.species(11).charge() == -1.0_a);
    CHECK(read_ions.species(11).mass() == 1837.17994584_a);
    CHECK(read_ions.positions()[11][0] == -4.0572419367_a);
    CHECK(read_ions.positions()[11][1] == 2.343260364_a);
    CHECK(read_ions.positions()[11][2] == 0.0_a);
		CHECK(read_ions.velocities()[11][0] == 0.0_a);
    CHECK(read_ions.velocities()[11][1] == 0.0_a);
    CHECK(read_ions.velocities()[11][2] == 0.0_a);

    CHECK(read_ions.species(12).atomic_number() == 17);
    CHECK(read_ions.species(12) == ionic::species(17));
    CHECK(read_ions.species(12).charge() == -17.0_a);
    CHECK(read_ions.species(12).mass() == 64614.105771_a);
    CHECK(read_ions.positions()[12][0] == -3.0_a);
    CHECK(read_ions.positions()[12][1] == 4.0_a);
    CHECK(read_ions.positions()[12][2] == 5.0_a);
		CHECK(read_ions.velocities()[12][0] == 0.0_a);
    CHECK(read_ions.velocities()[12][1] == 0.0_a);
    CHECK(read_ions.velocities()[12][2] == 0.0_a);

		CHECK(read_ions.velocities().size() == read_ions.positions().size());

		read_ions.clear();

		CHECK(read_ions.size() == 0);
		
  }
	
	SECTION("Read an xyz file with radius"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "benzene.xyz", /* radius = */ 5.0_b);

		CHECK(ions.cell().lattice(0)[0] == 18.1144839792_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 19.3692621259_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 10.0_a);
		CHECK(ions.cell().periodicity() == 0);
		
    CHECK(ions.size() == 12);
    
    CHECK(ions.species(2) == "C");
    CHECK(ions.species(2).charge() == -6.0_a);
    CHECK(ions.species(2).mass() == 21892.1617296_a);
    CHECK(ions.positions()[2][0] == 2.2846788549_a);
    CHECK(ions.positions()[2][1] == -1.3190288178_a);
    CHECK(ions.positions()[2][2] == 0.0_a);

    CHECK(ions.species(11) == "H");
    CHECK(ions.species(11).charge() == -1.0_a);
    CHECK(ions.species(11).mass() == 1837.17994584_a);
    CHECK(ions.positions()[11][0] == -4.0572419367_a);
    CHECK(ions.positions()[11][1] == 2.343260364_a);
    CHECK(ions.positions()[11][2] == 0.0_a);
		CHECK(ions.velocities()[11][0] == 0.0_a);
    CHECK(ions.velocities()[11][1] == 0.0_a);
    CHECK(ions.velocities()[11][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.positions().size());
		
    ions.insert("Cl", {-3.0_b, 4.0_b, 5.0_b});

    CHECK(ions.size() == 13);
    CHECK(ions.species(12).atomic_number() == 17);
    CHECK(ions.species(12) == ionic::species(17));
    CHECK(ions.species(12).charge() == -17.0_a);
    CHECK(ions.species(12).mass() == 64614.105771_a);
    CHECK(ions.positions()[12][0] == -3.0_a);
    CHECK(ions.positions()[12][1] == 4.0_a);
    CHECK(ions.positions()[12][2] == 5.0_a);
		CHECK(ions.velocities()[12][0] == 0.0_a);
    CHECK(ions.velocities()[12][1] == 0.0_a);
    CHECK(ions.velocities()[12][2] == 0.0_a);

		CHECK(ions.velocities().size() == ions.positions().size());
		
  }
	
	SECTION("CIF - Al"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Al.cif");

		CHECK(ions.cell().lattice(0)[0] == 7.634890386_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 7.634890386_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 7.634890386_a);		
		
		CHECK(ions.size() == 4);

		CHECK(ions.species(0) == "Al");
		CHECK(ions.species(1) == "Al");
		CHECK(ions.species(2) == "Al");		
		CHECK(ions.species(3) == "Al");

		CHECK(ions.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][2] == Approx(0.0).margin(1e-12));
		
		CHECK(ions.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][1] == -3.817445193_a);
		CHECK(ions.positions()[1][2] == -3.817445193_a);

		CHECK(ions.positions()[2][0] == -3.817445193_a);
		CHECK(ions.positions()[2][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[2][2] == -3.817445193_a);

		CHECK(ions.positions()[3][0] == -3.817445193_a);
		CHECK(ions.positions()[3][1] == -3.817445193_a);
		CHECK(ions.positions()[3][2] == Approx(0.0).margin(1e-12));
	}
	
	SECTION("CIF - SrTiO3"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "9002806.cif");

		CHECK(ions.cell().lattice(0)[0] == 10.4316661532_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[1] == 10.4316661532_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 14.7525249371_a);		
		
		CHECK(ions.size() == 20);

		//Conversions for units and cell convention to compare with obabel
		//generated positions
		//
		//  1.38005 ->  2.6079165383
		//  1.95168 ->  3.6881312343
		//  2.76010 -> -5.2158330766
		//  3.90335 -> -7.3762624686
		//  4.14015 -> -2.6079165383
		//  5.85503 -> -3.6881312343
		
		//Sr         0.00000        0.00000        1.95168
		CHECK(ions.species(0) == "Sr");
		CHECK(ions.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][2] == 3.6881312343_a);

		//Sr         0.00000        0.00000        5.85503
		CHECK(ions.species(1) == "Sr");
		CHECK(ions.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][2] == -3.6881312343_a);

		//Sr         2.76010        2.76010        5.85503
		CHECK(ions.species(2) == "Sr");
		CHECK(ions.positions()[2][0] == -5.2158330766_a);
		CHECK(ions.positions()[2][1] == -5.2158330766_a);
		CHECK(ions.positions()[2][2] == -3.6881312343_a);

		//Sr         2.76010        2.76010        1.95167
		CHECK(ions.species(3) == "Sr");
		CHECK(ions.positions()[3][0] == -5.2158330766_a);
		CHECK(ions.positions()[3][1] == -5.2158330766_a);
		CHECK(ions.positions()[3][2] ==  3.6881312343_a);

		//Ti         2.76010        0.00000        0.00000
		CHECK(ions.species(4) == "Ti");
		CHECK(ions.positions()[4][0] == -5.2158330766_a);
		CHECK(ions.positions()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[4][2] == Approx(0.0).margin(1e-12));

		//Ti         2.76010        0.00000        3.90335
		CHECK(ions.species(5) == "Ti");
		CHECK(ions.positions()[5][0] == -5.2158330766_a);
		CHECK(ions.positions()[5][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[5][2] == -7.3762624686_a);

		//Ti         0.00000        2.76010        3.90335
		CHECK(ions.species(6) == "Ti");
		CHECK(ions.positions()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[6][1] == -5.2158330766_a);
		CHECK(ions.positions()[6][2] == -7.3762624686_a);

		//Ti         0.00000        2.76010        0.00000
		CHECK(ions.species(7) == "Ti");
		CHECK(ions.positions()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[7][1] == -5.2158330766_a);
		CHECK(ions.positions()[7][2] == Approx(0.0).margin(1e-12));

		//O          0.00000        2.76010        1.95168
		CHECK(ions.species(8) == "O");
		CHECK(ions.positions()[8][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[8][1] == -5.2158330766_a);
		CHECK(ions.positions()[8][2] == 3.6881312343_a);

		//O          0.00000        2.76010        5.85503
		CHECK(ions.species(9) == "O");
		CHECK(ions.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[9][1] == -5.2158330766_a);
		CHECK(ions.positions()[9][2] == -3.6881312343_a);

		//O          2.76010        0.00000        5.85503
		CHECK(ions.species(10) == "O");
		CHECK(ions.positions()[10][0] == -5.2158330766_a);
		CHECK(ions.positions()[10][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[10][2] == -3.6881312343_a);

		//O          2.76010        0.00000        1.95167
		CHECK(ions.species(11) == "O");
		CHECK(ions.positions()[11][0] == -5.2158330766_a);
		CHECK(ions.positions()[11][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[11][2] ==  3.6881312343_a);

		//O          4.14015        1.38005        0.00000
		CHECK(ions.species(12) == "O");
		CHECK(ions.positions()[12][0] == -2.6079165383_a);
		CHECK(ions.positions()[12][1] ==  2.6079165383_a);
		CHECK(ions.positions()[12][2] == Approx(0.0).margin(1e-12));

		//4.14015        1.38005        3.90335
		CHECK(ions.species(13) == "O");
		CHECK(ions.positions()[13][0] == -2.6079165383_a);
		CHECK(ions.positions()[13][1] ==  2.6079165383_a);
		CHECK(ions.positions()[13][2] == -7.3762624686_a);

		//O          1.38005        4.14015        3.90335
		CHECK(ions.species(14) == "O");
		CHECK(ions.positions()[14][0] ==  2.6079165383_a);
		CHECK(ions.positions()[14][1] == -2.6079165383_a);
		CHECK(ions.positions()[14][2] == -7.3762624686_a);

		//O          1.38005        1.38005        3.90335
		CHECK(ions.species(15) == "O");
		CHECK(ions.positions()[15][0] ==  2.6079165383_a);
		CHECK(ions.positions()[15][1] ==  2.6079165383_a);
		CHECK(ions.positions()[15][2] == -7.3762624686_a);

		//O          4.14015        4.14015        3.90335
		CHECK(ions.species(16) == "O");
		CHECK(ions.positions()[16][0] == -2.6079165383_a);
		CHECK(ions.positions()[16][1] == -2.6079165383_a);
		CHECK(ions.positions()[16][2] == -7.3762624686_a);

		//O          4.14015        4.14015        0.00000
		CHECK(ions.species(17) == "O");
		CHECK(ions.positions()[17][0] == -2.6079165383_a);
		CHECK(ions.positions()[17][1] == -2.6079165383_a);
		CHECK(ions.positions()[17][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        1.38005        0.00000
		CHECK(ions.species(18) == "O");
		CHECK(ions.positions()[18][0] ==  2.6079165383_a);
		CHECK(ions.positions()[18][1] ==  2.6079165383_a);
		CHECK(ions.positions()[18][2] == Approx(0.0).margin(1e-12));

		//O          1.38005        4.14015        0.00000
		CHECK(ions.species(19) == "O");
		CHECK(ions.positions()[19][0] ==  2.6079165383_a);
		CHECK(ions.positions()[19][1] == -2.6079165383_a);
		CHECK(ions.positions()[19][2] == Approx(0.0).margin(1e-12));
		
	}

	SECTION("CIF - Ca2PI symmetrized"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Ca2PI_symm.cif");

		CHECK(ions.cell().lattice(0)[0] == 8.1469916149_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == -4.0734958074_a);
		CHECK(ions.cell().lattice(1)[1] ==  7.0555017029_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 42.0773092856_a);		
		
		CHECK(ions.size() == 12);
		
		CHECK(ions.species(0) == "Ca");
		CHECK(ions.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][2] == 9.6727962369_a);
		
		CHECK(ions.species(1) == "Ca");
		CHECK(ions.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][2] == -9.6727962369_a);

		CHECK(ions.species(2) == "Ca");
		CHECK(ions.positions()[2][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[2][1] ==   2.3518339010_a);
		CHECK(ions.positions()[2][2] == -18.3787432869_a);
		
		CHECK(ions.species(3) == "Ca");
		CHECK(ions.positions()[3][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[3][1] ==   2.3518339010_a);
		CHECK(ions.positions()[3][2] ==   4.3529735250_a);
		
		CHECK(ions.species(4) == "Ca");
		CHECK(ions.positions()[4][0] ==   4.0734958074_a);
		CHECK(ions.positions()[4][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[4][2] ==  -4.3529735250_a);
		
		CHECK(ions.species(5) == "Ca");
		CHECK(ions.positions()[5][0] ==   4.0734958074_a);
		CHECK(ions.positions()[5][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[5][2] ==  18.3787432869_a);

		CHECK(ions.species(6) == "P");
		CHECK(ions.positions()[6][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[6][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[6][2] == -21.0386546428_a);

		CHECK(ions.species(7) == "P");
		CHECK(ions.positions()[7][0] == -4.0734958074_a);
		CHECK(ions.positions()[7][1] ==  2.3518339010_a);
		CHECK(ions.positions()[7][2] == -7.0128848809_a);

		CHECK(ions.species(8) == "P");
		CHECK(ions.positions()[8][0] ==  4.0734958074_a);
		CHECK(ions.positions()[8][1] == -2.3518339010_a);
		CHECK(ions.positions()[8][2] ==  7.0128848809_a);

		CHECK(ions.species(9) == "I");
		CHECK(ions.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.species(10) == "I");
		CHECK(ions.positions()[10][0] == -4.0734958074_a);
		CHECK(ions.positions()[10][1] ==  2.3518339010_a);
		CHECK(ions.positions()[10][2] == 14.0257697619_a);

		CHECK(ions.species(11) == "I");
		CHECK(ions.positions()[11][0] ==   4.0734958074_a);
		CHECK(ions.positions()[11][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[11][2] == -14.0257697619_a);

		ions.remove(1);
		ions.remove(6);

		CHECK(ions.size() == 10);
				
		CHECK(ions.species(0) == "Ca");
		CHECK(ions.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][2] == 9.6727962369_a);
		
		CHECK(ions.species(1) == "Ca");
		CHECK(ions.positions()[1][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[1][1] ==   2.3518339010_a);
		CHECK(ions.positions()[1][2] == -18.3787432869_a);
		
		CHECK(ions.species(2) == "Ca");
		CHECK(ions.positions()[2][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[2][1] ==   2.3518339010_a);
		CHECK(ions.positions()[2][2] ==   4.3529735250_a);
		
		CHECK(ions.species(3) == "Ca");
		CHECK(ions.positions()[3][0] ==   4.0734958074_a);
		CHECK(ions.positions()[3][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[3][2] ==  -4.3529735250_a);
		
		CHECK(ions.species(4) == "Ca");
		CHECK(ions.positions()[4][0] ==   4.0734958074_a);
		CHECK(ions.positions()[4][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[4][2] ==  18.3787432869_a);

		CHECK(ions.species(5) == "P");
		CHECK(ions.positions()[5][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[5][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[5][2] == -21.0386546428_a);

		CHECK(ions.species(6) == "P");
		CHECK(ions.positions()[6][0] ==  4.0734958074_a);
		CHECK(ions.positions()[6][1] == -2.3518339010_a);
		CHECK(ions.positions()[6][2] ==  7.0128848809_a);

		CHECK(ions.species(7) == "I");
		CHECK(ions.positions()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[7][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[7][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.species(8) == "I");
		CHECK(ions.positions()[8][0] == -4.0734958074_a);
		CHECK(ions.positions()[8][1] ==  2.3518339010_a);
		CHECK(ions.positions()[8][2] == 14.0257697619_a);

		CHECK(ions.species(9) == "I");
		CHECK(ions.positions()[9][0] ==   4.0734958074_a);
		CHECK(ions.positions()[9][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[9][2] == -14.0257697619_a);
	}
	
	SECTION("CIF - Ca2PI not symmetrized"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Ca2PI.cif");

		CHECK(ions.cell().lattice(0)[0] == 8.1469916149_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == -4.0734958074_a);
		CHECK(ions.cell().lattice(1)[1] ==  7.0555017029_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[2] == 42.0773092856_a);		
		
		CHECK(ions.size() == 12);

		//the order here is to match the symmetrized test above
		CHECK(ions.species(1) == "Ca");
		CHECK(ions.positions()[1][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[1][2] == 9.6727962369_a);

		CHECK(ions.species(4) == "Ca");
		CHECK(ions.positions()[4][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[4][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[4][2] == -9.6727962369_a);

		CHECK(ions.species(3) == "Ca");
		CHECK(ions.positions()[3][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[3][1] ==   2.3518339010_a);
		CHECK(ions.positions()[3][2] == -18.3787432869_a);

		CHECK(ions.species(0) == "Ca");
		CHECK(ions.positions()[0][0] ==  -4.0734958074_a);
		CHECK(ions.positions()[0][1] ==   2.3518339010_a);
		CHECK(ions.positions()[0][2] ==   4.3529735250_a);

		CHECK(ions.species(5) == "Ca");
		CHECK(ions.positions()[5][0] ==   4.0734958074_a);
		CHECK(ions.positions()[5][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[5][2] ==  -4.3529735250_a);

		CHECK(ions.species(2) == "Ca");
		CHECK(ions.positions()[2][0] ==   4.0734958074_a);
		CHECK(ions.positions()[2][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[2][2] ==  18.3787432869_a);

		CHECK(ions.species(7) == "P");
		CHECK(ions.positions()[7][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[7][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[7][2] == -21.0386546428_a);

		CHECK(ions.species(8) == "P");
		CHECK(ions.positions()[8][0] == -4.0734958074_a);
		CHECK(ions.positions()[8][1] ==  2.3518339010_a);
		CHECK(ions.positions()[8][2] == -7.0128848809_a);
		
		CHECK(ions.species(6) == "P");
		CHECK(ions.positions()[6][0] ==  4.0734958074_a);
		CHECK(ions.positions()[6][1] == -2.3518339010_a);
		CHECK(ions.positions()[6][2] ==  7.0128848809_a);

		CHECK(ions.species(9) == "I");
		CHECK(ions.positions()[9][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[9][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[9][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.species(10) == "I");
		CHECK(ions.positions()[10][0] == -4.0734958074_a);
		CHECK(ions.positions()[10][1] ==  2.3518339010_a);
		CHECK(ions.positions()[10][2] == 14.0257697619_a);

		CHECK(ions.species(11) == "I");
		CHECK(ions.positions()[11][0] ==   4.0734958074_a);
		CHECK(ions.positions()[11][1] ==  -2.3518339010_a);
		CHECK(ions.positions()[11][2] == -14.0257697619_a);

	}
	
	SECTION("CIF - Na"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "Na.cif");

		//These lattice vectors match openbabel
		CHECK(ions.cell().lattice(0)[0] == 17.7976863062_a);
		CHECK(ions.cell().lattice(0)[1] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(0)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(1)[0] == 16.3923048366_a);
		CHECK(ions.cell().lattice(1)[1] == 6.9318092872_a);
		CHECK(ions.cell().lattice(1)[2] == Approx(0.0).margin(1e-12));
		CHECK(ions.cell().lattice(2)[0] == 16.3923048366_a);
		CHECK(ions.cell().lattice(2)[1] ==  3.3234384423_a);
		CHECK(ions.cell().lattice(2)[2] ==  6.0831518898_a);		
		
		CHECK(ions.size() == 3);
		
		CHECK(ions.species(0) == "Na");
		CHECK(ions.positions()[0][0] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][1] == Approx(0.0).margin(1e-12));
		CHECK(ions.positions()[0][2] == Approx(0.0).margin(1e-12));

		CHECK(ions.species(1) == "Na");
		CHECK(ions.positions()[1][0] == 11.2403978126_a);
		CHECK(ions.positions()[1][1] ==  2.2789211505_a);
		CHECK(ions.positions()[1][2] ==  1.3517980130_a);

		CHECK(ions.species(2) == "Na");
		CHECK(ions.positions()[2][0] == -11.2403978126_a);
		CHECK(ions.positions()[2][1] ==  -2.2789211505_a);
		CHECK(ions.positions()[2][2] ==  -1.3517980130_a);
	}
	
	SECTION("POSCAR - BN"){
	
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "bn.poscar");
		
		CHECK(ions.cell().lattice(0)[0] == 0.0_a);
		CHECK(ions.cell().lattice(0)[1] == 3.3731611325_a);
		CHECK(ions.cell().lattice(0)[2] == 3.3731611325_a);
		CHECK(ions.cell().lattice(1)[0] == 3.3731611325_a);
		CHECK(ions.cell().lattice(1)[1] == 0.0_a);
		CHECK(ions.cell().lattice(1)[2] == 3.3731611325_a);
		CHECK(ions.cell().lattice(2)[0] == 3.3731611325_a);
		CHECK(ions.cell().lattice(2)[1] == 3.3731611325_a);
		CHECK(ions.cell().lattice(2)[2] == 0.0_a);		
		
		CHECK(ions.size() == 2);

		CHECK(ions.species(0) == "B");
		CHECK(ions.species(1) == "N");

		CHECK(ions.positions()[0][0] == 0.0_a);
		CHECK(ions.positions()[0][1] == 0.0_a);
		CHECK(ions.positions()[0][2] == 0.0_a);
		CHECK(ions.positions()[1][0] == 1.6865805662_a);
		CHECK(ions.positions()[1][1] == 1.6865805662_a);
		CHECK(ions.positions()[1][2] == 1.6865805662_a);
	}

	SECTION("POSCAR - Al"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "al.poscar");
		
		CHECK(ions.cell().lattice(0)[0] == 7.6458319003_a);
		CHECK(ions.cell().lattice(0)[1] == 0.0_a);
		CHECK(ions.cell().lattice(0)[2] == 0.0_a);
		CHECK(ions.cell().lattice(1)[0] == 0.0_a);
		CHECK(ions.cell().lattice(1)[1] == 7.6458319003_a);
		CHECK(ions.cell().lattice(1)[2] == 0.0_a);
		CHECK(ions.cell().lattice(2)[0] == 0.0_a);
		CHECK(ions.cell().lattice(2)[1] == 0.0_a);
		CHECK(ions.cell().lattice(2)[2] == 7.6458319003_a);		
		
		CHECK(ions.size() == 4);

		CHECK(ions.species(0) == "Al");
		CHECK(ions.species(1) == "Al");
		CHECK(ions.species(2) == "Al");		
		CHECK(ions.species(3) == "Al");

		CHECK(ions.positions()[0][0] == 0.0_a);
		CHECK(ions.positions()[0][1] == 0.0_a);
		CHECK(ions.positions()[0][2] == 0.0_a);
		CHECK(ions.positions()[1][0] == 3.8229159501_a);
		CHECK(ions.positions()[1][1] == 3.8229159501_a);
		CHECK(ions.positions()[1][2] == 0.0_a);
		CHECK(ions.positions()[2][0] == 0.0_a);
		CHECK(ions.positions()[2][1] == 3.8229159501_a);
		CHECK(ions.positions()[2][2] == 3.8229159501_a);
		CHECK(ions.positions()[3][0] == 3.8229159501_a);
		CHECK(ions.positions()[3][1] == 0.0_a);
		CHECK(ions.positions()[3][2] == 3.8229159501_a);

	}

	SECTION("POSCAR - Ni"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "POSCAR");
		ions.velocities()[0] = vector3<double>{ 1.0,  2.0,  3.0};
		ions.velocities()[1] = vector3<double>{ 0.1,  5.5, -0.8};
		ions.velocities()[2] = vector3<double>{-1.0, -2.0, -3.0};
		ions.velocities()[3] = vector3<double>{-7.9,  0.6,  3.4};
		ions.velocities()[4] = vector3<double>{ 9.9,  2.6,  1.7};		
		
		CHECK(ions.cell().lattice(0)[0] == 3.33536661_a);
		CHECK(ions.cell().lattice(0)[1] == 3.33536661_a);
		CHECK(ions.cell().lattice(0)[2] == 0.0_a);
		CHECK(ions.cell().lattice(1)[0] == -3.33536661_a);
		CHECK(ions.cell().lattice(1)[1] == 3.33536661_a);
		CHECK(ions.cell().lattice(1)[2] == 0.0_a);
		CHECK(ions.cell().lattice(2)[0] == 0.0_a);
		CHECK(ions.cell().lattice(2)[1] == 0.0_a);
		CHECK(ions.cell().lattice(2)[2] == 33.3536660997_a);		
		
		CHECK(ions.size() == 5);

		CHECK(ions.species(0) == "Ni");
		CHECK(ions.species(1) == "Ni");
		CHECK(ions.species(2) == "Ni");
		CHECK(ions.species(3) == "Ni");		
		CHECK(ions.species(4) == "Ni");

		CHECK(ions.positions()[0][0] == 0.0_a);
		CHECK(ions.positions()[0][1] == 0.0_a);
		CHECK(ions.positions()[0][2] == 0.0_a);

		CHECK(ions.positions()[1][0] == 0.0_a);
		CHECK(ions.positions()[1][1] == 3.33536661_a);
		CHECK(ions.positions()[1][2] == 3.33536661_a);

		CHECK(ions.positions()[2][0] == 0.0_a);
		CHECK(ions.positions()[2][1] == 0.0_a);
		CHECK(ions.positions()[2][2] == 6.6707332199_a);

		CHECK(ions.positions()[3][0] == 0.0_a);
		CHECK(ions.positions()[3][1] == 3.33536661_a);
		CHECK(ions.positions()[3][2] == 10.0060998299_a);

		CHECK(ions.positions()[4][0] == 0.0_a);
		CHECK(ions.positions()[4][1] == 0.0_a);
		CHECK(ions.positions()[4][2] == 13.3414664399_a);
		
		CHECK(ions.velocities()[0][0] ==  1.0_a);
		CHECK(ions.velocities()[0][1] ==  2.0_a);
		CHECK(ions.velocities()[0][2] ==  3.0_a);

		CHECK(ions.velocities()[1][0] ==  0.1_a);
		CHECK(ions.velocities()[1][1] ==  5.5_a);
		CHECK(ions.velocities()[1][2] == -0.8_a);

		CHECK(ions.velocities()[2][0] ==  -1.0_a);
		CHECK(ions.velocities()[2][1] ==  -2.0_a);
		CHECK(ions.velocities()[2][2] ==  -3.0_a);

		CHECK(ions.velocities()[3][0] ==  -7.9_a);
		CHECK(ions.velocities()[3][1] ==   0.6_a);
		CHECK(ions.velocities()[3][2] ==   3.4_a);

		CHECK(ions.velocities()[4][0] ==  9.9_a);
		CHECK(ions.velocities()[4][1] ==  2.6_a);
		CHECK(ions.velocities()[4][2] ==  1.7_a);

		ions.save(comm, "ions_save_ni");
		auto read_ions = systems::ions::load("ions_save_ni");

		CHECK(read_ions.cell().lattice(0)[0] == 3.33536661_a);
		CHECK(read_ions.cell().lattice(0)[1] == 3.33536661_a);
		CHECK(read_ions.cell().lattice(0)[2] == 0.0_a);
		CHECK(read_ions.cell().lattice(1)[0] == -3.33536661_a);
		CHECK(read_ions.cell().lattice(1)[1] == 3.33536661_a);
		CHECK(read_ions.cell().lattice(1)[2] == 0.0_a);
		CHECK(read_ions.cell().lattice(2)[0] == 0.0_a);
		CHECK(read_ions.cell().lattice(2)[1] == 0.0_a);
		CHECK(read_ions.cell().lattice(2)[2] == 33.3536660997_a);

		CHECK(read_ions.positions()[0][0] == 0.0_a);
		CHECK(read_ions.positions()[0][1] == 0.0_a);
		CHECK(read_ions.positions()[0][2] == 0.0_a);

		CHECK(read_ions.positions()[1][0] == 0.0_a);
		CHECK(read_ions.positions()[1][1] == 3.33536661_a);
		CHECK(read_ions.positions()[1][2] == 3.33536661_a);

		CHECK(read_ions.positions()[2][0] == 0.0_a);
		CHECK(read_ions.positions()[2][1] == 0.0_a);
		CHECK(read_ions.positions()[2][2] == 6.6707332199_a);

		CHECK(read_ions.positions()[3][0] == 0.0_a);
		CHECK(read_ions.positions()[3][1] == 3.33536661_a);
		CHECK(read_ions.positions()[3][2] == 10.0060998299_a);

		CHECK(read_ions.positions()[4][0] == 0.0_a);
		CHECK(read_ions.positions()[4][1] == 0.0_a);
		CHECK(read_ions.positions()[4][2] == 13.3414664399_a);

		CHECK(read_ions.velocities()[0][0] ==  1.0_a);
		CHECK(read_ions.velocities()[0][1] ==  2.0_a);
		CHECK(read_ions.velocities()[0][2] ==  3.0_a);

		CHECK(read_ions.velocities()[1][0] ==  0.1_a);
		CHECK(read_ions.velocities()[1][1] ==  5.5_a);
		CHECK(read_ions.velocities()[1][2] == -0.8_a);

		CHECK(read_ions.velocities()[2][0] ==  -1.0_a);
		CHECK(read_ions.velocities()[2][1] ==  -2.0_a);
		CHECK(read_ions.velocities()[2][2] ==  -3.0_a);

		CHECK(read_ions.velocities()[3][0] ==  -7.9_a);
		CHECK(read_ions.velocities()[3][1] ==   0.6_a);
		CHECK(read_ions.velocities()[3][2] ==   3.4_a);

		CHECK(read_ions.velocities()[4][0] ==  9.9_a);
		CHECK(read_ions.velocities()[4][1] ==  2.6_a);
		CHECK(read_ions.velocities()[4][2] ==  1.7_a);
		
	}

	SECTION("POSCAR - CO 64"){
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "co.POSCAR");

		CHECK(ions.cell().lattice(0)[0] == 22.753846_a);
		CHECK(ions.cell().lattice(0)[1] == 0.0_a);
		CHECK(ions.cell().lattice(0)[2] == 0.0_a);
		CHECK(ions.cell().lattice(1)[0] == 0.0_a);
		CHECK(ions.cell().lattice(1)[1] == 22.753846_a);
		CHECK(ions.cell().lattice(1)[2] == 0.0_a);
		CHECK(ions.cell().lattice(2)[0] == 0.0_a);
		CHECK(ions.cell().lattice(2)[1] == 0.0_a);
		CHECK(ions.cell().lattice(2)[2] == 22.753846_a);
		
		CHECK(ions.size() == 128);

		CHECK(ions.species(0)   == "C");
		CHECK(ions.species(10)  == "C");
		CHECK(ions.species(20)  == "C");
		CHECK(ions.species(30)  == "C");
		CHECK(ions.species(40)  == "C");
		CHECK(ions.species(50)  == "C");
		CHECK(ions.species(60)  == "C");
		CHECK(ions.species(63)  == "C");
		CHECK(ions.species(64)  == "O");
		CHECK(ions.species(70)  == "O");
		CHECK(ions.species(80)  == "O");
		CHECK(ions.species(90)  == "O");
		CHECK(ions.species(100) == "O");
		CHECK(ions.species(110) == "O");
		CHECK(ions.species(120) == "O");
		CHECK(ions.species(127) == "O");
		
		CHECK(ions.positions()[1][0] == 17.2379607321_a);
		CHECK(ions.positions()[1][1] == 13.7235682342_a);
		CHECK(ions.positions()[1][2] == 1.4562639292_a);

		CHECK(ions.velocities()[1][0] ==  0.00044201619_a);
		CHECK(ions.velocities()[1][1] ==  0.00014473032_a);
		CHECK(ions.velocities()[1][2] == -0.00017120384_a);

		CHECK(ions.velocities()[127][0] == 0.000149239_a);
		CHECK(ions.velocities()[127][1] == 0.000643727_a);
		CHECK(ions.velocities()[127][2] == 0.00015515_a);
		
	}
	
}
#endif
