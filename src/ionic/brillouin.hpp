/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__IONS__BRILLOUIN
#define INQ__IONS__BRILLOUIN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/kpoints.hpp>
#include <systems/ions.hpp>

namespace inq {
namespace ionic {

class brillouin {

	std::vector<vector3<double, covariant>> kpoints_;
	std::vector<double> weights_;	

	brillouin(int size):
		kpoints_(size),
		weights_(size) {
	}
	
public:

	brillouin() = default;
	
  brillouin(inq::systems::ions const & ions, input::kpoints::grid const & kpts):
		kpoints_(kpts.size()),
		weights_(kpts.size())
  {
		auto num_atoms = std::max(1, ions.size());
		
		std::vector<int> types(num_atoms);
		std::vector<double> positions(3*num_atoms);
		
		//add a dummy atom, since spg doesn't work without atoms
		types[0] = 0;
		positions[0] = 0.0;
		positions[1] = 0.0;
		positions[2] = 0.0;		
		
		for(int iatom = 0; iatom < ions.size(); iatom++){
			types[iatom] = ions.species(iatom).atomic_number();
			auto pos = ions.cell().to_contravariant(ions.cell().position_in_cell(ions.positions()[iatom]));
			positions[3*iatom + 0] = pos[0];
			positions[3*iatom + 1] = pos[1];
			positions[3*iatom + 2] = pos[2];
		}

		double amat[9];
		amat[0] = ions.cell().lattice(0)[0];
		amat[1] = ions.cell().lattice(0)[1];
		amat[2] = ions.cell().lattice(0)[2];
		amat[3] = ions.cell().lattice(1)[0];
		amat[4] = ions.cell().lattice(1)[1];
		amat[5] = ions.cell().lattice(1)[2];
		amat[6] = ions.cell().lattice(2)[0];
		amat[7] = ions.cell().lattice(2)[1];
		amat[8] = ions.cell().lattice(2)[2];

		auto is_shifted = kpts.is_shifted();
		auto grid_address = std::vector<int>(3*kpts.size());
    auto map = std::vector<int>(kpts.size());
		
		spg_get_ir_reciprocal_mesh(reinterpret_cast<int (*)[3]>(grid_address.data()), map.data(), (int const *) &kpts.dims(), (int const *) &is_shifted, 0,
															 reinterpret_cast<double (*)[3]>(amat), reinterpret_cast<double (*)[3]>(positions.data()), types.data(), num_atoms, 1e-4);
		
		for(int ik = 0; ik < kpts.size(); ik++){
			auto kpr = vector3<double, covariant>{
				(grid_address[3*ik + 0] + 0.5*is_shifted[0])/kpts.dims()[0],
				(grid_address[3*ik + 1] + 0.5*is_shifted[1])/kpts.dims()[1],
				(grid_address[3*ik + 2] + 0.5*is_shifted[2])/kpts.dims()[2]};
			kpr.transform([](auto xx){ return (xx >= 0.5) ? xx - 1.0 : xx; });
			kpoints_[ik] = 2.0*M_PI*kpr;
			weights_[ik] = 1.0/kpts.size();
		}
		
  }
	
  brillouin(inq::systems::ions const & ions, input::kpoints::list const & kpts):
		kpoints_(kpts.size()),
		weights_(kpts.size())
  {

		double totalw = 0.0;
		for(int ik = 0; ik < kpts.size(); ik++){
			auto kpr = kpts.kpoint(ik);
			kpr.transform([](auto xx){ return (xx >= 0.5) ? xx - 1.0 : xx; });
			kpoints_[ik] = 2.0*M_PI*kpr;
			weights_[ik] = kpts.weight(ik);
			totalw += weights_[ik];
		}

		if(totalw < 1.0e-15) throw std::runtime_error("inq error: the k-points have zero total weight");

		for(int ik = 0; ik < kpts.size(); ik++) weights_[ik] /= totalw;

  }

  brillouin(inq::systems::ions const &, brillouin const & bz):
		brillouin(bz){
  }

	void clear() {
		kpoints_.clear();
		weights_.clear();
	}
	
  auto size() const {
    return (long) kpoints_.size();
  }

  auto kpoint(int ik) const {
		return kpoints_[ik];
  }
  
  auto kpoint_weight(int ik) const {
    return weights_[ik];
  }

	auto insert(vector3<double, covariant> kpt, double const & weight){
		kpt.transform([](auto xx){ return (xx >= 0.5) ? xx - 1.0 : xx; });
		kpoints_.emplace_back(2.0*M_PI*kpt);
		weights_.emplace_back(weight);
	}
	
	template<class OStream>
	friend OStream& operator<<(OStream& os, brillouin const & self){
		os << "Kpoints (" << self.size() << " total):\n";
		for(int ikpt = 0; ikpt < self.size(); ikpt++){
			auto kk = self.kpoint(ikpt)/(2.0*M_PI);
			tfm::format(os, "  k-point %7d = %7.3f %7.3f %7.3f   weight = %5.3f\n", ikpt, kk[0], kk[1], kk[2], self.kpoint_weight(ikpt));
		}
		os << std::endl;
		return os;
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the Brillouin zone to directory '" + dirname + "'.";

		utils::create_directory(comm, dirname);
		utils::save_value(comm, dirname + "/num_kpoints",   size(),    error_message);
		utils::save_container(comm, dirname + "/kpoints",       kpoints_,  error_message);
		utils::save_container(comm, dirname + "/weights",       weights_,  error_message);
	}
	
	static auto load(std::string const & dirname) {
		auto error_message = "INQ error: Cannot load the kpoints from directory '" + dirname + "'.";

		int num;
		utils::load_value(dirname + "/num_kpoints", num, error_message);

		brillouin bz(num);

		utils::load_array(dirname + "/kpoints", bz.kpoints_, error_message);
		utils::load_array(dirname + "/weights", bz.weights_, error_message);
		
		return bz;
	}
	
};

}
}
#endif

#ifdef INQ_IONIC_BRILLOUIN_UNIT_TEST
#undef INQ_IONIC_BRILLOUIN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using Catch::Approx;
	using namespace Catch::literals;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	SECTION("Diamond"){

		auto a =  3.567095_A;

		auto ions = systems::ions(systems::cell::lattice({0.0_b, a/2.0, a/2.0}, {a/2, 0.0_b, a/2.0}, {a/2.0, a/2.0, 0.0_b}));
		
		ions.insert_fractional("C", {0.0 , 0.0 , 0.0 });
		ions.insert_fractional("C", {0.25, 0.25, 0.25});

		auto bz1 = ionic::brillouin(ions, input::kpoints::gamma());

		CHECK(bz1.size() == 1);
		CHECK(bz1.kpoint(0)[0] == 0.0_a);
		CHECK(bz1.kpoint(0)[1] == 0.0_a);
		CHECK(bz1.kpoint(0)[2] == 0.0_a);

		bz1.insert({-0.5, -0.5, -0.5}, 0.0);
		
		CHECK(bz1.size() == 2);
		CHECK(bz1.kpoint(0)[0]/(2*M_PI) ==  0.0_a);
		CHECK(bz1.kpoint(0)[1]/(2*M_PI) ==  0.0_a);
		CHECK(bz1.kpoint(0)[2]/(2*M_PI) ==  0.0_a);
		CHECK(bz1.kpoint(1)[0]/(2*M_PI) == -0.5_a);
		CHECK(bz1.kpoint(1)[1]/(2*M_PI) == -0.5_a);
		CHECK(bz1.kpoint(1)[2]/(2*M_PI) == -0.5_a);

		bz1.clear();
		CHECK(bz1.size() == 0);
				
		auto bz2 = ionic::brillouin(ions, input::kpoints::grid({1, 2, 3}));

		CHECK(bz2.size() == 6);

		CHECK(bz2.kpoint(0)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(0)[1]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(0)[2]/(2*M_PI) == 0.0_a);
		
		CHECK(bz2.kpoint(1)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(1)[1]/(2*M_PI) == -0.5_a);
		CHECK(bz2.kpoint(1)[2]/(2*M_PI) == 0.0_a);
		
		CHECK(bz2.kpoint(2)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(2)[1]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(2)[2]/(2*M_PI) == 0.3333333333_a);
		
		CHECK(bz2.kpoint(3)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(3)[1]/(2*M_PI) == -0.5_a);
		CHECK(bz2.kpoint(3)[2]/(2*M_PI) == 0.3333333333_a);
		
		CHECK(bz2.kpoint(4)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(4)[1]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(4)[2]/(2*M_PI) == -0.3333333333_a);
		
		CHECK(bz2.kpoint(5)[0]/(2*M_PI) == 0.0_a);
		CHECK(bz2.kpoint(5)[1]/(2*M_PI) == -0.5_a);
		CHECK(bz2.kpoint(5)[2]/(2*M_PI) == -0.3333333333_a);

		auto bz3 = ionic::brillouin(ions, input::kpoints::grid({2, 2, 2}, true));

		CHECK(bz3.size() == 8);

		CHECK(bz3.kpoint(0)[0]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(0)[1]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(0)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(bz3.kpoint(1)[0]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(1)[1]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(1)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(bz3.kpoint(2)[0]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(2)[1]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(2)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(bz3.kpoint(3)[0]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(3)[1]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(3)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(bz3.kpoint(4)[0]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(4)[1]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(4)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(bz3.kpoint(5)[0]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(5)[1]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(5)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(bz3.kpoint(6)[0]/(2*M_PI) ==  0.25_a);
		CHECK(bz3.kpoint(6)[1]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(6)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(bz3.kpoint(7)[0]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(7)[1]/(2*M_PI) == -0.25_a);
		CHECK(bz3.kpoint(7)[2]/(2*M_PI) == -0.25_a);

		bz3.save(comm, "save_brillouin_3");
		auto read_bz3 = ionic::brillouin::load("save_brillouin_3");

		CHECK(read_bz3.size() == 8);

		CHECK(read_bz3.kpoint(0)[0]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(0)[1]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(0)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(read_bz3.kpoint(1)[0]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(1)[1]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(1)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(read_bz3.kpoint(2)[0]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(2)[1]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(2)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(read_bz3.kpoint(3)[0]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(3)[1]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(3)[2]/(2*M_PI) ==  0.25_a);
		
		CHECK(read_bz3.kpoint(4)[0]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(4)[1]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(4)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(read_bz3.kpoint(5)[0]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(5)[1]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(5)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(read_bz3.kpoint(6)[0]/(2*M_PI) ==  0.25_a);
		CHECK(read_bz3.kpoint(6)[1]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(6)[2]/(2*M_PI) == -0.25_a);
		
		CHECK(read_bz3.kpoint(7)[0]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(7)[1]/(2*M_PI) == -0.25_a);
		CHECK(read_bz3.kpoint(7)[2]/(2*M_PI) == -0.25_a);
		
		auto kpts = input::kpoints::list();
		
		kpts.insert({0.5, 0.5, -0.5}, 1.0);
		kpts.insert({0.1, 0.2,  0.3}, 2.0);
		kpts.insert({1.0, -0.01, 0.0}, 0.0);
		
		auto bz4 = ionic::brillouin(ions, kpts);

		CHECK(bz4.size() == 3);

		CHECK(bz4.kpoint(0)[0]/(2*M_PI) == -0.5_a);
		CHECK(bz4.kpoint(0)[1]/(2*M_PI) == -0.5_a);
		CHECK(bz4.kpoint(0)[2]/(2*M_PI) == -0.5_a);
		
		CHECK(bz4.kpoint(1)[0]/(2*M_PI) ==  0.1_a);
		CHECK(bz4.kpoint(1)[1]/(2*M_PI) ==  0.2_a);
		CHECK(bz4.kpoint(1)[2]/(2*M_PI) ==  0.3_a);
		
		CHECK(bz4.kpoint(2)[0]/(2*M_PI) ==  0.0_a);
		CHECK(bz4.kpoint(2)[1]/(2*M_PI) == -0.01_a);
		CHECK(bz4.kpoint(2)[2]/(2*M_PI) ==  0.0_a);

		CHECK(bz4.kpoint_weight(0) ==  0.33333333_a);
		CHECK(bz4.kpoint_weight(1) ==  0.66666666_a);
		CHECK(bz4.kpoint_weight(2) ==  0.0_a);

		bz4.save(comm, "save_brillouin_4");
		auto read_bz4 = ionic::brillouin::load("save_brillouin_4");

		CHECK(read_bz4.size() == 3);

		CHECK(read_bz4.kpoint(0)[0]/(2*M_PI) == -0.5_a);
		CHECK(read_bz4.kpoint(0)[1]/(2*M_PI) == -0.5_a);
		CHECK(read_bz4.kpoint(0)[2]/(2*M_PI) == -0.5_a);
		
		CHECK(read_bz4.kpoint(1)[0]/(2*M_PI) ==  0.1_a);
		CHECK(read_bz4.kpoint(1)[1]/(2*M_PI) ==  0.2_a);
		CHECK(read_bz4.kpoint(1)[2]/(2*M_PI) ==  0.3_a);
		
		CHECK(read_bz4.kpoint(2)[0]/(2*M_PI) ==  0.0_a);
		CHECK(read_bz4.kpoint(2)[1]/(2*M_PI) == -0.01_a);
		CHECK(read_bz4.kpoint(2)[2]/(2*M_PI) ==  0.0_a);

		CHECK(read_bz4.kpoint_weight(0) ==  0.33333333_a);
		CHECK(read_bz4.kpoint_weight(1) ==  0.66666666_a);
		CHECK(read_bz4.kpoint_weight(2) ==  0.0_a);
		
	}

}
#endif
