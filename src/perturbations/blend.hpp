/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__BLEND
#define INQ__PERTURBATIONS__BLEND

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <perturbations/absorbing.hpp>
#include <perturbations/kick.hpp>
#include <perturbations/laser.hpp>
#include <perturbations/none.hpp>

namespace inq {
namespace perturbations {

class blend {

	enum class pert_id { KICK = 0, LASER = 1 };
  using any = std::variant<kick, laser>;
  std::vector<any> perts_;
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, pert_id const & self) {
		
		if(self == pert_id::KICK){
			out << "kick";
		} else if(self == pert_id::LASER){
			out << "laser";
		}
		
		return out;
	}
	
	template<class IStream>
	friend IStream & operator>>(IStream & in, pert_id & self) {

		std::string readval;
		in >> readval;

		if(readval == "kick"){
			self = pert_id::KICK;
		} else if(readval == "laser"){
			self = pert_id::LASER;
		} else {
			throw std::runtime_error("INQ error: Invalid perturbation id");
		}
		
		return in;
	}
	
public:

  blend() = default;

  void clear() {
    perts_.clear();
  }

  template <class PertType>
  void add(PertType && pert){
    perts_.emplace_back(std::forward<PertType>(pert));
  }

  template <class PertType>
  void add(PertType const & pert){
    perts_.emplace_back(pert);
  }
	
	auto size() const {
		return (long) perts_.size();
	}

	template <typename PhiType>  
	void zero_step(PhiType & phi) const {
		for(auto & pert : perts_){
			std::visit([&](auto per) { per.zero_step(phi); }, pert);
		}
	}
	
	auto has_uniform_electric_field() const {
		for(auto & pert : perts_){
			auto has = std::visit([&](auto per) { return per.has_uniform_electric_field(); }, pert);
			if(has) return true;
		}
		return false;
	}

	auto uniform_electric_field(double time) const {
    auto total = vector3<double>{0.0, 0.0, 0.0};
		for(auto & pert : perts_){
			if(std::visit([&](auto per) { return per.has_uniform_electric_field(); }, pert)){
				total += std::visit([&](auto per) { return per.uniform_electric_field(time);}, pert);
			}
		}
    return total;
	}

	auto has_uniform_vector_potential() const {
		for(auto & pert : perts_){
			auto has = std::visit([&](auto per) { return per.has_uniform_vector_potential(); }, pert);
			if(has) return true;
		}
		return false;
	}

	auto uniform_vector_potential(double time) const {
    auto total = vector3<double>{0.0, 0.0, 0.0};
		for(auto & pert : perts_){
			if(std::visit([&](auto per) { return per.has_uniform_vector_potential(); }, pert)){
				total += std::visit([&](auto per) { return per.uniform_vector_potential(time);}, pert);
			}
		}
    return total;
	}

	template<typename PotentialType>
	void potential(double const time, PotentialType & potential) const {
		for(auto & pert : perts_){
			std::visit([&](auto per) { per.potential(time, potential); }, pert);
		}
	}

	auto has_magnetic_field() const {
		for(auto & pert : perts_){
			auto has = std::visit([&](auto per) { return per.has_magnetic_field(); }, pert);
			if(has) return true;
		}
		return false;
	}
	
	template<typename MagneticField>
	void magnetic_field(double const time, MagneticField & magnetic) const {
		for(auto & pert : perts_){
			std::visit([&](auto per) { per.magnetic_field(time, magnetic); }, pert);
		}
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the perturbations::blend to directory '" + dirname + "'.";

		utils::create_directory(comm, dirname);
		utils::save_value(comm, dirname + "/num_perturbations",  size(),  error_message);

		auto index = 0;
		for(auto & pert : perts_){
			auto subdir = dirname + "/pert" + utils::num_to_str(index);
			utils::create_directory(comm, subdir);
			utils::save_value(comm, subdir + "/type", pert_id(pert.index()),  error_message);
			std::visit([&](auto per) { per.save(comm, subdir + "/save"); }, pert);
			index++;
		}
	}

	static auto load(std::string const & dirname) {
    auto error_message = "INQ error: Cannot load perturbations::blend from directory '" + dirname + "'.";

		auto bl = perturbations::blend{};

    int num;
		try {utils::load_value(dirname + "/num_perturbations", num, error_message);}
		catch(...){ return bl; };
		
		for(int index = 0; index < num; index++){
			auto subdir = dirname + "/pert" + utils::num_to_str(index);

			pert_id id;
			utils::load_value(subdir + "/type", id, error_message);

			switch (id) {
			case pert_id::KICK:
				bl.add(perturbations::kick::load(subdir + "/save"));
				break;
			case pert_id::LASER:
				bl.add(perturbations::laser::load(subdir + "/save"));				
				break;
			}
		}
			
    return bl;
	}
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, blend const & self){
		out << "Perturbations (" << self.size() << " total):\n";
		for(auto & pert : self.perts_){
			std::visit([&](auto per) { out << per; }, pert);
		}
		return out;
	}
	
};
	
}
}
#endif

#ifdef INQ_PERTURBATIONS_BLEND_UNIT_TEST
#undef INQ_PERTURBATIONS_BLEND_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	auto cell = systems::cell::orthorhombic(4.2_b, 5.5_b, 6.4_b).periodic();
  auto kick = perturbations::kick(cell, {0.1, 0.2, 0.3}, perturbations::gauge::velocity);

	SECTION("2 elements"){
		auto ps = perturbations::blend{};
		ps.add(kick);
		ps.add(perturbations::laser({1.0, 1.0, 1.0}, 1.0_Ha, perturbations::gauge::length));

		CHECK(ps.size() == 2);
		CHECK(ps.has_uniform_electric_field());

		CHECK(ps.has_uniform_electric_field());
		CHECK(ps.uniform_electric_field(M_PI/2.0)[0] == 1.0);
		CHECK(ps.uniform_electric_field(M_PI/2.0)[1] == 1.0);
		CHECK(ps.uniform_electric_field(M_PI/2.0)[2] == 1.0);

		CHECK(ps.has_uniform_vector_potential());
		CHECK(ps.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(ps.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(ps.uniform_vector_potential(1.0)[2] == -0.3);
		
	}

	SECTION("repeated elements"){
		auto ps = perturbations::blend{};
		ps.add(kick);
		ps.add(kick);
		
		CHECK(ps.size() == 2);
		CHECK(not ps.has_uniform_electric_field());

		CHECK(ps.has_uniform_vector_potential());
		CHECK(ps.uniform_vector_potential(3.0)[0] == -0.2);
		CHECK(ps.uniform_vector_potential(2.0)[1] == -0.4);
		CHECK(ps.uniform_vector_potential(1.0)[2] == -0.6);
	}

	SECTION("3 elements"){
		auto ps = perturbations::blend{};
		ps.add(kick);
		ps.add(kick);
		ps.add(perturbations::laser({1.0, 1.0, 1.0}, 1.0_Ha, perturbations::gauge::length));

		CHECK(ps.size() == 3);

		CHECK(ps.has_uniform_electric_field());
		CHECK(ps.uniform_electric_field(M_PI/2.0)[0] == 1.0);
		CHECK(ps.uniform_electric_field(M_PI/2.0)[1] == 1.0);
		CHECK(ps.uniform_electric_field(M_PI/2.0)[2] == 1.0);
		  
		CHECK(ps.has_uniform_vector_potential());
		CHECK(ps.uniform_vector_potential(3.0)[0] == -0.2);
		CHECK(ps.uniform_vector_potential(2.0)[1] == -0.4);
		CHECK(ps.uniform_vector_potential(1.0)[2] == -0.6);

		std::cout << ps;

		ps.save(comm, "save_blend_3");
		auto read_ps = perturbations::blend::load("save_blend_3");

		CHECK(read_ps.size() == 3);

		CHECK(read_ps.has_uniform_electric_field());
		CHECK(read_ps.uniform_electric_field(M_PI/2.0)[0] == 1.0);
		CHECK(read_ps.uniform_electric_field(M_PI/2.0)[1] == 1.0);
		CHECK(read_ps.uniform_electric_field(M_PI/2.0)[2] == 1.0);
		  
		CHECK(read_ps.has_uniform_vector_potential());
		CHECK(read_ps.uniform_vector_potential(3.0)[0] == -0.2);
		CHECK(read_ps.uniform_vector_potential(2.0)[1] == -0.4);
		CHECK(read_ps.uniform_vector_potential(1.0)[2] == -0.6);
	}

	SECTION("zero step"){
		
		const int nvec = 12;
		
		double phi_absdif = 0.0;
		double phi_dif = 0.0;
		
		parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
		
		basis::real_space bas(systems::cell::orthorhombic(4.2_b, 5.5_b, 6.4_b).finite(), /*spacing =*/ 0.39770182, comm);
		
		CHECK(bas.cell().periodicity() == 0);
		
		basis::field_set<basis::real_space, complex> phi(bas, nvec);
		
		//Construct a field
		for(int ix = 0; ix < phi.basis().local_sizes()[0]; ix++){
			for(int iy = 0; iy < phi.basis().local_sizes()[1]; iy++){
				for(int iz = 0; iz < phi.basis().local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						phi.hypercubic()[ix][iy][iz][ist] = complex(cos(ist+(ix+iy+iz)), 1.3*sin(ist+(cos(ix-iy-iz))));
					}
				}
			}
		}
		
		auto phi_old = phi;
		auto kick = perturbations::kick(bas.cell(), {0.1, 0.0, 0.0});
		auto ps = perturbations::blend{};
		ps.add(kick);
		ps.add(kick);
		ps.add(perturbations::laser({1.0, 1.0, 1.0}, 1.0_Ha, perturbations::gauge::length));
		
		ps.zero_step(phi);
		
		for(int ix = 0; ix < phi.basis().local_sizes()[0]; ix++){
			for(int iy = 0; iy < phi.basis().local_sizes()[1]; iy++){
				for(int iz = 0; iz < phi.basis().local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						phi_absdif += norm(phi.hypercubic()[ix][iy][iz][ist]) - norm(phi_old.hypercubic()[ix][iy][iz][ist]);
						phi_dif += norm(phi.hypercubic()[ix][iy][iz][ist] - phi_old.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}
		
		CHECK(phi_absdif == Approx(0).margin(1.0e-9));
		CHECK(phi_dif > 1.0e-9);
	}
}
#endif
