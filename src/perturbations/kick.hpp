/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__KICK
#define INQ__PERTURBATIONS__KICK

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>
#include <basis/real_space.hpp>
#include <states/orbital_set.hpp>
#include <perturbations/gauge.hpp>
#include <perturbations/none.hpp>

namespace inq {
namespace perturbations {

class kick : public perturbations::none {

	vector3<double> efield_;
	vector3<double> vpot_;	
	int periodicity_;
	gauge gauge_;

	kick(int arg_periodicity, vector3<double> const & arg_kick_field, gauge arg_gauge):
		efield_(-arg_kick_field),
		vpot_(-arg_kick_field),		
		periodicity_(arg_periodicity),
		gauge_(arg_gauge)
	{
		if(gauge_ == gauge::automatic){
			gauge_ = gauge::velocity;
			if(periodicity_ == 0) gauge_ = gauge::length; 
		}

		if(gauge_ == gauge::length){
			if(periodicity_ > 0) throw std::runtime_error("INQ Error: the length gauge cannot be used for periodic or partially-periodic systems");
			vpot_ = {0.0, 0.0, 0.0};
		}

		if(gauge_ == gauge::velocity){
			efield_ = {0.0, 0.0, 0.0};
		}

		assert(efield_ + vpot_ == -arg_kick_field);
	}
	
public:

	kick(systems::cell const & cell, vector3<double> const & arg_kick_field, gauge arg_gauge = gauge::automatic):
		kick(cell.periodicity(), arg_kick_field, arg_gauge) {
	}

	template <typename PhiType>
	void zero_step(PhiType & phi) const {

		auto cov_efield = phi.basis().cell().to_covariant(efield_);
		
		gpu::run(phi.basis().local_sizes()[2], phi.basis().local_sizes()[1], phi.basis().local_sizes()[0],
						 [pop = phi.basis().point_op(), ph = begin(phi.hypercubic()), cov_efield, nst = phi.set_part().local_size()] GPU_LAMBDA (auto iz, auto iy, auto ix){
							 
							 auto rr = pop.rvector(ix, iy, iz);
							 auto kick_factor = exp(complex(0.0, dot(cov_efield, rr)));
							 for(int ist = 0; ist < nst; ist++) ph[ix][iy][iz][ist] *= kick_factor;
						 });
	}

	auto has_uniform_vector_potential() const {
		return true;
	}
	
	auto uniform_vector_potential(double /*time*/) const {
		return vpot_;
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the perturbations::kick to directory '" + dirname + "'.";
		
		utils::create_directory(comm, dirname);
		utils::save_value(comm, dirname + "/field",         -(efield_ + vpot_),  error_message);
		utils::save_value(comm, dirname + "/periodicity",   periodicity_,        error_message);
		utils::save_value(comm, dirname + "/gauge",         gauge_,              error_message);
	}

	static auto load(std::string const & dirname) {
    auto error_message = "INQ error: Cannot load perturbations::kick from directory '" + dirname + "'.";

		vector3<double> field;
		int per;
		gauge gau;
	
    utils::load_value(dirname + "/field",          field,   error_message);
    utils::load_value(dirname + "/periodicity",    per,     error_message);
    utils::load_value(dirname + "/gauge",          gau,     error_message);
    
    return kick(per, field, gau);
	}
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, kick const & self){
		using namespace magnitude;

		out << "Kick:\n";
		out << "  field [a.u.] = " << -(self.efield_ + self.vpot_) << "\n";
		out << "  gauge        = " << self.gauge_ << "\n";
		return out;
	}
	
};

}
}
#endif

#ifdef INQ_PERTURBATIONS_KICK_UNIT_TEST
#undef INQ_PERTURBATIONS_KICK_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	SECTION("finite"){
	
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
		
		kick.zero_step(phi);
		
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
	
	SECTION("periodic"){
		auto cell = systems::cell::orthorhombic(4.2_b, 3.5_b, 6.4_b);
		auto kick = perturbations::kick(cell, {0.1, 0.2, 0.3});

		CHECK(kick.has_uniform_vector_potential());
		CHECK(kick.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(kick.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(kick.uniform_vector_potential(1.0)[2] == -0.3);

		kick.save(comm, "save_kick_periodic");
		auto read_kick = perturbations::kick::load("save_kick_periodic");

		CHECK(read_kick.has_uniform_vector_potential());
		CHECK(read_kick.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(read_kick.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(read_kick.uniform_vector_potential(1.0)[2] == -0.3);
	}

	SECTION("semi periodic"){
		auto cell = systems::cell::orthorhombic(4.2_b, 3.5_b, 6.4_b).periodicity(2);
		auto kick = perturbations::kick(cell, {0.1, 0.2, 0.3});

		CHECK(kick.has_uniform_vector_potential());
		CHECK(kick.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(kick.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(kick.uniform_vector_potential(1.0)[2] == -0.3);

		kick.save(comm, "save_kick_semi_periodic");
		auto read_kick = perturbations::kick::load("save_kick_semi_periodic");

		CHECK(read_kick.has_uniform_vector_potential());
		CHECK(read_kick.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(read_kick.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(read_kick.uniform_vector_potential(1.0)[2] == -0.3);
	}

	SECTION("velocity gauge"){
		auto cell = systems::cell::orthorhombic(4.2_b, 3.5_b, 6.4_b).finite();
		auto kick = perturbations::kick(cell, {0.1, 0.2, 0.3}, perturbations::gauge::velocity);

		CHECK(kick.has_uniform_vector_potential());
		CHECK(kick.uniform_vector_potential(3.0)[0] == -0.1);
		CHECK(kick.uniform_vector_potential(2.0)[1] == -0.2);
		CHECK(kick.uniform_vector_potential(1.0)[2] == -0.3);
	}

	SECTION("length gauge"){
		auto cell = systems::cell::orthorhombic(4.2_b, 3.5_b, 6.4_b).finite();
		auto kick = perturbations::kick(cell, {0.1, 0.2, 0.3}, perturbations::gauge::length);

		CHECK(kick.has_uniform_vector_potential());
		CHECK(kick.uniform_vector_potential(3.0)[0] == 0.0);
		CHECK(kick.uniform_vector_potential(2.0)[1] == 0.0);
		CHECK(kick.uniform_vector_potential(1.0)[2] == 0.0);
		
		std::cout << kick;
		
	}

	SECTION("length gauge for periodic systems"){
		auto cell = systems::cell::orthorhombic(4.2_b, 3.5_b, 6.4_b).periodic();
		CHECK_THROWS(perturbations::kick(cell, {0.1, 0.2, 0.3}, perturbations::gauge::length));
	}

}
#endif
