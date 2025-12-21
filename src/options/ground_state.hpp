/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPTIONS__GROUND_STATE
#define INQ__OPTIONS__GROUND_STATE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <magnitude/energy.hpp>

#include <optional>
#include <cassert>

namespace inq {
namespace options {

class ground_state {

public:

	enum class scf_eigensolver { STEEPEST_DESCENT };

	template<class OStream>
	friend OStream & operator<<(OStream & out, scf_eigensolver const & self){
		if(self == scf_eigensolver::STEEPEST_DESCENT) out << "steepest_descent";
		return out;
	}

	template<class IStream>
	friend IStream & operator>>(IStream & in, scf_eigensolver & self){
		std::string readval;
		in >> readval;
		if(readval == "steepest_descent"){
			self = scf_eigensolver::STEEPEST_DESCENT;
		} else {
			throw std::runtime_error("INQ error: Invalid eigensolver");
		}
		return in;
	}

	enum class mixing_algo { LINEAR, BROYDEN };

	template<class OStream>
	friend OStream & operator<<(OStream & out, mixing_algo const & self){
		if(self == mixing_algo::LINEAR)     out << "linear";
		if(self == mixing_algo::BROYDEN)    out << "broyden";
		return out;
	}

	template<class IStream>
	friend IStream & operator>>(IStream & in, mixing_algo & self){
		std::string readval;
		in >> readval;
		if(readval == "linear"){
			self = mixing_algo::LINEAR;
		} else if(readval == "broyden"){
			self = mixing_algo::LINEAR;
		} else {
			throw std::runtime_error("INQ error: Invalid mixing algorithm");
		}
		return in;
	}
	
private:
		
	std::optional<scf_eigensolver> eigensolver_;
	std::optional<double> mixing_;
	std::optional<double> energy_tol_;
	std::optional<mixing_algo> mixing_algo_;
	std::optional<bool> verbose_;
	std::optional<bool> subspace_diag_;
	std::optional<int> max_steps_;
	std::optional<bool> calc_forces_;
	
public:

	auto steepest_descent(){
		ground_state solver = *this;;
		solver.eigensolver_ = scf_eigensolver::STEEPEST_DESCENT;
		return solver;
	}

	auto eigensolver() const {
		return eigensolver_.value_or(scf_eigensolver::STEEPEST_DESCENT);
	}

	auto mixing(double mixing_factor) {
		ground_state solver = *this;;
		solver.mixing_ = mixing_factor;
		return solver;
	}

	auto mixing() const {
		return mixing_.value_or(0.3);
	}

	auto energy_tolerance(quantity<magnitude::energy> etol) {
		ground_state solver = *this;;
		solver.energy_tol_ = etol.in_atomic_units();
		return solver;
	}
				
	auto energy_tolerance() const {
		return energy_tol_.value_or(1e-6);
	}
		
	auto linear_mixing(){
		ground_state solver = *this;;
		solver.mixing_algo_ = mixing_algo::LINEAR;
		return solver;
	}

	auto broyden_mixing(){
		ground_state solver = *this;;
		solver.mixing_algo_ = mixing_algo::BROYDEN;
		return solver;
	}
		
	auto mixing_algorithm() const {
		return mixing_algo_.value_or(mixing_algo::BROYDEN);
	}

	auto silent(){
		ground_state solver = *this;;
		solver.verbose_ = false;
		return solver;
	}
		
	auto verbose_output() const {
		return verbose_.value_or(true);
	}

	auto no_subspace_diag() {
		ground_state solver = *this;;
		solver.subspace_diag_ = false;
		return solver;
	}
		
	auto subspace_diag() const {
		return subspace_diag_.value_or(true);
	}

	auto max_steps(int val) {
		ground_state solver = *this;;
		solver.max_steps_ = val;
		return solver;
	}

	auto max_steps() const {
		return max_steps_.value_or(200);
	}

	auto calculate_forces() {
		ground_state solver = *this;;
		solver.calc_forces_ = true;
		return solver;
	}
		
	auto calc_forces() const {
		return calc_forces_.value_or(false);
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the options::ground_state to directory '" + dirname + "'.";

		utils::create_directory(comm, dirname);
		utils::save_optional(comm, dirname + "/eigensolver",      eigensolver_,   error_message);
		utils::save_optional(comm, dirname + "/mixing",           mixing_,        error_message);
		utils::save_optional(comm, dirname + "/energy_tol",       energy_tol_,    error_message);
		utils::save_optional(comm, dirname + "/mixing_algorithm", mixing_algo_,   error_message);
		utils::save_optional(comm, dirname + "/verbose",          verbose_,       error_message);
		utils::save_optional(comm, dirname + "/subspace_diag",    subspace_diag_, error_message);
		utils::save_optional(comm, dirname + "/max_steps",        max_steps_,     error_message);
		utils::save_optional(comm, dirname + "/calc_forces",      calc_forces_,   error_message);
	}
	
	static auto load(std::string const & dirname) {
		ground_state opts;

		utils::load_optional(dirname + "/eigensolver",      opts.eigensolver_);
		utils::load_optional(dirname + "/mixing",           opts.mixing_);
		utils::load_optional(dirname + "/energy_tol",       opts.energy_tol_);
		utils::load_optional(dirname + "/mixing_algorithm", opts.mixing_algo_);
		utils::load_optional(dirname + "/verbose",          opts.verbose_);
		utils::load_optional(dirname + "/subspace_diag",    opts.subspace_diag_);
		utils::load_optional(dirname + "/max_steps",        opts.max_steps_);
		utils::load_optional(dirname + "/calc_forces",      opts.calc_forces_);
		
		return opts;
	}
	
	template<class OStream>
	friend OStream & operator<<(OStream & out, ground_state const & self){

		using namespace magnitude;

		out << "Ground-state:\n";

		out << "  max_steps          = " << self.max_steps();
		if(not self.max_steps_.has_value()) out << " *";
		out << "\n";

		out << "  tolerance          = " << self.energy_tolerance();
		if(not self.energy_tol_.has_value()) out << " *";
		out << "\n";
		
		out << "  mixing             = " << self.mixing();
		if(not self.mixing_.has_value()) out << " *";
		out << "\n";
		
		out << "\n  * default values\n" << std::endl;
		
		return out;
	}
	
};
}
}
#endif

#ifdef INQ_OPTIONS_GROUND_STATE_UNIT_TEST
#undef INQ_OPTIONS_GROUND_STATE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
  
	using namespace inq;
	using namespace Catch::literals;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
 
	SECTION("Defaults"){

    options::ground_state solver;

    CHECK(solver.eigensolver() == options::ground_state::scf_eigensolver::STEEPEST_DESCENT);
    CHECK(solver.mixing() == 0.3_a);

  }

  SECTION("Composition"){

    auto solver = options::ground_state{}.calculate_forces().mixing(0.05).steepest_descent().linear_mixing();

		CHECK(solver.calc_forces());
    CHECK(solver.mixing() == 0.05_a);
    CHECK(solver.eigensolver() == options::ground_state::scf_eigensolver::STEEPEST_DESCENT);
    CHECK(solver.mixing_algorithm() == options::ground_state::mixing_algo::LINEAR);
		
		solver.save(comm, "save_options_ground_state");
		auto read_solver = options::ground_state::load("save_options_ground_state");

		CHECK(read_solver.calc_forces());
    CHECK(read_solver.mixing() == 0.05_a);
    CHECK(read_solver.eigensolver() == options::ground_state::scf_eigensolver::STEEPEST_DESCENT);
    CHECK(read_solver.mixing_algorithm() == options::ground_state::mixing_algo::LINEAR);

  }
}
#endif
