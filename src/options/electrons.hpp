/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPTIONS__ELECTRONS
#define INQ__OPTIONS__ELECTRONS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <magnitude/energy.hpp>
#include <states/ks_states.hpp>
#include <utils/load_save.hpp>

#include <cassert>
#include <optional>

namespace inq {
namespace options {

class electrons {

	std::optional<int> extra_states_;
	std::optional<double> extra_electrons_;
	std::optional<double> temperature_;
	std::optional<states::spin_config> spin_;
	std::optional<double> spacing_;
	std::optional<bool> double_grid_;	
	std::optional<double> density_factor_;
	std::optional<bool> spherical_grid_;

public:
	
	auto extra_states(int value){
		electrons conf = *this;
		conf.extra_states_ = value;
		return conf;
	}
	
	auto extra_states_val() const {
		return extra_states_.value_or(0);
	}

	auto extra_electrons(double value){
		electrons conf = *this;
		conf.extra_electrons_ = value;
		return conf;
	}

	auto extra_electrons_val() const {
		return extra_electrons_.value_or(0.0);
	}

	auto temperature(quantity<magnitude::energy> value){
		electrons conf = *this;
		conf.temperature_ = value.in_atomic_units();
		return conf;
	}

	auto temperature_val() const {
		return temperature_.value_or(0.0);
	}

	auto spin_unpolarized(){
		electrons conf = *this;
		conf.spin_ = states::spin_config::UNPOLARIZED;
		return conf;
	}
	
	auto spin_polarized(){
		electrons conf = *this;
		conf.spin_ = states::spin_config::POLARIZED;
		return conf;
	}

	auto spin_non_collinear(){
		electrons conf = *this;
		conf.spin_ = states::spin_config::NON_COLLINEAR;
		return conf;
	}
	
	auto spin_val() const {
		return spin_.value_or(states::spin_config::UNPOLARIZED);
	}
	
	auto num_spin_components_val() const {
		if(spin_val() == states::spin_config::POLARIZED) return 2;
		return 1;
	}

	auto cutoff(quantity<magnitude::energy> arg_ecut){
		electrons conf = *this;
		conf.spacing_ = M_PI*sqrt(0.5/arg_ecut.in_atomic_units());
		return conf;		
	}

	auto spacing(quantity<magnitude::length> arg_spacing){
		electrons conf = *this;
		conf.spacing_ = arg_spacing.in_atomic_units();
		return conf;		
	}

	auto spacing_value() const {
		if(not spacing_.has_value()) throw std::runtime_error("Error: the cutoff energy or the spacing have not been set");
		return *spacing_;
	}

	auto cutoff_value() const {
		return 0.5*pow(M_PI/spacing_value(), 2);
	}
	
	auto double_grid(){
		electrons conf = *this;
		conf.double_grid_ = true;
		return conf;				
	}
	
	auto double_grid_value() const {
		return double_grid_.value_or(false);
	}

	auto density_factor(double arg_factor){
		electrons conf = *this;
		conf.density_factor_ = arg_factor;
		return conf;
	}

	auto density_factor_value() const {
		return density_factor_.value_or(1.0);
	}

	void save(parallel::communicator & comm, std::string const & dirname) const {
		auto error_message = "INQ error: Cannot save the options::electrons to directory '" + dirname + "'.";

		utils::create_directory(comm, dirname);
		utils::save_optional(comm, dirname + "/extra_states", extra_states_, error_message);
		utils::save_optional(comm, dirname + "/extra_electrons", extra_electrons_, error_message);
		utils::save_optional(comm, dirname + "/temperature", temperature_, error_message);
		utils::save_optional(comm, dirname + "/spacing", spacing_, error_message);
		utils::save_optional(comm, dirname + "/double_grid", double_grid_, error_message);
		utils::save_optional(comm, dirname + "/density_factor", density_factor_, error_message);
		utils::save_optional(comm, dirname + "/spherical_grid", spherical_grid_, error_message);
		utils::save_optional(comm, dirname + "/spin",           spin_,           error_message);

	}
	
	static auto load(std::string const & dirname) {
		electrons opts;

		utils::load_optional(dirname + "/extra_states", opts.extra_states_);
		utils::load_optional(dirname + "/extra_electrons", opts.extra_electrons_);
		utils::load_optional(dirname + "/temperature", opts.temperature_);
		utils::load_optional(dirname + "/spacing", opts.spacing_);
		utils::load_optional(dirname + "/double_grid", opts.double_grid_);
		utils::load_optional(dirname + "/density_factor", opts.density_factor_);
		utils::load_optional(dirname + "/spherical_grid", opts.spherical_grid_);
		utils::load_optional(dirname + "/spin",           opts.spin_);
				
		return opts;
	}

	template<class OStream>
	friend OStream & operator<<(OStream & out, electrons const & self){

		using namespace magnitude;

		out << "Electrons:\n";

		out << "  extra-states       = " << self.extra_states_val();
		if(not self.extra_states_.has_value()) out << " *";
		out << "\n";
		
		out << "  extra-electrons    = " << self.extra_electrons_val();
		if(not self.extra_electrons_.has_value()) out << " *";
		out << "\n";

		out << "  temperature        = " << self.temperature_val() << " Ha | " << self.temperature_val()/in_atomic_units(1.0_eV) << " eV | " << self.temperature_val()/in_atomic_units(1.0_K) << " K";
		if(not self.temperature_.has_value()) out << " *";
		out << "\n";

		out << "  cutoff             = ";
		if(self.spacing_.has_value()) {
			out << self.cutoff_value() << " Ha | " << self.cutoff_value()/in_atomic_units(1.0_eV) << " eV | " << self.cutoff_value()/in_atomic_units(1.0_Ry) << " Ry";
		} else {
			out << "NOT SET *";
		}
		out << "\n";

		out << "  density-cutoff     = " << pow(self.density_factor_value(), 2) << "x";
		if(not self.density_factor_.has_value()) out << " *";
		out << "\n";
		
		out << "  spacing            = ";
		if(self.spacing_.has_value()) {
			out << self.spacing_value() << " bohr | " << self.spacing_value()/in_atomic_units(1.0_A) << " A";
		} else {
			out << "NOT SET *";
		}
		out << "\n";
		
		out << "  spin               = " << self.spin_val();
		if(not self.spin_.has_value()) out << " *";
		out << "\n";
		
		out << "\n  * default values\n" << std::endl;
		
		return out;
	}
	
};

}
}
#endif

#ifdef INQ_OPTIONS_ELECTRONS_UNIT_TEST
#undef INQ_OPTIONS_ELECTRONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	auto conf = options::electrons{}.spacing(23.1_b).extra_states(666).spin_non_collinear();

	CHECK(conf.extra_states_val() == 666);
	CHECK(conf.spacing_value() == 23.1_a);
	CHECK(conf.spin_val() == states::spin_config::NON_COLLINEAR);

	conf.save(comm, "options_electrons_save");
	auto read_conf = options::electrons::load("options_electrons_save");

	CHECK(read_conf.extra_states_val() == 666);
	CHECK(read_conf.spacing_value() == 23.1_a);
	CHECK(read_conf.spin_val() == states::spin_config::NON_COLLINEAR);
	
}
#endif
