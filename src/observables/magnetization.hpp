/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__MAGNETIZATION
#define INQ__OBSERVABLES__MAGNETIZATION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <basis/field_set.hpp>
#include <operations/spatial_partitions.hpp>

namespace inq {
namespace observables {

template <class Density>
GPU_FUNCTION auto local_magnetization(Density const & spin_density, int const & components) {
	vector3<double> mag_density;

	if(components == 4){
		mag_density[0] = 2.0*spin_density[2];
		mag_density[1] =-2.0*spin_density[3];
	} else {
		mag_density[0] = 0.0;
		mag_density[1] = 0.0;							 
	}

	if(components >= 2){
		mag_density[2] = spin_density[0] - spin_density[1];
	} else {
		mag_density[2] = 0.0;
	}

	return mag_density;
}


///////////////////////////////////////////////////////////////

auto spin_dens_matr_orient(vector3<double> const & mp) {
	[[maybe_unused]] constexpr double tol = 1.e-10;
	assert(norm(mp) <= 1.0+tol);

	return std::array<double, 4>{{
		/*[0] =*/ (1.0 + mp[2])/2.0,
		/*[1] =*/ (1.0 - mp[2])/2.0,
		/*[2] =*/      + mp[0] /2.0,
		/*[3] =*/      - mp[1] /2.0
	}};
}

basis::field<basis::real_space, vector3<double>> magnetization(basis::field_set<basis::real_space, double> const & spin_density){

	// The formula comes from here: https://gitlab.com/npneq/inq/-/wikis/Magnetization
	// Note that we store the real and imaginary parts of the off-diagonal density in components 3 and 4 respectively. 
	
	basis::field<basis::real_space, vector3<double>> magnet(spin_density.basis());

	gpu::run(magnet.basis().local_size(),
					 [mag = begin(magnet.linear()), den = begin(spin_density.matrix()), components = spin_density.set_size()] GPU_LAMBDA (auto ip){
						mag[ip] = local_magnetization(den[ip], components);
					 });
	
	return magnet;
	
}

auto total_magnetization(basis::field_set<basis::real_space, double> const & spin_density){
	if(spin_density.set_size() >= 2){
		return operations::integral(observables::magnetization(spin_density));
	} else {
		return vector3<double>{0.0, 0.0, 0.0};
	}
}

template <typename CellType>
auto compute_local_magnetic_moments_radii(basis::field_set<basis::real_space, double> const & spin_density, std::vector<double> const & magnetic_radii, std::vector<vector3<double, cartesian>> const & magnetic_centers, CellType const & cell) {
	auto nspin = spin_density.set_size();
	auto nmagc = static_cast<int>(magnetic_centers.size());
	auto local_field = inq::operations::local_radii_field(magnetic_centers, magnetic_radii, cell, spin_density.basis());
	basis::field_set<basis::real_space, vector3<double>> local_mag_density(spin_density.basis(), nmagc);
	local_mag_density.fill(vector3<double>{0.0, 0.0, 0.0});
	gpu::run(spin_density.basis().local_size(),
		[spd = begin(spin_density.matrix()), magd = begin(local_mag_density.matrix()), ph = begin(local_field.matrix()), nmagc, nspin] GPU_LAMBDA (auto ip){
			for (auto i = 0; i < nmagc; i++) {
				magd[ip][i] = local_magnetization(spd[ip], nspin)*ph[ip][i];
			}
		});
	return operations::integral_local(local_mag_density);
}

template <typename CellType>
auto compute_local_magnetic_moments_voronoi(basis::field_set<basis::real_space, double> const & spin_density, std::vector<vector3<double, cartesian>> const & magnetic_centers, CellType const & cell) {
	auto nspin = spin_density.set_size();
	auto nmagc = static_cast<int>(magnetic_centers.size());
	auto voronoi = inq::operations::voronoi_field(magnetic_centers, cell, spin_density.basis());
	basis::field_set<basis::real_space, vector3<double>> local_mag_density(spin_density.basis(), nmagc);
	local_mag_density.fill(vector3<double>{0.0, 0.0, 0.0});
	gpu::run(spin_density.basis().local_size(),
		[spd = begin(spin_density.matrix()), magd = begin(local_mag_density.matrix()), vor = begin(voronoi.linear()), nspin] GPU_LAMBDA (auto ip){
			auto ci = vor[ip];
			magd[ip][ci] = local_magnetization(spd[ip], nspin);
		});
	return operations::integral_local(local_mag_density);
}

template <typename CellType>
auto compute_local_magnetic_moments(basis::field_set<basis::real_space, double> const & spin_density, std::vector<vector3<double,cartesian>> const & magnetic_centers, CellType const & cell, std::vector<double> const & magnetic_radii = {}) {
	std::vector<vector3<double>> magnetic_moments = {};
	if (magnetic_radii.empty()) {
		magnetic_moments = compute_local_magnetic_moments_voronoi(spin_density, magnetic_centers, cell);
	}
	else {
		assert(magnetic_radii.size() == magnetic_centers.size());
		magnetic_moments = compute_local_magnetic_moments_radii(spin_density, magnetic_radii, magnetic_centers, cell);
	}
	return magnetic_moments;
}

}
}

#endif

#ifdef INQ_OBSERVABLES_MAGNETIZATION_UNIT_TEST
#undef INQ_OBSERVABLES_MAGNETIZATION_UNIT_TEST

#include <basis/trivial.hpp>
#include <math/complex.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	SECTION("SPIN POLARIZED INITIALIZATION"){
		
		auto par = input::parallelization(comm);
		auto cell = systems::cell::cubic(15.0_b).finite();
		auto ions = systems::ions(cell);
		ions.insert("Fe", {0.0_b, 0.0_b, 0.0_b});
		auto conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_polarized();
		auto electrons = systems::electrons(par, ions, conf);
		std::vector<vector3<double>> initial_magnetization = {{0.0, 0.0, 1.0}};

		ground_state::initial_guess(ions, electrons, initial_magnetization);
		auto mag = observables::total_magnetization(electrons.spin_density());
		std::vector<vector3<double,cartesian>> magnetic_centers = {ions.positions()[0]};
		auto magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		Approx target = Approx(mag[2]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][2] == target);

		initial_magnetization = {{0.0, 0.0, -1.0}};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		target = Approx(mag[2]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][2] == target);

		auto a = 6.1209928_A;
		cell = systems::cell::lattice({-a/2.0, 0.0_A, a/2.0}, {0.0_A, a/2.0, a/2.0}, {-a/2.0, a/2.0, 0.0_A});
		assert(cell.periodicity() == 3);
		ions = systems::ions(cell);
		ions.insert_fractional("Fe", {0.0, 0.0, 0.0});
		ions.insert_fractional("Fe", {0.5, 0.5, 0.5});
		conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_polarized();
		electrons = systems::electrons(par, ions, conf);
		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, -0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_centers = {ions.positions()[0], ions.positions()[1]};
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);

		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, 0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);
	}

	SECTION("SPIN NON COLLINEAR INITIALIZATION"){
		
		auto par = input::parallelization(comm);
		auto cell = systems::cell::cubic(15.0_b).finite();
		auto ions = systems::ions(cell);
		ions.insert("Fe", {0.0_b, 0.0_b, 0.0_b});
		auto conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_non_collinear();
		auto electrons = systems::electrons(par, ions, conf);
		std::vector<vector3<double>> initial_magnetization = {{1.0, 0.0, 0.0}};

		ground_state::initial_guess(ions, electrons, initial_magnetization);
		auto mag = observables::total_magnetization(electrons.spin_density());
		std::vector<vector3<double, cartesian>> magnetic_centers;
		magnetic_centers = {ions.positions()[0]};
		auto magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		Approx target = Approx(mag[0]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][0] == target);

		initial_magnetization = {{-1.0, 0.0, 0.0}};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		target = Approx(mag[0]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][0] == target);

		auto a = 6.1209928_A;
		cell = systems::cell::lattice({-a/2.0, 0.0_A, a/2.0}, {0.0_A, a/2.0, a/2.0}, {-a/2.0, a/2.0, 0.0_A});
		assert(cell.periodicity() == 3);
		ions = systems::ions(cell);
		ions.insert_fractional("Fe", {0.0, 0.0, 0.0});
		ions.insert_fractional("Ni", {0.5, 0.5, 0.5});
		conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_non_collinear();
		electrons = systems::electrons(par, ions, conf);
		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, -0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_centers = {ions.positions()[0], ions.positions()[1]};
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);

		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, 0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);
		
		std::vector<double> magnetic_radii = {1.0, 1.0};
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell, magnetic_radii);
		CHECK(Approx(magnetic_moments[0][2]).margin(1.e-7) == 4.4085);
		CHECK(Approx(magnetic_moments[1][2]).margin(1.e-7) == 5.18108);
	}

	SECTION("SPIN NON COLLINEAR INITIALIZATION"){
		
		auto par = input::parallelization(comm);
		auto cell = systems::cell::cubic(15.0_b).finite();
		auto ions = systems::ions(cell);
		ions.insert("Fe", {0.0_b, 0.0_b, 0.0_b});
		auto conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_non_collinear();
		auto electrons = systems::electrons(par, ions, conf);
		std::vector<vector3<double>> initial_magnetization = {{1.0, 0.0, 0.0}};

		ground_state::initial_guess(ions, electrons, initial_magnetization);
		auto mag = observables::total_magnetization(electrons.spin_density());
		std::vector<vector3<double>> magnetic_centers;
		auto nmagc = static_cast<int>(initial_magnetization.size());
		for (auto i=0; i<nmagc; i++) magnetic_centers.push_back(ions.positions()[i]);
		auto magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		Approx target = Approx(mag[0]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][0] == target);

		initial_magnetization = {{-1.0, 0.0, 0.0}};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		target = Approx(mag[0]).epsilon(1.e-10);
		CHECK(magnetic_moments[0][0] == target);

		auto a = 6.1209928_A;
		cell = systems::cell::lattice({-a/2.0, 0.0_A, a/2.0}, {0.0_A, a/2.0, a/2.0}, {-a/2.0, a/2.0, 0.0_A});
		assert(cell.periodicity() == 3);
		ions = systems::ions(cell);
		ions.insert_fractional("Fe", {0.0, 0.0, 0.0});
		ions.insert_fractional("Ni", {0.5, 0.5, 0.5});
		conf = options::electrons{}.cutoff(40.0_Ha).extra_states(10).temperature(300.0_K).spin_non_collinear();
		electrons = systems::electrons(par, ions, conf);
		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, -0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_centers = {};
		nmagc = static_cast<int>(initial_magnetization.size());
		for (auto i=0; i<nmagc; i++) magnetic_centers.push_back(ions.positions()[i]);
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);

		initial_magnetization = {
			{0.0, 0.0, 0.5}, 
			{0.0, 0.0, 0.5}
		};
		ground_state::initial_guess(ions, electrons, initial_magnetization);
		mag = observables::total_magnetization(electrons.spin_density());
		magnetic_moments = inq::observables::compute_local_magnetic_moments(electrons.spin_density(), magnetic_centers, cell);
		CHECK(Approx(magnetic_moments[0][2] + magnetic_moments[1][2]).margin(1.e-7) == mag[2]);
	}

}
#endif