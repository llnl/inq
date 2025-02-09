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

auto sdm_oriented(vector3<double> const & mp) {
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

template <typename PtType, typename LattType>
GPU_FUNCTION auto distance_two_grid_points(PtType const & r1, PtType const & r2, int const periodicity, LattType const & lattice) {
	auto DMAX = sqrt(norm(lattice[0])) + sqrt(norm(lattice[1])) + sqrt(norm(lattice[2]));
	DMAX = DMAX * 10.0;
	auto dd = DMAX;
	if (periodicity == 0) {
		dd = sqrt(norm(r1 - r2));
	}
	else if (periodicity == 1) {
		auto L1 = lattice[0];
		for (auto i1 = -1; i1 < 2; i1++) {
			if (sqrt(norm(r1 - r2 + i1*L1)) < dd) dd = sqrt(norm(r1 - r2 + i1*L1));
		}
	}
	else if (periodicity == 2) {
		auto L1 = lattice[0];
		auto L2 = lattice[1];
		for (auto i1 = -1; i1 < 2; i1++) {
			for (auto i2 = -1; i2 < 2; i2++) {
				if (sqrt(norm(r1 - r2 + i1*L1 + i2*L2)) < dd) dd = sqrt(norm(r1 - r2 + i1*L1 + i2*L2));
			}
		}
	}
	else if (periodicity == 3) {
		auto L1 = lattice[0];
		auto L2 = lattice[1];
		auto L3 = lattice[2];
		for (auto i1 = -1; i1 < 2; i1++) {
			for (auto i2 = -1; i2 < 2; i2++) {
				for (auto i3 = -1; i3 < 2; i3++) {
					if (sqrt(norm(r1 - r2 + i1*L1 + i2*L2 + i3*L3)) < dd) dd = sqrt(norm(r1 - r2 + i1*L1 + i2*L2 + i3*L3));
				}
			}
		}
	}
	return dd;
}

void integrate_magnet_density(basis::field_set<basis::real_space, vector3<double>> const & local_mag_density, std::vector<vector3<double>> & magnetic_moments) {
	auto nmagc = magnetic_moments.size();
	basis::field<basis::real_space, vector3<double>> rfield(local_mag_density.basis());
	for (auto i = 0; i < nmagc; i++) {
		rfield.fill(vector3<double>{0.0, 0.0, 0.0});
		gpu::run(local_mag_density.basis().local_size(),
			[magd = begin(local_mag_density.matrix()), rf = begin(rfield.linear()), index = i] GPU_LAMBDA (auto ip){
				rf[ip] = magd[ip][index];
			});
		magnetic_moments[i] = operations::integral(rfield);
	}
}

template <typename CellType>
void local_magnetic_moments_radii(basis::field_set<basis::real_space, double> const & spin_density, std::vector<double> const & magnetic_radii, std::vector<vector3<double>> const & magnetic_centers, int const periodicity, CellType const & cell, std::vector<vector3<double>> & magnetic_moments) {
	auto nspin = spin_density.set_size();
	auto nmagc = magnetic_moments.size();
	std::array<vector3<double>, 3> lattice = {cell.lattice(0), cell.lattice(1), cell.lattice(2)};
	gpu::array<vector3<double>, 1> lattice_ = lattice;
	gpu::array<vector3<double>, 1> magnetic_centers_ = magnetic_centers;
	gpu::array<double, 1> magnetic_radii_ = magnetic_radii;
	basis::field_set<basis::real_space, vector3<double>> local_mag_density(spin_density.basis(), nmagc);
	local_mag_density.fill(vector3<double>{0.0, 0.0, 0.0});
	gpu::run(spin_density.basis().local_sizes()[2], spin_density.basis().local_sizes()[1], spin_density.basis().local_sizes()[0],
		[spd = begin(spin_density.hypercubic()), magd = begin(local_mag_density.hypercubic()), point_op = spin_density.basis().point_op(), mrs = magnetic_radii_.begin(), mcs = magnetic_centers_.begin(), latt = lattice_.begin(), nspin, nmagc, periodicity] GPU_LAMBDA (auto iz, auto iy, auto ix){
			auto rr = point_op.rvector_cartesian(ix, iy, iz);
			for (auto i = 0; i < nmagc; i++) {
				auto dd = distance_two_grid_points(rr, mcs[i], periodicity, latt);
				if (dd <= mrs[i]) magd[ix][iy][iz][i] = local_magnetization(spd[ix][iy][iz], nspin);
			}
		});
	integrate_magnet_density(local_mag_density, magnetic_moments);
}

template <typename CellType>
void local_magnetic_moments_voronoi(basis::field_set<basis::real_space, double> const & spin_density, std::vector<vector3<double>> const & magnetic_centers, int const periodicity, CellType const & cell, std::vector<vector3<double>> & magnetic_moments) {
	auto nspin = spin_density.set_size();
	auto nmagc = magnetic_moments.size();
	std::array<vector3<double>, 3> lattice = {cell.lattice(0), cell.lattice(1), cell.lattice(2)};
	gpu::array<vector3<double>, 1> lattice_ = lattice;
	gpu::array<vector3<double>, 1> magnetic_centers_ = magnetic_centers;
	auto DMAX = sqrt(norm(lattice[0])) + sqrt(norm(lattice[1])) + sqrt(norm(lattice[2]));
	DMAX = DMAX * 10.0;
	basis::field_set<basis::real_space, vector3<double>> local_mag_density(spin_density.basis(), nmagc);
	local_mag_density.fill(vector3<double>{0.0, 0.0, 0.0});
	gpu::run(spin_density.basis().local_sizes()[2], spin_density.basis().local_sizes()[1], spin_density.basis().local_sizes()[0],
		[spd = begin(spin_density.hypercubic()), magd = begin(local_mag_density.hypercubic()), point_op = spin_density.basis().point_op(), mcs = magnetic_centers_.begin(), latt = lattice_.begin(), nspin, nmagc, periodicity, DMAX] GPU_LAMBDA (auto iz, auto iy, auto ix){
			auto rr = point_op.rvector_cartesian(ix, iy, iz);
			auto dd2 = DMAX;
			auto j = -1;
			for (auto i = 0; i < nmagc; i++) {
				auto dd = distance_two_grid_points(rr, mcs[i], periodicity, latt);
				if (dd < dd2) {
					dd2 = dd;
					j = i;
				}
			}
			magd[ix][iy][iz][j] = local_magnetization(spd[ix][iy][iz], nspin);
		});
	integrate_magnet_density(local_mag_density, magnetic_moments);
}

template <typename CellType>
auto compute_local_magnetic_moments(basis::field_set<basis::real_space, double> const & spin_density, std::vector<vector3<double>> const & magnetic_centers, CellType const & cell, std::vector<double> const & magnetic_radii = {}) {
	std::vector<vector3<double>> magnetic_moments;
	for (auto i = 0; i < magnetic_centers.size(); i++) magnetic_moments.push_back(vector3<double> {0.0, 0.0, 0.0});
	if (magnetic_radii.empty()) {
		local_magnetic_moments_voronoi(spin_density, magnetic_centers, cell.periodicity(), cell, magnetic_moments);
	}
	else {
		assert(magnetic_radii.size() == magnetic_centers.size());
		local_magnetic_moments_radii(spin_density, magnetic_radii, magnetic_centers, cell.periodicity(), cell, magnetic_moments);
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
		std::vector<vector3<double>> magnetic_centers;
		for (auto i=0; i<initial_magnetization.size(); i++) magnetic_centers.push_back(ions.positions()[i]);
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
		magnetic_centers = {};
		for (auto i=0; i<initial_magnetization.size(); i++) magnetic_centers.push_back(ions.positions()[i]);
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
		std::vector<vector3<double>> magnetic_centers;
		for (auto i=0; i<initial_magnetization.size(); i++) magnetic_centers.push_back(ions.positions()[i]);
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
		for (auto i=0; i<initial_magnetization.size(); i++) magnetic_centers.push_back(ions.positions()[i]);
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