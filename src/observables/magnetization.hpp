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

void local_magnetic_moments(basis::field_set<basis::real_space, double> const & spin_density, std::vector<double> const & magnetic_radii, std::vector<vector3<double>> const & magnetic_centers, std::vector<vector3<double>> & magnetic_moments) {
	auto nspin = spin_density.set_size();
	basis::field<basis::real_space, vector3<double>> local_mag_density(spin_density.basis());
	for (auto i = 0; i < magnetic_moments.size(); i++) {
		std::cout << "i: " << i << std::endl;
		local_mag_density.fill(vector3<double>{0.0, 0.0, 0.0});
		gpu::run(spin_density.local_set_size(), spin_density.basis().local_sizes()[2], spin_density.basis().local_sizes()[1], spin_density.basis().local_sizes()[0],
		[spd = begin(spin_density.hypercubic()), magd = begin(local_mag_density.cubic()), point_op = spin_density.basis().point_op(), mr = magnetic_radii[i], mcs = magnetic_centers, mc = magnetic_centers[i], nspin] GPU_LAMBDA (auto ist, auto iz, auto iy, auto ix){
			auto rr = point_op.rvector_cartesian(ix, iy, iz);
			auto dd = sqrt(norm(rr - mc));
			//for (auto i = 0; i < mcs.size(); i++) {

			//}
			if (dd <= mr) magd[ix][iy][iz] = local_magnetization(spd[ix][iy][iz], nspin);
		});
		magnetic_moments[i] = operations::integral(local_mag_density);
	}
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
	using namespace Catch::literals;
	using Catch::Approx;

}
#endif