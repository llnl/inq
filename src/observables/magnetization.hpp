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

	std::array<double, 4> ret{};
	double const tol = 1.e-10;

	assert(norm(mp) <= 1.0+tol);

	ret[0] = 0.5*(1.0 + mp[2]);
	ret[1] = 0.5*(1.0 - mp[2]);
	ret[2] = mp[0]/2.0;
	ret[3] = -mp[1]/2.0;
	
	return ret;
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