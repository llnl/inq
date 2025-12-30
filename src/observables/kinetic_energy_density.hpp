/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__KINETIC_ENERGY_DENSITY
#define INQ__OBSERVABLES__KINETIC_ENERGY_DENSITY

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <basis/real_space.hpp>
#include <basis/field.hpp>
#include <systems/electrons.hpp>
#include <operations/gradient.hpp>

namespace inq {
namespace observables {

basis::field_set<basis::real_space, double> kinetic_energy_density(systems::electrons const & electrons){

	CALI_CXX_MARK_FUNCTION;

	basis::field_set<basis::real_space, double> kdensity(electrons.states_basis(), electrons.states().num_density_components());

	kdensity.fill(0.0);

	auto iphi = 0;
	for(auto & phi : electrons.kpin()){
		auto gphi = operations::gradient(phi, /*factor = */ 1.0, /* shift = */ phi.kpoint());

		if(not phi.spinors()){
			
			gpu::run(kdensity.basis().part().local_size(),
							 [nst = gphi.set_part().local_size(), cell = kdensity.basis().cell(), ispin = phi.spin_index(),
								_occupations = electrons.occupations()[iphi].cbegin(), _gphi = gphi.matrix().cbegin(), _kdensity = kdensity.matrix().begin()] GPU_LAMBDA (auto ipoint){
								 for(int ist = 0; ist < nst; ist++) _kdensity[ipoint][ispin] += 0.5*_occupations[ist]*cell.norm(_gphi[ipoint][ist]);
							 });

		} else {
			
			gpu::run(kdensity.basis().part().local_size(),
							 [nst = gphi.local_spinor_set_size(), cell = kdensity.basis().cell(),
								_occupations = electrons.occupations()[iphi].cbegin(), _gphi = gphi.spinor_array().cbegin(), _kdensity = kdensity.matrix().begin()] GPU_LAMBDA (auto ipoint){
								 for(int ist = 0; ist < nst; ist++) {
									 _kdensity[ipoint][0] += 0.5*_occupations[ist]*cell.norm(_gphi[ipoint][0][ist]);
									 _kdensity[ipoint][1] += 0.5*_occupations[ist]*cell.norm(_gphi[ipoint][1][ist]);
									 auto crossterm = 0.5*_occupations[ist]*cell.dot(_gphi[ipoint][0][ist], _gphi[ipoint][1][ist]);
									 _kdensity[ipoint][2] += real(crossterm);
									 _kdensity[ipoint][3] += imag(crossterm);
								 }
							 });
			
		}

		iphi++;
	}

	kdensity.all_reduce(electrons.kpin_states_comm());
	return kdensity;
	
}

}
}
#endif

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef INQ_OBSERVABLES_KINETIC_ENERGY_DENSITY_UNIT_TEST
#undef INQ_OBSERVABLES_KINETIC_ENERGY_DENSITY_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;

}
#endif
