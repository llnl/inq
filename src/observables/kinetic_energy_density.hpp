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

basis::field<basis::real_space, double> kinetic_energy_density(systems::electrons const & electrons){

	CALI_CXX_MARK_FUNCTION;

	basis::field<basis::real_space, double> density(electrons.states_basis());

	density.fill(0.0);

	auto iphi = 0;
	for(auto & phi : electrons.kpin()){
		auto gphi = operations::gradient(phi, /*factor = */ 1.0, /* shift = */ phi.kpoint());
		
		gpu::run(density.basis().part().local_size(),
						 [nst = gphi.set_part().local_size(),
							occ = begin(electrons.occupations()[iphi]),
							gph = begin(gphi.matrix()),
							den = begin(density.linear()),
							cell = density.basis().cell()]
						 GPU_LAMBDA (auto ipoint){
							 for(int ist = 0; ist < nst; ist++) den[ipoint] += 0.5*occ[ist]*cell.norm(gph[ipoint][ist]);
						 });

		iphi++;
	}

	density.all_reduce(electrons.kpin_states_comm());
	return density;
	
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
