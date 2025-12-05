/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__REAL_TIME__CRANK_NICOLSON
#define INQ__REAL_TIME__CRANK_NICOLSON

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <observables/density.hpp>
#include <operations/preconditioner.hpp>
#include <solvers/steepest_descent.hpp>
#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <utils/profiling.hpp>

namespace inq {
namespace real_time {

template <class OpType>
struct crank_nicolson_op {
	OpType & op;
	complex factor;
	
	template <class PhiType>
	PhiType operator()(PhiType const & phi) const {
		auto opphi = op(phi);
		
		gpu::run(opphi.local_set_size(), opphi.basis().local_size(),
						 [opph = begin(opphi.matrix()), ph = begin(phi.matrix()), fac = factor] GPU_LAMBDA (auto ist, auto ip){
							 opph[ip][ist] = ph[ip][ist] + fac*opph[ip][ist];
						 });
		
		return opphi;
	}
};

template <class IonSubPropagator, class ForcesType, class HamiltonianType, class SelfConsistencyType, class EnergyType>
void crank_nicolson(double const time, double const dt, systems::ions & ions, systems::electrons & electrons, IonSubPropagator const & ion_propagator, ForcesType const & forces, HamiltonianType & ham, SelfConsistencyType & sc, EnergyType & energy){

	assert(not sc.has_induced_vector_potential());
	
	CALI_CXX_MARK_FUNCTION;

	operations::preconditioner prec;
	crank_nicolson_op<decltype(ham)> op{ham, complex{0.0, 0.5*dt}};
	crank_nicolson_op<decltype(ham)> op_rhs{ham, complex{0.0, -0.5*dt}};

	auto const dens_tol = 1e-5;
	auto const exxe_tol = 1e-6;

	//calculate the right hand side with H(t)
	std::vector<states::orbital_set<basis::real_space, complex>> rhs; 
	rhs.reserve(electrons.kpin_size());	
	for(auto & phi : electrons.kpin()) rhs.emplace_back(op_rhs(phi));
	
	//propagate ionic positions to t + dt
	ion_propagator.propagate_positions(dt, ions, forces);
	if(not ion_propagator.static_ions()) {
		sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot());
		ham.update_projectors(electrons.states_basis(), electrons.atomic_pot(), ions);
		energy.ion(ionic::interaction_energy(ions.cell(), ions, electrons.atomic_pot()));
	}

	using mix_arr_type = std::remove_reference_t<decltype(electrons.spin_density())>;

	//  THERE SEEMS TO BE AN ISSUE WITH BROYDEN RIGHT NOW, SO WE HAVE TO USE LINEAR MIXING FOR THE MOMENT
	//	auto mixer = mixers::broyden<mix_arr_type>(4, 0.3, electrons.spin_density().matrix().flatted().size());
	auto mixer = mixers::linear<mix_arr_type>(0.3);

	auto old_exxe = 0.0;
	auto update_hf = true;
	auto exxe_diff = 1.0;
	
	//now calculate the wave functions in t + dt by solving a self-consistent linear equation
	for(int istep = 0; istep < 300; istep++) {
		CALI_CXX_MARK_SCOPE("crank_nicolson:iteration");

		sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);

		if(update_hf) {
			auto exxe = ham.exchange().update(electrons);
			exxe_diff = fabs(exxe - old_exxe);
			old_exxe = exxe;
			update_hf = false;
		}
		
		auto res = 0.0;
		auto iphi = 0;
		for(auto & phi : electrons.kpin()){
			auto ires = solvers::steepest_descent(op, prec, rhs[iphi], phi);
			res += ires*electrons.kpin_weights()[iphi];
			iphi++;
		}

		if(electrons.kpin_states_comm().size() > 1) electrons.kpin_states_comm().all_reduce_in_place_n(&res, 1, std::plus<>{});
		
		auto new_density = observables::density::calculate(electrons);
		auto density_diff = operations::integral_sum_absdiff(electrons.spin_density(), new_density)/electrons.states().num_electrons();
		
		if(electrons.full_comm().root()) std::cout << istep << '\t' << density_diff << '\t' << res << '\t' << exxe_diff << std::endl;
		
		if(density_diff < dens_tol) {
			update_hf = true;
			if(exxe_diff < exxe_tol) break;
		}
		
		mixer(electrons.spin_density(), new_density);
		observables::density::normalize(electrons.spin_density(), electrons.states().num_electrons());
	}

	electrons.spin_density() = observables::density::calculate(electrons);
	sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
	ham.exchange().update(electrons);
	 
}

}
}
#endif

#ifdef INQ_REAL_TIME_CRANK_NICOLSON_UNIT_TEST
#undef INQ_REAL_TIME_CRANK_NICOLSON_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
