/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__REAL_TIME__VIEWABLES
#define INQ__REAL_TIME__VIEWABLES

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <systems/ions.hpp>
#include <operations/overlap_diagonal.hpp>
#include <observables/current.hpp>
#include <observables/dipole.hpp>
#include <observables/forces_stress.hpp>
#include <perturbations/none.hpp>
#include <systems/electrons.hpp>
#include <real_time/crank_nicolson.hpp>
#include <real_time/etrs.hpp>
#include <utils/profiling.hpp>

#include <chrono>

namespace inq {
namespace real_time {

template <class ForcesType, class HamiltonianType, class Perturbation>
class viewables {
	bool last_iter_;
	int iter_;
	double time_;
	systems::ions const & ions_;
	systems::electrons const & electrons_;
	hamiltonian::energy const & energy_;
	ForcesType forces_;
  HamiltonianType const & ham_;
	Perturbation const & pert_;
	
public:

	viewables(bool last_iter, int iter, double time, systems::ions const & ions, systems::electrons const & electrons, hamiltonian::energy const & energy, ForcesType const & forces, HamiltonianType const & ham, Perturbation const & pert)
		:last_iter_(last_iter), iter_(iter), time_(time), ions_(ions), electrons_(electrons), energy_(energy), forces_(forces), ham_(ham), pert_(pert){
	}

	auto iter() const {
		return iter_;
	}

	auto last_iter() const {
		return last_iter_;
	}

	auto every(int every_iter) const {
		if(iter() == 0) return false;
		return (iter()%every_iter == 0) or last_iter(); 
	}

	auto root() const {
		return electrons_.full_comm().root();
	}
	
	auto time() const {
		return time_;
	}
	
	auto positions() const {
		return ions_.positions();
	}
	
	auto velocities() const {
		return ions_.velocities();
	}

	auto forces() {
		if(forces_.size() == 0) forces_ = observables::forces_stress{ions_, electrons_, ham_, energy_}.forces;
		return forces_;
	}

	auto energy() const {
		return energy_;
	}

	auto dipole() const {
		return observables::dipole(ions_, electrons_);
	}

	auto laser_field() const {
		return pert_.uniform_electric_field(time_);
	}

	auto uniform_vector_potential() const{
		return ions_.cell().metric().to_cartesian(ham_.uniform_vector_potential());
	}

	auto num_electrons() const {
		return operations::integral(electrons_.density());
	}

  auto current() const {
    return ions_.cell().metric().to_cartesian(observables::current(ions_, electrons_, ham_));
  }
	auto projected_occupation(const systems::electrons & gs) {
		auto calc = [] (auto occ, auto v) {
			return occ*norm(v);
		};
		gpu::array<double, 2> occ({gs.kpin_part().size(), gs.kpin()[0].set_size()}, 0.0);
		for(int ilot = 0; ilot < gs.kpin_size(); ilot++) {

			auto ortho = matrix::all_gather(operations::overlap(electrons_.kpin()[ilot], gs.kpin()[ilot]));

			for (int it = 0; it < get<0>(sizes(ortho)); it++) {
				auto start = electrons_.kpin()[ilot].set_part().start();
				auto finish = electrons_.kpin()[ilot].set_part().end();
				occ[ilot + gs.kpin_part().start()][it] = operations::sum(electrons_.occupations()[ilot], ortho[it]({start, finish}), calc)/electrons_.kpin_weights()[ilot];
			}
		}

		if(electrons_.kpin_states_comm().size() > 1){
			electrons_.kpin_states_comm().all_reduce_n(raw_pointer_cast(occ.data_elements()), occ.num_elements(), std::plus<>{});
		}

		return occ;
	}

	auto electrons() const {
		return electrons_;
	}
	
};

}
}
#endif

#ifdef INQ_REAL_TIME_VIEWABLES_UNIT_TEST
#undef INQ_REAL_TIME_VIEWABLES_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
