/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__GROUND_STATE__CALCULATOR
#define INQ__GROUND_STATE__CALCULATOR

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cfloat>

#include <systems/ions.hpp>
#include <basis/real_space.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <states/ks_states.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <hamiltonian/self_consistency.hpp>
#include <hamiltonian/energy.hpp>
#include <basis/field_set.hpp>
#include <operations/randomize.hpp>
#include <operations/overlap.hpp>
#include <operations/orthogonalize.hpp>
#include <operations/preconditioner.hpp>
#include <operations/integral.hpp>
#include <observables/density.hpp>
#include <observables/forces_stress.hpp>
#include <parallel/gather.hpp>
#include <mixers/linear.hpp>
#include <mixers/broyden.hpp>
#include <eigensolvers/steepest_descent.hpp>
#include <math/complex.hpp>
#include <observables/dipole.hpp>
#include <observables/magnetization.hpp>
#include <options/ground_state.hpp>
#include <systems/electrons.hpp>
#include <ground_state/eigenvalue_output.hpp>
#include <ground_state/results.hpp>
#include <ground_state/subspace_diagonalization.hpp>

#include<tinyformat/tinyformat.h>

#include<spdlog/spdlog.h>
#include<spdlog/sinks/stdout_color_sinks.h>

#include<memory>

#include <utils/profiling.hpp>

namespace inq {
namespace ground_state {

template <typename Perturbation = perturbations::none>
class calculator {

public:

private:
	
	systems::ions const & ions_;
	options::theory inter_;
	options::ground_state solver_;
	hamiltonian::self_consistency<Perturbation> sc_;
	hamiltonian::ks_hamiltonian<double> ham_;

#ifdef ENABLE_CUDA
public:
#endif

	template <typename NormResType>
	static double state_convergence(systems::electrons & el, NormResType const & normres) {
		CALI_CXX_MARK_FUNCTION;
		
		auto state_conv = 0.0;
		
		for(int iphi = 0; iphi < el.kpin_size(); iphi++){
			assert(el.occupations()[iphi].size() == normres[iphi].size());

			state_conv += gpu::run(gpu::reduce(normres[iphi].size()), 0.0, [occ = begin(el.occupations()[iphi]), arr = begin(normres[iphi])] GPU_LAMBDA (auto ip) {
				return fabs(occ[ip]*arr[ip]);
			});
		}
		
		el.kpin_states_comm().all_reduce_n(&state_conv, 1);
		state_conv /= el.states().num_electrons();
		
		return state_conv;
	}

public:

	calculator(systems::ions const & ions, systems::electrons const & electrons, const options::theory & inter = {}, options::ground_state const & solver = {}, Perturbation const & pert = {})
		:ions_(ions),
		 inter_(inter),
		 solver_(solver),
		 sc_(inter, electrons.states_basis(), electrons.density_basis(), electrons.states().num_density_components(), pert),
		 ham_(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions_, sc_.exx_coefficient(), /* use_ace = */ true)
	{
	}

	results operator()(systems::electrons & electrons){
		
		CALI_CXX_MARK_FUNCTION;
		
		assert(electrons.kpin()[0].full_comm() == electrons.states_basis_comm());
		
		auto console = electrons.logger();
		if(solver_.verbose_output() and console) console->trace("ground-state calculation started");
		
		if(electrons.full_comm().root()) ham_.info(std::cout);
		
		results res;
		operations::preconditioner prec;
		
		using mix_arr_type = std::remove_reference_t<decltype(electrons.spin_density())>;
		
		auto mixer = [&]()->std::unique_ptr<mixers::base<mix_arr_type>>{
			switch(solver_.mixing_algorithm()){
			case options::ground_state::mixing_algo::LINEAR : return std::make_unique<mixers::linear <mix_arr_type>>(solver_.mixing());
			case options::ground_state::mixing_algo::BROYDEN: return std::make_unique<mixers::broyden<mix_arr_type>>(4, solver_.mixing(), electrons.spin_density().matrix().flatted().size());
			} __builtin_unreachable();
		}();
		
		auto old_energy = std::numeric_limits<double>::max();
		
		sc_.update_ionic_fields(electrons.states_comm(), ions_, electrons.atomic_pot());
		sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
		
		res.energy.ion(ionic::interaction_energy(ions_.cell(), ions_, electrons.atomic_pot()));
		
		double old_exe = ham_.exchange().update(electrons);
		double exe_diff = fabs(old_exe);
		auto update_hf = false;
		
		electrons.full_comm().barrier();
		auto iter_start_time = std::chrono::high_resolution_clock::now();

		auto converged = false;
		res.total_iter = solver_.max_steps();
		int conv_count = 0;
		for(int iiter = 0; iiter < solver_.max_steps(); iiter++){
			
			CALI_CXX_MARK_SCOPE("scf_iteration");
			
			if(solver_.subspace_diag()) {
				int ilot = 0;
				for(auto & phi : electrons.kpin()) {
					electrons.eigenvalues()[ilot] = subspace_diagonalization(ham_, phi);
					ilot++;
				}
				electrons.update_occupations(electrons.eigenvalues());
			}
			
			if(update_hf){
				auto exe = ham_.exchange().update(electrons);
				exe_diff = fabs(exe - old_exe);
				old_exe = exe;
			}
			
			for(auto & phi : electrons.kpin()) {
				auto fphi = operations::transform::to_fourier(std::move(phi));
				
				switch(solver_.eigensolver()){
					
				case options::ground_state::scf_eigensolver::STEEPEST_DESCENT:
					eigensolvers::steepest_descent(ham_, prec, fphi);
					break;
					
				default:
					assert(false);
				}
				
				phi = operations::transform::to_real(std::move(fphi));
			}
			
			CALI_MARK_BEGIN("mixing");
			
			double density_diff = 0.0;
			{
				auto new_density = observables::density::calculate(electrons);
				density_diff = operations::integral_sum_absdiff(electrons.spin_density(), new_density);
				density_diff /= electrons.states().num_electrons();
				
				if(inter_.self_consistent()) {
					mixer->operator()(electrons.spin_density(), new_density);
					observables::density::normalize(electrons.spin_density(), electrons.states().num_electrons());
				} else {
					electrons.spin_density() = std::move(new_density);
				}
			}
			
			sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
			
			CALI_MARK_END("mixing");
			
			{
				auto normres = res.energy.calculate(ham_, electrons);
				auto energy_diff = (res.energy.eigenvalues() - old_energy)/electrons.states().num_electrons();

				electrons.full_comm().barrier();
				std::chrono::duration<double> elapsed_seconds = std::chrono::high_resolution_clock::now() - iter_start_time;
				
				electrons.full_comm().barrier();
				iter_start_time = std::chrono::high_resolution_clock::now();
				
				auto state_conv = state_convergence(electrons, normres);
				auto ev_out = eigenvalues_output(electrons, normres);
				
				if(solver_.verbose_output() and console){
					console->info("\nSCF iter {} : wtime = {:5.2f}s e = {:.10f} de = {:5.0e} dexe = {:5.0e} dn = {:5.0e} dst = {:5.0e}\n{}", 
												iiter, elapsed_seconds.count(), res.energy.total(), energy_diff, exe_diff, density_diff, state_conv, ev_out);
				}
				
				if(fabs(energy_diff) < solver_.energy_tolerance()){
					conv_count++;
					if(conv_count > 2 and exe_diff < solver_.energy_tolerance()) {
						res.total_iter = iiter;
						converged = true;
						break;
					}
					if(conv_count > 2) update_hf = true;
				} else {
					conv_count = 0; 
				}

				old_energy = res.energy.eigenvalues();
			}
		}

		if(solver_.max_steps() > 0 and not converged) {
			throw std::runtime_error("The SCF calculation did not converge. Try reducing the mixing parameter.\n"); 
		}
		
		//make sure we have a density consistent with phi
		electrons.spin_density() = observables::density::calculate(electrons);
		sc_.update_hamiltonian(ham_, res.energy, electrons.spin_density());
		auto normres = res.energy.calculate(ham_, electrons);
			
		if(solver_.calc_forces()) {
			auto fas = observables::forces_stress{ions_, electrons, ham_, res.energy};
			res.forces = fas.forces;
			res.stress = fas.stress;
		}
		
		auto ev_out = eigenvalues_output(electrons, normres);		
		
		if(solver_.verbose_output() and console) {
			console->info("\nSCF ended after {} iterations with resulting eigenvalues and energies:\n\n{}{}", res.total_iter, ev_out.full(), res.energy);
		}
		
		res.dipole = observables::dipole(ions_, electrons);
		for(int idir = 0; idir < ions_.cell().periodicity(); idir++) res.dipole[idir] = 0.0;

		res.magnetization = observables::total_magnetization(electrons.spin_density());
		
		if(solver_.verbose_output() and console) console->trace("ground-state calculation ended normally");
		return res;
	}

	/////////////////////////////////////////

	auto & hamiltonian() const {
		return ham_;
	}
	
	
};
}
}
#endif

#ifdef INQ_GROUND_STATE_CALCULATOR_UNIT_TEST
#undef INQ_GROUND_STATE_CALCULATOR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif

