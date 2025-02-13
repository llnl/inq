/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OBSERVABLES__FORCES_STRESS
#define INQ__OBSERVABLES__FORCES_STRESS

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <observables/density.hpp>
#include <operations/gradient.hpp>
#include <solvers/poisson.hpp>
#include <systems/ions.hpp>
#include <systems/electrons.hpp>
#include <utils/raw_pointer_cast.hpp>

namespace inq {
namespace observables {

struct forces_stress {
	gpu::array<vector3<double>, 1> forces;
	vector3<vector3<double>> stress;

	forces_stress() = default;

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename HamiltonianType, typename Energy>
	forces_stress(systems::ions const & ions, systems::electrons const & electrons, HamiltonianType const & ham, Energy const & energy):
		forces(ions.size())
	{
		calculate(ions, electrons, ham, energy);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	
#ifndef ENABLE_GPU
private:
#endif

	GPU_FUNCTION static void stress_component(int const index, int & alpha, int & beta) {
		alpha = index;
		beta = index;
		if(index == 3) {
			alpha = 0;
			beta = 1;
		}
		if(index == 4) {
			alpha = 1;
			beta = 2;
		}
		if(index == 5) {
			alpha = 0;
			beta = 2;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Stress1D>
	auto tensor(Stress1D const & stress1d) {
		vector3<vector3<double>> stress;

		for(auto index = 0; index < 6; index++) {
			int alpha, beta;
			stress_component(index, alpha, beta);
			stress[alpha][beta] = stress1d[index];
			if(beta != alpha) stress[beta][alpha] = stress1d[index];
		}

		return stress;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename GPhi, typename Occupations>
	vector3<vector3<double>> stress_kinetic(GPhi const & gphi, Occupations const & occupations) {

		auto stress1d = gpu::run(6, gpu::reduce(gphi.local_set_size()), gpu::reduce(gphi.basis().local_size()), 0.0,
														 [metric = gphi.basis().cell().metric(), gph = begin(gphi.matrix()), occ = begin(occupations)] GPU_LAMBDA (auto index, auto ist, auto ip) {
															 int alpha, beta;
															 stress_component(index, alpha, beta);
															 auto grad_cart = metric.to_cartesian(gph[ip][ist]);
															 return occ[ist]*real(conj(grad_cart[alpha])*grad_cart[beta]);
														 });
		
		if(gphi.full_comm().size() > 1) gphi.full_comm().all_reduce_n(raw_pointer_cast(stress1d.data_elements()), 6);

		return -gphi.basis().volume_element()*tensor(stress1d);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Density>
	vector3<vector3<double>> stress_electrostatic(Density const & density) {

		auto potential = solvers::poisson::solve(density);
		auto efield = operations::gradient(potential);

		auto stress1d = gpu::run(6, gpu::reduce(efield.basis().local_size()), 0.0,
														 [metric = efield.basis().cell().metric(), ef = begin(efield.linear())] GPU_LAMBDA (auto index, auto ip) {
															 int alpha, beta;
															 stress_component(index, alpha, beta);
															 auto ef_cart = metric.to_cartesian(ef[ip]);
															 return ef_cart[alpha]*ef_cart[beta];
														 });

		if(efield.basis().comm().size() > 1) efield.basis().comm().all_reduce_n(raw_pointer_cast(stress1d.data_elements()), 6);

		return density.basis().volume_element()/(4.0*M_PI)*tensor(stress1d);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename HamiltonianType, typename Energy>
	void calculate(const systems::ions & ions, systems::electrons const & electrons, HamiltonianType const & ham, Energy const & energy){
		// This function calculates the force and the stress. Sources:
		//   - Force: Page 41, Eq. (2.41) of https://digital.csic.es/bitstream/10261/44512/1/xandrade_phd.pdf
		//   - Stress formulas: Eq. (33) of https://arxiv.org/pdf/1809.08157

		
		CALI_CXX_MARK_FUNCTION;

		// SET THE STRESS TO ZERO
		for(auto alpha = 0; alpha < 3; alpha++){
			for(auto beta = 0; beta < 3; beta++){
				stress[alpha][beta] = 0.0;
			}
		}
		
		basis::field<basis::real_space, vector3<double, covariant>> gdensity(electrons.density_basis());
		gdensity.fill(vector3<double, covariant>{0.0, 0.0, 0.0});
		
		gpu::array<vector3<double>, 1> forces_non_local(ions.size(), {0.0, 0.0, 0.0});

		auto iphi = 0;
		for(auto & phi : electrons.kpin()){
			
			auto gphi = operations::gradient(phi, /* factor = */ 1.0, /*shift = */ phi.kpoint() + ham.uniform_vector_potential());
			observables::density::calculate_gradient_add(electrons.occupations()[iphi], phi, gphi, gdensity);
			
			ham.projectors_all().force(phi, gphi, electrons.occupations()[iphi], phi.kpoint() + ham.uniform_vector_potential(), forces_non_local);
			for(auto & pr : ham.projectors_rel()) pr.force(phi, gphi, electrons.occupations()[iphi], phi.kpoint() + ham.uniform_vector_potential(), forces_non_local);

			stress += stress_kinetic(gphi, electrons.occupations()[iphi]);

			iphi++;
		}
		
		gdensity.all_reduce(electrons.kpin_states_comm());
		
		if(electrons.full_comm().size() > 1){
			CALI_CXX_MARK_SCOPE("forces_nonlocal::reduce");
			electrons.full_comm().all_reduce_n(raw_pointer_cast(forces_non_local.data_elements()), forces_non_local.size(), std::plus<>{});
		}
		
		auto ionic_forces = ionic::interaction_forces(ions.cell(), ions, electrons.atomic_pot());
		
		gpu::array<vector3<double>, 1> forces_local(ions.size(), {0.0, 0.0, 0.0});
		
		{ CALI_CXX_MARK_SCOPE("forces_local");
			
			//the force from the local potential
			for(int iatom = 0; iatom < ions.size(); iatom++){
				auto ionic_long_range = solvers::poisson::solve(electrons.atomic_pot().ionic_density(electrons.states_comm(), electrons.density_basis(), ions, iatom));
				auto ionic_short_range = electrons.atomic_pot().local_potential(electrons.states_comm(), electrons.density_basis(), ions, iatom);
				
				auto force_cov = -gpu::run(gpu::reduce(electrons.density_basis().local_size()), zero<vector3<double, inq::covariant>>(),
																	 [v1 = begin(ionic_long_range.linear()), v2 = begin(ionic_short_range.linear()), gdensityp = begin(gdensity.linear())] GPU_LAMBDA (auto ip) {
																		 return (v1[ip] + v2[ip])*gdensityp[ip];
																	 });

				forces_local[iatom] = electrons.density_basis().volume_element()*ions.cell().metric().to_cartesian(force_cov);
			}
			
			if(electrons.density_basis().comm().size() > 1){
				CALI_CXX_MARK_SCOPE("forces_local::reduce");
				electrons.density_basis().comm().all_reduce_n(reinterpret_cast<double *>(raw_pointer_cast(forces_local.data_elements())), 3*forces_local.size());
			}
		}

		auto forces_nlcc = electrons.atomic_pot().nlcc_forces(electrons.states_comm(), ham.vxc(), ions);

		for(int iatom = 0; iatom < ions.size(); iatom++){
			forces[iatom] = ionic_forces[iatom] + forces_local[iatom] + forces_non_local[iatom] + forces_nlcc[iatom];
		}

		// THE XC CONTRIBUTION TO THE STRESS
		for(int alpha = 0; alpha < 3; alpha++) {
			stress[alpha][alpha] += energy.xc() - energy.nvxc();
		}

		//missing: the XC gradient term

		stress += stress_electrostatic(electrons.density());
		
		// Divide by the cell volume
		for(auto alpha = 0; alpha < 3; alpha++){
			for(auto beta = 0; beta < 3; beta++){
				stress[alpha][beta] /= electrons.density_basis().cell().volume();
			}
		}
		
	}
	
};

}
}
#endif

#ifdef INQ_OBSERVABLES_FORCES_STRESS_UNIT_TEST
#undef INQ_OBSERVABLES_FORCES_STRESS_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;
	
}
#endif

