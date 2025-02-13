/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__KS_POTENTIAL
#define INQ__HAMILTONIAN__KS_POTENTIAL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <solvers/poisson.hpp>
#include <observables/density.hpp>
#include <operations/add.hpp>
#include <operations/integral.hpp>
#include <hamiltonian/xc_term.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/zeeman_coupling.hpp>
#include <options/theory.hpp>
#include <perturbations/none.hpp>
#include <solvers/velocity_verlet.hpp>
#include <utils/profiling.hpp>

namespace inq {
namespace hamiltonian {

template <typename Perturbation = perturbations::none>
class self_consistency {
	
	options::theory theory_;
	hamiltonian::xc_term xc_;
	basis::field<basis::real_space, double> vion_;
	basis::field<basis::real_space, double> core_density_;
	basis::real_space potential_basis_;
	basis::real_space density_basis_;
	Perturbation pert_;
	vector3<double, covariant> induced_vector_potential_vel_;
	vector3<double, covariant> induced_vector_potential_;

public:
	
	self_consistency(options::theory interaction, basis::real_space const & potential_basis, basis::real_space const & density_basis, int const spin_components, Perturbation const & pert = {}):
		theory_(interaction),
		xc_(interaction, spin_components),
		vion_(density_basis),
		core_density_(density_basis),
		potential_basis_(potential_basis),
		density_basis_(density_basis),
		pert_(pert),
		induced_vector_potential_vel_({0.0, 0.0, 0.0}),
		induced_vector_potential_({0.0, 0.0, 0.0})
	{
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	self_consistency(self_consistency && old, parallel::communicator new_comm):
		theory_(std::move(old.theory_)),
		xc_(std::move(old.xc_)),
		vion_(std::move(old.vion_), new_comm),
		core_density_(std::move(old.core_density_), new_comm),
		potential_basis_(std::move(old.potential_basis_), new_comm),
		density_basis_(std::move(old.density_basis_), new_comm),
		pert_(std::move(old.pert_)),
		induced_vector_potential_vel_(old.induced_vector_potential_vel_),
		induced_vector_potential_(old.induced_vector_potential_)
	{
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <class CommType, class ions_type>
	void update_ionic_fields(CommType & comm, const ions_type & ions, const hamiltonian::atomic_potential & atomic_pot){
		
		CALI_CXX_MARK_FUNCTION;
		
		auto ionic_long_range = solvers::poisson::solve(atomic_pot.ionic_density(comm, density_basis_, ions));
		auto ionic_short_range = atomic_pot.local_potential(comm, density_basis_, ions);
		vion_ = operations::add(ionic_long_range, ionic_short_range);
		
		core_density_ = atomic_pot.nlcc_density(comm, density_basis_, ions);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename HamiltonianType, typename EnergyType, typename FieldType>
	void update_hamiltonian(HamiltonianType & hamiltonian, EnergyType & energy, FieldType const & spin_density, double time = 0.0) const {

		CALI_CXX_MARK_FUNCTION;

		assert(spin_density.basis() == density_basis_);
		assert(core_density_.basis() == spin_density.basis());

		auto total_density = observables::density::total(spin_density);
			
		energy.external(operations::integral_product(total_density, vion_));

		//IONIC POTENTIAL
		auto vscalar = vion_;

		//Time-dependent perturbation
		if(pert_.has_uniform_electric_field()){
			auto efield = pert_.uniform_electric_field(time);

			gpu::run(vscalar.basis().local_sizes()[2], vscalar.basis().local_sizes()[1], vscalar.basis().local_sizes()[0],
								 [point_op = vscalar.basis().point_op(), efield, vk = begin(vscalar.cubic())] GPU_LAMBDA (auto iz, auto iy, auto ix){
									 auto rr = point_op.rvector_cartesian(ix, iy, iz);
									 vk[ix][iy][iz] += -dot(efield, rr);
								 });
		}

		pert_.potential(time, vscalar);
		
		// Hartree
		if(theory_.hartree_potential()){
			auto vhartree = solvers::poisson::solve(total_density);
			energy.hartree(0.5*operations::integral_product(total_density, vhartree));
			operations::increment(vscalar, vhartree);
		} else {
			energy.hartree(0.0);
		}

		auto vks = basis::field_set<basis::real_space, typename HamiltonianType::potential_type>(density_basis_, spin_density.set_size());

		gpu::run(spin_density.set_size(), density_basis_.local_size(), [vk = begin(vks.matrix()), vs = begin(vscalar.linear())] GPU_LAMBDA (auto ispin, auto ipoint) {
			if(ispin < 2) {
				vk[ipoint][ispin] = vs[ipoint];
			} else {
				vk[ipoint][ispin] = 0.0;
			}
		});
		
		// XC
		double exc, nvxc;
		hamiltonian.vxc_ = xc_(spin_density, core_density_, exc, nvxc);

		assert(hamiltonian.vxc_.set_size() == vks.set_size());
		
		gpu::run(hamiltonian.vxc_.local_set_size(), hamiltonian.vxc_.basis().local_size(),
						 [vx = begin(hamiltonian.vxc_.matrix()), vk = begin(vks.matrix())] GPU_LAMBDA (auto is, auto ip){
							 vk[ip][is] += vx[ip][is];
						 });

		energy.xc(exc);
		energy.nvxc(nvxc);

		// PUT THE CALCULATED POTENTIAL IN THE HAMILTONIAN
		if(potential_basis_ == vks.basis()){
			hamiltonian.scalar_potential_= std::move(vks);
		} else {
			hamiltonian.scalar_potential_ = operations::transfer::coarsen(vks, potential_basis_);
		}

		// THE VECTOR POTENTIAL
		
		if(pert_.has_uniform_vector_potential()){
			hamiltonian.uniform_vector_potential_ = potential_basis_.cell().metric().to_covariant(pert_.uniform_vector_potential(time));
		} else {
			hamiltonian.uniform_vector_potential_ = {0.0, 0.0, 0.0};
		}

		if(has_induced_vector_potential()){
			hamiltonian.uniform_vector_potential_ += induced_vector_potential_;
		}

		// THE MAGNETIC FIELD
		
		if (pert_.has_magnetic_field()) {
			basis::field<basis::real_space, vector3<double>> uniform_magnetic(spin_density.basis());
			uniform_magnetic.fill(vector3 {0.0, 0.0, 0.0});
			pert_.magnetic_field(time, uniform_magnetic);
			zeeman_coupling zc_(spin_density.set_size());
			auto zeeman_ener = 0.0;
			zc_(spin_density, uniform_magnetic, hamiltonian.scalar_potential_, zeeman_ener);
			energy.zeeman_energy(zeeman_ener);
		}
		
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	void propagate_induced_vector_potential(double const dt, vector3<double,covariant> const & current) {
		if(not has_induced_vector_potential()) return;
		solvers::velocity_verlet::propagate_positions(dt, theory_.alpha_value()*current/density_basis_.cell().volume(), induced_vector_potential_vel_, induced_vector_potential_);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	void propagate_induced_vector_potential_derivative(double const dt, vector3<double, covariant> const & current) {
		if(not has_induced_vector_potential()) return;
		solvers::velocity_verlet::propagate_velocities(dt, theory_.alpha_value()*current/density_basis_.cell().volume(), induced_vector_potential_vel_);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto exx_coefficient(){
		if(xc_.exchange().true_functional()) return xc_.exchange().exx_coefficient();
		return theory_.exchange_coefficient();
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	bool has_induced_vector_potential() const {
		return theory_.has_induced_vector_potential();
	}
		
	
};
}
}
#endif

#ifdef INQ_HAMILTONIAN_SELF_CONSISTENCY_UNIT_TEST
#undef INQ_HAMILTONIAN_SELF_CONSISTENCY_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;

}
#endif
