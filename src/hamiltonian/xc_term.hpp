/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__XC_TERM
#define INQ__HAMILTONIAN__XC_TERM

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <solvers/poisson.hpp>
#include <observables/density.hpp>
#include <observables/magnetization.hpp>
#include <operations/add.hpp>
#include <operations/integral.hpp>
#include <options/theory.hpp>
#include <hamiltonian/xc_functional.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <perturbations/none.hpp>
#include <utils/profiling.hpp>
#include <systems/electrons.hpp>
#include <cmath>

namespace inq {
namespace hamiltonian {

class xc_term {

	std::vector<hamiltonian::xc_functional> functionals_; 

public:
	
	xc_term(options::theory interaction, int const spin_components){
		functionals_.emplace_back(int(interaction.exchange()), std::min(spin_components, 2));
		functionals_.emplace_back(int(interaction.correlation()), std::min(spin_components, 2));
	}

  ////////////////////////////////////////////////////////////////////////////////////////////

	auto any_requires_gradient() const {
		for(auto & func : functionals_) if(func.requires_gradient()) return true;
		return false;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto any_true_functional() const {
		for(auto & func : functionals_) if(func.true_functional()) return true;
		return false;
	}
	
  ////////////////////////////////////////////////////////////////////////////////////////////

	template <typename SpinDensityType, typename CoreDensityType>
	SpinDensityType process_density(SpinDensityType const & spin_density, CoreDensityType const & core_density) const{

		SpinDensityType full_density(spin_density.basis(), std::min(2, spin_density.set_size()));

		if(spin_density.set_size() == 4) {
			gpu::run(spin_density.basis().local_size(),
							 [spi = begin(spin_density.matrix()), ful = begin(full_density.matrix()), cor = begin(core_density.linear())] GPU_LAMBDA (auto ip){
								 auto dtot = spi[ip][0] + spi[ip][1];
								 auto mag = observables::local_magnetization(spi[ip], 4);
								 auto dpol = mag.length();
								 ful[ip][0] = 0.5*(dtot + dpol);
								 ful[ip][1] = 0.5*(dtot - dpol);
								 for(int ispin = 0; ispin < 2; ispin++){
									 if(ful[ip][ispin] < 0.0) ful[ip][ispin] = 0.0;
									 ful[ip][ispin] += cor[ip]/2;
								 }
							 });
		} else {
			assert(spin_density.set_size() == 1 or spin_density.set_size() == 2);
			
			gpu::run(spin_density.basis().local_size(),
							 [spi = begin(spin_density.matrix()), ful = begin(full_density.matrix()), cor = begin(core_density.linear()), nspin = spin_density.set_size()] GPU_LAMBDA (auto ip){
								 for(int ispin = 0; ispin < nspin; ispin++){
									 ful[ip][ispin] = spi[ip][ispin];
									 if(ful[ip][ispin] < 0.0) ful[ip][ispin] = 0.0;
									 ful[ip][ispin] += cor[ip]/nspin;
								 }
							 });
		}
		
		return full_density;
	}

	///////////////////////////////////////////////////////////////////////////////////////////

	template <typename SpinDensityType, typename VXC>
	double compute_nvxc(SpinDensityType const & spin_density, VXC const & vxc) const {

		CALI_CXX_MARK_FUNCTION;
		
		auto nvxc = gpu::run(gpu::reduce(spin_density.basis().local_size()), 0.0,
												 [den = begin(spin_density.matrix()), vx = begin(vxc.matrix()), nspin = spin_density.local_set_size()] GPU_LAMBDA (auto ip){
													 if(nspin == 1) return den[ip][0]*vx[ip][0];
													 if(nspin == 2) return den[ip][0]*vx[ip][0] + den[ip][1]*vx[ip][1];
													 if(nspin == 4) return den[ip][0]*vx[ip][0] + den[ip][1]*vx[ip][1] + 2.0*den[ip][2]*vx[ip][2] - 2.0*den[ip][3]*vx[ip][3];
													 return 0.0;
												 });
												 
		if(spin_density.basis().comm().size() > 1) {
			spin_density.basis().comm().all_reduce_in_place_n(&nvxc, 1, std::plus<>{});
		}
			
		return nvxc*spin_density.basis().volume_element();
	}

  ////////////////////////////////////////////////////////////////////////////////////////////
	
  template <typename SpinDensityType, typename CoreDensityType>
  auto operator()(SpinDensityType const & spin_density, CoreDensityType const & core_density, double & exc, double & nvxc) const {

		basis::field_set<basis::real_space, double> vxc(spin_density.skeleton());
		vxc.fill(0.0);
		exc = 0.0;
		nvxc = 0.0;
		if(not any_true_functional()) return vxc;
		
		auto full_density = process_density(spin_density, core_density);
		
		double efunc = 0.0;
		
		basis::field_set<basis::real_space, double> vfunc(full_density.skeleton());
		auto density_gradient = std::optional<decltype(operations::gradient(full_density))>{};
		if(any_requires_gradient()) density_gradient.emplace(operations::gradient(full_density));

		for(auto & func : functionals_){
			if(not func.true_functional()) continue;

			evaluate_functional(func, full_density, density_gradient, efunc, vfunc);
			compute_vxc(spin_density, vfunc, vxc);

			exc += efunc;
		}

		nvxc += compute_nvxc(spin_density, vxc);

		return vxc;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename SpinDensityType, typename VxcType, typename VXC>
	static void compute_vxc(SpinDensityType const & spin_density, VxcType const & vfunc, VXC & vxc) {
		
		if (spin_density.set_size() == 4) {
			gpu::run(vfunc.basis().local_size(),
				[spi = begin(spin_density.matrix()), vxi = begin(vfunc.matrix()), vxf = begin(vxc.matrix())] GPU_LAMBDA (auto ip){
					auto b0 = 0.5*(vxi[ip][0] - vxi[ip][1]);
					auto v0 = 0.5*(vxi[ip][0] + vxi[ip][1]);
					auto mag = observables::local_magnetization(spi[ip], 4);
					auto dpol = mag.length();
					if (fabs(dpol) > 1.e-7) {
						auto e_mag = mag/dpol;
						vxf[ip][0] += v0 + b0*e_mag[2];
						vxf[ip][1] += v0 - b0*e_mag[2];
						vxf[ip][2] += b0*e_mag[0];
						vxf[ip][3] += b0*e_mag[1];
					}
					else {
						vxf[ip][0] += v0;
						vxf[ip][1] += v0;
					}
				});
		}
		else {
			assert(spin_density.set_size() == 1 or spin_density.set_size() == 2);
			gpu::run(vfunc.local_set_size(), vfunc.basis().local_size(),
				[vxi = begin(vfunc.matrix()), vxf = begin(vxc.matrix())] GPU_LAMBDA (auto is, auto ip){
					vxf[ip][is] += vxi[ip][is];
				});
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename DensityType, typename DensityGradientType>
	static void evaluate_functional(hamiltonian::xc_functional const & functional, DensityType const & density, DensityGradientType const & density_gradient,
																	double & efunctional, basis::field_set<basis::real_space, double> & vfunctional){
		CALI_CXX_MARK_FUNCTION;

		auto edens = basis::field<basis::real_space, double>(density.basis());

		assert(functional.nspin() == density.set_size());
		
		if(functional.family() == XC_FAMILY_LDA){
			xc_lda_exc_vxc(functional.libxc_func_ptr(), density.basis().local_size(), raw_pointer_cast(density.matrix().data_elements()),
										 raw_pointer_cast(edens.linear().data_elements()), raw_pointer_cast(vfunctional.matrix().data_elements()));
			gpu::sync();
			
		} else if(functional.family() == XC_FAMILY_GGA){

			auto nsig = (density.set_size() > 1) ? 3:1;
			
			basis::field_set<basis::real_space, double> sig(density.basis(), nsig);
			basis::field_set<basis::real_space, double> vsig(sig.skeleton());

			gpu::run(density.basis().local_size(),
							 [gr = begin(density_gradient->matrix()), si = begin(sig.matrix()), metric = density.basis().cell().metric(), nsig] GPU_LAMBDA (auto ip){
								 si[ip][0] = metric.norm(gr[ip][0]);
								 if(nsig > 1) si[ip][1] = metric.dot(gr[ip][0], gr[ip][1]);
								 if(nsig > 1) si[ip][2] = metric.norm(gr[ip][1]);
							 });

			xc_gga_exc_vxc(functional.libxc_func_ptr(), density.basis().local_size(), raw_pointer_cast(density.matrix().data_elements()), raw_pointer_cast(sig.matrix().data_elements()),
										 raw_pointer_cast(edens.linear().data_elements()), raw_pointer_cast(vfunctional.matrix().data_elements()), raw_pointer_cast(vsig.matrix().data_elements()));
			gpu::sync();

			basis::field_set<basis::real_space, vector3<double, covariant>> term(vfunctional.skeleton());

			gpu::run(density.basis().local_size(),
							 [vs = begin(vsig.matrix()), gr = begin(density_gradient->matrix()), te = begin(term.matrix()), nsig] GPU_LAMBDA (auto ip){
								 if(nsig == 1) te[ip][0] = -2.0*vs[ip][0]*gr[ip][0];
								 if(nsig == 3) te[ip][0] = -2.0*vs[ip][0]*gr[ip][0] - vs[ip][1]*gr[ip][1];
								 if(nsig == 3) te[ip][1] = -2.0*vs[ip][2]*gr[ip][1] - vs[ip][1]*gr[ip][0];
							 });

			auto div_term = operations::divergence(term);

			gpu::run(density.local_set_size(), density.basis().local_size(),
							 [di = begin(div_term.matrix()), vf = begin(vfunctional.matrix())] GPU_LAMBDA (auto ispin, auto ip){
								 vf[ip][ispin] += di[ip][ispin];
							 });

		} else {
			std::runtime_error("inq error: unsupported exchange correlation functional type");
		}
		
		efunctional = operations::integral_product(edens, observables::density::total(density));
		
	}
	
  ////////////////////////////////////////////////////////////////////////////////////////////
	
	auto & exchange() const {
		return functionals_[0];
	}
	
  ////////////////////////////////////////////////////////////////////////////////////////////
	
	auto & functionals() const {
		return functionals_;
	}
	
  ////////////////////////////////////////////////////////////////////////////////////////////

};
}
}
#endif

#ifdef INQ_HAMILTONIAN_XC_TERM_UNIT_TEST
#undef INQ_HAMILTONIAN_XC_TERM_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>
using namespace inq;

template<class occupations_array_type, class field_set_type, typename VxcType>
void compute_psi_vxc_psi_ofr(occupations_array_type const & occupations, field_set_type const & phi, VxcType const & vxc, basis::field<basis::real_space, double> & rfield) {

	assert(get<1>(sizes(phi.spinor_array())) == phi.spinor_dim());
	assert(get<2>(sizes(phi.spinor_array())) == phi.local_spinor_set_size());

	if (vxc.set_size() == 1) {
		gpu::run(phi.local_spinor_set_size(), phi.basis().local_size(),
			[ph = begin(phi.matrix()), rf = begin(rfield.linear()), vx=begin(vxc.matrix()), occ=begin(occupations)] GPU_LAMBDA (auto ist, auto ip) {
				rf[ip] += occ[ist] * vx[ip][0] * norm(ph[ip][ist]);
			});
	}
	else if (vxc.set_size() == 2) {
		gpu::run(phi.local_set_size(), phi.basis().local_size(),
			[ph = begin(phi.matrix()), rf = begin(rfield.linear()), vx = begin(vxc.matrix()), occ = begin(occupations), spi = phi.spin_index()] GPU_LAMBDA (auto ist, auto ip) {
				rf[ip] += occ[ist] * vx[ip][spi] * norm(ph[ip][ist]);
			});
	}
	else {
		assert(vxc.set_size() == 4);
		gpu::run(phi.local_spinor_set_size(), phi.basis().local_size(),
			[ph = begin(phi.spinor_array()), rf = begin(rfield.linear()), vx = begin(vxc.matrix()), occ = begin(occupations)] GPU_LAMBDA (auto ist, auto ip) {
				auto offdiag = vx[ip][2] + complex{0.0, 1.0}*vx[ip][3];
				auto cross = 2.0*occ[ist]*real(offdiag*conj(ph[ip][1][ist])*ph[ip][0][ist]);
				rf[ip] += occ[ist]*vx[ip][0]*norm(ph[ip][0][ist]);
				rf[ip] += occ[ist]*vx[ip][1]*norm(ph[ip][1][ist]);
				rf[ip] += cross;
			});
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

template<class CommType, typename CoreDensityType, typename SpinDensityType, class occupations_array_type, class kpin_type>
void eval_psi_vxc_psi(CommType & comm, options::theory interaction, CoreDensityType const & core_density, SpinDensityType const & spin_density, occupations_array_type const & occupations, kpin_type const & kpin, double & nvx, double & ex) {

	hamiltonian::xc_term xc_(interaction, spin_density.set_size());

	if (not xc_.any_true_functional()) {
		nvx += 0.0;
		ex += 0.0;
	}
	else {
		auto full_density = xc_.process_density(spin_density, core_density);
		double efunc = 0.0;
		basis::field_set<basis::real_space, double> vxc(spin_density.skeleton());
		vxc.fill(0.0);
		
		basis::field_set<basis::real_space, double> vfunc(full_density.skeleton());
		auto density_gradient = std::optional<decltype(operations::gradient(full_density))>{};
		if (xc_.any_requires_gradient()) density_gradient.emplace(operations::gradient(full_density));

		for (auto & func : xc_.functionals()){
			if (not func.true_functional()) continue;

			xc_.evaluate_functional(func, full_density, density_gradient, efunc, vfunc);
			xc_.compute_vxc(spin_density, vfunc, vxc);
			ex += efunc;
		}

		basis::field<basis::real_space, double> rfield(vxc.basis());
		rfield.fill(0.0);
		int iphi = 0;
		for (auto & phi : kpin) {
			compute_psi_vxc_psi_ofr(occupations[iphi], phi, vxc, rfield);
			iphi++;
		}

		rfield.all_reduce(comm);
		nvx += operations::integral(rfield);
	}
}

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude; 
	using namespace Catch::literals;
	using namespace operations;
	using Catch::Approx;
	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	if(comm.size() > 4) return; //FIXME: check the problem for size 5. It returns a multi error
	
	auto lx = 10.3;
	auto ly = 13.8;
	auto lz =  4.5;
	
	basis::real_space bas(systems::cell::orthorhombic(lx*1.0_b, ly*1.0_b, lz*1.0_b), /*spacing =*/ 0.40557787, comm);

	basis::field_set<basis::real_space, double> density_unp(bas, 1);  
	basis::field_set<basis::real_space, double> density_pol(bas, 2);
	
	//Define k-vector for test function
	auto kvec = 2.0*M_PI*vector3<double>(1.0/lx, 1.0/ly, 1.0/lz);
	
	auto ff = [] (auto & kk, auto & rr){
		return std::max(0.0, cos(dot(kk, rr)) + 1.0);
	};

	for(int ix = 0; ix < bas.local_sizes()[0]; ix++){
		for(int iy = 0; iy < bas.local_sizes()[1]; iy++){
			for(int iz = 0; iz < bas.local_sizes()[2]; iz++){
				auto vec = bas.point_op().rvector_cartesian(ix, iy, iz);
				density_unp.hypercubic()[ix][iy][iz][0] = ff(kvec, vec);
				auto pol = sin(norm(vec)/100.0);
				density_pol.hypercubic()[ix][iy][iz][0] = (1.0 - pol)*ff(kvec, vec);
				density_pol.hypercubic()[ix][iy][iz][1] = pol*ff(kvec, vec);
			}
		}
	}

	observables::density::normalize(density_unp, 42.0);
	observables::density::normalize(density_pol, 42.0);
	
	CHECK(operations::integral_sum(density_unp) == 42.0_a);
	CHECK(operations::integral_sum(density_pol) == 42.0_a); 
	
	auto grad_unp = std::optional{operations::gradient(density_unp)};
	auto grad_pol = std::optional{operations::gradient(density_pol)};

	if(bas.part().contains(5439)) {
		auto index = bas.part().global_to_local(parallel::global_index(5439));
		CHECK(density_unp.matrix()[index][0] == 0.0024885602_a);
		CHECK(density_pol.matrix()[index][0] == 0.0009452194_a);
		CHECK(density_pol.matrix()[index][1] == 0.0015433408_a);
	}

	basis::field_set<basis::real_space, double> vfunc_unp(bas, 1);  
	basis::field_set<basis::real_space, double> vfunc_pol(bas, 2);
	
	SECTION("LDA_X"){

		hamiltonian::xc_functional func_unp(XC_LDA_X, 1);
		hamiltonian::xc_functional func_pol(XC_LDA_X, 2);
		
		double efunc_unp = NAN;
		double efunc_pol = NAN;
		
		hamiltonian::xc_term::evaluate_functional(func_unp, density_unp, grad_unp, efunc_unp, vfunc_unp);
		hamiltonian::xc_term::evaluate_functional(func_pol, density_pol, grad_pol, efunc_pol, vfunc_pol);

		CHECK(efunc_unp == -14.0558385758_a);
		CHECK(efunc_pol == -15.1704508993_a);

		if(bas.part().contains(5439)) {
			auto index = bas.part().global_to_local(parallel::global_index(5439));
			CHECK(vfunc_unp.matrix()[index][0] == -0.1334462916_a);
			CHECK(vfunc_pol.matrix()[index][0] == -0.1217618773_a);
			CHECK(vfunc_pol.matrix()[index][1] == -0.1433797225_a);
		}

		if(bas.part().contains(4444)) {
			auto index = bas.part().global_to_local(parallel::global_index(4444));
			CHECK(vfunc_unp.matrix()[index][0] == -0.3276348215_a);
			CHECK(vfunc_pol.matrix()[index][0] == -0.3784052378_a);
			CHECK(vfunc_pol.matrix()[index][1] == -0.2527984139_a);
		}

	}

	SECTION("PBE_C"){
		
		hamiltonian::xc_functional func_unp(XC_GGA_C_PBE, 1);
		hamiltonian::xc_functional func_pol(XC_GGA_C_PBE, 2);
		
		double efunc_unp = NAN;
		double efunc_pol = NAN;
		
		hamiltonian::xc_term::evaluate_functional(func_unp, density_unp, grad_unp, efunc_unp, vfunc_unp);
		hamiltonian::xc_term::evaluate_functional(func_pol, density_pol, grad_pol, efunc_pol, vfunc_pol);

		CHECK(efunc_unp == -1.8220292936_a);
		CHECK(efunc_pol == -1.5664843681_a);

		if(bas.part().contains(5439)) {
			auto index = bas.part().global_to_local(parallel::global_index(5439));
			CHECK(vfunc_unp.matrix()[index][0] == 0.0005467193_a);
			CHECK(vfunc_pol.matrix()[index][0] == 0.0005956583_a);
			CHECK(vfunc_pol.matrix()[index][1] == 0.0005978958_a);
		}

		if(bas.part().contains(4444)) {
			auto index = bas.part().global_to_local(parallel::global_index(4444));
			CHECK(vfunc_unp.matrix()[index][0] == -0.0798456253_a);
			CHECK(vfunc_pol.matrix()[index][0] == -0.0667968142_a);
			CHECK(vfunc_pol.matrix()[index][1] == -0.0830118308_a);
		}

	}

	SECTION("B3LYP"){
		
		hamiltonian::xc_functional func_unp(XC_HYB_GGA_XC_B3LYP, 1);
		hamiltonian::xc_functional func_pol(XC_HYB_GGA_XC_B3LYP, 2);
		
		double efunc_unp = NAN;
		double efunc_pol = NAN;
		
		hamiltonian::xc_term::evaluate_functional(func_unp, density_unp, grad_unp, efunc_unp, vfunc_unp);
		hamiltonian::xc_term::evaluate_functional(func_pol, density_pol, grad_pol, efunc_pol, vfunc_pol);

		CHECK(efunc_unp == -13.2435562623_a);
		CHECK(efunc_pol == -13.8397387159_a);

		if(bas.part().contains(5439)) {
			auto index = bas.part().global_to_local(parallel::global_index(5439));
			CHECK(vfunc_unp.matrix()[index][0] == -0.6495909727_a);
			CHECK(vfunc_pol.matrix()[index][0] == -0.6398010386_a);
			CHECK(vfunc_pol.matrix()[index][1] == -0.6142058762_a);
		}

		if(bas.part().contains(4444)) {
			auto index = bas.part().global_to_local(parallel::global_index(4444));
			CHECK(vfunc_unp.matrix()[index][0] == -0.2879332051_a);
			CHECK(vfunc_pol.matrix()[index][0] == -0.3195127242_a);
			CHECK(vfunc_pol.matrix()[index][1] == -0.2368583776_a);
		}

	}

	SECTION("xc_term object") {

		auto hf = hamiltonian::xc_term(options::theory{}.hartree_fock(), 1);
		CHECK(hf.any_requires_gradient() == false);
		CHECK(hf.any_true_functional() == false);
		
		auto lda = hamiltonian::xc_term(options::theory{}.lda(), 1);
		CHECK(lda.any_requires_gradient() == false);
		CHECK(lda.any_true_functional() == true);

		auto pbe = hamiltonian::xc_term(options::theory{}.pbe(), 1);
		CHECK(pbe.any_requires_gradient() == true);
		CHECK(pbe.any_true_functional() == true);

	}

	SECTION("nvxc calculation unpolarized") {
		auto par = input::parallelization(comm);
		auto ions = systems::ions(systems::cell::cubic(10.0_b));
		ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_unpolarized());
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha));
		auto nvxc = result.energy.nvxc();
		auto exc = result.energy.xc();
		Approx target = Approx(nvxc).epsilon(1.e-10);
		Approx target2= Approx(exc).epsilon(1.e-10);

		auto core_density_ = electrons.atomic_pot().nlcc_density(electrons.states_comm(), electrons.spin_density().basis(), ions);
		auto nvxc2 = 0.0;
		auto exc2 = 0.0;
		eval_psi_vxc_psi(electrons.kpin_states_comm(), options::theory{}.lda(), core_density_, electrons.spin_density(), electrons.occupations(), electrons.kpin(), nvxc2, exc2);
		CHECK(nvxc2 == target);
		CHECK(exc2 == target2);
	}

	SECTION("nvxc calculation spin polarized") {
		auto par = input::parallelization(comm);
		auto ions = systems::ions(systems::cell::cubic(10.0_b));
		ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_polarized());
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha));
		auto nvxc = result.energy.nvxc();
		auto exc = result.energy.xc();
		Approx target = Approx(nvxc).epsilon(1.e-10);
		Approx target2 = Approx(exc).epsilon(1.e-10);

		auto core_density_ = electrons.atomic_pot().nlcc_density(electrons.states_comm(), electrons.spin_density().basis(), ions);
		auto nvxc2 = 0.0;
		auto exc2 = 0.0;
		eval_psi_vxc_psi(electrons.kpin_states_comm(), options::theory{}.lda(), core_density_, electrons.spin_density(), electrons.occupations(), electrons.kpin(), nvxc2, exc2);
		CHECK(nvxc2 == target);
		CHECK(exc2 == target2);
	}

	SECTION("nvxc calculation spin non collinear") {
		auto par = input::parallelization(comm);
		auto ions = systems::ions(systems::cell::cubic(10.0_b));
		ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_non_collinear());
		ground_state::initial_guess(ions, electrons);
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha));
		auto nvxc = result.energy.nvxc();
		auto exc = result.energy.xc();
		Approx target = Approx(nvxc).epsilon(1.e-10);
		Approx target2 = Approx(exc).epsilon(1.e-10);

		auto core_density_ = electrons.atomic_pot().nlcc_density(electrons.states_comm(), electrons.spin_density().basis(), ions);
		auto nvxc2 = 0.0;
		auto exc2 = 0.0;
		eval_psi_vxc_psi(electrons.kpin_states_comm(), options::theory{}.lda(), core_density_, electrons.spin_density(), electrons.occupations(), electrons.kpin(), nvxc2, exc2);
		CHECK(nvxc2 == target);
		CHECK(exc2 == target2);
	}
}
#endif
