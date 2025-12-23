/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__KS_HAMILTONIAN
#define INQ__HAMILTONIAN__KS_HAMILTONIAN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <multi/adaptors/fftw.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/exchange_operator.hpp>
#include <hamiltonian/projector.hpp>
#include <hamiltonian/projector_all.hpp>
#include <hamiltonian/relativistic_projector.hpp>
#include <hamiltonian/scalar_potential.hpp>
#include <input/environment.hpp>
#include <operations/transform.hpp>
#include <operations/divergence.hpp>
#include <operations/laplacian.hpp>
#include <operations/gradient.hpp>
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>

#include <utils/profiling.hpp>

#include <list>
#include <unordered_map>

namespace inq {
namespace hamiltonian {

template <typename PotentialType>
class ks_hamiltonian {
		
public:

	using potential_type = PotentialType;

private:

	exchange_operator exchange_;
	basis::field_set<basis::real_space, double> vxc_;
	basis::field_set<basis::real_space, PotentialType> scalar_potential_;
	std::optional<basis::field_set<basis::real_space, double>> vmgga_;
	vector3<double, covariant> uniform_vector_potential_;
	projector_all projectors_all_;		
	std::list<relativistic_projector> projectors_rel_;
	states::ks_states states_;

#ifdef ENABLE_CUDA
public:
#endif
		
		template <typename OccType, typename ArrayType>
		static double occ_sum(OccType const & occupations, ArrayType const & array) {
			CALI_CXX_MARK_FUNCTION;
			
			assert(occupations.size() == array.size());
			return gpu::run(gpu::reduce(array.size()), 0.0, [occ = begin(occupations), arr = begin(array)] GPU_LAMBDA (auto ip) {
				return occ[ip]*real(arr[ip]);
			});
		}

public:
	
	void update_projectors(const basis::real_space & basis, const atomic_potential & pot, systems::ions const & ions){
			
		CALI_CXX_MARK_FUNCTION;

		std::list<projector> projectors;
			
		for(int iatom = 0; iatom < ions.size(); iatom++){
			auto && ps = pot.pseudo_for_element(ions.species(iatom));

			if(ps.has_total_angular_momentum()){
				projectors_rel_.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors_rel_.back().empty()) projectors_rel_.pop_back();
			} else {
				projectors.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors.back().empty()) projectors.pop_back();
			}
		}

		projectors_all_ = projector_all(projectors);
	}
		

	////////////////////////////////////////////////////////////////////////////////////////////
		
	ks_hamiltonian(const basis::real_space & basis, ionic::brillouin const & bzone, states::ks_states const & states, atomic_potential const & pot, systems::ions const & ions,
								 const double exchange_coefficient, bool use_ace = false):
		exchange_(basis.cell(), bzone, exchange_coefficient, use_ace),
		vxc_(basis, states.num_density_components()),
		scalar_potential_(basis, states.num_density_components()),
		uniform_vector_potential_({0.0, 0.0, 0.0}),
		states_(states)
	{
		scalar_potential_.fill(0.0);
		update_projectors(basis, pot, ions);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto non_local(const states::orbital_set<basis::real_space, complex> & phi) const {

		CALI_CXX_MARK_FUNCTION;
 
		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);
		
		states::orbital_set<basis::real_space, complex> vnlphi(phi.skeleton());
		vnlphi.fill(0.0);
		
		projectors_all_.apply(proj, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		for(auto & pr : projectors_rel_) pr.apply(phi, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		return vnlphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Occupations>
	auto non_local_energy(states::orbital_set<basis::real_space, complex> const & phi, Occupations const & occupations, bool const reduce_states = true) const {

		CALI_CXX_MARK_FUNCTION;

		auto en = projectors_all_.energy(phi, phi.kpoint() + uniform_vector_potential_, occupations, reduce_states);
		for(auto & pr : projectors_rel_) en += pr.energy(phi, occupations, phi.kpoint() + uniform_vector_potential_);
		return en;
		
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	void mgga_term(states::orbital_set<basis::real_space, complex> const & phi, states::orbital_set<basis::real_space, complex> & hphi) const {
		//THIS PROBABLY CAN BE DONE MORE EFFICIENTLY BY REUSING THE FFTS IN THE HAMILTONIAN
		CALI_CXX_MARK_FUNCTION;
		
		if(vmgga_.has_value()) {
			auto gphi = operations::gradient(phi);
			gpu::run(gphi.local_set_size(), gphi.basis().local_size(),
							 [ispin = phi.spin_index(), _gphi = begin(gphi.matrix()), _vmgga = begin(vmgga_->matrix())] GPU_LAMBDA (auto ist, auto ip) {
								 _gphi[ip][ist] *= _vmgga[ip][ispin];
							 });
			auto div = operations::divergence(gphi);
			gpu::run(hphi.local_set_size(), hphi.basis().local_size(),
							 [_hphi = begin(hphi.matrix()), _div = begin(div.matrix())] GPU_LAMBDA (auto ist, auto ip) {
								 _hphi[ip][ist] -= _div[ip][ist];
							 });
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("hamiltonian_real");

		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);

		auto phi_fs = operations::transform::to_fourier(phi);
		
		auto hphi_fs = operations::laplacian(phi_fs, -0.5, -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_));
			
		auto hphi = operations::transform::to_real(hphi_fs);

		hamiltonian::scalar_potential_add(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().norm(phi.kpoint() + uniform_vector_potential_), phi, hphi);
		exchange_(phi, hphi);
		mgga_term(phi, hphi);
		
		for(auto & pr : projectors_rel_) pr.apply(phi, hphi, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi, phi.kpoint() + uniform_vector_potential_);

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::fourier_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("hamiltonian_fourier");

		auto phi_rs = operations::transform::to_real(phi);

		auto proj = projectors_all_.project(phi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi_rs = hamiltonian::scalar_potential(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().norm(phi.kpoint() + uniform_vector_potential_), phi_rs);
		exchange_(phi_rs, hphi_rs);
		mgga_term(phi_rs, hphi_rs);

		for(auto & pr : projectors_rel_) pr.apply(phi_rs, hphi_rs, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi = operations::transform::to_fourier(hphi_rs);

		operations::laplacian_add(phi, hphi, -0.5, -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_));

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto kinetic_expectation_value(states::orbital_set<basis::fourier_space, complex> const & phi) const {
		CALI_CXX_MARK_FUNCTION;

		return operations::laplacian_expectation_value(phi, -0.5, -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_));
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto kinetic_expectation_value(states::orbital_set<basis::real_space, complex> const & phi) const {
		CALI_CXX_MARK_FUNCTION;

		auto phi_fs = operations::transform::to_fourier(phi);
		return kinetic_expectation_value(phi_fs);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto overlap(const states::orbital_set<basis::real_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("overlap_real");

		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_, true);
			
		auto sphi = phi;
		projectors_all_.apply(proj, sphi, phi.kpoint() + uniform_vector_potential_);

		return sphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto momentum(const states::orbital_set<basis::real_space, complex> & phi) const{
		CALI_CXX_MARK_FUNCTION;

		return operations::gradient(phi, /* factor = */ 1.0, /*shift = */ phi.kpoint() + uniform_vector_potential_);
	}
		
	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto & projectors_all() const {
		return projectors_all_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto & projectors_rel() const {
		return projectors_rel_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <class output_stream>
	void info(output_stream & out) const {
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & scalar_potential() {
		return scalar_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & exchange() {
		return exchange_;
	}
	
	auto & exchange() const {
		return exchange_;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto & uniform_vector_potential() const {
		return uniform_vector_potential_;
	}
	auto & uniform_vector_potential() {
		return uniform_vector_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & vxc() const {
		return vxc_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Perturbation>
	friend class self_consistency;
	
};
}
}
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
#endif

#ifdef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST
#undef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>
#include <config/path.hpp>
TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Hamiltonian application") {
		
		parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});
		
		auto set_comm = basis::set_subcomm(cart_comm);
		auto basis_comm = basis::basis_subcomm(cart_comm);
		
		auto ions = systems::ions(systems::cell::cubic(10.0_b));

		basis::real_space rs(ions.cell(), /*spacing = */ 0.49672941, basis_comm);
		
		CHECK(rs.size() == 8000);
		CHECK(rs.rspacing()[0] == 0.5_a);
		CHECK(rs.rspacing()[1] == 0.5_a);	
		CHECK(rs.rspacing()[2] == 0.5_a);
		CHECK(rs.volume_element() == 0.125_a);
	
		hamiltonian::atomic_potential pot(ions.species_list(), rs.gcutoff());
		
		states::ks_states st(states::spin_config::UNPOLARIZED, 11.0);
		
		states::orbital_set<basis::real_space, complex> phi(rs, st.num_states(), 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);
		
		auto bzone = ionic::brillouin(ions, input::kpoints::gamma());
		
		hamiltonian::ks_hamiltonian<double> ham(rs, bzone, st, pot, ions, 0.0);
		
		auto const nst = phi.local_set_size();

		//Constant function
		{
			ham.scalar_potential().fill(0);
			phi.fill(1.0);

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));

			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [nst, hph_r = begin(hphi_r.hypercubic()), hph_f = begin(hphi_f.hypercubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {
													 
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++) {
															 acc_r += fabs(hph_r[ix][iy][iz][ist] - 0.0);
															 acc_f += fabs(hph_f[ix][iy][iz][ist] - 0.0);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();
		
			CHECK(real(diff) < 1e-14);
			CHECK(imag(diff) < 1e-14);
		
		}
		
		//Plane wave
		{
		
			double kk = 2.0*M_PI/rs.rlength()[0];
		
			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [kk, nst, pot = begin(ham.scalar_potential().hypercubic()), ph = begin(phi.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
								 pot[ix][iy][iz][0] = 0.0;

								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
					
								 for(int ist = 0; ist < nst; ist++){
								 
									 auto istg = set_part.local_to_global(ist);
									 double xx = point_op.rvector_cartesian(ixg, iyg, izg)[0];
									 ph[ix][iy][iz][ist] = complex(cos(istg.value()*kk*xx), sin(istg.value()*kk*xx));
								 }
							 });

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [kk, nst, ph = begin(phi.hypercubic()), hph_r = begin(hphi_r.hypercubic()), hph_f = begin(hphi_f.hypercubic()), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
													 
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++){
															 auto istg = set_part.local_to_global(ist);
															 acc_r += fabs(hph_r[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*ph[ix][iy][iz][ist]);
															 acc_f += fabs(hph_f[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*ph[ix][iy][iz][ist]);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();

			CHECK(real(diff) < 1e-14);
			CHECK(imag(diff) < 1e-14);
		
		}

		//Harmonic oscillator
		{
			double ww = 2.0;

			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [ww, nst, pot = begin(ham.scalar_potential().hypercubic()), ph = begin(phi.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {

								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
							 
								 double r2 = point_op.r2(ixg, iyg, izg);
								 pot[ix][iy][iz][0] = 0.5*ww*ww*r2;
							 
								 for(int ist = 0; ist < nst; ist++) ph[ix][iy][iz][ist] = exp(-ww*r2);
							 });

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [ww, nst, ph = begin(phi.hypercubic()), hph_r = begin(hphi_r.hypercubic()), hph_f = begin(hphi_f.hypercubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++)	{
															 acc_r += fabs(hph_r[ix][iy][iz][ist] - 1.5*ww*ph[ix][iy][iz][ist]);
															 acc_f += fabs(hph_f[ix][iy][iz][ist] - 1.5*ww*ph[ix][iy][iz][ist]);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();
		
			CHECK(real(diff) == 0.0051420503_a);
			CHECK(imag(diff) == 0.0051420503_a);
		}
	}
	
	SECTION("PAW Overlap operator"){

		auto cell = systems::cell::cubic(30.0_b).finite();
		systems::ions ions(cell);
		ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {0.0_b, 0.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {2.0_b, 0.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {0.0_b, 2.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {0.0_b, 0.0_b, 2.0_b});
		ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {0.0_b, 0.0_b, -10.0_b});
		systems::electrons electrons(ions, input::kpoints::gamma(), options::electrons{}.cutoff(300.0_Ha).extra_states(0));

		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.brillouin_zone(), electrons.states(), electrons.atomic_pot(), ions, 0.0);

		auto overlap_operator = [&ham](auto const & phi ){
			return ham.overlap(phi);
		};

		for(auto & phi : electrons.kpin()) {
			phi.fill(1.0);

			auto sphi = overlap_operator(phi);
			operations::shift(-1.0, sphi, phi);
			auto olap = operations::overlap(phi);
			auto olap_array = matrix::all_gather(olap);

			CHECK(real(olap_array[0][0]) == Approx(0.97031883));
			CHECK(imag(olap_array[0][0]) == 0.0);
		}
	}
	}
#endif
