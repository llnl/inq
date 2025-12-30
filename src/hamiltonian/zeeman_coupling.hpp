/* -*- indent-tabs-mode: t -*- */

#ifndef ZEEMAN_COUPLING_HPP
#define ZEEMAN_COUPLING_HPP

#include <inq_config.h>

namespace inq {
namespace hamiltonian {

class zeeman_coupling {

private:

	int spin_components_;

public:

	double ge = 2.00231930436256;

	zeeman_coupling(int const spin_components):
		spin_components_(spin_components) {
		assert(spin_components_ > 1);
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template<typename SpinDensityType, typename VKSType>
	void operator()(SpinDensityType const & spin_density, basis::field<basis::real_space, vector3<double>> const & magnetic_field, VKSType & vks, double & zeeman_ener) const {

		basis::field_set<basis::real_space, double> zeeman_pot(vks.skeleton());
		zeeman_pot.fill(0.0);

		assert(zeeman_pot.set_size() == spin_components_);

		compute_zeeman_potential(magnetic_field, zeeman_pot);

		gpu::run(zeeman_pot.local_set_size(), zeeman_pot.basis().local_size(),
						 [vz = begin(zeeman_pot.matrix()), vk = begin(vks.matrix())] GPU_LAMBDA (auto is, auto ip) {
							 vk[ip][is] += vz[ip][is];
						 });

		zeeman_ener += compute_zeeman_energy(spin_density, zeeman_pot);
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template<typename VZType>
	void compute_zeeman_potential(basis::field<basis::real_space, vector3<double>> const & magnetic_field, VZType & zeeman_pot) const {

		gpu::run(zeeman_pot.basis().local_size(),
						 [vz = begin(zeeman_pot.matrix()), magnetic_ = begin(magnetic_field.linear()), ge_=ge] GPU_LAMBDA (auto ip) {
							 vz[ip][0] +=-0.5*ge_*magnetic_[ip][2]/2.0;
							 vz[ip][1] += 0.5*ge_*magnetic_[ip][2]/2.0;
						 });
		if (zeeman_pot.set_size() == 4) {
			gpu::run(zeeman_pot.basis().local_size(),
							 [vz = begin(zeeman_pot.matrix()), magnetic_ = begin(magnetic_field.linear()), ge_=ge] GPU_LAMBDA (auto ip) {
								 vz[ip][2] +=-0.5*ge_*magnetic_[ip][0]/2.0;
								 vz[ip][3] +=-0.5*ge_*magnetic_[ip][1]/2.0;
							 });
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename SpinDensityType, typename VZType>
	double compute_zeeman_energy(SpinDensityType const & spin_density, VZType & zeeman_pot) const {

		auto zeeman_ener_ = 0.0;
		if (spin_density.set_size() == 4) {
			gpu::run(spin_density.local_set_size(), spin_density.basis().local_size(),
							 [vz = begin(zeeman_pot.matrix())] GPU_LAMBDA (auto is, auto ip) {
								 if (is == 2) { 
									 vz[ip][is] = 2.0*vz[ip][is];
								 }
								 else if (is == 3) {
									 vz[ip][is] = -2.0*vz[ip][is];
								 }
							 });
		}
		zeeman_ener_ += operations::integral_product_sum(spin_density, zeeman_pot);
		return zeeman_ener_;
	}
};

}
}
#endif

#ifdef INQ_HAMILTONIAN_ZEEMAN_COUPLING_UNIT_TEST
#undef INQ_HAMILTONIAN_ZEEMAN_COUPLING_UNIT_TEST

#include <perturbations/magnetic.hpp>
#include <catch2/catch_all.hpp>
using namespace inq;

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	if(comm.size() > 2) return;
	
	SECTION("Spin polarized zeeman calculation") {
		auto par = input::parallelization(comm);
		auto ions = systems::ions(systems::cell::cubic(10.0_b));
		ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_polarized());
		ground_state::initial_guess(ions, electrons);
		perturbations::magnetic magnetic_uniform{{0.0_amu, 0.0_amu, -1.0_amu}};
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform);
		auto mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 0.0);
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		== 0.0);
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		==-1.0);
	}

	SECTION("Spin non collinear zeeman calculation") {
		auto par = input::parallelization(comm);
		auto ions = systems::ions(systems::cell::cubic(10.0_b));
		ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_non_collinear());
		ground_state::initial_guess(ions, electrons);
		perturbations::magnetic magnetic_uniform{{0.0_amu, 0.0_amu, -1.0_amu}};

		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform);
		auto mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(sqrt(mag[0]*mag[0]+mag[1]*mag[1])/mag.length()).margin(1.e-7)		== 0.0);
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)	                                ==-1.0);

		auto zeeman_ener = result.energy.zeeman_energy();
		Approx target = Approx(zeeman_ener).epsilon(1.e-10);

		vector3 bvec = {1.0_amu/sqrt(2.0), 1.0_amu/sqrt(2.0), 0.0_amu};
		perturbations::magnetic magnetic_uniform2{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform2);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 0.0);
				
		auto zeeman_ener2 = result.energy.zeeman_energy();
		CHECK(zeeman_ener2 == target);

		bvec = {1.0_amu/sqrt(2.0), -1.0_amu/sqrt(2.0), 0.0_amu};
		perturbations::magnetic magnetic_uniform3{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform3);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 0.0);
				
		zeeman_ener2 = result.energy.zeeman_energy();
		CHECK(zeeman_ener2 == target);
		
		bvec = {1.0_amu/sqrt(3.0), 1.0_amu/sqrt(3.0), 1.0_amu/sqrt(3.0)};
		perturbations::magnetic magnetic_uniform4{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform4);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
				
		zeeman_ener2 = result.energy.zeeman_energy();
		CHECK(zeeman_ener2 == target);

		bvec = {0.0_amu, -1.0_amu/sqrt(2.0), 1.0_amu/sqrt(2.0)};
		perturbations::magnetic magnetic_uniform5{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform5);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 0.0);
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));

		zeeman_ener2 = result.energy.zeeman_energy();
		CHECK(zeeman_ener2 == target);
		
		bvec = {1.0_amu/sqrt(1.0+4.0+9.0/4), -2.0_amu/sqrt(1.0+4.0+9.0/4), 1.5_amu/sqrt(1.0+4.0+9.0/4)};
		perturbations::magnetic magnetic_uniform6{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform6);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(1.0+4.0+9.0/4));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-2.0/sqrt(1.0+4.0+9.0/4));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.5/sqrt(1.0+4.0+9.0/4));

		zeeman_ener2 = result.energy.zeeman_energy();
		CHECK(zeeman_ener2 == target);
				
		bvec = {4.0e+05_T/sqrt(16.0+4.0+1.0), -2.0e+05_T/sqrt(16.0+4.0+1.0), 1.0e+05_T/sqrt(16.0+4.0+1.0)};
		perturbations::magnetic magnetic_uniform7{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform7);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 4.0/sqrt(16.0+4.0+1.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-2.0/sqrt(16.0+4.0+1.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(16.0+4.0+1.0));
				
	}
}
#endif
