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

template<class occupations_array_type, class field_set_type, typename VZType, typename RFType>
void compute_psi_vz_psi_ofr(occupations_array_type const & occupations, field_set_type const & phi, VZType const & zeeman_pot, RFType & rfield) {

	assert(get<1>(sizes(phi.spinor_array())) == phi.spinor_dim());
	assert(get<2>(sizes(phi.spinor_array())) == phi.local_spinor_set_size());

	if (zeeman_pot.set_size() == 2){
		gpu::run(phi.local_set_size(), phi.basis().local_size(),
						 [ph = begin(phi.matrix()), rf = begin(rfield.linear()), vz = begin(zeeman_pot.matrix()), occ = begin(occupations), spi = phi.spin_index()] GPU_LAMBDA (auto ist, auto ip) {
							 rf[ip] += occ[ist]*vz[ip][spi]*norm(ph[ip][ist]);
						 });
	}
	else {
		assert(zeeman_pot.set_size() == 4);
		gpu::run(phi.local_spinor_set_size(), phi.basis().local_size(),
						 [ph = begin(phi.spinor_array()), rf = begin(rfield.linear()), vz = begin(zeeman_pot.matrix()), occ = begin(occupations)] GPU_LAMBDA (auto ist, auto ip) {
							 auto offdiag = vz[ip][2] + complex{0.0, 1.0}*vz[ip][3];
							 auto cross = 2.0*occ[ist]*real(offdiag*conj(ph[ip][1][ist])*ph[ip][0][ist]);
							 rf[ip] += occ[ist]*vz[ip][0]*norm(ph[ip][0][ist]);
							 rf[ip] += occ[ist]*vz[ip][1]*norm(ph[ip][1][ist]);
							 rf[ip] += cross;
						 });
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

template<class CommType, typename SpinDensityType, typename MagneticField, class occupations_array_type, class kpin_type>
void eval_psi_vz_psi(CommType & comm, SpinDensityType const & spin_density, MagneticField const & magnetic_field, occupations_array_type const & occupations, kpin_type const & kpin, double & zeeman_ener) {
				
	basis::field_set<basis::real_space, double> zeeman_pot(spin_density.skeleton());
	zeeman_pot.fill(0.0);
	hamiltonian::zeeman_coupling zc_(spin_density.set_size());
	zc_.compute_zeeman_potential(magnetic_field, zeeman_pot);

	basis::field<basis::real_space, double> rfield(zeeman_pot.basis());
	rfield.fill(0.0);
	int iphi = 0;
	for (auto & phi : kpin) {
		compute_psi_vz_psi_ofr(occupations[iphi], phi, zeeman_pot, rfield);
		iphi++;
	}

	rfield.all_reduce(comm);
	zeeman_ener += operations::integral(rfield);
}

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

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
		auto zeeman_ener = result.energy.zeeman_energy();
		Approx target = Approx(zeeman_ener).epsilon(1.e-10);

		basis::field<basis::real_space, vector3<double>> mag_field(electrons.spin_density().basis());
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform.magnetic_field(0.0, mag_field);
		auto zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target);
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
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)																== -1.0);

		auto zeeman_ener = result.energy.zeeman_energy();
		Approx target = Approx(zeeman_ener).epsilon(1.e-10);
		basis::field<basis::real_space, vector3<double>> mag_field(electrons.spin_density().basis());
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform.magnetic_field(0.0, mag_field);
		auto zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target);

		vector3 bvec = {1.0_amu/sqrt(2.0), 1.0_amu/sqrt(2.0), 0.0_amu};
		perturbations::magnetic magnetic_uniform2{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform2);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 0.0);
				
		zeeman_ener = result.energy.zeeman_energy();
		Approx target2 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform2.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target2);

		bvec = {1.0_amu/sqrt(2.0), -1.0_amu/sqrt(2.0), 0.0_amu};
		perturbations::magnetic magnetic_uniform3{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform3);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 0.0);
				
		zeeman_ener = result.energy.zeeman_energy();
		Approx target3 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform3.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target3);
				
		bvec = {1.0_amu/sqrt(3.0), 1.0_amu/sqrt(3.0), 1.0_amu/sqrt(3.0)};
		perturbations::magnetic magnetic_uniform4{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform4);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(3.0));
				
		zeeman_ener = result.energy.zeeman_energy();
		Approx target4 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform4.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target4);

		bvec = {0.0_amu, -1.0_amu/sqrt(2.0), 1.0_amu/sqrt(2.0)};
		perturbations::magnetic magnetic_uniform5{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform5);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 0.0);
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-1.0/sqrt(2.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(2.0));

		zeeman_ener = result.energy.zeeman_energy();
		Approx target5 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform5.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target5);
				
		bvec = {1.0_amu/sqrt(1.0+4.0+9.0/4), -2.0_amu/sqrt(1.0+4.0+9.0/4), 1.5_amu/sqrt(1.0+4.0+9.0/4)};
		perturbations::magnetic magnetic_uniform6{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform6);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 1.0/sqrt(1.0+4.0+9.0/4));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-2.0/sqrt(1.0+4.0+9.0/4));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.5/sqrt(1.0+4.0+9.0/4));

		zeeman_ener = result.energy.zeeman_energy();
		Approx target6 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform6.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target6);
				
		bvec = {4.0e+05_T/sqrt(16.0+4.0+1.0), -2.0e+05_T/sqrt(16.0+4.0+1.0), 1.0e+05_T/sqrt(16.0+4.0+1.0)};
		perturbations::magnetic magnetic_uniform7{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(200).mixing(0.1), magnetic_uniform7);
		mag = observables::total_magnetization(electrons.spin_density());
		CHECK(Approx(mag[0]/mag.length()).margin(1.e-7)		== 4.0/sqrt(16.0+4.0+1.0));
		CHECK(Approx(mag[1]/mag.length()).margin(1.e-7)		==-2.0/sqrt(16.0+4.0+1.0));
		CHECK(Approx(mag[2]/mag.length()).margin(1.e-7)		== 1.0/sqrt(16.0+4.0+1.0));

		zeeman_ener = result.energy.zeeman_energy();
		Approx target7 = Approx(zeeman_ener).epsilon(1.e-10);
		mag_field.fill(vector3 {0.0, 0.0, 0.0});
		magnetic_uniform7.magnetic_field(0.0, mag_field);
		zeeman_ener2 = 0.0;
		eval_psi_vz_psi(electrons.kpin_states_comm(), electrons.spin_density(), mag_field, electrons.occupations(), electrons.kpin(), zeeman_ener2);
		CHECK(zeeman_ener2 == target7);
				
	}
}
#endif
