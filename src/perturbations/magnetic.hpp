/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__MAGNET
#define INQ__PERTURBATIONS__MAGNET

#include <inq_config.h>
#include <magnitude/magfield.hpp>

namespace inq {
namespace perturbations {

class magnetic : public none {

	vector3<double> magnetic_vector_;
		
public:

	magnetic(vector3<quantity<magnitude::magfield>> const & value):
		magnetic_vector_(vector3<double> {value[0].in_atomic_units(), value[1].in_atomic_units(), value[2].in_atomic_units()}) {
	}

	auto has_magnetic_field() const {
		return true;
	}

	template<typename MagneticField>
	void magnetic_field(const double time, MagneticField & magnetic) const {
		gpu::run(magnetic.basis().local_size(),
						 [magnetic_ = begin(magnetic.linear()), mv = magnetic_vector_] GPU_LAMBDA (auto ip){
							 magnetic_[ip] += mv;
						 });
	}

	template<class OStream>
	friend OStream & operator<<(OStream & out, magnetic const & self){
		return out;
	}

};

}
}
#endif

#ifdef INQ_PERTURBATIONS_MAGNETIC_UNIT_TEST
#undef INQ_PERTURBATIONS_MAGNETIC_UNIT_TEST

#include <catch2/catch_all.hpp>
using namespace inq;
using Catch::Approx;

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	if(comm.size() > 2) return;

	vector3<quantity<magnitude::magfield>> bvec = {0.0_amu, 0.0_amu, 1.0_amu};
	perturbations::magnetic uniform_magnetic{bvec};

	basis::real_space bas(systems::cell::cubic(5.0_b), /*spacing*/ 0.1, comm);
	basis::field<basis::real_space, vector3<double>> mag_field(bas);
	mag_field.fill(vector3<double> {0.0, 0.0, 0.0});

	CHECK(uniform_magnetic.has_magnetic_field());
	uniform_magnetic.magnetic_field(/*time*/ 0.0, mag_field);
	CHECK(mag_field.linear()[0]			== vector3<double>{0.0, 0.0, 1.0});
	CHECK(mag_field.linear()[1]			== vector3<double>{0.0, 0.0, 1.0});

	uniform_magnetic.magnetic_field(/*time*/ 1000.0, mag_field);
	CHECK(mag_field.linear()[0]			== vector3<double>{0.0, 0.0, 2.0});

	auto par = input::parallelization(comm);
	auto ions = systems::ions(systems::cell::cubic(10.0_b));
	ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
		
	SECTION("Hydrogen atom perturbation collinear calculation") {
		auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(5).spin_polarized());
		ground_state::initial_guess(ions, electrons);
		bvec = {0.0_beV, 0.0_beV, 0.1_beV};
		perturbations::magnetic B{bvec};
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(200).mixing(0.1), B);
		CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 0.1);

		bvec ={0.0_beV, 0.0_beV, 1.0_beV};
		perturbations::magnetic B2{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(200).mixing(0.1), B2);
		CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 1.0);

		bvec ={0.0_T, 0.0_T, 17255.974545750545_T};
		perturbations::magnetic B3{bvec};
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(200).mixing(0.1), B3);
		CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 1.0);
	}

    SECTION("Hydrogen atom perturbation non collinear calculation") {
        auto electrons = systems::electrons(par, ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_non_collinear());
        std::vector<vector3<double>> initial_magnetization = {{0.0, 0.0, 1.0}};
        ground_state::initial_guess(ions, electrons, initial_magnetization);
        bvec = {0.0_beV, 0.0_beV, 0.1_beV};
        perturbations::magnetic B(bvec);
        auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(1000).mixing(0.1), B);
        CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 0.1);

		bvec = {0.0_beV, 0.0_beV, 0.5_beV};
		perturbations::magnetic B2(bvec);
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(1000).mixing(0.1), B2);
		CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 0.5);

		bvec = {0.0_T, 0.0_T, 17255.974545750545_T};
		perturbations::magnetic B3(bvec);
		result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(1000).mixing(0.1), B3);
		CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 1.0);

        bvec = {0.1_beV, 0.0_beV, 0.0_beV};
        initial_magnetization = {{1.0, 0.0, 0.0}};
        ground_state::initial_guess(ions, electrons, initial_magnetization);
        perturbations::magnetic B4(bvec);
        result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-9_Ha).max_steps(1000).mixing(0.1), B4);
        CHECK(Approx(fabs(result.energy.zeeman_energy())*27.2114).margin(1.e-4) == 0.1);
    }
}
#endif
