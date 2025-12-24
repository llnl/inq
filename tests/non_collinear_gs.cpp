/* -*- indent-tabs-mode: t -*- */

#include <inq/inq.hpp>

using namespace inq;
using namespace inq::magnitude;

inq::utils::match match(3.0e-4);

int main (int argc, char ** argv) {
	auto env = input::environment{};

	auto ions = systems::ions(systems::cell::cubic(10.0_b));
	ions.insert("H", {0.0_b, 0.0_b, 0.0_b});
	std::vector<vector3<double>> initial_magnetization = {{0.0, 0.0, 1.0}};

	auto electrons = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_polarized());
	ground_state::initial_guess(ions, electrons, initial_magnetization);
	auto result_col = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(1000));
	auto mag = observables::total_magnetization(electrons.spin_density());

	auto electrons_nc = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_non_collinear());
	ground_state::initial_guess(ions, electrons_nc, initial_magnetization);
	auto result_nc = ground_state::calculate(ions, electrons_nc, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(1000).mixing(0.1));
	auto mag_nc = observables::total_magnetization(electrons_nc.spin_density());

	match.check("total energy",								result_col.energy.total(),	 result_nc.energy.total());
	match.check("nvxc",                                     result_col.energy.nvxc(),    result_nc.energy.nvxc());
	match.check("total magnetization",				        mag.length(),				 mag_nc.length());

	auto d = 1.21_A;
	ions = systems::ions(systems::cell::cubic(10.0_b));
	ions.insert("O", {0.0_b, 0.0_b, d/2});
	ions.insert("O", {0.0_b, 0.0_b,-d/2});
	initial_magnetization = {{0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}};

	electrons = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_polarized());
    ground_state::initial_guess(ions, electrons, initial_magnetization);
    result_col = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(1000));
    mag = observables::total_magnetization(electrons.spin_density());

    electrons_nc = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(2).spin_non_collinear());
    ground_state::initial_guess(ions, electrons_nc, initial_magnetization);
    result_nc = ground_state::calculate(ions, electrons_nc, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(10000).mixing(0.1));
    mag_nc = observables::total_magnetization(electrons_nc.spin_density());

    match.check("total energy",               result_col.energy.total(),   result_nc.energy.total());
    match.check("nvxc",                       result_col.energy.nvxc(),    result_nc.energy.nvxc());
    match.check("total magnetization",        mag.length(),                mag_nc.length());

	return match.fail();
}
