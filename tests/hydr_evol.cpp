/* -*- indent-tabs-mode: t -*- */

#include <inq/inq.hpp>
#include <perturbations/magnetic.hpp>
#include <perturbations/magnetic_pulse.hpp>

using namespace inq;
using namespace inq::magnitude;

inq::utils::match match(3.0e-4);

int main (int argc, char ** argv){
    auto env = input::environment{};
    vector3<quantity<magnitude::magfield>> b = {0.0_amu, 0.0_amu, 1.0_amu};
    perturbations::magnetic B{b};

    auto ions = systems::ions(systems::cell::cubic(10.0_b));
    ions.insert("H", {0.0_b, 0.0_b, 0.0_b});

    {
        auto electrons = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(10).spin_non_collinear());
        std::vector<vector3<double>> initial_magnetization = {{0.0, 0.0, 1.0}};
        ground_state::initial_guess(ions, electrons, initial_magnetization);

        auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(1000).mixing(0.1), B);
        auto mag = observables::total_magnetization(electrons.spin_density());
        match.check("magnetization direction",        mag[2],         1.0);
        
        std::vector<double> mz;
        std::vector<double> mx;
        std::vector<double> my;
        auto output = [&](auto data){
            mz.push_back(data.magnetization()[2]);
            mx.push_back(data.magnetization()[0]);
            my.push_back(data.magnetization()[1]);
        };

        vector3<quantity<magnitude::magfield>> b2 = {3.0_amu, 0.0_amu, 0.0_amu};
        perturbations::magnetic B2{b2};
        real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(100).dt(0.03_atomictime), B2);

        match.check("magnetization module: ", (mx[10]*mx[10] + my[10]*my[10] + mz[10]*mz[10]), 1.0);
        match.check("magnetization module: ", (mx[30]*mx[30] + my[30]*my[30] + mz[30]*mz[30]), 1.0);
        match.check("magnetization module: ", (mx[50]*mx[50] + my[50]*my[50] + mz[50]*mz[50]), 1.0);
        match.check("magnetization module: ", (mx[70]*mx[70] + my[70]*my[70] + mz[70]*mz[70]), 1.0);
        match.check("magnetization module: ", (mx[85]*mx[85] + my[85]*my[85] + mz[85]*mz[85]), 1.0);
        match.check("magnetization module: ", (mx[99]*mx[99] + my[99]*my[99] + mz[99]*mz[99]), 1.0);
    }

    if (match.fail()) return match.fail();
    
    {
        auto electrons = systems::electrons(env.par(), ions, options::electrons{}.cutoff(30.0_Ha).extra_states(10).spin_non_collinear());
        std::vector<vector3<double>> initial_magnetization = {{0.0, 0.0, 1.0}};
        ground_state::initial_guess(ions, electrons, initial_magnetization);

        auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), inq::options::ground_state{}.steepest_descent().energy_tolerance(1.e-8_Ha).max_steps(1000).mixing(0.1), B);
        auto mag = observables::total_magnetization(electrons.spin_density());
        match.check("magnetization direction",      mag[2],        1.0);
        

        std::vector<double> mz;
        std::vector<double> mx;
        std::vector<double> my;
        auto output = [&](auto data){
            mz.push_back(data.magnetization()[2]);
            mx.push_back(data.magnetization()[0]);
            my.push_back(data.magnetization()[1]);
        };
        
        vector3<quantity<magnitude::magfield>> Db = {6.0_amu, 0.0_amu, -6.0_amu};
        vector3<quantity<magnitude::magfield>> b2 = {1.0_amu, 0.0_amu, -1.0_amu};
        perturbations::magnetic_pulse B2{b, Db, b2, 10.0_ev, 1000.0_attosecond, 1.0_invfs2, 100.0_attosecond, 0.0};
        real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(100).dt(0.03_atomictime), B2);
        
        match.check("magnetization module: ", (mx[10]*mx[10] + my[10]*my[10] + mz[10]*mz[10]), 1.0);
        match.check("magnetization module: ", (mx[30]*mx[30] + my[30]*my[30] + mz[30]*mz[30]), 1.0);
        match.check("magnetization module: ", (mx[50]*mx[50] + my[50]*my[50] + mz[50]*mz[50]), 1.0);
        match.check("magnetization module: ", (mx[70]*mx[70] + my[70]*my[70] + mz[70]*mz[70]), 1.0);
        match.check("magnetization module: ", (mx[85]*mx[85] + my[85]*my[85] + mz[85]*mz[85]), 1.0);
        match.check("magnetization module: ", (mx[99]*mx[99] + my[99]*my[99] + mz[99]*mz[99]), 1.0);
    }
    
    if (match.fail()) return match.fail();
}