/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MAGNITUDE__MAGFIELD
#define INQ__MAGNITUDE__MAGFIELD

#include <utils/lowercase.hpp>

namespace inq {
namespace magnitude {

struct magfield;

auto operator ""_amu(long double val){
    return inq::quantity<magfield>::from_atomic_units(val);
}

auto operator ""_AMU(long double val){
    return inq::quantity<magfield>::from_atomic_units(val);
}

auto operator ""_tesla(long double val){
    return inq::quantity<magfield>::from_atomic_units(val/2.3505e+05);
}

auto operator ""_T(long double val){
    return inq::quantity<magfield>::from_atomic_units(val/2.3505e+05);
}

auto operator ""_beV(long double val){
    auto muB = 5.7883818060E-5;     // eV T^-1
    auto ge = 2.00231930436256;
    auto val2 = 2.0*val/(ge*muB);
    return inq::quantity<magfield>::from_atomic_units(val2/2.3505e+05);
}

struct magfield {
    
    static auto parse(double value, std::string units){

        units = utils::lowercase(units);

        if (units == "AU" or units == "au") {
            return value*1.0_AMU;
        } else if (units == "tesla" or units == "T") {
            return value*1.0_T;
        } else {
            throw std::runtime_error("inq error: unknown magnetic field units '" + units + "'.");
        }
    }
};



}
}
#endif

#ifdef INQ_MAGNITUDE_MAGFIELD_UNIT_TEST
#undef INQ_MAGNITUDE_MAGFIELD_UNIT_TEST

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    using namespace inq;
    using namespace magnitude;
    using Catch::Approx;

    auto ge = 2.00231930436256;
    auto muB = 5.78883942514688e-05;

    {
        auto mf = 100.0_AMU;
        CHECK(mf.in_atomic_units() == 100.0);
    }

    {
        auto mf = 10.0_amu;
        CHECK(mf.in_atomic_units() == 10.0);
    }

    {
        auto mf = 2.3505e+05_tesla;
        CHECK(Approx(mf.in_atomic_units()).margin(1.e-4) == 1.0);
    }

    {
        auto mf = 6.0e+05_T;
        CHECK(Approx(mf.in_atomic_units()).margin(1.e-4) == 2.55265);
    }

    {
        auto mf = 1.0_amu;
        CHECK(Approx((mf.in_atomic_units()*2.3505e+05*muB*1.0_ev).in_atomic_units()).margin(1.e-4) == 0.5);
    }

    {
        auto mf = 1.0_beV;
        CHECK(Approx(mf.in_atomic_units()*2.3505e+05*ge*muB/2.0).margin(1.e-4) == 1.0);
    }

    {
        auto mf = 0.1_beV;
        CHECK(Approx(mf.in_atomic_units()*2.3505e+05*ge*muB/2.0).margin(1.e-4) == 0.1);
    }

    {
        auto mf = 0.01_beV;
        CHECK(Approx(mf.in_atomic_units()*2.3505e+05*ge*muB/2.0).margin(1.e-4) == 0.01);
    }

}

#endif