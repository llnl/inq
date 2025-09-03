/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPERATIONS__SPATIAL_PARTITIONS
#define INQ__OPERATIONS__SPATIAL_PARTITIONS

#include <inq_config.h>
#include <cfloat>
#include <operations/sum.hpp>

namespace inq {
namespace operations {

template <typename CellType>
basis::field<basis::real_space, int> voronoi_field(std::vector<vector3<double, cartesian>> const & local_centers, CellType const & cell, basis::real_space const & bas) {
    auto nloc = static_cast<int>(local_centers.size());
    assert(nloc > 0);

    basis::field<basis::real_space, int> local_field(bas);
    local_field.fill(0);
    auto range = 0.0;
    for (auto idir = 0; idir < cell.periodicity(); idir++) {
        auto L = sqrt(norm(cell[idir]));
        if (L > range) range = L;
    }
    std::vector<std::vector<vector3<double>>> replicas;
    for (auto ii = 0; ii < nloc; ii++) {
        auto rep = inq::ionic::periodic_replicas(cell, local_centers[ii], range);
        std::vector<vector3<double>> rep_;
        for (unsigned irep = 0; irep < rep.size(); irep++) {
            rep_.push_back(rep[irep]);
        }
        replicas.push_back(rep_);
    }

    gpu::run(bas.local_sizes()[2], bas.local_sizes()[1], bas.local_sizes()[0],
        [ph = begin(local_field.cubic()), point_op = bas.point_op(), rep = replicas.begin(), nloc, cell] GPU_LAMBDA (auto iz, auto iy, auto ix){
            auto rr = point_op.rvector_cartesian(ix, iy, iz);
            auto ci = -1;
            auto dd = DBL_MAX;
            for (auto ii = 0; ii < nloc; ii++) {
                auto dd2 = DBL_MAX;
                for (auto irep = 0; irep < rep[ii].size(); irep++) {
                    auto dd3 = cell.metric().distance(rr, rep[ii][irep]);
                    if (dd3 < dd2) dd2 = dd3;
                }
                if (dd2 < dd) {
                    dd = dd2;
                    ci = ii;
                }
            }
            ph[ix][iy][iz] = ci;
        });

    return local_field;
}

template <typename CellType>
basis::field_set<basis::real_space, int> local_radii_field(std::vector<vector3<double, cartesian>> const & local_centers, std::vector<double> const & local_radii, CellType const & cell, basis::real_space const & bas){
    auto nloc = static_cast<int>(local_centers.size());
    assert(nloc == local_radii.size());

    basis::field_set<basis::real_space, int> local_field(bas, nloc);
    local_field.fill(0);
    auto range = 0.0;
    for (auto idir = 0; idir < cell.periodicity(); idir++) {
        auto L = sqrt(norm(cell[idir]));
        if (L > range) range = L;
    }
    std::vector<std::vector<vector3<double>>> replicas;
    for (auto ii = 0; ii < nloc; ii++) {
        auto rep = inq::ionic::periodic_replicas(cell, local_centers[ii], range);
        std::vector<vector3<double>> rep_;
        for (unsigned irep = 0; irep < rep.size(); irep++) {
            rep_.push_back(rep[irep]);
        }
        replicas.push_back(rep_);
    }

    gpu::run(bas.local_sizes()[2], bas.local_sizes()[1], bas.local_sizes()[0],
        [ph = begin(local_field.hypercubic()), point_op = bas.point_op(), rd = local_radii.begin(), rep = replicas.begin(), nloc, cell] GPU_LAMBDA (auto iz, auto iy, auto ix){
            auto rr = point_op.rvector_cartesian(ix, iy, iz);
            for (auto ii = 0; ii < nloc; ii++) {
                auto dd = DBL_MAX;
                for (auto irep = 0; irep < rep[ii].size(); irep++) {
                    auto dd2 = cell.metric().distance(rr, rep[ii][irep]);
                    if (dd2 < dd) dd = dd2;
                }
                if (dd < rd[ii]) ph[ix][iy][iz][ii] = 1;
            }
        });
    return local_field;
}

}
}
#endif

#ifdef INQ_OPERATIONS_SPATIAL_PARTITIONS_UNIT_TEST
#undef INQ_OPERATIONS_SPATIAL_PARTITIONS_UNIT_TEST

using namespace inq;

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

    parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

    basis::real_space bas(systems::cell::cubic(5.0_b), /*spacing*/ 0.1, comm);
    basis::field<basis::real_space, int> voronoi(bas);

    CHECK(voronoi.size() == 50*50*50);

    auto cell = systems::cell::cubic(15.0_b).finite();
    basis::real_space bas2(cell, /*spacing*/ 0.1, comm);
    auto ions = systems::ions(cell);
    ions.insert("H", {0.0_b, 0.0_b, 0.0_b});

    std::vector<vector3<double, cartesian>> local_centers;
    local_centers.push_back(ions.positions()[0]);
    voronoi = operations::voronoi_field(local_centers, cell, bas2);
    CHECK(voronoi.size() == 150*150*150);
    CHECK(voronoi.linear()[0] == 0);
    CHECK(voronoi.linear()[1] == 0);
    CHECK(voronoi.linear()[2] == 0);
    CHECK(operations::sum(voronoi.linear()) == 0);

    ions.insert("H", {0.0_b, 0.0_b, 1.0_b});
    local_centers.push_back(ions.positions()[1]);
    voronoi = operations::voronoi_field(local_centers, cell, bas2);
    CHECK(voronoi.linear()[0] == 0);
    CHECK(voronoi.linear()[1] == 0);
    CHECK(voronoi.linear()[2] == 0);

    std::vector<double> local_radii = {0.0, 0.0};
    auto local_field = operations::local_radii_field(local_centers, local_radii, cell, bas2);
    CHECK(local_field.matrix()[0][0] == 0);
    CHECK(local_field.matrix()[1][0] == 0);
    CHECK(local_field.matrix()[0][1] == 0);
    CHECK(local_field.matrix()[1][1] == 0);

    local_radii = {100.0, 0.0};
    local_field = operations::local_radii_field(local_centers, local_radii, cell, bas2);
    CHECK(local_field.matrix()[0][0] == 1);
    CHECK(local_field.matrix()[1][0] == 1);
    CHECK(local_field.matrix()[10][0] == 1);
    CHECK(local_field.matrix()[20][0] == 1);
    CHECK(local_field.matrix()[0][1] == 0);
    CHECK(local_field.matrix()[1][1] == 0);
}
#endif