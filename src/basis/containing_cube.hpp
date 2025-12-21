/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__CONTAINING_CUBE
#define INQ__BASIS__CONTAINING_CUBE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>

namespace inq {
namespace basis {

//returns the cube that contains the sphere, this makes the initialization O(1) instead of O(N)
void containing_cube(basis::real_space const & grid, vector3<double> const & pos, double radius, vector3<int> & lo, vector3<int> & hi){

	for(int idir = 0; idir < 3; idir++){
		auto rec = grid.cell().reciprocal(idir);
		auto lointer = pos - radius/length(rec)*rec;
		auto hiinter = pos + radius/length(rec)*rec;

		auto dlo = grid.cell().to_contravariant(lointer)[idir];
		auto dhi = grid.cell().to_contravariant(hiinter)[idir];

		lo[idir] = lround(floor(dlo/grid.contravariant_spacing()[idir]));
		hi[idir] = lround(ceil(dhi/grid.contravariant_spacing()[idir])) + 1;

		#if defined(__cpp_lib_clamp) and  (__cpp_lib_clamp >= 201603L)
		lo[idir] = std::clamp(lo[idir], grid.symmetric_range_begin(idir), grid.symmetric_range_end(idir));
		hi[idir] = std::clamp(hi[idir], grid.symmetric_range_begin(idir), grid.symmetric_range_end(idir));
		#else
		lo[idir] = std::max(lo[idir], grid.symmetric_range_begin(idir));
		lo[idir] = std::min(lo[idir], grid.symmetric_range_end  (idir));

		hi[idir] = std::max(hi[idir], grid.symmetric_range_begin(idir));
		hi[idir] = std::min(hi[idir], grid.symmetric_range_end  (idir));
		#endif
	}
	
}

}
}
#endif

#ifdef INQ_BASIS_CONTAINING_CUBE_UNIT_TEST
#undef INQ_BASIS_CONTAINING_CUBE_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <gpu/array.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	SECTION("Orthogonal box"){
		
		basis::real_space rs(systems::cell::orthorhombic(12.0_b, 14.0_b, 16.0_b), /*spacing =*/ 0.33115294, comm);

		auto center = vector3{3.0, 2.0, 1.0};
		auto radius = 3.0;

		vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == 0);
		CHECK(lo[1] == -3);
		CHECK(lo[2] == -6);
		CHECK(hi[0] == 18);
		CHECK(hi[1] == 16);
		CHECK(hi[2] == 13);		
		
		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto ii = rs.point_op().to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));
					CHECK(dist2 > radius*radius);
				}
			}
		}
		
	}
	
	SECTION("Non-orthogonal box"){

		auto aa = 23.3_b;
		basis::real_space rs(systems::cell::lattice({0.0_b, aa/2.0, aa/2.0}, {aa/2, 0.0_b, aa/2.0}, {aa/2.0, aa/2.0, 0.0_b}),  /*spacing =*/ 0.25650997, comm);

		auto center = vector3{-0.5, 0.666, -1.0};
		auto radius = 4.2;

		vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == -20);
		CHECK(lo[1] == -26);
		CHECK(lo[2] == -17);
		CHECK(hi[0] == 22);
		CHECK(hi[1] == 16);
		CHECK(hi[2] == 25);

		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					
					auto ii = rs.to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));
					CHECK(dist2 > radius*radius);
				}
			}
		}
				
	}

	SECTION("Non-orthogonal box 2"){

		auto aa = 5.5_b;
		basis::real_space rs(systems::cell::lattice({0.0_b, aa/2.0, aa/2.0}, {aa/2, 0.0_b, aa/2.0}, {aa/2.0, aa/2.0, 0.0_b}), /*spacing =*/ 0.21995548, comm);

		auto center = vector3{-0.5, 0.666, -1.0};
		auto radius = 4.2;

		vector3<int> lo, hi;
		containing_cube(rs, center, radius, lo, hi);

		CHECK(lo[0] == -9);
		CHECK(lo[1] == -9);
		CHECK(lo[2] == -9);
		CHECK(hi[0] == 9);
		CHECK(hi[1] == 9);
		CHECK(hi[2] == 9);		
		
		for(int ix = 0; ix < rs.sizes()[0]; ix++){
			for(int iy = 0; iy < rs.sizes()[1]; iy++){
				for(int iz = 0; iz < rs.sizes()[2]; iz++){
					auto ii = rs.point_op().to_symmetric_range(ix, iy, iz);
					
					if(ii[0] >= lo[0] and ii[0] < hi[0] and
						 ii[1] >= lo[1] and ii[1] < hi[1] and
						 ii[2] >= lo[2] and ii[2] < hi[2]) continue;

					auto dist2 = norm(center - rs.point_op().rvector_cartesian(parallel::global_index(ix), parallel::global_index(iy), parallel::global_index(iz)));					
					CHECK(dist2 > radius*radius);
				}
			}
		}
		
	}

}
#endif

