/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATRIX__DISTRIBUTED
#define INQ__MATRIX__DISTRIBUTED

// Copyright (C) 2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gpu/array.hpp>
#include <gpu/run.hpp>
#include <parallel/partition.hpp>
#include <parallel/communicator.hpp>

namespace inq {
namespace matrix {

template <class Type>
class distributed {

  using array_type = gpu::array<Type, 2>;

  mutable parallel::cartesian_communicator<2> comm_;
	parallel::partition partx_;
	parallel::partition party_;	
  array_type block_;
	
public:
	
	using element_type = Type;
	
  distributed(parallel::cartesian_communicator<2> comm, long sizex, long sizey):
    comm_(std::move(comm)),
		partx_(sizex, comm_.axis(0)),
		party_(sizey, comm_.axis(1)),
		block_({partx_.local_size(), party_.local_size()}) {
  }

  auto & block() const {
    return block_;
  }

  auto & block() {
    return block_;
  }

  auto & comm() const {
    return comm_;
  }

	auto is_local(parallel::global_index ix, parallel::global_index iy) const {
		return partx_.contains(ix) and party_.contains(iy);
	}

	auto & partx() const {
		return partx_;
	}

	auto & party() const {
		return party_;
	}

	auto sizex() const {
		return partx().size();
	}

	auto sizey() const {
		return party().size();
	}
	
 	auto local_sizex() const {
		return partx().local_size();
	}

	auto local_sizey() const {
		return party().local_size();
	} 
};

}
}
#endif

#ifdef INQ_MATRIX_DISTRIBUTED_UNIT_TEST
#undef INQ_MATRIX_DISTRIBUTED_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

  parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	parallel::cartesian_communicator<2> cart_comm(comm, {});

	auto sizex = 1000;
	auto sizey = 500;
	
  matrix::distributed<double> mm(cart_comm, sizex, sizey);
	
	for(int ix = 0; ix < sizex; ix++){
		for(int iy = 0; iy < sizey; iy++){
			auto ixg = parallel::global_index(ix);
			auto iyg = parallel::global_index(iy);			
			if(mm.is_local(ixg, iyg)) mm.block()[mm.partx().global_to_local(ixg)][mm.party().global_to_local(iyg)] = sqrt((ix + 1.0)*(iy + 1.0));
		}
	}
	
	for(int ix = 0; ix < sizex; ix++){
		for(int iy = 0; iy < sizey; iy++){
			auto ixg = parallel::global_index(ix);
			auto iyg = parallel::global_index(iy);			
			if(mm.is_local(ixg, iyg)) CHECK(mm.block()[mm.partx().global_to_local(ixg)][mm.party().global_to_local(iyg)] == sqrt((ix + 1.0)*(iy + 1.0)));
		}
	}

}
#endif
