/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__GRID
#define INQ__BASIS__GRID

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <array>

#include <basis/base.hpp>
#include <math/vector3.hpp>

namespace inq {
namespace basis {

class grid : public base {

protected:

	std::array<inq::parallel::partition, 3> cubic_part_;
	std::array<int, 3> sizes_;
	std::array<int, 3> local_sizes_;
	long npoints_;
	int par_dim_;

public:
	
	grid(std::array<int, 3> sizes, parallel::communicator & comm, int par_dim) :
		base(sizes[par_dim], comm),
		cubic_part_({inq::parallel::partition(sizes[0]), inq::parallel::partition(sizes[1]), inq::parallel::partition(sizes[2])}),
		sizes_(sizes),
		par_dim_(par_dim)
	{

		cubic_part_[par_dim] = base::part_;
		
		for(int idir = 0; idir < 3; idir++) {
			if(idir != par_dim) base::part_ *= sizes_[idir];
			local_sizes_[idir] = cubic_part_[idir].local_size();
		}

		npoints_ = sizes_[0]*long(sizes_[1])*sizes_[2];

	}

	grid(grid && old, parallel::communicator new_comm):
		grid(old.sizes_, new_comm, old.par_dim_)
	{
	}
		
	long size() const {
		return npoints_;
	}

	GPU_FUNCTION
	friend auto sizes(const grid & gr){
		return gr.sizes_;
	}

	GPU_FUNCTION
	auto & sizes() const {
		return sizes_;
	}

	auto & local_sizes() const {
		return local_sizes_;
	}
		
	constexpr auto & cubic_part(int dim) const {
		return cubic_part_[dim];
	}

	GPU_FUNCTION static auto to_symmetric_range(std::array<int, 3> const & sizes, const int ix, const int iy, const int iz) {
		vector3<int> ii{ix, iy, iz};
		for(int idir = 0; idir < 3; idir++) {
			if(ii[idir] >= (sizes[idir] + 1)/2) ii[idir] -= sizes[idir];
		}
		return ii;
	}

	GPU_FUNCTION static auto to_symmetric_range(std::array<int, 3> const & sizes, parallel::global_index ix, parallel::global_index iy, parallel::global_index iz) {
		return to_symmetric_range(sizes, ix.value(), iy.value(), iz.value());
	}

	GPU_FUNCTION static auto from_symmetric_range(std::array<int, 3> const & sizes, vector3<int> ii) {
		for(int idir = 0; idir < 3; idir++) {
			if(ii[idir] < 0) ii[idir] += sizes[idir];
		}
		return ii;
	}

	GPU_FUNCTION auto to_symmetric_range(const int ix, const int iy, const int iz) const {
		return to_symmetric_range(sizes_, ix, iy, iz);
	}

	GPU_FUNCTION auto to_symmetric_range(parallel::global_index ix, parallel::global_index iy, parallel::global_index iz) const {
		return to_symmetric_range(sizes_, ix.value(), iy.value(), iz.value());
	}
			
	GPU_FUNCTION auto from_symmetric_range(vector3<int> ii) const {
		return from_symmetric_range(sizes_, ii);
	}

	auto symmetric_range_begin(int idir) const {
		return -sizes_[idir]/2;
	}

	auto symmetric_range_end(int idir) const {
		return sizes_[idir]/2 + sizes_[idir]%2;
	}

	auto local_contains(vector3<int> const & ii) const {
		bool contains = true;
		for(int idir = 0; idir < 3; idir++){
			contains = contains and cubic_part_[idir].contains(ii[idir]);
		}
		return contains;
	}

	void shift() {
		base::part_.shift();
		for(int idir = 0; idir < 3; idir++) {
			cubic_part_[idir].shift();
			local_sizes_[idir] = cubic_part_[idir].local_size();
		}
		
	}
		
};

}
}
#endif

#ifdef INQ_BASIS_GRID_UNIT_TEST
#undef INQ_BASIS_GRID_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
  
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	basis::grid gr({45, 120, 77}, comm,  /*par_dim = */ 1);

	CHECK(gr.sizes()[0] == 45);
	CHECK(gr.sizes()[1] == 120);
	CHECK(gr.sizes()[2] == 77);

	CHECK(gr.cubic_part(0).local_size() == 45);
	if(comm.size() == 1) CHECK(gr.cubic_part(1).local_size() == 120);
	if(comm.size() == 2) CHECK(gr.cubic_part(1).local_size() == 60);
	if(comm.size() == 3) CHECK(gr.cubic_part(1).local_size() == 40);
	if(comm.size() == 4) CHECK(gr.cubic_part(1).local_size() == 30);
	if(comm.size() == 5) CHECK(gr.cubic_part(1).local_size() == 24);
	if(comm.size() == 6) CHECK(gr.cubic_part(1).local_size() == 20);
	if(comm.size() == 8) CHECK(gr.cubic_part(1).local_size() == 15);
	if(comm.size() == 10) CHECK(gr.cubic_part(1).local_size() == 12);
	if(comm.size() == 12) CHECK(gr.cubic_part(1).local_size() == 10);
	CHECK(gr.cubic_part(2).local_size() == 77);

	for(int ix = 0; ix < gr.local_sizes()[0]; ix++){
		for(int iy = 0; iy < gr.local_sizes()[1]; iy++){
			for(int iz = 0; iz < gr.local_sizes()[2]; iz++){

				auto ii = gr.from_symmetric_range(gr.to_symmetric_range(ix, iy, iz));

				CHECK(ii[0] == ix);
				CHECK(ii[1] == iy);
				CHECK(ii[2] == iz);
				
			}
		}
	}

	CHECK(gr.symmetric_range_begin(0) == -22);
	CHECK(gr.symmetric_range_end(0) == 23);
	
	CHECK(gr.symmetric_range_begin(1) == -60);
	CHECK(gr.symmetric_range_end(1) == 60);

	CHECK(gr.symmetric_range_begin(2) == -38);
	CHECK(gr.symmetric_range_end(2) == 39);	

	auto new_gr = basis::grid(basis::grid(gr), parallel::communicator{boost::mpi3::environment::get_self_instance()});

	CHECK(gr.sizes() == new_gr.sizes());
	CHECK(new_gr.local_sizes() == new_gr.sizes());
	
}
#endif

