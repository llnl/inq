/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PARALLEL__TRANSPOSE
#define INQ__PARALLEL__TRANSPOSE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <cstdlib>

#include <gpu/array.hpp>
#include <utils/raw_pointer_cast.hpp>

#include <inq_config.h>
#include <mpi.h>
#include <parallel/communicator.hpp>
#include <gpu/run.hpp>

namespace inq {
namespace parallel {

template <typename PartX, typename PartY, typename Matrix>
void transpose(parallel::communicator & comm, PartX const & partx, PartY const & party, Matrix & matrix){
	CALI_CXX_MARK_FUNCTION;

	auto bsize = partx.max_local_size()*party.max_local_size();
	
	Matrix buffer({comm.size(), bsize});

	gpu::run(party.max_local_size(), partx.max_local_size(), comm.size(),
					 [mat = begin(matrix), buf = begin(buffer), partx, party] GPU_LAMBDA (auto iy, auto ix, auto iproc) {
						 if(long(ix) < partx.local_size(iproc) and long(iy) < party.local_size()) buf[iproc][ix + iy*partx.max_local_size()] = mat[partx.start(iproc) + ix][iy];
					 });

	parallel::alltoall(buffer, comm);

	matrix.reextent({party.size(), partx.local_size()});

	gpu::run(partx.max_local_size(), party.max_local_size(), comm.size(),
					 [mat = begin(matrix), buf = begin(buffer), partx, party] GPU_LAMBDA (auto ix, auto iy, auto iproc) {
						 if(long(iy) < party.local_size(iproc) and long(ix) < partx.local_size()) mat[party.start(iproc) + iy][ix] = buf[iproc][ix + iy*partx.max_local_size()];
					 });

}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename PartX, typename PartY, typename Array>
void transpose_forward(parallel::communicator & comm, PartX const & partx, PartY const & party, Array & array){
	CALI_CXX_MARK_FUNCTION;

	assert(comm.size() == party.comm_size());
	assert(comm.size() == partx.comm_size());
	
	assert(get<0>(sizes(array)) == partx.size());
	assert(get<1>(sizes(array)) == party.local_size());
	auto nz = get<2>(sizes(array));
	auto nw = get<3>(sizes(array));
	
	Array buffer({comm.size(), party.max_local_size()*partx.max_local_size(), nz, nw});

	for(int iproc = 0; iproc < comm.size(); iproc++) {
		gpu::run(nw, nz, party.max_local_size(), partx.max_local_size(),
						 [arr = begin(array), buf = begin(buffer), partx, party, iproc] GPU_LAMBDA (auto iw, auto iz, auto iy, auto ix) {
							 if(long(ix) < partx.local_size(iproc) and long(iy) < party.local_size()) buf[iproc][ix + iy*partx.max_local_size()][iz][iw] = arr[partx.start(iproc) + ix][iy][iz][iw];
						 });
	}
	
	parallel::alltoall(buffer, comm);

	array.reextent({nz, party.size(), partx.local_size(), nw});

	for(int iproc = 0; iproc < comm.size(); iproc++) {
		gpu::run(nw, nz, partx.max_local_size(), party.max_local_size(),
						 [arr = begin(array), buf = begin(buffer), partx, party, iproc] GPU_LAMBDA (auto iw, auto iz, auto ix, auto iy) {
							 if(long(iy) < party.local_size(iproc) and long(ix) < partx.local_size()) arr[iz][party.start(iproc) + iy][ix][iw] = buf[iproc][ix + iy*partx.max_local_size()][iz][iw];
						 });
	}

}

////////////////////////////////////////////////////////////////////////////////////////////

template <typename PartX, typename PartY, typename Array>
void transpose_backward(parallel::communicator & comm, PartX const & partx, PartY const & party, Array & array){
	CALI_CXX_MARK_FUNCTION;

	assert(comm.size() == party.comm_size());
	assert(comm.size() == partx.comm_size());

	auto nz = get<0>(sizes(array));
	assert(get<1>(sizes(array)) == party.size());
	assert(get<2>(sizes(array)) == partx.local_size());
	auto nw = get<3>(sizes(array));
	
	Array buffer({comm.size(), party.max_local_size()*partx.max_local_size(), nz, nw});

	for(int iproc = 0; iproc < comm.size(); iproc++) {
		gpu::run(nw, nz, party.max_local_size(), partx.max_local_size(),
						 [arr = begin(array), buf = begin(buffer), partx, party, iproc] GPU_LAMBDA (auto iw, auto iz, auto iy, auto ix) {
							 if(long(iy) < party.local_size(iproc) and long(ix) < partx.local_size()) buf[iproc][ix + iy*partx.max_local_size()][iz][iw] = arr[iz][party.start(iproc) + iy][ix][iw];
						 });
	}
	
	parallel::alltoall(buffer, comm);

	array.reextent({partx.size(), party.local_size(), nz, nw});

	for(int iproc = 0; iproc < comm.size(); iproc++) {
		gpu::run(nw, nz, partx.max_local_size(), party.max_local_size(),
						 [arr = begin(array), buf = begin(buffer), partx, party, iproc] GPU_LAMBDA (auto iw, auto iz, auto ix, auto iy) {
							 if(long(ix) < partx.local_size(iproc) and long(iy) < party.local_size()) arr[partx.start(iproc) + ix][iy][iz][iw] = buf[iproc][ix + iy*partx.max_local_size()][iz][iw];
						 });
	}

}

}
}
#endif

#ifdef INQ_PARALLEL_TRANSPOSE_UNIT_TEST
#undef INQ_PARALLEL_TRANSPOSE_UNIT_TEST

#include <gpu/run.hpp>
#include <mpi3/environment.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

  using namespace inq;
  using namespace Catch::literals;
  
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	SECTION("4x6 matrix") {
		if(comm.size() == 2) {
			
			auto nx = 4;
			auto ny = 6;
			
			auto partx = parallel::partition(nx, comm);
			auto party = parallel::partition(ny, comm);	
			
			gpu::array<int, 2> matrix({nx, party.local_size()});
			
			if(comm.rank() == 0) {
				matrix[0][0] =  1;
				matrix[0][1] =  2;
				matrix[0][2] =  3;
				matrix[1][0] =  7;
				matrix[1][1] =  8;
				matrix[1][2] =  9;
				matrix[2][0] = 13;
				matrix[2][1] = 14;
				matrix[2][2] = 15;
				matrix[3][0] = 19;
				matrix[3][1] = 20;
				matrix[3][2] = 21;
			}
			
			if(comm.rank() == 1) {
				matrix[0][0] =  4;
				matrix[0][1] =  5;
				matrix[0][2] =  6;
				matrix[1][0] = 10;
				matrix[1][1] = 11;
				matrix[1][2] = 12;
				matrix[2][0] = 16;
				matrix[2][1] = 17;
				matrix[2][2] = 18;
				matrix[3][0] = 22;
				matrix[3][1] = 23;
				matrix[3][2] = 24;
			}
			
			parallel::transpose(comm, partx, party, matrix);
			
			CHECK(get<0>(sizes(matrix)) == party.size());
			CHECK(get<1>(sizes(matrix)) == partx.local_size());	
			
			if(comm.rank() == 0) {
				CHECK(matrix == gpu::array<int, 2>{
						{1,  7},	
						{2,  8},
						{3,  9},
						{4, 10},
						{5, 11},
						{6, 12}});
			}
			
			if(comm.rank() == 1) {
				CHECK(matrix == gpu::array<int, 2>{
						{13, 19},	
						{14, 20},
						{15, 21},
						{16, 22},
						{17, 23},
						{18, 24}});
			}
		}
	}

	SECTION("Multiple sizes") {

		std::vector<int> dims({235, 666, 757, 1080, 1427});

		for(auto nx : dims) {
			for(auto ny : dims) {

				auto partx = parallel::partition(nx, comm);
				auto party = parallel::partition(ny, comm);

				gpu::array<complex, 2> matrix({nx, party.local_size()}, NAN);

				gpu::run(party.local_size(), partx.size(),
								 [ma = begin(matrix), party_start = party.start()] GPU_LAMBDA (auto iy, auto ix) {
									 ma[ix][iy] = complex{double(ix), party_start + double(iy)};
								 });

				parallel::transpose(comm, partx, party, matrix);

				CHECK(get<0>(sizes(matrix)) == party.size());
				CHECK(get<1>(sizes(matrix)) == partx.local_size());

				auto fails = gpu::array<int, 1>(2, 0);

				gpu::run(partx.local_size(), party.size(),
								 [ma = begin(matrix), partx_start = partx.start(), fa = begin(fails)] GPU_LAMBDA (auto ix, auto iy) {
									 if(real(ma[iy][ix]) != partx_start + double(ix)) gpu::atomic(fa[0])++;
									 if(imag(ma[iy][ix]) != double(iy))               gpu::atomic(fa[1])++;
								 });

				CHECK(fails[0] == 0);
				CHECK(fails[1] == 0);
				
			}
		}

	}
	
	SECTION("forward") {
		
		auto nx = 20;
		auto ny = 24;
		auto nz = 17;
		auto nw = 32;

		auto partx = parallel::partition(nx, comm);
		auto party = parallel::partition(ny, comm);
		
		gpu::array<complex, 4> array({nx, party.local_size(), nz, nw}, NAN);

		gpu::run(nw, nz, party.local_size(), partx.size(),
						 [ar = begin(array), party_start = party.start()] GPU_LAMBDA (auto iw, auto iz, auto iy, auto ix) {
							 ar[ix][iy][iz][iw] = complex{ix*1000.0 + iw, party_start + iy + iz*10000.0};
						 });
		
		parallel::transpose_forward(comm, partx, party, array);

		CHECK(get<0>(sizes(array)) == nz);
		CHECK(get<1>(sizes(array)) == party.size());
		CHECK(get<2>(sizes(array)) == partx.local_size());
		CHECK(get<3>(sizes(array)) == nw);
		
 		auto fails = gpu::array<int, 1>(2, 0);
		gpu::run(nw, partx.local_size(), party.size(), nz,
						 [ar = begin(array), partx_start = partx.start(), fa = begin(fails)] GPU_LAMBDA (auto iw, auto ix, auto iy, auto iz) {
							 if(real(ar[iz][iy][ix][iw]) != (partx_start + ix)*1000.0 + iw) gpu::atomic(fa[0])++;
							 if(imag(ar[iz][iy][ix][iw]) != iy + iz*10000.0)                gpu::atomic(fa[1])++;
						 });
		
		CHECK(fails[0] == 0);
		CHECK(fails[1] == 0);

	}

	SECTION("backward") {
		
		auto nx = 201;
		auto ny = 59;
		auto nz = 137;
		auto nw = 67;

		auto partx = parallel::partition(nx, comm);
		auto party = parallel::partition(ny, comm);
		
		gpu::array<complex, 4> array({nz, ny, partx.local_size(), nw}, NAN);

		gpu::run(nw, partx.local_size(), party.size(), nz,
						 [ar = begin(array), partx_start = partx.start()] GPU_LAMBDA (auto iw, auto ix, auto iy, auto iz) {
							 ar[iz][iy][ix][iw] = complex{(partx_start + ix)*1000.0 + iw, iy + iz*10000.0};
						 });
		
		parallel::transpose_backward(comm, partx, party, array);
		
		CHECK(get<0>(sizes(array)) == partx.size());
		CHECK(get<1>(sizes(array)) == party.local_size());
		CHECK(get<2>(sizes(array)) == nz);
		CHECK(get<3>(sizes(array)) == nw);

		auto fails = gpu::array<int, 1>(2, 0);
		gpu::run(nw, nz, party.local_size(), partx.size(),
						 [ar = begin(array), party_start = party.start(), fa = begin(fails)] GPU_LAMBDA (auto iw, auto iz, auto iy, auto ix) {
							 if(real(ar[ix][iy][iz][iw]) != ix*1000.0 + iw)                gpu::atomic(fa[0])++;
							 if(imag(ar[ix][iy][iz][iw]) != party_start + iy + iz*10000.0) gpu::atomic(fa[1])++;
						 });
		
		CHECK(fails[0] == 0);
		CHECK(fails[1] == 0);

	}
	
	
}


#endif
