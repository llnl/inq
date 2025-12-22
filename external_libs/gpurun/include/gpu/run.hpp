/* -*- indent-tabs-mode: t -*- */

#ifndef GPURUN__GPU__RUN
#define GPURUN__GPU__RUN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <gpu/host.hpp>
#include <gpu/indices.hpp>

#define CUDA_MAX_DIM1 2147483647ULL
#define CUDA_MAX_DIM23 65535

namespace gpu {

//finds fact1, fact2 < thres such that fact1*fact2 >= val
inline static void factorize(const std::size_t val, const std::size_t thres, std::size_t & fact1, std::size_t & fact2){
	fact1 = val;
	fact2 = 1;
	while (fact1 > thres){
		fact1 = (fact1 + 1)/2;
		fact2 *= 2;
	}

	assert(fact1*fact2 >= val);
}

#ifdef ENABLE_GPU
template <class kernel_type>
__global__ void run_kernel_0(kernel_type kernel){
	kernel();
}
#endif

template <class kernel_type>
void run(kernel_type kernel){
	
#ifdef ENABLE_GPU

	run_kernel_0<<<1, 1>>>(kernel);
	check_error(last_error());
	sync();
	
#else
	kernel();
#endif
}

#ifdef ENABLE_GPU
template <class kernel_type>
__global__ void run_kernel_1(unsigned size, kernel_type kernel){
	auto ii = blockIdx.x*blockDim.x + threadIdx.x;
	if(ii < size) kernel(ii);
}
#endif

template <class kernel_type>
void run(size_t size, kernel_type kernel){
	
#ifdef ENABLE_GPU
	if(size == 0) return;
		
	assert(size <= CUDA_MAX_DIM1);

	auto blocksize = max_blocksize(run_kernel_1<kernel_type>);

	unsigned nblock = num_blocks(size, blocksize);
  
	run_kernel_1<<<nblock, blocksize>>>(size, kernel);
	check_error(last_error());
	sync();
	
#else
	for(size_t ii = 0; ii < size; ii++) kernel(ii);
#endif
  
}

#ifdef ENABLE_GPU
template <class kernel_type>
__global__ void run_kernel_2(unsigned sizex, unsigned sizey, unsigned dim2, kernel_type kernel){
	auto i1 = blockIdx.x*blockDim.x + threadIdx.x;
	auto i2 = blockIdx.y*blockDim.y + threadIdx.y;
	auto i3 = blockIdx.z*blockDim.z + threadIdx.z;
	
	auto ix = i1;
	auto iy = i2 + dim2*i3;
	if(ix < sizex && iy < sizey) kernel(ix, iy);
}
#endif

template <class kernel_type>
void run(size_t sizex, size_t sizey, kernel_type kernel){

#ifdef ENABLE_GPU
	if(sizex == 0 or sizey == 0) return;

	auto blocksize = max_blocksize(run_kernel_2<kernel_type>);

	//OPTIMIZATION, this is not ideal if sizex < blocksize
	unsigned nblock = num_blocks(sizex, blocksize);
	
	size_t dim2, dim3;
	factorize(sizey, CUDA_MAX_DIM23, dim2, dim3);
	
	struct dim3 dg{nblock, unsigned(dim2), unsigned(dim3)};
	struct dim3 db{unsigned(blocksize), 1, 1};
	run_kernel_2<<<dg, db>>>(sizex, sizey, dim2, kernel);
	check_error(last_error());
		
	sync();
	
#else
	for(size_t iy = 0; iy < sizey; iy++){
		for(size_t ix = 0; ix < sizex; ix++){
			kernel(ix, iy);
		}
	}
#endif
}

#ifdef ENABLE_GPU
template <class kernel_type>
__global__ void run_kernel_3(unsigned sizex, unsigned sizey, unsigned sizez, kernel_type kernel){
	auto ix = blockIdx.x*blockDim.x + threadIdx.x;
	auto iy = blockIdx.y*blockDim.y + threadIdx.y;
	auto iz = blockIdx.z*blockDim.z + threadIdx.z;
	if(ix < sizex && iy < sizey && iz < sizez) kernel(ix, iy, iz);
	
}
#endif

template <class kernel_type>
void run(size_t sizex, size_t sizey, size_t sizez, kernel_type kernel){
	
#ifdef ENABLE_GPU
	if(sizex == 0 or sizey == 0 or sizez == 0) return;

	auto blocksize = max_blocksize(run_kernel_3<kernel_type>);
	
	//OPTIMIZATION, this is not ideal if sizex < blocksize
	unsigned nblock = num_blocks(sizex, blocksize);
	struct dim3 dg{nblock, unsigned(sizey), unsigned(sizez)};
	struct dim3 db{unsigned(blocksize), 1, 1};
	run_kernel_3<<<dg, db>>>(sizex, sizey, sizez, kernel);
	check_error(last_error());
	
	sync();
	
#else
	for(size_t iz = 0; iz < sizez; iz++){
		for(size_t iy = 0; iy < sizey; iy++){
			for(size_t ix = 0; ix < sizex; ix++){
				kernel(ix, iy, iz);
			}
		}
	}
#endif
}
	
#ifdef ENABLE_GPU
template <class kernel_type>
__global__ void run_kernel_4(unsigned sizex, unsigned sizey, unsigned sizez, unsigned sizew, kernel_type kernel){
	auto ix = blockIdx.x*blockDim.x + threadIdx.x;
	auto iy = blockIdx.y*blockDim.y + threadIdx.y;
	auto iz = blockIdx.z*blockDim.z + threadIdx.z;
	if(ix < sizex && iy < sizey && iz < sizez){
		for(int iw = 0; iw < sizew; iw++){
			kernel(ix, iy, iz, iw);
		}
	}
}
#endif
 
template <class kernel_type>
void run(size_t sizex, size_t sizey, size_t sizez, size_t sizew, kernel_type kernel){

	if(sizex == 1){
		run(sizey, sizez, sizew, [kernel] GPU_LAMBDA (auto iy, auto iz, auto iw){ kernel(0, iy, iz, iw); });
		return;
	}
	
#ifdef ENABLE_GPU
	if(sizex == 0 or sizey == 0 or sizez == 0 or sizew == 0) return;

	auto blocksize = max_blocksize(run_kernel_4<kernel_type>);
	
	//OPTIMIZATION, this is not ideal if sizex < blocksize
	unsigned nblock = num_blocks(sizex, blocksize);
	struct dim3 dg{nblock, unsigned(sizey), unsigned(sizez)};
	struct dim3 db{unsigned(blocksize), 1, 1};
	run_kernel_4<<<dg, db>>>(sizex, sizey, sizez, sizew, kernel);
	check_error(last_error());
	
	sync();

#else
	for(size_t iw = 0; iw < sizew; iw++){
		for(size_t iz = 0; iz < sizez; iz++){
			for(size_t iy = 0; iy < sizey; iy++){
				for(size_t ix = 0; ix < sizex; ix++){
					kernel(ix, iy, iz, iw);
				}
			}
		}
	}
#endif
}

}
#endif

#ifdef GPURUN__RUN__UNIT_TEST
#undef GPURUN__RUN__UNIT_TEST

#include <gpu/array.hpp>
#include <mpi3/environment.hpp>
#include <catch2/catch_all.hpp>
#include <gpu/atomic.hpp>

long check_run(long size){
	
	gpu::array<long, 1> list(size, 0l);

	gpu::run(size,
					 [itlist = begin(list)] GPU_LAMBDA (auto ii){
						 gpu::atomic(itlist[ii]) += ii + 1;
					 });
	
	long diff = 0;
	for(long ii = 0; ii < size; ii++) {
		diff += ii + 1 - list[ii];
	}
	return diff;
}

long check_run(long size1, long size2){
	
	gpu::array<long, 3> list({size1, size2, 2}, 0l);
	
	gpu::run(size1, size2, 
					 [itlist = begin(list)] GPU_LAMBDA (auto ii, auto jj){
						 gpu::atomic(itlist[ii][jj][0]) += ii + 1;
						 gpu::atomic(itlist[ii][jj][1]) += jj + 1;
					 });
	
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			diff += ii + 1 - list[ii][jj][0];
			diff += jj + 1 - list[ii][jj][1];
		}
	}
		
	return diff;
}

long check_run(long size1, long size2, long size3){
	
	gpu::array<long, 4> list({size1, size2, size3, 3}, 0l);

	gpu::run(size1, size2, size3,
					 [itlist = begin(list)] GPU_LAMBDA (auto ii, auto jj, auto kk){
						 gpu::atomic(itlist[ii][jj][kk][0]) += ii + 1;
						 gpu::atomic(itlist[ii][jj][kk][1]) += jj + 1;
						 gpu::atomic(itlist[ii][jj][kk][2]) += kk + 1;
					 });
		
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			for(long kk = 0; kk < size3; kk++) {
				diff += ii + 1 - list[ii][jj][kk][0];
				diff += jj + 1 - list[ii][jj][kk][1];
				diff += kk + 1 - list[ii][jj][kk][2];
			}
		}
	}

	return diff;
}
	
long check_run(long size1, long size2, long size3, long size4){

	gpu::array<long, 5> list({size1, size2, size3, size4, 4}, 0l);

	gpu::run(size1, size2, size3, size4,
					 [itlist = begin(list)] GPU_LAMBDA (auto ii, auto jj, auto kk, auto ll){
						 gpu::atomic(itlist[ii][jj][kk][ll][0]) += ii + 1;
						 gpu::atomic(itlist[ii][jj][kk][ll][1]) += jj + 1;
						 gpu::atomic(itlist[ii][jj][kk][ll][2]) += kk + 1;
						 gpu::atomic(itlist[ii][jj][kk][ll][3]) += ll + 1;
					 });
		
	long diff = 0;
	for(long ii = 0; ii < size1; ii++) {
		for(long jj = 0; jj < size2; jj++) {
			for(long kk = 0; kk < size3; kk++) {
				for(long ll = 0; ll < size4; ll++) {
					diff += ii + 1 - list[ii][jj][kk][ll][0];
					diff += jj + 1 - list[ii][jj][kk][ll][1];
					diff += kk + 1 - list[ii][jj][kk][ll][2];
					diff += ll + 1 - list[ii][jj][kk][ll][3];
				}
			}
		}
	}

	return diff;
}

TEST_CASE(GPURUN_TEST_FILE, GPURUN_TEST_TAG) {

	using namespace Catch::literals;

	SECTION("1D"){
		CHECK(check_run(200) == 0);
		CHECK(check_run(1024) == 0);
		CHECK(check_run(6666) == 0);
	}
	
	SECTION("2D"){
		CHECK(check_run(200, 200) == 0);
		CHECK(check_run(256, 1200) == 0);
		CHECK(check_run(2023, 4) == 0);
		CHECK(check_run(7, 57*57*57) == 0);
	}

	SECTION("3D"){
		CHECK(check_run(2, 2, 2) == 0);
		CHECK(check_run(7, 2, 2) == 0);
		CHECK(check_run(7, 57, 57) == 0);
		CHECK(check_run(32, 23, 18) == 0);
		CHECK(check_run(213, 27, 78) == 0);
		CHECK(check_run(2500, 10, 12) == 0);
		CHECK(check_run(7, 1023, 12) == 0);	
		CHECK(check_run(1, 11, 1229) == 0);	
	}
	
	SECTION("4D"){
		CHECK(check_run(2, 2, 2, 2) == 0);
		CHECK(check_run(7, 2, 2, 2) == 0);
		CHECK(check_run(7, 57, 57, 57) == 0);
		CHECK(check_run(32, 23, 45, 18) == 0);
		CHECK(check_run(35, 213, 27, 78) == 0);
		CHECK(check_run(2500, 10, 11, 12) == 0);
		CHECK(check_run(7, 1023, 11, 12) == 0);
		CHECK(check_run(1, 1, 11, 1229) == 0);
		CHECK(check_run(1, 1023, 11, 12) == 0);
	}

}

#endif
