/* -*- indent-tabs-mode: t -*- */

#ifndef GPURUN__GPU__REDUCE
#define GPURUN__GPU__REDUCE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <cassert>

#ifndef ENABLE_GPU
#include <numeric>
#else
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#endif

#include <gpu/run.hpp>
#include <gpu/array.hpp>
#include <gpu/host.hpp>
#include <gpu/indices.hpp>

namespace gpu {

struct reduce {
	explicit reduce(long arg_size):
		size(arg_size){		
	}
	long size;
};

template <typename ArrayType>
struct array_access {
  ArrayType array;

  GPU_FUNCTION auto operator()(long ii) const {
    return array[ii];
  }

  GPU_FUNCTION auto operator()(long ix, long iy) const {
    return array[ix][iy];
  }
  
};

template <typename Type, typename KernelType>
Type run(reduce const & red, Type const init, KernelType kernel) {

	auto const size = red.size;
	auto range = boost::multi::extension_t{0l, size};

#ifndef ENABLE_GPU
	return std::transform_reduce(range.begin(), range.end(), init, std::plus<>{}, kernel);
#else
	return thrust::transform_reduce(thrust::device, range.begin(), range.end(), kernel, init, std::plus<>{});
#endif
}

template <typename Type, typename KernelType>
Type run(gpu::reduce const & redx, gpu::reduce const & redy, Type const init, KernelType kernel) {

	auto const sizex = redx.size;	
	auto const sizey = redy.size;	

	if(sizex == 0 or sizey == 0) return init;
	
#ifndef ENABLE_GPU

  auto accumulator = init;
	for(long iy = 0; iy < sizey; iy++){
		for(long ix = 0; ix < sizex; ix++){
			accumulator += kernel(ix, iy);
		}
	}
  return accumulator;

#else

	auto range = boost::multi::extension_t{0l, sizex*sizey};
	return thrust::transform_reduce(thrust::device, range.begin(), range.end(),
																	[sizex, kernel] GPU_LAMBDA (auto ind) {

																		int ix, iy;
																		linear_to_bidimensional(ind, sizex, ix, iy);
																		return kernel(ix, iy);
																		
																	}, init, std::plus<>{});
  
#endif
}

template <typename Type, typename KernelType>
Type run(reduce const & redx, reduce const & redy, reduce const & redz, Type const init, KernelType kernel) {

	auto const sizex = redx.size;	
	auto const sizey = redy.size;
	auto const sizez = redz.size;
	
	if(sizex == 0 or sizey == 0 or sizez == 0) return init;
	
#ifndef ENABLE_GPU

  auto accumulator = init;
	for(long iy = 0; iy < sizey; iy++){
		for(long ix = 0; ix < sizex; ix++){
			for(long iz = 0; iz < sizez; iz++){
				accumulator += kernel(ix, iy, iz);
			}
		}
	}
  return accumulator;
	
#else

	auto range = boost::multi::extension_t{0l, sizex*sizey*sizez};
	return thrust::transform_reduce(thrust::device, range.begin(), range.end(),
																	[sizex, sizey, kernel] GPU_LAMBDA (auto ind) {

																		int ix, iy, iz;
																		linear_to_tridimensional(ind, sizex, sizey, ix, iy, iz);
																		return kernel(ix, iy, iz);
																		
																	}, init, std::plus<>{});
    
#endif
}

#ifdef ENABLE_GPU
template <typename KernelType, typename ArrayType>
__global__ void reduce_kernel_vr(long sizex, long sizey, KernelType kernel, ArrayType odata) {

	extern __shared__ char shared_mem[];
	auto reduction_buffer = (typename ArrayType::element *) shared_mem; // {blockDim.x, blockDim.y}
	
	// each thread loads one element from global to shared mem
  unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.z*blockDim.y + threadIdx.y;
	unsigned int iy = gridDim.y*blockDim.y*(blockIdx.z*blockDim.z + threadIdx.z) + blockIdx.y*blockDim.y + threadIdx.y;

	if(ix >= sizex) return;
	
	if(iy < sizey){
		reduction_buffer[threadIdx.x + blockDim.x*tid] = kernel(ix, iy);
	} else {
		reduction_buffer[threadIdx.x + blockDim.x*tid] = (typename ArrayType::element) 0.0;
	}

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = (blockDim.y*blockDim.z)/2; s > 0; s >>= 1){
		if (tid < s) {
			reduction_buffer[threadIdx.x + blockDim.x*tid] += reduction_buffer[threadIdx.x + blockDim.x*(tid + s)];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if (tid == 0) odata[blockIdx.z*gridDim.y + blockIdx.y][ix] = reduction_buffer[threadIdx.x];

}
#endif


template <typename Type, typename KernelType>
gpu::array<Type, 1> run(long sizex, reduce const & redy, Type const init, KernelType kernel) { 

	auto const sizey = redy.size;	

#ifndef ENABLE_GPU

  gpu::array<Type, 1> accumulator(sizex, init);

  for(long iy = 0; iy < sizey; iy++){
    for(long ix = 0; ix < sizex; ix++){
      accumulator[ix] += kernel(ix, iy);
    }
  }
  
  return accumulator;
  
#else

	gpu::array<Type, 2> result;
	
	auto blocksize = pow2_floor(max_blocksize(reduce_kernel_vr<KernelType, decltype(begin(result))>));
	
	unsigned bsizex = 4; //this seems to be the optimal value
	if(sizex <= 2) bsizex = sizex;
	unsigned bsizey = blocksize/bsizex;
	unsigned bsizez = 1;

	assert(bsizey > 1);

	unsigned nblockx  = num_blocks(sizex, bsizex);
	unsigned nblockyz = num_blocks(sizey, bsizey);
	unsigned nblockz  = num_blocks(nblockyz, MAX_DIM_YZ);
	unsigned nblocky  = num_blocks(nblockyz, nblockz);

	assert(nblocky*nblockz >= nblockyz);
	assert(nblocky <= MAX_DIM_YZ);
	assert(nblockz <= MAX_DIM_YZ);
	
	result.reextent({nblockyz, sizex});

	struct dim3 dg{nblockx, nblocky, nblockz};
  struct dim3 db{bsizex, bsizey, bsizez};

  auto shared_mem_size = blocksize*sizeof(Type);

  assert(shared_mem_size <= 48*1024);
  reduce_kernel_vr<<<dg, db, shared_mem_size>>>(sizex, sizey, kernel, begin(result));	
  check_error(last_error());
	
  if(nblockyz == 1) {
    gpu::sync();

		assert(result[0].size() == sizex);

		gpu::run(result[0].size(), [res = begin(result[0]), init] GPU_LAMBDA (auto ii) {
			res[ii] += init;
		});
			
    return result[0];
  } else {
    return run(sizex, reduce(nblockyz), init, array_access<decltype(begin(result.transposed()))>{begin(result.transposed())});
  }
  
#endif

}

#ifdef ENABLE_GPU
template <typename KernelType, typename ArrayType>
__global__ void reduce_kernel_vrr(long sizex, long sizey, long sizez, KernelType kernel, ArrayType odata) {

	extern __shared__ char shared_mem[];
	auto reduction_buffer = (typename ArrayType::element *) shared_mem;
	
	// each thread loads one element from global to shared mem
  unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.z*blockDim.y + threadIdx.y;
	unsigned int iyz = gridDim.y*blockDim.y*(blockIdx.z*blockDim.z + threadIdx.z) + blockIdx.y*blockDim.y + threadIdx.y;

	if(ix >= sizex) return;

	int iy, iz;
	linear_to_bidimensional(iyz, sizey, iy, iz);
	
	if(iy < sizey and iz < sizez){
		reduction_buffer[threadIdx.x + blockDim.x*tid] = kernel(ix, iy, iz);
	} else {
		reduction_buffer[threadIdx.x + blockDim.x*tid] = (typename ArrayType::element) 0.0;
	}
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = (blockDim.y*blockDim.z)/2; s > 0; s >>= 1){
		if (tid < s) {
			reduction_buffer[threadIdx.x + blockDim.x*tid] += reduction_buffer[threadIdx.x + blockDim.x*(tid + s)];
		}
		__syncthreads();
	}
	
	// write result for this block to global mem
	if (tid == 0) odata[blockIdx.z*gridDim.y + blockIdx.y][ix] = reduction_buffer[threadIdx.x];

}
#endif

template <typename Type, typename KernelType>
gpu::array<Type, 1>  run(long sizex, reduce const & redy, reduce const & redz, Type const init, KernelType kernel) {

	auto const sizey = redy.size;
	auto const sizez = redz.size;	
	
#ifndef ENABLE_GPU

  gpu::array<Type, 1> accumulator(sizex, init);

	for(long iz = 0; iz < sizez; iz++){
		for(long iy = 0; iy < sizey; iy++){
			for(long ix = 0; ix < sizex; ix++){
				accumulator[ix] += kernel(ix, iy, iz);
			}
		}
	}
  
  return accumulator;
  
#else

	gpu::array<Type, 2> result;
	
	auto blocksize = pow2_floor(max_blocksize(reduce_kernel_vrr<KernelType, decltype(begin(result))>));
	
	unsigned bsizex = 4; //this seems to be the optimal value
	if(sizex <= 2) bsizex = sizex;
	unsigned bsizey = blocksize/bsizex;
	unsigned bsizez = 1;

	assert(bsizey > 1);
	assert(bsizex*bsizey*bsizez == blocksize);
	
	unsigned nblockx  = num_blocks(sizex, bsizex);
	unsigned nblockyz = num_blocks(sizey*sizez, bsizey);
	unsigned nblockz  = num_blocks(nblockyz, MAX_DIM_YZ);
	unsigned nblocky  = num_blocks(nblockyz, nblockz);
	
	result.reextent({nblockyz, sizex});

	struct dim3 dg{nblockx, nblocky, nblockz};
  struct dim3 db{bsizex, bsizey, bsizez};

  auto shared_mem_size = blocksize*sizeof(Type);

  assert(shared_mem_size <= 48*1024);
  
  reduce_kernel_vrr<<<dg, db, shared_mem_size>>>(sizex, sizey, sizez, kernel, begin(result));	
  check_error(last_error());
	
  if(nblockyz == 1) {
    gpu::sync();

		assert(result[0].size() == sizex);

		gpu::run(result[0].size(), [res = begin(result[0]), init] GPU_LAMBDA (auto ii) {
			res[ii] += init;
		});
				
    return result[0];
  } else {
		auto && reduce_buffer = result.transposed();
    return run(sizex, reduce(nblockyz), init, array_access<decltype(begin(reduce_buffer))>{begin(reduce_buffer)});
  }
	
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Type, typename KernelType>
gpu::array<Type, 1>  run(long sizex, reduce const & redy, reduce const & redz, reduce const & redw, Type const init, KernelType kernel) {
	
#ifndef ENABLE_GPU

	auto const sizey = redy.size;
	auto const sizez = redz.size;
	auto const sizew = redw.size;
	
  gpu::array<Type, 1> accumulator(sizex, init);

	for(long iw = 0; iw < sizew; iw++){
		for(long iz = 0; iz < sizez; iz++){
			for(long iy = 0; iy < sizey; iy++){
				for(long ix = 0; ix < sizex; ix++){
					accumulator[ix] += kernel(ix, iy, iz, iw);
				}
			}
		}
	}
  
  return accumulator;
  
#else

	//this is a crude implementation for now
	gpu::array<Type, 1> result(sizex);
	for(auto ix = 0l; ix < sizex; ix++) {
		result[ix] = run(redy, redz, redw, init, [ix, kernel] GPU_LAMBDA (auto iy, auto iz, auto iw) { return kernel(ix, iy, iz, iw); });
	}
	return result;

#endif
}

}
#endif

#ifdef GPURUN__REDUCE__UNIT_TEST
#undef GPURUN__REDUCE__UNIT_TEST

#include <mpi3/environment.hpp>
#include <catch2/catch_all.hpp>

TEST_CASE(GPURUN_TEST_FILE, GPURUN_TEST_TAG) {
  
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("r"){
		CHECK(gpu::run(gpu::reduce(0), -232.8, [] GPU_LAMBDA (auto ii) -> double { return double(ii);} ) == Approx(-232.8));
		
		const long maxsize = 129140163;
		
		for(long nn = 1; nn <= maxsize; nn *= 3){
			CHECK(gpu::run(gpu::reduce(nn), -232.8, [] GPU_LAMBDA (auto ii) -> double { return double(ii);} ) == Approx(-232.8 + (nn*(nn - 1.0)/2.0)));
		}
		
	}

	SECTION("rr"){

		CHECK(gpu::run(gpu::reduce(100), gpu::reduce(  0), 2.23,  [] GPU_LAMBDA (auto ix, auto iy) -> double {return double(ix)*double(iy);}) == 2.23_a);
		CHECK(gpu::run(gpu::reduce(  0), gpu::reduce(100), 2.23,  [] GPU_LAMBDA (auto ix, auto iy) -> double {return double(ix)*double(iy);}) == 2.23_a);		

		const long maxsize = 2*625;

		for(long nx = 1; nx <= maxsize; nx *= 5){
			for(long ny = 1; ny <= maxsize; ny *= 5){

				auto res = gpu::run(gpu::reduce(nx), gpu::reduce(ny), 2.23,  [] GPU_LAMBDA (auto ix, auto iy) -> double {return double(ix)*double(iy);});
				
				CHECK(typeid(decltype(res)) == typeid(double));
				CHECK(res == Approx(2.23 + nx*(nx - 1.0)/2.0*ny*(ny - 1.0)/2.0));
			}
		}
		
  }

	SECTION("rrr"){
		CHECK(gpu::run(gpu::reduce(  0), gpu::reduce(100), gpu::reduce(100), 17.89, [] GPU_LAMBDA (auto ix, auto iy, auto iz) -> double {return double(ix)*double(iy)*double(iz);}) == 17.89_a);
		CHECK(gpu::run(gpu::reduce(100), gpu::reduce(  0), gpu::reduce(100), 17.89, [] GPU_LAMBDA (auto ix, auto iy, auto iz) -> double {return double(ix)*double(iy)*double(iz);}) == 17.89_a);
		CHECK(gpu::run(gpu::reduce(100), gpu::reduce(100), gpu::reduce(  0), 17.89, [] GPU_LAMBDA (auto ix, auto iy, auto iz) -> double {return double(ix)*double(iy)*double(iz);}) == 17.89_a);		
		
		const long maxsize = 125;

		for(long nx = 1; nx <= 10000; nx *= 10){
			for(long ny = 1; ny <= maxsize; ny *= 5){
				for(long nz = 1; nz <= maxsize; nz *= 5){
					
					auto res = gpu::run(gpu::reduce(nx), gpu::reduce(ny), gpu::reduce(nz), 17.89, [] GPU_LAMBDA (auto ix, auto iy, auto iz) -> double {return double(ix)*double(iy)*double(iz);});
					
					CHECK(typeid(decltype(res)) == typeid(double));
					CHECK(res == Approx(17.89 + nx*(nx - 1.0)/2.0*ny*(ny - 1.0)/2.0*nz*(nz - 1.0)/2.0));
				}
			}
		}
		
  }
	
	SECTION("vr"){

		const long maxsize = 390625;

		for(long nx = 1; nx <= 10000; nx *= 10){
			for(long ny = 1; ny <= maxsize; ny *= 5){

				auto res = gpu::run(nx, gpu::reduce(ny), -7.7, [] GPU_LAMBDA (auto ix, auto iy) {return double(ix)*double(iy);});
					
				CHECK(typeid(decltype(res)) == typeid(gpu::array<double, 1>));
				CHECK(res.size() == nx);
				for(long ix = 0; ix < 1; ix++) CHECK(res[ix] == Approx(-7.7 + double(ix)*ny*(ny - 1.0)/2.0));
			}
		}
		
		auto dim = 65535*338 + 27;
		auto lred = gpu::run(4, gpu::reduce(dim), 0.0, [] GPU_LAMBDA (auto ix, auto iy) {return double(ix + 1)*double(iy + 1.0);});
		CHECK(lred[0] == 1*dim*(dim + 1.0)/2.0);
		CHECK(lred[1] == 2*dim*(dim + 1.0)/2.0);
		CHECK(lred[2] == 3*dim*(dim + 1.0)/2.0);
		CHECK(lred[3] == 4*dim*(dim + 1.0)/2.0);
  }

	SECTION("vrr"){

		const long maxsize = 625;

		for(long nx = 1; nx <= 10000; nx *= 10){
			for(long ny = 1; ny <= maxsize; ny *= 5){
				for(long nz = 1; nz <= maxsize; nz *= 5){
					
					auto res = gpu::run(nx, gpu::reduce(ny), gpu::reduce(nz), 10.0, [] GPU_LAMBDA (auto ix, auto iy, auto iz) {return double(ix)*double(iy)*double(iz);});
					
					CHECK(typeid(decltype(res)) == typeid(gpu::array<double, 1>));
					
					CHECK(res.size() == nx);
					for(long ix = 0; ix < nx; ix++) CHECK(res[ix] == Approx(10.0 + double(ix)*ny*(ny - 1.0)/2.0*nz*(nz - 1.0)/2.0));
				}
			}
		}

		auto dim = 65535*338 + 27;
		auto lred = gpu::run(4, gpu::reduce(3), gpu::reduce(dim), 0.0, [] GPU_LAMBDA (auto ix, auto iy, auto iz) {return double(ix + 1)*double(iz + 1.0);});
		CHECK(lred[0] ==  3*dim*(dim + 1.0)/2.0);
		CHECK(lred[1] ==  6*dim*(dim + 1.0)/2.0);
		CHECK(lred[2] ==  9*dim*(dim + 1.0)/2.0);
		CHECK(lred[3] == 12*dim*(dim + 1.0)/2.0);
		
  }

}
#endif

