/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PARALLEL__PARTITION
#define INQ__PARALLEL__PARTITION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <parallel/global_index.hpp>
#include <utils/raw_pointer_cast.hpp>

#include <parallel/communicator.hpp>
#include <mpi3/environment.hpp>
#include <mpi3/detail/datatype.hpp>

#include <cassert>
#include <array>

namespace inq{
namespace parallel {

class partition {

protected:
	
	long size_;
	long start_;
	long end_;
	long bsize_;
	int  comm_size_;
	int  rank_;

public:

	constexpr auto local_size() const {
		return end_ - start_;
	}
	
	partition(const long size, int comm_size, int comm_rank)
		:size_(size),
		 comm_size_(comm_size),
		 rank_(comm_rank)
	{
		
		bsize_ = (size_ + comm_size_ - 1)/comm_size_;
		
		if(size_ > 0) assert(bsize_ > 0);

		start_ = std::min(bsize_*comm_rank, size_);
		end_ = std::min(bsize_*(comm_rank + 1), size_);
		
		assert(local_size() <= bsize_);
		assert(end_ >= start_);
		assert(end_ <= size);
	}

	template <typename CommType>
	partition(const long size, CommType const & comm)
		:partition(size, comm.size(), comm.rank()){
	}
	
	partition(const long size)
		:partition(size, 1, 0){
	}
	
	auto operator*=(const long factor) {
		size_ *= factor;
		start_ *= factor;
		end_ *= factor;
		bsize_ *= factor;
		
		return *this;
	}

	friend auto operator*(const long factor, partition part){
		part *= factor;
		return part;
	}
	
	constexpr auto size() const {
		return size_;
	}
	
	constexpr auto start() const {
		return start_;
	}

	constexpr auto start(int part) const {
		return std::min(bsize_*part, size_);
	}
	
	constexpr auto end() const {
		return end_;
	}
	
	constexpr auto end(int part) const {
		return std::min(bsize_*(part + 1), size_);
	}
	
	constexpr auto local_size(int part) const {
		return end(part) - start(part);
	}
	
	constexpr auto parallel() const {
		return comm_size_ > 1;
	}
	
	constexpr auto contains(long index) const {
		return start() <= index and index < end();
	}

	constexpr auto contains(long index, int part) const {
		return start(part) <= index and index < end(part);
	}

	constexpr auto contains(parallel::global_index index) const {
		return start() <= index.value() and index.value() < end();
	}

	constexpr auto local_to_global(long local_i) const {
		return global_index(start_ + local_i);
	}
	
	constexpr auto global_to_local(global_index global_i) const {
		return global_i.value() - start_;
	}
	
	constexpr auto comm_size() const {
		return comm_size_;
	}

	constexpr auto max_local_size() const {
		return bsize_;
	}
	
	constexpr auto location(long global_i) const {
		return global_i/bsize_;
	}

	constexpr auto location(global_index global_i) const {
		return global_i.value()/bsize_;
	}
	
	constexpr auto waste() const {
		auto total_elements = bsize_*comm_size();
		return (total_elements - size())/double(size());
	}

	constexpr auto & rank() const {
		return rank_;
	}

	void shift() {
		rank_++;
		if(rank_ == comm_size_) rank_ = 0;
		start_ = start(rank_);
		end_   = end(rank_);
	}
	
};
}
}
#endif

#ifdef INQ_PARALLEL_PARTITION_UNIT_TEST
#undef INQ_PARALLEL_PARTITION_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
  
	using namespace inq;
	using namespace Catch::literals;
  
  const int NN = 1033;

  parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
  
	inq::parallel::partition part(NN, comm);

	CHECK(part.comm_size() == comm.size());
	CHECK(part.rank() == comm.rank());
	
  auto next = comm.rank() + 1;
  if(next == comm.size()) next = 0;
  
  auto prev = comm.rank() - 1;
  if(prev == -1) prev = comm.size() - 1;

  SECTION("Total"){

    CHECK(NN == part.size());

    auto calculated_size = part.local_size();
    
    comm.all_reduce_in_place_n(&calculated_size, 1, std::plus<>{});
    
    CHECK(NN == calculated_size);

  }

	SECTION("Waste"){
		if(comm.size() == 1) CHECK(part.waste() == 0.0_a);
		if(comm.size() == 2) CHECK(part.waste() == 0.0009680542_a);
		if(comm.size() == 3) CHECK(part.waste() == 0.0019361084_a);
		if(comm.size() == 4) CHECK(part.waste() == 0.0029041626_a);
		if(comm.size() == 5) CHECK(part.waste() == 0.0019361084_a);
	}
	
  SECTION("Upper bound"){
    auto boundary_value = part.end();
    
    comm.send_receive_replace_n(&boundary_value, 1, /* dest = */ next, /* source = */ prev, 0, 0);
    
    if(comm.rank() != 0){
      CHECK(boundary_value == part.start());
    } else {
      CHECK(boundary_value == NN);
    }
  }

  SECTION("Lower bound"){
    auto boundary_value = part.start();

    comm.send_receive_replace_n(&boundary_value, 1, /* dest = */ prev, /* source = */ next, 1, 1);
    
    if(comm.rank() != comm.size() - 1){
      CHECK(boundary_value == part.end());
    } else {
      CHECK(boundary_value == 0);
    }
  }

	SECTION("Location"){

		for(long ig = part.start(); ig < part.end(); ig++){
			CHECK(part.location(ig) == comm.rank());
		}
	}

	long factor = 13;

	part *= factor;
	
	SECTION("Scaled - Total"){
		
    CHECK(NN*factor == part.size());
		
    auto calculated_size = part.local_size();
    
    comm.all_reduce_in_place_n(&calculated_size, 1, std::plus<>{});
    
    CHECK(NN*factor == calculated_size);
  }
	

  SECTION("Scaled - Upper bound"){
    auto boundary_value = part.end();
    
    comm.send_receive_replace_n(&boundary_value, 1, /* dest = */ next, /* source = */ prev, 0, 0);
    
    if(comm.rank() != 0){
      CHECK(boundary_value == part.start());
    } else {
      CHECK(boundary_value == NN*factor);
    }
  }

  SECTION("Scaled - Lower bound"){
    auto boundary_value = part.start();

    comm.send_receive_replace_n(&boundary_value, 1, /* dest = */ prev, /* source = */ next, 1, 1);
    
    if(comm.rank() != comm.size() - 1){
      CHECK(boundary_value == part.end());
    } else {
      CHECK(boundary_value == 0);
    }
  }
	
	SECTION("Scaled - Location"){
			
		for(long ig = part.start(); ig < part.end(); ig++){
			CHECK(part.location(ig) == comm.rank());
		}
	}

	SECTION("Small sizes"){
		for(int ii = 0; ii < 20; ii++) inq::parallel::partition part(ii, comm);
	}

	SECTION("Check partition sizes 16 in 4"){
		inq::parallel::partition part(16, 4, 0);

		CHECK(part.max_local_size() == 4);
		CHECK(part.local_size() == 4);
		
		CHECK(part.local_size(0) == 4);
		CHECK(part.local_size(1) == 4);
		CHECK(part.local_size(2) == 4);
		CHECK(part.local_size(3) == 4);

		CHECK(part.start(0) == 0);
		CHECK(part.start(1) == 4);
		CHECK(part.start(2) == 8);
		CHECK(part.start(3) == 12);

		CHECK(part.end(0) == 4);
		CHECK(part.end(1) == 8);
		CHECK(part.end(2) == 12);
		CHECK(part.end(3) == 16);		
		
		CHECK(part.waste() == 0.0_a);
	}
	
	SECTION("Check partition sizes 16 in 5"){
		inq::parallel::partition part(16, 5, 0);

		CHECK(part.max_local_size() == 4);
		CHECK(part.local_size() == 4);
		
		CHECK(part.local_size(0) == 4);
		CHECK(part.local_size(1) == 4);
		CHECK(part.local_size(2) == 4);
		CHECK(part.local_size(3) == 4);
		CHECK(part.local_size(4) == 0);

		CHECK(part.start(0) == 0);
		CHECK(part.start(1) == 4);
		CHECK(part.start(2) == 8);
		CHECK(part.start(3) == 12);
		CHECK(part.start(4) == 16);

		CHECK(part.end(0) == 4);
		CHECK(part.end(1) == 8);
		CHECK(part.end(2) == 12);
		CHECK(part.end(3) == 16);
		CHECK(part.end(4) == 16);

		CHECK(part.waste() == 0.25_a);		
	}

	SECTION("Check partition sizes 17 in 5"){
		inq::parallel::partition part(17, 5, 0);

		CHECK(part.max_local_size() == 4);
		CHECK(part.local_size() == 4);
		
		CHECK(part.local_size(0) == 4);
		CHECK(part.local_size(1) == 4);
		CHECK(part.local_size(2) == 4);
		CHECK(part.local_size(3) == 4);
		CHECK(part.local_size(4) == 1);

		CHECK(part.start(0) == 0);
		CHECK(part.start(1) == 4);
		CHECK(part.start(2) == 8);
		CHECK(part.start(3) == 12);
		CHECK(part.start(4) == 16);

		CHECK(part.end(0) == 4);
		CHECK(part.end(1) == 8);
		CHECK(part.end(2) == 12);
		CHECK(part.end(3) == 16);
		CHECK(part.end(4) == 17);

		CHECK(part.waste() == 0.1764705882_a);
	}

	SECTION("Shift"){
		inq::parallel::partition part(17, 5, 0);

		CHECK(part.max_local_size() == 4);

		//part 0
		CHECK(part.rank()       == 0);
		CHECK(part.local_size() == 4);
		CHECK(part.start()      == 0);
		CHECK(part.end()        == 4);

		//part 1
		part.shift();
		CHECK(part.rank()       == 1);
		CHECK(part.local_size() == 4);
		CHECK(part.start()      == 4);
		CHECK(part.end()        == 8);

		//part 2
		part.shift();
		CHECK(part.rank()       == 2);
		CHECK(part.local_size() == 4);
		CHECK(part.start()      == 8);
		CHECK(part.end()        == 12);

		//part 3
		part.shift();
		CHECK(part.rank()       == 3);
		CHECK(part.local_size() == 4);
		CHECK(part.start()      == 12);
		CHECK(part.end()        == 16);

		//part 4
		part.shift();
		CHECK(part.rank()       == 4);
		CHECK(part.local_size() == 1);
		CHECK(part.start()      == 16);
		CHECK(part.end()        == 17);

		//part 0
		part.shift();
		CHECK(part.rank()       == 0);
		CHECK(part.local_size() == 4);
		CHECK(part.start()      == 0);
		CHECK(part.end()        == 4);

	}

	

}
#endif
