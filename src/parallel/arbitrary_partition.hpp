/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PARALLEL__ARBITRARY_PARTITION
#define INQ__PARALLEL__ARBITRARY_PARTITION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <parallel/global_index.hpp>
#include <parallel/communicator.hpp>
#include <parallel/partition.hpp>

#include <cassert>
#include <array>

namespace inq{
namespace parallel {

class arbitrary_partition {

	long local_size_;
	std::vector<long> lsizes_;
	std::vector<long> starts;
	long size_;
	long max_local_size_;
	long start_;
	long end_;
	int comm_size_;
	int rank_;
	
public:
	
	auto local_size() const {
		return local_size_;
	}

	arbitrary_partition() = default;
	
	template <typename CommType>
	arbitrary_partition(long const local_size, CommType comm):
		local_size_(local_size),
		lsizes_(comm.size()),
		comm_size_(comm.size()),
		rank_(comm.rank())
	{

		MPI_Allgather(&local_size, 1, MPI_LONG, lsizes_.data(), 1, MPI_LONG, comm.get());

		assert(lsizes_[rank_] == local_size);
		
		size_ = 0;
		max_local_size_ = 0;
		start_ = 0;
		end_ = 0;
		
		for(unsigned ipart = 0; ipart < lsizes_.size(); ipart++){
			size_ += lsizes_[ipart];
			max_local_size_ = std::max(max_local_size_, lsizes_[ipart]);
			if(ipart < (unsigned) rank_) start_ += lsizes_[ipart];
		}
		end_ = start_ + local_size_;
	}
	
	arbitrary_partition(parallel::partition const & part):
		local_size_(part.local_size()),
		lsizes_(part.comm_size()),
		comm_size_(part.comm_size())
	{
		for(unsigned ipart = 0; ipart < lsizes_.size(); ipart++) lsizes_[ipart] = part.local_size(ipart);

		size_ = part.size();
		max_local_size_ = part.max_local_size();
		start_ = part.start();
		end_ = part.end();
	}

	template <typename LocalSizes>
	arbitrary_partition(LocalSizes const & local_sizes, int rank):
		local_size_(local_sizes[rank]),
		lsizes_(local_sizes.begin(), local_sizes.end()),
		comm_size_(local_sizes.size()),
		rank_(rank)
	{

		size_ = 0;
		max_local_size_ = 0;
		start_ = 0;
		end_ = 0;

		for(unsigned ipart = 0; ipart < lsizes_.size(); ipart++){
			size_ += lsizes_[ipart];
			max_local_size_ = std::max(max_local_size_, lsizes_[ipart]);
			if(ipart < (unsigned) rank_) start_ += lsizes_[ipart];
		}
		end_ = start_ + local_size_;
	}

	template <typename Integer>
	arbitrary_partition(std::initializer_list<Integer> const & local_sizes, int rank):
		arbitrary_partition(std::vector<Integer>{local_sizes.begin(), local_sizes.end()}, rank){
	}
	
	auto size() const {
		return size_;
	}
	
	constexpr auto start() const {
		return start_;
	}

	auto start(int part) const {
		long val = 0;
		for(int ipart = 0; ipart < part; ipart++) val += lsizes_[ipart];
		return val;
	}
	
	constexpr auto end() const {
		return end_;
	}

	auto end(int part) const {
		long val = 0;
		for(int ipart = 0; ipart <= part; ipart++) val += lsizes_[ipart];
		return val;
	}
	
	auto local_size(int part) const {
		return lsizes_[part];
	}
	
	auto parallel() const {
		return comm_size_ > 1;
	}
	
	auto contains(long index) const {
		return start() <= index and index < end();
	}

	auto contains(long index, int part) const {
		return start(part) <= index and index < end(part);
	}
	
	constexpr auto local_to_global(long local_i) const {
		return global_index(start_ + local_i);
	}
	
	constexpr auto global_to_local(global_index global_i) const {
		return global_i.value() - start_;
	}
	
	auto comm_size() const {
		return comm_size_;
	}

	auto max_local_size() const {
		return max_local_size_;
	}
	
	auto waste() const {
		auto total_elements = max_local_size()*comm_size();
		return (total_elements - size())/double(size());
	}
	
	constexpr auto & rank() const {
		return rank_;
	}

	void shift() {
		rank_++;
		if(rank_ == comm_size_) rank_ = 0;
		local_size_ = local_size(rank_);
		start_      = start(rank_);
		end_        = end(rank_);
	}
	
};
}
}
#endif

#ifdef INQ_PARALLEL_ARBITRARY_PARTITION_UNIT_TEST
#undef INQ_PARALLEL_ARBITRARY_PARTITION_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
  
	using namespace inq;
	using namespace Catch::literals;
  
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
  
	auto local_size = 100 + 3*comm.rank();

	inq::parallel::arbitrary_partition part(local_size, comm);

	CHECK(comm.rank() == part.rank());
	CHECK(comm.size() == part.comm_size());
	
  auto next = comm.rank() + 1;
  if(next == comm.size()) next = 0;
  
  auto prev = comm.rank() - 1;
  if(prev == -1) prev = comm.size() - 1;

  SECTION("Total"){
		
		auto total_size = local_size;

		comm.all_reduce_in_place_n(&total_size, 1, std::plus<>{});
    CHECK(total_size == part.size());

    auto calculated_size = part.local_size();
    
    comm.all_reduce_in_place_n(&calculated_size, 1, std::plus<>{});
    
    CHECK(total_size == calculated_size);

		CHECK(part.start() == part.start(comm.rank()));
		CHECK(part.end() == part.end(comm.rank()));
		
  }

	SECTION("Waste"){
		if(comm.size() == 1) CHECK(part.waste() == 0.0_a);
		if(comm.size() == 2) CHECK(part.waste() == 0.0147783251_a);
		if(comm.size() == 3) CHECK(part.waste() == 0.0291262136_a);
		if(comm.size() == 4) CHECK(part.waste() == 0.043062201_a);
		if(comm.size() == 5) CHECK(part.waste() == 0.0566037736_a);
	}

  SECTION("Upper bound"){
    auto boundary_value = part.end();
    
    comm.send_receive_replace_n(&boundary_value, 1, /* dest = */ next, /* source = */ prev, 0, 0);
    
    if(comm.rank() != 0){
      CHECK(boundary_value == part.start());
    } else {
      CHECK(boundary_value == part.size());
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

	SECTION("Construct from parallel:partition"){

		inq::parallel::partition part(1384, comm);

		auto apart = parallel::arbitrary_partition(part);

		CHECK(apart.size() == part.size());
		CHECK(apart.local_size() == part.local_size());
		CHECK(apart.start() == part.start());
		CHECK(apart.end() == part.end());

		for(int ipart = 0; ipart < comm.size(); ipart++){
			CHECK(apart.local_size(ipart) == part.local_size(ipart));
			CHECK(apart.start(ipart) == part.start(ipart));
			CHECK(apart.end(ipart) == part.end(ipart));
		}

		for(int ii = 0; ii < part.size(); ii++){
			CHECK(apart.contains(ii) == part.contains(ii));
			for(int ipart = 0; ipart < comm.size(); ipart++) CHECK(apart.contains(ii, ipart) == part.contains(ii, ipart));
		}
			
	}

	SECTION("Construct from array"){
		auto apart = parallel::arbitrary_partition({12, 4, 7, 10, 2}, 1);

		CHECK(apart.size()            == 35);
		CHECK(apart.comm_size()       == 5);
		CHECK(apart.max_local_size()  == 12);

		CHECK(apart.rank()        == 1);
		CHECK(apart.local_size()  == 4);
		CHECK(apart.start()       == 12);
		CHECK(apart.end()         == 16);

		CHECK(apart.local_size(0) == 12);
		CHECK(apart.start(0)      == 0);
		CHECK(apart.end(0)        == 12);

		CHECK(apart.local_size(1) == 4);
		CHECK(apart.start(1)      == 12);
		CHECK(apart.end(1)        == 16);

		CHECK(apart.local_size(2) == 7);
		CHECK(apart.start(2)      == 16);
		CHECK(apart.end(2)        == 23);

		CHECK(apart.local_size(3) == 10);
		CHECK(apart.start(3)      == 23);
		CHECK(apart.end(3)        == 33);

		CHECK(apart.local_size(4) == 2);
		CHECK(apart.start(4)      == 33);
		CHECK(apart.end(4)        == 35);
	
	}

		SECTION("Shift"){
		auto apart = parallel::arbitrary_partition({12, 4, 7, 10, 2}, 1);

		CHECK(apart.size()            == 35);
		CHECK(apart.comm_size()       == 5);
		CHECK(apart.max_local_size()  == 12);

		CHECK(apart.rank()        == 1);
		CHECK(apart.local_size()  == 4);
		CHECK(apart.start()       == 12);
		CHECK(apart.end()         == 16);

		apart.shift();
		CHECK(apart.rank()        == 2);
		CHECK(apart.local_size()  == 7);
		CHECK(apart.start()       == 16);
		CHECK(apart.end()         == 23);

		apart.shift();
		CHECK(apart.rank()        == 3);
		CHECK(apart.local_size()  == 10);
		CHECK(apart.start()       == 23);
		CHECK(apart.end()         == 33);

		apart.shift();
		CHECK(apart.rank()        == 4);
		CHECK(apart.local_size()  == 2);
		CHECK(apart.start()       == 33);
		CHECK(apart.end()         == 35);

		apart.shift();
		CHECK(apart.rank()        == 0);
		CHECK(apart.local_size()  == 12);
		CHECK(apart.start()       == 0);
		CHECK(apart.end()         == 12);

		apart.shift();
		CHECK(apart.rank()        == 1);
		CHECK(apart.local_size()  == 4);
		CHECK(apart.start()       == 12);
		CHECK(apart.end()         == 16);


	
	}

}
#endif
