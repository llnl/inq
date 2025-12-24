/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INPUT__PARALLELIZATION
#define INQ__INPUT__PARALLELIZATION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <parallel/communicator.hpp>

#include <parallel/partition.hpp>
#include <utils/factors.hpp>
#include <utils/num_str.hpp>

namespace inq {
namespace input {

  class parallelization {

  public:

		static auto dimension_kpoints(){
			return 2;
		}
		
		static auto dimension_domains(){
			return 0;
		}
		
		static auto dimension_states(){
			return 1;
		}

		static auto optimal_nprocs(int size, int max_comm_size, double threshold){
			for(utils::factors_reverse fac(max_comm_size); fac != fac.end(); ++fac){
				parallel::partition part(size, *fac, 0);

				if(part.local_size(*fac - 1) == 0) continue; //avoid empty partitions
				if(part.waste() <= threshold) return *fac;
			}
			return 1;
		}

		template <class CommType>
    explicit parallelization(CommType & comm):
			nproc_kpts_(boost::mpi3::fill),
			nproc_states_(1),
			nproc_domains_(boost::mpi3::fill),
      comm_(comm)			
		{
    }

		auto & comm() const {
			return comm_;
		}

		auto cart_comm(int nspin, int nkpoints, int nstates) const {
			assert(nspin == 1 or nspin == 2);

			{
				auto requested_procs = std::max(1, nproc_domains_)*std::max(1, nproc_states_)*std::max(1, nproc_kpts_);
				if(requested_procs > comm_.size()) {
					throw std::runtime_error("INQ Error: the number of processors requested (" + utils::num_to_str("%d", requested_procs) + ") is larger than the number of processors (" + utils::num_to_str("%d", comm_.size()) + ").");
				}
			}
			
			auto avail_procs = comm_.size();
			if(nproc_domains_ != boost::mpi3::fill) {
				if(nproc_domains_ > avail_procs or avail_procs%nproc_domains_ != 0) {
					throw std::runtime_error("INQ Error: invalid number of parallel partitions requested (" + utils::num_to_str("%d", nproc_domains_) + ") for the number of available processors (" + utils::num_to_str("%d", avail_procs) + ").");
				}
				avail_procs /= nproc_domains_;
			}

			if(nproc_states_ != boost::mpi3::fill) {
				if(nproc_states_ > avail_procs or avail_procs%nproc_states_ != 0) {
					throw std::runtime_error("INQ Error: invalid number of parallel states processors requested (" + utils::num_to_str("%d", nproc_states_) + ") for the number of available processors (" + utils::num_to_str("%d", avail_procs) + ").");
				}
				avail_procs /= nproc_states_;
			}

			auto actual_nproc_kpts = optimal_nprocs(nkpoints*nspin, avail_procs, efficiency_threshold);
			if(nproc_kpts_ != boost::mpi3::fill) {
				if(nproc_kpts_ > avail_procs or avail_procs%nproc_kpts_ != 0) {
					throw std::runtime_error("INQ Error: invalid number of parallel kpoints processors requested (" + utils::num_to_str("%d", nproc_kpts_) + ") for the number of available processors (" + utils::num_to_str("%d", avail_procs) + ").");
				}
				actual_nproc_kpts = nproc_kpts_;
			} else {
				avail_procs /= actual_nproc_kpts;
			}

			auto actual_nproc_states = nproc_states_;
			if(actual_nproc_states == boost::mpi3::fill and nproc_domains_ == boost::mpi3::fill) {
				actual_nproc_states = optimal_nprocs(nstates, avail_procs, efficiency_threshold);
			}
			
			std::array<int, 3> nprocs;
			nprocs[dimension_kpoints()] = actual_nproc_kpts;
			nprocs[dimension_domains()] = nproc_domains_;
			nprocs[dimension_states()] = actual_nproc_states;

			assert(std::max(1, nprocs[dimension_kpoints()])*std::max(1, nprocs[dimension_domains()])*std::max(1, nprocs[dimension_states()]) <= comm_.size());
			for(int idim = 0; idim < 3; idim++)	assert(nprocs[idim] == boost::mpi3::fill or comm_.size()%nprocs[idim] == 0);
			
			return parallel::cartesian_communicator<3>(comm_, nprocs);
    }

		auto states(int num = boost::mpi3::fill){
			auto ret = *this;
			ret.nproc_states_ = num;
			return ret;
		}
		
		auto domains(int num = boost::mpi3::fill){
			auto ret = *this;
			ret.nproc_domains_ = num;
			return ret;
		}

		auto kpoints(int num = boost::mpi3::fill){
			auto ret = *this;
			ret.nproc_kpts_ = num;
			return ret;
		}

		auto size() const {
			return comm_.size();
		}

	private:

		int nproc_kpts_;
		int nproc_states_;
		int nproc_domains_;
    mutable parallel::communicator comm_;
		constexpr static double const efficiency_threshold = 0.1;
  };
    
}
}
#endif

#ifdef INQ_INPUT_PARALLELIZATION_UNIT_TEST
#undef INQ_INPUT_PARALLELIZATION_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <mpi3/environment.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

  using namespace inq;
	using namespace Catch::literals;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	input::parallelization par(comm);
	
	auto cart_comm = par.kpoints(1).states(comm.size()).domains(1).cart_comm(2, 10, 23);

	CHECK(par.comm() == comm);
	CHECK(cart_comm.size() == comm.size());

	SECTION("optimize parallelization"){
		CHECK(input::parallelization::optimal_nprocs(16, 4, 0.05) == 4);
		CHECK(input::parallelization::optimal_nprocs(15, 8, 0.1) == 8);
		CHECK(input::parallelization::optimal_nprocs(31, 4, 0.05) == 4);
		CHECK(input::parallelization::optimal_nprocs(20, 38, 0.05) == 2);
		CHECK(input::parallelization::optimal_nprocs(34785, 78, 0.05) == 78);
		CHECK(input::parallelization::optimal_nprocs(12, 64, 0.05) == 4);
		CHECK(input::parallelization::optimal_nprocs(1, 737730, 0.05) == 1);		
	}
	
}
#endif
