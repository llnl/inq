/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INPUT__ENVIRONMENT
#define INQ__INPUT__ENVIRONMENT

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/parallelization.hpp>
#include <utils/profiling.hpp>

#include <spdlog/spdlog.h>

#include <inq_config.h>

#include <mpi3/environment.hpp>
#include <cassert>
#include <optional>

namespace inq {
namespace input {

class environment {
		
	boost::mpi3::environment mpi_env_;
	cali::ConfigManager calimgr_;
	mutable parallel::communicator base_comm_;

	static auto & threaded_impl() {
		static bool threaded_ = false;
		
		return threaded_;
	}
	
	auto initialization_(bool use_threads){
		
		if(use_threads) assert(mpi_env_.thread_support() == boost::mpi3::thread_level::multiple);
		
		threaded_impl() = use_threads;
		
		if(not use_threads and base_comm_.rank() == 0){
			calimgr_.add("runtime-report");
			calimgr_.start();
		}
		
		CALI_MARK_BEGIN("inq_environment");		

	}
	
public:
	
	static auto const & threaded() {
		return threaded_impl();
	}

	environment(bool use_threads = false):
		mpi_env_(use_threads?boost::mpi3::thread_level::multiple:boost::mpi3::thread_level::single),
		base_comm_(mpi_env_.get_world_instance())
	{
		initialization_(use_threads);
	}
	
	~environment(){
		
		CALI_MARK_END("inq_environment");
		
		base_comm_.barrier();
		
		if(not threaded() and base_comm_.rank() == 0){
			calimgr_.flush(); // write performance results
		}
		
	}
	
	auto par() const {
		return parallelization(base_comm_);
	}

	auto & comm() const {
		return base_comm_;
	}

	static auto const & global() {
		static environment global_env;
		return global_env;
	}
		
};

}
}
#endif

#ifdef INQ_INPUT_ENVIRONMENT_UNIT_TEST
#undef INQ_INPUT_ENVIRONMENT_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
