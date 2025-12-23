/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PARALLEL__ALLTOALL
#define INQ__PARALLEL__ALLTOALL

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

template <typename ArrayType>
void alltoall(ArrayType & buf, parallel::communicator & comm){
	CALI_CXX_MARK_FUNCTION;

	assert(buf.size() == comm.size());
	if(comm.size() == 1) return;
	
	using type = typename ArrayType::element_type;
	auto mpi_type = boost::mpi3::detail::basic_datatype<type>();
	auto count = buf[0].num_elements();

	auto method = std::getenv("INQ_COMM");

	if(method == NULL or method == std::string("collective")){

		{
			CALI_CXX_MARK_SCOPE("alltoall:mpi");
			MPI_Alltoall(MPI_IN_PLACE, count, mpi_type, raw_pointer_cast(buf.data_elements()), count, mpi_type, comm.get());
		}
		
	} else if(method == std::string("point")) {

#ifdef HAVE_MPI_ISENDRECV_REPLACE

		std::vector<MPI_Request> reqs(comm.size(), MPI_REQUEST_NULL);

		CALI_CXX_MARK_SCOPE("alltoall:mpi");
		
		for(int iproc = 0; iproc < comm.size(); iproc++){
			MPI_Isendrecv_replace(raw_pointer_cast(buf[iproc].base()), count, mpi_type, iproc, iproc, iproc, comm.rank(), comm.get(), &reqs[iproc]);
		}
		
#else

		ArrayType copy(buf);
		std::vector<MPI_Request> reqs(comm.size()*2, MPI_REQUEST_NULL);
		
		CALI_CXX_MARK_SCOPE("alltoall:mpi");
		
		for(int iproc = 0; iproc < comm.size(); iproc++){
			MPI_Irecv(raw_pointer_cast(buf[iproc].base()), count, mpi_type, iproc, comm.rank(), comm.get(), &reqs[2*iproc]);
			MPI_Isend(raw_pointer_cast(copy[iproc].base()), count, mpi_type, iproc, iproc, comm.get(), &reqs[2*iproc + 1]);
		}

#endif

		std::vector<MPI_Status> stats(reqs.size());
		MPI_Waitall(reqs.size(), reqs.data(), stats.data());
		
	} else if(method == std::string("nccl")) {

#ifndef ENABLE_NCCL
		assert(false and "inq was compiled without nccl support");		
#else
		comm.nccl_init();
		
		ArrayType copy(buf);

		CALI_CXX_MARK_SCOPE("alltoall:nccl");

		ncclGroupStart();
		for(int iproc = 0; iproc < comm.size(); iproc++){
			ncclRecv(raw_pointer_cast(buf[iproc].base()), count*sizeof(type)/sizeof(double), ncclDouble, iproc, &comm.nccl_comm(), 0);
			ncclSend(raw_pointer_cast(copy[iproc].base()), count*sizeof(type)/sizeof(double), ncclDouble, iproc, &comm.nccl_comm(), 0);
		}
		ncclGroupEnd();

		gpu::sync();
#endif

	} else {
		assert(false and "uknown communication method");		
	}	
}

}
}
#endif

#ifdef INQ_PARALLEL_ALLTOALL_UNIT_TEST
#undef INQ_PARALLEL_ALLTOALL_UNIT_TEST

#include <gpu/run.hpp>
#include <mpi3/environment.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

  using namespace inq;
  using namespace Catch::literals;
  
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
  
  int blocksize = 100;
  
  gpu::array<int, 2> buffer({comm.size(), blocksize}, comm.rank());

  parallel::alltoall(buffer, comm);
	gpu::sync();

  for(int iproc = 0; iproc < comm.size(); iproc++){
    for(int ib = 0; ib < blocksize; ib++){
			CHECK(buffer[iproc][ib] == iproc);
    }
  }

}

#endif
