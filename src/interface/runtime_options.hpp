/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__RUNTIME_OPTIONS
#define INQ__INTERFACE__RUNTIME_OPTIONS

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

namespace inq {
namespace interface {

struct runtime_options {
	bool quiet;
	bool debug;
	int  par_domains;
	int  par_states;
	int  par_kpoints;
  runtime_options():
		quiet(false),
		debug(false),
		par_domains(boost::mpi3::fill),
		par_states (boost::mpi3::fill),
		par_kpoints(boost::mpi3::fill) {
  }
};

}
}
#endif

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef INQ_INTERFACE_RUNTIME_OPTIONS_UNIT_TEST
#undef INQ_INTERFACE_RUNTIME_OPTIONS_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;
	
	using namespace interface;

	runtime_options ro;

	CHECK(ro.quiet == false);
	CHECK(ro.debug == false);
}

#endif
