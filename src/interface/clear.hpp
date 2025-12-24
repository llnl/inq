/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__CLEAR
#define INQ__INTERFACE__CLEAR

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <input/environment.hpp>
#include <interface/actions.hpp>
#include <interface/runtime_options.hpp>

namespace inq {
namespace interface {

struct {

	constexpr auto name() const {
		return "clear";
	}

	constexpr auto one_line() const {
		return "Removes any inq information from the current directory";
	}

	constexpr auto help() const {
		return R""""(

The 'clear' command
===================

The 'clear' command removes all inq information from the current
directory. It doesn't take any arguments.

Shell example:  `pinq clear`
Python example: `pinq.clear()`


)"""";
	}
	
	static void clear() {
		if(input::environment::global().comm().root()) std::filesystem::remove_all(".inq");
		input::environment::global().comm().barrier();
	}
	
	void operator()() const {
		clear();
	}

	template <typename ArgsType>
	void command(ArgsType const & args, runtime_options const &) const {
		if(args.size() != 0) actions::error(input::environment::global().comm(), "The 'clear' command doesn't take arguments");
		clear();
		actions::normal_exit();
	}

#ifdef INQ_PYTHON_INTERFACE
	template <class PythonModule>
	void python_interface(PythonModule & module) const {
		module.def("clear", &clear, help());
	}
#endif
	
}	const clear;

}
}
#endif

#ifdef INQ_INTERFACE_CLEAR_UNIT_TEST
#undef INQ_INTERFACE_CLEAR_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

}
#endif
