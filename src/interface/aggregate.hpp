/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__INTERFACE__AGGREGATE
#define INQ__INTERFACE__AGGREGATE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>
#include <interface/runtime_options.hpp>


namespace inq {
namespace interface {

auto list_item(std::string const & name, std::string const & one_line){
	auto align = 35ul;
	assert(name.size() < align);
	auto pad = std::string(align - name.size(), ' ');
	return "  " + name + pad + one_line + '\n';
}

template <typename Type>
struct item : public Type {

	item(Type const & arg):
		Type(arg){
	}
	
	auto list() const {
		return list_item(Type::name(), Type::one_line());
	}

	auto execute(std::string const & comm, std::vector<std::string> const & args, runtime_options const & run_opts) const {
		if(comm == Type::name()) Type::command(args, run_opts);
	}

	auto help(std::string const & comm) const {
		if(comm == Type::name()) {
			std::cout << Type::help();
			actions::normal_exit();
		}
	}
	
};

template <typename TypeA, typename TypeB>
class aggregate {

  TypeA agg1_;
  TypeB agg2_;
  
public:

  aggregate(TypeA arg_agg1, TypeB arg_agg2):
    agg1_(arg_agg1),
    agg2_(arg_agg2)
  {
  }

	auto list() const {
		return agg1_.list() + agg2_.list();
	}

	auto execute(std::string const & comm, std::vector<std::string> const & args, runtime_options const & run_opts) const {
		agg1_.execute(comm, args, run_opts);
		agg2_.execute(comm, args, run_opts);
	}

	auto help(std::string const & comm) const {
		agg1_.help(comm);
		agg2_.help(comm);
	}
	
};

template <typename TypeA, typename TypeB>
auto operator+(item<TypeA> agg1, item<TypeB> agg2){
  return interface::aggregate(agg1, agg2);
}

template <typename TypeA, typename TypeB, typename TypeC>
auto operator+(aggregate<TypeA, TypeB> agg1, item<TypeC> agg2){
  return interface::aggregate(agg1, agg2);
}


template <typename TypeA, typename TypeB, typename TypeC>
auto operator+(item<TypeA> agg1, aggregate<TypeB, TypeC> agg2){
  return interface::aggregate(agg1, agg2);
}

}
}
#endif

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef INQ_INTERFACE_AGGREGATE_UNIT_TEST
#undef INQ_INTERFACE_AGGREGATE_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

#include <interface/cell.hpp>
#include <interface/ions.hpp>
#include <interface/run.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using namespace magnitude;
	
	using namespace interface;
	
	auto aggregate = item(interface::cell) + item(interface::ions) + item(interface::run);

	std::cout << aggregate.list() << std::endl;
	
}

#endif
