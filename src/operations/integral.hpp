/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPERATIONS__INTEGRAL
#define INQ__OPERATIONS__INTEGRAL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <numeric>

#include <basis/field.hpp>
#include <basis/field_set.hpp>
#include <operations/sum.hpp>
#include <math/complex.hpp>

namespace inq {
namespace operations {

template <class BasisType, class ElementType>
auto integral(basis::field<BasisType, ElementType> const & phi){
	CALI_CXX_MARK_FUNCTION;
	
	auto integral_value = phi.basis().volume_element()*sum(phi.linear());
	if(phi.basis().comm().size() > 1) phi.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType>
auto integral_sum(basis::field_set<BasisType, ElementType> const & phi){
	CALI_CXX_MARK_FUNCTION;
	
	auto integral_value = phi.basis().volume_element()*sum(phi.matrix().flatted());
	if(phi.full_comm().size() > 1) phi.full_comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType>
ElementType integral_partial_sum(basis::field_set<BasisType, ElementType> const & phi, int const & max_index){
	CALI_CXX_MARK_FUNCTION;

	assert(phi.local_set_size() >= max_index);
	basis::field<BasisType, ElementType> rphi(phi.basis());
	rphi.fill(0.0);
	gpu::run(phi.basis().local_size(),
			[ph = begin(phi.matrix()), rph = begin(rphi.linear()), mi = max_index] GPU_LAMBDA (auto ip){ 
				for (auto i=0; i<mi; i++) {
					rph[ip] += ph[ip][i];
				}
			});
	auto integral_value = rphi.basis().volume_element()*sum(rphi.linear());
	if (phi.full_comm().size() > 1) phi.full_comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType>
double integral_abs(basis::field<BasisType, ElementType> const & phi){
	CALI_CXX_MARK_FUNCTION;

	gpu::array<ElementType, 1> sum_array(phi.basis().local_size());
	gpu::run(phi.basis().local_size(),
					 [su = begin(sum_array), ph = begin(phi.linear())] GPU_LAMBDA (auto ip) {
						 su[ip] = fabs(ph[ip]);
					 });

	auto integral_value = phi.basis().volume_element()*sum(sum_array);
	if(phi.basis().comm().size() > 1) phi.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType1, class ElementType2>
auto integral_product(basis::field<BasisType, ElementType1> const & phi1, basis::field<BasisType, ElementType2> const & phi2) -> decltype(ElementType1{}*ElementType2{}) {
	CALI_CXX_MARK_FUNCTION;

	assert(phi1.basis() == phi2.basis());
	
	using type = decltype(ElementType1{}*ElementType2{});
	gpu::array<type, 1> sum_array(phi1.basis().local_size());

	gpu::run(phi1.basis().local_size(),
					 [su = begin(sum_array), p1 = begin(phi1.linear()), p2 = begin(phi2.linear())] GPU_LAMBDA (auto ip) {
						 su[ip] = p1[ip]*p2[ip];
					 });

	auto integral_value = phi1.basis().volume_element()*sum(sum_array);
	if(phi1.basis().comm().size() > 1) phi1.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType1, class ElementType2>
auto integral_product_sum(basis::field_set<BasisType, ElementType1> const & phi1, basis::field_set<BasisType, ElementType2> const & phi2) -> decltype(ElementType1{}*ElementType2{}) {
	CALI_CXX_MARK_FUNCTION;

	assert(phi1.basis() == phi2.basis());
	
	using type = decltype(ElementType1{}*ElementType2{});
	gpu::array<type, 1> sum_array(phi1.matrix().flatted().size());

	gpu::run(phi1.matrix().flatted().size(),
					 [su = begin(sum_array), p1 = begin(phi1.matrix().flatted()), p2 = begin(phi2.matrix().flatted())] GPU_LAMBDA (auto ip) {
						 su[ip] = p1[ip]*p2[ip];
					 });

	auto integral_value = phi1.basis().volume_element()*sum(sum_array);
	if(phi1.basis().comm().size() > 1) phi1.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType1, class ElementType2>
double integral_absdiff(basis::field<BasisType, ElementType1> const & phi1, basis::field<BasisType, ElementType2> const & phi2){
	CALI_CXX_MARK_FUNCTION;
	
	assert(phi1.basis() == phi2.basis());
	
	gpu::array<double, 1> sum_array(phi1.basis().local_size());

	gpu::run(phi1.basis().local_size(),
					 [su = begin(sum_array), p1 = begin(phi1.linear()), p2 = begin(phi2.linear())] GPU_LAMBDA (auto ip) {
						 su[ip] = fabs(p1[ip] - p2[ip]);
					 });

	auto integral_value = phi1.basis().volume_element()*sum(sum_array);
	if(phi1.basis().comm().size() > 1) phi1.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template <class BasisType, class ElementType1, class ElementType2>
double integral_sum_absdiff(basis::field_set<BasisType, ElementType1> const & phi1, basis::field_set<BasisType, ElementType2> const & phi2){
	CALI_CXX_MARK_FUNCTION;

	assert(phi1.basis() == phi2.basis());
	
	gpu::array<double, 1> sum_array(phi1.matrix().flatted().size());

	gpu::run(phi1.matrix().flatted().size(),
					 [su = begin(sum_array), p1 = begin(phi1.matrix().flatted()), p2 = begin(phi2.matrix().flatted())] GPU_LAMBDA (auto ip) {
						 su[ip] = fabs(p1[ip] - p2[ip]);
					 });

	auto integral_value = phi1.basis().volume_element()*sum(sum_array);
	if(phi1.basis().comm().size() > 1) phi1.basis().comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
	return integral_value;
}

template<class BasisType, class ElementType>
std::vector<ElementType> integral_local(basis::field_set<BasisType, ElementType> const & phi){
	CALI_CXX_MARK_FUNCTION;

	auto n = phi.set_size();
	std::vector<ElementType> integral_set(n);
	for (auto i = 0; i < n; i++){
		basis::field<BasisType, ElementType> phi_loc(phi.basis());
		gpu::run(phi.basis().local_size(),
			[ph = begin(phi.matrix()), phl = begin(phi_loc.linear()), index = i] GPU_LAMBDA (auto ip){
				phl[ip] = ph[ip][index];
			});
		auto integral_value = phi_loc.basis().volume_element()*sum(phi_loc.linear());
		if (phi.full_comm().size() > 1) phi.full_comm().all_reduce_in_place_n(&integral_value, 1, std::plus<>{});
		integral_set[i] = integral_value;
	}
	return integral_set;
}

}
}
#endif

#ifdef INQ_OPERATIONS_INTEGRAL_UNIT_TEST
#undef INQ_OPERATIONS_INTEGRAL_UNIT_TEST

#include <basis/field.hpp>
#include <catch2/catch_all.hpp>
#include <basis/trivial.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	const int N = 1000;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
		
	basis::trivial bas(N, comm);
	
	SECTION("Integral double"){
		
		basis::field<basis::trivial, double> aa(bas);

		aa.fill(1.0);

		CHECK(operations::integral(aa) == 1.0_a);

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	aa.linear()[ii] = aa.basis().part().local_to_global(ii).value();

		CHECK(operations::integral(aa) == Approx(0.5*N*(N - 1.0)*bas.volume_element()));

	}

	SECTION("Integral complex"){
		
		basis::field<basis::trivial, complex> aa(bas);

		aa.fill(complex(1.0, 1.0));

		CHECK(real(operations::integral(aa)) == 1.0_a);
		CHECK(imag(operations::integral(aa)) == 1.0_a);

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) {
			auto iig = aa.basis().part().local_to_global(ii);
			aa.linear()[ii] = complex(iig.value(), -3.0*iig.value());
		}

		CHECK(real(operations::integral(aa)) == Approx(0.5*N*(N - 1.0)*bas.volume_element()));
		CHECK(imag(operations::integral(aa)) == Approx(-1.5*N*(N - 1.0)*bas.volume_element()));

	}

	SECTION("integral_sum double"){

		int nvec = 6;
		
		basis::field_set<basis::trivial, double> aa(bas, nvec);

		aa.fill(1.0);

		CHECK(operations::integral_sum(aa) == Approx(nvec));

		for(int ii = 0; ii < aa.basis().part().local_size(); ii++) {
			for(int ist = 0; ist < nvec; ist++){
				aa.matrix()[ii][ist] = aa.basis().part().local_to_global(ii).value();
			}
		}

		CHECK(operations::integral_sum(aa) == Approx(nvec*0.5*N*(N - 1.0)*bas.volume_element()));
	}
	
	SECTION("Integral product double"){
		
		basis::field<basis::trivial, double> aa(bas);
		basis::field<basis::trivial, double> bb(bas);
		
		aa.fill(2.0);
		bb.fill(0.8);
		
		CHECK(operations::integral_product(aa, bb) == 1.6_a);
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	{
			auto iig = aa.basis().part().local_to_global(ii);
			aa.linear()[ii] = pow(iig.value() + 1, 2);
			bb.linear()[ii] = 1.0/(iig.value() + 1);
		}
		
		CHECK(operations::integral_product(aa, bb) == Approx(0.5*N*(N + 1.0)*bas.volume_element()));
		
	}
	
	SECTION("Integral product complex"){
		
		basis::field<basis::trivial, complex> aa(bas);
		basis::field<basis::trivial, complex> bb(bas);
		
		aa.fill(complex(2.0, -0.3));
		bb.fill(complex(0.8, 0.01));
		
		CHECK(real(operations::integral_product(aa, bb)) == 1.603_a);
		CHECK(imag(operations::integral_product(aa, bb)) == -0.22_a);
		
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	{
			auto iig = aa.basis().part().local_to_global(ii);
			aa.linear()[ii] = pow(iig.value() + 1, 2)*exp(complex(0.0, M_PI/8 + M_PI/7*iig.value()));
			bb.linear()[ii] = 1.0/(iig.value() + 1)*exp(complex(0.0, M_PI/8 - M_PI/7*iig.value()));
		}
		
		CHECK(real(operations::integral_product(aa, bb)) == Approx(sqrt(2.0)*0.25*N*(N + 1.0)*bas.volume_element()));
		CHECK(real(operations::integral_product(aa, bb)) == Approx(sqrt(2.0)*0.25*N*(N + 1.0)*bas.volume_element()));
		
	}
	
	
	SECTION("Integral absdiff double"){
		
		basis::field<basis::trivial, double> aa(bas);
		basis::field<basis::trivial, double> bb(bas);
		
		aa.fill(-13.23);
		bb.fill(-13.23);
		
		CHECK(fabs(operations::integral_absdiff(aa, bb)) < 1e-14);

		double sign = 1.0;
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	{
			auto iig = aa.basis().part().local_to_global(ii);
			aa.linear()[ii] = sign*2.0*(iig.value() + 1);
			bb.linear()[ii] = sign*1.0*(iig.value() + 1);
			sign *= -1.0;
		}
		
		CHECK(operations::integral_absdiff(aa, bb) == Approx(0.5*N*(N + 1.0)*bas.volume_element()));
		
	}
	
	SECTION("Integral absdiff complex"){
		
		basis::field<basis::trivial, complex> aa(bas);
		basis::field<basis::trivial, complex> bb(bas);
		
		aa.fill(-13.23*exp(complex(0.0, M_PI/3.63)));
		bb.fill(-13.23*exp(complex(0.0, M_PI/3.63)));
		
		CHECK(fabs(operations::integral_absdiff(aa, bb)) < 1e-14);

		double sign = 1.0;
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	{
			auto iig = aa.basis().part().local_to_global(ii);
			aa.linear()[ii] = sign*2.0*(iig.value() + 1)*exp(complex(0.0, 0.123*iig.value()));
			bb.linear()[ii] = sign*1.0*(iig.value() + 1)*exp(complex(0.0, 0.123*iig.value()));
			sign *= -1.0;
		}
		
		CHECK(operations::integral_absdiff(aa, bb) == Approx(0.5*N*(N + 1.0)*bas.volume_element()));
		
	}

	SECTION("integral_sum_absdiff double"){
		int nvec = 6;
		
		basis::field_set<basis::trivial, double> aa(bas, nvec);
		basis::field_set<basis::trivial, double> bb(bas, nvec);
		
		aa.fill(-13.23);
		bb.fill(-13.23);
		
		CHECK(fabs(operations::integral_sum_absdiff(aa, bb)) < 1e-14);

		double sign = 1.0;
		for(int ii = 0; ii < aa.basis().part().local_size(); ii++)	{
			for(int ist = 0; ist < nvec; ist++){
				auto iig = aa.basis().part().local_to_global(ii);
				aa.matrix()[ii][ist] = sign*2.0*(iig.value() + 1);
				bb.matrix()[ii][ist] = sign*1.0*(iig.value() + 1);
				sign *= -1.0;
			}
		}
		
		CHECK(operations::integral_sum_absdiff(aa, bb) == Approx(nvec*0.5*N*(N + 1.0)*bas.volume_element()));
		
	}
	
}
#endif
