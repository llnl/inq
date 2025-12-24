/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATH__NEXTPROD
#define INQ__MATH__NEXTPROD

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <stdexcept>
#include <limits>
#include <numeric>

namespace inq {
namespace math {

// calculates the smallest number that is larger than n and it is a
// multplication of powers of the prime factors.
long nextprod(std::vector<int> const & factors, long n) {
	if (n < 0) throw std::invalid_argument("nextprod: n must be non-negative");
	if (*std::min_element(factors.begin(), factors.end()) <= 1) throw std::invalid_argument("nextprod: all factors must be larger than 1");	

	if (n <= 1) return 1;
  
	// Use a min-priority queue to efficiently find the smallest candidate
	std::priority_queue<long, std::vector<long>, std::greater<long>> pq;
	// Use a set to keep track of visited numbers and avoid duplicates
	std::set<long> visited;
	
	// Initialize the priority queue with the factors themselves
	for (int f : factors) {
		pq.push(f);
		visited.insert(f);
	}
	
	long current_prod = 1;
	while (!pq.empty()) {
		current_prod = pq.top();
		pq.pop();
		
		if (current_prod >= n) return current_prod;

		// Generate new candidates by multiplying the current product by each factor
		for (int f : factors) {
			// Check for potential overflow before multiplication
			if (std::numeric_limits<long>::max()/f < current_prod) continue; // Skip if multiplication would overflow

			long next_candidate = current_prod*f;
			if (visited.find(next_candidate) != visited.end()) continue;
			pq.push(next_candidate);
			visited.insert(next_candidate);
		}
	}
  
	return current_prod; // Should not be reached if factors are provided
}

}
}

#endif

#ifdef INQ_MATH_NEXTPROD_UNIT_TEST
#undef INQ_MATH_NEXTPROD_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;

	CHECK(math::nextprod({2, 3, 5, 7, 11}, 0) == 1);
	CHECK(math::nextprod({2, 3, 5, 7, 11}, 1) == 1);
	CHECK(math::nextprod({2, 3}, 10) == 12);
	CHECK(math::nextprod({2, 3}, 20) == 24);
	CHECK(math::nextprod({2, 3, 5}, 100) == 100);
	CHECK(math::nextprod({7}, 50) == 343);
	CHECK(math::nextprod({2, 3, 5, 7}, 237) == 240);
	CHECK(math::nextprod({2, 3, 5, 7, 11}, 1078) == 1078);
}

#endif
