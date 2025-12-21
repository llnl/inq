/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__FOURIER_SPACE
#define INQ__BASIS__FOURIER_SPACE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#include <math/vector3.hpp>
#include "grid.hpp"
#include <cassert>
#include <array>

namespace inq {
namespace basis {

class real_space;

class fourier_space : public grid{

	bool reverse_;
	systems::cell cell_;
	
public:

	using reciprocal_space = real_space;
		
	fourier_space(real_space const & rs):
		grid({rs.sizes()[(rs.comm().size() > 1)?2:0], rs.sizes()[1], rs.sizes()[(rs.comm().size() > 1)?0:2]}, rs.comm(), /*par_dim = */ 2),
		reverse_(rs.comm().size() > 1),
		cell_(rs.cell())
	{
	}

	constexpr static auto covspacing(int) {
		return 2.0*M_PI;
	}
	
	auto & cell() const {
		return cell_;
	}

	auto volume_element() const {
		return cell().volume()/(size()*size());
	}
		
	friend auto operator==(const fourier_space & fs1, const fourier_space & fs2){
		return fs1.cell_ == fs2.cell_ and fs1.sizes_[0] == fs2.sizes_[0] and fs1.sizes_[1] == fs2.sizes_[1] and fs1.sizes_[2] == fs2.sizes_[2];
	}

	auto & reverse_order() const {
		return reverse_;
	}
	
	class point_operator {

		bool reverse_;
		std::array<int, 3> sizes_;
		std::array<inq::parallel::partition, 3> cubic_part_;
		systems::cell cell_;
		
	public:
		
		point_operator(bool reverse, std::array<int, 3> const & nr, std::array<inq::parallel::partition, 3> const & dist, systems::cell cell):
			reverse_(reverse),
			sizes_(nr),
			cubic_part_(dist),
			cell_(cell)
		{
		}

		GPU_FUNCTION auto gvector(parallel::global_index i0, parallel::global_index i1, parallel::global_index i2) const {
				
			//FFTW generates a grid from 0 to 2pi/h, so we convert it to a
			//grid from -pi/h to pi/h
				
			auto ii = grid::to_symmetric_range(sizes_, i0, i1, i2);
			if(reverse_) return vector3<double, covariant>{ii[2]*covspacing(2), ii[1]*covspacing(1), ii[0]*covspacing(0)};
			return vector3<double, covariant>{ii[0]*covspacing(0), ii[1]*covspacing(1), ii[2]*covspacing(2)};
		}

		GPU_FUNCTION auto gvector(int i0, int i1, int i2) const {
			auto i0g = cubic_part_[0].local_to_global(i0);
			auto i1g = cubic_part_[1].local_to_global(i1);
			auto i2g = cubic_part_[2].local_to_global(i2);
				
			return gvector(i0g, i1g, i2g);
		}

		GPU_FUNCTION auto gvector_cartesian(int i0, int i1, int i2) const {
			return cell_.to_cartesian(gvector(i0, i1, i2));
		}
			
		GPU_FUNCTION auto outside_sphere(int i0, int i1, int i2) const {
			auto ivec = grid::to_symmetric_range(sizes_, cubic_part_[0].local_to_global(i0), cubic_part_[1].local_to_global(i1), cubic_part_[2].local_to_global(i2));
			double xx = double(ivec[0])/sizes_[0];
			double yy = double(ivec[1])/sizes_[1];
			double zz = double(ivec[2])/sizes_[2];
			return xx*xx + yy*yy + zz*zz > 0.25;
		}
			
		GPU_FUNCTION bool g_is_zero(parallel::global_index i0, parallel::global_index i1, parallel::global_index i2) const {
			return (i0.value() == 0 and i1.value() == 0 and i2.value() == 0);
		}

		GPU_FUNCTION bool g_is_zero(int i0, int i1, int i2) const {
			auto i0g = cubic_part_[0].local_to_global(i0);
			auto i1g = cubic_part_[1].local_to_global(i1);
			auto i2g = cubic_part_[2].local_to_global(i2);
				
			return g_is_zero(i0g, i1g, i2g);
		}
			
		GPU_FUNCTION double g2(parallel::global_index i0, parallel::global_index i1, parallel::global_index i2) const {
			return cell_.norm(gvector(i0, i1, i2));
		}
			
		GPU_FUNCTION double g2(int i0, int i1, int i2) const {
			return cell_.norm(gvector(i0, i1, i2));
		}

		GPU_FUNCTION auto & cell() const {
			return cell_;
		}

		GPU_FUNCTION auto to_symmetric_range(int i0, int i1, int i2) const {
			return grid::to_symmetric_range(sizes_, i0, i1, i2);
		}
			
		GPU_FUNCTION auto from_symmetric_range(vector3<int> ii) const {
			return grid::from_symmetric_range(sizes_, ii);
		}
						
		GPU_FUNCTION auto & cubic_part(int idim) const {
			return cubic_part_[idim];
		}

		GPU_FUNCTION auto local_contains(vector3<int> const & ii) const {
			bool contains = true;
			for(int idir = 0; idir < 3; idir++){
				contains = contains and cubic_part_[idir].contains(ii[idir]);
			}
			return contains;
		}
			
	};

	auto point_op() const {
		return point_operator{reverse_, sizes_, cubic_part_, cell_};
	}

	template <typename ReciprocalBasis = reciprocal_space>
	auto reciprocal() const {
		if(reverse_) return ReciprocalBasis(cell_, {sizes_[2], sizes_[1], sizes_[0]}, comm_);
		return ReciprocalBasis(cell_, {sizes_[0], sizes_[1], sizes_[2]}, comm_);
	}

};

}
}
#endif

#ifdef INQ_BASIS_FOURIER_SPACE_UNIT_TEST
#undef INQ_BASIS_FOURIER_SPACE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

}
#endif
