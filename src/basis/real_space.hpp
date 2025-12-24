/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__REAL_SPACE
#define INQ__BASIS__REAL_SPACE

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <array>

#include <basis/grid.hpp>
#include <gpu/run.hpp>
#include <math/vector3.hpp>
#include <math/nextprod.hpp>

namespace inq {
namespace basis {

class fourier_space;

class real_space : public grid {

	systems::cell cell_;
	vector3<double> rspacing_;
	vector3<double, contravariant> conspacing_;
	vector3<double> rlength_;

	template <typename Int>
	static auto good_size(Int size) {
		return math::nextprod({2, 3, 5, 7}, size);
	}
	
public:

	using reciprocal_space = fourier_space;
	
	real_space(systems::cell const & cell, std::array<int, 3> sizes, parallel::communicator comm):
		grid(sizes, comm, /*par_dim = */ 1),
		cell_(cell)
	{
		for(int idir = 0; idir < 3; idir++){
			rlength_[idir] = length(cell_[idir]);
			rspacing_[idir] = rlength_[idir]/sizes_[idir];
			conspacing_[idir] = 1.0/sizes_[idir];
		}
	}

	real_space(systems::cell const & cell, double const & spacing, parallel::communicator comm):
		real_space(cell, calculate_dimensions(cell, spacing), comm)
	{
	}
		
	real_space(real_space && old, parallel::communicator & new_comm):
		real_space(old.cell_, old.sizes_, new_comm)
	{
	}

	real_space(real_space && old, parallel::communicator && new_comm):
		real_space(old.cell_, old.sizes_, new_comm)
	{
	}

	auto & cell() const {
		return cell_;
	}
	
	GPU_FUNCTION const vector3<double> & rspacing() const{
		return rspacing_;
	}

		
	GPU_FUNCTION const auto & contravariant_spacing() const{
		return conspacing_;
	}
		
	double diagonal_length() const {
		return length(rlength_);
	}
		
	GPU_FUNCTION const vector3<double> & rlength() const{
		return rlength_;
	}

	double min_rlength() const{
		return std::min(rlength_[0], std::min(rlength_[1], rlength_[2]));
	}

	template<class OStream>
	friend OStream& operator<<(OStream& out, real_space const& self){
		out << "Basis set:" << std::endl;
		out << "  Spacing                 = " << self.rspacing()[0] << " " << self.rspacing()[1] << " " << self.rspacing()[2] << " bohr" << std::endl;
		out << "  Grid size               = " << self.sizes()[0] << " x " << self.sizes()[1] << " x " << self.sizes()[2] << std::endl;
		out << "  Number of grid points   = " << self.sizes()[0]*self.sizes()[1]*self.sizes()[2] << std::endl;
		out << std::endl;
		return out;
	}
	
	class point_operator {

		std::array<int, 3> sizes_;
		vector3<double, contravariant> rspacing_;
		std::array<inq::parallel::partition, 3> cubic_part_;
		systems::cell cell_;

	public:

		point_operator(std::array<int, 3> const & nr, vector3<double, contravariant> const & rspacing, std::array<inq::parallel::partition, 3> const & dist, systems::cell cell):
			sizes_(nr),
			rspacing_(rspacing),
			cubic_part_(dist),
			cell_(cell)
		{
		}

		GPU_FUNCTION auto to_symmetric_range(int ix, int iy, int iz) const {
			return grid::to_symmetric_range(sizes_, ix, iy, iz);
		}
			
		GPU_FUNCTION auto from_symmetric_range(vector3<int> ii) const {
			return grid::from_symmetric_range(sizes_, ii);
		}

		GPU_FUNCTION auto rvector(parallel::global_index ix, parallel::global_index iy, parallel::global_index iz) const {
			auto ii = grid::to_symmetric_range(sizes_, ix, iy, iz);
			return vector3<int, contravariant>{ii[0], ii[1], ii[2]}*rspacing_;
		}
			
		GPU_FUNCTION auto rvector(int ix, int iy, int iz) const {
			auto ixg = cubic_part_[0].local_to_global(ix);
			auto iyg = cubic_part_[1].local_to_global(iy);
			auto izg = cubic_part_[2].local_to_global(iz);
				
			return rvector(ixg, iyg, izg);
		}

		GPU_FUNCTION auto rvector_cartesian(int ix, int iy, int iz) const {
			return cell_.to_cartesian(rvector(ix, iy, iz));
		}

		GPU_FUNCTION auto rvector_cartesian(parallel::global_index ix, parallel::global_index iy, parallel::global_index iz) const {
			return cell_.to_cartesian(rvector(ix, iy, iz));
		}
			
		template <class int_array>
		GPU_FUNCTION auto rvector(const int_array & indices) const {
			return rvector(indices[0], indices[1], indices[2]);
		}
			
		template <typename IndexType>
		GPU_FUNCTION double r2(IndexType ix, IndexType iy, IndexType iz) const {
			return cell_.norm(rvector(ix, iy, iz));
		}

		template <typename IndexType>
		GPU_FUNCTION double rlength(IndexType ix, IndexType iy, IndexType iz) const {
			return cell_.length(rvector(ix, iy, iz));
		}
			
		GPU_FUNCTION auto & cubic_part(int idim) const {
			return cubic_part_[idim];
		}
			
		GPU_FUNCTION auto & cell() const {
			return cell_;
		}
			
		GPU_FUNCTION auto local_contains(vector3<int> const & ii) const {
			bool contains = true;
			for(int idir = 0; idir < 3; idir++){
				contains = contains and cubic_part_[idir].contains(ii[idir]);
			}
			return contains;
		}
			
	};
			
	friend auto operator==(const real_space & rs1, const real_space & rs2){
		return rs1.cell_ == rs2.cell_ and rs1.sizes_[0] == rs2.sizes_[0] and rs1.sizes_[1] == rs2.sizes_[1] and rs1.sizes_[2] == rs2.sizes_[2];
	}

	auto enlarge(int factor) const {

		auto new_sizes = sizes_;
		for(auto & size : new_sizes) {
			size = good_size(round(factor*size));
		}

		return real_space(cell_.enlarge(factor), new_sizes, this->comm());
	}

	auto enlarge(vector3<int> factor) const {

		auto new_sizes = sizes_;
		auto idir = 0;
		for(auto & size : new_sizes) {
			size = good_size(round(factor[idir]*size));
			idir++;
		}

		return real_space(cell_.enlarge(factor), new_sizes,  this->comm());
	}
		
	auto refine(double factor) const {
		assert(factor > 0.0);
		auto new_sizes = sizes_;
		for(auto & size : new_sizes) {
			size = good_size(round(factor*size));
		}
		return real_space(cell_, new_sizes, this->comm());
	}
		
	auto volume_element() const {
		return cell().volume()/size();
	}

	auto gcutoff() const {
		auto max_spacing = std::max({rspacing_[0], rspacing_[1],rspacing_[2]});

		return M_PI/max_spacing;
	}

	auto point_op() const {
		return point_operator(sizes_, conspacing_, cubic_part_, cell_);
	}

	template <typename ReciprocalBasis = reciprocal_space>
	auto reciprocal() const {
		return ReciprocalBasis(*this);
	}

	static auto gcutoff(systems::cell const & cell, double const & spacing){
		auto nr = calculate_dimensions(cell, spacing);

		auto max_spacing = 0.0;
		for(int idir = 0; idir < 3; idir++){
			auto actual_spacing = length(cell[idir])/nr[idir];
			max_spacing = std::max(max_spacing, actual_spacing);
		}

		return M_PI/max_spacing;
	}
		
private:

	static std::array<int, 3> calculate_dimensions(systems::cell const & cell, double const & spacing, bool optimize = true){
		std::array<int, 3> nr;
			
		// make the spacing conmensurate with the grid
		// OPTIMIZATION: we can select a good size here for the FFT
		for(int idir = 0; idir < 3; idir++){
			double rlength = length(cell[idir]);
			nr[idir] = round(rlength/spacing);
			if(optimize) nr[idir] = good_size(nr[idir]);
		}
			
		return nr;
	}

};

}
}
#endif

#ifdef INQ_BASIS_REAL_SPACE_UNIT_TEST
#undef INQ_BASIS_REAL_SPACE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using Catch::Approx;

	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
  {
    
    SECTION("Cubic cell"){

      basis::real_space rs(systems::cell::cubic(10.0_b), /* spacing = */ 0.49672941, comm);

      CHECK(rs.size() == 8000);

			CHECK(rs.cell().volume() == rs.volume_element()*rs.size());
      
      CHECK(rs.rspacing()[0] == 0.5_a);
      CHECK(rs.rspacing()[1] == 0.5_a);
      CHECK(rs.rspacing()[2] == 0.5_a);
      
      CHECK(rs.sizes()[0] == 20);
      CHECK(rs.sizes()[1] == 20);
      CHECK(rs.sizes()[2] == 20);

			CHECK(rs.gcutoff() == 6.2831853072_a);
			CHECK(rs.gcutoff() == basis::real_space::gcutoff(systems::cell::cubic(10.0_b), /* spacing = */ 0.49672941));
			
			basis::real_space new_rs(basis::real_space(rs), parallel::communicator{boost::mpi3::environment::get_self_instance()});
			
			CHECK(rs.sizes() == new_rs.sizes());
			CHECK(new_rs.local_sizes() == new_rs.sizes());	
			CHECK(rs.cell().periodicity() == new_rs.cell().periodicity());

			CHECK(rs.part().rank() == comm.rank());
			CHECK(rs.cubic_part(0).rank() == comm.rank()%rs.cubic_part(0).comm_size());
			CHECK(rs.cubic_part(1).rank() == comm.rank()%rs.cubic_part(1).comm_size());
			CHECK(rs.cubic_part(2).rank() == comm.rank()%rs.cubic_part(2).comm_size());
			
			rs.shift();
			CHECK(rs.part().rank() == (comm.rank() + 1)%comm.size());
			CHECK(rs.cubic_part(0).rank() == (comm.rank() + 1)%rs.cubic_part(0).comm_size());
			CHECK(rs.cubic_part(1).rank() == (comm.rank() + 1)%rs.cubic_part(1).comm_size());
			CHECK(rs.cubic_part(2).rank() == (comm.rank() + 1)%rs.cubic_part(2).comm_size());			
			
    }

    SECTION("Parallelepipedic cell"){

      basis::real_space rs(systems::cell::orthorhombic(77.7_b, 14.14_b, 23.25_b), /*spacing =*/ 0.36063925, comm);

      CHECK(rs.size() == 552960);

			CHECK(rs.cell().volume() == rs.volume_element()*rs.size());
			
      CHECK(rs.rspacing()[0] == 0.3597222222_a);
      CHECK(rs.rspacing()[1] == 0.3535_a);
      CHECK(rs.rspacing()[2] == 0.36328125_a);
      
      CHECK(rs.sizes()[0] == 216);
      CHECK(rs.sizes()[1] == 40);
			CHECK(rs.sizes()[2] == 64);

			CHECK(rs.gcutoff() == 8.6478249389_a);
			CHECK(rs.gcutoff() == basis::real_space::gcutoff(systems::cell::orthorhombic(77.7_b, 14.14_b, 23.25_b), /*spacing =*/ 0.36063925));
			
			auto rs3x = rs.enlarge(3);
			
      CHECK(rs3x.rspacing()[0] == 0.3597222222_a);
      CHECK(rs3x.rspacing()[1] == 0.3535_a);
      CHECK(rs3x.rspacing()[2] == 0.36328125_a);
      
      CHECK(rs3x.sizes()[0] == 3*216);
      CHECK(rs3x.sizes()[1] == 3*40);
			CHECK(rs3x.sizes()[2] == 3*64);

			CHECK(rs3x.cubic_part(0).comm_size() == 1);
			CHECK(rs3x.cubic_part(1).comm_size() == rs.comm().size());
			CHECK(rs3x.cubic_part(2).comm_size() == 1);

			CHECK(rs3x.part().comm_size() == rs.comm().size());
			CHECK(rs3x.part().local_size() == rs3x.cubic_part(1).local_size()*rs3x.cubic_part(0).size()*rs3x.cubic_part(2).size());
			
			auto rs_fine = rs.refine(2);
			
      CHECK(rs_fine.rspacing()[0] == Approx(0.5*0.3597222222));
      CHECK(rs_fine.rspacing()[1] == Approx(0.5*0.3535));
      CHECK(rs_fine.rspacing()[2] == Approx(0.5*0.36328125));
      
      CHECK(rs_fine.sizes()[0] == 2*216);
      CHECK(rs_fine.sizes()[1] == 2*40);
			CHECK(rs_fine.sizes()[2] == 2*64);

			CHECK(rs == rs.refine(1));
			
			auto rs_155 = rs.refine(1.55);
			
      CHECK(rs_155.rspacing()[0] == Approx(0.23125));
      CHECK(rs_155.rspacing()[1] == Approx(0.2244444444));
      CHECK(rs_155.rspacing()[2] == Approx(0.2325));
      
      CHECK(rs_155.sizes()[0] == 336);
      CHECK(rs_155.sizes()[1] == 63);
			CHECK(rs_155.sizes()[2] == 100);

    }

    SECTION("Non-orthogonal cell"){

			auto a = 3.567095_A;
      basis::real_space rs(systems::cell::lattice({0.0_b, a/2.0, a/2.0}, {a/2, 0.0_b, a/2.0}, {a/2.0, a/2.0, 0.0_b}), /*spacing = */ a/30.0_b, comm);

      CHECK(rs.size() == 9261);

			CHECK(rs.cell().volume() == Approx(0.25*pow(a.in_atomic_units(), 3)));
			CHECK(rs.cell().volume() == rs.volume_element()*rs.size());
			
      CHECK(rs.rspacing()[0] == 0.2269756405_a);
      CHECK(rs.rspacing()[1] == 0.2269756405_a);
      CHECK(rs.rspacing()[2] == 0.2269756405_a);
      
      CHECK(rs.sizes()[0] == 21);
      CHECK(rs.sizes()[1] == 21);
			CHECK(rs.sizes()[2] == 21);

			CHECK(rs.gcutoff() == 13.8411005126_a);
			CHECK(rs.gcutoff() == basis::real_space::gcutoff(systems::cell::lattice({0.0_b, a/2.0, a/2.0}, {a/2, 0.0_b, a/2.0}, {a/2.0, a/2.0, 0.0_b}), /*spacing = */ a/30.0_b));

    }

  }
}
#endif
