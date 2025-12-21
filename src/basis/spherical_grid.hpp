/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__BASIS__SPHERICAL_GRID
#define INQ__BASIS__SPHERICAL_GRID

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h>

#ifdef ENABLE_GPU
#include <thrust/remove.h>  // for thrust::remove_if
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#endif 
#include <algorithm> //max, min, sort

#include <math/vector3.hpp>
#include <gpu/atomic.hpp>
#include <gpu/run.hpp>
#include <gpu/reduce.hpp>
#include <ionic/periodic_replicas.hpp>
#include <basis/containing_cube.hpp>
#include <basis/real_space.hpp>
#include <cassert>
#include <array>
#include <gpu/array.hpp>

#include <utils/profiling.hpp>

namespace inq {
namespace basis {

  class spherical_grid {

	public: //because of CUDA
		struct point_data {
			vector3<int> coords_;
			float distance_; //I don't think we need additional precision for this, and we get aligned memory
			vector3<float, contravariant> relative_pos_;

			GPU_FUNCTION friend auto operator<(point_data const & aa, point_data const & bb){
				if(aa.coords_[0] < bb.coords_[0]) return true;
				if(aa.coords_[0] > bb.coords_[0]) return false;
				if(aa.coords_[1] < bb.coords_[1]) return true;
				if(aa.coords_[1] > bb.coords_[1]) return false;
				return (aa.coords_[2] < bb.coords_[2]);
			};
			
		};

  public:

		//we need to make an additional public function to make cuda happy
		template <class basis>
		void initialize(const basis & parent_grid, const vector3<double> & center_point, const double radius){
			CALI_CXX_MARK_SCOPE("spherical_grid::initialize");
			
			ionic::periodic_replicas const rep(parent_grid.cell(), center_point, parent_grid.diagonal_length());

			vector3<int> local_sizes = parent_grid.local_sizes();
			
			long upper_count = 0;

			//FIRST PASS: we count the cubes, this gives us an upper bound for the memory allocation
			for(unsigned irep = 0; irep < rep.size(); irep++){
				vector3<int> lo, hi;
				containing_cube(parent_grid, rep[irep], radius, lo, hi);
				upper_count += (hi[0] - lo[0])*(hi[1] - lo[1])*(hi[2] - lo[2]);
			}

			points_.reextent({upper_count});
			
			upper_count = 0;
			for(unsigned irep = 0; irep < rep.size(); irep++){

				vector3<int> lo, hi;
				containing_cube(parent_grid, rep[irep], radius, lo, hi);

				auto cube_size = (hi[0] - lo[0])*(hi[1] - lo[1])*(hi[2] - lo[2]);     

				if(cube_size == 0) continue;

				auto cube_dims = hi - lo;
				
				//OPTIMIZATION: this iteration should be done only over the local points
				auto buffer = points_({upper_count, upper_count + cube_size}).partitioned(cube_dims[0]*cube_dims[1]).partitioned(cube_dims[0]);

				assert(get<0>(sizes(buffer)) == cube_dims[0]);
				assert(get<1>(sizes(buffer)) == cube_dims[1]);
				assert(get<2>(sizes(buffer)) == cube_dims[2]);

				gpu::run(cube_dims[2], cube_dims[1], cube_dims[0],
								 [lo, local_sizes, point_op = parent_grid.point_op(), re = rep[irep], buf = begin(buffer), radius] GPU_LAMBDA (auto iz, auto iy, auto ix){
									 
									 (&buf[ix][iy][iz])->coords_ = local_sizes;
									 (&buf[ix][iy][iz])->distance_ = -1.0;
									 
									 auto ii = point_op.from_symmetric_range({int(lo[0] + ix), int(lo[1] + iy), int(lo[2] + iz)});
									 
									 parallel::global_index ii0(ii[0]);
									 parallel::global_index ii1(ii[1]);
									 parallel::global_index ii2(ii[2]);
									 
									 int ixl = point_op.cubic_part(0).global_to_local(ii0);
									 int iyl = point_op.cubic_part(1).global_to_local(ii1);
									 int izl = point_op.cubic_part(2).global_to_local(ii2);
									 
									 if(ixl < 0 or ixl >= local_sizes[0]) return;
									 if(iyl < 0 or iyl >= local_sizes[1]) return;
									 if(izl < 0 or izl >= local_sizes[2]) return;
									 
									 auto rpoint = point_op.rvector_cartesian(ii0, ii1, ii2);
									 
									 auto n2 = norm(rpoint - re);
									 if(n2 > radius*radius) return;
									 
									 (&buf[ix][iy][iz])->coords_ = {ixl, iyl, izl};
									 (&buf[ix][iy][iz])->distance_ = sqrt(n2);
									 (&buf[ix][iy][iz])->relative_pos_ = static_cast<vector3<float, contravariant>>(point_op.cell().to_contravariant(rpoint - re));
								 });
				
				upper_count += cube_size;
			}

			assert(upper_count == points_.size());

			{
				CALI_CXX_MARK_SCOPE("spherical_grid::compact");       
#ifdef ENABLE_GPU
				using thrust::remove_if;
				auto it = remove_if(thrust::device, begin(points_), end(points_), [] GPU_LAMBDA (auto value){ return value.distance_ < 0.0;});
				size_ = it - begin(points_);        
#else
				using std::remove_if;
				auto it = remove_if(begin(points_), end(points_), [] GPU_LAMBDA (auto value){ return value.distance_ < 0.0;});
				size_ = it - begin(points_);        
#endif
			}

			if(size_ == 0) {
				points_.clear();
				assert(points_.size() == 0);        
				return;
			}
			
			assert(size_ == 0 or point_data(points_[size_ - 1]).distance_ >= 0.0);

			{
				CALI_CXX_MARK_SCOPE("spherical_grid::reextent");
				auto points2 = +points_({0, size_});
				points_ = std::move(points2);
				assert(points_.size() == size_);
			}
		}
		
		const static int dimension = 1;
		
		template <class basis>
		spherical_grid(const basis & parent_grid, const vector3<double> & center_point, const double radius):
			volume_element_(parent_grid.volume_element()),
			center_(center_point){

			initialize(parent_grid, center_point, radius);
    }
		
    long size() const {
      return size_;
    }
    
		template <class array_4d>
		gpu::array<typename array_4d::element, 2> gather(const array_4d & grid) const {

			CALI_CXX_MARK_SCOPE("spherical_grid::gather(4d)");

			const int nst = get<3>(sizes(grid));

			CALI_MARK_BEGIN("spherical_grid::gather(4d)::allocation");
			gpu::array<typename array_4d::element, 2> subgrid({this->size(), nst});
			CALI_MARK_END("spherical_grid::gather(4d)::allocation");

			gpu::run(nst, size(),
							 [sgr = begin(subgrid), gr = begin(grid), poi = begin(points_)] GPU_LAMBDA (auto ist, auto ipoint){
								 sgr[ipoint][ist] = gr[poi[ipoint].coords_[0]][poi[ipoint].coords_[1]][poi[ipoint].coords_[2]][ist];
							 });
			
			return subgrid;
    }

    template <class array_2d, class array_4d>
    void scatter_add(const array_2d & subgrid, array_4d && grid) const{
			CALI_CXX_MARK_SCOPE("spherical_grid::scatter_add");

			gpu::run(get<1>(sizes(subgrid)), size(),
							 [sgr = begin(subgrid), gr = begin(grid), poi = begin(points_)] GPU_LAMBDA (auto ist, auto ipoint){
								 gr[poi[ipoint].coords_[0]][poi[ipoint].coords_[1]][poi[ipoint].coords_[2]][ist] += sgr[ipoint][ist];
							 });
    }

		const double & volume_element() const {
			return volume_element_;
		}

		friend auto sizes(const spherical_grid & sphere){
			return std::array<long, dimension>{sphere.size()};
		}

		auto & center() const {
			return center_;
		}

		template <typename PointsType>
		struct sphere_ref {

			PointsType points_;

			GPU_FUNCTION auto & grid_point(int ii) const {
				return (&points_[ii])->coords_;
			}

			GPU_FUNCTION auto & distance(int ii) const {
				return (&points_[ii])->distance_;
			}

			GPU_FUNCTION auto & point_pos(int ii) const {
				return (&points_[ii])->relative_pos_;
			}
			
		};

		auto ref() const {
			return sphere_ref<decltype(cbegin(points_))>{cbegin(points_)};
		}   
		
  private:

		gpu::array<point_data, 1> points_;
		double volume_element_;
		vector3<double> center_;
		int size_;
		
  };

}
}
#endif

#ifdef INQ_BASIS_SPHERICAL_GRID_UNIT_TEST
#undef INQ_BASIS_SPHERICAL_GRID_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <gpu/array.hpp>
#include <math/complex.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	auto ll = 10.0;
	basis::real_space pw(systems::cell::cubic(ll*1.0_b), /*spacing = */ 0.49672941, comm);
  auto radius = 2.0;
	auto theo_vol = 4.0/3.0*M_PI*pow(radius, 3);
	
  SECTION("Point 0 0 0"){
    
    basis::spherical_grid sphere(pw, {0.0, 0.0, 0.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);

    gpu::array<complex, 4> grid({pw.local_sizes()[0], pw.local_sizes()[1], pw.local_sizes()[2], 1});

    for(long ii = 0; ii < grid.num_elements(); ii++) grid.data_elements()[ii] = 0.0;
    
    auto subgrid = sphere.gather(grid);

    for(unsigned ii = 0; ii < subgrid.size(); ii++) subgrid[ii][0] = 1.0; 
    
    sphere.scatter_add(subgrid, grid);

    double sum = 0.0;
		for(long ii = 0; ii < grid.num_elements(); ii++) sum += real(grid.data_elements()[ii]);
		comm.all_reduce_in_place_n(&sum, 1, std::plus<>{});
		
    CHECK(sum == 257.0_a);
    
  }

  SECTION("Point -l/2 0 0"){
    
    basis::spherical_grid sphere(pw, {-ll/2.0, 0.0, 0.0}, radius);
		
		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);
    
    gpu::array<complex, 4> grid({pw.local_sizes()[0], pw.local_sizes()[1], pw.local_sizes()[2], 20}, 0.0);

    for(long ii = 0; ii < grid.num_elements(); ii++) grid.data_elements()[ii] = 1.0;
    
    auto subgrid = sphere.gather(grid);

    double sum = 0.0;
		for(long ii = 0; ii < subgrid.num_elements(); ii++) sum += real(subgrid.data_elements()[ii]);
		comm.all_reduce_in_place_n(&sum, 1, std::plus<>{});

    CHECK(sum == Approx(20.0*257.0));
    
    for(long ii = 0; ii < subgrid.num_elements(); ii++) subgrid.data_elements()[ii] = -1.0;
    
		sphere.scatter_add(subgrid, grid);

    sum = 0.0;
		for(long ii = 0; ii < grid.num_elements(); ii++) sum += real(grid.data_elements()[ii]);
		comm.all_reduce_in_place_n(&sum, 1, std::plus<>{});
		
		CHECK(sum == Approx(20.0*(pw.size() - size)));
		
  }

  SECTION("Point l/2 0 0"){
    
    basis::spherical_grid sphere(pw, {ll/2.0, 0.0, 0.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);

  }

  SECTION("Point -l/2 -l/2 -l/2"){
    
    basis::spherical_grid sphere(pw, {-ll/2.0, -ll/2.0, -ll/2.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);
    
  }

  SECTION("Point l/2 l/2 l/2"){
    
    basis::spherical_grid sphere(pw, {ll/2.0, ll/2.0, ll/2.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);
    
  }

  SECTION("Point l/2 l/2 l/2"){
    
    basis::spherical_grid sphere(pw, {ll/2.0, ll/2.0, ll/2.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);
		CHECK(size*pw.volume_element()/theo_vol == 0.9586598525_a);
  }

  SECTION("Point l/4 l/4 l/4"){
    
    basis::spherical_grid sphere(pw, {ll/4.0, ll/4.0, ll/4.0}, radius);

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});
    CHECK(size == 257);
		CHECK(size*pw.volume_element()/theo_vol == 0.9586598525_a);
		
  }
    
  SECTION("Non-cubic grid"){

		auto radius = 5.71;
		
		basis::real_space rs(systems::cell::lattice({7.57325_b, 0.0_b, 0.0_b}, {-3.78662_b, 6.55862_b, 0.0_b}, {5.78917e-16_b, 1.00271e-15_b, 9.45443_b}), /*spacing = */ 0.405578, comm);
    basis::spherical_grid sphere(rs, {3.78659, -2.18619, 1.65434}, radius);   

		auto size = sphere.size();
		comm.all_reduce_in_place_n(&size, 1, std::plus<>{});

		CHECK(size == 15963);
		
		auto theo_vol = 4.0/3.0*M_PI*pow(radius, 3);
		CHECK(size*rs.volume_element()/theo_vol == 1.0013282291_a);

	}
}
#endif

