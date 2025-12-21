/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SYSTEMS__CELL
#define INQ__SYSTEMS__CELL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <magnitude/length.hpp>
#include <utils/load_save.hpp>

#include <stdexcept>
#include <filesystem>
#include <fstream>

#include <math/vector3.hpp>
#include <valarray>
#include <array>

namespace inq {
namespace systems {

  class cell{
		using vector_type = vector3<double>;

  private:
		vector3<vector_type> lattice_;
		vector3<vector_type> reciprocal_;
    double volume_;
		int periodicity_;
		
  public:

    cell(vector3<double> const& a0, vector3<double> const& a1, vector3<double> const& a2, int arg_periodicity = 3){
			
			periodicity_ = arg_periodicity;

			assert(periodicity_ >= 0);
			assert(periodicity_ <= 3);			
			
			lattice_[0] = a0;
			lattice_[1] = a1;
			lattice_[2] = a2;
			volume_ = a0.dot(cross(a1, a2));

			if(volume_ < 1e-10) throw std::runtime_error("inq error: the lattice volume " + std::to_string(volume_) + " is not positive");
		
			reciprocal_[0] = 2.0*M_PI/volume_*cross(a1, a2);
			reciprocal_[1] = 2.0*M_PI/volume_*cross(a2, a0);
			reciprocal_[2] = 2.0*M_PI/volume_*cross(a0, a1);
		}

		static auto cubic(quantity<magnitude::length> lat_par){
			auto aa = lat_par.in_atomic_units();
			return cell{vector3<double>(aa, 0.0, 0.0), vector3<double>(0.0, aa, 0.0), vector3<double>(0.0, 0.0, aa)};
		}

		static auto orthorhombic(quantity<magnitude::length> aa, quantity<magnitude::length> bb, quantity<magnitude::length> cc){
			return cell{vector3<double>(aa.in_atomic_units(), 0.0, 0.0), vector3<double>(0.0, bb.in_atomic_units(), 0.0), vector3<double>(0.0, 0.0, cc.in_atomic_units())};
		}

		static auto lattice(vector3<quantity<magnitude::length>> aa, vector3<quantity<magnitude::length>> bb, vector3<quantity<magnitude::length>> cc){
			return cell{vector3<double>(aa[0].in_atomic_units(), aa[1].in_atomic_units(), aa[2].in_atomic_units()), 
											 vector3<double>(bb[0].in_atomic_units(), bb[1].in_atomic_units(), bb[2].in_atomic_units()), 
											 vector3<double>(cc[0].in_atomic_units(), cc[1].in_atomic_units(), cc[2].in_atomic_units())};
		}
		
		vector_type const& operator[](int ii) const {
			assert(ii >= 0 and ii < 3);         
			return lattice_[ii];
		}
    
    vector_type const& lattice(int ii) const {
			assert(ii >= 0 and ii < 3);
			return lattice_[ii];
		}
		
		vector_type const& reciprocal(int ii) const {
			assert(ii >= 0 and ii < 3);
			return reciprocal_[ii];
		}
		
		auto enlarge(int factor) const {
			return cell(factor*lattice_[0], factor*lattice_[1], factor*lattice_[2], periodicity_);
		}

		auto enlarge(vector3<int> factor) const {
			return cell(factor[0]*lattice_[0], factor[1]*lattice_[1], factor[2]*lattice_[2], periodicity_);
		}
			
    double volume() const { return volume_; }

		template<class OStream>
		friend OStream & operator<<(OStream & out, cell const& self){
			out << "Cell:" << '\n';
			out << "  Lattice vectors [b] = " << self.lattice_[0] << '\n';
			out << "                        " << self.lattice_[1] << '\n';
			out << "                        " << self.lattice_[2] << '\n';
			out << "  Volume [b^3]        = " << self.volume_ << '\n';
			out << "  Periodicity         = " << self.periodicity_;
			if(self.periodicity_ == 0) out << "d (finite)\n";
			if(self.periodicity_ == 1) out << "d (wire)\n";
			if(self.periodicity_ == 2) out << "d (slab)\n";
			if(self.periodicity_ == 3) out << "d (fully periodic)\n";
			out << std::endl;
			return out;
	  }
		
		////////////////////////////////////////////////////////////////////////////////

    bool operator==(const cell& c) const {
			return ( lattice_[0]==c.lattice_[0] && lattice_[1]==c.lattice_[1] && lattice_[2]==c.lattice_[2] );
		}
			
    bool operator!=(const cell& c) const {
			return ! ( *this == c );
		}

		auto diagonal_length() const {
			return sqrt(lattice_[0].norm() + lattice_[1].norm() + lattice_[2].norm());
		}

		auto periodicity() const {
			return periodicity_;
		}

		auto & periodicity(int const pval) {
			if(pval > 3 or pval < 0) throw std::runtime_error("inq error: the requested periodicity (" + std::to_string(pval) + ") does not make sense.");
			if(pval == 1) throw std::runtime_error("inq error: periodicity 1 is not implemented yet.");
			periodicity_ = pval;
			return *this;
		}
				
		auto & finite() {
			periodicity_ = 0;
			return *this;
		}

		auto & periodic() {
			periodicity_ = 3;
			return *this;
		}
		
		template <class Type1, class Space1, class Type2, class Space2>
		GPU_FUNCTION auto dot(vector3<Type1, Space1> const & vv1, vector3<Type2, Space2> const & vv2) const {
			return to_cartesian(vv1).dot(to_cartesian(vv2));
		}
		
		template <class Type, class Space>
		GPU_FUNCTION auto norm(vector3<Type, Space> const & vv) const {
			return real(dot(vv, vv));
		}
		
		template <class Type, class Space>
		GPU_FUNCTION auto length(vector3<Type, Space> const & vv) const {
			return sqrt(norm(vv));
		}
		
		template <class Type>
		GPU_FUNCTION auto to_covariant(vector3<Type, covariant> const & vv) const {
				return vv;
		}
		
		template <class Type>
		GPU_FUNCTION auto to_covariant(vector3<Type, cartesian> const & vv) const {
			return vector3<Type, covariant>{lattice_[0].dot(vv), lattice_[1].dot(vv), lattice_[2].dot(vv)};
		}
		
		template <class Type>
		GPU_FUNCTION auto to_covariant(vector3<Type, contravariant> const & vv) const {
			return to_covariant(to_cartesian(vv));
		}
		
		template <class Type>
		GPU_FUNCTION auto to_contravariant(vector3<Type, contravariant> const & vv) const {
				return vv;
		}
		
		template <class Type>
		GPU_FUNCTION auto to_contravariant(vector3<Type, covariant> const & vv) const {
			return to_contravariant(to_cartesian(vv));
		}
		
		template <class Type>
			GPU_FUNCTION auto to_contravariant(vector3<Type, cartesian> const & vv) const {
			return vector3<Type, contravariant>{reciprocal_[0].dot(vv), reciprocal_[1].dot(vv), reciprocal_[2].dot(vv)}/(2.0*M_PI);
		}
		
		template <class Type>
		GPU_FUNCTION auto to_cartesian(vector3<Type, cartesian> const & vv) const {
			return vv;
		}
			
		template <class Type>
		GPU_FUNCTION auto to_cartesian(vector3<Type, contravariant> const & vv) const {
			return lattice_[0]*vv[0] + lattice_[1]*vv[1] + lattice_[2]*vv[2];
		}
		
		template <class Type>
		GPU_FUNCTION auto to_cartesian(vector3<Type, covariant> const & vv) const {
			return (reciprocal_[0]*vv[0] + reciprocal_[1]*vv[1] + reciprocal_[2]*vv[2])/(2.0*M_PI);
		}

    bool contains(vector3<double, contravariant> point) const {
			return point[0] >= -0.5 && point[0] < 0.5 && point[1] >= -0.5 && point[1] < 0.5 && point[2] >= -0.5 && point[2] <= 0.5;
		}

		bool contains(vector3<double> point) const {
			return contains(to_contravariant(point));
		}
		
		auto position_in_cell(vector3<double, contravariant> crystal_pos) const {
			for(int idir = 0; idir < 3; idir++) {
				crystal_pos[idir] -= floor(crystal_pos[idir]);
				if(crystal_pos[idir] >= 0.5) crystal_pos[idir] -= 1.0;
			}
			assert(contains(crystal_pos));
			return crystal_pos;
		}
		
		auto position_in_cell(vector3<double> const & pos) const {
			auto crystal_pos = to_contravariant(pos);
			return to_cartesian(position_in_cell(crystal_pos));
		}

		auto is_orthogonal() const {
			return fabs(dot(lattice_[0], lattice_[1])) < 1e-8 and fabs(dot(lattice_[1], lattice_[2])) < 1e-8 and fabs(dot(lattice_[2], lattice_[0])) < 1e-8;
		}

		auto is_cartesian() const {
			return is_orthogonal() and lattice_[0][1] < 1e-8 and lattice_[0][2] < 1e-8;
		}

		void save(parallel::communicator & comm, std::string const & dirname) const {
			auto error_message = "INQ error: Cannot save the cell to directory '" + dirname + "'.";

			utils::create_directory(comm, dirname);
			utils::save_container(comm, dirname + "/lattice",     lattice_,     error_message);
			utils::save_value(comm, dirname + "/periodicity", periodicity_, error_message);
		}

		static auto load(std::string const & dirname) {
			auto error_message = "INQ error: Cannot load the cell from directory '" + dirname + "'.";
			
			std::array<vector3<double>, 3> lat;
			int per;
			utils::load_array(dirname + "/lattice", lat, error_message);
			utils::load_value(dirname + "/periodicity", per, error_message);
			
			return cell(lat[0], lat[1], lat[2], per);
		}
		
  };

}
}
#endif

#ifdef INQ_SYSTEMS_CELL_UNIT_TEST
#undef INQ_SYSTEMS_CELL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
  {
		SECTION("Cubic"){
				
			auto cell = systems::cell::cubic(10.2_b).periodic();
			
			CHECK(cell[0][0] == 10.2_a);
			CHECK(cell[0][1] == 0.0_a);
			CHECK(cell[0][2] == 0.0_a);
			CHECK(cell[1][0] == 0.0_a);
			CHECK(cell[1][1] == 10.2_a);
			CHECK(cell[1][2] == 0.0_a);
			CHECK(cell[2][0] == 0.0_a);
			CHECK(cell[2][1] == 0.0_a);
			CHECK(cell[2][2] == 10.2_a);
			
			CHECK(cell.periodicity() == 3);
			
		}

		SECTION("Cubic finite"){
				
			auto cell = systems::cell::cubic(10.2_b).finite();
			
			CHECK(cell[0][0] == 10.2_a);
			CHECK(cell[0][1] == 0.0_a);
			CHECK(cell[0][2] == 0.0_a);
			CHECK(cell[1][0] == 0.0_a);
			CHECK(cell[1][1] == 10.2_a);
			CHECK(cell[1][2] == 0.0_a);
			CHECK(cell[2][0] == 0.0_a);
			CHECK(cell[2][1] == 0.0_a);
			CHECK(cell[2][2] == 10.2_a);
			
			CHECK(cell.periodicity() == 0);
			
		}
		
		SECTION("Parallelepipedic"){
			
			auto cell = systems::cell::orthorhombic(10.2_b, 5.7_b, 8.3_b).periodicity(2);
			
			CHECK(cell[0][0] == 10.2_a);
			CHECK(cell[0][1] == 0.0_a);
			CHECK(cell[0][2] == 0.0_a);
			CHECK(cell[1][0] == 0.0_a);
			CHECK(cell[1][1] == 5.7_a);
			CHECK(cell[1][2] == 0.0_a);
			CHECK(cell[2][0] == 0.0_a);
			CHECK(cell[2][1] == 0.0_a);
			CHECK(cell[2][2] == 8.3_a);
			CHECK(cell.periodicity() == 2);
		
		}

		SECTION("Non-orthogonal"){
			
			auto cell = systems::cell::lattice({0.0_A, 1.0_A, 1.0_A}, {1.0_A, 0.0_b, 1.0_A}, {1.0_A, 1.0_A, 0.0_A});
			
			CHECK(cell[0][0] == 0.0_a);
			CHECK(cell[0][1] == 1.8897261246_a);
			CHECK(cell[0][2] == 1.8897261246_a);
			CHECK(cell[1][0] == 1.8897261246_a);
			CHECK(cell[1][1] == 0.0_a);
			CHECK(cell[1][2] == 1.8897261246_a);
			CHECK(cell[2][0] == 1.8897261246_a);
			CHECK(cell[2][1] == 1.8897261246_a);
			CHECK(cell[2][2] == 0.0_a);
			CHECK(cell.periodicity() == 3);
        
			cell.save(comm, "cell_non_orthogonal_save");
			auto read_cell = systems::cell::load("cell_non_orthogonal_save");
			
			CHECK(read_cell[0][0] == 0.0_a);
			CHECK(read_cell[0][1] == 1.8897261246_a);
			CHECK(read_cell[0][2] == 1.8897261246_a);
			CHECK(read_cell[1][0] == 1.8897261246_a);
			CHECK(read_cell[1][1] == 0.0_a);
			CHECK(read_cell[1][2] == 1.8897261246_a);
			CHECK(read_cell[2][0] == 1.8897261246_a);
			CHECK(read_cell[2][1] == 1.8897261246_a);
			CHECK(read_cell[2][2] == 0.0_a);
			CHECK(read_cell.periodicity() == 3);
			
		}
		
		SECTION("Cubic cell"){
			
      systems::cell cell(vector3<double>(10.0, 0.0, 0.0), vector3<double>(0.0, 10.0, 0.0), vector3<double>(0.0, 0.0, 10.0));

      CHECK(cell[0][0] == 10.0_a);
      CHECK(cell[0][1] ==  0.0_a);
      CHECK(cell[0][2] ==  0.0_a);
      CHECK(cell[1][0] ==  0.0_a);
      CHECK(cell[1][1] == 10.0_a);
      CHECK(cell[1][2] ==  0.0_a);
      CHECK(cell[2][0] ==  0.0_a);
      CHECK(cell[2][1] ==  0.0_a);
      CHECK(cell[2][2] == 10.0_a);
      
      CHECK(cell.lattice(0)[0] == 10.0_a);
      CHECK(cell.lattice(0)[1] ==  0.0_a);
      CHECK(cell.lattice(0)[2] ==  0.0_a);
      CHECK(cell.lattice(1)[0] ==  0.0_a);
      CHECK(cell.lattice(1)[1] == 10.0_a);
      CHECK(cell.lattice(1)[2] ==  0.0_a);
      CHECK(cell.lattice(2)[0] ==  0.0_a);
      CHECK(cell.lattice(2)[1] ==  0.0_a);
      CHECK(cell.lattice(2)[2] == 10.0_a);
    
      CHECK(cell.reciprocal(0)[0] == 0.6283185307_a);
      CHECK(cell.reciprocal(0)[1] ==  0.0_a);
      CHECK(cell.reciprocal(0)[2] ==  0.0_a);
      CHECK(cell.reciprocal(1)[0] ==  0.0_a);
      CHECK(cell.reciprocal(1)[1] == 0.6283185307_a);
      CHECK(cell.reciprocal(1)[2] ==  0.0_a);
      CHECK(cell.reciprocal(2)[0] ==  0.0_a);
      CHECK(cell.reciprocal(2)[1] ==  0.0_a);
      CHECK(cell.reciprocal(2)[2] == 0.6283185307_a);
    
      CHECK(cell.volume() == 1000.0_a);

      CHECK(!cell.contains(vector3<double>(5.0, 5.0, 5.0)));
      CHECK(!cell.contains(vector3<double>(-5.0, 5.0, 5.0)));
      CHECK(!cell.contains(vector3<double>(5.0, -5.0, 5.0)));
      CHECK(!cell.contains(vector3<double>(5.0, 5.0, -5.0)));

      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[0] == 2.0_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[1] == -5.0_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[2] == 8.67_a);

      CHECK(cell.to_contravariant(vector3<double>(6.66, -3.77, 27.2))[0] == 0.666_a);
      CHECK(cell.to_contravariant(vector3<double>(6.66, -3.77, 27.2))[1] == -0.377_a);
      CHECK(cell.to_contravariant(vector3<double>(6.66, -3.77, 27.2))[2] == 2.72_a);

			auto in_cell = cell.position_in_cell(vector3<double>(6.66, 25.0, -18.33));

			CHECK(in_cell[0] == -3.34_a);
			CHECK(in_cell[1] == -5.00_a);
			CHECK(in_cell[2] == 1.67_a);

			{
				auto vv = cell.to_contravariant(vector3<double, cartesian>{10.0, 10.0, 10.0});
				CHECK(vv[0] == 1.0_a);      
				CHECK(vv[1] == 1.0_a);
				CHECK(vv[2] == 1.0_a);
			}

			{
				auto vv = cell.to_covariant(vector3<double, cartesian>{M_PI/10.0, M_PI/10.0, M_PI/10.0});
				CHECK(vv[0] == Approx(M_PI));       
				CHECK(vv[1] == Approx(M_PI));
				CHECK(vv[2] == Approx(M_PI));
			}

			{
				auto vv = vector3<double, contravariant>{1.0, 0.0, 0.0};
				CHECK(cell.length(vv) == 10.0);
				CHECK(dot(cell.to_covariant(vv), vv) == 100.0_a);
				CHECK(cell.dot(cell.to_covariant(vv), vv) == 100.0_a);

			}

			CHECK(cell.is_orthogonal());
			CHECK(cell.is_cartesian());         
    }

    SECTION("Parallelepipedic cell"){

			systems::cell cell(vector3<double>(28.62, 0.0, 0.0), vector3<double>(0.0, 90.14, 0.0), vector3<double>(0.0, 0.0, 12.31));
			
      CHECK(cell.lattice(0)[0] == 28.62_a);
      CHECK(cell.lattice(0)[1] ==  0.0_a);
      CHECK(cell.lattice(0)[2] ==  0.0_a);
      CHECK(cell.lattice(1)[0] ==  0.0_a);
      CHECK(cell.lattice(1)[1] == 90.14_a);
      CHECK(cell.lattice(1)[2] ==  0.0_a);
      CHECK(cell.lattice(2)[0] ==  0.0_a);
      CHECK(cell.lattice(2)[1] ==  0.0_a);
      CHECK(cell.lattice(2)[2] == 12.31_a);
    
      CHECK(cell.reciprocal(0)[0] == 0.2195382707_a);
      CHECK(cell.reciprocal(0)[1] ==  0.0_a);
      CHECK(cell.reciprocal(0)[2] ==  0.0_a);
      CHECK(cell.reciprocal(1)[0] ==  0.0_a);
      CHECK(cell.reciprocal(1)[1] == 0.0697047405_a);
      CHECK(cell.reciprocal(1)[2] ==  0.0_a);
      CHECK(cell.reciprocal(2)[0] ==  0.0_a);
      CHECK(cell.reciprocal(2)[1] ==  0.0_a);
      CHECK(cell.reciprocal(2)[2] == 0.5104131038_a);
    
      CHECK(cell.volume() == 31757.421708_a);

      CHECK(cell.contains(vector3<double>(5.0, 5.0, 5.0)));
      CHECK(cell.contains(vector3<double>(-5.0, 5.0, 5.0)));
      CHECK(cell.contains(vector3<double>(5.0, -5.0, 5.0)));
      CHECK(cell.contains(vector3<double>(5.0, 5.0, -5.0)));

      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[0] == 5.724_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[1] == -45.07_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[2] == 10.67277_a);

      CHECK(cell.to_contravariant(vector3<double>(6.66, -203.77, 927.2))[0] == 0.2327044025_a);
      CHECK(cell.to_contravariant(vector3<double>(6.66, -203.77, 927.2))[1] == -2.2605946306_a);
      CHECK(cell.to_contravariant(vector3<double>(6.66, -203.77, 927.2))[2] == 75.3208773355_a);

			auto in_cell = cell.position_in_cell(vector3<double>(6.66, 225.0, -18.33));

			CHECK(in_cell[0] == 6.66_a);
			CHECK(in_cell[1] == 44.72_a);
			CHECK(in_cell[2] == -6.02_a);
			
			{
				auto vv = cell.to_contravariant(vector3<double, cartesian>{28.62, 90.14, 12.31});
				CHECK(vv[0] == 1.0_a);      
				CHECK(vv[1] == 1.0_a);
				CHECK(vv[2] == 1.0_a);

				auto vv2 = cell.to_cartesian(vv);
				CHECK(vv2[0] == 28.62_a);       
				CHECK(vv2[1] == 90.14_a);
				CHECK(vv2[2] == 12.31_a);

				CHECK(norm(vv2) == Approx(dot(vv, cell.to_covariant(vv))));
			}

			{
				auto vv1 = cell.to_covariant(vector3<double, cartesian>{M_PI/28.62, 0.0, 0.0});
				CHECK(vv1[0] == Approx(M_PI));      
				CHECK(vv1[1] == Approx(0.0));
				CHECK(vv1[2] == Approx(0.0));

				auto vv2 = cell.to_covariant(vector3<double, cartesian>{0.0, M_PI/90.14, 0.0});
				CHECK(vv2[0] == Approx(0.0));       
				CHECK(vv2[1] == Approx(M_PI));
				CHECK(vv2[2] == Approx(0.0));

				auto vv3 = cell.to_covariant(vector3<double, cartesian>{0.0, 0.0, M_PI/12.31});
				CHECK(vv3[0] == Approx(0.0));       
				CHECK(vv3[1] == Approx(0.0));
				CHECK(vv3[2] == Approx(M_PI));

				CHECK(cell.dot(vv1, vector3<double, cartesian>{28.62, 0.0, 0.0}) == Approx(M_PI));
				
			}
						
			CHECK(cell.length(vector3<double, contravariant>{1.0, 0.0, 0.0}) == 28.62);

			CHECK(cell.is_orthogonal());
			CHECK(cell.is_cartesian());         
		}

    SECTION("Rotated cell"){

			double ll = 10;
      systems::cell cell({ll/sqrt(2.0), ll/sqrt(2.0), 0.0}, {-ll/sqrt(2.0), ll/sqrt(2.0), 0.0}, {0.0, 0.0, ll});
      
      CHECK(cell.lattice(0)[0] == 7.0710678119_a);
      CHECK(cell.lattice(0)[1] == 7.0710678119_a);
      CHECK(cell.lattice(0)[2] == 0.0_a);
      CHECK(cell.lattice(1)[0] == -7.0710678119_a);
      CHECK(cell.lattice(1)[1] == 7.0710678119_a);
      CHECK(cell.lattice(1)[2] == 0.0_a);
      CHECK(cell.lattice(2)[0] == 0.0_a);
      CHECK(cell.lattice(2)[1] == 0.0_a);
      CHECK(cell.lattice(2)[2] == 10.0_a);
			
      CHECK(cell.reciprocal(0)[0] == 0.4442882938_a);
      CHECK(cell.reciprocal(0)[1] == 0.4442882938_a);
      CHECK(cell.reciprocal(0)[2] == 0.0_a);
      CHECK(cell.reciprocal(1)[0] == -0.4442882938_a);
      CHECK(cell.reciprocal(1)[1] == 0.4442882938_a);
      CHECK(cell.reciprocal(1)[2] == 0.0_a);
      CHECK(cell.reciprocal(2)[0] == 0.0_a);
      CHECK(cell.reciprocal(2)[1] == 0.0_a);
      CHECK(cell.reciprocal(2)[2] == 0.6283185307_a);

      CHECK(cell.volume() == 1000.0_a);

			CHECK(cell.to_cartesian(vector3<double, contravariant>(1.0, 0.0, 0.0))[0] == 7.0710678119_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(1.0, 0.0, 0.0))[1] == 7.0710678119_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(1.0, 0.0, 0.0))[2] == 0.0_a);

			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 1.0, 0.0))[0] == -7.0710678119_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 1.0, 0.0))[1] == 7.0710678119_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 1.0, 0.0))[2] == 0.0_a);

			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 0.0, 1.0))[0] == 0.0_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 0.0, 1.0))[1] == 0.0_a);
			CHECK(cell.to_cartesian(vector3<double, contravariant>(0.0, 0.0, 1.0))[2] == 10.0_a);

			CHECK(dot(cell.reciprocal(0), cell.lattice(0)) == Approx(2.0*M_PI));
			CHECK(dot(cell.reciprocal(1), cell.lattice(0)) < 1e-14);
			CHECK(dot(cell.reciprocal(2), cell.lattice(0)) < 1e-14);
			CHECK(dot(cell.reciprocal(0), cell.lattice(1)) < 1e-14);
			CHECK(dot(cell.reciprocal(1), cell.lattice(1)) == 2.0*M_PI);
			CHECK(dot(cell.reciprocal(2), cell.lattice(1)) < 1e-14);
			CHECK(dot(cell.reciprocal(0), cell.lattice(2)) < 1e-14);
			CHECK(dot(cell.reciprocal(1), cell.lattice(2)) < 1e-14);
			CHECK(dot(cell.reciprocal(2), cell.lattice(2)) == 2.0*M_PI);        

			CHECK(cell.to_contravariant(vector3<double>(7.0710678119, 7.0710678119, 0.0))[0] == 1.0_a);
			CHECK(cell.to_contravariant(vector3<double>(7.0710678119, 7.0710678119, 0.0))[1] == (0.0_a).margin(1e-12) );
			CHECK(cell.to_contravariant(vector3<double>(7.0710678119, 7.0710678119, 0.0))[2] == (0.0_a).margin(1e-12) );
			
			CHECK(cell.is_orthogonal());
			CHECK(not cell.is_cartesian());
    }
		
    SECTION("Non-orthogonal cell"){

      systems::cell cell({6.942, 8.799, 4.759}, {9.627, 7.092, 4.819}, {4.091, 0.721, 1.043});
      
      CHECK(cell.lattice(0)[0] == 6.942_a);
      CHECK(cell.lattice(0)[1] == 8.799_a);
      CHECK(cell.lattice(0)[2] == 4.759_a);
      CHECK(cell.lattice(1)[0] == 9.627_a);
      CHECK(cell.lattice(1)[1] == 7.092_a);
      CHECK(cell.lattice(1)[2] == 4.819_a);
      CHECK(cell.lattice(2)[0] == 4.091_a);
      CHECK(cell.lattice(2)[1] == 0.721_a);
      CHECK(cell.lattice(2)[2] == 1.043_a);
    
      CHECK(cell.reciprocal(0)[0] == 3.3736397602_a);
      CHECK(cell.reciprocal(0)[1] == 8.3200742872_a);
      CHECK(cell.reciprocal(0)[2] == -18.9840209206_a);
      CHECK(cell.reciprocal(1)[0] == -4.942140131_a);
      CHECK(cell.reciprocal(1)[1] == -10.517582818_a);
      CHECK(cell.reciprocal(1)[2] == 26.6552948109_a);
      CHECK(cell.reciprocal(2)[0] == 7.4410562534_a);
      CHECK(cell.reciprocal(2)[1] == 10.6318294029_a);
      CHECK(cell.reciprocal(2)[2] == -30.5117208294_a);
    
      CHECK(cell.volume() == 7.305321831_a);

			CHECK(dot(cell.reciprocal(0), cell.lattice(0)) == Approx(2.0*M_PI));
			CHECK(dot(cell.reciprocal(1), cell.lattice(0)) < 1e-12);
			CHECK(dot(cell.reciprocal(2), cell.lattice(0)) < 1e-12);
			CHECK(dot(cell.reciprocal(0), cell.lattice(1)) < 1e-12);
			CHECK(dot(cell.reciprocal(1), cell.lattice(1)) == Approx(2.0*M_PI));
			CHECK(dot(cell.reciprocal(2), cell.lattice(1)) < 1e-12);
			CHECK(dot(cell.reciprocal(0), cell.lattice(2)) < 1e-12);
			CHECK(dot(cell.reciprocal(1), cell.lattice(2)) < 1e-12);
			CHECK(dot(cell.reciprocal(2), cell.lattice(2)) == Approx(2.0*M_PI));
			
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[0] == 0.121797_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[1] == -1.161093_a);
      CHECK(cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[2] == -0.553419_a);

      CHECK(cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[0] == -39.3396165136_a);
      CHECK(cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[1] == 50.8091863243_a);
      CHECK(cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[2] == -52.6483546581_a);

      CHECK(!cell.contains(vector3<double, contravariant>(0.5, 0.5, 0.5)));
			CHECK(!cell.contains(vector3<double, contravariant>(1.5, 0.5, 0.5)));
      CHECK(!cell.contains(vector3<double, contravariant>(0.5, -0.1, 0.0)));
      CHECK(!cell.contains(vector3<double, contravariant>(0.5, 0.5, -1.0)));

			{
				auto vv = cell.to_contravariant(vector3<double, cartesian>{9.627, 7.092, 4.819});
				CHECK(fabs(vv[0]) < 1e-12);     
				CHECK(vv[1] == 1.0_a);
				CHECK(fabs(vv[2]) < 1e-12);

				auto vv2 = cell.to_cartesian(vv);
				CHECK(vv2[0] == 9.627_a);       
				CHECK(vv2[1] == 7.092_a);
				CHECK(vv2[2] == 4.819_a);

				CHECK(norm(vv2) == Approx(dot(vv, cell.to_covariant(vv))));
			}

			CHECK(not cell.is_orthogonal());
			CHECK(not cell.is_cartesian());

			cell.save(comm, "cell_non_orthogonal_save_2");

			auto read_cell = systems::cell::load("cell_non_orthogonal_save_2");

			CHECK(read_cell.lattice(0)[0] == 6.942_a);
			CHECK(read_cell.lattice(0)[1] == 8.799_a);
			CHECK(read_cell.lattice(0)[2] == 4.759_a);
			CHECK(read_cell.lattice(1)[0] == 9.627_a);
			CHECK(read_cell.lattice(1)[1] == 7.092_a);
			CHECK(read_cell.lattice(1)[2] == 4.819_a);
			CHECK(read_cell.lattice(2)[0] == 4.091_a);
			CHECK(read_cell.lattice(2)[1] == 0.721_a);
			CHECK(read_cell.lattice(2)[2] == 1.043_a);
		
			CHECK(read_cell.reciprocal(0)[0] == 3.3736397602_a);
			CHECK(read_cell.reciprocal(0)[1] == 8.3200742872_a);
			CHECK(read_cell.reciprocal(0)[2] == -18.9840209206_a);
			CHECK(read_cell.reciprocal(1)[0] == -4.942140131_a);
			CHECK(read_cell.reciprocal(1)[1] == -10.517582818_a);
			CHECK(read_cell.reciprocal(1)[2] == 26.6552948109_a);
			CHECK(read_cell.reciprocal(2)[0] == 7.4410562534_a);
			CHECK(read_cell.reciprocal(2)[1] == 10.6318294029_a);
			CHECK(read_cell.reciprocal(2)[2] == -30.5117208294_a);
		
			CHECK(read_cell.volume() == 7.305321831_a);

			CHECK(dot(read_cell.reciprocal(0), read_cell.lattice(0)) == Approx(2.0*M_PI));
			CHECK(dot(read_cell.reciprocal(1), read_cell.lattice(0)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(2), read_cell.lattice(0)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(0), read_cell.lattice(1)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(1), read_cell.lattice(1)) == Approx(2.0*M_PI));
			CHECK(dot(read_cell.reciprocal(2), read_cell.lattice(1)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(0), read_cell.lattice(2)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(1), read_cell.lattice(2)) < 1e-12);
			CHECK(dot(read_cell.reciprocal(2), read_cell.lattice(2)) == Approx(2.0*M_PI));
			
			CHECK(read_cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[0] == 0.121797_a);
			CHECK(read_cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[1] == -1.161093_a);
			CHECK(read_cell.to_cartesian(vector3<double, contravariant>(0.2, -0.5, 0.867))[2] == -0.553419_a);

			CHECK(read_cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[0] == -39.3396165136_a);
			CHECK(read_cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[1] == 50.8091863243_a);
			CHECK(read_cell.to_contravariant(vector3<double>(0.66, -23.77, 2.72))[2] == -52.6483546581_a);

			CHECK(!read_cell.contains(vector3<double, contravariant>(0.5, 0.5, 0.5)));
			CHECK(!read_cell.contains(vector3<double, contravariant>(1.5, 0.5, 0.5)));
			CHECK(!read_cell.contains(vector3<double, contravariant>(0.5, -0.1, 0.0)));
			CHECK(!read_cell.contains(vector3<double, contravariant>(0.5, 0.5, -1.0)));

			{
				auto vv = read_cell.to_contravariant(vector3<double, cartesian>{9.627, 7.092, 4.819});
				CHECK(fabs(vv[0]) < 1e-12);			
				CHECK(vv[1] == 1.0_a);
				CHECK(fabs(vv[2]) < 1e-12);

				auto vv2 = read_cell.to_cartesian(vv);
				CHECK(vv2[0] == 9.627_a);				
				CHECK(vv2[1] == 7.092_a);
				CHECK(vv2[2] == 4.819_a);

				CHECK(norm(vv2) == Approx(dot(vv, read_cell.to_covariant(vv))));
			}

			CHECK(not read_cell.is_orthogonal());
			CHECK(not read_cell.is_cartesian());
			
    }

    SECTION("Bad load and save"){

			CHECK_THROWS(systems::cell::load("this_directory_doesnt_exist"));

			auto cell = systems::cell::cubic(10.2_b).periodic();
			
			CHECK_THROWS(cell.save(comm, "\0"));

			//generate a directory with the name of the file we will try to create
			utils::create_directory(comm, "dummy_dir/periodicity");
				
			CHECK_THROWS(cell.save(comm, "dummy_dir"));

		}
  }
}
#endif
