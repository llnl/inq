/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__SINGULARITY_CORRECTION
#define INQ__HAMILTONIAN__SINGULARITY_CORRECTION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <ionic/brillouin.hpp>

namespace inq {
namespace hamiltonian {

// Implements the method for the singularity correction of Carrier et al. PRB 75 205126 (2007)
// https://doi.org/10.1103/PhysRevB.75.205126
class singularity_correction {

	gpu::array<double, 1> fk_;
	double fzero_;
	int nkpoints_;
	double cell_volume_;

public:

	template <class Vector3>
	GPU_FUNCTION static auto cell_projection(systems::cell const & cell, Vector3 const & qpoint) {
		auto qp_cart = cell.to_cartesian(qpoint);
		return vector3<double>{dot(cell.lattice(0), qp_cart),
													 dot(cell.lattice(1), qp_cart),
													 dot(cell.lattice(2), qp_cart)};
	}
	
	// the function defined in Eq. 16
	GPU_FUNCTION static auto auxiliary(vector3<double> const & dp1, vector3<double> const & dp2, vector3<double> const & dp3){

		double s0, s1, s2, c0, c1, c2;
		sincos(dp3[0], &s0, &c0);
		sincos(dp3[1], &s1, &c1);
		sincos(dp3[2], &s2, &c2);

		auto v1 = dp1[0]*(1.0 - c0) + dp1[1]*(1.0 - c1) + dp1[2]*(1.0 - c2);
		auto v2 = dp2[0]*s0*s1 + dp2[1]*s1*s2 + dp2[1]*s2*s0;
		
		return 4.0*M_PI*M_PI/(0.5*v1 + v2);
	}
	
	static auto auxiliary(systems::cell const & cell, vector3<double, covariant> const & qpoint){
		vector3<double> dp1, dp2;
		for(int jj = 0; jj < 3; jj++){
			auto jjp1 = jj + 1;
			if(jjp1 == 3) jjp1 = 0;
			dp1[jj] = 4.0*cell.dot(cell.reciprocal(jj), cell.reciprocal(jj));
			dp2[jj] = 2.0*cell.dot(cell.reciprocal(jj), cell.reciprocal(jjp1));
		}
		
		return auxiliary(dp1, dp2, cell_projection(cell, qpoint));
	}

	GPU_FUNCTION static auto length(int istep) {
		return 1.0/pow(3.0, istep);
	}

	static double calculate_fzero(vector3<double> const & dp1, vector3<double> const & dp2, systems::cell const & cell) {

		CALI_CXX_MARK_SCOPE("singularity_correction::fzero");
		
		auto const nsteps = 7;
		auto const nk = 60;
		auto const kvol_element = pow(2.0*M_PI/(2.0*nk + 1.0), 3)/cell.volume();

		auto fzero = gpu::run(gpu::reduce(nk + 1), gpu::reduce(2*nk + 1), gpu::reduce(2*nk + 1), 0.0,
											[nk, cell, dp1, dp2] GPU_LAMBDA (auto ikx, auto iky, auto ikz) {
												iky -= nk;
												ikz -= nk;								 
												
												if(3*ikx <= nk and 3*std::abs(iky) <= nk and 3*std::abs(ikz) <= nk) return 0.0;
												
												auto qpoint = (M_PI/nk)*vector3<int, covariant>(ikx, iky, ikz);
												auto dp3 = cell_projection(cell, qpoint);

												auto psum = 0.0;
												for(int istep = 0; istep < nsteps; istep++){
													auto ll = length(istep);
													psum += ll*ll*ll*auxiliary(dp1, dp2, dp3*ll);
												}

												return psum;
											});

		fzero *= 8.0*M_PI/pow(2.0*M_PI, 3)*kvol_element;
		fzero += 4.0*M_PI*pow(3.0/(4.0*M_PI), 1.0/3.0)*pow(cell.volume(), 2.0/3.0)/M_PI/cell.volume()*length(nsteps - 1);
		
		return fzero;
	}

	static gpu::array<double, 1> calculate_fk(vector3<double> const & dp1, vector3<double> const & dp2, systems::cell const & cell, ionic::brillouin const & bzone) {
		CALI_CXX_MARK_SCOPE("singularity_correction::fk");

		auto foka    = gpu::array<double, 1>(bzone.size(), 0.0);
		auto weights = gpu::array<double, 1>(bzone.size());
		auto kpoints = gpu::array<vector3<double>, 1>(bzone.size());

		for(int ik = 0; ik < bzone.size(); ik++){
			weights[ik] = 4.0*M_PI/cell.volume()*bzone.kpoint_weight(ik);
			kpoints[ik] = cell.to_cartesian(bzone.kpoint(ik));
		}

		gpu::run(bzone.size(), bzone.size(),
						 [dp1, dp2, fk = begin(foka), we = begin(weights), kp = begin(kpoints), cell] GPU_LAMBDA (auto jk, auto ik) {
							 if(jk <= ik) return;
							 
							 auto qpoint = kp[ik] - kp[jk];
							 if(cell.norm(qpoint) < 1e-6) return;
							 auto aux = auxiliary(dp1, dp2, cell_projection(cell, qpoint));
							 //  cell_projection is odd, and auxiliary is odd (with respect to the 3rd argument) so we can use aux for both cases
							 gpu::atomic::add(&fk[ik], we[jk]*aux);
							 gpu::atomic::add(&fk[jk], we[ik]*aux);
						 });

		return foka;
	}
	
	singularity_correction(systems::cell const & cell, ionic::brillouin const & bzone):
		nkpoints_(bzone.size()),
		cell_volume_(cell.volume())
	{
		CALI_CXX_MARK_SCOPE("singularity_correction::constructor");

		vector3<double> dp1, dp2;
		for(int jj = 0; jj < 3; jj++){
			auto jjp1 = jj + 1;
			if(jjp1 == 3) jjp1 = 0;
			dp1[jj] = 4.0*cell.dot(cell.reciprocal(jj), cell.reciprocal(jj));
			dp2[jj] = 2.0*cell.dot(cell.reciprocal(jj), cell.reciprocal(jjp1));
		}

		fk_ = calculate_fk(dp1, dp2, cell, bzone);
		fzero_ = calculate_fzero(dp1, dp2, cell);
	}

	auto fk(int ik) const {
		return fk_[ik];
	}

	auto fzero() const {
		return fzero_;
	}	 

	auto operator()(int ik) const {
		return -nkpoints_*cell_volume_*(fk(ik) - fzero());
	}
	
};
}
}

#endif

#ifdef INQ_HAMILTONIAN_SINGULARITY_CORRECTION_UNIT_TEST
#undef INQ_HAMILTONIAN_SINGULARITY_CORRECTION_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Auxiliary function cubic"){
		auto aa = 10.18_b;

		auto ions = systems::ions(systems::cell::lattice({aa, 0.0_b, 0.0_b}, {0.0_b, aa, 0.0_b}, {0.0_b, 0.0_b, aa}));
		auto const & cell = ions.cell();

		auto bzone = ionic::brillouin(ions, input::kpoints::grid({2, 2, 2}));

		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{0.0, -0.5, 0.0}) == 25.9081_a);
		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{8.3333333333333332E-003, 7.4999999999999997E-002, 0.26666666666666666}) == 42.650855183181122_a);
		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{0.11666666666666667, 0.20000000000000001, 0.21666666666666667}) == 29.780683447124286_a);		
		
		auto sing = hamiltonian::singularity_correction(cell, bzone);

		CHECK(sing.fzero() == 0.30983869660201141_a);

		CHECK(sing.fk(0) == 0.18644848345224296_a);
		CHECK(sing.fk(1) == 0.18644848345224296_a);
		CHECK(sing.fk(2) == 0.18644848345224296_a);
		CHECK(sing.fk(3) == 0.18644848345224296_a);
		CHECK(sing.fk(4) == 0.18644848345224296_a);
		CHECK(sing.fk(5) == 0.18644848345224296_a);
		CHECK(sing.fk(6) == 0.18644848345224296_a);
		CHECK(sing.fk(7) == 0.18644848345224296_a);
		
		CHECK(sing(0) == 1041.3915164701_a);
		CHECK(sing(1) == 1041.3915164701_a);
		CHECK(sing(2) == 1041.3915164701_a);
		CHECK(sing(3) == 1041.3915164701_a);
		CHECK(sing(4) == 1041.3915164701_a);
		CHECK(sing(5) == 1041.3915164701_a);
		CHECK(sing(6) == 1041.3915164701_a);
		CHECK(sing(7) == 1041.3915164701_a);
		
	}
		
	SECTION("Auxiliary function non-orthogonal"){
		auto aa = 6.7408326;
		systems::cell cell(aa*vector3<double>(0.0, 0.5, 0.5), aa*vector3<double>(0.5, 0.0, 0.5), aa*vector3<double>(0.5, 0.5, 0.0));
		auto ions = systems::ions(cell);
		
		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{1.6666666666666666E-002, 0.28333333333333333, 0.39166666666666666}) == 2.77471621018199290_a);
		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{0.12500000000000000,-0.20833333333333334, -0.23333333333333334}) == 3.6560191647005245_a);
		CHECK(hamiltonian::singularity_correction::auxiliary(cell, 2.0*M_PI*vector3<double, covariant>{ 0.14999999999999999, 0.25000000000000000, -3.3333333333333333E-002}) == 5.8717108336249790_a);

		{
			
			auto bzone = ionic::brillouin(ions, input::kpoints::grid({32, 32, 32}));
			auto sing = hamiltonian::singularity_correction(cell, bzone);
			
			CHECK(sing.fzero() == 0.6557402601_a);
			
			CHECK(sing.fk(    0) == 0.6397684561_a);
			CHECK(sing.fk(   25) == 0.6397684561_a);
			CHECK(sing.fk(  369) == 0.6397684561_a);
			CHECK(sing.fk( 1331) == 0.6397684561_a);
			CHECK(sing.fk(10789) == 0.6397684561_a);
			CHECK(sing.fk(20334) == 0.6397684561_a);
			CHECK(sing.fk(29000) == 0.6397684561_a);
			CHECK(sing.fk(32767) == 0.6397684561_a);
			
			CHECK(sing(32767) == 40076.0160362316_a);
		}
		
		{
			
			auto bzone = ionic::brillouin(ions, input::kpoints::grid({2, 2, 2}, true));
			bzone.insert({0.0, 0.0, 0.0}, 0.5);
			bzone.insert({0.5, 0.5, 0.5}, 0.0);
			
			auto sing = hamiltonian::singularity_correction(cell, bzone);

			CHECK(sing.fk(0) == 0.6861351991_a);
			CHECK(sing.fk(1) == 0.5618541064_a);
			CHECK(sing.fk(2) == 0.5618541064_a);
			CHECK(sing.fk(3) == 0.5618541064_a);
			CHECK(sing.fk(4) == 0.5618541064_a);
			CHECK(sing.fk(5) == 0.5618541064_a);
			CHECK(sing.fk(6) == 0.5618541064_a);
			CHECK(sing.fk(7) == 0.6861351991_a);
			CHECK(sing.fk(8) == 0.4349838243_a);
			CHECK(sing.fk(9) == 0.5385514015_a);
			
			CHECK(sing(0) == -23.2745831_a);
			CHECK(sing(1) ==  71.8922676398_a);
			CHECK(sing(2) ==  71.8922676398_a);
			CHECK(sing(3) ==  71.8922676398_a);
			CHECK(sing(4) ==  71.8922676398_a);
			CHECK(sing(5) ==  71.8922676398_a);
			CHECK(sing(6) ==  71.8922676398_a);
			CHECK(sing(7) == -23.2745831_a);
			CHECK(sing(8) == 169.0417611035_a);
			CHECK(sing(9) ==  89.7360521536_a);
		
		}
		
	}	 
}
#endif
