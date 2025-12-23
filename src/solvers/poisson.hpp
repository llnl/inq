/* -*- indent-tabs-mode: t -*- */

#ifndef SOLVERS_POISSON
#define SOLVERS_POISSON

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/complex.hpp>
#include <math/vector3.hpp>
#include <gpu/array.hpp>
#include <basis/field.hpp>
#include <basis/fourier_space.hpp>
#include <operations/transform.hpp>
#include <operations/transfer.hpp>

#include <utils/profiling.hpp>

namespace inq {
namespace solvers {

class poisson {

public:

	poisson() = delete;
	
	struct poisson_kernel_3d {
		GPU_FUNCTION auto operator()(vector3<double, cartesian> gg, double const zeroterm) const {
			auto g2 = norm(gg);
			if(g2 < 1e-6) return zeroterm;
			return -1.0/g2;
		}
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////

	struct poisson_kernel_2d {
		double rc_;
		
		GPU_FUNCTION auto operator()(vector3<double, cartesian> gg, double const) const {
			auto gpar = hypot(gg[0], gg[1]);
			auto gz = fabs(gg[2]);
			auto g2 = norm(gg);
			
			if(g2 < 1e-6) return 0.5*rc_*rc_;
			if(gpar < 1e-12) return -(1.0 - cos(gz*rc_) - gz*rc_*sin(gz*rc_))/g2;
			return -(1.0 + exp(-gpar*rc_)*(gz*sin(gz*rc_)/gpar - cos(gz*rc_)))/g2;
		}
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////

	struct poisson_kernel_0d {
		double rc_;
		
		GPU_FUNCTION auto operator()(vector3<double, cartesian> gg, double const) const {
			auto g2 = norm(gg);

			// this is the kernel of C. A. Rozzi et al., Phys. Rev. B 73, 205119 (2006).
			if(g2 < 1e-6) return -0.5*rc_*rc_;
			return -(1.0 - cos(rc_*sqrt(g2)))/g2;
		}
	};
	
	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename KernelType, typename FieldSetType>
	static void poisson_apply_kernel(KernelType const kernel, FieldSetType & density, vector3<double> const & gshift = {0.0, 0.0, 0.0}, double const zeroterm = 0.0) {

		static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only makes sense in fourier_space");

		CALI_CXX_MARK_FUNCTION;
		
		const double scal = (-4.0*M_PI)/density.basis().size();
		
		gpu::run(density.basis().local_sizes()[2], density.basis().local_sizes()[1], density.basis().local_sizes()[0],
						 [point_op = density.basis().point_op(), dens = begin(density.hypercubic()), scal, nst = density.local_set_size(), kernel, gshift, zeroterm] GPU_LAMBDA (auto i2, auto i1, auto i0){
							 
							 auto kerg = kernel(point_op.gvector_cartesian(i0, i1, i2) + gshift, zeroterm/(-4*M_PI));
							 for(int ist = 0; ist < nst; ist++) dens[i0][i1][i2][ist] *= scal*kerg;
						 });
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////

private:
	
	static auto poisson_solve_3d(basis::field<basis::real_space, complex> const & density) {

		CALI_CXX_MARK_FUNCTION;
		
		auto potential_fs = operations::transform::to_fourier(density);
		poisson_apply_kernel(poisson_kernel_3d{}, potential_fs);
		return operations::transform::to_real(std::move(potential_fs),  /*normalize = */ false);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	static void poisson_solve_in_place_3d(basis::field_set<basis::real_space, complex> & density, vector3<double> const & gshift, double const zeroterm) {

		CALI_CXX_MARK_FUNCTION;

		auto potential_fs = operations::transform::to_fourier(std::move(density));
		poisson_apply_kernel(poisson_kernel_3d{}, potential_fs, gshift, zeroterm);
		density = operations::transform::to_real(std::move(potential_fs),  /*normalize = */ false);
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////////	
	
	static basis::field<basis::real_space, complex> poisson_solve_2d(basis::field<basis::real_space, complex> const & density) {

		CALI_CXX_MARK_FUNCTION;

		auto potential2x = operations::transfer::enlarge(density, density.basis().enlarge({1, 1, 2}));
		auto potential_fs = operations::transform::to_fourier(potential2x);

		const auto cutoff_radius = density.basis().rlength()[2];
		poisson_apply_kernel(poisson_kernel_2d{cutoff_radius}, potential_fs);

		potential2x = operations::transform::to_real(potential_fs,  /*normalize = */ false);
		auto potential = operations::transfer::shrink(potential2x, density.basis());

		return potential;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	static void poisson_solve_in_place_2d(basis::field_set<basis::real_space, complex> & density, vector3<double> const & gshift, double const zeroterm) {

		CALI_CXX_MARK_FUNCTION;

		auto potential2x = operations::transfer::enlarge(density, density.basis().enlarge({1, 1, 2}));
		auto potential_fs = operations::transform::to_fourier(std::move(potential2x));
			
		const auto cutoff_radius = density.basis().rlength()[2];
		poisson_apply_kernel(poisson_kernel_2d{cutoff_radius}, potential_fs, gshift, zeroterm);
		
		potential2x = operations::transform::to_real(std::move(potential_fs),  /*normalize = */ false);
		density = operations::transfer::shrink(potential2x, density.basis());
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	static basis::field<basis::real_space, complex> poisson_solve_0d(basis::field<basis::real_space, complex> const & density) {

		CALI_CXX_MARK_FUNCTION;

		auto potential2x = operations::transfer::enlarge(density, density.basis().enlarge(2));
		auto potential_fs = operations::transform::to_fourier(potential2x);
			
		const auto cutoff_radius = potential2x.basis().min_rlength()/2.0;
		poisson_apply_kernel(poisson_kernel_0d{cutoff_radius}, potential_fs);
		
		potential2x = operations::transform::to_real(potential_fs,  /*normalize = */ false);
		auto potential = operations::transfer::shrink(potential2x, density.basis());

		return potential;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	static void poisson_solve_in_place_0d(basis::field_set<basis::real_space, complex> & density, vector3<double> const & gshift, double const zeroterm) {

		CALI_CXX_MARK_FUNCTION;

		auto potential2x = operations::transfer::enlarge(density, density.basis().enlarge(2));
		auto potential_fs = operations::transform::to_fourier(std::move(potential2x));
			
		const auto cutoff_radius = potential2x.basis().min_rlength()/2.0;
		poisson_apply_kernel(poisson_kernel_0d{cutoff_radius}, potential_fs, gshift, zeroterm);

		potential2x = operations::transform::to_real(std::move(potential_fs),  /*normalize = */ false);
		density = operations::transfer::shrink(potential2x, density.basis());
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
public:
	
	static auto solve(const basis::field<basis::real_space, complex> & density) {

		CALI_CXX_MARK_SCOPE("poisson(complex)");
		
		if(density.basis().cell().periodicity() == 3){
			return poisson_solve_3d(density);
		} else if(density.basis().cell().periodicity() == 2){
			return poisson_solve_2d(density);
		} else {
			return poisson_solve_0d(density);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Space = cartesian>
	static void in_place(basis::field_set<basis::real_space, complex> & density, vector3<double, Space> const & gshift = {0.0, 0.0, 0.0}, double const zeroterm = 0.0) {

		CALI_CXX_MARK_SCOPE("poisson(complex)");

		auto gshift_cart = density.basis().cell().to_cartesian(gshift);
		
		if(density.basis().cell().periodicity() == 3){
			poisson_solve_in_place_3d(density, gshift_cart, zeroterm);
		} else if(density.basis().cell().periodicity() == 2){
			return poisson_solve_in_place_2d(density, gshift_cart, zeroterm);
		} else {
			poisson_solve_in_place_0d(density, gshift_cart, zeroterm);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	static basis::field<basis::real_space, double> solve(const basis::field<basis::real_space, double> & density) {

		CALI_CXX_MARK_SCOPE("poisson(real)");
		
		auto complex_potential = solve(complex_field(density));
		return real_field(complex_potential);
	}
		
};    
	
}
}
#endif

#ifdef INQ_SOLVERS_POISSON_UNIT_TEST
#undef INQ_SOLVERS_POISSON_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>
#include <operations/integral.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	namespace multi = boost::multi;
	using namespace basis;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};
	
	SECTION("Periodic"){

		basis::real_space rs(systems::cell::orthorhombic(10.0_b, 10.0_b, 13.7_b), /*spacing =*/ 0.1, comm);

		CHECK(rs.cell().periodicity() == 3);
		
		CHECK(rs.sizes()[0] == 100);
		CHECK(rs.sizes()[1] == 100);
		CHECK(rs.sizes()[2] == 140);
		
		int const nst = 5;
		
		field<real_space, complex> density(rs);
		field_set<real_space, complex> density_set(rs, nst);
		
		//Plane wave
		{

			double kk = 2.0*M_PI/rs.rlength()[0];

			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [kk, dens = begin(density.cubic()), dset = begin(density_set.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2)] GPU_LAMBDA (auto iz, auto iy, auto ix) {
								 
								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
								 
								 double xx = point_op.rvector_cartesian(ixg, iyg, izg)[0];
								 dens[ix][iy][iz] = complex(cos(kk*xx), sin(kk*xx));
								 for(int ist = 0; ist < nst; ist++) dset[ix][iy][iz][ist] = (1.0 + ist)*dens[ix][iy][iz];
							 });

			auto potential = solvers::poisson::solve(density);
			solvers::poisson::in_place(density_set);

			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), 0.0,
													 [kk, pot = begin(potential.cubic()), dens = begin(density.cubic()), dset = begin(density_set.hypercubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {

														 auto acc = fabs(pot[ix][iy][iz] - 4*M_PI/(kk*kk)*dens[ix][iy][iz]);
														 for(int ist = 0; ist < nst; ist++) acc += fabs(dset[ix][iy][iz][ist]/(1.0 + ist) - 4*M_PI/(kk*kk)*dens[ix][iy][iz]);
														 return acc;
													 });

			comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});

			diff /= rs.size()*(1.0 + nst);
		
			CHECK(diff < 1.0e-13);
	
		}

		//Real plane wave
		{

			field<real_space, double> density(rs);

			double kk = 8.0*M_PI/rs.rlength()[1];

			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [kk, dens = begin(density.cubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2)] GPU_LAMBDA (auto iz, auto iy, auto ix) {
								 
								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
								 double yy = point_op.rvector_cartesian(ixg, iyg, izg)[1];
								 dens[ix][iy][iz] = cos(kk*yy);
							 });


			auto potential = solvers::poisson::solve(density);
			
			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), 0.0,
													 [kk, pot = begin(potential.cubic()), dens = begin(density.cubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {
														 return fabs(pot[ix][iy][iz] - 4*M_PI/(kk*kk)*dens[ix][iy][iz]);
													 });

			comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});

			diff /= rs.size();
		
			CHECK(diff < 1e-8);

		}
	}


	SECTION("Point charge finite") {
		basis::real_space rs(systems::cell::cubic(8.0_b).finite(), /*spacing =*/ 0.09, comm);

		CHECK(rs.cell().periodicity() == 0);
		
		CHECK(rs.sizes()[0] == 90);
		CHECK(rs.sizes()[1] == 90);
		CHECK(rs.sizes()[2] == 90);

		int const nst = 3;
		
		field<real_space, complex> density(rs);
		field_set<real_space, complex> density_set(rs, nst);

		gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
						 [dens = begin(density.cubic()), dset = begin(density_set.hypercubic()), point_op = rs.point_op(), vol_el = rs.volume_element()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
								 
							 dens[ix][iy][iz] = 0.0;
							 for(int ist = 0; ist < nst; ist++) dset[ix][iy][iz][ist] = 0.0;
							 if(point_op.r2(ix, iy, iz) < 1e-10) {
								 dens[ix][iy][iz] = -1.0/vol_el;
								 for(int ist = 0; ist < nst; ist++) dset[ix][iy][iz][ist] = -(1.0 + ist)/vol_el;
							 }
						 });

			CHECK(real(operations::integral(density)) == -1.0_a);
			
			auto potential = solvers::poisson::solve(density);
			solvers::poisson::in_place(density_set);

			auto errors = gpu::array<int, 1>(2, 0);
			
			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [pot = begin(potential.cubic()), dset = begin(density_set.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2), er = begin(errors)] GPU_LAMBDA (auto iz, auto iy, auto ix) {

								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
											 
								 auto rr = point_op.rlength(ixg, iyg, izg);

								 // it should be close to -1/r
								 if(rr <= 1) return;
								 if(fabs(pot[ix][iy][iz]*rr + 1.0) >= 0.025) gpu::atomic(er[0])++;
								 for(int ist = 0; ist < nst; ist++) if(fabs(dset[ix][iy][iz][ist]*rr/(1.0 + ist) + 1.0) >= 0.025) gpu::atomic(er[1])++;
							 });

			CHECK(errors[0] == 0);
			CHECK(errors[1] == 0);
	}

	SECTION("Point charge 2d periodic"){
		
		basis::real_space rs(systems::cell::orthorhombic(6.0_b, 6.0_b, 9.0_b).periodicity(2), /*spacing =*/ 0.12, comm);
		
		CHECK(rs.cell().periodicity() == 2);
		
		CHECK(rs.sizes()[0] == 50);
		CHECK(rs.sizes()[1] == 50);
		CHECK(rs.sizes()[2] == 75);
		
		int const nst = 3;
		
		field<real_space, complex> density(rs);
		field_set<real_space, complex> density_set(rs, nst);

		gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
						 [dens = begin(density.cubic()), dset = begin(density_set.hypercubic()), point_op = rs.point_op(), vol_el = rs.volume_element()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
							 
							 dens[ix][iy][iz] = 0.0;
							 for(int ist = 0; ist < nst; ist++) dset[ix][iy][iz][ist] = 0.0;
							 if(point_op.r2(ix, iy, iz) < 1e-10) {
								 dens[ix][iy][iz] = -1.0/vol_el;
								 for(int ist = 0; ist < nst; ist++) dset[ix][iy][iz][ist] = -(1.0 + ist)/vol_el;
							 }
						 });
		
		CHECK(real(operations::integral(density)) == -1.0_a);
		
		auto potential = solvers::poisson::solve(density);
		solvers::poisson::in_place(density_set);

		//MISSING: CHECKING THE RESULTS
	}
}
#endif
