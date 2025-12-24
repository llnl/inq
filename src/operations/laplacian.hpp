/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__LAPLACIAN
#define OPERATIONS__LAPLACIAN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>

#include <gpu/run.hpp>
#include <basis/field_set.hpp>
#include <basis/fourier_space.hpp>
#include <math/vector3.hpp>
#include <operations/transform.hpp>
#include <utils/profiling.hpp>

namespace inq {
namespace operations {

template <typename FieldSetType, typename FactorType = double>
void laplacian_add(FieldSetType const & ff, FieldSetType & laplff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");
			
	gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), factor, gradcoeff]
					 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 laplffcub[i0][i1][i2][ist] += lapl*ffcub[i0][i1][i2][ist];
					 });
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorType = double>
void laplacian_in_place(FieldSetType & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");

	gpu::run(ff.set_part().local_size(), ff.basis().local_sizes()[2], ff.basis().local_sizes()[1], ff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(),
						ffcub = begin(ff.hypercubic()), factor, gradcoeff] GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 ffcub[i0][i1][i2][ist] = ffcub[i0][i1][i2][ist]*lapl;
					 });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorType = double>
FieldSetType laplacian(FieldSetType const & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	if constexpr(std::is_same_v<typename FieldSetType::basis_type, basis::real_space>) {
		return operations::transform::to_real(operations::laplacian(operations::transform::to_fourier(ff), factor));		
	} else {
		
		static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for real or fourier_space");
		
		FieldSetType laplff(ff.skeleton());
		
		gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
						 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), factor, gradcoeff]
						 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
							 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
							 laplffcub[i0][i1][i2][ist] = lapl*ffcub[i0][i1][i2][ist];
						 });
		
		return laplff;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorType = double>
gpu::array<double, 1> laplacian_expectation_value(FieldSetType & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");

	
	auto k2 = -0.25*ff.basis().cell().dot(gradcoeff, gradcoeff);
	
	auto evs = gpu::run(ff.local_set_size(), gpu::reduce(ff.basis().local_sizes()[2]), gpu::reduce(ff.basis().local_sizes()[1]), gpu::reduce(ff.basis().local_sizes()[0]), 0.0,
											[point_op = ff.basis().point_op(), ffcub = begin(ff.hypercubic()), fac = factor*ff.basis().volume_element(), gradcoeff, k2]
											GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
												auto lapl = fac*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)) + k2);
												return real(conj(ffcub[i0][i1][i2][ist])*lapl*ffcub[i0][i1][i2][ist]);
											});
	
	if(ff.spinor_dim() == 2) {
		auto spinor_evs = decltype(evs)(ff.local_spinor_set_size());
		gpu::run(ff.local_spinor_set_size(), [sev = begin(spinor_evs), ev = begin(evs), nst = ff.local_spinor_set_size()] GPU_LAMBDA (auto ist) {
			sev[ist] = ev[ist] + ev[ist + nst];
		});
		
		evs = std::move(spinor_evs);
	}

	if(ff.basis().comm().size() > 1) {
		ff.basis().comm().all_reduce_in_place_n(raw_pointer_cast(evs.data_elements()), evs.num_elements(), std::plus<>{});
	}
	
	return evs;
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////


}
}
#endif

#ifdef INQ_OPERATIONS_LAPLACIAN_UNIT_TEST
#undef INQ_OPERATIONS_LAPLACIAN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using namespace operations;
	using Catch::Approx;
	
	parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});
	auto set_comm = basis::set_subcomm(cart_comm);
	auto basis_comm = basis::basis_subcomm(cart_comm);	

	SECTION("Plane-wave -- field_set"){
		
		double lx = 9;
		double ly = 12;
		double lz = 10;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::orthorhombic(lx*1.0_b, ly*1.0_b, lz*1.0_b), /*spacing =*/ 0.49672941, basis_comm);

		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		//Define k-vector for test function
		auto kvec = 2.0*M_PI*vector3<double>(1.0/lx, 1.0/ly, 1.0/lz);
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-8) ;
		CHECK(diff_in_place < 1.0e-8);
		CHECK(diff_add < 5.0e-7);
		
	}
	
	SECTION("Plane-wave -- rotated"){
		
		double ll = 9;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::lattice({ll*1.0_b/sqrt(2), ll*1.0_b/sqrt(2), 0.0_b}, {-ll*1.0_b/sqrt(2), ll*1.0_b/sqrt(2), 0.0_b}, {0.0_b, 0.0_b, ll*1.0_b}), /*spacing =*/ 0.49672941, basis_comm);
		
		CHECK(rs.cell().volume() == ll*ll*ll);
		
		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		auto kvec = 2.0*M_PI*vector3<double>(sqrt(2.0)/ll, sqrt(2.0)/ll, 0.0);
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};

		CHECK(norm(kvec) == Approx(rs.cell().norm(rs.cell().to_covariant(kvec))));
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-8) ;
		CHECK(diff_in_place < 1.0e-8);
		CHECK(diff_add < 5.0e-7);
		
	}
	
	SECTION("Plane-wave -- non-orthogonal"){
		
		double ll = 5.89;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::lattice({0.0_b, ll*1.0_b, ll*1.0_b}, {ll*1.0_b, 0.0_b, ll*1.0_b}, {ll*1.0_b, ll*1.0_b, 0.0_b}), /*spacing =*/ 0.49672941, basis_comm);
		
		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		auto kvec = rs.cell().to_cartesian(2.0*rs.cell().reciprocal(2) + 3.0*rs.cell().reciprocal(2) - 1.0*rs.cell().reciprocal(2));
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};

		CHECK(norm(kvec) == Approx(rs.cell().norm(rs.cell().to_covariant(kvec))));
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-7) ;
		CHECK(diff_in_place < 1.0e-7);
		CHECK(diff_add < 5.0e-7);
		
	}
	
}
#endif

