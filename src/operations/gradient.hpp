/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__GRADIENT
#define OPERATIONS__GRADIENT

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


template <typename FieldSetType, typename FactorType = double,
					typename ResultType = typename FieldSetType::template template_type<typename FieldSetType::basis_type, vector3<typename FieldSetType::element_type, covariant>>>
ResultType gradient(FieldSetType const & ff, double factor = 1.0, vector3<double, covariant> const & shift = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	if constexpr(std::is_same_v<typename FieldSetType::basis_type, basis::real_space>) {
		if constexpr(std::is_same_v<typename FieldSetType::element_type, double>) {		
			return real_field(operations::transform::to_real(operations::gradient(operations::transform::to_fourier(complex_field(ff)), factor, shift)));
		} else {
			return operations::transform::to_real(operations::gradient(operations::transform::to_fourier(ff), factor, shift));
		}
	} else {
		
		static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for real or fourier_space");
		
		ResultType gradff(ff.skeleton());
		
		gpu::run(gradff.local_set_size(), gradff.basis().local_sizes()[2], gradff.basis().local_sizes()[1], gradff.basis().local_sizes()[0],
						 [point_op = ff.basis().point_op(), gradffcub = begin(gradff.hypercubic()), ffcub = begin(ff.hypercubic()), factor, shift]
						 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
							 auto grad = factor*complex(0.0, 1.0)*(point_op.gvector(i0, i1, i2) + shift);
							 gradffcub[i0][i1][i2][ist] = grad*ffcub[i0][i1][i2][ist];
						 });
		return gradff;
	}
}

}
}
#endif

#ifdef INQ_OPERATIONS_GRADIENT_UNIT_TEST
#undef INQ_OPERATIONS_GRADIENT_UNIT_TEST

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
		
		auto gradff = [ff] (auto & kk, auto & rr) {
			return complex(0.0, 1.0)*kk*ff(kk, rr);
		};
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto grad = operations::gradient(func, factor);
		
		auto diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = rs.cell().to_covariant((ist + 1.0)*factor*gradff(kvec, vec));
						diff += fabs(grad.hypercubic()[ix][iy][iz][ist] - anvalue);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		CHECK(diff < 3.0e-8) ;
		
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
		
		auto gradff = [ff] (auto & kk, auto & rr) {
			return complex(0.0, 1.0)*kk*ff(kk, rr);			
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

		auto grad = operations::gradient(func, factor);
		
		auto diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = rs.cell().to_covariant((ist + 1.0)*factor*gradff(kvec, vec));
						diff += fabs(grad.hypercubic()[ix][iy][iz][ist] - anvalue);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		CHECK(diff < 3.0e-8);
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
		
		auto gradff = [ff] (auto & kk, auto & rr) {
			return complex(0.0, 1.0)*kk*ff(kk, rr);
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

		auto grad = operations::gradient(func, factor);

		auto diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = rs.cell().to_covariant((ist + 1.0)*factor*gradff(kvec, vec));
						diff += fabs(grad.hypercubic()[ix][iy][iz][ist] - anvalue);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		CHECK(diff < 3.0e-7);

	}
	
}
#endif

