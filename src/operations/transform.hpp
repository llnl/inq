/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__OPERATIONS__TRANSFORM
#define INQ__OPERATIONS__TRANSFORM

// Copyright (C) 2019-2025 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq_config.h> 

#include <gpu/run.hpp>
#include <parallel/alltoall.hpp>
#include <basis/field.hpp>
#include <basis/field_set.hpp>
#include <basis/fourier_space.hpp>
#include <parallel/transpose.hpp>
#include <states/orbital_set.hpp>

#ifdef ENABLE_GPU
#include <multi/adaptors/fft.hpp>
#else
#include <multi/adaptors/fftw.hpp>
#endif

#include <utils/profiling.hpp>
#include <utils/raw_pointer_cast.hpp>

#include <cassert>

namespace inq {
namespace operations {
namespace transform {

///////////////////////////////////////////////////////////////

template <typename FieldSetType>
void zero_outside_sphere(FieldSetType & fphi){
	CALI_CXX_MARK_FUNCTION;
	
	gpu::run(fphi.local_set_size(), fphi.basis().local_sizes()[2], fphi.basis().local_sizes()[1], fphi.basis().local_sizes()[0],
					 [fphicub = begin(fphi.hypercubic()), point_op = fphi.basis().point_op()] GPU_LAMBDA
					 (auto ist, auto iz, auto iy, auto ix){
						 if(point_op.outside_sphere(ix, iy, iz)) fphicub[ix][iy][iz][ist] = zero<typename FieldSetType::element_type>();
					 });
}

///////////////////////////////////////////////////////////////

template <class InArray4D, class OutArray4D>
void to_fourier_array(basis::real_space const & real_basis, basis::fourier_space const & fourier_basis, InArray4D const & array_rs, OutArray4D && array_fs) {

	CALI_CXX_MARK_FUNCTION;
	assert(get<0>(sizes(array_rs)) == real_basis.local_sizes()[0]);
	assert(get<1>(sizes(array_rs)) == real_basis.local_sizes()[1]);
	assert(get<2>(sizes(array_rs)) == real_basis.local_sizes()[2]);

	assert(get<0>(sizes(array_fs)) == fourier_basis.local_sizes()[0]);
	assert(get<1>(sizes(array_fs)) == fourier_basis.local_sizes()[1]);
	assert(get<2>(sizes(array_fs)) == fourier_basis.local_sizes()[2]);

	assert(get<3>(sizes(array_rs)) == get<3>(sizes(array_fs)));
	
	namespace multi = boost::multi;
#ifdef ENABLE_GPU
	namespace fft = multi::fft;
#else
	namespace fft = multi::fftw;
#endif
	
	if(not real_basis.part().parallel()) {
		CALI_CXX_MARK_SCOPE("fft_forward_3d");

		assert(not fourier_basis.reverse_order());
		assert(extensions(array_rs) == extensions(array_fs));
		
		fft::dft_forward({true, true, true, false}, array_rs, array_fs);
		gpu::sync();

	} else {

		assert(fourier_basis.reverse_order());
		
		gpu::array<complex, 4> tmp(extensions(array_rs));

		{
			CALI_CXX_MARK_SCOPE("fft_forward_1d");
			fft::dft_forward({true, false, false, false}, array_rs, tmp);
			gpu::sync();
		}

		parallel::transpose_forward(real_basis.comm(), fourier_basis.cubic_part(2), real_basis.cubic_part(1), tmp);

		assert(get<0>(sizes(tmp)) == fourier_basis.local_sizes()[0]);
		assert(get<1>(sizes(tmp)) == fourier_basis.local_sizes()[1]);
		assert(get<2>(sizes(tmp)) == fourier_basis.local_sizes()[2]);
		assert(get<3>(sizes(array_rs)) == get<3>(sizes(tmp)));
	
		{
			CALI_CXX_MARK_SCOPE("fft_forward_1d");
			
			fft::dft_forward({true, true, false, false}, tmp, array_fs);
			gpu::sync();
		}
		
	}
}

///////////////////////////////////////////////////////////////

template <class InArray4D, class OutArray4D>
void to_real_array(basis::fourier_space const & fourier_basis, basis::real_space const & real_basis, InArray4D const & array_fs, OutArray4D && array_rs, bool normalize) {

	CALI_CXX_MARK_FUNCTION;

	namespace multi = boost::multi;
#ifdef ENABLE_GPU
	namespace fft = multi::fft;
#else
	namespace fft = multi::fftw;
#endif
	
	assert(get<0>(sizes(array_rs)) == real_basis.local_sizes()[0]);
	assert(get<1>(sizes(array_rs)) == real_basis.local_sizes()[1]);
	assert(get<2>(sizes(array_rs)) == real_basis.local_sizes()[2]);

	assert(get<0>(sizes(array_fs)) == fourier_basis.local_sizes()[0]);
	assert(get<1>(sizes(array_fs)) == fourier_basis.local_sizes()[1]);
	assert(get<2>(sizes(array_fs)) == fourier_basis.local_sizes()[2]);

	assert(get<3>(sizes(array_rs)) == get<3>(sizes(array_fs)));
	
	if(not real_basis.part().parallel()) {
		CALI_CXX_MARK_SCOPE("fft_backward_3d");

		assert(not fourier_basis.reverse_order());
		
		assert(extensions(array_rs) == extensions(array_fs));
		fft::dft_backward({true, true, true, false}, array_fs, array_rs);
		gpu::sync();

	} else {

		assert(fourier_basis.reverse_order());
		
		auto & partx = fourier_basis.cubic_part(2);
		auto & party = real_basis.cubic_part(1);

		gpu::array<complex, 4> tmp2(extensions(array_fs));
		
		{
			CALI_CXX_MARK_SCOPE("fft_backward_2d");
			
			fft::dft_backward({true, true, false, false}, array_fs, tmp2);
			gpu::sync();
		}

		parallel::transpose_backward(fourier_basis.comm(), partx, party, tmp2);

		assert(get<0>(sizes(tmp2)) == real_basis.cubic_part(0).local_size());
		assert(get<1>(sizes(tmp2)) == real_basis.cubic_part(1).local_size());
		assert(get<2>(sizes(tmp2)) == real_basis.cubic_part(2).local_size());
		
		{
			CALI_CXX_MARK_SCOPE("fft_backward_1d");

			fft::dft_backward({true, false, false, false}, tmp2, array_rs);
			gpu::sync();
		}

	}

	if(normalize){
		CALI_CXX_MARK_SCOPE("fft_normalize");
		gpu::run(size(array_rs[0][0][0])*real_basis.local_size(), 
						 [ar = begin(array_rs.flatted().flatted().flatted()), factor = 1.0/real_basis.size()] GPU_LAMBDA (auto ip){
							 ar[ip] = factor*ar[ip];
						 });
	}
}

///////////////////////////////////////////////////////////////

template <class FieldSetType>
auto to_fourier(const FieldSetType & phi){

	CALI_CXX_MARK_SCOPE("to_fourier");

	auto fphi = FieldSetType::reciprocal(phi.skeleton());

	assert(phi.local_set_size() == fphi.local_set_size());  
	
	using type = typename FieldSetType::element_type;
	
	if constexpr (not is_vector3<type>::value){
		static_assert(std::is_same<type, complex>::value, "Only implemented for complex");
		
		to_fourier_array(phi.basis(), fphi.basis(), phi.hypercubic(), fphi.hypercubic());

	} else {

		static_assert(std::is_same<typename type::element_type, complex>::value, "Only implemented for complex vector3");
		
		auto &&    fphi_as_scalar = fphi.hypercubic().template reinterpret_array_cast<complex      >(3).rotated().rotated().rotated().flatted().rotated();
		auto const& phi_as_scalar = phi .hypercubic().template reinterpret_array_cast<complex const>(3).rotated().rotated().rotated().flatted().rotated();

		to_fourier_array(phi.basis(), fphi.basis(), phi_as_scalar, fphi_as_scalar);
	}

	// this is disabled since it causes some issues I need to check, XA
	//  zero_outside_sphere(fphi);
	
	return fphi;
}

///////////////////////////////////////////////////////////////

template <class FieldSetType>
auto to_real(const FieldSetType & fphi, bool const normalize = true){

	CALI_CXX_MARK_SCOPE("to_real");

	auto phi = FieldSetType::reciprocal(fphi.skeleton());

	assert(phi.local_set_size() == fphi.local_set_size());  
	
	using type = typename FieldSetType::element_type;
	
	if constexpr (not is_vector3<type>::value){

		static_assert(std::is_same<type, complex>::value, "Only valid for complex");
		
		to_real_array(fphi.basis(), phi.basis(), fphi.hypercubic(), phi.hypercubic(), normalize);

	} else {

		static_assert(std::is_same<typename type::element_type, complex>::value, "Only valid for complex vector3");
		
		auto const& fphi_as_scalar = fphi.hypercubic().template reinterpret_array_cast<complex const>(3).rotated().rotated().rotated().flatted().rotated();
		auto &&     phi_as_scalar  = phi .hypercubic().template reinterpret_array_cast<complex      >(3).rotated().rotated().rotated().flatted().rotated();
		
		to_real_array(fphi.basis(), phi.basis(), fphi_as_scalar, phi_as_scalar, normalize);

	}
	
	return phi;
}

///////////////////////////////////////////////////////////////

}
}
}
#endif

#ifdef INQ_OPERATIONS_TRANSFORM_UNIT_TEST
#undef INQ_OPERATIONS_TRANSFORM_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	
	parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});
	auto basis_comm = basis::basis_subcomm(cart_comm);

	
	SECTION("zero_outside_sphere"){

		basis::real_space rs(systems::cell::cubic(16.66_b), /*spacing =*/ 0.306320257, basis_comm);
		auto fs = basis::fourier_space(rs);
		basis::field<basis::fourier_space, double> ff(fs);

		ff.fill(1.0);
		auto vol = operations::integral(ff);
		
		CHECK(vol == 0.0293659268_a);
	
		operations::transform::zero_outside_sphere(ff);
		
		CHECK(operations::integral(ff)/vol == 0.5240308896_a /* The limit is M_PI/6.0 for zero spacing */);
	}
	
	SECTION("Gaussian"){

		basis::real_space rs(systems::cell::cubic(16.66_b), /*spacing =*/ 0.306320257, basis_comm);
		auto fs = basis::fourier_space(rs);
		
		basis::field_set<basis::real_space, complex> phi(rs, 7, cart_comm);
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					double r2 = rs.point_op().r2(ix, iy, iz);
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						double sigma = 0.5*(ist + 1);
						phi.hypercubic()[ix][iy][iz][ist] = exp(-sigma*r2);
					}
				}
			}
		}
		
		auto fphi = operations::transform::to_fourier(phi);

		auto fs_vol = fs.size()*fs.volume_element();
		CHECK(fs_vol == 0.0293659268_a);
		
		double diff = 0.0;
		for(int i0 = 0; i0 < fphi.basis().local_sizes()[0]; i0++){
			for(int i1 = 0; i1 < fphi.basis().local_sizes()[1]; i1++){
				for(int i2 = 0; i2 < fphi.basis().local_sizes()[2]; i2++){
					double g2 = fphi.basis().point_op().g2(i0, i1, i2);
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						double sigma = 0.5*(ist + 1);
						diff += fabs(fphi.hypercubic()[i0][i1][i2][ist] - pow(M_PI/sigma, 3.0/2.0)/fs_vol*exp(-0.25*g2/sigma));
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		diff /= fs.size();
		CHECK(diff < 1e-3);

		auto phi2 = operations::transform::to_real(fphi);

		diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						diff += fabs(phi.hypercubic()[ix][iy][iz][ist] - phi2.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}
		
		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});   

		diff /= phi2.hypercubic().num_elements();
		
		CHECK(diff < 1e-15);
		
	}

	SECTION("Gaussian rotated"){

		auto aa = 16.66_b;
			
		basis::real_space rs(systems::cell::lattice({aa/sqrt(2.0), aa/2, aa/2}, {-aa/sqrt(2), aa/2, aa/2}, {0.0_b, -aa/sqrt(2.0), aa/sqrt(2.0)}), /*spacing =*/ 0.306320257, basis_comm);
		auto fs = basis::fourier_space(rs);
		
		basis::field_set<basis::real_space, complex> phi(rs, 7, cart_comm);
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					double r2 = rs.point_op().r2(ix, iy, iz);
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						double sigma = 0.5*(ist + 1);
						phi.hypercubic()[ix][iy][iz][ist] = exp(-sigma*r2);
					}
				}
			}
		}
		
		auto fphi = operations::transform::to_fourier(phi);

		auto fs_vol = fs.size()*fs.volume_element();
		CHECK(fs_vol == 0.0293659268_a);
		
		double diff = 0.0;
		for(int i0 = 0; i0 < fphi.basis().local_sizes()[0]; i0++){
			for(int i1 = 0; i1 < fphi.basis().local_sizes()[1]; i1++){
				for(int i2 = 0; i2 < fphi.basis().local_sizes()[2]; i2++){
					double g2 = fphi.basis().point_op().g2(i0, i1, i2);
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						double sigma = 0.5*(ist + 1);
						diff += fabs(fphi.hypercubic()[i0][i1][i2][ist] - pow(M_PI/sigma, 3.0/2.0)/fs_vol*exp(-0.25*g2/sigma));
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		diff /= fs.size();
		CHECK(diff < 1e-3);

		auto phi2 = operations::transform::to_real(fphi);

		diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						diff += fabs(phi.hypercubic()[ix][iy][iz][ist] - phi2.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}
		
		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});   

		diff /= phi2.hypercubic().num_elements();
		
		CHECK(diff < 1e-15);
		
	}
	
}
#endif

