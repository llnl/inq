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

	assert(get<3>(sizes(array_rs)) == get<3>(sizes(array_fs)));
	
	namespace multi = boost::multi;
#ifdef ENABLE_GPU
	namespace fft = multi::fft;
#else
	namespace fft = multi::fftw;
#endif
	
	if(not real_basis.part().parallel()) {
		CALI_CXX_MARK_SCOPE("fft_forward_3d");

		fft::dft_forward({true, true, true, false}, array_rs, array_fs);
		gpu::sync();

	} else {

		auto & comm = real_basis.comm();
		
		int xblock = real_basis.cubic_part(0).max_local_size();
		int zblock = fourier_basis.cubic_part(2).max_local_size();
		assert(real_basis.local_sizes()[1] == fourier_basis.local_sizes()[1]);

		auto last_dim = get<3>(sizes(array_rs));

		gpu::array<complex, 4> tmp({xblock, real_basis.local_sizes()[1], zblock*comm.size(), last_dim});
		
		gpu::prefetch(tmp);
		
		{
			CALI_CXX_MARK_SCOPE("fft_forward_2d");

			auto const real_x = real_basis.local_sizes();
			fft::dft_forward({false, true, true, false}, array_rs, tmp({0, real_x[0]}, {0, real_x[1]}, {0, real_x[2]}));
			gpu::sync();
		}
		
		CALI_MARK_BEGIN("fft_forward_transpose");   
		gpu::array<complex, 5> buffer({comm.size(), xblock, real_basis.local_sizes()[1], zblock, last_dim});

		for(int i4 = 0; i4 < comm.size(); i4++){
			gpu::run(last_dim, zblock, real_basis.local_sizes()[1], xblock, 
							 [i4,
								buf = begin(buffer),
								rot = begin(tmp.unrotated().unrotated().partitioned(comm.size()).transposed().rotated().transposed().rotated())]
							 GPU_LAMBDA (auto i0, auto i1, auto i2, auto i3){
								 buf[i4][i3][i2][i1][i0] = rot[i4][i3][i2][i1][i0];
							 });
		}
		CALI_MARK_END("fft_forward_transpose");

		assert(get<4>(sizes(buffer)) == last_dim);
		
		tmp.clear();

		{
			CALI_CXX_MARK_SCOPE("fft_forward_alltoall");
			parallel::alltoall(buffer, comm);
		}

		{
			CALI_CXX_MARK_SCOPE("fft_forward_1d");
			
			auto const fourier_x = fourier_basis.local_sizes();
			fft::dft_forward({true, false, false, false}, buffer.flatted()({0, fourier_x[0]}, {0, fourier_x[1]}, {0, fourier_x[2]}), array_fs);
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

	if(not real_basis.part().parallel()) {
		CALI_CXX_MARK_SCOPE("fft_backward_3d");

		fft::dft_backward({true, true, true, false}, array_fs, array_rs);
		gpu::sync();

	} else {

		auto & comm = fourier_basis.comm();

		int xblock = real_basis.cubic_part(0).max_local_size();
		int zblock = fourier_basis.cubic_part(2).max_local_size();

		auto last_dim = get<3>(sizes(array_fs));
		
		gpu::array<complex, 5> buffer({comm.size(), xblock, real_basis.local_sizes()[1], zblock, last_dim});
		gpu::prefetch(buffer);
		
		{
			CALI_CXX_MARK_SCOPE("fft_backward_2d");
			
			fft::dft_backward({true, true, false, false}, array_fs, buffer.flatted()({0, fourier_basis.local_sizes()[0]}, {0, fourier_basis.local_sizes()[1]}, {0, fourier_basis.local_sizes()[2]}));
			gpu::sync();
		}

		{
			CALI_CXX_MARK_SCOPE("fft_backward_alltoall");
			parallel::alltoall(buffer, comm);
		}
		
		gpu::array<complex, 4> tmp({real_basis.local_sizes()[0], real_basis.local_sizes()[1], zblock*comm.size(), last_dim});
		gpu::prefetch(tmp);
		
		{
			CALI_CXX_MARK_SCOPE("fft_backward_transpose");
			for(int i4 = 0; i4 < comm.size(); i4++){
				gpu::run(last_dim, zblock, real_basis.local_sizes()[1], real_basis.local_sizes()[0],
								 [i4, 
									buf = begin(buffer),
									rot = begin(tmp.unrotated().unrotated().partitioned(comm.size()).transposed().rotated().transposed().rotated())]
								 GPU_LAMBDA (auto i0, auto i1, auto i2, auto i3){
									 rot[i4][i3][i2][i1][i0] = buf[i4][i3][i2][i1][i0];
								 });
			}
		}

		{
			CALI_CXX_MARK_SCOPE("fft_backward_1d");

			auto tmpsub = tmp({0, real_basis.local_sizes()[0]}, {0, real_basis.local_sizes()[1]}, {0, real_basis.local_sizes()[2]});

#ifdef ENABLE_GPU
			//when using cuda, do a loop explicitly over the last dimension (n1), otherwise multi makes a loop over the 2 first ones (n0 and n1). And we know that n3 << n0*n1.
			for(auto ii : extension(tmpsub.unrotated())) {
				fft::dft_backward({false, false, true}, tmpsub.unrotated()[ii], array_rs.unrotated()[ii]);
			}
#else
			fft::dft_backward({false, false, true, false}, tmpsub, array_rs);
#endif
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
	basis::real_space rs(systems::cell::cubic(6.66_b), /*spacing =*/ 0.46320257, basis_comm);
	auto fs = basis::fourier_space(rs);
	
	basis::field_set<basis::real_space, complex> phi(rs, 7, cart_comm);

	SECTION("zero_outside_sphere"){
		
		basis::field<basis::fourier_space, double> ff(fs);
		
		ff.fill(1.0);
		auto vol = operations::integral(ff);

		CHECK(vol == 0.1076560845_a);
		
		operations::transform::zero_outside_sphere(ff);
		
		CHECK(operations::integral(ff)/vol == 0.5160349854_a /* The limit is M_PI/6.0 for zero spacing */);
	}
	
	SECTION("Zero"){
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++) phi.hypercubic()[ix][iy][iz][ist] = 0.0;
				}
			}
		}
		
		auto fphi = operations::transform::to_fourier(phi);
		
		double diff = 0.0;
		for(int ix = 0; ix < fphi.basis().local_sizes()[0]; ix++){
			for(int iy = 0; iy < fphi.basis().local_sizes()[1]; iy++){
				for(int iz = 0; iz < fphi.basis().local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						diff += fabs(fphi.hypercubic()[ix][iy][iz][ist]);
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		diff /= fphi.hypercubic().num_elements();

		CHECK(diff < 1e-15);
		
		auto phi2 = operations::transform::to_real(fphi);

		diff = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					for(int ist = 0; ist < phi.set_part().local_size(); ist++)  diff += fabs(phi.hypercubic()[ix][iy][iz][ist]);
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		diff /= phi2.hypercubic().num_elements();

		CHECK(diff < 1e-15);
		
	}
	
	SECTION("Gaussian"){
		
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
		
		double diff = 0.0;
		for(int ix = 0; ix < fphi.basis().local_sizes()[0]; ix++){
			for(int iy = 0; iy < fphi.basis().local_sizes()[1]; iy++){
				for(int iz = 0; iz < fphi.basis().local_sizes()[2]; iz++){
					double g2 = fphi.basis().point_op().g2(ix, iy, iz);
					for(int ist = 0; ist < phi.set_part().local_size(); ist++){
						double sigma = 0.5*(ist + 1);
						diff += fabs(fphi.hypercubic()[ix][iy][iz][ist] - pow(M_PI/sigma, 3.0/2.0)*exp(-0.25*g2/sigma));
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		
		diff /= fphi.hypercubic().num_elements();

		//not sure what is wrong here
		std::cout << "DIFF1 " << diff << std::endl;

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

