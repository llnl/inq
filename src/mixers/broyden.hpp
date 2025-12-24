/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MIXERS__BROYDEN
#define INQ__MIXERS__BROYDEN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/complex.hpp>
#include <math/vector3.hpp>
#include <gpu/array.hpp>
#include <solvers/least_squares.hpp>
#include <mixers/base.hpp>

#include <parallel/communicator.hpp>
#include <parallel/reduce_nrm2.hpp>
#include <mpi3/environment.hpp>
#include <utils/raw_pointer_cast.hpp>

namespace inq {
namespace mixers {

template <class FieldType>
class broyden : public base<FieldType> {

public:

	using element_type = typename FieldType::element_type;

private:
	
	int iter_;
	int max_size_;
	double mix_factor_;
	gpu::array<element_type, 2> dv_;
	gpu::array<element_type, 2> df_;
	gpu::array<element_type, 1> f_old_;
	gpu::array<element_type, 1> vin_old_;
	element_type gamma_;
	int last_pos_;
	
public:
	
	broyden(const int arg_steps, const double arg_mix_factor, const long long dim):
		iter_(0),
		max_size_(arg_steps),
		mix_factor_(arg_mix_factor),
		dv_({max_size_, dim}, NAN),
		df_({max_size_, dim}, NAN),
		f_old_(dim, NAN),
		vin_old_(dim, NAN),
		last_pos_(-1){
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	template <typename Comm, typename Array>
	void broyden_extrapolation(Comm & comm, Array & input_value, int const iter_used, gpu::array<element_type, 1> const & ff){

		CALI_CXX_MARK_SCOPE("broyden_extrapolation");
		
		double const w0 = 0.01;
		double const ww = 5.0;

		if(iter_used == 0){
			gpu::run(input_value.size(),
							 [iv = begin(input_value), ffp = begin(ff),  mix = mix_factor_] GPU_LAMBDA (auto ip){
								 iv[ip] += mix*ffp[ip];
							 });

			return;
		}

		namespace blas = boost::multi::blas;
		
		auto subdf = df_({0, iter_used}, {0, input_value.size()});
		auto beta = +blas::gemm(ww*ww, subdf, blas::H(subdf));
		auto work = +blas::gemv(1.0, subdf, ff({0, input_value.size()})); 
		
		gpu::run(iter_used, [w0, ww, be = begin(beta), dfactor = 1.0/comm.size()] GPU_LAMBDA (auto ii){ be[ii][ii] = dfactor*(w0*w0 + ww*ww); });
		
		if(comm.size() > 1){
			CALI_CXX_MARK_SCOPE("broyden_extrapolation::reduce");
			comm.all_reduce_in_place_n(raw_pointer_cast(beta.data_elements()), beta.num_elements());
			comm.all_reduce_in_place_n(raw_pointer_cast(work.data_elements()), work.num_elements());
		}
		
		solvers::least_squares(beta, work);
		
		gpu::run(input_value.size(), 
						 [iv = begin(input_value),  ffp = begin(ff), ww, wo = begin(work), mix = mix_factor_, df = begin(df_), dv = begin(dv_), iter_used] GPU_LAMBDA (auto ip){
							 iv[ip] += mix*ffp[ip];							 
							 for(int ii = 0; ii < iter_used; ii++){
								 iv[ip] -= ww*ww*wo[ii]*(mix*df[ii][ip] + dv[ii][ip]);
							 }
						 });
			
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
		
	void operator()(FieldType & input_field, FieldType const & output_field){

		CALI_CXX_MARK_SCOPE("broyden_mixing");
		
		auto input_value = input_field.matrix().flatted();
		auto output_value = output_field.matrix().flatted();
		
		assert((typename gpu::array<double, 2>::size_type) input_value.size() == dv_[0].size());
		assert((typename gpu::array<double, 2>::size_type) output_value.size() == dv_[0].size());

		iter_++;

		gpu::array<element_type, 1> ff(input_value.size());

		gpu::run(input_value.size(),
						 [iv = begin(input_value), ov = begin(output_value), ffp = begin(ff)] GPU_LAMBDA (auto ip){
								 ffp[ip] = ov[ip] - iv[ip]; 
						 });
		
		if(iter_ > 1){

			auto pos = (last_pos_ + 1)%max_size_;
				
			df_[pos] = ff;
			dv_[pos] = input_value;

			gpu::run(input_value.size(),
							 [pos, df = begin(df_), f_old = begin(f_old_), dv = begin(dv_), vin_old = begin(vin_old_)] GPU_LAMBDA (auto ip){
								 df[pos][ip] -= f_old[ip];
								 dv[pos][ip] -= vin_old[ip];
							 });

			gamma_ = boost::multi::blas::nrm2(df_[pos]);
			parallel::reduce_nrm2(gamma_, input_field.full_comm());
			gamma_ = std::max(1e-8, gamma_);

			gpu::run(input_value.size(),
							 [pos, df = begin(df_), dv = begin(dv_), invgamma = 1.0/gamma_] GPU_LAMBDA (auto ip){
								 df[pos][ip] *= invgamma;
								 dv[pos][ip] *= invgamma;
							 });

			last_pos_ = pos;
				
		}

		vin_old_ = input_value;
		f_old_ = ff;

		auto iter_used = std::min(iter_ - 1, max_size_);

		broyden_extrapolation(input_field.full_comm(), input_value, iter_used, ff);

	}
	
};
	
}
}
#endif

#ifdef INQ_MIXERS_BROYDEN_UNIT_TEST
#undef INQ_MIXERS_BROYDEN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;

	auto comm = parallel::communicator{boost::mpi3::environment::get_self_instance()};
	
	basis::trivial bas(2, comm);
	
	basis::field_set<basis::trivial, double> vin(bas, 1);
	vin.matrix()[0][0] =    10.0;
	vin.matrix()[1][0] =   -20.0;

	basis::field_set<basis::trivial, double> vout(bas, 1);
	vout.matrix()[0][0] =    0.0;
	vout.matrix()[1][0] =   22.2;

	mixers::broyden<decltype(vin)> mixer(5, 0.5, 2);
	
	mixer(vin, vout);
  
	CHECK(vin.matrix()[0][0] == 5.0_a);
	CHECK(vin.matrix()[1][0] == 1.1_a);

	vout.matrix()[0][0] =   4.0;
	vout.matrix()[1][0] =   5.5;
	
	mixer(vin, vout);
  
	CHECK(vin.matrix()[0][0] == 4.4419411001_a);
	CHECK(vin.matrix()[1][0] == 3.5554591594_a);
	
}
#endif
