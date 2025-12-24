// -*-indent-tabs-mode:t;-*-

#ifndef INQ__BASIS__FIELD
#define INQ__BASIS__FIELD

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gpu/array.hpp>
#include <tinyformat/tinyformat.h>
#include <algorithm>
#include <utils/skeleton_wrapper.hpp>
#include <utils/raw_pointer_cast.hpp>
#include <basis/real_space.hpp>
#include <math/complex.hpp>
#include <parallel/get_remote_points.hpp>

#include <mpi3/environment.hpp>

#include <fstream>

namespace inq {
namespace basis {
	
template<class Basis, typename Type>
class field {
	
public:
	
	using element_type = Type;
	using basis_type = Basis;
	using internal_array_type = gpu::array<element_type, 1>;
	
private:
	
	internal_array_type linear_;
	basis_type basis_;
	
public:
	
		template <typename BType, typename EType>
		using template_type = field<BType, EType>;
		
		field(const basis_type & basis):
			linear_(basis.part().max_local_size()),
			basis_(basis){
			prefetch();
		}

		template <class OtherType>
		field(inq::utils::skeleton_wrapper<field<basis_type, OtherType>> const & skeleton)
			:field(skeleton.base.basis()){
		}

		template <class OtherType>
		static auto reciprocal(inq::utils::skeleton_wrapper<field<basis_type, OtherType>> const & skeleton){
			return field<typename basis_type::reciprocal_space, Type>(skeleton.base.basis().reciprocal());
		}
		
		template<class, class> friend class field;
		template<typename OtherType>
		field(field<basis_type, OtherType> const& o) 
		: linear_(o.linear_), basis_(o.basis_){
			static_assert(std::is_constructible<element_type, Type>{}, "!");
		}

		auto skeleton() const {
			return inq::utils::skeleton_wrapper<field<basis_type, element_type>>(*this);
		}

		field(field && old, parallel::communicator new_comm)
			:field(basis_type(std::move(old.basis_), new_comm)){

			if(new_comm == old.basis().comm()){
				linear_ = std::move(old.linear_);
				return;
			}
			
			gpu::array<int, 1> rem_points(basis().local_size());
			for(long ip = 0; ip < basis().local_size(); ip++) rem_points[ip] = basis().part().local_to_global(ip).value();
			linear() = parallel::get_remote_points(old, rem_points);
		}
		
		explicit field(const field & coeff) = default;      //avoid unadverted copies
		field(field && coeff) = default;
		field & operator=(const field & coeff) = default;
		field & operator=(field && coeff) = default;
		
	template <typename ScalarType>
	void fill(ScalarType const & scalar) {
		CALI_CXX_MARK_SCOPE("fill(field_set)");
		
		gpu::run(basis_.local_size(), [lin = begin(linear_), scalar] GPU_LAMBDA (auto ip) {
			lin[ip] = scalar;
		});
	}
	
		template<typename OtherType>
		field& operator=(field<basis_type, OtherType> const& o){
			static_assert( std::is_assignable<element_type&, OtherType>{}, "!" );
			assert(o.basis_ == basis_);
			linear() = o.linear();
			return *this;
		}

	void shift_domains() {
		if(basis_.comm().size() == 1) return;

		auto next_proc = (basis_.comm().rank() + 1)%basis_.comm().size();
		auto prev_proc = basis_.comm().rank() - 1;
		if(prev_proc == -1) prev_proc = basis_.comm().size() - 1;

		auto tag = basis_.part().rank() - basis_.comm().rank();
		if(tag < 0) tag += basis_.comm().size();
		assert(tag >= 0 and tag < basis_.comm().size());
		
		auto mpi_type = boost::mpi3::detail::basic_datatype<Type>();
		assert(linear_.num_elements() == basis_.part().max_local_size());
		MPI_Sendrecv_replace(data(), basis_.part().max_local_size(), mpi_type, prev_proc, tag, next_proc, tag, basis_.comm().get(), MPI_STATUS_IGNORE);
		basis_.shift();
	}
	
		auto size() const {
			return basis_.size();
		}
		
		const auto & basis() const {
			return basis_;
		}

	auto cubic() const {
		assert(basis_.local_size() > 0);
		return linear().partitioned(basis_.cubic_part(1).local_size()*basis_.cubic_part(0).local_size()).partitioned(basis_.cubic_part(0).local_size());
	}
	
	auto cubic() {
		assert(basis().local_size() > 0);
		return linear().partitioned(basis_.cubic_part(1).local_size()*basis_.cubic_part(0).local_size()).partitioned(basis_.cubic_part(0).local_size());
	}
	
		auto data() {
			return raw_pointer_cast(linear_.data_elements());
		}

		auto data() const {
			return raw_pointer_cast(linear_.data_elements());
		}
	
	auto linear() const {
		return linear_({0, basis_.local_size()});
	}
	
	auto linear() {
		return linear_({0, basis_.local_size()});
	}
	
		// emulate a field_set

		auto hypercubic() const {
			return cubic().template reinterpret_array_cast<Type>(1);
		}
		
		auto hypercubic() {
			return cubic().template reinterpret_array_cast<Type>(1);
		}

		auto matrix() const {
			return linear().template reinterpret_array_cast<Type const>(1);
		}
		
		auto matrix() {
			return linear().template reinterpret_array_cast<Type>(1);
		}
		
		constexpr auto set_size() const {
			return 1;
		}

		constexpr auto local_set_size() const {
			return 1;
		}

		template <int dir = 2>
		friend void print_debug(const field & fld, const std::string & filename){

			std::ofstream file(filename);

			std::array<int, 3> point = {0, 0, 0};

			auto size = get<dir>(sizes(fld.cubic()));
			
			for(int ii = 0; ii < size; ii++){
				auto ip = ii + size/2;
				if(ip >= size) ip -= size;
				point[dir] = ip;
				auto rr = fld.basis().rvector(point);
				tfm::format(file, "%f %e %e\n", rr[dir], inq::real(fld.cubic()[point[0]][point[1]][point[2]]), imag(fld.cubic()[point[0]][point[1]][point[2]]));
			}
		}

		void prefetch() const {
			gpu::prefetch(linear_);
		}
		
		template <typename CommunicatorType, typename OpType = std::plus<>>
		void all_reduce(CommunicatorType & comm, OpType op = OpType{}){
			if(comm.size() < 2) return;
			comm.all_reduce_in_place_n(data(), linear().num_elements(), op);
		}

};

field<basis::real_space, complex> complex_field(field<basis::real_space, double> const & rfield) {
	field<basis::real_space, complex> cfield(rfield.skeleton());        
	
	gpu::run(rfield.basis().part().local_size(),
					 [cp = begin(cfield.linear()), rp = begin(rfield.linear())] GPU_LAMBDA (auto ip){
						 cp[ip] = inq::complex(rp[ip], 0.0);
					 });
	
	return cfield;
}

template <class VectorSpace>
field<basis::real_space, vector3<inq::complex, VectorSpace>> complex_field(field<basis::real_space, vector3<double, VectorSpace>> const & rfield) {
	field<basis::real_space, vector3<inq::complex, VectorSpace>> cfield(rfield.skeleton());
	
	gpu::run(3, rfield.basis().part().local_size(),
					 [cp = begin(cfield.linear()), rp = begin(rfield.linear())] GPU_LAMBDA (auto idir, auto ip){
						 cp[ip][idir] = inq::complex(rp[ip][idir], 0.0);
					 });
	
	return cfield;
}

field<basis::real_space, double> real_field(field<basis::real_space, complex> const & cfield) {
	field<basis::real_space, double> rfield(cfield.skeleton());     
	gpu::run(rfield.basis().local_size(),
					 [rf = begin(rfield.linear()), cf = begin(cfield.linear())] GPU_LAMBDA (auto ip) {
						 rf[ip] = real(cf[ip]);
					 });
	return rfield;
}

template <class VectorSpace>
field<basis::real_space, vector3<double, VectorSpace>> real_field(field<basis::real_space, vector3<complex, VectorSpace>> const & cfield) {
	field<basis::real_space, vector3<double, VectorSpace>> rfield(cfield.skeleton());       
	
	gpu::run(3, cfield.basis().part().local_size(),
					 [rp = begin(rfield.linear()), cp = begin(cfield.linear())] GPU_LAMBDA (auto idir, auto ip){
						 rp[ip][idir] = inq::real(cp[ip][idir]);
					 });
	
	return rfield;
}


}
}
#endif

#ifdef INQ_BASIS_FIELD_UNIT_TEST
#undef INQ_BASIS_FIELD_UNIT_TEST

#include <basis/real_space.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude; 
	using namespace Catch::literals;
	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	basis::real_space rs(systems::cell::orthorhombic(4.0_b, 10.0_b, 7.0_b), /*spacing = */ 0.35124074, comm);

	basis::field<basis::real_space, double> ff(rs);

	basis::field<basis::real_space, complex> ff_complex = complex_field(ff);
	basis::field<basis::real_space, double> ff2 = real_field(ff_complex);

	ff2.fill(0.0);

	CHECK(( sizes(rs) == decltype(sizes(rs)){12, 28, 20} ));

	CHECK(get<0>(sizes(ff.cubic())) == 12);
	
	if(comm.size() == 1) CHECK(ff.linear().size() == 6720);
	if(comm.size() == 2) CHECK(ff.linear().size() == 6720/2);
	if(comm.size() == 3 and comm.rank() != 2) CHECK(ff.linear().size() == 2400);
	if(comm.size() == 3 and comm.rank() == 2) CHECK(ff.linear().size() == 1920);
	if(comm.size() == 4) CHECK(ff.linear().size() == 1680);

	if(comm.size() == 1) CHECK(get<1>(sizes(ff.cubic())) == 28);
	if(comm.size() == 2) CHECK(get<1>(sizes(ff.cubic())) == 14);
	if(comm.size() == 4) CHECK(get<1>(sizes(ff.cubic())) == 7);

	CHECK(get<2>(sizes(ff.cubic())) == 20);

	ff.fill(12.2244);

	for(int ii = 0; ii < rs.part().local_size(); ii++) CHECK(ff.linear()[ii] == 12.2244_a); 

	basis::field<basis::real_space, double> ff_copy(ff.skeleton());

	CHECK(get<0>(sizes(ff_copy.cubic())) == 12);
	CHECK(get<2>(sizes(ff_copy.cubic())) == 20);

	auto zff = complex_field(ff);
	
	static_assert(std::is_same<decltype(zff), basis::field<basis::real_space, complex>>::value, "complex() should return a complex field");
	
	CHECK(get<0>(sizes(zff.cubic())) == 12);
	CHECK(get<2>(sizes(zff.cubic())) == 20);

	auto dff = real_field(zff);

	static_assert(std::is_same<decltype(dff), basis::field<basis::real_space, double>>::value, "real() should return a double field");

	CHECK(get<0>(sizes(dff.cubic())) == 12);
	CHECK(get<2>(sizes(dff.cubic())) == 20);

	CHECK(get<0>(sizes(ff.hypercubic())) == 12);
	CHECK(get<2>(sizes(ff.hypercubic())) == 20);
	CHECK(get<3>(sizes(ff.hypercubic())) == 1);    

	//Make sure the hypercubic array is correctly ordered, so it can be flattened
	auto strd = strides(ff.hypercubic());

	CHECK(get<0>(strd) >= get<1>(strd));
	CHECK(get<1>(strd) >= get<2>(strd));
	CHECK(get<2>(strd) >= get<3>(strd));
	
	basis::field<basis::real_space, double> red(basis::field<basis::real_space, double>(ff), parallel::communicator{boost::mpi3::environment::get_self_instance()});

	CHECK(red.basis().local_size() == red.linear().size());
	
	for(long ip = 0; ip < red.basis().local_size(); ip++){
		parallel::global_index ipg(ip);
		if(ff.basis().part().contains(ip)) CHECK(red.linear()[ip] == ff.linear()[ff.basis().part().global_to_local(ipg)]);
	}
	
	red.fill(1.0/comm.size());
	red.all_reduce(comm);

	for(int ii = 0; ii < red.basis().part().local_size(); ii++){
		CHECK(red.linear()[ii] == 1.0_a);
	}

	SECTION("Shift") {
		basis::field<basis::real_space, complex> fie(rs, comm);

		for(auto ip = 0; ip < fie.basis().local_size(); ip++){
			fie.linear()[ip] = complex{double(fie.basis().part().start() + ip), double(comm.rank())};
		}

		auto part = fie.basis().part();

		for(int ishift = 0; ishift < 2*comm.size(); ishift++) {
			auto shift_rank = (comm.rank() + ishift)%comm.size();

			CHECK(fie.basis().part().rank() == shift_rank);
			CHECK(fie.basis().part().start() == part.start(shift_rank));
			CHECK(fie.basis().part().local_size() == part.local_size(shift_rank));
			
			for(auto ip = 0; ip < fie.basis().local_size(); ip++){
				CHECK(real(fie.linear()[ip]) == fie.basis().part().start() + ip);
				CHECK(imag(fie.linear()[ip]) == shift_rank);
			}
			
			fie.shift_domains();
		}
		
		
	}
}
#endif
