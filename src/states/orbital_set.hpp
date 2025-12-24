/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__STATES__ORBITAL_SET
#define INQ__STATES__ORBITAL_SET

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field_set.hpp>
#include <states/index.hpp>

namespace inq {
namespace states {
	
template<class Basis, class Type>
class orbital_set {

public:
	
	using element_type = Type;
	using basis_type = Basis;
	using kpoint_type = vector3<double, covariant>;

	template <typename BType, typename EType>
	using template_type = orbital_set<BType, EType>;

private:

	basis::field_set<Basis, Type> fields_;
	int spinor_dim_;
	kpoint_type kpoint_;
	int spin_index_;
	parallel::partition spinor_set_part_;
		
public:
	
	orbital_set(Basis const & basis, int const num_vectors, int const spinor_dim, kpoint_type const & kpoint, int spin_index, parallel::cartesian_communicator<2> comm)
		:fields_(basis, spinor_dim*parallel::partition(num_vectors, basis::set_subcomm(comm)), comm),
		 spinor_dim_(spinor_dim),
		 kpoint_(kpoint),
		 spin_index_(spin_index),
		 spinor_set_part_(num_vectors, fields_.set_comm()){
		assert(spinor_dim_ == 1 or spinor_dim_ == 2);
		assert(fields_.local_set_size()%spinor_dim_ == 0);
	}
	
	orbital_set(orbital_set && oldset, parallel::cartesian_communicator<2> new_comm)
	:fields_(std::move(oldset.fields_), new_comm),
		spinor_dim_(oldset.spinor_dim_),
		kpoint_(oldset.kpoint()),
		spin_index_(oldset.spin_index()),
		spinor_set_part_(fields_.set_size()/spinor_dim_, fields_.set_comm()){
		assert(spinor_dim_ == 1 or spinor_dim_ == 2);
		assert(fields_.local_set_size()%spinor_dim_ == 0);
	}
			 
	template <class any_type>
	orbital_set(inq::utils::skeleton_wrapper<orbital_set<Basis, any_type>> const & skeleton)
		:orbital_set(skeleton.base.basis(), skeleton.base.spinor_set_size(), skeleton.base.spinor_dim(), skeleton.base.kpoint(), skeleton.base.spin_index(), skeleton.base.full_comm()){
	}
	
	auto skeleton() const {
		return inq::utils::skeleton_wrapper<orbital_set<Basis, Type>>(*this);
	}
	
	template <class OtherType>
	static auto reciprocal(inq::utils::skeleton_wrapper<orbital_set<basis_type, OtherType>> const & skeleton){
		return orbital_set<typename basis_type::reciprocal_space, element_type>(skeleton.base.basis().reciprocal(), skeleton.base.spinor_set_size(), skeleton.base.spinor_dim(), skeleton.base.kpoint(), skeleton.base.spin_index(), skeleton.base.full_comm());
	}
	
	template <typename ScalarType>
	auto fill(ScalarType const & scalar){
		fields_.fill(scalar);     
	}

	auto shift_domains() {
		fields_.shift_domains();
	}
	
	auto shift_states() {
		fields_.shift_states();
	}
	
	auto & spinor_dim() const {
		return spinor_dim_;
	}

	bool spinors() const {
		return spinor_dim_ - 1;
	}
	
	auto & kpoint() const {
		return kpoint_;
	}
	
	auto & spin_index() const {
			assert(spin_index_ >= 0 and spin_index_ < 2);
			return spin_index_;
	}
	
	auto & set_part() const {
		return fields_.set_part();
	}
	
	auto local_set_size() const {
		return fields_.local_set_size();
	}

	auto local_spinor_set_size() const {
		return fields_.local_set_size()/spinor_dim_;
	}

	auto set_size() const {
		return fields_.set_size();
	}
	
	auto spinor_set_size() const {
		return fields_.set_size()/spinor_dim_;
	}

	auto & basis() const {
		return fields_.basis();
	}
	
	auto & basis() {
		return fields_.basis();
	}
	
	auto & matrix() const {
		return fields_.matrix();
	}
	
	auto & matrix() {
		return fields_.matrix();
	}

	auto spinor_array() const {
		return fields_.matrix().rotated().partitioned(spinor_dim()).unrotated();
	}
	
	auto spinor_array() {
		return fields_.matrix().rotated().partitioned(spinor_dim()).unrotated();
	}
	
	auto spinor_matrix() const {
		return fields_.matrix().flatted().partitioned(basis().local_size()*spinor_dim_);
	}
	
	auto spinor_matrix() {
		return fields_.matrix().flatted().partitioned(basis().local_size()*spinor_dim_);
	}
	
	auto hypercubic() const {
		return fields_.hypercubic();
	}

	auto hypercubic() {
		return fields_.hypercubic();
	}

	auto spinor_hypercubic() const {
		return fields_.hypercubic().rotated().rotated().rotated().partitioned(spinor_dim()).unrotated().unrotated().unrotated();
	}

	auto spinor_hypercubic() {
		return fields_.hypercubic().rotated().rotated().rotated().partitioned(spinor_dim()).unrotated().unrotated().unrotated();
	}
	
	auto & full_comm() const {
		return fields_.full_comm();
	}
	
	auto & set_comm() const {
		return fields_.set_comm();
	}
	
	template <typename CommunicatorType, typename OpType = std::plus<>>
	void all_reduce(CommunicatorType & comm, OpType op = OpType{}){
		fields_.all_reduce(comm, op);
	}

	auto & spinor_set_part() const {
		return spinor_set_part_;
	}

	auto key() const {
		return states::key{kpoint(), spin_index()};
	}
};

}
}
#endif

#ifdef INQ_STATES_ORBITAL_SET_UNIT_TEST
#undef INQ_STATES_ORBITAL_SET_UNIT_TEST

#include <basis/real_space.hpp>

#include <catch2/catch_all.hpp>

#include <parallel/communicator.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){
  
	using namespace inq;
	using namespace inq::magnitude; 
	using namespace Catch::literals;
	
	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	parallel::cartesian_communicator<2> cart_comm(comm, {});

	auto set_comm = basis::set_subcomm(cart_comm);
	auto basis_comm = basis::basis_subcomm(cart_comm);  

	if(cart_comm.size() >= 5) return;

  basis::real_space rs(systems::cell::orthorhombic(10.0_b, 4.0_b, 7.0_b), /*spacing =*/ 0.35124074, basis_comm);

	states::orbital_set<basis::real_space, double> orb(rs, 12, 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);
	CHECK(not orb.spinors());
	
	CHECK(sizes(orb.basis())[0] == 28);
	CHECK(sizes(orb.basis())[1] == 12);
	CHECK(sizes(orb.basis())[2] == 20);

	CHECK(orb.local_set_size() == orb.local_set_size());
	CHECK(orb.set_size() == orb.set_size());

	CHECK(orb.matrix().size() == orb.basis().local_size());
	CHECK(orb.matrix().transposed().size() ==  orb.local_set_size());

	CHECK(orb.hypercubic().rotated().rotated().rotated().size() == orb.local_set_size());

	CHECK(orb.spinor_array().is_compact());
	CHECK(orb.spinor_matrix().is_compact());
	
	states::orbital_set<basis::real_space, double> orbk(rs, 12, 1, {0.4, 0.22, -0.57}, 0, cart_comm);

	CHECK(not orbk.spinors());
	CHECK(sizes(orbk.basis())[0] == 28);
	CHECK(sizes(orbk.basis())[1] == 12);
	CHECK(sizes(orbk.basis())[2] == 20);

	CHECK(orbk.kpoint()[0] == 0.4_a);
	CHECK(orbk.kpoint()[1] == 0.22_a);
	CHECK(orbk.kpoint()[2] == -0.57_a);

	CHECK(orbk.local_set_size() == orbk.local_set_size());
	CHECK(orbk.set_size() == orbk.set_size());

	CHECK(orbk.matrix().size() == orb.basis().local_size());
	CHECK(orbk.matrix().transposed().size() ==  orb.local_set_size());
	
	states::orbital_set<basis::real_space, double> orb_copy(orbk.skeleton());

	CHECK(not orb_copy.spinors());
	CHECK(sizes(orb_copy.basis()) == sizes(orbk.basis()));
	CHECK(orb_copy.kpoint() == orbk.kpoint());
	CHECK(orb_copy.local_set_size() == orbk.local_set_size());
	CHECK(orb_copy.set_size() == orbk.set_size());

	CHECK(orb_copy.matrix().size() == orb_copy.basis().local_size());
	CHECK(orb_copy.matrix().transposed().size() ==  orb_copy.local_set_size());

	states::orbital_set<basis::real_space, double> sporb(rs, 12, 2, {0.4, 0.22, -0.57}, 0, cart_comm);

	CHECK(sporb.spinors());
	CHECK(sizes(sporb.basis())[0] == 28);
	CHECK(sizes(sporb.basis())[1] == 12);
	CHECK(sizes(sporb.basis())[2] == 20);

	CHECK(sporb.kpoint()[0] == 0.4_a);
	CHECK(sporb.kpoint()[1] == 0.22_a);
	CHECK(sporb.kpoint()[2] == -0.57_a);

	CHECK(sporb.set_part().local_size() == sporb.spinor_set_part().local_size()*sporb.spinor_dim());

	CHECK(sporb.spinor_array().is_compact());
	CHECK(sporb.spinor_matrix().is_compact());

	CHECK(sporb.spinor_hypercubic().is_compact());

	if(cart_comm.size() == 1){
		CHECK(sporb.matrix().size() == sporb.basis().local_size());
		CHECK(sporb.matrix().transposed().size() == 24);

		CHECK(get<0>(sizes(sporb.spinor_array())) == sporb.basis().local_size());
		CHECK(get<1>(sizes(sporb.spinor_array())) == 2);
		CHECK(get<2>(sizes(sporb.spinor_array())) == 12);

		CHECK(get<0>(strides(sporb.spinor_array())) == 24);
		CHECK(get<1>(strides(sporb.spinor_array())) == 12);
		
		CHECK(get<0>(sizes(sporb.spinor_matrix())) == sporb.basis().local_size()*2);
		CHECK(get<1>(sizes(sporb.spinor_matrix())) == 12);

		//CHECK THE ORDER IS CORRECT IN THE SPINOR MATRIX
		for(int ip = 0; ip < sporb.basis().local_size(); ip++){
			for(int ii = 0; ii < 12; ii++){
				sporb.spinor_array()[ip][0][ii] =  ii + 1.0 + 100000.0*ip;
				sporb.spinor_array()[ip][1][ii] = -ii - 1.0 + 100000.0*ip;
			}
		}

		for(int ip = 0; ip < sporb.basis().local_size(); ip++){
			for(int ii = 0; ii < 24; ii++) {
				auto sign = (ii < 12)? 1.0:-1.0;
				CHECK(sporb.matrix()[ip][ii] == Approx(sign*(ii%12 + 1.0) + 100000.0*ip));
			}
		}

		CHECK(get<0>(sizes(sporb.spinor_hypercubic())) == 28);
		CHECK(get<1>(sizes(sporb.spinor_hypercubic())) == 12);
		CHECK(get<2>(sizes(sporb.spinor_hypercubic())) == 20);
		CHECK(get<3>(sizes(sporb.spinor_hypercubic())) == 2);
		CHECK(get<4>(sizes(sporb.spinor_hypercubic())) == 12);
		
	}
	
	states::orbital_set<basis::real_space, double> rr(rs, 12, 1, {0.4, 0.22, -0.57}, 0, cart_comm);
	rr.fill(1.0/set_comm.size());
	rr.all_reduce(set_comm);

	for(int ii = 0; ii < rr.basis().local_size(); ii++){
		for(int jj = 0; jj < rr.local_set_size(); jj++){
			CHECK(rr.matrix()[ii][jj] == 1.0_a);
		}
	}
	
}
#endif
