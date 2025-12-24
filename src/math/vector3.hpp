/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__MATH__VECTOR3
#define INQ__MATH__VECTOR3

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gpu/run.hpp>
#include <math/complex.hpp>
#include <parallel/communicator.hpp>

#include <inq_config.h>

#include<array>

namespace inq {

template <class Type, class Space> class vector3;
template<class> struct is_vector3 : std::false_type{};
template<class Type, class Space> struct is_vector3<vector3<Type, Space>> : std::true_type {};

struct cartesian {
  using DualSpace = cartesian;
};

struct covariant;

struct contravariant {
  using DualSpace = covariant;
};

struct covariant {
  using DualSpace = contravariant;
};

template <class Type, class Space = cartesian>
class vector3 {

	Type vec_[3];
	
public:

	using element_type = Type;

	vector3() = default;

	explicit GPU_FUNCTION vector3(Type const & scal){
		vec_[0] = scal;
		vec_[1] = scal;
		vec_[2] = scal;
	}
		
	GPU_FUNCTION vector3(Type const & v0, Type const & v1, Type const & v2){
		vec_[0] = v0;
		vec_[1] = v1;
		vec_[2] = v2;
	}
	
	template<class OtherType, class = decltype(Type{std::declval<OtherType const&>()})>
	vector3(OtherType const & v0, OtherType const & v1, OtherType const & v2) : vector3(Type{v0}, Type{v1}, Type{v2}) {}

	GPU_FUNCTION vector3(std::initializer_list<Type> const & list){
		vec_[0] = list.begin()[0];
		vec_[1] = list.begin()[1];
		vec_[2] = list.begin()[2];
	}

	explicit GPU_FUNCTION vector3(Type const * const arr){
		vec_[0] = arr[0];
		vec_[1] = arr[1];
		vec_[2] = arr[2];
	}
		
	GPU_FUNCTION vector3(std::array<Type, 3> const & arr){
		vec_[0] = arr[0];
		vec_[1] = arr[1];
		vec_[2] = arr[2];
	}
		
	GPU_FUNCTION auto & operator[](int ii){
		return vec_[ii];
	}

	GPU_FUNCTION auto & operator[](int ii) const {
		return vec_[ii];
	}

	auto data() {
		return vec_;
	}

	auto data() const {
		return vec_;
	}

	constexpr auto size() const {
		return 3;
	}

	constexpr auto empty() const {
		return false;
	}

	auto begin() const {
		return vec_;
	}

	auto end() const {
		return vec_ + size();
	}

	auto begin() {
		return vec_;
	}

	auto end() {
		return vec_ + size();
	}
	
	//TRANSFORMATION
	template <typename Kernel>
	void transform(Kernel transformation){
		vec_[0] = transformation(vec_[0]);
		vec_[1] = transformation(vec_[1]);
		vec_[2] = transformation(vec_[2]);      
	}
	
	//COMPARISON
	GPU_FUNCTION bool operator==(const vector3 & other) const {
		return vec_[0] == other.vec_[0] && vec_[1] == other.vec_[1] && vec_[2] == other.vec_[2];
	}

	GPU_FUNCTION bool operator!=(const vector3 & other) const {
		return ! (*this == other);
	}

	//ADDITION AND SUBSTRACTION
	GPU_FUNCTION vector3 & operator+=(const vector3 & other){
		vec_[0] += other.vec_[0];
		vec_[1] += other.vec_[1];
		vec_[2] += other.vec_[2];
		return *this;
	}

	GPU_FUNCTION vector3 operator+(const vector3 & other) const {
		vector3 result;
		result.vec_[0] = vec_[0] + other.vec_[0];
		result.vec_[1] = vec_[1] + other.vec_[1];
		result.vec_[2] = vec_[2] + other.vec_[2];
		return result;
	}

	GPU_FUNCTION vector3 & operator-=(const vector3 & other){
		vec_[0] -= other.vec_[0];
		vec_[1] -= other.vec_[1];
		vec_[2] -= other.vec_[2];
		return *this;
	}

	GPU_FUNCTION vector3 operator-(const vector3 & other) const {
		vector3 result;
		result.vec_[0] = vec_[0] - other.vec_[0];
		result.vec_[1] = vec_[1] - other.vec_[1];
		result.vec_[2] = vec_[2] - other.vec_[2];
		return result;
	}
		
	// MULTIPLICATION
		
	//element-wise multiplication and division
	template <class TypeB>
	friend GPU_FUNCTION vector3<decltype(Type()*TypeB()), Space> operator*(const vector3 & vv1, const vector3<TypeB, Space> & vv2){
		return {vv1[0]*vv2[0], vv1[1]*vv2[1], vv1[2]*vv2[2]};
	}

	template <class TypeB>
	friend GPU_FUNCTION vector3<decltype(Type()/TypeB()), Space> operator/(const vector3 & vv1, const vector3<TypeB, Space> & vv2){
		return {vv1[0]/vv2[0], vv1[1]/vv2[1], vv1[2]/vv2[2]};
	}

	//scalar multiplication and division
	template<class TypeA, class=std::enable_if_t<not is_vector3<TypeA>{}>>
	friend GPU_FUNCTION vector3<decltype(TypeA()*Type()), Space> operator*(TypeA const& scalar, vector3 const& vv){
		return {scalar*vv[0], scalar*vv[1], scalar*vv[2]};
	}
		
	template<class TypeB, class=std::enable_if_t<not is_vector3<TypeB>{}> >
	friend GPU_FUNCTION vector3<decltype(Type()*TypeB()), Space> operator*(vector3 const& vv, TypeB const& scalar){
		return {vv[0]*scalar, vv[1]*scalar, vv[2]*scalar};
	}
		
	template <class TypeB, class=std::enable_if_t<not is_vector3<TypeB>{}> >
	friend GPU_FUNCTION vector3<decltype(Type()/TypeB()), Space> operator/(const vector3 & vv, const TypeB & scalar){
		return {vv[0]/scalar, vv[1]/scalar, vv[2]/scalar};
	}
		
	friend GPU_FUNCTION vector3 operator-(vector3 const & vv){
		return -1*vv;
	}

	GPU_FUNCTION vector3 & operator*=(Type const & factor){
		vec_[0] *= factor;
		vec_[1] *= factor;
		vec_[2] *= factor;
		return *this;
	}

	GPU_FUNCTION vector3 & operator/=(Type const & factor){
		vec_[0] /= factor;
		vec_[1] /= factor;
		vec_[2] /= factor;
		return *this;
	}
	
	// ELEMENTWISE OPERATIONS
	template <class Function>
	friend GPU_FUNCTION auto elementwise(Function const & func, vector3 const & vv) -> vector3<decltype(func(Type{})), Space> {
		return {func(vv[0]), func(vv[1]), func(vv[2])};
	}

	friend GPU_FUNCTION auto conj(vector3 const & vv){
		return vector3{conj(vv[0]), conj(vv[1]), conj(vv[2])};
	}

	friend GPU_FUNCTION auto real(vector3 const & vv){
		return vector3<double, Space>{real(vv[0]), real(vv[1]), real(vv[2])};
	}
		
	friend GPU_FUNCTION auto imag(vector3 const & vv){
		return elementwise([] (auto x) { return imag(x);}, vv);
	}

	// VECTORIAL PRODUCTS
		
	//internal product
	template <class OtherType>
	friend GPU_FUNCTION auto dot(vector3 const & vv1, vector3<OtherType, typename Space::DualSpace> const & vv2) {
		return conj(vv1[0])*vv2[0] + conj(vv1[1])*vv2[1] + conj(vv1[2])*vv2[2];         
	}

	template <class OtherType>
	GPU_FUNCTION auto dot(vector3<OtherType, typename Space::DualSpace> const & vv2) const {
		return conj(vec_[0])*vv2[0] + conj(vec_[1])*vv2[1] + conj(vec_[2])*vv2[2];          
	}
	
	//cross product
	friend GPU_FUNCTION auto cross(vector3 const & vv1, vector3 const & vv2) {
		return vector3(vv1[1]*vv2[2] - vv1[2]*vv2[1], vv1[2]*vv2[0] - vv1[0]*vv2[2], vv1[0]*vv2[1] - vv1[1]*vv2[0]);
	}

	//norm
	GPU_FUNCTION auto norm() const {
		return real(this->dot(*this));
	}
	
	friend GPU_FUNCTION auto norm(vector3 const & vv) {
		return real(vv.dot(vv));
	}

	friend GPU_FUNCTION auto length(vector3 const & vv) {
		return sqrt(real(vv.dot(vv)));
	}
		
	GPU_FUNCTION auto length() const{
		return sqrt(real(this->dot(*this)));
	}

	friend GPU_FUNCTION auto product(vector3 const & vv) {
		return vv[0]*vv[1]*vv[2];
	}

	friend GPU_FUNCTION auto fabs(vector3 const & vv){
		return fabs(vv[0]) + fabs(vv[1]) + fabs(vv[2]);
	}
	
	// INPUT OUTPUT
	friend std::ostream& operator <<(std::ostream & out, vector3 const & vv){
		out << vv.vec_[0] << '\t' << vv.vec_[1] << '\t' << vv.vec_[2];
		return out;
	}

	friend std::istream& operator >>(std::istream & in, vector3 & vv){
		in >> vv.vec_[0] >> vv.vec_[1] >> vv.vec_[2] ;
		return in;
	}

	// CAST
	template <typename NewType>
	explicit GPU_FUNCTION operator vector3<NewType, Space>() const{
		return {static_cast<NewType>(vec_[0]), static_cast<NewType>(vec_[1]), static_cast<NewType>(vec_[2])};
	}

	explicit operator std::vector<Type>() const {
		std::vector<Type> stdvec(3);
		stdvec[0] = vec_[0];
		stdvec[1] = vec_[1];
		stdvec[2] = vec_[2];
		return stdvec;
	}
	
};

template<std::size_t Index, class T>
decltype(auto) get(inq::vector3<T, inq::cartesian>& self) {
	static_assert(Index >=0 and Index < 3);
	return self[Index];
}

template<std::size_t Index, class T>
decltype(auto) get(inq::vector3<T, inq::cartesian>&& self) {
	static_assert(Index >=0 and Index < 3);
	return std::move(self)[Index];
}

template<std::size_t Index, class T>
decltype(auto) get(inq::vector3<T, inq::cartesian> const& self) {
	static_assert(Index >=0 and Index < 3);
	return self[Index];
}

template<std::size_t Index, class T>
decltype(auto) get(inq::vector3<T, inq::cartesian> const&& self) {
	static_assert(Index >=0 and Index < 3);
	return self[Index];
}

}

namespace std {
	template<class T>
	struct tuple_size<inq::vector3<T, inq::cartesian> > : integral_constant<size_t, 3> {};

	template<std::size_t N, class T>
	struct tuple_element<N, inq::vector3<T, inq::cartesian> > {using type = T;};
}

template<typename T, typename Space> struct boost::mpi3::datatype<inq::vector3<T, Space>> : boost::mpi3::struct_<T, T, T>{};

namespace std{
template<class T> // workaround for types that recursively have a TDC workaround
struct is_trivially_default_constructible<inq::vector3<T>> : 
		is_trivially_default_constructible<T>{};
}

#endif

#ifdef INQ_MATH_VECTOR3_UNIT_TEST
#undef INQ_MATH_VECTOR3_UNIT_TEST

#include <catch2/catch_all.hpp>

#include <iostream>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Default initialization"){
		vector3<int> vv; (void)vv;
	}

	SECTION("Zero by default constructor"){
		vector3<int> vv{};
		
		CHECK(vv[0] == 0.0);
		CHECK(vv[1] == 0.0);
		CHECK(vv[2] == 0.0);
	}

	SECTION("Scalar"){
		vector3<double> vv(-45.677);
		
		CHECK(vv[0] == -45.677);
		CHECK(vv[1] == -45.677);
		CHECK(vv[2] == -45.677);
	}
		
	SECTION("Initializer list"){
		vector3<int> vv({10, 20, 30});
		
		CHECK(vv[0] == 10);
		CHECK(vv[1] == 20);
		CHECK(vv[2] == 30);
	}

	SECTION("Initialization by element"){
		vector3<int> vv{1000, 2000, 3000};
		
		CHECK(vv[0] == 1000);
		CHECK(vv[1] == 2000);
		CHECK(vv[2] == 3000);
	}

	SECTION("Initialization from std::array"){

		std::array<int, 3> arr;

		arr[0] = 500;
		arr[1] = 600;
		arr[2] = 700;
		
		vector3<int> vv(arr);
		
		CHECK(vv[0] == 500);
		CHECK(vv[1] == 600);
		CHECK(vv[2] == 700);
	}
	
	SECTION("Copy, assignment and comparison"){

		vector3<double> vv({0.1, 0.2, 0.3});
		vector3<double> vv2(vv);
		
		CHECK(vv2[0] == 0.1);
		CHECK(vv2[1] == 0.2);
		CHECK(vv2[2] == 0.3);

		vector3<double> vv3;

		vv3 = vv;

		CHECK(vv3[0] == 0.1);
		CHECK(vv3[1] == 0.2);
		CHECK(vv3[2] == 0.3);
		
		CHECK(vv == vv2);
		CHECK(not (vv != vv2));
		
		vector3<double> vv4({20.1, 0.32, 0.53});
		
		CHECK(vv4 != vv2);

		CHECK(vv4 == vector3<double>{20.1, 0.32, 0.53});
		
	}
		
	SECTION("Addition and substraction"){
		
		vector3<double> vv1({10.0, 5.0, -3.4});
		vector3<double> vv2({1.0, -7.8, 5.6});

		CHECK((vv1 + vv2)[0] == 11.0_a);
		CHECK((vv1 + vv2)[1] == -2.8_a);
		CHECK((vv1 + vv2)[2] ==  2.2_a);

		CHECK((vv1 - vv2)[0] ==  9.0_a);
		CHECK((vv1 - vv2)[1] == 12.8_a);
		CHECK((vv1 - vv2)[2] == -9.0_a);
		
		vector3<double> vv3 = vv1;

		vv3 += vv2;

		CHECK(vv3[0] == 11.0_a);
		CHECK(vv3[1] == -2.8_a);
		CHECK(vv3[2] ==  2.2_a);

		vv3 -= vv2;
		CHECK(vv3 == vv1);

	}
		
	SECTION("Multiplication"){
		
		vector3<double> vv1({10.0, 5.0, -3.4});
		vector3<int> vv2({12, -3, 4});

		auto vv3 = vv1*vv2;
		
		CHECK(vv3[0] == 120.0_a);
		CHECK(vv3[1] == -15.0_a);
		CHECK(vv3[2] == -13.6_a);

		CHECK(product(vv3) == Approx(120.0*15.0*13.6));
		
		auto zvv = complex(0.0, 1.0)*vv1;

		CHECK(real(zvv[0]) ==  (0.0_a).margin(1e-12));
		CHECK(imag(zvv[0]) ==  10.0_a);
		CHECK(real(zvv[1]) ==  (0.0_a).margin(1e-12));
		CHECK(imag(zvv[1]) ==   5.0_a);
		CHECK(real(zvv[2]) ==  (0.0_a).margin(1e-12));
		CHECK(imag(zvv[2]) ==  -3.4_a);
		
		auto zvv2 = zvv/complex(0.0, -1.0);
		zvv /= complex(0.0, -1.0);
		
		CHECK(real(zvv[0]) == -10.0_a);
		CHECK(imag(zvv[0]) ==  (0.0_a).margin(1e-12));
		CHECK(real(zvv[1]) ==  -5.0_a);
		CHECK(imag(zvv[1]) ==  (0.0_a).margin(1e-12));
		CHECK(real(zvv[2]) ==   3.4_a);
		CHECK(imag(zvv[2]) ==  (0.0_a).margin(1e-12));
		
		CHECK(zvv == zvv2);
		
	}

	SECTION("Vector operations"){
		vector3<double> dv(3.0, -1.1, 0.1);

		CHECK( dot(dv, dv) == norm(dv));        
		CHECK(fabs(dv) == 3.0 + 1.1 + 0.1);
		
		vector3<complex> vv1({complex(0.0, 2.0), complex(0.2, -1.1), complex(0.1, 0.1)});
		vector3<complex> vv2({complex(-4.55, 9.0), complex(-0.535, -33.3), complex(2.35, -0.4)});
		
		CHECK(dot(vv1, vv2) == conj(dot(vv2, vv1)));
		CHECK(dot(vv1, vv2) == vv1.dot(vv2));
		
		CHECK(real(dot(vv1, vv2)) == 54.7180_a);
		CHECK(imag(dot(vv1, vv2)) == 1.5765_a);

		CHECK(norm(vv1) == 5.2700_a);
		CHECK(vv1.norm() == 5.2700_a);
		CHECK(norm(vv1) == Approx(real(dot(vv1, vv1))));
		CHECK(imag(dot(vv1, vv1)) == (0.0_a).margin(1e-12));
		
		CHECK(norm(vv1) == Approx(4.0 + 0.04 + 1.21 + 0.01 + 0.01));
		CHECK(length(vv1) == Approx(sqrt(4.0 + 0.04 + 1.21 + 0.01 + 0.01)));
		
		CHECK(real(vv1)[0] ==  (0.0_a).margin(1e-12));
		CHECK(imag(vv1)[0] ==  2.0_a);
		CHECK(real(vv1)[1] ==  0.2_a);
		CHECK(imag(vv1)[1] == -1.1_a);
		
		SECTION("contravariant and covariant"){

		vector3<double, contravariant> vec(2.0, 3.0, 5.0);
		vector3<double, covariant> covec(4.1, 0.76, 2.4);

		CHECK(dot(covec, vec) == 22.48_a);
		
		}
		
		CHECK(real(vv1)[2] ==  0.1_a);
		CHECK(imag(vv1)[2] ==  0.1_a);

	}
	
	SECTION("Elementwise Multiplication same type"){

		vector3<double> vv1 = {10.0,  5.0, -3.4};
		vector3<double> vv2 = {12.0, -3.0,  4.0};

		CHECK( (vv1*vv2)[0] == vv1[0]*vv2[0] );
		CHECK( (2.*vv1)[0] == 2.*vv1[0] );
		CHECK( 2.*vv1 == vv1*2. );
	}
	
	SECTION("Elementwise Multiplication cross type"){
		
		vector3<double> vv1 = {10.0,  5.0, -3.4};
		vector3<complex> vv2 = {12.0, -3.0,  4.0};
		
		CHECK( (vv1*vv2)[0] == vv1[0]*vv2[0] );
		CHECK( (complex{2.}*vv1)[0] == complex{2.}*vv1[0] );
		
	}

	SECTION("Old vector3<double> tests"){
	
		vector3<double> x1{1.0, 2.0, 3.0};
		vector3<double> x2{0.1, 0.2, 0.3};

		CHECK(x1[0] == 1.0_a);
		CHECK(x1[1] == 2.0_a);
		CHECK(x1[2] == 3.0_a);

		CHECK(norm(x1) == 14.0_a);
		CHECK(length(x1) == 3.7416573868_a);
	
		CHECK((2.4*x1)[0] == 2.4_a);
		CHECK((2.4*x1)[1] == 4.8_a);
		CHECK((2.4*x1)[2] == 7.2_a);

		CHECK((x1/0.4166666666667)[0] == 2.4_a);
		CHECK((x1/0.4166666666667)[1] == 4.8_a);
		CHECK((x1/0.4166666666667)[2] == 7.2_a);

		CHECK(-x1 == -1.0*x1);
	
		vector3<double> x3 = x1 + x2;

		CHECK(x3[0] == 1.1_a);
		CHECK(x3[1] == 2.2_a);
		CHECK(x3[2] == 3.3_a);

		CHECK((x3/1.1)[0] == 1.0_a);
		CHECK((x3/1.1)[1] == 2.0_a);
		CHECK((x3/1.1)[2] == 3.0_a);

		x3 /= 2.2;

		CHECK(x3[0] == 0.5_a);
		CHECK(x3[1] == 1.0_a);
		CHECK(x3[2] == 1.5_a);
	
		x3 = x1 - x2;

		CHECK(x3[0] == 0.9_a);
		CHECK(x3[1] == 1.8_a);
		CHECK(x3[2] == 2.7_a);
	
		CHECK(dot(x1, x2) == 1.4_a);

		auto crss = cross(x1, vector3<double>{-1.0, -0.5, 3.33});

		CHECK(crss[0] == 8.16_a);
		CHECK(crss[1] == -6.33_a);
		CHECK(crss[2] == 1.5_a);

		vector3<double> scal(6.66);

		CHECK(scal[0] == 6.66_a);
		CHECK(scal[1] == 6.66_a);
		CHECK(scal[2] == 6.66_a);

		double arr[] = {-45.0, 0.2277, 3.1};

		vector3<double> x4(arr);

		CHECK(x4[0] == -45.0_a);
		CHECK(x4[1] == 0.2277_a);
		CHECK(x4[2] == 3.1_a);

		x4 = vector3<double>(-3.3);
	
		CHECK(x4[0] == -3.3_a);
		CHECK(x4[1] == -3.3_a);
		CHECK(x4[2] == -3.3_a);

	}

	SECTION("contravariant and covariant"){

		vector3<double, contravariant> vec(2.0, 3.0, 5.0);
		vector3<double, covariant> covec(4.1, 0.76, 2.4);

		CHECK(dot(covec, vec) == 22.48_a);
		
	}
	
	SECTION("transformation"){

		vector3<double> vec(2.0, -3.0, 5.0);
		vec.transform([](auto xx){ return std::max(xx, 0.0); });

		CHECK(vec[0] == 2.0_a);
		CHECK(vec[1] ==(0.0_a).margin(1e-12));
		CHECK(vec[2] == 5.0_a);
	}

	SECTION("structured bindings"){
		vector3<double> vec{2.0, -3.0, 5.0};
		auto const [x, y, z] = vec;
		CHECK(x ==  2.0_a);
		CHECK(y == -3.0_a);
		CHECK(z ==  5.0_a);
	}

	SECTION("cast"){
		vector3<double> vec{-12.0, -66.0, 77.0};
		auto vecf = static_cast<vector3<float>>(vec);

		CHECK(vecf[0] == Approx(-12.0f));
		CHECK(vecf[1] == Approx(-66.0f));
		CHECK(vecf[2] == Approx( 77.0f));

		auto stdvec = static_cast<std::vector<double>>(vec);

		CHECK(stdvec.size() == 3);
		CHECK(stdvec[0] == Approx(-12.0));
		CHECK(stdvec[1] == Approx(-66.0));
		CHECK(stdvec[2] == Approx( 77.0));
		
	}

}
#endif
