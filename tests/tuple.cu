//
// Original code Copyright (c) Electronic Arts Inc. All rights reserved
// Modifications/Rewrite Copyright (c) 2020 Eyal Rozenberg.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Note: Original code was retrieved from https://github.com/electronicarts/EASTL/ ,
// master branch, on 2020-03-06.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/tuple.hpp>

#include <cuda/api_wrappers.hpp>

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>

//#include "EASTLTest.h"

//EA_DISABLE_VC_WARNING(4623 4625 4413 4510)

namespace test_structs {

struct default_constructible
{
	enum : int { default_value = 0x1EE7C0DE };
	KAT_HD default_constructible() : value(default_value) {}
	int value;
};

namespace op_counts {
__device__ int default_constructions = 0;
__device__ int int_constructions = 0;
__device__ int copy_constructions = 0;
__device__ int move_constructions = 0;
__device__ int copy_assignments = 0;
__device__ int move_assignments = 0;
__device__ int destructions = 0;
}


struct op_counting
{
	KAT_HD op_counting() : value() {
#ifndef __CUDA_ARCH__
		++default_constructions;
#else
		++op_counts::default_constructions;
#endif
	}
	KAT_HD op_counting(int x) : value(x) {
#ifndef __CUDA_ARCH__
		++int_constructions;
#else
		++op_counts::int_constructions;
#endif
	}
	KAT_HD op_counting(const op_counting& x) : value(x.value) {
#ifndef __CUDA_ARCH__
		++copy_constructions;
#else
		++op_counts::copy_constructions;
#endif
}
	KAT_HD op_counting(op_counting&& x) : value(x.value)
	{
#ifndef __CUDA_ARCH__
		++move_constructions;
#else
		++op_counts::move_constructions;
#endif
		x.value = 0;
	}
	KAT_HD op_counting& operator=(const op_counting& x)
	{
		value = x.value;
#ifndef __CUDA_ARCH__
		++copy_assignments;
#else
		++op_counts::copy_assignments;
#endif
		return *this;
	}
	KAT_HD op_counting& operator=(op_counting&& x)
	{
		value = x.value;
		x.value = 0;
#ifndef __CUDA_ARCH__
		++move_assignments;
#else
		++op_counts::move_assignments;
#endif
		return *this;
	}
	KAT_HD ~op_counting() {
#ifndef __CUDA_ARCH__
		++destructions;
#else
		++op_counts::destructions;
#endif
	}

	int value;

	KAT_HD static void reset_counters()
	{
#ifndef __CUDA_ARCH__
		default_constructions = 0;
		int_constructions = 0;
		copy_constructions = 0;
		move_constructions = 0;
		copy_assignments = 0;
		move_assignments = 0;
		destructions = 0;
#else
		op_counts::default_constructions = 0;
		op_counts::int_constructions = 0;
		op_counts::copy_constructions = 0;
		op_counts::move_constructions = 0;
		op_counts::copy_assignments = 0;
		op_counts::move_assignments = 0;
		op_counts::destructions = 0;
#endif
	}

	static int default_constructions;
	static int int_constructions;
	static int copy_constructions;
	static int move_constructions;
	static int copy_assignments;
	static int move_assignments;
	static int destructions;
};

int op_counting::default_constructions = 0;
int op_counting::int_constructions = 0;
int op_counting::copy_constructions = 0;
int op_counting::move_constructions = 0;
int op_counting::copy_assignments = 0;
int op_counting::move_assignments = 0;
int op_counting::destructions = 0;

// move_only_type - useful for verifying containers that may hold, e.g., unique_ptrs to make sure move ops are implemented
struct move_only_type
{
	move_only_type() = delete;
	KAT_HD move_only_type(int val) : value(val) {}
	move_only_type(const move_only_type&) = delete;
	KAT_HD move_only_type(move_only_type&& x) : value(x.value) { x.value = 0; }
	move_only_type& operator=(const move_only_type&) = delete;
	KAT_HD move_only_type& operator=(move_only_type&& x)
	{
		value = x.value;
		x.value = 0;
		return *this;
	}
	KAT_HD bool operator==(const move_only_type& o) const { return value == o.value; }

	int value;
};


} // namespace test_structs

using kat::tuple;
using kat::tuple_size;
using kat::tuple_element_t;
using kat::get;
using kat::make_tuple;
using std::is_same;
using namespace test_structs;



TEST_SUITE("tuple") {

TEST_CASE("static assertions")
{
	using kat::tuple;
	using kat::tuple_size;
	using kat::tuple_element_t;
	using std::is_same;

	static_assert(tuple_size<tuple<int>>::value == 1, "tuple_size<tuple<T>> test failed.");
	static_assert(tuple_size<const tuple<int>>::value == 1, "tuple_size<const tuple<T>> test failed.");
	static_assert(tuple_size<const tuple<const int>>::value == 1, "tuple_size<const tuple<const T>> test failed.");
	static_assert(tuple_size<volatile tuple<int>>::value == 1, "tuple_size<volatile tuple<T>> test failed.");
	static_assert(tuple_size<const volatile tuple<int>>::value == 1, "tuple_size<const volatile tuple<T>> test failed.");
	static_assert(tuple_size<tuple<int, float, bool>>::value == 3, "tuple_size<tuple<T, T, T>> test failed.");

	static_assert(is_same<tuple_element_t<0, tuple<int>>, int>::value, "tuple_element<I, T> test failed.");
	static_assert(is_same<tuple_element_t<1, tuple<float, int>>, int>::value, "tuple_element<I, T> test failed.");
	static_assert(is_same<tuple_element_t<1, tuple<float, const int>>, const int>::value, "tuple_element<I, T> test failed.");
	static_assert(is_same<tuple_element_t<1, tuple<float, volatile int>>, volatile int>::value, "tuple_element<I, T> test failed.");
	static_assert(is_same<tuple_element_t<1, tuple<float, const volatile int>>, const volatile int>::value, "tuple_element<I, T> test failed.");
	static_assert(is_same<tuple_element_t<1, tuple<float, int&>>, int&>::value, "tuple_element<I, T> test failed.");
}

TEST_CASE("get")
{
	tuple<int> single_element(1);
	CHECK( get<0>(single_element) == 1  );
	get<0>(single_element) = 2;
	CHECK( get<0>(single_element) == 2  );
	get<int>(single_element) = 3;
	CHECK( get<int>(single_element) == 3  );

	const tuple<int> const_single_element(3);
	CHECK( get<int>(const_single_element) == 3  );

	tuple<default_constructible> default_constructed;
	CHECK( get<0>(default_constructed).value == default_constructible::default_value  );
}

TEST_CASE("method invocation counts")
{
	op_counting::reset_counters();
	{
		tuple<op_counting> an_op_counter;
		CHECK_UNARY(
			(op_counting::default_constructions == 1 &&
			get<0>(an_op_counter).value == 0) );
		get<0>(an_op_counter).value = 1;
		tuple<op_counting> another_op_counter(an_op_counter);
		CHECK( true == (
			op_counting::default_constructions == 1 &&
			op_counting::copy_constructions == 1 &&
			get<0>(another_op_counter).value == 1 )	);
		get<0>(an_op_counter).value = 2;
		another_op_counter = an_op_counter;
		CHECK_UNARY(
			op_counting::default_constructions == 1 &&
			op_counting::copy_constructions == 1 &&
			op_counting::copy_assignments == 1 &&
			get<0>(another_op_counter).value == 2 );

		op_counting::reset_counters();

		tuple<op_counting> yet_another_op_counter(op_counting(5));
		CHECK_UNARY( (
			op_counting::move_constructions == 1 && op_counting::default_constructions == 0 &&
			op_counting::copy_constructions == 0 && get<0>(yet_another_op_counter).value == 5 ) );
	}

	CHECK( op_counting::destructions == 4  );
}

TEST_CASE("get")
{
	// Test constructor
	tuple<int, float, bool> a_tuple(1, 1.0f, true);
	CHECK( get<0>(a_tuple) == 1  );
	CHECK( get<1>(a_tuple) == 1.0f  );
	CHECK( get<2>(a_tuple) == true  );
	CHECK( get<int>(a_tuple) == 1  );
	CHECK( get<float>(a_tuple) == 1.0f  );
	CHECK( get<bool>(a_tuple) == true  );

	get<1>(a_tuple) = 2.0f;
	CHECK( get<1>(a_tuple) == 2.0f );

	// Test copy constructor
	tuple<int, float, bool> another_tuple(a_tuple);
	CHECK_UNARY( get<0>(another_tuple) == 1 && get<1>(another_tuple) == 2.0f && get<2>(another_tuple) == true );

	// Test copy assignment
	tuple<int, float, bool> yet_another_tuple(2, 3.0f, true);
	CHECK_UNARY( get<0>(yet_another_tuple) == 2 && get<1>(yet_another_tuple) == 3.0f &&
				  get<2>(yet_another_tuple) == true);
	yet_another_tuple = another_tuple;
	CHECK_UNARY( get<0>(yet_another_tuple) == 1 && get<1>(yet_another_tuple) == 2.0f &&
				  get<2>(yet_another_tuple) == true);

	// Test converting 'copy' constructor (from a tuple of different type whose members are each convertible)
	tuple<double, double, bool> a_different_tuple(a_tuple);
	CHECK_UNARY( get<0>(a_different_tuple) == 1.0 && get<1>(a_different_tuple) == 2.0 &&
				  get<2>(a_different_tuple) == true);

	// Test converting assignment operator (from a tuple of different type whose members are each convertible)
	tuple<double, double, bool> another_different_tuple;
	CHECK_UNARY( get<0>(another_different_tuple) == 0.0 && get<1>(another_different_tuple) == 0.0 &&
				  get<2>(another_different_tuple) == false);
	another_different_tuple = another_tuple;
	CHECK_UNARY( get<0>(another_different_tuple) == 1.0 && get<1>(another_different_tuple) == 2.0 &&
				  get<2>(another_different_tuple) == true);

	// Test default initialization (built in types should be value initialized rather than default initialized)
	tuple<int, float, bool> a_default_initialized_tuple;
	CHECK_UNARY( get<0>(a_default_initialized_tuple) == 0 && get<1>(a_default_initialized_tuple) == 0.0f &&
				  get<2>(a_default_initialized_tuple) == false);
}

TEST_CASE("more typed get")
{
	// Test some other cases with typed-getter
	tuple<double, double, bool> a_tuple_with_repeated_type(1.0f, 2.0f, true);
	CHECK( get<bool>(a_tuple_with_repeated_type) == true );

	tuple<double, bool, double> another_tuple_with_repeated_type(1.0f, true, 2.0f);
	CHECK( get<bool>(another_tuple_with_repeated_type) == true );

	tuple<bool, double, double> yet_another_tupleWithRepeatedType(true, 1.0f, 2.0f);
	CHECK( get<bool>(another_tuple_with_repeated_type) == true );

	struct one_float { float val; };
	struct second_float { float val; };
	tuple<one_float, second_float> a_tuple_of_structs({ 1.0f }, { 2.0f } );
	CHECK( get<one_float>(a_tuple_of_structs).val == 1.0f );
	CHECK( get<second_float>(a_tuple_of_structs).val == 2.0f );

	const tuple<double, double, bool> aConstTuple(a_tuple_with_repeated_type);
	const bool& constRef = get<bool>(aConstTuple);
	CHECK( constRef == true );

	const bool&& constRval = get<bool>(kat::move(a_tuple_with_repeated_type));
	CHECK( constRval == true );
}

TEST_CASE("more tuple methods")
{
	tuple<int, float> a_tuple_with_default_init(1, {});

	// tuple construction from pair

	std::pair<int, float> a_pair(1, 2.0f);
	tuple<int, float> a_tuple(a_pair);
	CHECK_UNARY( get<0>(a_tuple) == 1 && get<1>(a_tuple) == 2.0f );
	tuple<double, double> another_tuple(a_pair);
	CHECK_UNARY( get<0>(another_tuple) == 1.0 && get<1>(another_tuple) == 2.0 );
	another_tuple = std::make_pair(2, 3);
	CHECK_UNARY( get<0>(another_tuple) == 2.0 && get<1>(another_tuple) == 3.0 );

	// operators: ==, !=, <
	another_tuple = a_tuple;
	CHECK( a_tuple == another_tuple );
	CHECK_UNARY( !(a_tuple < another_tuple) && !(another_tuple < a_tuple) );
	tuple<double, double> a_default_init_tuple;
	CHECK( a_tuple != a_default_init_tuple );
	CHECK( a_default_init_tuple < a_tuple );

	tuple<int, int, int> a_lesser_tuple(1, 2, 3);
	tuple<int, int, int> a_greater_tuple(1, 2, 4);
	CHECK_UNARY( a_lesser_tuple < a_greater_tuple && !(a_greater_tuple < a_lesser_tuple) && a_greater_tuple > a_lesser_tuple &&
				  !(a_lesser_tuple > a_greater_tuple));

// We don't have the library's TestObject here
//	tuple<int, float, TestObject> value_tuple(2, 2.0f, TestObject(2));
//	tuple<int&, float&, TestObject&> refTup(value_tuple);
//	tuple<const int&, const float&, const TestObject&> const_ref_to_tuple(value_tuple);
//
//	CHECK( get<0>(refTup) == get<0>(value_tuple) );
//	CHECK( get<1>(refTup) == get<1>(value_tuple) );
//	CHECK( refTup == value_tuple );
//	CHECK( get<0>(refTup) == get<0>(const_ref_to_tuple) );
//	CHECK( get<1>(refTup) == get<1>(const_ref_to_tuple) );
//	CHECK( const_ref_to_tuple == value_tuple );
//	CHECK( const_ref_to_tuple == refTup );

	// swap
	swap(a_lesser_tuple, a_greater_tuple);
	CHECK_UNARY( get<2>(a_lesser_tuple) == 4 && get<2>(a_greater_tuple) == 3 );
	swap(a_greater_tuple, a_lesser_tuple);
	CHECK( a_lesser_tuple < a_greater_tuple );
}


TEST_CASE("move-only contained type")
{
	static_assert(std::is_constructible<move_only_type, move_only_type>::value, "is_constructible type trait giving confusing answers.");
	static_assert(std::is_constructible<move_only_type, move_only_type&&>::value, "is_constructible type trait giving wrong answers.");
	static_assert(std::is_constructible<move_only_type&&, move_only_type&&>::value, "is_constructible type trait giving bizarre answers.");
	tuple<move_only_type> a_tuple_with_move_only_member(1);
	CHECK( get<0>(a_tuple_with_move_only_member).value == 1 );
	get<0>(a_tuple_with_move_only_member) = move_only_type(2);
	CHECK( get<0>(a_tuple_with_move_only_member).value == 2 );

	tuple<const move_only_type&> a_tuple_with_ref_to_move_only_member(a_tuple_with_move_only_member);
	CHECK( get<0>(a_tuple_with_ref_to_move_only_member).value == 2 );

	tuple<const move_only_type&> aTupleWithConstRefToGetMoveOnly(get<0>(a_tuple_with_move_only_member));
	CHECK( get<0>(aTupleWithConstRefToGetMoveOnly).value == 2 );

	tuple<move_only_type&> a_tuple_with_ref_to_get_move_only(get<0>(a_tuple_with_move_only_member));
	CHECK( get<0>(a_tuple_with_ref_to_get_move_only).value == 2 );
}

TEST_CASE("make_tuple")
{
	auto a_made_tuple = make_tuple(1, 2.0, true);
	CHECK_UNARY( get<0>(a_made_tuple) == 1 && get<1>(a_made_tuple) == 2.0 && get<2>(a_made_tuple) == true );

	// TODO: reference_wrapper implementation needs to be finished to enable this code
	{
		int a = 2;
		float b = 3.0f;
		auto a_made_tuple_2 = make_tuple(kat::ref(a), b);
		get<0>(a_made_tuple_2) = 3;
		get<1>(a_made_tuple_2) = 4.0f;
		CHECK_UNARY( get<0>(a_made_tuple_2) == 3 && get<1>(a_made_tuple_2) == 4.0f && a == 3 && b == 3.0f );
	}
}

TEST_CASE("forward_as_tuple")
{
	auto forward_test = [](tuple<move_only_type&&, move_only_type&&> x) -> tuple<move_only_type, move_only_type>
	{
		return tuple<move_only_type, move_only_type>(move(x));
	};

	tuple<move_only_type, move_only_type> a_movable_tuple(
		forward_test(kat::forward_as_tuple(move_only_type(1), move_only_type(2))));

	CHECK_UNARY( get<0>(a_movable_tuple).value == 1 && get<1>(a_movable_tuple).value == 2 );
}

TEST_CASE("tie")
{
	int a = 0;
	double b = 0.0f;
	static_assert(std::is_assignable<const kat::detail::ignore_t<int>&, int>::value, "ignore_t not assignable");
	static_assert(kat::detail::tuple_assignable<tuple<const kat::detail::ignore_t<int>&>, tuple<int>>::value, "Not assignable");
	kat::tie(a, kat::ignore, b) = kat::make_tuple(1, 3, 5);
	CHECK_UNARY( a == 1 && b == 5.0f );
}

TEST_CASE("tuple_cat")
{
	int a = 0;
	double b = 0.0f;

	auto concatenated_tuple = tuple_cat(make_tuple(1, 2.0f), make_tuple(3.0, true));
	CHECK_UNARY( get<0>(concatenated_tuple) == 1 && get<1>(concatenated_tuple) == 2.0f && get<2>(concatenated_tuple) == 3.0 &&
			get<3>(concatenated_tuple) == true);

	auto concatenated_tuple_2 = tuple_cat(make_tuple(1, 2.0f), make_tuple(3.0, true), make_tuple(5u, '6'));
	CHECK_UNARY( get<0>(concatenated_tuple_2) == 1 && get<1>(concatenated_tuple_2) == 2.0f && get<2>(concatenated_tuple_2) == 3.0 &&
			get<3>(concatenated_tuple_2) == true && get<4>(concatenated_tuple_2) == 5u && get<5>(concatenated_tuple_2) == '6');

	auto a_catted_ref_tuple = tuple_cat(make_tuple(1), kat::tie(a, kat::ignore, b));
	get<1>(a_catted_ref_tuple) = 2;
	CHECK( a == 2 );
}

TEST_CASE("empty tuple")
{
	tuple<> empty_tuple;
	CHECK( tuple_size<decltype(empty_tuple)>::value == 0 );
	empty_tuple = make_tuple();
	auto another_empty_tuple = make_tuple();
	swap(another_empty_tuple, empty_tuple);
}

TEST_CASE("std::tuple compatibility") {
	{
		tuple<> empty_tuple;

		auto empty_std_tuple_1 { static_cast< std::tuple<> >(empty_tuple) };
		auto empty_std_tuple_2 { static_cast< std::tuple<> >(kat::make_tuple()) };
		std::tuple<> empty_std_tuple_3 = empty_tuple;
		// empty_tuple = empty_std_tuple_1;
		CHECK (std::is_same<std::tuple<>,decltype(empty_std_tuple_1)>::value);
		CHECK (kat::detail::tuple_convertible< std::tuple<>, tuple<> >::value);
	}

	{
		tuple<int, float, bool> a_tuple(1, 1.0f, true);

		auto std_tuple_1 { static_cast< std::tuple<int, float, bool> >(a_tuple) };
		auto std_tuple_2 { static_cast< std::tuple<int, float, bool> >(kat::make_tuple(1, 1.0f, true)) };
		std::tuple<int, float, bool> std_tuple_3 = a_tuple;
		// a_tuple = std_tuple_1;
		CHECK (std::is_same<std::tuple<int, float, bool>,decltype(std_tuple_1)>::value);
//		CHECK (kat::detail::tuple_convertible< std::tuple<int, float, bool>, tuple<int, float, bool> >::value);
//		CHECK (kat::tuple_size<std::tuple<int, float, bool>>::value == 3);
//		CHECK (kat::detail::tuple_assignable< tuple<int, float, bool>, std::tuple<int, float, bool> >::value);
//		std::cout
//			<< "tuple_size<typename std::remove_reference<tuple<int, float, bool>>::type>::value = "
//			<< tuple_size<typename std::remove_reference<tuple<int, float, bool>>::type>::value << '\n'
//			<< "tuple_size<std::tuple<int, float, bool>>::value) = "
//			<< tuple_size<std::tuple<int, float, bool>>::value << '\n'
//			<< "kat::detail::make_tuple_types_t<tuple<int, float, bool>> = "
//			<< util::type_name<kat::detail::make_tuple_types_t<tuple<int, float, bool> > >() << '\n'
//			<< "make_tuple_types_t<std::tuple<int, float, bool> > = "
//			<< util::type_name< kat::detail::make_tuple_types_t<std::tuple<int, float, bool> > >() << '\n';
//		CHECK (kat::detail::tuple_assignable< tuple<int, float, bool>, tuple<int, float, bool> >::value);
	}

//	std::tuple<> empty_std_tuple;

//	tuple<> empty_tuple_1 { static_cast< kat::tuple<> >(empty_tuple_1) };
//	tuple<> empty_tuple_2 {  static_cast< kat::tuple<> >(std::make_tuple()) };
//	tuple<> empty_tuple_3 = empty_std_tuple_1;

//	swap(empty_tuple, empty_std_tuple);
//	swap(empty_std_tuple, empty_tuple);

//	{
//		tuple<move_only_type> a_tuple_with_move_only_member(1);
//		auto std_tuple_1 { static_cast< std::tuple<move_only_type> >(a_tuple_with_move_only_member) };
//
//		tuple<const move_only_type&> a_tuple_with_ref_to_move_only_member(a_tuple_with_move_only_member);
//		std::tuple<> std_tuple_2 { static_cast< std::tuple<const move_only_type> >(a_tuple_with_ref_to_move_only_member) };
//
//		tuple<const move_only_type&> aTupleWithConstRefToGetMoveOnly(get<0>(a_tuple_with_move_only_member));
//		std::tuple<const move_only_type&> std_tuple_3 { static_cast< std::tuple<const move_only_type&> >(a_tuple_with_ref_to_move_only_member) };
//
//		tuple<move_only_type&> a_tuple_with_ref_to_get_move_only(get<0>(a_tuple_with_move_only_member));
//		std::tuple<move_only_type&> std_tuple_4 { static_cast< std::tuple<move_only_type&> >(a_tuple_with_ref_to_move_only_member) };
//	}

	{
		// operators: ==, !=, <
		tuple<int, float, bool> a_tuple(1, 1.0f, true);
		std::tuple<int, float, bool> an_std_tuple = a_tuple;
		CHECK( a_tuple == an_std_tuple );
		CHECK_UNARY( !(a_tuple < an_std_tuple) && !(an_std_tuple < a_tuple) );
	}

	{
		tuple<int, int, int> a_lesser_tuple(1, 2, 3);
		tuple<int, int, int> a_greater_tuple(1, 2, 4);
		std::tuple<int, int, int> a_lesser_std_tuple(1, 2, 3);
		std::tuple<int, int, int> a_greater_std_tuple(1, 2, 4);
		CHECK_UNARY(
			a_lesser_tuple < a_greater_std_tuple &&
			!(a_greater_tuple < a_lesser_std_tuple) &&
			a_greater_tuple > a_lesser_std_tuple &&
			!(a_lesser_tuple > a_greater_std_tuple)
			);

		CHECK_UNARY(
			a_lesser_std_tuple < a_greater_tuple &&
			!(a_greater_std_tuple < a_lesser_tuple) &&
			a_greater_std_tuple > a_lesser_tuple &&
			!(a_lesser_std_tuple > a_greater_tuple)
			);
	}
}

// TODO: Enable this when we've introduced compatibility code of kat::tuple
// and std::tuple on the host side. Also, if we get kat::pair, replicate the
// following tests for that class as well.

/*
TEST_CASE("piecewise_construction")
{
	{
		struct local
		{
			local() = default;
			local(int a, int b) : mA(a), mB(b) {}

			int mA = 0;
			int mB = 0;
		};

		auto t = kat::make_tuple(42, 43);

		std::pair<local, local> p(std::piecewise_construct, t, t);

		CHECK( p.first.mA  == 42 );
		CHECK( p.second.mA == 42 );
		CHECK( p.first.mB  == 43 );
		CHECK( p.second.mB == 43 );
	}

	{
		struct local
		{
			local() = default;
			local(int a, int b, int c, int d) : mA(a), mB(b), mC(c), mD(d) {}

			int mA = 0;
			int mB = 0;
			int mC = 0;
			int mD = 0;
		};

		auto t = kat::make_tuple(42, 43, 44, 45);

		std::pair<local, local> p(std::piecewise_construct, t, t);

		CHECK( p.first.mA  == 42 );
		CHECK( p.second.mA == 42 );

		CHECK( p.first.mB  == 43 );
		CHECK( p.second.mB == 43 );

		CHECK( p.first.mC  == 44 );
		CHECK( p.second.mC == 44 );

		CHECK( p.first.mD  == 45 );
		CHECK( p.second.mD == 45 );
	}

	{
		struct local1
		{
			local1() = default;
			local1(int a) : mA(a) {}
			int mA = 0;
		};

		struct local2
		{
			local2() = default;
			local2(char a) : mA(a) {}
			char mA = 0;
		};

		auto t1 = kat::make_tuple(42);
		auto t2 = kat::make_tuple('a');

		std::pair<local1, local2> p(std::piecewise_construct, t1, t2);

		CHECK( p.first.mA  == 42 );
		CHECK( p.second.mA == 'a' );
	}
}
*/

#if __cplusplus >=  201703L

TEST_CASE("apply")
{
	// test with tuples
	{
		{
			auto result = kat::apply([](int i) { return i; }, make_tuple(1));
			CHECK( result == 1 );
		}

		{
			auto result = kat::apply([](int i, int j) { return i + j; }, make_tuple(1, 2));
			CHECK( result == 3 );
		}


		{
			auto result = kat::apply([](int i, int j, int k, int m) { return i + j + k + m; }, make_tuple(1, 2, 3, 4));
			CHECK( result == 10 );
		}
	}

//	// test with pair
//	{
//		auto result = kat::apply([](int i, int j) { return i + j; }, make_pair(1, 2));
//		CHECK( result == 3 );
//	}

	// TODO: Test apply with arrays?
}

TEST_CASE("tuple structured bindings") {
	kat::tuple<int, int, int> t = {1,2,3};
	auto [x,y,z] = t;
	CHECK( x == 1 );
	CHECK( y == 2 );
	CHECK( z == 3 );
}

#endif // __cplusplus >= 201703L

TEST_CASE("tuple_cat") {
	void* empty = nullptr;
	auto t = kat::make_tuple(empty, true);
	auto tc = kat::tuple_cat(kat::make_tuple("asd", 1), t);

	static_assert(std::is_same<decltype(tc), kat::tuple<const char*, int, void*, bool>>::value, "type mismatch");

	CHECK( std::string("asd") == kat::get<0>(tc) );
	CHECK( kat::get<1>(tc) == 1 );
	CHECK( kat::get<2>(tc) == nullptr );
	CHECK( kat::get<3>(tc) == true );
}

} // TEST_SUITE("tuple")

// EA_RESTORE_VC_WARNING()
