// Copyright (C) 2011-2020 Free Software Foundation, Inc.
// Copyright (C) 2020 Eyal Rozenberg <eyalroz@technion.ac.il>
//
// This file is based on the kat::array test code from the GNU ISO C++
// Library.  It is free software. Thus, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
//
// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation, i.e.:
//
// You may use this file as part of a free software library without
// restriction.  Specifically, if other files instantiate templates or
// use macros or inline functions from this file, or you compile this file
// and link it with other files to produce an executable, this file does
// not by itself cause the resulting executable to be covered by the GNU
// General Public License. This exception does not however invalidate any
// other reasons why the executable file might be covered by the GNU
// General Public License.
//
// A copy of the GNU General Public License and a copy of the GCC Runtime
// Library Exception are available at <http://www.gnu.org/licenses/>.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/containers/array.hpp>
#include <array>
#include <memory>
#include <kat/tuple.hpp>


#include <stdio.h>
#include <doctest.h>
#include <cuda/runtime_api.hpp>

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>


// We can use doctest's CHECK() macro on the GPU. Instead,
// whenever we check some boolean expression, we'll keep
// the raw result and the line number where we checked in one of
// these, and send it back to the host
//
// Note: Not including the function name for now, but
// maybe we should add it. It's not simple, obviously,
// unless we use a plain array and limit the length
struct result_of_check {
	bool result;
	kat::size_t line_number;
};

#define GPU_CHECK(check_expression) \
do { \
	results[check_index++] = result_of_check{ ( check_expression ) , __LINE__ }; \
} while(false);


namespace kernels {

template <typename F, typename... Ts>
__global__ void run_on_gpu(
	F            function,
	Ts...        args
)
{
	function(std::forward<Ts>(args)...);
}


template <typename F>//, typename... Ts>
__global__ void run_simple_test(
	F                             test_function,
	result_of_check*  __restrict  results,
	kat::size_t                   num_checks
//	, Ts...                         args
	)
{
	test_function(results, num_checks);//, std::forward<Ts>(args)...);
}

} // namespace kernels


#if __cplusplus >= 201703L
template<typename T, typename U> struct require_same;
template<typename T> struct require_same<T, T> { using type = void; };

template<typename T, typename U>
  typename require_same<T, U>::type
  check_type(U&) { }
#endif


// TODO: Don't pass the number of checks, have the device function return
// a dynamically-allocated std-vector-like object, and carefull copy it to
// the host side (first its size, then its data after corresponding allocation
// on the host side). Perhaps with thrust device_vector? Or roll my own?
template <typename F>
auto execute_simple_testcase_on_gpu(
	F                                testcase_device_function,
	size_t                           num_checks = 0)
{
	cuda::device_t device { cuda::device::current::get() };
	auto host_side_results { std::vector<result_of_check>(num_checks) };
	if (num_checks == 0) {
		cuda::launch(
			kernels::run_simple_test<F>,
			single_thread_launch_config(),
			testcase_device_function,
			nullptr,
			num_checks
			);
	}
	else {
		auto device_side_results { cuda::memory::device::make_unique<result_of_check[]>(device, num_checks) };
		cuda::memory::device::zero(device_side_results.get(), num_checks * sizeof(result_of_check)); // just to be on the safe side

		cuda::launch(
			kernels::run_simple_test<F>,
			single_thread_launch_config(),
			testcase_device_function,
			device_side_results.get(),
			num_checks
			);

		cuda::memory::copy(host_side_results.data(), device_side_results.get(), sizeof(result_of_check) * num_checks);
	}
	device.synchronize(); // Probably unnecessary, but let's just be on the safe side
	return host_side_results;
}

void check_results(
	std::string             test_or_testcase_name,
	const result_of_check*  results,
	kat::size_t             num_checks)
{
	std::stringstream ss;
	// Note that it's possible for there to be _no_ results
	for(kat::size_t i = 0; i < num_checks; i++) {
		ss.str("");
		ss << test_or_testcase_name << " failed check #" << (i+1) << " (1-based) at source line " << results[i].line_number;
		auto message = ss.str();
		CHECK_MESSAGE(results[i].result, message);
	}
}

template <typename F>
void execute_simple_testcase_on_gpu_and_check(
	std::string                      testcase_name,
	F                                testcase_device_function,
	size_t                           num_checks)
{
	auto results = execute_simple_testcase_on_gpu(testcase_device_function, num_checks);
	check_results(testcase_name, results.data(), results.size());
}

constexpr const auto checks { 1 };


// Notes:
// The test suites prefixed with libstdcxx are derived from the
// stdlibc++ tests for std::array. They are therefore subject to the
// same license as the code for kat::array itself - see
// <kat/array.hpp> for details.
//
// ... however... we can't use those unit tests which are expected to
// fail, nor do we process exit's/abort's. At least, it seems that's
// not possible with doctest and in the same file as other tests. So,
// in particular, none of the tests ending with "-neg" are used.

#if __cplusplus >= 201703L

struct unswappable_type { };

void swap(unswappable_type&, unswappable_type&) = delete;
namespace kat {
void swap(unswappable_type&, unswappable_type&) = delete;
}

// Not swappable, and pair not swappable via the generic std::swap.
struct immovable_type { immovable_type(immovable_type&&) = delete; };
#endif




namespace test_hdfs {

namespace capacity {

struct empty {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	{
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };

		GPU_CHECK( not a.empty() );
	}
	{
		const size_t len = 0;
		typedef kat::array<int, len> array_type;
		array_type a;

		GPU_CHECK( a.empty() );
	}
}
};

struct max_size {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;

	{
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };

		GPU_CHECK( a.max_size() == len );
	}

	{
		const size_t len = 0;
		typedef kat::array<int, len> array_type;
		array_type a;

		GPU_CHECK( a.max_size() == len );
	}
}
};


struct size {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;

	{
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };

		GPU_CHECK( a.size() == len );
	}

	{
		const size_t len = 0;
		typedef kat::array<int, len> array_type;
		array_type a;

		GPU_CHECK( a.size() == len );
	}
}
};

} // namespace capacity

namespace comparison_operators {

struct equal {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;

	 const size_t len = 5;
	 typedef kat::array<int, len> array_type;
	 array_type a = { { 0, 1, 2, 3, 4 } };
	 array_type b = { { 0, 1, 2, 3, 4 } };
	 array_type c = { { 0, 1, 2, 3 } };

	 GPU_CHECK( a == b );
	 GPU_CHECK( !(a == c) );
}
};

struct greater {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3, 4 } };
	array_type c = { { 0, 1, 2, 3, 7 } };

	GPU_CHECK( !(a > b) );
	GPU_CHECK( c > a );
}
};

struct greater_or_equal {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3, 4 } };
	array_type c = { { 0, 1, 2, 3, 7 } };

	GPU_CHECK( a >= b );
	GPU_CHECK( c >= a );
}
};

struct less {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3, 4 } };
	array_type c = { { 0, 1, 2, 3, 7 } };

	GPU_CHECK( !(a < b) );
	GPU_CHECK( a < c );
}
};


struct less_or_equal {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3, 4 } };
	array_type c = { { 0, 1, 2, 3, 7 } };

	GPU_CHECK( a <= b );
	GPU_CHECK( a <= c );
}
};

struct not_equal {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3, 4 } };
	array_type c = { { 0, 1, 2, 3 } };

	GPU_CHECK( !(a != b) );
	GPU_CHECK( a != c );
}
};


} // namespace comparison_operators

namespace cons {

struct aggregate_init {
KAT_HD void operator()(
	result_of_check*,
	kat::size_t)
{
	typedef kat::array<int, 5> array_type;

	array_type a = { { 0, 1, 2, 3, 4 } };
	array_type b = { { 0, 1, 2, 3 } };

	a = b;
	b = a;
}
};

#if __cplusplus >= 201703L
struct deduction {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::array a1{ 1, 2, 3 };
	check_type<kat::array<int, 3>>(a1);
	int y = 2;
	const int z = 3;
	kat::array a2{ 1, y, z };
	check_type<kat::array<int, 3>>(a2);
	kat::array a3{ 'a', 'b', 'c', 'd', 'e' };
	check_type<kat::array<char, 5>>(a3);

	kat::array copy = a1;
	check_type<decltype(a1)>(copy);
	kat::array move = std::move(a1);
	check_type<decltype(a1)>(move);
}
};
#endif

} // namespace cons

namespace element_access {

struct tc54338 {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;

	struct A
	{
		bool valid = true;
		KAT_HD ~A() { valid = false; }
	};

#pragma push
#pragma nv_diag_suppress = missing_default_constructor_on_const
	const kat::array<A, 1> a;
	const A& aa = a.at(0);
	GPU_CHECK(aa.valid);
#pragma pop
}
};


struct back {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;

	{
		array_type a = { { 0, 1, 2, 3, 4 } };
		int& ri = a.back();
		GPU_CHECK( ri == 4 );
	}

	{
		const array_type ca = { { 4, 3, 2, 1, 0 } };
		const int& cri = ca.back();
		GPU_CHECK( cri == 0 );
	}
}
};

struct data {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;

	{
		array_type a = { { 0, 1, 2, 3, 4 } };
		int* pi = a.data();
		GPU_CHECK( *pi == 0 );
	}

	{
		const array_type ca = { { 4, 3, 2, 1, 0 } };
		const int* pci = ca.data();
		GPU_CHECK( *pci == 4 );
	}
}
};

struct front {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;

	{
		array_type a = { { 0, 1, 2, 3, 4 } };
		int& ri = a.front();
		GPU_CHECK( ri == 0 );
	}

	{
		const array_type ca = { { 4, 3, 2, 1, 0 } };
		const int& cri = ca.front();
		GPU_CHECK( cri == 4 );
	}
}
};

} // namespace element_access

namespace iterators {

struct end_is_one_past {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };

	array_type::iterator b = a.begin();
	array_type::iterator e = a.end();

	GPU_CHECK( e != (b + a.size() - 1) );
}
};

} // namespace iterators

namespace requirements {


struct contiguous {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };

	// &a[n] == &a[0] + n for all 0 <= n < N.
	for (size_t i = 0; i < len; ++i)
	{
		GPU_CHECK( &a[i] == &a[0] + i );
	}
}
};

struct fill {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;

	const size_t len = 3;
	typedef kat::array<int, len> array_type;

	array_type a = { { 0, 1, 2 } };
	const int value = 5;

	a.fill(value);
	GPU_CHECK( a[0] == value );
	GPU_CHECK( a[1] == value );
	GPU_CHECK( a[2] == value );
}
};

struct member_swap {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;

	array_type a = { { 0, 1, 2, 3, 4 } };
	const array_type a_ref = a;

	array_type b = { { 4, 3, 2, 1, 0 } };
	const array_type b_ref = b;

	a.swap(b);
	GPU_CHECK( a == b_ref );
	GPU_CHECK( b == a_ref );
}
};


struct zero_sized_arrays {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 0;
	typedef kat::array<int, len> array_type;

	// 1: ?
	array_type a = { };

	// 2
	array_type b;

	// 3
	// begin() == end()
	GPU_CHECK( a.begin() == a.end() );
	GPU_CHECK( b.begin() == b.end() );
}
};


} // namespace requirements

namespace specialized_algorithms {

struct swap {
KAT_HD void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	const size_t len = 5;
	typedef kat::array<int, len> array_type;

	array_type a = { { 0, 1, 2, 3, 4 } };
	const array_type a_ref = a;

	array_type b = { { 4, 3, 2, 1, 0 } };
	const array_type b_ref = b;

	kat::swap(a, b);
	GPU_CHECK( a == b_ref );
	GPU_CHECK( b == a_ref );
}
};


#if __cplusplus >= 201703L
struct swap_cpp17 {
KAT_HD void operator()(
	result_of_check*,
	kat::size_t  )
{
//	TODO: The next line fails if enabled - though it really shouldn't!
//	static_assert( not std::is_swappable<kat::array<unswappable_type, 42>>::value );
//	TODO: The next line triggers an error, rather than SFINAE'ing into making is_swappable return false
//	static_assert( not std::is_swappable<kat::array<immovable_type, 42>>::value );
}
};

#endif

} // namespace specialized_algorithms


struct range_access {
KAT_DEV void operator()(
	result_of_check* results,
	kat::size_t   num_checks)
{
	kat::size_t check_index = 0;
	kat::array<int, 3> a{{1, 2, 3}};
	auto b = kat::begin(a);
	GPU_CHECK(&(*b) == &(a[0]));
	auto e = kat::end(a);
	GPU_CHECK(&(*e) == &a[a.size()]);
	printf("*b is %d", (int) *b);
}
};


} // namespace test_hdfs

TEST_SUITE("device-side-libstdcxx") {

TEST_CASE("capacity") {
	SUBCASE("empty")            { execute_simple_testcase_on_gpu_and_check("empty",            test_hdfs::capacity::empty{},                        2 * checks); }
	SUBCASE("max_size")         { execute_simple_testcase_on_gpu_and_check("max_size",         test_hdfs::capacity::max_size{},                     2 * checks); }
//	SUBCASE("size")             { execute_simple_testcase_on_gpu_and_check("size",             test_hdfs::capacity::size{},                         2 * checks); }

} // TEST_CASE("capacity")
/*
TEST_CASE("comparison operators") {

	SUBCASE("equal")            { execute_simple_testcase_on_gpu_and_check("empty",            test_hdfs::comparison_operators::equal{},            2 * checks); }
	SUBCASE("greater")          { execute_simple_testcase_on_gpu_and_check("greater",          test_hdfs::comparison_operators::greater{},          2 * checks); }
	SUBCASE("greater_or_equal") { execute_simple_testcase_on_gpu_and_check("greater_or_equal", test_hdfs::comparison_operators::greater_or_equal{}, 2 * checks); }
	SUBCASE("less")             { execute_simple_testcase_on_gpu_and_check("less",             test_hdfs::comparison_operators::less{},             2 * checks); }
	SUBCASE("less_or_equal")    { execute_simple_testcase_on_gpu_and_check("less_or_equal",    test_hdfs::comparison_operators::less_or_equal{},    2 * checks); }
	SUBCASE("not_equal")        { execute_simple_testcase_on_gpu_and_check("not_equal",        test_hdfs::comparison_operators::not_equal{},        2 * checks); }

}

TEST_CASE("cons") {

	SUBCASE("aggregate_init")   { execute_simple_testcase_on_gpu_and_check("aggregate_init",   test_hdfs::cons::aggregate_init{},                   0 * checks); }
#if __cplusplus >= 201703L
	SUBCASE("deduction")        { execute_simple_testcase_on_gpu_and_check("deduction",        test_hdfs::cons::deduction{},                        0 * checks); }
#endif

}

TEST_CASE("element_access") {
	SUBCASE("54338")            { execute_simple_testcase_on_gpu_and_check("54338",            test_hdfs::element_access::tc54338{},                1 * checks); }
	// Not testing at_invalid_index, as for device_side, that's a test where the kernel will fail
	// Not including test in 60497.cc - it involes std::debug::
	SUBCASE("back")             { execute_simple_testcase_on_gpu_and_check("back",             test_hdfs::element_access::back{},                   2 * checks); }
	SUBCASE("data")             { execute_simple_testcase_on_gpu_and_check("data",             test_hdfs::element_access::data{},                   2 * checks); }
	SUBCASE("front")            { execute_simple_testcase_on_gpu_and_check("front",            test_hdfs::element_access::front{},                  2 * checks); }
}

// Note: Not including anything from the debug/ folder

TEST_CASE("iterators") {
	SUBCASE("end_is_one_past")  { execute_simple_testcase_on_gpu_and_check("end_is_one_past",  test_hdfs::iterators::end_is_one_past{},             1 * checks); }
}

TEST_CASE("requirements") {
	SUBCASE("contiguous")       { execute_simple_testcase_on_gpu_and_check("contiguous",       test_hdfs::requirements::contiguous{},               5 * checks); }
	SUBCASE("fill")             { execute_simple_testcase_on_gpu_and_check("fill",             test_hdfs::requirements::fill{},                     3 * checks); }
	SUBCASE("member_swap")      { execute_simple_testcase_on_gpu_and_check("member_swap",      test_hdfs::requirements::member_swap{},              2 * checks); }
	// Not including the non_default_constructible test - where's the test in there?
	SUBCASE("zero_sized_arrays"){ execute_simple_testcase_on_gpu_and_check("zero_sized_arrays",test_hdfs::requirements::zero_sized_arrays{},        2 * checks); }
}

TEST_CASE("specialized algorithms") {
	SUBCASE("swap")             { execute_simple_testcase_on_gpu_and_check("swap",             test_hdfs::specialized_algorithms::swap{},           2 * checks); }
#if __cplusplus >= 201703L
	SUBCASE("swap_cpp17")       { execute_simple_testcase_on_gpu_and_check("swap_cpp17",       test_hdfs::specialized_algorithms::swap_cpp17{},     0 * checks); }
#endif
}

TEST_CASE("range_access") {
	execute_simple_testcase_on_gpu_and_check("range_access",     test_hdfs::range_access{},     0 * checks);
}
*/
} // TEST_SUITE("device-side-libstdcxx")


TEST_SUITE("host-side-libstdcxx") {

TEST_CASE("capacity") {

	SUBCASE("empty") {
		SUBCASE("") {
			const size_t len = 5;
			typedef kat::array<int, len> array_type;
			array_type a = { { 0, 1, 2, 3, 4 } };

			CHECK_FALSE( a.empty()  );
		}

		SUBCASE("") {
			const size_t len = 0;
			typedef kat::array<int, len> array_type;
			array_type a;

			CHECK( a.empty() );
		}
	}

	SUBCASE("max_size") {

		SUBCASE("") {
			const size_t len = 5;
			typedef kat::array<int, len> array_type;
			array_type a = { { 0, 1, 2, 3, 4 } };

			CHECK( a.max_size() == len );
		}

		SUBCASE("") {
			const size_t len = 0;
			typedef kat::array<int, len> array_type;
			array_type a;

			CHECK( a.max_size() == len );
		}
	}

	SUBCASE("size") {

		SUBCASE("") {
			const size_t len = 5;
			typedef kat::array<int, len> array_type;
			array_type a = { { 0, 1, 2, 3, 4 } };

			CHECK( a.size() == len );
		}

		SUBCASE("") {
			const size_t len = 0;
			typedef kat::array<int, len> array_type;
			array_type a;

			CHECK( a.size() == len );
		}
	}

} // TEST_CASE("capacity")

TEST_CASE("comparison operators") {

	SUBCASE("equal") {
		 const size_t len = 5;
		 typedef kat::array<int, len> array_type;
		 array_type a = { { 0, 1, 2, 3, 4 } };
		 array_type b = { { 0, 1, 2, 3, 4 } };
		 array_type c = { { 0, 1, 2, 3 } };

		 CHECK( a == b );
		 CHECK( !(a == c) );
	}

	SUBCASE("greater") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3, 4 } };
		array_type c = { { 0, 1, 2, 3, 7 } };

		CHECK( !(a > b) );
		CHECK( c > a );
	}

	SUBCASE("greater_or_equal") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3, 4 } };
		array_type c = { { 0, 1, 2, 3, 7 } };

		CHECK( a >= b );
		CHECK( c >= a );
	}

	SUBCASE("less") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3, 4 } };
		array_type c = { { 0, 1, 2, 3, 7 } };

		CHECK( !(a < b) );
		CHECK( a < c );	}

	SUBCASE("less_or_equal") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3, 4 } };
		array_type c = { { 0, 1, 2, 3, 7 } };

		CHECK( a <= b );
		CHECK( a <= c );
	}

	SUBCASE("not_equal") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3, 4 } };
		array_type c = { { 0, 1, 2, 3 } };

		CHECK( !(a != b) );
		CHECK( a != c );
	}

}

TEST_CASE("cons") {

	SUBCASE("aggregate initialization") {
		 typedef kat::array<int, 5> array_type;

		array_type a = { { 0, 1, 2, 3, 4 } };
		array_type b = { { 0, 1, 2, 3 } };

		a = b;
		b = a;
	}

#if __cplusplus >= 201703L
	SUBCASE("deduction") {
		kat::array a1{ 1, 2, 3 };
		check_type<kat::array<int, 3>>(a1);
		int y = 2;
		const int z = 3;
		kat::array a2{ 1, y, z };
		check_type<kat::array<int, 3>>(a2);
		kat::array a3{ 'a', 'b', 'c', 'd', 'e' };
		check_type<kat::array<char, 5>>(a3);

		kat::array copy = a1;
		check_type<decltype(a1)>(copy);
		kat::array move = std::move(a1);
		check_type<decltype(a1)>(move);
	}

// doctest doesn't support testcases which you expect to fail to _compile_.
#if 0
	SUBCASE("deduction" * should_fail(true)) {
		kat::array a1{};
		kat::array a2{1, 2u, 3};
	}
#endif

#endif

}

// Using nothing from the "constrution" and "debug" testcase directory

TEST_CASE("element access") {

	SUBCASE("54338") {

		struct A
		{
		bool valid = true;
		~A() { valid = false; }
		};
#pragma push
#pragma nv_diag_suppress = missing_default_constructor_on_const
		const kat::array<A, 1> a;
		const A& aa = a.at(0);
		CHECK(aa.valid);
#pragma pop
	}

	// Not including test in 60497.cc - it involes std::debug::

// Not testing what happens when accessing an array at an invalid
// index, because the intended behavior is execution termination,
// and we don't support testing that.
#if 0
	SUBCASE("at_invalid_index") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };

		try
			{
				a.at(len);
				CHECK( false );
			}
		catch(std::out_of_range& obj)
			{
				// Expected.
				CHECK( true );
			}
		catch(...)
			{
				// Failed.
				CHECK( false );
			}
	}
#endif

	SUBCASE("back") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;

		{
			array_type a = { { 0, 1, 2, 3, 4 } };
			int& ri = a.back();
			CHECK( ri == 4 );
		}

		{
			const array_type ca = { { 4, 3, 2, 1, 0 } };
			const int& cri = ca.back();
			CHECK( cri == 0 );
		}
	}

	SUBCASE("") {
		 const size_t len = 5;
		typedef kat::array<int, len> array_type;

		{
			array_type a = { { 0, 1, 2, 3, 4 } };
			int* pi = a.data();
			CHECK( *pi == 0 );
		}

		{
			const array_type ca = { { 4, 3, 2, 1, 0 } };
			const int* pci = ca.data();
			CHECK( *pci == 4 );
		}
	}

	SUBCASE("") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;

		{
			array_type a = { { 0, 1, 2, 3, 4 } };
			int& ri = a.front();
			CHECK( ri == 0 );
		}

		{
			const array_type ca = { { 4, 3, 2, 1, 0 } };
			const int& cri = ca.front();
			CHECK( cri == 4 );
		}
	}

}

TEST_CASE("iterators") {

	SUBCASE("end is one past") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;
		array_type a = { { 0, 1, 2, 3, 4 } };

		array_type::iterator b = a.begin();
		array_type::iterator e = a.end();

		CHECK( e != (b + a.size() - 1) );
	}

}

TEST_CASE("requirements") {

	SUBCASE("contiguous") {
	const size_t len = 5;
	typedef kat::array<int, len> array_type;
	array_type a = { { 0, 1, 2, 3, 4 } };

	// &a[n] == &a[0] + n for all 0 <= n < N.
	for (size_t i = 0; i < len; ++i)
		{
		CHECK( &a[i] == &a[0] + i );
		}
	}

	SUBCASE("fill") {
		 const size_t len = 3;
		typedef kat::array<int, len> array_type;

		array_type a = { { 0, 1, 2 } };
		const int value = 5;

		a.fill(value);
		CHECK( a[0] == value );
		CHECK( a[1] == value );
		CHECK( a[2] == value );
	}

	SUBCASE("member swap") {
		 const size_t len = 5;
		typedef kat::array<int, len> array_type;

		array_type a = { { 0, 1, 2, 3, 4 } };
		const array_type a_ref = a;

		array_type b = { { 4, 3, 2, 1, 0 } };
		const array_type b_ref = b;

		a.swap(b);
		CHECK( a == b_ref );
		CHECK( b == a_ref );
	}

	// Not include the non_default_constructible test - where's the test in there?

	SUBCASE("zero-sized arrays") {
		 const size_t len = 0;
		typedef kat::array<int, len> array_type;

		// 1: ?
		array_type a = { };

		// 2
		array_type b;

		// 3
		// begin() == end()
		CHECK( a.begin() == a.end() );
		CHECK( b.begin() == b.end() );
	}
}

TEST_CASE("specialized algorithms") {

	SUBCASE("swap") {
		const size_t len = 5;
		typedef kat::array<int, len> array_type;

		array_type a = { { 0, 1, 2, 3, 4 } };
		const array_type a_ref = a;

		array_type b = { { 4, 3, 2, 1, 0 } };
		const array_type b_ref = b;

		kat::swap(a, b);
		CHECK( a == b_ref );
		CHECK( b == a_ref );
	}

#if __cplusplus >= 201703L
	SUBCASE("swap C++17") {
//	TODO: The next line fails if enabled - though it really shouldn't!
//		static_assert( not std::is_swappable<kat::array<unswappable_type, 42>>::value );
//	TODO: The next line triggers an error, rather than SFINAE'ing into making is_swappable return false
//		static_assert( not std::is_swappable<kat::array<immovable_type, 42>>::value );
	}
#endif

}

TEST_CASE("range access") {
	SUBCASE("") {
		kat::array<int, 3> a{{1, 2, 3}};
		auto b = kat::begin(a);
		CHECK(&(*b) == &(a[0]));
		auto e = kat::end(a);
		CHECK(&(*e) == &(a[a.size()]));
	}
}

} // TEST_SUITE("host-side-libstdcxx")

TEST_SUITE("libstdcxx-constexpr") {

TEST_CASE("capacity") {

	SUBCASE("constexpr functions") {
		constexpr const auto s { 60 };
		constexpr const kat::array<long, s> arr{};
		constexpr const auto size = arr.size();         (void) size;
		constexpr const auto max_size = arr.max_size(); (void) max_size;
		constexpr const auto is_empty = arr.empty();    (void) is_empty;
	}
}

TEST_CASE("comparison operators") {

#if __cplusplus >= 202001L
	SUBCASE("constexpr") {
		constexpr const kat::array<const int, 3> a1{{1, 2, 3}};
		constexpr const kat::array<const int, 3> a2{{4, 5, 6}};
		constexpr const kat::array<const int, 3> a3{{1, 2, 4}};
		constexpr const kat::array<const int, 3> a4{{1, 3, 3}};

		static_assert(a1 == a1);
		static_assert(a1 != a2);
		static_assert(a1 < a3);
		static_assert(a4 > a1);
		static_assert(a1 <= a3);
		static_assert(a4 >= a1);
		static_assert(std::is_eq(a1 <=> a1));
		static_assert(std::is_neq(a1 <=> a2));
		static_assert(std::is_lt(a1 <=> a3));
		static_assert(std::is_gt(a4 <=> a1));

		constexpr const kat::array<unsigned char, 3> a5{{1, 2, 3}};
		constexpr const kat::array<unsigned char, 3> a6{{4, 5, 6}};
		constexpr const kat::array<unsigned char, 3> a7{{1, 2, 4}};
		constexpr const kat::array<unsigned char, 3> a8{{1, 3, 3}};

		static_assert(a5 == a5);
		static_assert(a5 != a6);
		static_assert(a5 < a7);
		static_assert(a8 > a5);
		static_assert(a5 <= a7);
		static_assert(a8 >= a5);
		static_assert(std::is_eq(a5 <=> a5));
		static_assert(std::is_neq(a5 <=> a6));
		static_assert(std::is_lt(a5 <=> a7));
		static_assert(std::is_gt(a8 <=> a5));
	}
#endif
}

TEST_CASE("construction") {

	SUBCASE("constexpr") {
#if __cplusplus >= 202001L
		SUBCASE("") {
			const char x[6]{};
			kat::array<char, 6> y = std::to_array(x);

			constexpr char x2[] = "foo";
			constexpr kat::array<char, 4> y2 = std::to_array(x2);
			static_assert( std::equal(y2.begin(), y2.end(), x2) );
		}
#endif

#if __cplusplus >= 202001L
		SUBCASE("") {
			struct MoveOnly
			{
				constexpr MoveOnly(int i = 0) : i(i) { }
				constexpr MoveOnly(MoveOnly&& m) : i(m.i + 100) { }
				int i;
			};

			struct X {
				MoveOnly m[3];
			};
			X x;
			kat::array<MoveOnly, 3> y = std::to_array(std::move(x).m);

			constexpr kat::array<MoveOnly, 3> y2 = std::to_array(X{{1, 2, 3}}.m);
			static_assert( y2[0].i == 101 && y2[1].i == 102 && y2[2].i == 103 );
		}
#endif
	}
}

TEST_CASE("element access") {

	SUBCASE("constexpr element access") {
		typedef kat::array<std::size_t, 6> array_type;
		constexpr array_type a = { { 0, 55, 66, 99, 4115, 2 } };
		constexpr auto v1 = a[1];      (void) v1;
#if __cplusplus >= 201703L
		constexpr auto v2 = a.at(2);   (void) v2;
#endif
		constexpr auto v3 = a.front(); (void) v3;
		constexpr auto v4 = a.back();  (void) v4;
	}
}

constexpr bool test_fill() {
	auto ok = true;

	kat::array<float,3> fa{};
	fa.fill(3.333f);

	ok = ok && (fa[0] == fa[2]);

	return ok;
}

constexpr int test_iter()
{
  constexpr kat::array<int, 3> a1{{1, 2, 3}};
  static_assert(1 == *a1.begin());
  auto n = a1[0] * a1[1]* a1[2];
  static_assert(1 == *a1.cbegin());

  kat::array<int, 3> a2{{0, 0, 0}};
  auto a1i = a1.begin();
  auto a1e = a1.end();
  auto a2i = a2.begin();
  while (a1i != a1e)
    *a2i++ = *a1i++;

  return n;
}

#if __cplusplus >= 202001L
constexpr bool test_swap()
{
  auto ok = true;
  kat::array<float,3> fa{{1.1f, 2.2f, 3.3f}};
  kat::array<float,3> fb{{4.4f, 5.5f, 6.6f}};
  fb.swap(fa);
  ok = ok && (fa[0] == 4.4f);
  std::swap(fa, fb);
  ok = ok && (fa[0] == 1.1f);
  return ok;
}
#endif


TEST_CASE("requirements") {

	// Not including citerators test

	// Note: Not really constexpr
	SUBCASE("constexpr fill") {
		static_assert(test_fill());
	}

	SUBCASE("constexpr iter") {
		static_assert(test_iter());
	}

#if __cplusplus >= 202001L
	SUBCASE("constexpr swap") {
		static_assert(test_swap());
	}
#endif
}

// Modified from the libstdc++ test - which seems like it should work

// The following segment is part of the tuple interface testcase, below,
// but must appear in file scope, not within a function.

namespace testcase_tuple_interface {
namespace subcase_get {

kat::array<int, 5> ai;
const kat::array<int, 5> cai(ai);

constexpr const int& cri = kat::get<0>(cai);
constexpr int&  ri = kat::get<0>(ai);
constexpr int&& rri = kat::get<0>(std::move(ai));

} //namespace subcase_get

namespace subcase_tuple_element_cpp14 {

constexpr const std::size_t len = 3;
using array_type = kat::array<int, len>;

} // subcase_tuple_element_cpp14


} // namespace testcase_tuple_interface


TEST_CASE("tuple interface") {


	SUBCASE("tuple_element") {
		using kat::array;
		using std::tuple_element;
		using std::is_same;

		const std::size_t len = 3;
		typedef array<int, len> array_type;

		static_assert(is_same<tuple_element<0, array_type>::type, int>::value, "" );
		static_assert(is_same<tuple_element<1, array_type>::type, int>::value, "" );
		static_assert(is_same<tuple_element<2, array_type>::type, int>::value, "");

		static_assert(is_same<tuple_element<0, const array_type>::type,
			            const int>::value, "");
		static_assert(is_same<tuple_element<1, const array_type>::type,
			            const int>::value, "");
		static_assert(is_same<tuple_element<2, const array_type>::type,
			            const int>::value, "");

		static_assert(is_same<tuple_element<0, volatile array_type>::type,
			            volatile int>::value, "");
		static_assert(is_same<tuple_element<1, volatile array_type>::type,
			            volatile int>::value, "");
		static_assert( (is_same<tuple_element<2, volatile array_type>::type,
			       volatile int>::value == true), "" );

		static_assert(is_same<tuple_element<0, const volatile array_type>::type,
			            const volatile int>::value, "");
		static_assert(is_same<tuple_element<1, const volatile array_type>::type,
			            const volatile int>::value, "");
		static_assert(is_same<tuple_element<2, const volatile array_type>::type,
			            const volatile int>::value, "");
	}


#if __cplusplus >= 201402L
	SUBCASE("tuple_element C++14") {
		using std::is_same;
		using std::tuple_element;
		using std::tuple_element_t;
		// For some reason, this has to be outside the subcase, at file scope, rather
		// then defined here. Why? My guess is some weird CUDA bug.
		using testcase_tuple_interface::subcase_tuple_element_cpp14::array_type;


		static_assert(is_same<tuple_element_t<0, array_type>, int>::value, "");
		static_assert(is_same<tuple_element_t<1, array_type>, int>::value, "");
		static_assert(is_same<tuple_element_t<2, array_type>, int>::value, "");

		static_assert(is_same<tuple_element_t<0, const array_type>,
			            const int>::value, "");
		static_assert(is_same<tuple_element_t<1, const array_type>,
			            const int>::value, "");
		static_assert(is_same<tuple_element_t<2, const array_type>,
			            const int>::value, "");

		static_assert(is_same<tuple_element_t<0, volatile array_type>,
			            volatile int>::value, "");
		static_assert(is_same<tuple_element_t<1, volatile array_type>,
			            volatile int>::value, "");
		static_assert(is_same<tuple_element_t<2, volatile array_type>,
			            volatile int>::value, "");

		static_assert(is_same<tuple_element_t<0, const volatile array_type>,
			            const volatile int>::value, "");
		static_assert(is_same<tuple_element_t<1, const volatile array_type>,
			            const volatile int>::value, "");
		static_assert(is_same<tuple_element_t<2, const volatile array_type>,
			            const volatile int>::value, "");
	}
#endif

	SUBCASE("get") {
		// see above
	}

	SUBCASE("tuple_size") {
		using kat::array;
		using std::tuple_size;
		using std::size_t;
		// This relies on the fact that <utility> includes <type_traits>:
		using std::is_same;

		SUBCASE("") {
			const size_t len = 5;
			typedef array<int, len> array_type;
			static_assert(tuple_size<array_type>::value == 5, "");
			static_assert(tuple_size<const array_type>::value == 5, "");
			static_assert(tuple_size<volatile array_type>::value == 5, "");
			static_assert(tuple_size<const volatile array_type>::value == 5, "");
		}

		SUBCASE("") {
			const size_t len = 0;
			typedef array<float, len> array_type;
			static_assert(tuple_size<array_type>::value == 0, "");
			static_assert(tuple_size<const array_type>::value == 0, "");
			static_assert(tuple_size<volatile array_type>::value == 0, "");
			static_assert(tuple_size<const volatile array_type>::value == 0, "");
		}
	}

} // TESTCASE("tuple interface")

} // TEST_SUITE("libstdcxx-constexpr")


//TEST_SUITE("span-device-side") {
//} // TEST_SUITE("span-device-side")

