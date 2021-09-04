//
// Original code Copyright (c) Electronic Arts Inc. All rights reserved
// Modifications/Rewrite Copyright (c) 2021 Eyal Rozenberg.
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
// master branch, on 2021-09-04.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/pair.hpp>

#include <cuda/runtime_api.hpp>

#include <type_traits>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <cstring>

using std::strcmp;

namespace test_structs {

template <typename T>
struct use_self             // : public unary_function<T, T> // Perhaps we want to make it a subclass of unary_function.
	{
	typedef T result_type;

	const T& operator()(const T& x) const
	{ return x; }
	};

template <typename Pair>
struct use_first {
	typedef Pair argument_type;
	typedef typename Pair::first_type result_type;

	const result_type& operator()(const Pair& x) const
	{ return x.first; }
};

template <typename Pair>
struct use_second {
	typedef Pair argument_type;
	typedef typename Pair::second_type result_type;

	const result_type& operator()(const Pair& x) const
	{ return x.second; }
};

} // namespace test_structs

using kat::pair;
using namespace test_structs;

namespace kernels {

__global__ void combine_a_pair(kat::pair<int, double> p, int* result)
{
	*result = static_cast<int>(p.first * 10000 + (int) (p.second * 100));
}

__global__ void write_a_pair(kat::pair<double, int>* result)
{
	*result = pair<double,int>(12.34, 56);
}


} // namespace kernels

TEST_SUITE("pair") {

TEST_CASE("construction")
{
	int _0 = 0, _2 = 2, _3 = 3;
	float _1f = 1.f;

	// pair();
	pair<int, float> ifPair1;
	CHECK(ifPair1.first == 0);
	CHECK(ifPair1.second == 0.f);

	// pair(const T1& x, const T2& y);
	pair<int, float> ifPair2(_0, _1f);
	CHECK(ifPair2.first == 0);
	CHECK(ifPair2.second == 1.f);

	// template <typename U, typename V>
	// pair(U&& u, V&& v);
	pair<int, float> ifPair3(int(0), float(1.f));
	CHECK(ifPair3.first == 0);
	CHECK(ifPair3.second == 1.f);

	// template <typename U>
	// pair(U&& x, const T2& y);
	const float fConst1 = 1.f;
	pair<int, float> ifPair4(int(0), fConst1);
	CHECK(ifPair4.first == 0);
    CHECK((ifPair4.second == 1.f));

	// template <typename V>
	// pair(const T1& x, V&& y);
	const int intConst0 = 0;
	pair<int, float> ifPair5(intConst0, float(1.f));
	CHECK(ifPair5.first == 0);
    CHECK((ifPair5.second == 1.f));

	pair<const int, const int> constIntPair(_2, _3);
	CHECK(constIntPair.first == 2);
    CHECK((constIntPair.second == 3));

	// pair(const pair&) = default;
	pair<int, float> ifPair2Copy(ifPair2);
	CHECK(ifPair2Copy.first == 0);
    CHECK((ifPair2Copy.second == 1.f));

	pair<const int, const int> constIntPairCopy(constIntPair);
	CHECK(constIntPairCopy.first == 2);
    CHECK((constIntPairCopy.second == 3));

	// template<typename U, typename V>
	// pair(const pair<U, V>& p);
	pair<long, double> idPair2(ifPair2);
	CHECK(idPair2.first == 0);
    CHECK((idPair2.second == 1.0));

	// pair(pair&& p);

	// template<typename U, typename V>
	// pair(pair<U, V>&& p);
}

TEST_CASE("assignment and swap operators")
{
	// pair& operator=(const pair& p);

	// template<typename U, typename V>
	// pair& operator=(const pair<U, V>& p);

	// pair& operator=(pair&& p);

	// template<typename U, typename V>
	// pair& operator=(pair<U, V>&& p);

	// void swap(pair& p);
	
}

TEST_CASE("use_self, use_first, use_second")
{
	int _0 = 0;
	float _1f = 1.f;

	pair<int, float> ifPair2(_0, _1f);

	// use_self, use_first, use_second
	use_self<pair<int, float> > usIFPair;
	use_first<pair<int, float> > u1IFPair;
	use_second<pair<int, float> > u2IFPair;


	ifPair2 = usIFPair(ifPair2);
	CHECK(ifPair2.first == 0);
    CHECK((ifPair2.second == 1));

	int first = u1IFPair(ifPair2);
	CHECK(first == 0);

	float second = u2IFPair(ifPair2);
	CHECK(second == 1);
}

TEST_CASE("make_pair")
{
	// make_pair
	pair<int, float> p1 = kat::make_pair(int(0), float(1));
	CHECK(p1.first == 0);
    CHECK((p1.second == 1.f));

	pair<const char*, int> p3 = kat::make_pair("a", 1);
	CHECK(strcmp(p3.first, "a") == 0);
    CHECK((p3.second == 1));

	pair<const char*, int> p4 = kat::make_pair<const char*, int>("a", 1);
	CHECK(strcmp(p4.first, "a") == 0);
    CHECK((p4.second == 1));

	pair<int, const char*> p5 = kat::make_pair<int, const char*>(1, "b");
	CHECK(p5.first == 1);
    CHECK((strcmp(p5.second, "b") == 0));

	auto p60 = kat::make_pair("a", "b");  // Different strings of same length of 1.
	CHECK(strcmp(p60.first, "a") == 0);
    CHECK((strcmp(p60.second, "b") == 0));

	auto p61 = kat::make_pair("ab", "cd");  // Different strings of same length > 1.
	CHECK(strcmp(p61.first, "ab") == 0);
    CHECK((strcmp(p61.second, "cd") == 0));

	auto p62 = kat::make_pair("abc", "bcdef");  // Different strings of different length.
	CHECK(strcmp(p62.first, "abc") == 0);
    CHECK((strcmp(p62.second, "bcdef") == 0));

	char strA[] = "a";
	auto p70 = kat::make_pair(strA, strA);
	CHECK(strcmp(p70.first, "a") == 0);
    CHECK((strcmp(p70.second, "a") == 0));

	char strBC[] = "bc";
	auto p71 = kat::make_pair(strA, strBC);
	CHECK(strcmp(p71.first, "a") == 0);
    CHECK((strcmp(p71.second, "bc") == 0));

	const char cstrA[] = "a";
	auto p80 = kat::make_pair(cstrA, cstrA);
	CHECK(strcmp(p80.first, "a") == 0);
    CHECK((strcmp(p80.second, "a") == 0));

	const char cstrBC[] = "bc";
	auto p81 = kat::make_pair(cstrA, cstrBC);
	CHECK(strcmp(p81.first, "a") == 0);
    CHECK((strcmp(p81.second, "bc") == 0));
}


TEST_CASE("EASTL regressions")
{
	// The following code was giving an EDG compiler:
	//       error 145: a value of type "int" cannot be used to initialize
	//       an entity of type "void *" second(kat::forward<V>(v)) {}
	// template <typename U, typename V>
	// pair(U&& u, V&& v);
	typedef kat::pair<float*, void*> TestPair1;
	float fOne = 1.f;
	TestPair1 testPair1(&fOne, (void*) NULL);
	CHECK(*testPair1.first == 1.f);
}

#if __cplusplus >= 201701L

TEST_CASE("structured bindings")
{

	{
		kat::pair<int, int> t = {1,2};
		auto [x,y] = t;
		CHECK(x == 1);
		CHECK(y == 2);
	}

	{
		auto t = kat::make_pair(1, 2);
		auto [x,y] = t;
		CHECK(x == 1);
		CHECK(y == 2);
	}

	{ // reported user-regression structured binding unpacking for iterators
		std::vector<int> v = {1,2,3,4,5,6};
		auto t = kat::make_pair(v.begin(), v.end() - 1);
		auto [x,y] = t;
		CHECK(*x == 1);
		CHECK(*y == 6);
	}

	{ // reported user-regression structured binding unpacking for iterators
		std::vector<int> v = {1,2,3,4,5,6};
		auto t = kat::make_pair(v.begin(), v.end());
		auto [x,y] = t;
		CHECK(*x == 1);
		UNUSED(y);
	}

{ // reported user-regression for const structured binding unpacking for iterators
	std::vector<int> v = {1,2,3,4,5,6};
	const auto [x,y] = kat::make_pair(v.begin(), v.end());;
	CHECK(*x == 1);
	EA_UNUSED(y);
}

}

#endif

TEST_CASE("pass a pair to a kernel") {
	if (not cpu_is_little_endian()) {
		INFO("pair tests for big-endian platforms not implemented yet.");
		return;
	}
	cuda::device_t dev = cuda::device::current::get();
	auto p = kat::make_pair(12, 3.4);
	auto uptr = cuda::memory::device::make_unique<int>(dev);
	cuda::memory::device::zero(uptr.get(), sizeof(int));
	auto kernel = kernels::combine_a_pair;
	dev.launch(
		kernel,
		cuda::make_launch_config(1, 1),
		p,
		uptr.get()
		);
	dev.synchronize();
	cuda::outstanding_error::ensure_none();
	int result;
	cuda::memory::copy_single(&result, uptr.get());
	CHECK( result == 120340 );
}

TEST_CASE("write a pair in a kernel") {
	if (not cpu_is_little_endian()) {
		INFO("pair tests for big-endian platforms not implemented yet.");
		return;
	}
	using pair_type = kat::pair<double, int>;
	cuda::device_t dev = cuda::device::current::get();
	auto uptr = cuda::memory::device::make_unique<unsigned char[]>(dev, sizeof(pair_type));
	cuda::memory::device::zero(uptr.get(), sizeof(pair_type));
	auto kernel = kernels::write_a_pair;
	dev.launch(
		kernel,
		cuda::make_launch_config(1, 1),
		reinterpret_cast<pair_type*>(uptr.get())
		);
	dev.synchronize();
	cuda::outstanding_error::ensure_none();
	pair_type result;
	cuda::memory::copy_single(&result, reinterpret_cast<pair_type*>(uptr.get()));
	CHECK( result.first == 12.34);
	CHECK( result.second == 56);
}

} // TEST_SUITE("pair")


