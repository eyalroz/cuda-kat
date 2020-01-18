#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "util/type_name.hpp"
#include "util/miscellany.cuh"
#include <kat/containers/span.hpp>
//#include <kat/detail/execution_space_specifiers>

#include <doctest.h>
#include <cuda/api_wrappers.hpp>


#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>

//constexpr const auto num_grid_blocks {  2 };
//constexpr const auto block_size      {  kat::warp_size + 1 };
//
//constexpr const auto sleep_elongation_multiplicative_factor { 10000 };
//	// we want each sleep command to take a not-insignificant amount of time
//
//constexpr const auto sleep_elongation_additive_factor { 10 };
//	// we want each sleep command to take a not-insignificant amount of time


cuda::launch_configuration_t singleton_grid_config()
{
	auto num_grid_blocks { 1 };
	auto block_size { 1 };
	return cuda::make_launch_config(num_grid_blocks, block_size);
}

// Poor man's addressof
template< class T >
T* addressof(T& arg)
{
        return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char&>(arg)));
}


namespace detail {

enum everything_checks {
	shorts_is_empty
#if __cplusplus >= 201703L
	, shorts_is_std_empty
#endif
	, shorts_data_is_null
	, shorts_begin_equal_to_end
	, shorts_cbegin_equal_to_cend
#if __cplusplus >= 202001L
	, definitely_reinterpret_casted
	, definitely_equivalent
#endif
	, num_checks
};

__host__ __device__ void lwg_3225_constructibility_with_c_array()
{
	static_assert( std::is_constructible<kat::span<int, 1>, int(&)[1]>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, int(&)[1]>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, const int(&)[1]>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, int(&)[2]>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, int(&)[2]>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, const int(&)[2]>::value, "");

	static_assert( std::is_constructible<kat::span<int>, int(&)[2]>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, int(&)[2]>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, const int(&)[2]>::value, "");
}

__host__ __device__ void lwg_3225_constructibility_with_kat_array()
{
	static_assert( std::is_constructible<kat::span<const int* const>, kat::array<int*, 2>>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, kat::array<const int, 4>>::value, "");

	static_assert( std::is_constructible<kat::span<int, 1>, kat::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, kat::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, kat::array<const int, 1>&>::value, "");

	static_assert( std::is_constructible<kat::span<const int, 1>, const kat::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, const kat::array<const int, 1>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, kat::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, kat::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, kat::array<const int, 2>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, const kat::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, const kat::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, const kat::array<const int, 2>&>::value, "");

	static_assert( std::is_constructible<kat::span<int>, kat::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, kat::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, kat::array<const int, 2>&>::value, "");

	static_assert( std::is_constructible<kat::span<const int>, const kat::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, const kat::array<const int, 2>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, kat::array<const int, 1>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int, 1>, const kat::array<int, 1>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int, 1>, const kat::array<const int, 1>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int>, kat::array<const int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int>, const kat::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int>, const kat::array<const int, 2>&>::value, "");
}

__host__ __device__ void lwg_3225_constructibility_with_std_array()
{
	static_assert( std::is_constructible<kat::span<const int* const>, std::array<int*, 2>>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, std::array<const int, 4>>::value, "");

	static_assert( std::is_constructible<kat::span<int, 1>, std::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, std::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, std::array<const int, 1>&>::value, "");

	static_assert( std::is_constructible<kat::span<const int, 1>, const std::array<int, 1>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int, 1>, const std::array<const int, 1>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, std::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, std::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, std::array<const int, 2>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, const std::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, const std::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<const int, 1>, const std::array<const int, 2>&>::value, "");

	static_assert( std::is_constructible<kat::span<int>, std::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, std::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, std::array<const int, 2>&>::value, "");

	static_assert( std::is_constructible<kat::span<const int>, const std::array<int, 2>&>::value, "");
	static_assert( std::is_constructible<kat::span<const int>, const std::array<const int, 2>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int, 1>, std::array<const int, 1>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int, 1>, const std::array<int, 1>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int, 1>, const std::array<const int, 1>&>::value, "");

	static_assert( not std::is_constructible<kat::span<int>, std::array<const int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int>, const std::array<int, 2>&>::value, "");
	static_assert( not std::is_constructible<kat::span<int>, const std::array<const int, 2>&>::value, "");
}


namespace nothrow_construcitibility {
template<bool DoesntThrow>
struct sentinel { int* p; };

template<bool DoesntThrow>
bool operator==(sentinel<DoesntThrow> s, int* p) noexcept { return s.p == p; }

template<bool DoesntThrow>
std::ptrdiff_t operator-(sentinel<DoesntThrow> s, int* p) noexcept(DoesntThrow) { return s.p - p; }

template<bool DoesntThrow>
std::ptrdiff_t operator-(int* p, sentinel<DoesntThrow> s) noexcept { return p - s.p; }
}

__host__ __device__ void nothrow_constructibility()
	{
	using kat::span;
	using namespace nothrow_construcitibility;

	static_assert( std::is_nothrow_constructible< kat::span<int>>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int, 0>>::value, "" );

	static_assert( std::is_nothrow_constructible< kat::span<int>, span<int>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<const int>, span<int>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int>, span<int, 1>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<const int>, span<int, 1>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int, 1>, span<int, 1>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<const int, 1>, span<int, 1>&>::value, "" );

	static_assert( std::is_nothrow_constructible< kat::span<int>, int(&)[1]>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int, 1>, int(&)[1]>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int>, std::array<int, 1>&>::value, "" );
	static_assert( std::is_nothrow_constructible< kat::span<int, 1>, std::array<int, 1>&>::value, "" );

//
//		static_assert(std::sized_sentinel_for<sentinel<true>, int*>);
//		static_assert(std::sized_sentinel_for<sentinel<false>, int*>);

	static_assert(std::is_nothrow_constructible< kat::span<int>, int*, std::size_t>::value, "");
#if __cpplusplus >= 202001L
	constexpr const bool throws_exceptions = false;
	// These tests require a construct with a different type for the beginning iterator and the sentinel;
	// while they may theoretically be made available for C++ versions earlier than C++20, we'll just leave them out.
	// Stick to more conservative arguments for now.
	static_assert(std::is_nothrow_constructible< kat::span<int>, int*, const int*>::value, "");
	static_assert(std::is_nothrow_constructible< kat::span<int>, int*, sentinel<not throws_exceptions>>::value, "");
	static_assert(not std::is_nothrow_constructible< kat::span<int>, int*, sentinel<throws_exceptions>>::value, "");
#endif
}

__host__ __device__ void everything(bool* results)
{
	struct alignas(256) strawman
	// struct strawman
	  {
	    int x;
	    int y;
	    bool z;
	    int w;
	  };

	  struct naked_span
	  {
	    char* p;
	    std::size_t n;
	  };

	  struct strawman_span
	  {
	    strawman* p;
	    std::size_t n;
	  };

#if __cplusplus >= 202001L
	  // In C++20, span's Extent is allowed to not take up any space if it's an empty struct -
	  // and have the same starting address as the next field; this uses [[no_unique_address]
	  // but ... we don't have that (unless we swap the span implementation for GSL's :-(
	  //
	  static_assert(sizeof(kat::span<char, 0>) <= sizeof(char*), "");
	  static_assert(sizeof(kat::span<const char, 0>) <= sizeof(const char*), "");
	  static_assert(sizeof(kat::span<strawman, 0>) <= sizeof(strawman*), "");
	  static_assert(sizeof(kat::span<strawman, 1>) <= sizeof(strawman*), "");
#endif
	  static_assert(sizeof(kat::span<char>) <= sizeof(naked_span), "");
	  static_assert(sizeof(kat::span<strawman>) <= sizeof(strawman_span), "");

	  constexpr static const kat::array<int, 9> arr_data{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	  constexpr auto arr_data_span = kat::span<const int, sizeof(arr_data) / sizeof(int)>(arr_data);
	  static_assert(arr_data_span.size() == 9, "");
	  static_assert(arr_data_span.size_bytes() == 9 * sizeof(int), "");
	  static_assert(*arr_data_span.begin() == 0, "");
	  static_assert(*arr_data_span.data() == 0, "");
	  static_assert(arr_data_span.front() == 0, "");
	  static_assert(arr_data_span.back() == 8, "");
	  static_assert(arr_data_span[0] == 0, "");
	  static_assert(arr_data_span[1] == 1, "");
	  static_assert(arr_data_span[2] == 2, "");
	  static_assert(arr_data_span[3] == 3, "");
	  static_assert(arr_data_span[4] == 4, "");
	  static_assert(arr_data_span[5] == 5, "");
	  static_assert(arr_data_span[6] == 6, "");
	  static_assert(arr_data_span[7] == 7, "");
	  static_assert(arr_data_span[8] == 8, "");
	  static_assert(!arr_data_span.empty(), "");
	  static_assert(decltype(arr_data_span)::extent == 9, "");

	  constexpr static int data[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	  constexpr auto data_span    = kat::span<const int, sizeof(data) / sizeof(int)>(data);
	  static_assert(data_span.size() == 9, "");
	  static_assert(data_span.size_bytes() == 9 * sizeof(int), "");
	  static_assert(*data_span.begin() == 0, "");
	  static_assert(*data_span.data() == 0, "");
	  static_assert(data_span.front() == 0, "");
	  static_assert(data_span.back() == 8, "");
	  static_assert(data_span[0] == 0, "");
	  static_assert(data_span[1] == 1, "");
	  static_assert(data_span[2] == 2, "");
	  static_assert(data_span[3] == 3, "");
	  static_assert(data_span[4] == 4, "");
	  static_assert(data_span[5] == 5, "");
	  static_assert(data_span[6] == 6, "");
	  static_assert(data_span[7] == 7, "");
	  static_assert(data_span[8] == 8, "");
	  static_assert(!data_span.empty(), "");
	  static_assert(decltype(data_span)::extent == 9, "");

	  constexpr auto data_span_first = data_span.first<3>();
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_first)>::type, kat::span<const int, 3>>::value, "");
	  static_assert(decltype(data_span_first)::extent == 3, "");
	  static_assert(data_span_first.size() == 3, "");
	  static_assert(data_span_first.front() == 0, "");
	  static_assert(data_span_first.back() == 2, "");
	  static_assert(std::tuple_size<decltype(data_span_first)>::value == 3, "");
	  static_assert(std::is_same<std::tuple_element_t<0, decltype(data_span_first)>, const int>::value, "");

	  constexpr auto data_span_first_dyn = data_span.first(4);
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_first_dyn)>::type, kat::span<const int>>::value, "");
	  static_assert(decltype(data_span_first_dyn)::extent == kat::dynamic_extent, "");
	  static_assert(data_span_first_dyn.size() == 4, "");
	  static_assert(data_span_first_dyn.front() == 0, "");
	  static_assert(data_span_first_dyn.back() == 3, "");

	  constexpr auto data_span_last = data_span.last<5>();
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_last)>::type, kat::span<const int, 5>>::value, "");
	  static_assert(decltype(data_span_last)::extent == 5, "");
	  static_assert(data_span_last.size() == 5, "");
	  static_assert(data_span_last.front() == 4, "");
	  static_assert(data_span_last.back() == 8, "");
	  static_assert(std::tuple_size<decltype(data_span_last)>::value == 5, "");
	  static_assert(std::is_same<std::tuple_element_t<0, decltype(data_span_last)>, const int>::value, "");

	  constexpr auto data_span_last_dyn = data_span.last(6);
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_last_dyn)>::type, kat::span<const int>>::value, "");
	  static_assert(decltype(data_span_last_dyn)::extent == kat::dynamic_extent, "");
	  static_assert(data_span_last_dyn.size() == 6, "");
	  static_assert(data_span_last_dyn.front() == 3, "");
	  static_assert(data_span_last_dyn.back() == 8, "");

	  constexpr auto data_span_subspan = data_span.subspan<1, 3>();
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_subspan)>::type, kat::span<const int, 3>>::value, "");
	  static_assert(decltype(data_span_subspan)::extent == 3, "");
	  static_assert(data_span_subspan.size() == 3, "");
	  static_assert(data_span_subspan.front() == 1, "");
	  static_assert(data_span_subspan.back() == 3, "");

	  //		  constexpr auto data_span_subspan_offset = data_span.subspan<8>();
	    constexpr auto data_span_subspan_offset = data_span.subspan<8, 1>();
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_subspan_offset)>::type, kat::span<const int, 1>>::value, "");
	  static_assert(decltype(data_span_subspan_offset)::extent == 1, "");
	  static_assert(data_span_subspan_offset.size() == 1, "");
	  static_assert(data_span_subspan_offset.front() == 8, "");
	  static_assert(data_span_subspan_offset.back() == 8, "");

	  constexpr auto data_span_subspan_empty = data_span.subspan(9, 0);
	  static_assert(
	    std::is_same<typename std::remove_cv<decltype(data_span_subspan_empty)>::type, kat::span<const int>>::value, "");
	  static_assert(decltype(data_span_subspan_empty)::extent == kat::dynamic_extent, "");
	  static_assert(data_span_subspan_empty.size() == 0, "");

	  // TODO: The following line should work, i.e. deduction should give us
	  // the second template argument as Extent - Offset, but somehow it doesn't.
	  // Perhaps it's because I broke a method up into two cases to avoid if constexpr;
	  // perhaps it's because of NVCC - who knows.
	  //
	  // constexpr auto data_span_subspan_empty_static = data_span.subspan<9>();
	  //
	  // instead, well use the following line:
	  constexpr auto data_span_subspan_empty_static = data_span.subspan<9,0>();
	  static_assert(std::is_same<typename std::remove_cv<decltype(data_span_subspan_empty_static)>::type,
	    kat::span<const int, 0>>::value, "");
//		  std::cout << std::hash<decltype(data_span_subspan_empty_static)>() << std::endl;
	  static_assert(decltype(data_span_subspan_empty_static)::extent == 0, "");
	  static_assert(data_span_subspan_empty.size() == 0, "");

	  kat::span<short> shorts{};
	  results[shorts_is_empty] = shorts.empty();

#if __cplusplus >= 201703L
	  results[shorts_is_std_empty] = std::empty(shorts);
#else
#endif
	  results[shorts_data_is_null] = shorts.data() == nullptr;
	  results[shorts_begin_equal_to_end]  = shorts.begin() == shorts.end();
	  results[shorts_cbegin_equal_to_cend]  = shorts.cbegin() == shorts.cend();

#if __cplusplus >= 202001L
	  std::vector<std::int_least32_t> value{ 0 };
	  kat::span<int32_t> muh_span(value);
	  VERIFY(muh_span.size() == 1);
	  std::byte* original_bytes                  = reinterpret_cast<std::byte*>(value.data());
	  original_bytes[0]                          = static_cast<std::byte>(1);
	  original_bytes[1]                          = static_cast<std::byte>(2);
	  original_bytes[2]                          = static_cast<std::byte>(3);
	  original_bytes[3]                          = static_cast<std::byte>(4);
	  kat::span<const std::byte> muh_byte_span   = std::as_bytes(muh_span);
	  kat::span<std::byte> muh_mutable_byte_span = std::as_writable_bytes(muh_span);
	  kat::span<std::byte> muh_original_byte_span(original_bytes, original_bytes + 4);
	  bool definitely_reinterpret_casted0 = std::equal(muh_byte_span.cbegin(), muh_byte_span.cend(),
	    muh_original_byte_span.cbegin(), muh_original_byte_span.cend());
	  bool definitely_reinterpret_casted1 = std::equal(muh_mutable_byte_span.cbegin(),
	    muh_mutable_byte_span.cend(), muh_original_byte_span.cbegin(), muh_original_byte_span.cend());
	  results[definitely_reinterpret_casted] =
	    definitely_reinterpret_casted0 && definitely_reinterpret_casted1;

	  kat::span<std::byte> muh_original_byte_span_ptr_size(original_bytes, 4);
	  results[definitely_equivalent] =
	    std::equal(muh_original_byte_span_ptr_size.cbegin(), muh_original_byte_span_ptr_size.cend(),
		 muh_original_byte_span.cbegin(), muh_original_byte_span.cend());

#endif

}

} // namespace detail

namespace kernels {

__global__ void lwg_3225_constructibility_with_c_array()
{
	detail::lwg_3225_constructibility_with_c_array();
}

__global__ void lwg_3225_constructibility_with_kat_array()
{
	detail::lwg_3225_constructibility_with_kat_array();
}

__global__ void lwg_3225_constructibility_with_std_array()
{
	detail::lwg_3225_constructibility_with_std_array();
}

__global__ void nothrow_constructibility()
{
	detail::nothrow_constructibility();
}

} // namespace kernels


template <typename T, T Value>
struct value_as_type {
	static constexpr const T value { Value };
};

TEST_SUITE("span-host-side-libstdcxx") {

// Note:
// These tests are inspired and/or derived from the stdlibc++ tests
// for kat::span. They are therefore subject to the same license
// as the code for kat::span itself - see <kat/span.hpp> for details.
//
// ... however... we can't use those unit tests which test for assertion
// failure, or process exit/abort - not with doctest, anyway

TEST_CASE("LWG-3225-constructibility-with-C-array")
{
	detail::lwg_3225_constructibility_with_c_array();
}

TEST_CASE("LWG-3225-constructibility-with-kat-array")
{
	detail::lwg_3225_constructibility_with_kat_array();
}

TEST_CASE("LWG-3225-constructibility-with-std-array")
{
	detail::lwg_3225_constructibility_with_std_array();
}

TEST_CASE("nothrow-construcitibility") {
	detail::nothrow_constructibility();
}

TEST_CASE("everything") {
	bool results[detail::num_checks] = {};
	detail::everything(results);
	CHECK(results[detail::shorts_is_empty]);
#if __cplusplus >= 201703L
	CHECK(shorts_is_std_empty);
#endif
	CHECK(results[detail::shorts_data_is_null]);
	CHECK(results[detail::shorts_begin_equal_to_end]);
	CHECK(results[detail::shorts_cbegin_equal_to_cend]);
#if __cplusplus >= 202001L
	CHECK(results[detail::definitely_reinterpret_casted);
	CHECK(results[detail::definitely_equivalent]);
#endif
}

} // TEST_SUITE("host-side")

// The following tests are adapted from the Microsoft implementation
// of the GSL - C++ core guidelines support library. They are licensed
// under the MIT License (MIT):
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

TEST_SUITE("span-host-side-gsl") {

using kat::span;
using kat::make_span;

struct AddressOverloaded {
#if (__cplusplus > 201402L)
	[[maybe_unused]]
#endif
	AddressOverloaded operator&() const
	{
		return {};
	}
};


TEST_CASE("constructors")
{
	span<int> s;
	CHECK(s.size() == 0);
	CHECK(s.data() == nullptr);

	span<const int> cs;
	CHECK(cs.size() == 0);
	CHECK(cs.data() == nullptr);
}

TEST_CASE("constructors_with_extent")
{
	span<int, 0> s;
	CHECK(s.size() == 0);
	CHECK(s.data() == nullptr);

	span<const int, 0> cs;
	CHECK(cs.size() == 0);
	CHECK(cs.data() == nullptr);
}

TEST_CASE("constructors_with_bracket_init")
{
	span<int> s {};
	CHECK(s.size() == 0);
	CHECK(s.data() == nullptr);

	span<const int> cs {};
	CHECK(cs.size() == 0);
	CHECK(cs.data() == nullptr);
}

TEST_CASE("from_pointer_length_constructor")
{
	int arr[4] = {1, 2, 3, 4};

	{
		for (int i = 0; i < 4; ++i)
		{
			{
				span<int> s = {&arr[0], std::size_t {i}};
				CHECK(s.size() == i);
				CHECK(s.data() == &arr[0]);
				CHECK(s.empty() == (i == 0));
				for (int j = 0; j < i; ++j)
				{
					CHECK(arr[j] == s[j]);
					// These are supported in GSL, but not by our span
					// CHECK(arr[j] == s.at(j));
					// CHECK(arr[j] == s(j));
				}
			}
		}
	}

	{
		span<int, 2> s {&arr[0], 2};
		CHECK(s.size() == 2);
		CHECK(s.data() == &arr[0]);
		CHECK(s[0] == 1);
		CHECK(s[1] == 2);
	}

	{
		auto s = kat::make_span(&arr[0], 2);
		CHECK(s.size() == 2);
		CHECK(s.data() == &arr[0]);
		CHECK(s[0] == 1);
		CHECK(s[1] == 2);
	}

}

TEST_CASE("from_pointer_pointer_construction")
{
	int arr[4] = {1, 2, 3, 4};

	{
		span<int> s {&arr[0], &arr[2]};
		CHECK(s.size() == 2);
		CHECK(s.data() == &arr[0]);
		CHECK(s[0] == 1);
		CHECK(s[1] == 2);
	}
	{
		span<int, 2> s {&arr[0], &arr[2]};
		CHECK(s.size() == 2);
		CHECK(s.data() == &arr[0]);
		CHECK(s[0] == 1);
		CHECK(s[1] == 2);
	}

	{
		span<int> s {&arr[0], &arr[0]};
		CHECK(s.size() == 0);
		CHECK(s.data() == &arr[0]);
	}

	{
		span<int, 0> s {&arr[0], &arr[0]};
		CHECK(s.size() == 0);
		CHECK(s.data() == &arr[0]);
	}

	{
		int* p = nullptr;
		span<int> s {p, p};
		CHECK(s.size() == 0);
		CHECK(s.data() == nullptr);
	}

	{
		int* p = nullptr;
		span<int, 0> s {p, p};
		CHECK(s.size() == 0);
		CHECK(s.data() == nullptr);
	}

	{
		auto s = make_span(&arr[0], &arr[2]);
		CHECK(s.size() == 2);
		CHECK(s.data() == &arr[0]);
		CHECK(s[0] == 1);
		CHECK(s[1] == 2);
	}

	{
		auto s = make_span(&arr[0], &arr[0]);
		CHECK(s.size() == 0);
		CHECK(s.data() == &arr[0]);
	}

	{
		int* p = nullptr;
		auto s = make_span(p, p);
		CHECK(s.size() == 0);
		CHECK(s.data() == nullptr);
	}
}

TEST_CASE("from_array_constructor")
{
	int arr[5] = {1, 2, 3, 4, 5};

	{
		const span<int> s {arr};
		CHECK(s.size() == 5);
		CHECK(s.data() == &arr[0]);
	}

	{
		const span<int, 5> s {arr};
		CHECK(s.size() == 5);
		CHECK(s.data() == &arr[0]);
	}

	int arr2d[2][3] = {1, 2, 3, 4, 5, 6};

	{
		const span<int[3]> s {addressof(arr2d[0]), 1};
		CHECK(s.size() == 1);
		CHECK(s.data() == addressof(arr2d[0]));
	}

	int arr3d[2][3][2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

	{
		const span<int[3][2]> s {addressof(arr3d[0]), 1};
		CHECK(s.size() == 1);
	}

	{
		const auto s = make_span(arr);
		CHECK(s.size() == 5);
		CHECK(s.data() == addressof(arr[0]));
	}

	{
		const auto s = make_span(addressof(arr2d[0]), 1);
		CHECK(s.size() == 1);
		CHECK(s.data() == addressof(arr2d[0]));
	}

	{
		const auto s = make_span(addressof(arr3d[0]), 1);
		CHECK(s.size() == 1);
		CHECK(s.data() == addressof(arr3d[0]));
	}
}

TEST_CASE("from_dynamic_array_constructor")
{
	double(*arr)[3][4] = new double[100][3][4];

	{
		span<double> s(&arr[0][0][0], 10);
		CHECK(s.size() == 10);
		CHECK(s.data() == &arr[0][0][0]);
	}

	{
		auto s = make_span(&arr[0][0][0], 10);
		CHECK(s.size() == 10);
		CHECK(s.data() == &arr[0][0][0]);
	}

	delete[] arr;
}

TEST_CASE("from_std_array_constructor")
{
	std::array<int, 4> arr = {1, 2, 3, 4};

	{
		span<int> s {arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());

		span<const int> cs {arr};
//         CHECK(cs.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(cs.data() == arr.data());
	}

	{
		span<int, 4> s {arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());

		span<const int, 4> cs {arr};
//         CHECK(cs.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(cs.data() == arr.data());
	}

	{
		std::array<int, 0> empty_arr {};
		span<int> s {empty_arr};
		CHECK(s.size() == 0);
		CHECK(s.empty());
	}

//	std::array<AddressOverloaded, 4> ao_arr {};
//
//	{
//		span<AddressOverloaded, 4> fs {ao_arr};
//         CHECK(fs.size() == narrow_cast<ptrdiff_t>(ao_arr.size()));
//		CHECK(ao_arr.data() == fs.data());
//	}

	{
		auto get_an_array = []() -> std::array<int, 4> {return {1, 2, 3, 4};};
		auto take_a_span = [](span<const int> s) {static_cast<void>(s);};
		// try to take a temporary std::array
		take_a_span(get_an_array());
	}

	{
		auto s = make_span(arr);
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}

	// This test checks for the bug found in gcc 6.1, 6.2, 6.3, 6.4, 6.5 7.1, 7.2, 7.3 - issue #590
	{
		span<int> s1 = make_span(arr);

		static span<int> s2;
		s2 = s1;

		CHECK(s1.size() == s2.size());
	}
}

TEST_CASE("from_const_std_array_constructor")
{
	const std::array<int, 4> arr = {1, 2, 3, 4};

	{
		span<const int> s {arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}

	{
		span<const int, 4> s {arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}

//	const std::array<AddressOverloaded, 4> ao_arr {};
//
//	{
//		span<const AddressOverloaded, 4> s {ao_arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(ao_arr.size()));
//		CHECK(s.data() == ao_arr.data());
//	}

	{
		auto get_an_array = []() -> const std::array<int, 4> {return {1, 2, 3, 4};};
		auto take_a_span = [](span<const int> s) {static_cast<void>(s);};
		// try to take a temporary std::array
		take_a_span(get_an_array());
	}

	{
		auto s = make_span(arr);
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}
}

TEST_CASE("from_std_array_const_constructor")
{
	std::array<const int, 4> arr = {1, 2, 3, 4};

	{
		span<const int> s {arr};
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}

	{
		span<const int, 4> s {arr};
//         CHECK(s.size() ==  narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}

	{
		auto s = make_span(arr);
//         CHECK(s.size() == narrow_cast<ptrdiff_t>(arr.size()));
		CHECK(s.data() == arr.data());
	}
}

// TODO: These don't work- but doesn't. And we don't have a Container& constructor, either... and
// if we enable one (lifted from GSL), this still doesn't pass.
TEST_CASE("from_container_constructor" * doctest::skip())
{
	std::vector<int> v = {1, 2, 3};
	const std::vector<int> cv = v;
//
//	{
//		span<int> s {v};
//	//        CHECK(s.size() == narrow_cast<std::ptrdiff_t>(v.size()));
//		CHECK(s.data() == v.data());
//
//		span<const int> cs {v};
// //         CHECK(cs.size() == narrow_cast<std::ptrdiff_t>(v.size()));
//		CHECK(cs.data() == v.data());
//	}
//
//	std::string str = "hello";
//	const std::string cstr = "hello";
//
//	{
//		span<const char> cs {str};
// //         CHECK(cs.size() == narrow_cast<std::ptrdiff_t>(str.size()));
//		CHECK(cs.data() == str.data());
//	}
//
//	{
//		span<const char> cs {cstr};
// //         CHECK(cs.size() == narrow_cast<std::ptrdiff_t>(cstr.size()));
//		CHECK(cs.data() == cstr.data());
//	}
//
//	{
//		auto get_temp_vector = []() -> std::vector<int> {return {};};
//		auto use_span = [](span<const int> s) {static_cast<void>(s);};
//		use_span(get_temp_vector());
//	}
//
//
//	{
//		auto get_temp_string = []() -> std::string {return {};};
//		auto use_span = [](span<const char> s) {static_cast<void>(s);};
//		use_span(get_temp_string());
//	}
//
//	{
//		auto get_temp_string = []() -> const std::string {return {};};
//		auto use_span = [](span<const char> s) {static_cast<void>(s);};
//		use_span(get_temp_string());
//	}
//
//
//	{
//		auto s = make_span(v);
// //         CHECK(s.size() == narrow_cast<std::ptrdiff_t>(v.size()));
//		CHECK(s.data() == v.data());
//
//		auto cs = make_span(cv);
// //         CHECK(cs.size() == narrow_cast<std::ptrdiff_t>(cv.size()));
//		CHECK(cs.data() == cv.data());
//	}
}

TEST_CASE("from_convertible_span_constructor")
{
	{
		struct BaseClass { };
		struct DerivedClass : BaseClass { };

		span<DerivedClass> avd;
		span<const DerivedClass> avcd = avd;
		static_cast<void>(avcd);
	}

}

TEST_CASE("copy_move_and_assignment")
 {
     span<int> s1;
     CHECK(s1.empty());

     int arr[] = {3, 4, 5};

     span<const int> s2 = arr;
     CHECK(s2.size() ==  3);
     CHECK(s2.data() == &arr[0]);

     s2 = s1;
     CHECK(s2.empty());

     auto get_temp_span = [&]() -> span<int> { return {&arr[1], 2}; };
     auto use_span = [&](span<const int> s) {
         CHECK(s.size() ==  2);
         CHECK(s.data() == &arr[1]);
     }; use_span(get_temp_span());

     s1 = get_temp_span();
     CHECK(s1.size() ==  2);
     CHECK(s1.data() == &arr[1]);
 }

TEST_CASE("first")
 {
     int arr[5] = {1, 2, 3, 4, 5};

     {
         span<int, 5> av = arr;
         CHECK(av.first<2>().size() == 2);
         CHECK(av.first(2).size() == 2);
     }

     {
         span<int, 5> av = arr;
         CHECK(av.first<0>().size() == 0);
         CHECK(av.first(0).size() == 0);
     }

     {
         span<int, 5> av = arr;
         CHECK(av.first<5>().size() == 5);
         CHECK(av.first(5).size() == 5);
     }

     {
         span<int, 5> av = arr;
     }

     {
         span<int> av;
         CHECK(av.first<0>().size() == 0);
         CHECK(av.first(0).size() == 0);
     }
 }

TEST_CASE("last")
 {
    std::set_terminate([] {
        std::cerr << "Expected Death. last";
        std::abort();
    });
     int arr[5] = {1, 2, 3, 4, 5};

     {
         span<int, 5> av = arr;
         CHECK(av.last<2>().size() == 2);
         CHECK(av.last(2).size() == 2);
     }

     {
         span<int, 5> av = arr;
         CHECK(av.last<0>().size() == 0);
         CHECK(av.last(0).size() == 0);
     }

     {
         span<int, 5> av = arr;
         CHECK(av.last<5>().size() == 5);
         CHECK(av.last(5).size() == 5);
     }

     {
         span<int, 5> av = arr;
     }

     {
         span<int> av;
         CHECK(av.last<0>().size() == 0);
         CHECK(av.last(0).size() == 0);
     }
 }

TEST_CASE("subspan")
{
	int arr[5] = {1, 2, 3, 4, 5};

	{
		span<int, 5> av = arr;
		CHECK((av.subspan<2, 2>().size()) == 2);
		CHECK(decltype(av.subspan<2, 2>())::extent == 2);
		CHECK(av.subspan(2, 2).size() == 2);
		CHECK(av.subspan(2, 3).size() == 3);
	}

	{
		span<int, 5> av = arr;
		CHECK((av.subspan<0, 0>().size()) == 0);
		CHECK(decltype(av.subspan<0, 0>())::extent == 0);
		CHECK(av.subspan(0, 0).size() == 0);
	}

	{
		span<int, 5> av = arr;
		CHECK((av.subspan<0, 5>().size()) == 5);
		CHECK(decltype(av.subspan<0, 5>())::extent == 5);
		CHECK(av.subspan(0, 5).size() == 5);

	}

	{
		span<int, 5> av = arr;
		CHECK((av.subspan<4, 0>().size()) == 0);
		CHECK(decltype(av.subspan<4, 0>())::extent == 0);
		CHECK(av.subspan(4, 0).size() == 0);
		CHECK(av.subspan(5, 0).size() == 0);
	}

	{
		span<int, 5> av = arr;
		// TODO: This should work without specifying the extent!
		// CHECK(av.subspan<1>().size() == 4);
		// CHECK(decltype(av.subspan<1>())::extent == 4);
		CHECK(av.subspan<1,4>().size() == 4);
		CHECK(decltype(av.subspan<1,4>())::extent == 4);
	}

	{
		span<int> av;
		CHECK((av.subspan<0, 0>().size()) == 0);
		CHECK(decltype(av.subspan<0, 0>())::extent == 0);
		CHECK(av.subspan(0, 0).size() == 0);
	}

	{
		span<int> av;
		CHECK(av.subspan(0).size() == 0);
	}

	{
		span<int> av = arr;
		CHECK(av.subspan(0).size() == 5);
		CHECK(av.subspan(1).size() == 4);
		CHECK(av.subspan(4).size() == 1);
		CHECK(av.subspan(5).size() == 0);
		const auto av2 = av.subspan(1);
		for (int i = 0; i < 4; ++i) CHECK(av2[i] == i + 2);
	}

	{
		span<int, 5> av = arr;
		CHECK(av.subspan(0).size() == 5);
		CHECK(av.subspan(1).size() == 4);
		CHECK(av.subspan(4).size() == 1);
		CHECK(av.subspan(5).size() == 0);
		const auto av2 = av.subspan(1);
		for (int i = 0; i < 4; ++i) CHECK(av2[i] == i + 2);
	}
}


//	We don't have the GSL size optimization...
//	TEST(span_test, size_optimization)
//	{
//	    span<int> s;
//	    CHECK(sizeof(s) == sizeof(int*) + sizeof(ptrdiff_t));
//
//	    span<int, 0> se;
//	    CHECK(sizeof(se) == sizeof(int*));
//	}

} // TEST_SUITE("span-host-side-gsl")

TEST_SUITE("span-device-side") {

TEST_CASE("LWG-3225-constructibility-with-C-array")
{
	cuda::device_t<> device {cuda::device::current::get()};
	cuda::launch(kernels::lwg_3225_constructibility_with_c_array, singleton_grid_config());
	device.synchronize();
}

TEST_CASE("LWG-3225-constructibility-with-kat-array")
{
	cuda::device_t<> device {cuda::device::current::get()};
	cuda::launch(kernels::lwg_3225_constructibility_with_kat_array, singleton_grid_config());
	device.synchronize();
}

TEST_CASE("LWG-3225-constructibility-with-std-array")
{
	cuda::device_t<> device {cuda::device::current::get()};
	cuda::launch(kernels::lwg_3225_constructibility_with_std_array, singleton_grid_config());
	device.synchronize();
}

TEST_CASE("nothrow-constructibility")
{
	cuda::device_t<> device {cuda::device::current::get()};
	cuda::launch(kernels::nothrow_constructibility, singleton_grid_config());
	device.synchronize();
}

} // TEST_SUITE("span-device-side")
