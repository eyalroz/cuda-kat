#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"

#include <kat/ranges.hpp>
#include <kat/on_device/ranges.cuh>

#include <cuda/api_wrappers.hpp>

#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

using std::size_t;
using fake_bool = int8_t; // so as not to have trouble with vector<bool>
static_assert(sizeof(bool) == sizeof(fake_bool), "unexpected size mismatch");



template <typename T>
const auto make_exact_comparison { optional<T>{} };

namespace kernels {

template <typename F, typename T, typename... Is>
__global__ void execute_testcase(
	F                           testcase_device_function,
	size_t                      num_values_to_populate,
	T*         __restrict__     values_to_populate,
	const Is*  __restrict__ ... inputs
	)
{
	testcase_device_function(num_values_to_populate, values_to_populate, inputs...);
}

} // namespace kernels


template <typename T>
std::size_t set_width_for_up_to(T max)
{
//	assert(std::is_integral<I>::value, "Only integer types supported for now");
	std::stringstream ss;
	ss << std::dec << max;
	return ss.str().length();
}

namespace detail {

template <typename T>
auto tolerance_gadget(std::true_type, T x, optional<T> tolerance) {
	auto eps = tolerance.value_or(0);
	return doctest::Approx(x).epsilon(eps);
}


template <typename T>
T tolerance_gadget(std::false_type, T x, optional<T>) { return x; }

} // namespace detail

template <typename T>
auto tolerance_gadget(T x, optional<T> tolerance)
{
	constexpr const auto is_arithmetic = std::is_arithmetic< std::decay_t<T> >::value;
	return
		detail::tolerance_gadget(std::integral_constant<bool, is_arithmetic>{}, x, tolerance);
}

template <typename T, typename F, typename... Is>
void check_results(
	std::string               title,
	size_t                    num_values_to_check,
	const T*  __restrict__    actual_values,
	F                         expected_value_retriever,
	optional<T>               comparison_tolerance_fraction,
	const Is* __restrict__... inputs)
{
	std::stringstream ss;
	auto index_width = set_width_for_up_to(num_values_to_check);

	// TODO: Consider using the maximum/minimum result values to set field widths.

	for(size_t i = 0; i < num_values_to_check; i++) {
		ss.str("");
		ss
			<< "Assertion " << std::setw(index_width) << (i+1) << " for " << title
			// << " :\n"
			<< "(" << std::make_tuple(inputs[i]...) << ")"
		;
		auto mismatch_message { ss.str() };
		if (comparison_tolerance_fraction) {
			const auto& actual = actual_values[i];
			const auto expected = tolerance_gadget(expected_value_retriever(i), comparison_tolerance_fraction);
			CHECK_MESSAGE(actual == expected, mismatch_message);
		}
		else {
			const auto& ev = expected_value_retriever(i);
			const auto& actual = actual_values[i];
			const auto expected = expected_value_retriever(i);
			CHECK_MESSAGE(actual == expected, mismatch_message);
		}
	}
}

template <typename T, typename F, typename... Is>
void check_results(
	size_t                    num_values_to_check,
	const T*  __restrict__    actual_values,
	F                         expected_value_retriever,
	optional<T>               comparison_tolerance_fraction,
	const Is* __restrict__... inputs)
{
	return check_results(
		std::string("testcase ") + doctest::current_test_name(),
		num_values_to_check,
		actual_values,
		expected_value_retriever,
		comparison_tolerance_fraction,
		inputs...);
}


template <typename T>
struct tag {};

/**
 * @brief Executes a testcase intended to make certain checks using a GPU kernel
 * which produces the values to check for.
 *
 * @note The actual checks are eventually conducted on the host side, since doctest
 * code can't actually do anything useful on the GPU. So on the GPU side we "merely"
 * compute the values to check and let the test logic peform the actual comparison later
 * on.
 */
template <typename F, typename K, typename T, typename... Is, size_t... Indices>
auto execute_testcase_on_gpu(
	tag<T>,
	std::index_sequence<Indices...>,
	K                                testcase_kernel,
	F                                testcase_device_function,
	cuda::launch_configuration_t     launch_config,
	size_t                           num_values_to_populate,
	Is* __restrict__ ...             inputs)
{
	cuda::device_t device { cuda::device::current::get() };
	auto device_side_results { cuda::memory::device::make_unique<T[]>(device, num_values_to_populate) };
	cuda::memory::device::zero(device_side_results.get(), num_values_to_populate * sizeof(T)); // just to be on the safe side
	auto host_side_results { std::vector<T>(num_values_to_populate) };

	auto make_device_side_input = [&device, num_values_to_populate](auto input, size_t n) {
		using input_type = std::remove_reference_t<decltype(*input)>;
		auto device_side_input = cuda::memory::device::make_unique<input_type[]>(device, n);
		cuda::memory::copy(device_side_input.get(), input, num_values_to_populate * sizeof(input_type));
		return std::move(device_side_input);
	};
	auto device_side_inputs = std::make_tuple( make_device_side_input(inputs, num_values_to_populate)... );
	ignore(make_device_side_input); // for the case of no inputs
	ignore(device_side_inputs); // for the case of no inputs

	cuda::launch(
		testcase_kernel,
		launch_config,
		testcase_device_function,
		num_values_to_populate,
		device_side_results.get(),
		std::get<Indices>(device_side_inputs).get()... );

	cuda::memory::copy(host_side_results.data(), device_side_results.get(), sizeof(T) * num_values_to_populate);

	return host_side_results;
}

template <typename F, typename ExpectedResultRetriever, typename T, typename... Is>
void execute_non_uniform_testcase_on_gpu_and_check(
	F                               testcase_device_function,
	ExpectedResultRetriever         expected_value_retriever,
	size_t                          num_values_to_populate,
	cuda::grid::dimensions_t        grid_dimensions,
	cuda::grid::block_dimensions_t  block_dimensions,
	optional<T>                     comparison_tolerance_fraction,
	Is* __restrict__ ...            inputs)
{
	auto launch_config { cuda::make_launch_config(grid_dimensions, block_dimensions) };

	auto host_side_results = execute_testcase_on_gpu(
		tag<T>{},
		typename std::make_index_sequence<sizeof...(Is)> {},
		kernels::execute_testcase<F, T, Is...>,
		testcase_device_function,
		launch_config,
		num_values_to_populate,
		inputs...
	);

	check_results (
		num_values_to_populate,
		// perhaps add another parameter for specific testcase details?
		host_side_results.data(),
		expected_value_retriever,
		comparison_tolerance_fraction,
		inputs...);
}

template <typename F, typename T, typename... Is>
auto execute_non_uniform_testcase_on_gpu(
	tag<T>,
	F                               testcase_device_function,
	size_t                          num_values_to_populate,
	cuda::grid::dimensions_t        grid_dimensions,
	cuda::grid::block_dimensions_t  block_dimensions,
	Is* __restrict__ ...            inputs)
{
	auto launch_config { cuda::make_launch_config(grid_dimensions, block_dimensions) };

	return execute_testcase_on_gpu(
		tag<T>{},
		typename std::make_index_sequence<sizeof...(Is)> {},
		kernels::execute_testcase<F, T, Is...>,
		testcase_device_function,
		launch_config,
		num_values_to_populate,
		inputs...
	);
}

template <typename T1, typename T2>
struct poor_mans_pair { T1 first; T2 second; };

template <typename T1, typename T2>
bool operator==(const poor_mans_pair<T1, T2>& lhs, const poor_mans_pair<T1, T2>& rhs)
{
	return lhs.first == rhs.first and lhs.second == rhs.second;
}


TEST_SUITE("host-side") {

TEST_CASE("irange coverage")
{
	using tc_type = int;
	constexpr const auto a { 3 };
	constexpr const auto b { 15 };

	std::vector<tc_type> expected;
	std::vector<tc_type> with_kat_range;
	for(auto i : kat::irange(a, b)) {
		with_kat_range.push_back(i);
	}
	for(auto i = a; i < b; i ++) {
		expected.push_back(i);
	}
	CHECK(with_kat_range.size() == expected.size());
	for(auto i = 0; i < expected.size(); i ++) {
		CHECK(with_kat_range[i] == expected[i]);
	}
}

TEST_CASE("zero-based irange coverage")
{
	using tc_type = int;
	constexpr const auto b { 15 };

	std::vector<tc_type> expected;
	std::vector<tc_type> with_kat_range;
	for(auto i : kat::irange(b)) {
		with_kat_range.push_back(i);
	}
	for(auto i = 0; i < b; i ++) {
		expected.push_back(i);
	}
	CHECK(with_kat_range.size() == expected.size());
	for(auto i = 0; i < expected.size(); i ++) {
		CHECK(with_kat_range[i] == expected[i]);
	}
}

TEST_CASE("strided irange coverage")
{
	using tc_type = int;
	constexpr const auto a { 4 };
	constexpr const auto b { 15 };
	constexpr const auto stride { 3 };

	std::vector<tc_type> expected;
	std::vector<tc_type> with_kat_range;
	for(auto i : kat::strided_irange(a, b, stride)) {
		with_kat_range.push_back(i);
	}
	for(auto i = a; i < b; i += stride) {
		expected.push_back(i);
	}
	CHECK(with_kat_range.size() == expected.size());
	for(auto i = 0; i < expected.size(); i ++) {
		CHECK(with_kat_range[i] == expected[i]);
	}
}

TEST_CASE("zero-based strided irange coverage")
{
	using tc_type = int;
	constexpr const auto b { 15 };
	constexpr const auto stride { 3 };

	std::vector<tc_type> expected;
	std::vector<tc_type> with_kat_range;
	for(auto i : kat::strided_irange(b, stride)) {
		with_kat_range.push_back(i);
	}
	for(auto i = 0; i < b; i += stride) {
		expected.push_back(i);
	}
	CHECK(with_kat_range.size() == expected.size());
	for(auto i = 0; i < expected.size(); i ++) {
		CHECK(with_kat_range[i] == expected[i]);
	}
}

} // TEST_SUITE("host-side")
