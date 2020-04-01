#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"

#include <kat/on_device/sequence_ops/grid.cuh>
#include <kat/on_device/sequence_ops/block.cuh>
#include <kat/on_device/sequence_ops/warp.cuh>

#include <cuda/api_wrappers.hpp>

#include <limits>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

using std::size_t;
using fake_bool = int8_t; // so as not to have trouble with vector<bool>
static_assert(sizeof(bool) == sizeof(fake_bool), "unexpected size mismatch");


namespace klcg = kat::linear_grid::collaborative::grid;
namespace klcb = kat::linear_grid::collaborative::block;
// namespace kcg  = kat::collaborative::grid;
namespace kcb  = kat::collaborative::block;
namespace kcw  = kat::collaborative::warp;


#if __cplusplus < 201701L
#include <experimental/optional>
template <typename T>
using optional = std::experimental::optional<T>;
#else
template <typename T>
#include <optional>
using optional = std::optional<T>;
#endif

template <typename T>
const auto make_exact_comparison { optional<T>{} };

namespace kcw  = ::kat::collaborative::warp;
namespace klcw = ::kat::linear_grid::collaborative::warp;


std::ostream& operator<<(std::ostream& os, klcw::detail::predicate_computation_length_slack_t ss)
{
	switch(ss) {
	case klcw::detail::predicate_computation_length_slack_t::has_no_slack:
		os << "has_no_slack"; break;
	case klcw::detail::predicate_computation_length_slack_t::may_have_arbitrary_slack:
		os << "may_have_arbitrary_slack"; break;
	case klcw::detail::predicate_computation_length_slack_t::may_have_full_warps_of_slack:
	default:
		os << "may_have_full_warps_of_slack:";
	}
	return os;
}

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


namespace kat {
namespace linear_grid {
namespace collaborative {
namespace warp {

template <typename T>
std::ostream& operator<<(std::ostream& os, search_result_t<T> sr)
{
	if (not sr.is_set()) {
		return os << "(not found)";
	}
	return os << "value " << sr.value << " in lane " << sr.lane_index;
}


template <typename T>
KAT_FHD bool operator==(const search_result_t<T>& lhs, const search_result_t<T>& rhs)
{
	return
		(lhs.lane_index == rhs.lane_index)
		and ( (not lhs.is_set() ) or (lhs.value == rhs.value) );
}

} // namespace warp
} // namespace collaborative
} // namespace linear_grid
} // namespace kat

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


TEST_SUITE("warp-to-grid") {

TEST_CASE("append_to_global_memory")
{
//	template <typename T, typename Size = size_t>
//	KAT_FD void collaborative_append_to_global_memory(
//		T*     __restrict__  global_output,
//		Size*  __restrict__  global_output_length,
//		T*     __restrict__  fragment_to_append,
//		Size   __restrict__  fragment_length)

}

} // TEST_SUITE("warp-to-grid")

TEST_SUITE("block-level - linear grid") {

TEST_CASE("fill")
{
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	auto resolve_fill_value = [] KAT_HD (unsigned block_id) -> checked_value_type {
		constexpr const checked_value_type fill_value_base { 456 };
		return fill_value_base + (block_id + 1) * 10000;
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_block * gi::block::id();
			auto end = start + length_to_cover_per_block;
			auto fill_value = resolve_fill_value(gi::block::id());
			klcb::fill(start, end, fill_value);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		auto processing_block_id = pos / length_to_cover_per_block;
		return resolve_fill_value(processing_block_id);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("fill_n") {
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	auto resolve_fill_value = [] KAT_HD (unsigned block_id) -> checked_value_type {
		constexpr const checked_value_type fill_value_base { 456 };
		return fill_value_base + (block_id + 1) * 10000;
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_block * gi::block::id();
			auto fill_value = resolve_fill_value(gi::block::id());
			klcb::fill_n(start, length_to_cover_per_block, fill_value);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		auto processing_block_id = pos / length_to_cover_per_block;
		return resolve_fill_value(processing_block_id);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("memzero") {
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_block * gi::block::id();
			auto end = start + length_to_cover_per_block;
			klcb::memzero(start, end);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		return checked_value_type(0);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("memzero_n") {
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_block * gi::block::id();
			klcb::memzero_n(start, length_to_cover_per_block);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		return checked_value_type(0);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("transform") {
	using checked_value_type = int32_t;
	using input_value_type = uint8_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> uint8_t { return uint8_t('a' + pos % ('z'-'a' + 1)); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto op = [] KAT_HD (input_value_type x) -> checked_value_type { return -x; };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_block * gi::block::id();
			auto source_end = source_start + length_to_cover_per_block;
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::transform(source_start, source_end, block_target_start, op);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		return op(generator(pos));
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("transform_n") {
	using checked_value_type = int32_t;
	using input_value_type = uint8_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> uint8_t { return uint8_t('a' + pos % ('z'-'a' + 1)); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto op = [] KAT_HD (input_value_type x) -> checked_value_type { return -x; };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_block * gi::block::id();
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::transform_n(source_start, length_to_cover_per_block, block_target_start, op);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		return op(generator(pos));
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("cast_and_copy") {
	using checked_value_type = int32_t;
	using input_value_type = float;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80 + 0.123; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_block * gi::block::id();
			auto source_end = source_start + length_to_cover_per_block;
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::cast_and_copy(source_start, source_end, block_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("cast_and_copy_n") {
	using checked_value_type = int32_t;
	using input_value_type = float;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80 + 0.123; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = input + length_to_cover_per_block * gi::block::id();
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::cast_and_copy_n(start, length_to_cover_per_block, block_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}


TEST_CASE("copy") {
	using checked_value_type = int32_t;
	using input_value_type = checked_value_type;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_block * gi::block::id();
			auto source_end = source_start + length_to_cover_per_block;
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::copy(source_start, source_end, block_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("copy_n") {
	using checked_value_type = int32_t;
	using input_value_type = checked_value_type;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = input + length_to_cover_per_block * gi::block::id();
			auto block_target_start = target + length_to_cover_per_block * gi::block::id();
			klcb::copy_n(start, length_to_cover_per_block, block_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("lookup") {
	using checked_value_type = int32_t;
	using index_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t num_indices_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = num_indices_per_block * num_grid_blocks;

	std::vector<checked_value_type> data = {
		101, 202, 303, 404, 505, 606, 707, 808, 909, 1010
	};

	std::vector<index_type> indices;
	auto generator = [](size_t pos) -> index_type { return (7 * pos) % 10; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(indices), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*       __restrict target,
			const checked_value_type* __restrict data,
			const index_type*         __restrict indices
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto block_indices_start = indices + num_indices_per_block * gi::block::id();
			auto block_target_start = target + num_indices_per_block * gi::block::id();
			klcb::lookup(block_target_start, data, block_indices_start, num_indices_per_block);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return data[generator(pos)];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		data.data(),
		indices.data()
	);
}

TEST_CASE("reduce - all threads obtain result") {
	constexpr const bool all_threads_do_obtain_result { true };
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			target[gi::thread::global_id()] =
				klcb::reduce<
					checked_value_type,
					decltype(plus),
					all_threads_do_obtain_result
				>(thread_input, plus);
		};

	std::vector<checked_value_type> block_sums;
	block_sums.reserve(num_grid_blocks);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_sum = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_sum += make_thread_value(block_id,thread_id);
		}
		block_sums.push_back(block_sum);
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		auto block_id = pos / num_threads_per_block;
		return block_sums[block_id];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("reduce - not all threads obtain result") {
	constexpr const bool not_all_threads_obtain_result { true };
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			target[gi::thread::global_id()] =
				klcb::reduce<checked_value_type, decltype(plus), not_all_threads_obtain_result>(thread_input, plus);
		};

	std::vector<checked_value_type> block_sums;
	block_sums.reserve(num_grid_blocks);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_sum = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_sum += make_thread_value(block_id,thread_id);
		}
		block_sums.push_back(block_sum);
	}

	auto host_results = execute_non_uniform_testcase_on_gpu(
		tag<checked_value_type>{},
		testcase_device_function,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		input.data()
	);

	CHECK(host_results.size() == num_values_to_populate);

	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		auto first_block_thread_glbal_id = block_id * num_threads_per_block;
		CHECK(host_results[first_block_thread_glbal_id] == block_sums[block_id]);
	}
}

TEST_CASE("sum - all threads obtain result") {
	constexpr const bool all_threads_do_obtain_result { true };
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			target[gi::thread::global_id()] =
				klcb::sum<checked_value_type, all_threads_do_obtain_result>(thread_input);
		};

	std::vector<checked_value_type> block_sums;
	block_sums.reserve(num_grid_blocks);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_sum = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_sum += make_thread_value(block_id,thread_id);
		}
		block_sums.push_back(block_sum);
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		auto block_id = pos / num_threads_per_block;
		return block_sums[block_id];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}


TEST_CASE("inclusive scan with specified scratch area") {
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			static __shared__ checked_value_type scratch[kat::warp_size]; // assumes that there are no than warp_size warps per block
			target[gi::thread::global_id()] =
				klcb::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Inclusive>(checked_value_type(thread_input), plus, scratch);
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_scan += make_thread_value(block_id,thread_id);
			scans.push_back(block_scan);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("inclusive scan without specified scratch area") {
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			target[gi::thread::global_id()] =
				klcb::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Inclusive>(checked_value_type(thread_input), plus);
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_scan += make_thread_value(block_id,thread_id);
			scans.push_back(block_scan);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive scan with specified scratch area") {
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			static __shared__ checked_value_type scratch[kat::warp_size]; // assumes that there are no than warp_size warps per block
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			target[gi::thread::global_id()] =
				klcb::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Exclusive>(checked_value_type(thread_input), plus, scratch);
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			scans.push_back(block_scan);
			block_scan += make_thread_value(block_id,thread_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive scan without specified scratch area") {
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			target[gi::thread::global_id()] =
				klcb::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Exclusive>(checked_value_type(thread_input), plus);
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		checked_value_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			scans.push_back(block_scan);
			block_scan += make_thread_value(block_id,thread_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}


TEST_CASE("inclusive scan_and_reduce with specified scratch area") {
	using scan_result_type = int32_t;
	using reduction_result_type = int32_t;
	using checked_value_type = poor_mans_pair<scan_result_type, reduction_result_type >;
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			static __shared__ scan_result_type scratch[kat::warp_size]; // assumes that there are no than warp_size warps per block
			const auto plus = [](scan_result_type& x, scan_result_type y) { x += y; };
			checked_value_type result;
			klcb::scan_and_reduce<scan_result_type, decltype(plus), kat::collaborative::inclusivity_t::Inclusive>(
				scratch, scan_result_type(thread_input), plus, result.first, result.second);
			target[gi::thread::global_id()] = result;
		};

	std::vector<checked_value_type> scans_and_reductions;
	scans_and_reductions.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		scan_result_type block_reduction = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_reduction += make_thread_value(block_id,thread_id);
		}
		scan_result_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_scan += make_thread_value(block_id,thread_id);
			checked_value_type p { block_scan, block_reduction };
			scans_and_reductions.push_back(p);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans_and_reductions[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive scan_and_reduce with specified scratch area") {
	using scan_result_type = int32_t;
	using reduction_result_type = int32_t;
	using checked_value_type = poor_mans_pair<scan_result_type, reduction_result_type >;
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			static __shared__ scan_result_type scratch[kat::warp_size]; // assumes that there are no than warp_size warps per block
			const auto plus = [](scan_result_type& x, scan_result_type y) { x += y; };
			checked_value_type result;
			klcb::scan_and_reduce<scan_result_type, decltype(plus), kat::collaborative::inclusivity_t::Exclusive>(
				scratch, scan_result_type(thread_input), plus, result.first, result.second);
			target[gi::thread::global_id()] = result;
		};

	std::vector<checked_value_type> scans_and_reductions;
	scans_and_reductions.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		scan_result_type block_reduction = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_reduction += make_thread_value(block_id,thread_id);
		}
		scan_result_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			checked_value_type p { block_scan, block_reduction };
			scans_and_reductions.push_back(p);
			block_scan += make_thread_value(block_id,thread_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans_and_reductions[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}


TEST_CASE("inclusive scan_and_reduce with specified scratch area") {
	using scan_result_type = int32_t;
	using reduction_result_type = int32_t;
	using checked_value_type = poor_mans_pair<scan_result_type, reduction_result_type >;
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			static __shared__ scan_result_type scratch[kat::warp_size]; // assumes that there are no than warp_size warps per block
			const auto plus = [](scan_result_type& x, scan_result_type y) { x += y; };
			checked_value_type result;
			klcb::scan_and_reduce<scan_result_type, decltype(plus), kat::collaborative::inclusivity_t::Inclusive>(
				scratch, scan_result_type(thread_input), plus, result.first, result.second);
			target[gi::thread::global_id()] = result;
		};

	std::vector<checked_value_type> scans_and_reductions;
	scans_and_reductions.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		scan_result_type block_reduction = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_reduction += make_thread_value(block_id,thread_id);
		}
		scan_result_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_scan += make_thread_value(block_id,thread_id);
			checked_value_type p { block_scan, block_reduction };
			scans_and_reductions.push_back(p);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans_and_reductions[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive scan_and_reduce without specified scratch area") {
	using scan_result_type = int32_t;
	using reduction_result_type = int32_t;
	using checked_value_type = poor_mans_pair<scan_result_type, reduction_result_type >;
	using input_value_type = int16_t;
	cuda::grid::dimension_t num_grid_blocks { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t block_id,
		cuda::grid::block_dimension_t thread_id) -> input_value_type
		{ return ((1+block_id) * 1000) + (10 + 10 * thread_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto block_id = pos / num_threads_per_block;
			auto thread_id = pos % num_threads_per_block;
			pos++;
			return make_thread_value(block_id, thread_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](scan_result_type& x, scan_result_type y) { x += y; };
			checked_value_type result;
			klcb::scan_and_reduce<scan_result_type, decltype(plus), kat::collaborative::inclusivity_t::Exclusive>(
				scan_result_type(thread_input), plus, result.first, result.second);
			target[gi::thread::global_id()] = result;
		};

	std::vector<checked_value_type> scans_and_reductions;
	scans_and_reductions.reserve(num_values_to_populate);
	for(int block_id = 0; block_id < num_grid_blocks; block_id++ ) {
		scan_result_type block_reduction = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			block_reduction += make_thread_value(block_id,thread_id);
		}
		scan_result_type block_scan = 0;
		for(int thread_id = 0; thread_id < num_threads_per_block; thread_id++ ) {
			checked_value_type p { block_scan, block_reduction };
			scans_and_reductions.push_back(p);
			block_scan += make_thread_value(block_id,thread_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans_and_reductions[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("elementwise accumulate")
{
	using checked_value_type = int32_t;
	using input_value_type = checked_value_type;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 3 };
	size_t length_to_cover_per_block { num_threads_per_block * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	std::vector<checked_value_type> input_dest;
	auto dest_generator = [](size_t pos) -> checked_value_type { return 1000 + pos % 8000; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input_dest), num_values_to_populate, [&]() { return dest_generator(pos++); });

	std::vector<input_value_type> input_src;
	auto src_generator = [](size_t pos) -> input_value_type { return 10 + pos % 80; };
	pos = 0;
	std::generate_n(std::back_inserter(input_src), num_values_to_populate, [&]() { return src_generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*        __restrict result,
			const checked_value_type*  __restrict input_dest,
			const input_value_type*    __restrict input_src
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto block_result = result + length_to_cover_per_block * gi::block::id();
			auto block_dest = input_dest + length_to_cover_per_block * gi::block::id();
			klcb::copy_n(block_dest, length_to_cover_per_block, block_result);
			auto block_src = input_src + length_to_cover_per_block * gi::block::id();
			const auto plus = [](checked_value_type& x, input_value_type y) { x += y; };
			// So, you might think we should be accumulating into _dest - but we can't do that since it's
			// read-only. So first let's make a copy of it into the result column, then accumulate there.
			klcb::elementwise_accumulate(plus, block_result, block_src, length_to_cover_per_block);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return dest_generator(pos) + src_generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input_dest.data(),
		input_src.data()
	);
}


} // TEST_SUITE("block-level - linear grid")

TEST_SUITE("warp-level") {

TEST_CASE("reduce")
{
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1+global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
//			printf("Warp %u Lane %2u input is %4d\n", (unsigned) gi::warp::global_id(), (unsigned) gi::lane::id(), (int) thread_input);
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			auto warp_reduction_result = kcw::reduce(thread_input, plus);
//			printf("Warp %u reduction result is %d\n", (unsigned) gi::warp::global_id(), (int) warp_reduction_result);
			target[gi::thread::global_id()] = warp_reduction_result;
		};

	std::vector<checked_value_type> warp_sums;
	warp_sums.reserve(num_grid_blocks);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_sum = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			warp_sum += make_thread_value(global_warp_id, lane_id);
		}
		warp_sums.push_back(warp_sum);
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		auto global_warp_id = pos / kat::warp_size;
		return warp_sums[global_warp_id];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("sum")
{
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1+global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			auto warp_sum = kcw::sum(thread_input);
			target[gi::thread::global_id()] = warp_sum;
		};

	std::vector<checked_value_type> warp_sums;
	warp_sums.reserve(num_grid_blocks);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_sum = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			warp_sum += make_thread_value(global_warp_id, lane_id);
		}
		warp_sums.push_back(warp_sum);
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		auto global_warp_id = pos / kat::warp_size;
		return warp_sums[global_warp_id];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("inclusive scan")
{
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1 + global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			auto warp_scan_result = kcw::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Inclusive>(thread_input, plus);
			target[gi::thread::global_id()] = warp_scan_result;
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_scan = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			warp_scan += make_thread_value(global_warp_id, lane_id);
			scans.push_back(warp_scan);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive scan")
{
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1 + global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			const auto plus = [](checked_value_type& x, checked_value_type y) { x += y; };
			auto warp_scan_result = kcw::scan<checked_value_type, decltype(plus), kat::collaborative::inclusivity_t::Exclusive>(thread_input, plus);
			target[gi::thread::global_id()] = warp_scan_result;
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_scan = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			scans.push_back(warp_scan);
			warp_scan += make_thread_value(global_warp_id, lane_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("exclusive_prefix_sum")
{
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1 + global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			auto warp_exclusive_prefix_sum = kcw::exclusive_prefix_sum(thread_input);
			target[gi::thread::global_id()] = warp_exclusive_prefix_sum;
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_scan = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			scans.push_back(warp_scan);
			warp_scan += make_thread_value(global_warp_id, lane_id);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("prefix_sum") {
	using checked_value_type = int32_t; // TODO: Try with some other types, e.g. int64_t
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
		// TODO: What about non-full warps?

	auto num_values_to_populate = num_threads_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	input.reserve(num_values_to_populate);
	auto make_thread_value = [](
		cuda::grid::dimension_t global_warp_id,
		cuda::grid::block_dimension_t lane_id) -> checked_value_type
		{ return ((1 + global_warp_id) * 1000) + (10 + 10 * lane_id % 9); };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate,
		[&]() {
			auto global_warp_id = pos / kat::warp_size;
			auto lane_id = pos % kat::warp_size;
			pos++;
			return make_thread_value(global_warp_id, lane_id);
		}
	);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const checked_value_type* __restrict input
			)
		{
			// Note: Every thread will set a target value, but there is still just one reduction result
			// per block. In this variant of reduce, all block threads must obtain the result.

			namespace gi = kat::linear_grid::grid_info;
			auto thread_input = input[gi::thread::global_id()];
			auto warp_prefix_sum = kcw::prefix_sum(thread_input);
			target[gi::thread::global_id()] = warp_prefix_sum;
		};

	std::vector<checked_value_type> scans;
	scans.reserve(num_values_to_populate);
	for(int global_warp_id = 0; global_warp_id < num_grid_blocks * num_warps_per_block; global_warp_id++ ) {
		checked_value_type warp_scan = 0;
		for(int lane_id = 0; lane_id < kat::warp_size; lane_id++ ) {
			warp_scan += make_thread_value(global_warp_id, lane_id);
			scans.push_back(warp_scan);
		}
	}

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return scans[pos];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}


TEST_CASE("cast_and_copy_n")
{
	using checked_value_type = int32_t;
	using input_value_type = float;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80 + 0.123; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_warp * gi::warp::global_id();
			auto warp_target_start = target + length_to_cover_per_warp * gi::warp::global_id();
			kcw::cast_and_copy_n(source_start, length_to_cover_per_warp, warp_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("cast_and_copy")
{
	using checked_value_type = int32_t;
	using input_value_type = float;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<input_value_type> input;
	auto generator = [](size_t pos) -> input_value_type { return 10 + pos % 80 + 0.123; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* __restrict target,
			const input_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_warp * gi::warp::global_id();
			auto source_end = source_start + length_to_cover_per_warp;
			auto warp_target_start = target + length_to_cover_per_warp * gi::warp::global_id();
			kcw::cast_and_copy(source_start, source_end, warp_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("copy_n")
{
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	auto generator = [](size_t pos) -> checked_value_type { return 10 + pos % 80; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*     __restrict target,
			const checked_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_warp * gi::warp::global_id();
			auto warp_target_start = target + length_to_cover_per_warp * gi::warp::global_id();
			kcw::copy_n(source_start, length_to_cover_per_warp, warp_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("copy")
{
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<checked_value_type> input;
	auto generator = [](size_t pos) -> checked_value_type { return 10 + pos % 80; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*     __restrict target,
			const checked_value_type* __restrict input
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto source_start = input + length_to_cover_per_warp * gi::warp::global_id();
			auto source_end = source_start + length_to_cover_per_warp;
			auto warp_target_start = target + length_to_cover_per_warp * gi::warp::global_id();
			kcw::copy(source_start, source_end, warp_target_start);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input.data()
	);
}

TEST_CASE("fill")
{
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	auto resolve_fill_value = [] KAT_HD (unsigned warp_id) -> checked_value_type {
		constexpr const checked_value_type fill_value_base { 456 };
		return fill_value_base + (warp_id + 1) * 10000;
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_warp * gi::warp::global_id();
			auto end = start + length_to_cover_per_warp;
			auto fill_value = resolve_fill_value(gi::warp::global_id());
			kcw::fill(start, end, fill_value);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		auto processing_warp_global_id = pos / length_to_cover_per_warp;
		return resolve_fill_value(processing_warp_global_id);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("fill_n") {
	using checked_value_type = int32_t;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	auto resolve_fill_value = [] KAT_HD (unsigned warp_id) -> checked_value_type {
		constexpr const checked_value_type fill_value_base { 456 };
		return fill_value_base + (warp_id + 1) * 10000;
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* buffer_to_fill_by_entire_grid )
		{
			namespace gi = kat::linear_grid::grid_info;
			auto start = buffer_to_fill_by_entire_grid + length_to_cover_per_warp * gi::warp::global_id();
			auto fill_value = resolve_fill_value(gi::warp::global_id());
			kcw::fill_n(start, length_to_cover_per_warp, fill_value);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		auto processing_warp_global_id = pos / length_to_cover_per_warp;
		return resolve_fill_value(processing_warp_global_id);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("lookup")
{
	using checked_value_type = int32_t;
	using index_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t num_indices_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = num_indices_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<checked_value_type> data = {
		101, 202, 303, 404, 505, 606, 707, 808, 909, 1010
	};

	std::vector<index_type> indices;
	auto generator = [](size_t pos) -> index_type { return (7 * pos) % 10; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(indices), num_values_to_populate, [&]() { return generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*       __restrict target,
			const checked_value_type* __restrict data,
			const index_type*         __restrict indices
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto warp_indices_start = indices + num_indices_per_warp * gi::warp::global_id();
			auto warp_target_start = target + num_indices_per_warp * gi::warp::global_id();
			kcw::lookup(warp_target_start, data, warp_indices_start, num_indices_per_warp);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return data[generator(pos)];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		data.data(),
		indices.data()
	);

}

TEST_CASE("elementwise accumulate")
{
	using checked_value_type = int32_t;
	using input_value_type = checked_value_type;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_warps_per_block { 3 };
	cuda::grid::block_dimension_t num_threads_per_block { num_warps_per_block * kat::warp_size };
	size_t length_to_cover_per_warp { kat::warp_size * 2 + 7 };

	auto num_values_to_populate = length_to_cover_per_warp * num_warps_per_block * num_grid_blocks;

	std::vector<checked_value_type> input_dest;
	auto dest_generator = [](size_t pos) -> checked_value_type { return 1000 + pos % 8000; };
	size_t pos = 0;
	std::generate_n(std::back_inserter(input_dest), num_values_to_populate, [&]() { return dest_generator(pos++); });

	std::vector<input_value_type> input_src;
	auto src_generator = [](size_t pos) -> input_value_type { return 10 + pos % 80; };
	pos = 0;
	std::generate_n(std::back_inserter(input_src), num_values_to_populate, [&]() { return src_generator(pos++); });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type*        __restrict result,
			const checked_value_type*  __restrict input_dest,
			const input_value_type*    __restrict input_src
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto warp_result = result + length_to_cover_per_warp * gi::warp::global_id();
			auto warp_dest = input_dest + length_to_cover_per_warp * gi::warp::global_id();
			kcw::copy_n(warp_dest, length_to_cover_per_warp, warp_result);
			auto warp_src = input_src + length_to_cover_per_warp * gi::warp::global_id();
			const auto plus = [](checked_value_type& x, input_value_type y) { x += y; };
			// So, you might think we should be accumulating into _dest - but we can't do that since it's
			// read-only. So first let's make a copy of it into the result column, then accumulate there.
			kcw::elementwise_accumulate(plus, warp_result, warp_src, length_to_cover_per_warp);
		};

	auto expected_value_retriever = [=] (size_t pos) -> checked_value_type {
		return dest_generator(pos) + src_generator(pos);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>,
		input_dest.data(),
		input_src.data()
	);
}

} // TEST_SUITE("warp-level")
