/*
 * Note: The tests in this file involve run-time work on the GPU.
 * The way this is managed is with just a _single_ kernel for any
 * and all possible testcase - which is very generic: It
 * runs arbitrary test-case specific code, which is intended to
 * produce a sequence of values. These values are not necessarily
 * "results" - that depends on what it is you're running - but
 * they're values to then _check_ afterwards on the host side.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"

#include <kat/on_device/collaboration/grid.cuh>
#include <kat/on_device/collaboration/block.cuh>
#include <kat/on_device/collaboration/warp.cuh>
#include <kat/on_device/atomics.cuh>

using std::size_t;

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

namespace klcg = kat::linear_grid::collaborative::grid;
namespace klcb = kat::linear_grid::collaborative::block;
// namespace kcg  = kat::collaborative::grid;
namespace kcb  = kat::collaborative::block;
namespace kcw  = kat::collaborative::warp;

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


// TODO: Take iterator templates rather than pointers
template <typename T, typename F, typename... Is>
void check_results(
	size_t                    num_values_to_check,
	// perhaps add another parameter for specific individual-check details?
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
			<< "Assertion " << std::setw(index_width) << (i+1) << " for testcase " << doctest::current_test_name()
			// << " :\n"
			<< "(" << std::make_tuple(inputs[i]...) << ")"
		;
		auto mismatch_message { ss.str() };
		if (comparison_tolerance_fraction) {
			CHECK_MESSAGE(actual_values[i] ==  tolerance_gadget(expected_value_retriever(i), comparison_tolerance_fraction), mismatch_message);
		}
		else {
			CHECK_MESSAGE(actual_values[i] == expected_value_retriever(i), mismatch_message);
		}
	}
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
	// TODO: Should we check that num_values_to_populate is equal to the number of grid threads?

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

TEST_SUITE("grid-level") {

// Note: Types for instantiation are chosen based on what's actually available in CUDA

TEST_CASE("at_grid_stride")
{
	using checked_value_type = uint32_t;
	// No inputs, nor concrete expected results.

	auto testcase_device_function = [] KAT_DEV (size_t length, checked_value_type* results) {
		auto f_inner = [&] (size_t pos) {
			results[pos] = kat::linear_grid::grid_info::thread::global_index();
		};
		klcg::at_grid_stride(length, f_inner);
	};

	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 2 };
	auto total_num_threads = num_grid_blocks * num_threads_per_block;

	auto expected_value_retriever = [total_num_threads] (size_t pos) {
		// Which thread processes position pos?
		return checked_value_type(pos % total_num_threads);
	};

	auto num_values_to_populate = total_num_threads * 2 + kat::warp_size / 2 - 1;

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("at_block_stride")
{
	using checked_value_type = uint32_t; // The type for number of grids in a thread. Should we typedef that?
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 2 };
	auto  total_num_threads = num_grid_blocks * num_threads_per_block;

	size_t length_to_cover = total_num_threads * 2 + kat::warp_size / 2 - 1;
		// We don't actually create input data, we just need each element in the
		// range 0 ... length_to_cover-1 to be attended to by some thread
		//
		// In this test case, there's a single common range which the whole grid covers
		// (as opposed to block-level or warp-level collaboration)


	auto testcase_device_function = [] KAT_DEV (size_t length, checked_value_type* results) {
		auto f_inner = [&] (size_t pos) {
//			printf("Thread %u in block %u got pos %u of %u\n", threadIdx.x, blockIdx.x, (unsigned) pos, (unsigned) length);
			results[pos] = kat::linear_grid::grid_info::thread::global_index();
		};
		auto serialization_factor =
			length / kat::linear_grid::grid_info::grid::num_threads() + (length % kat::linear_grid::grid_info::grid::num_threads() != 0);
		klcg::at_block_stride(length, f_inner, serialization_factor);
	};

	auto serialization_factor = div_rounding_up(length_to_cover, total_num_threads);
	auto elements_processed_per_block = serialization_factor * num_threads_per_block;

//	std::cout << "length_to_cover = " << length_to_cover << ", num_threads_per_block = " << num_threads_per_block << ", elements_per_block  = " << serialization_factor << '\n';

	auto expected_value_retriever = [=] (size_t pos) {
		// Which thread processes position pos?

		auto processing_block_index = pos / elements_processed_per_block;
		auto processing_thread_index = pos % num_threads_per_block;
			// which is the same as (pos % processing_block_index) % num_threads_per_block
		return checked_value_type(processing_block_index * num_threads_per_block + processing_thread_index);
	};

//	for(int i = 0; i < length_to_cover; i++) {
//		if (i % 10 == 0) { std::cout << '\n' << std::setw(3) << i << ": "; }
//		std::cout << std::setw(3) << expected_value_retriever(i) << " ";
//	}
//	std::cout << "\n\n";

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		length_to_cover,
		num_grid_blocks,
		num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);


}

struct attending_threads_info {
	struct {
		uint32_t grid_size_minus_first, last;
			// We use grid_size_minus_first rather than first, so that
			// zero-initialization would be semantically acceptable
	} extrema;
	uint32_t num;
}; // Note: All of this gets zero-initialized

std::ostream& operator<<(std::ostream& os, const attending_threads_info& ati)
{
	return os << "{ {" << ati.extrema.grid_size_minus_first << ", " << ati.extrema.last << " }, " << ati.num << " }";
}

bool operator==(const attending_threads_info& lhs, const attending_threads_info & rhs)
{
	return
		lhs.extrema.grid_size_minus_first == rhs.extrema.grid_size_minus_first and
		lhs.extrema.last == rhs.extrema.last and
		lhs.num == rhs.num;
}

TEST_CASE("warp_per_input_element::at_grid_stride")
{
	using checked_value_type = attending_threads_info;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 15 };
	auto  total_num_threads = num_grid_blocks * num_threads_per_block;
	auto length_to_cover = total_num_threads / 4 + 1;
		// We don't actually create input data, we just need each element in the
		// range 0 ... length_to_cover-1 to be attended to by some full warp
	auto num_values_to_populate = length_to_cover;

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t length_of_attending_threads_info,
			checked_value_type* attending_threads_info)
		{
			namespace gi = kat::linear_grid::grid_info;
			const auto my_index = gi::thread::global_index();
			auto grid_size_minus_my_index = gi::grid::num_threads() - my_index;
			auto f_inner = [&] (size_t pos) {
//				printf("Thead %d of block %d is handling pos %lu\n", threadIdx.x, blockIdx.x, pos);
				kat::atomic::increment(&attending_threads_info[pos].num);
				kat::atomic::max(&attending_threads_info[pos].extrema.grid_size_minus_first, grid_size_minus_my_index);
				kat::atomic::max(&attending_threads_info[pos].extrema.last, my_index);
			};

			klcg::warp_per_input_element::at_grid_stride(length_to_cover, f_inner);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		// Which threads have handled position pos?

		auto total_num_warps = total_num_threads / kat::warp_size;
		auto modular_pos = pos % total_num_warps;
		uint32_t first_thread_to_handle_element = modular_pos * kat::warp_size;
		uint32_t grid_size_minus_first = total_num_threads - first_thread_to_handle_element;
		uint32_t last = (modular_pos+1) * kat::warp_size - 1;
		uint32_t num = kat::warp_size;
		return attending_threads_info { { grid_size_minus_first, last }, num };
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}


} // TEST_SUITE("grid-level")
