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

template <typename T>
const auto make_exact_comparison { optional<T>{} };

namespace klcg = kat::linear_grid::grid;
namespace klcb = kat::linear_grid::block;
// namespace kcg  = kat::grid;
namespace kcb  = kat::block;
namespace kcw  = kat::warp;

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
void execute_non_uniform_builtin_testcase_on_gpu_and_check(
	F                               testcase_device_function,
	ExpectedResultRetriever         expected_value_retriever,
	size_t                          num_values_to_populate,
	cuda::grid::dimensions_t        grid_dimensions,
	cuda::grid::block_dimensions_t  block_dimensions,
	optional<T>                     comparison_tolerance_fraction,
	Is* __restrict__ ...            inputs)
{
	static_assert(
		::std::is_trivially_copy_constructible<cuda::detail_::kernel_parameter_decay_t<F>>::value,
		"testcase_device_function must have a trivially copyable (decayed) type.");
	static_assert(
		::std::is_trivially_copy_constructible<cuda::detail_::kernel_parameter_decay_t<ExpectedResultRetriever>>::value,
		"ExpectedResultRetriever must be a trivially copyable (decayed) type.");

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


TEST_SUITE("block-level - linear grid") {

TEST_CASE("at_block_stride")
{
	using checked_value_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 2 };
//	auto total_num_threads = num_grid_blocks * num_threads_per_block;
	size_t length_to_cover_per_block { 271 };
		// We don't actually create input data, we just need each element in the
		// range 0 ... length_to_cover-1 to be attended to.
		//
		// In this test case - it's 0 ... length_to_cover-1 attended to by _each_ of the blocks.
		// In a real life kernel it might be, say num_grid_blocks * length_to_cover elements to
		// process.

	auto num_values_to_populate = length_to_cover_per_block * num_grid_blocks;

	auto testcase_device_function =
		[length_to_cover_per_block] KAT_DEV (
			size_t num_grid_threads,
			checked_value_type* pos_attendent_thread_indices
		)
		{
			namespace gi = kat::linear_grid;
			auto offset_into_attendant_array = length_to_cover_per_block * gi::block::id();
			auto f_inner = [&] (size_t pos) {
				pos_attendent_thread_indices[offset_into_attendant_array + pos] = gi::thread::index_in_grid();
			};
			klcb::at_block_stride(length_to_cover_per_block, f_inner);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		// Which thread processes position pos?

		auto intra_block_pos = pos % length_to_cover_per_block;
		auto processing_block_id = pos / length_to_cover_per_block;
		auto processing_thread_index = intra_block_pos % num_threads_per_block;
		return checked_value_type(processing_thread_index  + processing_block_id * num_threads_per_block);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);

}

TEST_CASE("share_per_warp_data - specific writer lane")
{
	using datum_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
	auto num_warps_per_block = num_threads_per_block / kat::warp_size;

	auto num_values_to_populate = num_warps_per_block * num_grid_blocks;

	auto make_warp_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t warp_id_within_block)
		{
			return datum_type{(warp_id_within_block + 1) + (block_id + 1) * 10000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* warp_data_for_all_blocks
		)
		{
			namespace gi = kat::linear_grid;
			datum_type thread_datum { make_warp_datum(gi::block::id(), gi::warp::id()) };
				// same for all threads in warp!
			constexpr auto max_possible_num_warps_per_block = 32; // Note: Important assumption here...
			__shared__ datum_type block_warps_data [max_possible_num_warps_per_block];
			constexpr const auto writing_lane_index = 3u; // just for kicks
			klcb::share_per_warp_data(thread_datum, block_warps_data, writing_lane_index);
			// We've run the synchronized variant, so no need for extra sync

			if (gi::thread::is_first_in_block()) {

				// Now we're populating what's going to be checked outside the kernel.
				auto warp_data_for_this_block = warp_data_for_all_blocks + gi::block::id() * num_warps_per_block;
				for(int i = 0; i < num_warps_per_block; i++) {
					warp_data_for_this_block[i] = block_warps_data[i];
				}
			}
		};

	auto expected_value_retriever = [=] (size_t i) {
		auto relevant_block_id = i / num_warps_per_block;
		auto warp_id_within_block = i % num_warps_per_block;
		return make_warp_datum(relevant_block_id, warp_id_within_block);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks,
		num_threads_per_block,
		make_exact_comparison<datum_type>
	);
}

TEST_CASE("share_per_warp_data - inspecific writer lane")
{
	using datum_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
	auto num_warps_per_block = num_threads_per_block / kat::warp_size;

	auto num_values_to_populate = num_warps_per_block * num_grid_blocks;

	auto make_warp_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t warp_id_within_block)
		{
			return datum_type{(warp_id_within_block + 1) + (block_id + 1) * 10000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* warp_data_for_all_blocks
		)
		{
			namespace gi = kat::linear_grid;
			datum_type thread_datum { make_warp_datum(gi::block::id(), gi::warp::id()) };
				// same for all threads in warp!
			constexpr auto max_possible_num_warps_per_block = 32; // Note: Important assumption here...
			__shared__ datum_type warp_data [max_possible_num_warps_per_block];
			klcb::share_per_warp_data(thread_datum, warp_data);
			// We've run the synchronized variant, so no need for extra sync

			if (gi::thread::is_first_in_block()) {
				// Now we're populating what's going to be checked outside the kernel.
				auto warp_data_for_this_block = warp_data_for_all_blocks + gi::block::id() * num_warps_per_block;
				for(int i = 0; i < num_warps_per_block; i++) {
					warp_data_for_this_block[i] = warp_data[i];
				}
			}
		};

	auto expected_value_retriever = [=] (size_t i) {
		auto relevant_block_id = i / num_warps_per_block;
		auto warp_id_within_block = i % num_warps_per_block;
		return make_warp_datum(relevant_block_id, warp_id_within_block);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks,
		num_threads_per_block,
		make_exact_comparison<datum_type>
	);
}

TEST_CASE("get_from_thread")
{
	using datum_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	constexpr const cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
	auto num_total_threads = num_threads_per_block * num_grid_blocks;
	auto num_values_to_populate = num_total_threads;

//	static const auto make_thread_datum =
//		[] KAT_HD (
//			cuda::grid::dimension_t block_id,
//			cuda::grid::block_dimension_t thread_index)
//		{
//			return datum_type{(thread_index + 1) + (block_id + 1) * 10000};
//		};
//
//	static const auto make_source_thread_index =
//		[] KAT_HD (cuda::grid::dimension_t block_id)
//		{
//			return unsigned{(block_id + 1) * 11 % num_threads_per_block};
//		};

	auto testcase_device_function =
		[] KAT_DEV (
			size_t,
			datum_type* thread_obtained_values
		)
		{
			// Sorry for the code duplication - need to avoid capturing. In C++17,
			// make the lambdas outside constexpr as well, and you should be able
			// to remove them here
			auto make_source_thread_index =
			[] KAT_HD (cuda::grid::dimension_t block_id)
			{
				constexpr const cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
				return unsigned{(block_id + 1) * 11 % num_threads_per_block};
			};

			auto make_thread_datum =
				[] KAT_HD (
				cuda::grid::dimension_t block_id,
				cuda::grid::block_dimension_t thread_index)
				{
					return datum_type{(thread_index + 1) + (block_id + 1) * 10000};
				};

			namespace gi = kat::linear_grid;
			datum_type thread_datum { make_thread_datum(gi::block::id(), gi::thread::id()) };
			auto source_thread_index { make_source_thread_index(gi::block::id()) };
			auto obtained_value { klcb::get_from_thread(thread_datum, source_thread_index) };
			// We've run the synchronized variant, so no need for extra sync

			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [] (size_t global_thread_index) {
		// Sorry for the code duplication - need to avoid capturing. In C++17,
		// make the lambdas outside constexpr as well, and you should be able
		// to remove them here
		auto make_source_thread_index =
			[] KAT_HD (cuda::grid::dimension_t block_id)
			{
				constexpr const cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
				return unsigned{(block_id + 1) * 11 % num_threads_per_block};
			};

		auto make_thread_datum =
			[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t thread_index)
			{
				return datum_type{(thread_index + 1) + (block_id + 1) * 10000};
			};


		auto block_id { global_thread_index / num_threads_per_block };
		auto source_thread_index { make_source_thread_index(block_id) };
		return make_thread_datum(block_id, source_thread_index);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks,
		num_threads_per_block,
		make_exact_comparison<datum_type>
	);
}


TEST_CASE("get_from_first_thread")
{
	using datum_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 2 };
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 7 };
	auto num_total_threads = num_threads_per_block * num_grid_blocks;
	auto num_values_to_populate = num_total_threads;

	auto make_thread_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t thread_index)
		{
			return datum_type{(thread_index + 1) + (block_id + 1) * 10000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* thread_obtained_values
		)
		{
			namespace gi = kat::linear_grid;
			datum_type thread_datum { make_thread_datum(gi::block::id(), gi::thread::id()) };
			auto obtained_value { klcb::get_from_first_thread(thread_datum) };
			// We've run the synchronized variant, so no need for extra sync

			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_index) {
		auto block_id { global_thread_index / num_threads_per_block };
		auto source_thread_index { 0 };
		return make_thread_datum(block_id, source_thread_index);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		num_grid_blocks,
		num_threads_per_block,
		make_exact_comparison<datum_type>
	);
}


TEST_CASE("barrier")
{
	// Note: Not much to test with the barrier() collaboration primitive, seeing how there could be "natural barrier'ing" even without the call
}

} // TEST_SUITE("block-level - linear grid")


TEST_SUITE("block-level - general") {

TEST_CASE("share_per_warp_data - specific writer lane")
{
	using datum_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 2, 2 };
	cuda::grid::block_dimensions_t block_dimensions { kat::warp_size, 3, 3 };
	auto num_warps_per_block = block_dimensions.volume() / kat::warp_size;

	auto num_values_to_populate = num_warps_per_block * grid_dimensions.volume();

	auto make_warp_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t warp_id_within_block)
		{
			return datum_type{(warp_id_within_block + 1) + (block_id + 1) * 1000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* warp_data_for_all_blocks
		)
		{
			namespace gi = kat;
			datum_type thread_datum { make_warp_datum(gi::block::id(), gi::warp::id()) };
			constexpr auto max_possible_num_warps_per_block = 32; // Note: Important assumption here...
			__shared__ datum_type warp_data [max_possible_num_warps_per_block];

			constexpr const auto writing_lane_index = 3u;
			kcb::share_per_warp_data(thread_datum, warp_data, writing_lane_index);
			// We've run the synchronized variant, so no need for extra sync

			if (gi::thread::is_first_in_block()) {
				// Now we're populating what's going to be checked outside the kernel.
				auto warp_data_for_this_block = warp_data_for_all_blocks + gi::block::id() * num_warps_per_block;
				for(int i = 0; i < num_warps_per_block; i++) {
					warp_data_for_this_block[i] = warp_data[i];
				}
			}
		};

	auto expected_value_retriever = [=] (size_t i) {
		auto relevant_block_id = i / num_warps_per_block;
		auto warp_id_within_block = i % num_warps_per_block;
		return make_warp_datum(relevant_block_id, warp_id_within_block);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<datum_type>
	);
}

TEST_CASE("share_per_warp_data - inspecific writer lane")
{
	using datum_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 2, 2 };
	cuda::grid::block_dimensions_t block_dimensions { kat::warp_size * 2, 5, 3 };
	auto num_warps_per_block = block_dimensions.volume() / kat::warp_size;

	auto num_values_to_populate = num_warps_per_block * grid_dimensions.volume();

	auto make_warp_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t warp_id_within_block)
		{
			return datum_type{(warp_id_within_block + 1) + (block_id + 1) * 1000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* warp_data_for_all_blocks
		)
		{
			namespace gi = kat;
			datum_type thread_datum { make_warp_datum(gi::block::id(), gi::warp::id()) };
			constexpr auto max_possible_num_warps_per_block = 32; // Note: Important assumption here...
			__shared__ datum_type warp_data [max_possible_num_warps_per_block];

			kcb::share_per_warp_data(thread_datum, warp_data);
			// We've run the synchronized variant, so no need for extra sync

			if (gi::thread::is_first_in_block()) {
				// Now we're populating what's going to be checked outside the kernel.
				auto warp_data_for_this_block = warp_data_for_all_blocks + gi::block::id() * num_warps_per_block;
				for(int i = 0; i < num_warps_per_block; i++) {
					warp_data_for_this_block[i] = warp_data[i];
				}
			}
		};

	auto expected_value_retriever = [=] (size_t i) {
		auto relevant_block_id = i / num_warps_per_block;
		auto warp_id_within_block = i % num_warps_per_block;
		return make_warp_datum(relevant_block_id, warp_id_within_block);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<datum_type>
	);
}


TEST_CASE("get_from_thread")
{
	using datum_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	constexpr const cuda::grid::block_dimensions_t block_dimensions = { kat::warp_size, 2, 1 };
	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t thread_id)
		{
			return datum_type{(thread_id + 1) + (block_id + 1) * 10000};
		};

	const auto make_source_thread_index =
		[] KAT_HD (cuda::grid::dimension_t block_id)
		{
			return kat::position_t { block_id, 1, 0 };
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* thread_obtained_values
		)
		{
			namespace gi = kat;
			datum_type thread_datum { make_thread_datum(gi::block::id(), gi::thread::id()) };
			auto source_thread_index { make_source_thread_index(gi::block::id()) };
			auto obtained_value { kcb::get_from_thread(thread_datum, source_thread_index) };
			// We've run the synchronized variant, so no need for extra sync
//			printf("Thread (%2u %2u %2u) = %4u in block %4u had datum %5d, used source thread index %2u,%2u,%2u and got value %4u\n",
//				threadIdx.x,threadIdx.y,threadIdx.z,
//				(unsigned) gi::thread::id(), (unsigned) gi::block::id(), thread_datum, source_thread_index.x, source_thread_index.y, source_thread_index.z, obtained_value   );

			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [] (size_t global_thread_index) {
		// Sorry, code duplication to avoid capture; maybe I can avoid it with C++17?
		const auto make_source_thread_index =
			[] KAT_HD (cuda::grid::dimension_t block_id)
			{ return kat::position_t { block_id, 1, 0 }; };

		auto make_thread_datum =
			[] KAT_HD (cuda::grid::dimension_t block_id,cuda::grid::block_dimension_t thread_id)
			{ return datum_type{(thread_id + 1) + (block_id + 1) * 10000}; };

		constexpr const cuda::grid::block_dimensions_t block_dimensions = { kat::warp_size, 2, 1 };
		auto block_id { global_thread_index / block_dimensions.volume() };
		auto source_thread_index { make_source_thread_index(block_id) };
		auto source_thread_id = kat::detail::row_major_linearization(source_thread_index, uint3(block_dimensions));
		return make_thread_datum(block_id, source_thread_id);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<datum_type>
	);
}


TEST_CASE("get_from_first_thread")
{
	using datum_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 2, 2 };
	cuda::grid::block_dimensions_t block_dimensions { kat::warp_size * 3, 3 };
	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_datum =
		[] KAT_HD (
			cuda::grid::dimension_t block_id,
			cuda::grid::block_dimension_t thread_id)
		{
			return datum_type{(thread_id + 1) + (block_id + 1) * 10000};
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			datum_type* thread_obtained_values
		)
		{
		namespace gi = kat;
			datum_type thread_datum { make_thread_datum(gi::block::id(), gi::thread::id()) };
			auto obtained_value { kcb::get_from_first_thread(thread_datum) };
			// We've run the synchronized variant, so no need for extra sync

			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_index) {
		auto block_id { global_thread_index / block_dimensions.volume() };
		auto source_thread_id = 0; // the first one...
		return make_thread_datum(block_id, source_thread_id);
	};

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<datum_type>
	);
}


TEST_CASE("barrier")
{
	// Note: Not much to test with the barrier() collaboration primitive, seeing how there could be "natural barrier'ing" even without the call
}

} // TEST_SUITE("block-level - general")

