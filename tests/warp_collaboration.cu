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
using kat::warp_size;
using fake_bool = int8_t; // so as not to have trouble with vector<bool>

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

TEST_SUITE("warp-level - general grid") {

TEST_CASE("barrier") {
	// TODO: Check with different lane masks and conditional
	// execution by lanes outside the mask

	using checked_value_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value =
		[] KAT_HD (unsigned warp_id, unsigned lane_id) {
			return checked_value_type( (warp_id+1) * 100 + (lane_id + 1) );
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			__shared__ int shared_array[warp_size];
			namespace gi = kat::linear_grid::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };

			shared_array[gi::lane::id()] = 0;

			kat::collaborative::block::barrier();

			if (gi::lane::id() == gi::warp::id_in_block()) {
				shared_array[gi::lane::id()] = make_thread_value(gi::warp::id_in_block(), gi::lane::id());
			};
			kcw::barrier();
			thread_obtained_values[gi::thread::global_id()] = shared_array[gi::warp::id_in_block()];
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto writing_lane_id { warp_id };
		return make_thread_value(warp_id, writing_lane_id);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);

}


TEST_CASE("all_lanes_satisfy") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::all_lanes_satisfy(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		predicate_type warp_values[warp_size];
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		bool all_satisfy { true };
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			warp_values[lane_id] = make_thread_value(warp_id, lane_id);
			all_satisfy = all_satisfy and warp_values[lane_id];
		}
		return predicate_type{all_satisfy};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);

}

TEST_CASE("no_lanes_satisfy") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::no_lanes_satisfy(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		predicate_type warp_values[warp_size];
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		bool any_satisfy { false };
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			warp_values[lane_id] = make_thread_value(warp_id, lane_id);
			any_satisfy = any_satisfy or warp_values[lane_id];
		}
		return predicate_type{not any_satisfy};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}

TEST_CASE("all_lanes_agree_on") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::all_lanes_agree_on(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		predicate_type warp_values[warp_size];
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		bool any_satisfy { false };
		bool any_dont_satisfy { false };
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			warp_values[lane_id] = make_thread_value(warp_id, lane_id);
			any_satisfy = any_satisfy or warp_values[lane_id];
			any_dont_satisfy = any_dont_satisfy or (not warp_values[lane_id]);
		}
		return predicate_type{
			(any_satisfy and not any_dont_satisfy) or
			(any_dont_satisfy and not any_satisfy)
		};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}

TEST_CASE("some_lanes_satisfy") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::some_lanes_satisfy(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		predicate_type warp_values[warp_size];
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		bool any_satisfy { false };
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			warp_values[lane_id] = make_thread_value(warp_id, lane_id);
			any_satisfy = any_satisfy or warp_values[lane_id];
		}
		return predicate_type{any_satisfy};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}

TEST_CASE("num_lanes_agreeing_on") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::num_lanes_agreeing_on(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		auto thread_value = make_thread_value(warp_id, lane_id);
		auto num_lanes_agreeing { 0 };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			if (thread_value == make_thread_value(warp_id, other_lane_id))
				{ num_lanes_agreeing++; }
		}
		return predicate_type{num_lanes_agreeing};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}

TEST_CASE("majority_vote") {
	using predicate_type = int;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id % 2);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::majority_vote(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		auto thread_value = make_thread_value(warp_id, lane_id);
		int vote_balance { 0 };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			vote_balance += make_thread_value(warp_id, other_lane_id) ? 1 : -1;
		}
		return predicate_type{vote_balance > 0};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}

#if 0
#if !defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700
TEST_CASE("in_unique_lane_with") {
	cuda::device_t device { cuda::device::current::get() };
	if (device.properties().compute_capability() < cuda::device::make_compute_capability(7,0)) {
		return;
	}

	using predicate_type = int;
	using datum_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return datum_type{12};
			case 1:  return datum_type{34};
			case 2:  return lane_id;
			default:  return (lane_id == 5 ? lane_id + 20 : lane_id);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			predicate_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::in_unique_lane_with(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		auto thread_value = make_thread_value(warp_id, lane_id);
		bool am_unique { true };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			if ((other_lane_id != lane_id) and (make_thread_value(warp_id, other_lane_id) == thread_value)) {
				am_unique = false;
			}
		}
		return predicate_type{am_unique};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<predicate_type>
	);
}
#endif //  !defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700
#endif // if 0

TEST_CASE("get_from_lane") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto make_source_lane = [] KAT_HD (unsigned thread_id)
		{
			auto warp_id = thread_id / warp_size;
			return (warp_id + 1) % warp_size;
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::thread::id()) };
			auto source_lane { make_source_lane(gi::thread::id()) };
			auto obtained_value { kcw::get_from_lane(thread_value, source_lane) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto source_lane { make_source_lane(thread_id) };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + source_lane };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("get_from_first_lane") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::thread::id()) };
			auto obtained_value { kcw::get_from_first_lane(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + 0 };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}


TEST_CASE("get_from_last_lane") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::thread::id()) };
			auto obtained_value { kcw::get_from_last_lane(thread_value) };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + (warp_size - 1) };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("have_a_single_lane_compute") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto make_source_lane = [] KAT_HD (unsigned thread_id)
		{
			auto warp_id = thread_id / warp_size;
			return (warp_id + 1) % warp_size;
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto source_lane { make_source_lane(gi::thread::id()) };
			auto obtained_value =
				kcw::have_a_single_lane_compute(
					[=]() { return make_thread_value(gi::thread::id()); },
					source_lane);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto source_lane { make_source_lane(thread_id) };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + source_lane };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}


TEST_CASE("have_first_lane_compute") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto obtained_value =
				kcw::have_first_lane_compute(
					[=]() { return make_thread_value(gi::thread::id()); }
				);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + 0 };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("have_last_lane_compute") {
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 3 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned thread_id)
		{
			return checked_value_type((thread_id + 1) * 10);
		};

	auto testcase_device_function =
		[=] KAT_DEV (size_t, checked_value_type* thread_obtained_values)
		{
			namespace gi = kat::grid_info;
			auto obtained_value =
				kcw::have_last_lane_compute(
					[=]() { return make_thread_value(gi::thread::id()); }
				);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};

	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto thread_id { global_thread_id % block_dimensions.volume() };
		auto warp_id { thread_id / warp_size };
		auto source_thread_id { warp_id * warp_size + (warp_size - 1) };
		checked_value_type source_thread_value { make_thread_value(source_thread_id) };
		return source_thread_value;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("first_lane_satisfying") {
	using predicate_type = int;
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 4 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0);
			case 1:  return predicate_type(1);
			case 2:  return predicate_type(lane_id > 5);
			default:  return predicate_type(lane_id != 7);
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			auto thread_value { make_thread_value(gi::warp::id_in_block(), gi::lane::id()) };
			auto obtained_value { kcw::first_lane_satisfying(thread_value) };
			// if (threadIdx.x >= 64) printf("Thread %u value %u obtained %u\n", gi::thread::global_id(), thread_value, obtained_value);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
//		auto lane_id { global_thread_id % warp_size };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			if (make_thread_value(warp_id, other_lane_id))
				{ return checked_value_type{other_lane_id}; }
		}
		return checked_value_type{warp_size};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}


TEST_CASE("get_active_lanes") {
	using kat::lane_mask_t;
	constexpr const lane_mask_t default_lane_mask_for_threads_who_go_inactive = 0xDEADBEEFu;

	using predicate_type = bool;
	using checked_value_type = lane_mask_t;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 4 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto determine_whether_to_stay_active = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0); // all go inactive
			case 1:  return predicate_type(1); // all stay active
			case 2:  return predicate_type(lane_id > 5); // lanes 6 and up stay active
			default:  return predicate_type(lane_id != 7); // only lane 7 goes inactive
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			bool should_stay_active { determine_whether_to_stay_active(gi::warp::id_in_block(), gi::lane::id()) };
			// if (threadIdx.x < 32)
			//	printf("Thread %u %s stay active\n", gi::thread::id(), (should_stay_active ? "SHOULD" : "SHOULD NOT"));
			if (not should_stay_active) {
				thread_obtained_values[gi::thread::global_id()] =
					default_lane_mask_for_threads_who_go_inactive;
				return;
			}
			auto obtained_value { kcw::get_active_lanes() };
			// if (threadIdx.x >= 64) printf("Thread %u value %u obtained %u\n", gi::thread::global_id(), thread_value, obtained_value);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		if (not determine_whether_to_stay_active(warp_id, lane_id)) {
			return default_lane_mask_for_threads_who_go_inactive;
		}
		lane_mask_t mask { 0 };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			if (determine_whether_to_stay_active(warp_id, other_lane_id)) {
				mask |= (1u << other_lane_id);
			}
		}
		return mask;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("num_active_lanes") {
	using kat::lane_mask_t;
	using checked_value_type = unsigned;

	constexpr const checked_value_type invalid_num_active_lanes = 33;

	using predicate_type = bool;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 4 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto determine_whether_to_stay_active = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0); // all go inactive
			case 1:  return predicate_type(1); // all stay active
			case 2:  return predicate_type(lane_id > 5); // lanes 6 and up stay active
			default:  return predicate_type(lane_id != 7); // only lane 7 goes inactive
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			bool should_stay_active { determine_whether_to_stay_active(gi::warp::id_in_block(), gi::lane::id()) };
			if (not should_stay_active) {
				thread_obtained_values[gi::thread::global_id()] = invalid_num_active_lanes;
				return;
			}
			auto obtained_value { kcw::num_active_lanes() };
			// if (threadIdx.x >= 64) printf("Thread %u value %u obtained %u\n", gi::thread::global_id(), thread_value, obtained_value);
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		if (not determine_whether_to_stay_active(warp_id, lane_id)) {
			return invalid_num_active_lanes;
		}
		auto active_count { 0u };
		for(unsigned other_lane_id = 0; other_lane_id < warp_size; other_lane_id++) {
			if (determine_whether_to_stay_active(warp_id, other_lane_id)) {
				active_count++;
			}
		}
		return checked_value_type{active_count};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("am_leader_lane") {
	using kat::lane_mask_t;
	using checked_value_type = fake_bool;
		// Not using actual to not get in trouble with __nv_bool and with std::vector<bool>
	using predicate_type = bool;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 4 };
		// TODO: What about when the last warp is not full?
	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_warps_in_grid = num_total_threads / warp_size;
	auto num_values_to_populate = num_total_threads;

	auto determine_whether_to_stay_active = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0); // all go inactive
			case 1:  return predicate_type(1); // all stay active
			case 2:  return predicate_type(lane_id > 5); // lanes 6 and up stay active
			default:  return predicate_type(lane_id != 7); // only lane 7 goes inactive
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			namespace gi = kat::linear_grid::grid_info;
			bool should_stay_active { determine_whether_to_stay_active(gi::warp::id_in_block(), gi::lane::id()) };
			if (not should_stay_active) {
				thread_obtained_values[gi::thread::global_id()] = false;
				// if (threadIdx.x < 32) printf("Thread %u goes inactive\n", gi::thread::id());
				return;
			}
			auto obtained_value { kcw::am_leader_lane() };
			// if (threadIdx.x < 32) printf("Thread %u value believes it %s the leader\n", gi::thread::id(), (obtained_value ? "IS" : "IS NOT"));
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	// Note: Deviating from our usual checks, since the leader selection is not required
	// to choose the first thread - so, we can't compare against a generated value;
	// and the checks are at the warp level


	auto thread_obtained_values = execute_non_uniform_testcase_on_gpu(
		tag<checked_value_type>{},
		testcase_device_function,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions
	);

	for(size_t warp_id = 0; warp_id < num_warps_in_grid; warp_id++) {
		unsigned num_active_lanes { 0 };
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			if (determine_whether_to_stay_active(warp_id, lane_id)) { num_active_lanes++; }
		}
		auto warp_results_begin = thread_obtained_values.data() + warp_id * warp_size;
		auto warp_results_end = warp_results_begin + warp_size;
		auto num_presumptive_leaders  = std::count(warp_results_begin, warp_results_end, true);
		// std::cout << "warp " << warp_id << ": num_presumptive_leaders  = " << num_presumptive_leaders << ", num_active_lanes = " << num_active_lanes << '\n';
		bool have_presumptive_leaders = (num_presumptive_leaders > 0);
		bool have_active_lanes = (num_active_lanes > 0);
		if (have_active_lanes and not have_presumptive_leaders) {
			auto error_message = "In warp " + std::to_string(warp_id) + " of the grid, some lanes were active, but no lane recognized itself as the leader";
			if (have_active_lanes and not have_presumptive_leaders) {
				CHECK_MESSAGE(false, error_message);
			}
		}
		if (num_presumptive_leaders > 1) {
			auto error_message = "In warp " + std::to_string(warp_id) + " of the grid, multiple lanes concluded they were the single leader lane";
			CHECK_MESSAGE(false, error_message);
		}
	}

}

TEST_CASE("index_among_active_lanes") {
	using kat::lane_mask_t;
	using checked_value_type = unsigned;

	constexpr const checked_value_type invalid_index = 33;

	using predicate_type = bool;
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 4 };
		// TODO: What about when the last warp is not full?

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto determine_whether_to_stay_active = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return predicate_type(0); // all go inactive
			case 1:  return predicate_type(1); // all stay active
			case 2:  return predicate_type(lane_id > 5); // lanes 6 and up stay active
			default:  return predicate_type(lane_id != 7); // only lane 7 goes inactive
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_obtained_values
		)
		{
			namespace gi = kat::grid_info;
			bool should_stay_active { determine_whether_to_stay_active(gi::warp::id_in_block(), gi::lane::id()) };
			if (not should_stay_active) {
				thread_obtained_values[gi::thread::global_id()] = invalid_index;
				return;
			}
			auto obtained_value { kcw::index_among_active_lanes() };
			thread_obtained_values[gi::thread::global_id()] = obtained_value;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		if (not determine_whether_to_stay_active(warp_id, lane_id)) {
			return invalid_index;
		}
		auto num_active_before_this_thread { 0u };
		for(unsigned other_lane_id = 0; other_lane_id < lane_id; other_lane_id++) {
			if (determine_whether_to_stay_active(warp_id, other_lane_id)) {
				num_active_before_this_thread++;
			}
		}
		return checked_value_type{num_active_before_this_thread};
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("at_warp_stride")
{
	using checked_value_type = uint32_t;
	cuda::grid::dimensions_t num_grid_blocks { 1 };
	cuda::grid::block_dimensions_t num_threads_per_block { warp_size * 3 }; // Meeting full warp constraint
	auto num_grid_warps = num_grid_blocks.volume() * num_threads_per_block.volume() / warp_size;
	size_t length_to_cover_per_warp { 71 };
		// We don't actually create input data, we just need each element in the
		// range 0 ... length_to_cover-1 to be attended to.
		//
		// In this test case - it's 0 ... length_to_cover-1 attended to by _each_ of the warps.

	auto num_values_to_populate = length_to_cover_per_warp * num_grid_warps;

	auto testcase_device_function =
		[length_to_cover_per_warp] KAT_DEV (
			size_t num_grid_threads,
			checked_value_type* pos_attendent_thread_indices
		)
		{
			namespace gi = kat::grid_info;
			auto offset_into_attendant_array = length_to_cover_per_warp * gi::warp::id_in_block();
			auto f_inner = [&] (size_t pos) {
				pos_attendent_thread_indices[offset_into_attendant_array + pos] = gi::thread::id_in_grid();
			};
			kcw::at_warp_stride(length_to_cover_per_warp, f_inner);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		// Which thread processes position pos?

		auto intra_warp_pos = pos % length_to_cover_per_warp;
		auto processing_warp_index = pos / length_to_cover_per_warp;
		auto processing_lane_index = intra_warp_pos % warp_size;
		return checked_value_type(processing_lane_index  + processing_warp_index * warp_size);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);

}

TEST_CASE("active_lanes_atomically_increment")
{
	using incremeted_type = int; // TODO: Should we also test other types?
	using checked_value_type = incremeted_type;
	using predicate_type = bool;
	cuda::grid::dimensions_t grid_dimensions { 2 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 5 };
	auto num_grid_threads = grid_dimensions.volume() * block_dimensions.volume();
	constexpr const incremeted_type invalid_value_for_inactives { -0xB7AB7A };

	auto num_values_to_populate = num_grid_threads;

	cuda::device_t device { cuda::device::current::get() };
	auto device_side_increment_target { cuda::memory::device::make_unique<incremeted_type>(device) };
	incremeted_type* device_side_increment_target_raw = device_side_increment_target.get();
	cuda::memory::zero(device_side_increment_target_raw, sizeof(incremeted_type));

	auto determine_whether_to_stay_active = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id % 5) {
			case 0:   return predicate_type(0); // all go inactive
			case 1:   return predicate_type(1); // all stay active
			case 2:   return predicate_type(lane_id > 5); // lanes 6 and up stay active
			case 3:   return predicate_type(lane_id % 2); // odd lanes stay active
			default:  return predicate_type(lane_id != 7); // only lane 7 goes inactive
			}
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* thread_values_before_increment
		)
		{
			namespace gi = kat::grid_info;
			if (not determine_whether_to_stay_active(gi::warp::id_in_block(), gi::lane::id() )) {
				thread_values_before_increment[gi::thread::global_id()] = invalid_value_for_inactives;
				return;
			}
			thread_values_before_increment[gi::thread::global_id()] =
				kcw::active_lanes_atomically_increment(device_side_increment_target_raw);
		};

	incremeted_type expected_num_active_threads { 0 };
	for(incremeted_type global_thread_id = 0; global_thread_id < num_grid_threads; global_thread_id++) {
		auto in_block_thread_id = global_thread_id % block_dimensions.volume();
		auto warp_id = in_block_thread_id / warp_size;
		auto lane_id = global_thread_id % warp_size;
		if (determine_whether_to_stay_active(warp_id, lane_id)) { expected_num_active_threads++; }
	}

	auto thread_values_before_increment = execute_non_uniform_testcase_on_gpu(
		tag<incremeted_type>{},
		testcase_device_function,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions
	);

	incremeted_type final_incremented;
	cuda::memory::copy_single(&final_incremented, device_side_increment_target_raw);
	CHECK_MESSAGE(final_incremented == expected_num_active_threads,
		"The increment target, originally 0, was not incremented by 1 for every thread in the test grid");

	std::vector<incremeted_type> active_thread_values;
	std::copy_if(thread_values_before_increment.cbegin(), thread_values_before_increment.cend(),
		std::back_inserter(active_thread_values), [](incremeted_type x) { return x != invalid_value_for_inactives; });
	std::sort(active_thread_values.begin(), active_thread_values.end());
	for(size_t i = 0; i < active_thread_values.size(); i++) {
		CHECK_MESSAGE(i == active_thread_values[i],
			"Not every intermediate value between 0 and final incremented value appears as some thread's pre-increment value");
	}
}

} // TEST_SUITE("warp-level - general grid")

TEST_SUITE("warp-level - linear grid") {


TEST_CASE("at_warp_stride")
{
	using checked_value_type = uint32_t;
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { warp_size * 3 }; // Meeting full warp constraint
	auto num_grid_warps = num_grid_blocks * num_threads_per_block / warp_size;
	size_t length_to_cover_per_warp { 71 };
		// We don't actually create input data, we just need each element in the
		// range 0 ... length_to_cover-1 to be attended to.
		//
		// In this test case - it's 0 ... length_to_cover-1 attended to by _each_ of the warps.

	auto num_values_to_populate = length_to_cover_per_warp * num_grid_warps;

	auto testcase_device_function =
		[length_to_cover_per_warp] KAT_DEV (
			size_t num_grid_threads,
			checked_value_type* pos_attendent_thread_indices
		)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto offset_into_attendant_array = length_to_cover_per_warp * gi::warp::id_in_block();
			auto f_inner = [&] (size_t pos) {
				pos_attendent_thread_indices[offset_into_attendant_array + pos] = gi::thread::id_in_grid();
			};
			klcw::at_warp_stride(length_to_cover_per_warp, f_inner);
		};

	auto expected_value_retriever = [=] (size_t pos) {
		// Which thread processes position pos?

		auto intra_warp_pos = pos % length_to_cover_per_warp;
		auto processing_warp_index = pos / length_to_cover_per_warp;
		auto processing_lane_index = intra_warp_pos % warp_size;
		return checked_value_type(processing_lane_index  + processing_warp_index * warp_size);
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);
}

TEST_CASE("multisearch") {
/*	using thread_value_type = int;
	using checked_value_type = kat::linear_grid::collaborative::warp::search_result_t<thread_value_type>;
	static_assert(std::is_trivially_constructible<checked_value_type>::value,
		"search results are supposed to be trivially-constructible!");
	cuda::grid::dimensions_t grid_dimensions { 1 };
	cuda::grid::block_dimensions_t block_dimensions { warp_size * 7 };

	auto num_total_threads = block_dimensions.volume() * grid_dimensions.volume();
	auto num_values_to_populate = num_total_threads;

	auto make_thread_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			switch (warp_id) {
			case 0:  return thread_value_type(-2);
			case 1:  return thread_value_type(2);
			case 2:  return thread_value_type(lane_id < 5 ? -2 : 2);
			case 3:  return thread_value_type(lane_id < 4 ? -2 : lane_id == 4 ? -1 : 2);
			case 4:  return thread_value_type(lane_id - warp_size/2);
			case 5:  return thread_value_type(lane_id == 0 ? -2 : 2);
			case 6:
			default: return thread_value_type(lane_id < warp_size - 1 ? -2 : 2);
			}
		};

	auto make_search_value = [] KAT_HD (unsigned warp_id, unsigned lane_id)
		{
			return -2 + ((int) lane_id % 5);
		};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* search_results
		)
		{
			namespace gi = kat::grid_info;
			auto haystack_straw = make_thread_value(gi::warp::id_in_block(), gi::lane::id());
			thread_value_type needle_to_search_for = make_search_value(gi::warp::id_in_block(), gi::lane::id());
			auto search_result = klcw::multisearch(needle_to_search_for, haystack_straw);
			search_results[gi::thread::global_id()] = search_result;
		};


	auto expected_value_retriever = [=] (size_t global_thread_id) -> checked_value_type {
		auto warp_id { (global_thread_id % block_dimensions.volume()) / warp_size };
		auto lane_id { global_thread_id % warp_size };
		auto needle = make_search_value(warp_id, lane_id);
		thread_value_type warp_values[warp_size];
		// checked_value_type warp_search_results[warp_size];
		for(unsigned lane_id = 0; lane_id < warp_size; lane_id++) {
			warp_values[lane_id] = make_thread_value(warp_id, lane_id);
		}
		auto ptr = std::find_if(warp_values, warp_values + warp_size,
			[=](thread_value_type x) { return x <= needle; });
		return {
			unsigned(ptr - warp_values),
			(ptr - warp_values == warp_size) ? 0 : *ptr
		};
		// return expected_search_result;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		num_values_to_populate,
		grid_dimensions,
		block_dimensions,
		make_exact_comparison<checked_value_type>
	);*/
}

TEST_CASE_TEMPLATE ("compute_predicate_at_warp_stride", SlackSetting,
	std::integral_constant<klcw::detail::predicate_computation_length_slack_t, klcw::detail::predicate_computation_length_slack_t::may_have_full_warps_of_slack>,
	std::integral_constant<klcw::detail::predicate_computation_length_slack_t, klcw::detail::predicate_computation_length_slack_t::has_no_slack>,
	std::integral_constant<klcw::detail::predicate_computation_length_slack_t, klcw::detail::predicate_computation_length_slack_t::may_have_arbitrary_slack>
)
{
	using checked_value_type = uint32_t; // but as a bit container
	cuda::grid::dimension_t num_grid_blocks { 1 };
	cuda::grid::block_dimension_t num_threads_per_block { warp_size * 3 }; // Meeting full warp constraint
	auto num_grid_warps = num_grid_blocks * num_threads_per_block / warp_size;
	size_t length_to_cover_per_warp;
	switch (SlackSetting::value) {
		case klcw::detail::predicate_computation_length_slack_t::has_no_slack:
			length_to_cover_per_warp = warp_size * warp_size * 3;
			break;
		case klcw::detail::predicate_computation_length_slack_t::may_have_full_warps_of_slack:
			length_to_cover_per_warp = warp_size * 3;
			break;
		case klcw::detail::predicate_computation_length_slack_t::may_have_arbitrary_slack:
		default:
			length_to_cover_per_warp = warp_size + 1;
			break;
	}

	auto num_bit_containers_per_warp = ::div_rounding_up<uint32_t>(length_to_cover_per_warp, warp_size);
	auto num_bit_containers = num_bit_containers_per_warp  * num_grid_warps;
		// Note that we populate a slack for every one of the warps' stretches
	auto pos_attendants_length = length_to_cover_per_warp * num_grid_warps;

	cuda::device_t device { cuda::device::current::get() };
	auto device_side_pos_attendants { cuda::memory::device::make_unique<uint32_t[]>(device, pos_attendants_length) };
	uint32_t* device_side_pos_attendants_raw = device_side_pos_attendants.get();
	cuda::memory::zero(device_side_pos_attendants_raw, sizeof(uint32_t));


	auto pred = [] KAT_HD (uint32_t intra_warp_stretch_pos) -> bool { return (intra_warp_stretch_pos % 7 == 0); };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t num_grid_threads,
			checked_value_type* computed_predicate
		)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto instrumented_pred = [=] (size_t pos) -> bool {
				device_side_pos_attendants_raw[gi::warp::global_id() * length_to_cover_per_warp + pos] = gi::thread::id_in_grid();
				return pred(pos);
			};
			auto computed_predicate_for_this_warp  = computed_predicate + num_bit_containers_per_warp * gi::warp::global_id();
			klcw::compute_predicate_at_warp_stride(computed_predicate_for_this_warp, instrumented_pred, length_to_cover_per_warp);
		};

	auto expected_bit_container_retriever = [=] (uint32_t bit_container_pos) -> uint32_t {
		auto intra_warp_stretch_bit_container_pos = bit_container_pos % num_bit_containers_per_warp;
		checked_value_type bit_container { 0 };
		for(auto lane_id = 0; lane_id < warp_size; lane_id++) {
			auto intra_warp_pos = intra_warp_stretch_bit_container_pos * warp_size + lane_id;
			if (intra_warp_pos < length_to_cover_per_warp) {
				bit_container |= ( pred(intra_warp_pos) << lane_id );
			}
		}
		return bit_container;
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_bit_container_retriever,
		num_bit_containers, num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);

	std::vector<uint32_t> pos_attendants(pos_attendants_length);
	cuda::memory::copy(pos_attendants.data(), device_side_pos_attendants_raw, sizeof(uint32_t) * pos_attendants_length);
	device.synchronize();

	auto expected_attendant_retriever = [=] (uint32_t pos) -> uint32_t {
		auto attending_warp = pos / length_to_cover_per_warp;
		auto attending_lane = (pos % length_to_cover_per_warp) % warp_size;
		return attending_warp * warp_size + attending_lane;
	};

	auto check_title = std::string(std::string("which thread attended which position in testcase ") + doctest::current_test_name());
	check_results(
		check_title,
		pos_attendants_length,
		pos_attendants.data(),
		expected_attendant_retriever,
		make_exact_comparison<uint32_t>);

}

/*
TEST_CASE("merge_sorted_half_warps - in-register")
{
	using lane_value_type = int;
	using checked_value_type = lane_value_type;

	std::vector<lane_value_type> half_warps {
		 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
		 0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,
		 1,  1,  1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1,  1,  1,
		 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
		 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
		 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 31,
		 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,
	};
	constexpr const auto half_warp_size = warp_size / 2;
	auto num_half_warps = half_warps.size() / half_warp_size;

	cuda::device_t device { cuda::device::current::get() };
	auto num_threads_per_block { warp_size *
		std::min<cuda::grid::block_dimension_t>(device.properties().max_warps_per_block(), num_half_warps / 2) }; // Meeting full warp constraint
	cuda::grid::dimension_t num_grid_blocks { std::max<cuda::grid::dimension_t>(1, (warp_size * num_half_warps) / num_threads_per_block) };
	auto total_num_threads = num_threads_per_block * num_grid_blocks;

	auto device_side_half_warps { cuda::memory::device::make_unique<lane_value_type[]>(device, half_warps.size()) };
	auto device_side_half_warps_raw = device_side_half_warps.get();
	cuda::memory::copy(device_side_half_warps_raw, half_warps.data(), sizeof(lane_value_type) * half_warps.size() );

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			checked_value_type* merged_data
		)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto half_warp_pair_index = gi::warp::global_id(); // Each warp gets a different pair of half-warps to merge
			auto first_half_warp_index = half_warp_pair_index / num_half_warps;
			auto second_half_warp_index = half_warp_pair_index % num_half_warps;
			auto my_half_warp_index = gi::lane::is_in_first_half_warp() ?
				first_half_warp_index : second_half_warp_index;
			lane_value_type lane_value = device_side_half_warps_raw[my_half_warp_index * half_warp_size + gi::lane::id()];
			auto post_merge_value = klcw::merge_sorted_half_warps(lane_value);
			// printf("Thread %3u had %2d now has %2d\n", gi::thread::global_id(), lane_value, post_merge_value);
			merged_data[gi::thread::global_id()] = post_merge_value;
		};

	auto expected_value_retriever = [=] (size_t pos) {
		auto half_warp_pair_index = pos / warp_size;
		auto first_half_warp_index = half_warp_pair_index / num_half_warps;
		auto first_half_warp_begin = half_warps.cbegin() +  first_half_warp_index * half_warp_size;
		auto second_half_warp_index = half_warp_pair_index % num_half_warps;
		auto second_half_warp_begin = half_warps.cbegin() +  second_half_warp_index * half_warp_size;
		std::array<lane_value_type, warp_size> merged;
		std::merge(
			first_half_warp_begin, first_half_warp_begin + half_warp_size,
			second_half_warp_begin, second_half_warp_begin + half_warp_size,
			merged.begin());
		auto lane_index = pos % warp_size;
		return merged[lane_index];
	};

	execute_non_uniform_testcase_on_gpu_and_check(
		testcase_device_function,
		expected_value_retriever,
		total_num_threads,
		num_grid_blocks, num_threads_per_block,
		make_exact_comparison<checked_value_type>
	);

}
*/

} // TEST_SUITE("warp-level - linear grid")


