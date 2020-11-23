#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/math.cuh>
#include <kat/on_device/shuffle.cuh>
#include <kat/containers/array.hpp>
#include <kat/on_device/atomics.cuh>
#include <kat/on_device/grid_info.cuh>
#include <random>
#include <numeric>
#include <cmath>

// TODO: Run some/all tests for half-precision floating-point values, e.g __half from:
// #include <cuda_fp16.h>

// TODO: Also test behavior with warps with some inactive/exited lanes

#include <kat/detail/execution_space_specifiers.hpp>

using std::vector;
using std::uniform_int_distribution;

template <typename F> inline F test_epsilon(std::size_t num_operations) { return 0; }

template <>
inline float test_epsilon<float>(std::size_t num_operations) { return 1e-5 * num_operations; };
template <>
inline double test_epsilon<double>(std::size_t num_operations) { return 1e-10 * num_operations; };

struct seq_type {
	enum { ops_per_thread  = 4 };
	bool elements[ops_per_thread];
	KAT_HD bool do_increment(int i) { return elements[i]; }
	KAT_HD bool do_decrement(int i) { return not elements[i]; }
	KAT_HD bool& operator[](int i) { return elements[i]; }
	KAT_HD bool operator[](int i) const { return elements[i]; }
};

template <typename T>
inline std::enable_if_t<std::is_floating_point<T>::value, std::size_t> nice_big_value()
{
	// This is very hacky :-(
	return T(10e6);
}

template <typename T>
inline std::enable_if_t<std::is_integral<T>::value, std::size_t> nice_big_value()
{
	return std::is_signed<T>::value ?
		std::numeric_limits<T>::max() :
		std::numeric_limits<T>::max() / 2;
}

template <typename T>
inline T middle_of_domain()
{
	// TODO: Check T is linearly ordered
	static_assert(std::is_arithmetic<T>::value, "Invalid type");
	if (std::is_floating_point<T>::value) { return T(0); }
	else {
		return T(std::numeric_limits<T>::max() / 2.0 + std::numeric_limits<T>::min() / 2.0);
	}
}


namespace kernels {


template <typename T, unsigned ElementsPerThread>
__global__ void test_add(
	T*       __restrict__ result,
	const T* __restrict__ data,
	std::size_t           data_size)
{
	// Notes:
	// * The access pattern used here is sub-optimal. But it doesn't matter since we're
	//   not trying to optimize speed.
	// * We could have used something like kat::collaborative::grid::linear::at_grid_stride
	//   but we want to minimize dependencies here.
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;

	std::size_t pos = global_thread_index * ElementsPerThread;
	for(int i = 0; (i < ElementsPerThread) and (pos < data_size); i++)
	{
		kat::atomic::add(result, data[pos]);
	    pos++;
	}
}

template <typename T, unsigned ElementsPerThread>
__global__ void test_subtract(
	T*       __restrict__ result,
	const T* __restrict__ data,
	std::size_t           data_size)
{
	// Notes:
	// * The access pattern used here is sub-optimal. But it doesn't matter since we're
	//   not trying to optimize speed.
	// * We could have used something like kat::collaborative::grid::linear::at_grid_stride
	//   but we want to minimize dependencies here.
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;

	std::size_t pos = global_thread_index * ElementsPerThread;
	for(int i = 0; (i < ElementsPerThread) and (pos < data_size); i++)
	{
		kat::atomic::subtract(result, data[pos]);
	    pos++;
	}
}


template <typename T, unsigned ElementsPerThread>
__global__ void test_exchange(
	T* __restrict__  extra_datum,
	T* __restrict__  data,
	std::size_t      data_size)
{
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;

	std::size_t pos = global_thread_index * ElementsPerThread;
	for(int i = 0; (i < ElementsPerThread) and (pos < data_size); i++)
	{
		auto datum_to_use = data[pos];
		auto previous_extra_datum = kat::atomic::exchange(extra_datum, datum_to_use);
		data[pos] = previous_extra_datum;
	    pos++;
	}
}


template <typename T, typename SeqType>
__global__ void test_inc_dec_sequences(
	T*       __restrict__  aggregate,
	SeqType* __restrict__  inc_dec_sequences)
{
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;

	auto seq = inc_dec_sequences[global_thread_index];

	for(int i = 0; i < SeqType::ops_per_thread;i ++)
	{
		if (seq[i]) {
			kat::atomic::increment(aggregate);
		}
		else {
			kat::atomic::decrement(aggregate);
		}
	}
};

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
	std::index_sequence<Indices...>,
	K                                testcase_kernel,
	F                                testcase_device_function,
	cuda::launch_configuration_t     launch_config,
	size_t                           num_values_to_populate,
	T                                result_initial_fill_value,
	Is* __restrict__ ...             inputs)
{
	cuda::device_t device { cuda::device::current::get() };
	auto device_side_results { cuda::memory::device::make_unique<T[]>(device, num_values_to_populate) };
	cuda::launch(
		kernels::fill<T, kat::size_t>,
		make_busy_config(device),
		device_side_results.get(),
		result_initial_fill_value,
		num_values_to_populate);
	auto host_side_results { vector<T>(num_values_to_populate) };

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

template <typename F, typename T, typename... Is>
auto execute_non_uniform_testcase_on_gpu(
	F                               testcase_device_function,
	size_t                          num_values_to_populate,
	T                               result_initial_fill_value,
	cuda::grid::dimensions_t        grid_dimensions,
	cuda::grid::block_dimensions_t  block_dimensions,
	Is* __restrict__ ...            inputs)
{
	auto launch_config { cuda::make_launch_config(grid_dimensions, block_dimensions) };

	return execute_testcase_on_gpu(
		typename std::make_index_sequence<sizeof...(Is)> {},
		kernels::execute_testcase<F, T, Is...>,
		testcase_device_function,
		launch_config,
		num_values_to_populate,
		result_initial_fill_value,
		inputs...
	);
}

size_t num_threads_in_grid(const cuda::launch_configuration_t& launch_config)
{
	return static_cast<size_t>(launch_config.grid_dimensions.volume()) * launch_config.block_dimensions.volume();
}

enum { largest_type_size = 8  };

TEST_SUITE("atomics") {

/*
 * Test description:
 *
 * - Generate I.I.D. data using random sampling
 * - Add up all the elements using atomic add
 * - Make sure the final value is correct
 *
 */
TEST_CASE_TEMPLATE("add", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	const auto num_grid_blocks { 3 };
	const auto block_size { kat::warp_size * 3 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	constexpr const auto elements_per_thread { 5 };
	constexpr auto data_size { num_grid_blocks * block_size * elements_per_thread };
	constexpr auto data_size_plus_alignment { data_size + (data_size % largest_type_size == 0 ? 0 : (largest_type_size - data_size % largest_type_size)) };
	struct {
		decltype(cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment)) data;
		decltype(cuda::memory::device::make_unique<T[]>(device, largest_type_size)) result;
			// Using largest_type_size so we can safely CAS into the result
	} device_side;
	device_side.data = cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment);
	device_side.result = cuda::memory::device::make_unique<T[]>(device, largest_type_size);

	struct  {
		vector<T> data;
		T result;
	} host_side;

	auto mean { 1 };
	auto standard_deviation { 11 };
	std::normal_distribution<double> distribution(mean, standard_deviation);
	host_side.data.reserve(data_size);
	util::random::insertion_generate_n(std::back_inserter(host_side.data), data_size, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	T expected_result {
		std::accumulate(host_side.data.begin(), host_side.data.end(), T{0})
	};

	cuda::memory::copy(device_side.data.get(), host_side.data.data(), host_side.data.size() * sizeof(T) );

	host_side.result = T{0};
	cuda::memory::copy(device_side.result.get(), &host_side.result, sizeof(T));

	cuda::launch(
		::kernels::test_add<T, elements_per_thread>,
		launch_config,
		device_side.result.get(),
		device_side.data.get(),
		data_size);

	cuda::outstanding_error::ensure_none();

	cuda::memory::copy(&host_side.result, device_side.result.get(), sizeof(T));
	int64_t i64result;
	cuda::memory::copy(&i64result, device_side.result.get(), 8);

	bool print_results { false };

	if (std::is_floating_point<T>::value) {
		CHECK(host_side.result == doctest::Approx(expected_result).epsilon(test_epsilon<T>(data_size)));
	}
	else {
		CHECK(host_side.result == expected_result);
	}

	if (print_results) {
		std::cout << "Results match for type " << util::type_name<T>() << ".\n";
	}
}

/*
 * Test description:
 *
 * - Generate I.I.D. data using random sampling
 * - Initialize a device-memory value to the sum of all data, plus an arbitrary extra value (you'll see why in a moment)
 * - Use atomic subtract to subtract all data from the sum - reaching, not zero, but the arbitrary value
 */
TEST_CASE_TEMPLATE("subtract", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	const auto num_grid_blocks { 3 };
	const auto block_size { kat::warp_size * 3 };
	constexpr const auto elements_per_thread { 5 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	constexpr auto data_size { num_grid_blocks * block_size * elements_per_thread };
	constexpr auto data_size_plus_alignment { data_size + (data_size % largest_type_size == 0 ? 0 : (largest_type_size - data_size % largest_type_size)) };
	struct {
		decltype(cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment)) data;
		decltype(cuda::memory::device::make_unique<T[]>(device, largest_type_size)) result;
			// Using largest_type_size so we can safely CAS into the result
	} device_side;
	device_side.data = cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment);
	device_side.result = cuda::memory::device::make_unique<T[]>(device, largest_type_size);

	struct  {
		vector<T> data;
		T result;
	} host_side;

	auto mean { 1 };
	auto standard_deviation { 11 };
	host_side.data.reserve(data_size);
	std::normal_distribution<double> distribution(mean, standard_deviation);
	util::random::insertion_generate_n(std::back_inserter(host_side.data), data_size, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	T sum_of_data {
		std::accumulate(host_side.data.begin(), host_side.data.end(), T{0})
	};
	T arbitrary_extra_value = T{12};
	T expected_result = arbitrary_extra_value;


	cuda::memory::copy(device_side.data.get(), host_side.data.data(), host_side.data.size() * sizeof(T) );

	host_side.result = sum_of_data + arbitrary_extra_value;
		// We're initializing the result value to something high so we can't subtract without underflow

	cuda::memory::copy(device_side.result.get(), &host_side.result, sizeof(T));

	cuda::launch(
		::kernels::test_subtract<T, elements_per_thread>,
		launch_config,
		device_side.result.get(),
		device_side.data.get(),
		data_size);

	cuda::outstanding_error::ensure_none();

	cuda::memory::copy(&host_side.result, device_side.result.get(), sizeof(T));

	bool print_results { false };

	if (std::is_floating_point<T>::value) {
		CHECK(host_side.result == doctest::Approx(expected_result).epsilon(test_epsilon<T>(data_size)));
	}
	else {
		CHECK(host_side.result == expected_result);
	}

	if (print_results) {
		std::cout << "Results match for type " << util::type_name<T>() << ".\n";
	}
}

/*
 * Test description:
 *
 * - Generate I.I.D. data using random sampling
 * - Set some arbitrary value in global device memory
 * - Play "musical chairs" between the data and the global memory value using atomic exchange
 * - Finally, make sure we still have all the original data, and the original global-mem single value,
 *   when we consider the entire array + what's in the single-value location (but of course, it
 *   doesn't have to end up in the same positions as we started with)
 */
TEST_CASE_TEMPLATE("exchange", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	const auto num_grid_blocks { 3 };
	const auto block_size { kat::warp_size * 5 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	constexpr const auto elements_per_thread { 5 };
	constexpr auto data_size { num_grid_blocks * block_size * elements_per_thread };
	constexpr auto data_size_plus_alignment { data_size + (data_size % largest_type_size == 0 ? 0 : (largest_type_size - data_size % largest_type_size)) };
	struct {
		decltype(cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment)) data;
		decltype(cuda::memory::device::make_unique<T[]>(device, largest_type_size)) extra_datum;
			// Using largest_type_size so we can safely CAS into the result
	} device_side;
	device_side.data = cuda::memory::device::make_unique<T[]>(device, data_size_plus_alignment);
	device_side.extra_datum = cuda::memory::device::make_unique<T[]>(device, largest_type_size);

	struct  {
		vector<T> input_data;
		vector<T> output_data;
		T extra_datum;
	} host_side;

	host_side.input_data.reserve(data_size + 1); // we're not using the +1 for now.
	auto base_value = static_cast<T>( M_PI / 4 ); // This will be 0 for integers and around 0.78525 for floating-point types
	T delta { 1 }; // This will be 0 for integers and around 0.78525 for floating-point types
	for(size_t i = 0; i < data_size; i++) {
		host_side.input_data.push_back(base_value + i * delta);
			// Note that this may overflow and roll over to negative values, so don't assume
			// the input data is sorted. In fact, it might be better to just sample randomly here...
	}
	std::shuffle(host_side.input_data.begin(), host_side.input_data.end(), util::random::engine);
	host_side.extra_datum = base_value + data_size * delta;

	cuda::memory::copy(device_side.data.get(), host_side.input_data.data(), host_side.input_data.size() * sizeof(T) );
	cuda::memory::copy(device_side.extra_datum.get(), &host_side.extra_datum, sizeof(T));

	cuda::launch(
		::kernels::test_exchange<T, elements_per_thread>,
		launch_config,
		device_side.extra_datum.get(),
		device_side.data.get(),
		data_size);

	cuda::outstanding_error::ensure_none();

	host_side.output_data.resize(data_size + 1);
	cuda::memory::copy(host_side.output_data.data(), device_side.data.get(), data_size * sizeof(T) );
	cuda::memory::copy(host_side.output_data.data() + data_size, device_side.extra_datum.get(), sizeof(T));

	bool print_results { false };

	host_side.input_data.push_back(host_side.extra_datum);
	std::sort(host_side.output_data.begin(), host_side.output_data.end());
	std::sort(host_side.input_data.begin(), host_side.input_data.end());

	auto mismatch_pair = std::mismatch(host_side.input_data.begin(), host_side.input_data.end(), host_side.output_data.begin());

	CHECK(mismatch_pair.first == host_side.input_data.end());
	if (mismatch_pair.first != host_side.input_data.end()) {
		std::cerr
			<< "Input data element #" << std::distance(host_side.input_data.begin(), mismatch_pair.first)
			<< "(in sorted order) is " << promote_for_streaming(*(mismatch_pair.first))
			<< " while output data element #" << std::distance(host_side.output_data.begin(), mismatch_pair.second)
			<< "(in sorted order) is " << promote_for_streaming(*(mismatch_pair.second)) << std::endl;
	}

	if (print_results) {
		std::cout << "Results as expected for type " << util::type_name<T>() << ".\n";
	}
}

/*

kind of test:

Sample I.I.D. short sequence sequence of increments and decrements
(sample a modulus perhaps? not right now)
(perhaps have each warp wait for some random amount of time? not right now)
Initialize an arbitrary global value
each thread applies increments and decrements according to one sampled sequence
ensure the overall result is what it should be

increment  (T* address, T modulus) + decrement  (T* address, T modulus);
*/

TEST_CASE_TEMPLATE("increment and decrement", T,
	short, int, long, long long,
	unsigned short, unsigned int, unsigned long, unsigned long long,
	float, double
	)
{
	cuda::device_t device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	const auto num_grid_blocks { 3 };
	const auto block_size { kat::warp_size * 5 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	auto num_grid_threads = num_threads_in_grid(launch_config);
	struct {
		decltype(cuda::memory::device::make_unique<seq_type[]>(device, num_grid_threads)) inc_dec_sequences;
		decltype(cuda::memory::device::make_unique<T>(device)) aggregate;
			// Using largest_type_size so we can safely CAS into the result
	} device_side;
	device_side.inc_dec_sequences = cuda::memory::device::make_unique<seq_type[]>(device, num_grid_threads);
	device_side.aggregate = cuda::memory::device::make_unique<T>(device);

	struct  {
		vector<seq_type> inc_dec_sequences;
		T aggregate;
		T expected_aggregate;
	} host_side;

	host_side.inc_dec_sequences.reserve(num_grid_threads);
	auto base_value = static_cast<T>(std::numeric_limits<T>::max() / 2);
		// Want to make sure we don't underflow or overflow with many decrements
		// or increments. That's still a danger with 8-bit T's though.
	if (std::is_floating_point<T>::value) { base_value += T(1/3.0); }
	std::normal_distribution<float> distribution(0, 1);
	auto& engine = util::random::engine;
	host_side.aggregate = base_value + 5 * std::sqrt(num_grid_threads);
	host_side.expected_aggregate = host_side.aggregate;

	for(size_t i = 0; i < num_grid_threads; i++) {
		seq_type seq;
		for(auto j = 0; j < seq_type::ops_per_thread; j++) {
			bool do_inc = (util::random::sample_from(distribution, engine) > 0);
			seq[j] = do_inc;
			host_side.expected_aggregate += do_inc ? 1 : -1;
		}
		host_side.inc_dec_sequences.push_back(seq);
	}

	cuda::memory::copy(device_side.inc_dec_sequences.get(), host_side.inc_dec_sequences.data(), host_side.inc_dec_sequences.size() * sizeof(seq_type) );
	cuda::memory::copy_single(device_side.aggregate.get(), &host_side.aggregate);

	cuda::launch(
		::kernels::test_inc_dec_sequences<T, seq_type>,
		launch_config,
		device_side.aggregate.get(),
		device_side.inc_dec_sequences.get());

	cuda::outstanding_error::ensure_none();

	cuda::memory::copy_single(&host_side.aggregate, device_side.aggregate.get());

	if (std::is_floating_point<T>::value) {
		// For some strange reason, this fails:
		//
		//   const T tolerance = T(10^-5);
		//   CHECK(host_side.aggregate == doctest::Approx(host_side.expected_aggregate).epsilon(tolerance));
		//
		// while this succeeds:
		CHECK(host_side.aggregate == host_side.expected_aggregate);
		// ... isn't that weird?
	}
	else {
		CHECK(host_side.aggregate == host_side.expected_aggregate);
	}
}

// Note: Testcases before this point in the file were written before the infrastructure
// in later test suite files was available (e.g. block_collaboration, sequence_ops) -
// even though it now appears this file
// From here on we'll try using that


TEST_CASE_TEMPLATE("min - random values from host", T, INTEGER_TYPES, FLOAT_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = num_grid_blocks * num_threads_per_block;

	vector<T> input_data;
	auto mean { middle_of_domain<T>() };
	auto standard_deviation { nice_big_value<T>() / 16.0 };
	std::normal_distribution<double> distribution(mean, standard_deviation);
	input_data.reserve(input_length);
	util::random::insertion_generate_n(std::back_inserter(input_data), input_length, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	T expected_result {
		*std::min_element(input_data.begin(), input_data.end())
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::min(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	constexpr const T fill_value { std::numeric_limits<T>::max() };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	// Note: We check for exact equality here even for floating-point types,
	// since there is no arithmetic performed
	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("max - random values from host", T, INTEGER_TYPES, FLOAT_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = num_grid_blocks * num_threads_per_block;

	vector<T> input_data;
	auto mean { middle_of_domain<T>() };
	auto standard_deviation { nice_big_value<T>() / 16.0 };
	std::normal_distribution<double> distribution(mean, standard_deviation);
	input_data.reserve(input_length);
	util::random::insertion_generate_n(std::back_inserter(input_data), input_length, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	T expected_result {
		*std::max_element(input_data.begin(), input_data.end())
	};

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::max(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	constexpr const T fill_value { std::numeric_limits<T>::min() };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	// Note: We check for exact equality here even for floating-point types,
	// since there is no arithmetic performed
	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("min - single outlier", T, INTEGER_TYPES, FLOAT_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = num_grid_blocks * num_threads_per_block;

	auto uniform_value { middle_of_domain<T>() };
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos]--; // this is the minimum!

	auto expected_result { T(uniform_value - T(1)) };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::min(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	constexpr const T fill_value { std::numeric_limits<T>::max() };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	// Note: We check for exact equality here even for floating-point types,
	// since there is no arithmetic performed
	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("max - single outlier", T, INTEGER_TYPES, FLOAT_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	auto uniform_value { middle_of_domain<T>() };
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos]++; // this is the maximum!

	auto expected_result { T(uniform_value + T(1)) };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::max(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	constexpr const T fill_value { std::numeric_limits<T>::min() };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	// Note: We check for exact equality here even for floating-point types,
	// since there is no arithmetic performed
	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_and - single outlier", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = num_grid_blocks * num_threads_per_block;

	T uniform_value(true);
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos] = false; // this is the conjunction value!

	T expected_result(false);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::logical_and(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	const T fill_value { uniform_value };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_or - single outlier", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = num_grid_blocks * num_threads_per_block;

	T uniform_value(false);
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos] = true; // this is the disjunction value!

	T expected_result(true);

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::logical_or(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	const T fill_value { uniform_value };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_xor - single outlier 0", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	T uniform_value(1);
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos] = 0;

	const T fill_value { 0 };
	T expected_result( (input_length-1) % 2 );

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::logical_xor(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_xor - single outlier 1", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	T uniform_value(0);
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);
	input_data[outlier_pos] = 1;

	T expected_result( 1 );
	const T fill_value { 0 };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::logical_xor(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}



TEST_CASE_TEMPLATE("logical_not - single non-negator", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);

	const T fill_value { 0 };
	T expected_result( (input_length-1) % 2 );

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate)
		{
			namespace gi = kat::linear_grid::grid_info;

			// TODO: Should I sleep here? Use a block::barrier()?
			if (not (outlier_pos == gi::thread::global_id()) ) {
				kat::atomic::logical_not(aggregate);
			}
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_not - single negater", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	auto outlier_pos = util::random::sample_from(distribution);

	T expected_result( 1 );
	const T fill_value { 0 };

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate)
		{
			namespace gi = kat::linear_grid::grid_info;

			if (outlier_pos == gi::thread::global_id() ) {
				kat::atomic::logical_not(aggregate);
			}
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("logical_not - by random threads", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	vector<fake_bool> perform_op_indicators;
	uniform_int_distribution<fake_bool> distribution(0, 1);
	perform_op_indicators.reserve(input_length);
	util::random::insertion_generate_n(std::back_inserter(perform_op_indicators), input_length, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	constexpr const T fill_value (0);
	T expected_result (fill_value);
	std::for_each(
		std::cbegin(perform_op_indicators), std::cend(perform_op_indicators),
		[&](bool b){ if(b) expected_result = not expected_result; });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T*               __restrict target,
			const fake_bool* __restrict perform_op_indicators)
		{
			namespace gi = kat::linear_grid::grid_info;
			bool perform_op = perform_op_indicators[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			if (perform_op) {
				kat::atomic::logical_not(target);
			}
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		perform_op_indicators.data()
	);
	T result { result_container[0] };

	// Note: We check for exact equality here even for floating-point types,
	// since there is no arithmetic performed
	CHECK(result == expected_result);
}


TEST_CASE_TEMPLATE("bitwise_and - single outliers", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	T uniform_value(~T(0));
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	constexpr const auto no_outlier_at_this_bit_index = kat::size_in_bits<T>();
	vector<decltype(kat::size_in_bits<T>())> outlier_positions(kat::size_in_bits<T>(), no_outlier_at_this_bit_index);
	T expected_result(~T(0));
	// An outlier for every second bit, starting from the LSB
	for(auto bit_index = 1; bit_index < kat::size_in_bits<T>(); bit_index += 2) {
		outlier_positions[bit_index] = util::random::sample_from(distribution);
		input_data[outlier_positions[bit_index]] &= ~(T(1) << bit_index);
		expected_result &= ~(T(1) << bit_index);
	}

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::bitwise_and(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	const T fill_value { uniform_value };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("bitwise_or - single outliers", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	T uniform_value(0);
	vector<T> input_data(input_length, uniform_value);
	uniform_int_distribution<std::size_t> distribution(0, input_length - 1);
	constexpr const auto no_outlier_at_this_bit_index = kat::size_in_bits<T>();
	vector<decltype(kat::size_in_bits<T>())> outlier_positions(kat::size_in_bits<T>(), no_outlier_at_this_bit_index);
	T expected_result(0);
	// An outlier for every second bit, starting from the LSB
	for(auto bit_index = 1; bit_index < kat::size_in_bits<T>(); bit_index += 2) {
		outlier_positions[bit_index] = util::random::sample_from(distribution);
		input_data[outlier_positions[bit_index]] |= (T(1) << bit_index);
		expected_result |= (T(1) << bit_index);
	}

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict aggregate,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::bitwise_or(aggregate, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	const T fill_value { uniform_value };
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("bitwise_xor - random values from host", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };
//	constexpr const unsigned elements_per_thread { 3 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	vector<T> input_data;
	uniform_int_distribution<T> distribution(0, std::numeric_limits<T>::max());
	input_data.reserve(input_length);
	util::random::insertion_generate_n(std::back_inserter(input_data), input_length, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	T expected_result (0);
	std::for_each(std::cbegin(input_data), std::cend(input_data), [&](T x){ expected_result ^= x; });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict target,
			const T* __restrict input_data)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto thread_element = input_data[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			auto prev = kat::atomic::bitwise_xor(target, thread_element);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	constexpr const T fill_value (0);
	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		input_data.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("bitwise_not - by random threads", T, INTEGER_TYPES) {
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	auto input_length = // elements_per_thread *
		num_grid_blocks * num_threads_per_block;

	vector<fake_bool> perform_op_indicators;
	uniform_int_distribution<fake_bool> distribution(0, 1);
	perform_op_indicators.reserve(input_length);
	util::random::insertion_generate_n(std::back_inserter(perform_op_indicators), input_length, distribution);
		// Note that we could easily have overflow/underflow occurring with smaller types

	constexpr const T fill_value (T(0xDEADBEEFCAFEBABEllu));
	T expected_result (fill_value);
	std::for_each(
		std::cbegin(perform_op_indicators), std::cend(perform_op_indicators),
		[&](bool b){ if(b) expected_result = ~expected_result; });

	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T*               __restrict target,
			const fake_bool* __restrict perform_op_indicators)
		{
			namespace gi = kat::linear_grid::grid_info;
			bool perform_op = perform_op_indicators[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			if (perform_op) {
				kat::atomic::bitwise_not(target);
			}
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		perform_op_indicators.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("set_bit - few outliers", T, long int) { // INTEGER_TYPES) {
	using bit_index_type = unsigned;
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	size_t input_length = num_grid_blocks * num_threads_per_block;
	const size_t num_outliers = size_in_bits<T>() / 3;

	vector<bit_index_type> half_bit_indices;
	uniform_int_distribution<bit_index_type> half_bit_index_distribution(0, size_in_bits<T>() / 2 - 1);
	util::random::insertion_generate_n(std::back_inserter(half_bit_indices), input_length , half_bit_index_distribution);

	vector<bit_index_type> bit_indices;
	std::transform(std::cbegin(half_bit_indices), std::cend(half_bit_indices), std::back_inserter(bit_indices),
		[](auto half_bit_index) { return half_bit_index * 2 + 1; }
	);

	auto outlier_positions = util::random::sample_index_subset(input_length, num_outliers);
	auto outlier_half_bit_indices = util::random::sample_index_subset(size_in_bits<T>() / 2, num_outliers);

	auto outlier_positions_iter = std::cbegin(outlier_positions);
	auto outlier_half_bit_indices_iter = std::cbegin(outlier_half_bit_indices);
	for(size_t i = 0; i < num_outliers; i++) {
		auto outlier_position = *(outlier_positions_iter++);
		auto half_bit_index = *(outlier_half_bit_indices_iter++);
		bit_indices[outlier_position] -= 1;
			// the outliers set even bits instead of odd ones
	}

	constexpr const T fill_value = T(0xDEADBEEFCAFED00Dlu);
	T expected_result (fill_value);

	for(auto bit_index : bit_indices) { expected_result |= (T(1) << bit_index); }


	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict target,
			const bit_index_type* __restrict bit_indices
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto bit_index = bit_indices[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			kat::atomic::set_bit(target, bit_index);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		bit_indices.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

TEST_CASE_TEMPLATE("unset_bit - few outliers", T, long int) { // INTEGER_TYPES) {
	using bit_index_type = unsigned;
	auto device = cuda::device::current::get();
	cuda::grid::dimension_t num_grid_blocks = device.properties().multiProcessorCount * 2;
	cuda::grid::block_dimension_t num_threads_per_block { kat::warp_size * 5 };

	size_t input_length = num_grid_blocks * num_threads_per_block;
	const size_t num_outliers = size_in_bits<T>() / 3;

	vector<bit_index_type> half_bit_indices;
	uniform_int_distribution<bit_index_type> half_bit_index_distribution(0, size_in_bits<T>() / 2 - 1);
	util::random::insertion_generate_n(std::back_inserter(half_bit_indices), input_length , half_bit_index_distribution);

	vector<bit_index_type> bit_indices;
	std::transform(std::cbegin(half_bit_indices), std::cend(half_bit_indices), std::back_inserter(bit_indices),
		[](auto half_bit_index) { return half_bit_index * 2 + 1; }
	);

	auto outlier_positions = util::random::sample_index_subset(input_length, num_outliers);
	auto outlier_half_bit_indices = util::random::sample_index_subset(size_in_bits<T>() / 2, num_outliers);

	auto outlier_positions_iter = std::cbegin(outlier_positions);
	auto outlier_half_bit_indices_iter = std::cbegin(outlier_half_bit_indices);
	for(size_t i = 0; i < num_outliers; i++) {
		auto outlier_position = *(outlier_positions_iter++);
		auto half_bit_index = *(outlier_half_bit_indices_iter++);
		bit_indices[outlier_position] -= 1;
			// the outliers set even bits instead of odd ones
	}

	constexpr const T fill_value = T(0xDEADBEEFCAFED00Dlu);
	T expected_result (fill_value);

	for(auto bit_index : bit_indices) { expected_result &= ~(T(1) << bit_index); }


	auto testcase_device_function =
		[=] KAT_DEV (
			size_t,
			T* __restrict target,
			const bit_index_type* __restrict bit_indices
			)
		{
			namespace gi = kat::linear_grid::grid_info;
			auto bit_index = bit_indices[gi::thread::global_index()];

			// TODO: Should I sleep here? Use a block::barrier()?
			kat::atomic::unset_bit(target, bit_index);
		};

	const auto num_values_to_populate { input_length };
		// Note: The result array will also have this many values - but we won't be using them.

	auto result_container = execute_non_uniform_testcase_on_gpu(
		testcase_device_function,
		num_values_to_populate,
		fill_value,
		num_grid_blocks, num_threads_per_block,
		bit_indices.data()
	);
	T result { result_container[0] };

	CHECK(result == expected_result);
}

// Note: Not testing apply_atomically, since half the tests here actually test just that -
// functions that are implemented using apply_atomically().

} // TEST_SUITE("atomics")
