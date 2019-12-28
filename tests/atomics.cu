#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/math.cuh>
#include <kat/on_device/shuffle.cuh>
#include <kat/containers/array.hpp>
#include <kat/on_device/atomics.cuh>
#include <random>
#include <numeric>
#include <cmath>

// TODO: Run some/all tests for half-precision floating-point values, e.g __half from:
// #include <cuda_fp16.h>

// TODO: Also test behavior with warps with some inactive/exited lanes

#include <kat/define_specifiers.hpp>

constexpr const auto num_grid_blocks {   2 }; // wanted to make sure I sature a GPU's SM's

// constexpr const auto num_full_warps  {   1 }; // wanted to make sure I sature a GPU's SM's warp scheduler


// constexpr const auto block_size { num_full_warps * kat::warp_size };
constexpr const auto block_size { 32 };
	// TODO: It may be useful to test with non-full warps
constexpr const auto elements_per_thread { 3 }; // just something that's not 1

namespace kernels {


template <typename T>
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

	std::size_t pos = global_thread_index * elements_per_thread;
	for(int i = 0; (i < elements_per_thread) and (pos < data_size); i++)
	{
		kat::atomic::add(result, data[pos]);
	    pos++;
	}
}

template <typename T>
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

	std::size_t pos = global_thread_index * elements_per_thread;
	for(int i = 0; (i < elements_per_thread) and (pos < data_size); i++)
	{
		//thread_printf("Using datum %f at pos %2u", (double) data[pos], pos);
		//::kat::linear_grid::tprintline("Using datum %f at pos %2u", (double) data[pos], pos);

//		{
//			auto prefixed_format_str = kat::linear_grid::detail::get_prefixed_format_string("Using datum %f at pos %2u");
//			printf(prefixed_format_str, (double) data[pos], pos);
//			free(prefixed_format_str);
//		}

		kat::atomic::subtract(result, data[pos]);
	    pos++;
	}
}


template <typename T>
__global__ void test_exchange(
	T* __restrict__  extra_datum,
	T* __restrict__  data,
	std::size_t      data_size)
{
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;

	std::size_t pos = global_thread_index * elements_per_thread;
	for(int i = 0; (i < elements_per_thread) and (pos < data_size); i++)
	{
		auto datum_to_use = data[pos];
		auto previous_extra_datum = kat::atomic::exchange(extra_datum, datum_to_use);
		data[pos] = previous_extra_datum;
	    pos++;
	}
}



} // namespace kernels

// TODO:
// * Test between_or_equal and strictly_between with differing types for all 3 arguments
// * Some floating-point tests
// * gcd tests with values of different types
// * Some tests with negative values

#define INSTANTIATE_CONSTEXPR_MATH_TEST(_tp) \
	compile_time_execution_results<_tp> UNIQUE_IDENTIFIER(test_struct_); \

#define INTEGER_TYPES \
	int8_t, int16_t, \
	int32_t, int64_t, \
	uint8_t, uint16_t, uint32_t, uint64_t, \
	char, short, int, long, long long, \
	signed char, signed short, signed int, signed long, signed long long, \
	unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long

#define FLOAT_TYPES float, double

#define ARRAY_TYPES_BY_SIZE  \
	kat::array<uint8_t, 4>, \
	kat::array<uint8_t, 7>, \
	kat::array<uint8_t, 8>, \
	kat::array<uint8_t, 9>, \
	kat::array<uint8_t, 15>, \
	kat::array<uint8_t, 16>, \
	kat::array<uint8_t, 17>, \
	kat::array<uint8_t, 31>, \
	kat::array<uint8_t, 32>, \
	kat::array<uint8_t, 33>


size_t num_threads_in_grid(const cuda::launch_configuration_t& launch_config)
{
	return static_cast<size_t>(launch_config.grid_dimensions.volume()) * launch_config.block_dimensions.volume();
}

enum { largest_type_size = 8  };

TEST_SUITE("atomics") {

TEST_CASE_TEMPLATE("add", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
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
		std::vector<T> data;
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
		::kernels::test_add<T>,
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
		CHECK(host_side.result == doctest::Approx(expected_result));
	}
	else {
		CHECK(host_side.result == expected_result); // what about floating-point types?
	}

	if (host_side.result != doctest::Approx(expected_result)) {
		std::cout << "Atomic add kernel result: " << promote_for_streaming(host_side.result) << ' ' << std::hex << promote_for_streaming(host_side.result) <<  std::dec <<  '\n';
		std::cout << "Expected result:          " << promote_for_streaming(expected_result)  << ' ' << std::hex << promote_for_streaming(expected_result) <<  std::dec <<'\n';
	}

	if (print_results) {
		std::cout << "Results match for type " << util::type_name<T>() << ".\n";
	}
}

TEST_CASE_TEMPLATE("subtract", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
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
		std::vector<T> data;
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
		::kernels::test_subtract<T>,
		launch_config,
		device_side.result.get(),
		device_side.data.get(),
		data_size);

	cuda::outstanding_error::ensure_none();

	cuda::memory::copy(&host_side.result, device_side.result.get(), sizeof(T));

	bool print_results { false };

	if (std::is_floating_point<T>::value) {
		CHECK(host_side.result == doctest::Approx(expected_result));
	}
	else {
		CHECK(host_side.result == expected_result); // what about floating-point types?
	}

	if (host_side.result != doctest::Approx(expected_result)) {
		std::cout << "Atomic subtract kernel result: " << promote_for_streaming(host_side.result) << ' ' << std::hex << promote_for_streaming(host_side.result) <<  std::dec <<  '\n';
		std::cout << "Expected result:               " << promote_for_streaming(expected_result)  << ' ' << std::hex << promote_for_streaming(expected_result) <<  std::dec <<'\n';
	}

	if (print_results) {
		std::cout << "Results match for type " << util::type_name<T>() << ".\n";
	}
}

TEST_CASE_TEMPLATE("exchange", T, INTEGER_TYPES, FLOAT_TYPES )
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
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
		std::vector<T> input_data;
		std::vector<T> output_data;
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
		::kernels::test_exchange<T>,
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

uniformly sampled permutation of a lot (?) of data - may have problem with range of possible values for small types
perhaps have each warp wait for some random amount of time?
each thread applies the atomic op with the value
ensure the overall result is what it should be


add        (T* address);
subtract   (T* address);
exchange   (T* address);

sample a modulus perhaps?
perhaps have each warp wait for some random amount of time?
each thread applies the atomic op with the value
ensure the overall result is what it should be

increment  (T* address, T modulus);
decrement  (T* address, T modulus);


uniformly sampled permutation of a lot (?) of data - may have problem with range of possible values for small types
perhaps have each warp wait for some random amount of time?
have each thread use 1 (maybe more?) atomics sequentially

min        (T* address);
max        (T* address);

all-zeros, all-ones, single-zero, single-one, all-random (and flip one if necessary for xor result)
perhaps have each warp wait for some random amount of time?
have each thread use 1 (maybe more?) atomics sequentially

logical_and(T* address);
logical_or (T* address);
logical_xor(T* address);

these are basically apply_atomically, so just test that I guess.

bitwise_or (T* address);
bitwise_and(T* address);
bitwise_xor(T* address);
bitwise_not(T* address);
set_bit    (T* address);
unset_bit  (T* address);
apply_atomically


here we'll want to check that the (distinct?) elements of a set just get shuffled around
and none of them gets duplicated

compare_and_swap


 */

#include <kat/undefine_specifiers.hpp>

} // TEST_SUITE("atomics")
