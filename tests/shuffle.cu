#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/shuffle.cuh>
#include <kat/containers/array.hpp>

// TODO: Run some/all tests for half-precision floating-point values, e.g __half from:
// #include <cuda_fp16.h>

// TODO: Also test behavior with warps with some inactive/exited lanes

#include <kat/define_specifiers.hpp>

namespace kernels {

template <typename T, std::size_t N>
__fhd__  kat::array<T, N>& operator++(::kat::array<T, N>& x)
{
	for(auto& e : x) { e++; }
	return x;
}

template <typename T, std::size_t N>
__fhd__ kat::array<T, N> operator++(::kat::array<T, N>& x, int)
{
	kat::array<T, N> copy;
	for(auto& e : x) { e++; }
	return copy;
}

// TODO: Add __restrict__ to these kernels ... but that triggers a bug, for some reason, with CUDA 9.2


template <typename T>
__global__ void test_shuffle_up(
	const                 T* unshuffled,
	                      T* shuffled,
	unsigned              delta)
{
	assert(gridDim.y == 1 and blockDim.y == 1);
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	T datum { kat::shuffle_up(unshuffled[global_thread_index], delta) };
	shuffled[global_thread_index] = datum;
}

template <typename T>
__global__ void test_shuffle_down(
	const                  T* unshuffled,
	                       T* shuffled,
	unsigned              delta)
{
	assert(gridDim.y == 1 and blockDim.y == 1);
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	T datum { kat::shuffle_down(unshuffled[global_thread_index], delta) };
	shuffled[global_thread_index] = datum;
}

template <typename T>
__global__ void test_shuffle_xor(
	const                 T* unshuffled,
	                      T* shuffled,
	const int             mask)
{
	assert(gridDim.y == 1 and blockDim.y == 1);
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	// thread_printf("__shfl_xor_sync(%X, %d, %X, %d)", kat::full_warp_mask, 123, mask, kat::warp_size);
	T datum {
		// unshuffled[global_thread_index]
		kat::shuffle_xor(unshuffled[global_thread_index], mask)
		// kat::builtins::warp::shuffle::xor_(unshuffled[global_thread_index], mask)
		// shfl_xor_sync(kat::full_warp_mask, unshuffled[global_thread_index], mask, kat::warp_size)
		// 1000 + unshuffled[global_thread_index]
		//123
	};
	shuffled[global_thread_index] = datum;
}

template <typename T, typename F>
__global__ void test_arbitrary_shuffle(
	const                 T* unshuffled,
	                      T* shuffled,
	                      F get_source_lane_for)
{
	assert(gridDim.y == 1 and blockDim.y == 1);
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	auto lane_index = threadIdx.x % kat::warp_size;
	auto shuffle_source_lane = get_source_lane_for(lane_index);
	T datum { kat::shuffle_arbitrary(unshuffled[global_thread_index], shuffle_source_lane) };
	shuffled[global_thread_index] = datum;
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

constexpr const auto num_full_warps { 2 }; // this is aribtrary; didn't just want to have 1.
constexpr const auto block_size { num_full_warps * kat::warp_size };


TEST_SUITE("shuffle") {

TEST_CASE_TEMPLATE("up", I, INTEGER_TYPES, FLOAT_TYPES ) //, ARRAY_TYPES_BY_SIZE)
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	auto device_side_unshuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	auto device_side_shuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	std::array<I, block_size> host_side_unshuffled;
	std::array<I, block_size> host_side_shuffled;

	std::iota(host_side_unshuffled.begin(),host_side_unshuffled.end(), 0);

	std::array<I, block_size> host_side_expected_shuffled;

	for(int delta = 0; delta < kat::warp_size; delta++) {

		for(std::size_t pos { 0 }; pos < host_side_expected_shuffled.size(); pos ++) {
			// Note: I wonder if it's a good idea not to use a typedef for lane indices.
			unsigned lane_index = pos % kat::warp_size;
			auto shuffle_origin_pos = (lane_index >= delta) ? (pos - delta) : pos;
			host_side_expected_shuffled[pos] = host_side_unshuffled[shuffle_origin_pos];
		}

		cuda::memory::copy(device_side_unshuffled.get(), host_side_unshuffled.data(), sizeof(host_side_unshuffled));

		cuda::launch(
			::kernels::test_shuffle_up<I>,
			launch_config,
			device_side_unshuffled.get(),
			device_side_shuffled.get(),
			delta);

		cuda::memory::copy(host_side_shuffled.data(), device_side_shuffled.get(), sizeof(host_side_shuffled));

		constexpr const auto print_results { false };
		auto found_discrepancy { false };
		for(auto i { 0 }; i < block_size; i++) {
			CHECK(host_side_shuffled[i] == host_side_expected_shuffled[i]);
			if (host_side_shuffled[i] != host_side_expected_shuffled[i]) {
				found_discrepancy = true;
				MESSAGE("index of discrepancy was: " << i);
			}
		}
		if (print_results) {
			if (found_discrepancy) {
				std::cout << "Unshuffled input:\n" << host_side_unshuffled << '\n';
				std::cout << "Input shuffled up with delta = " << delta <<  ":\n" << host_side_unshuffled << '\n';
				std::cout << "Expected shuffled up output :                   \n" << host_side_expected_shuffled << '\n';
			}
			else {
				std::cout << "No discrepancies for type = " << util::type_name<I>() << ", delta = " << delta << ".\n";
			}
		}
	}
}

TEST_CASE_TEMPLATE("down", I, INTEGER_TYPES, FLOAT_TYPES ) //, ARRAY_TYPES_BY_SIZE)
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	auto device_side_unshuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	auto device_side_shuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	std::array<I, block_size> host_side_unshuffled;
	std::array<I, block_size> host_side_shuffled;

	std::iota(host_side_unshuffled.begin(),host_side_unshuffled.end(), 0);

	std::array<I, block_size> host_side_expected_shuffled;

	for(int delta = 0; delta < kat::warp_size; delta++) {

		for(std::size_t pos { 0 }; pos < host_side_expected_shuffled.size(); pos ++) {
			// Note: I wonder if it's a good idea not to use a typedef for lane indices.
			unsigned lane_index = pos % kat::warp_size;
			auto shuffle_origin_pos = (lane_index < kat::warp_size - delta) ? (pos + delta) : pos;
			host_side_expected_shuffled[pos] = host_side_unshuffled[shuffle_origin_pos];
		}

		cuda::memory::copy(device_side_unshuffled.get(), host_side_unshuffled.data(), sizeof(host_side_unshuffled));

		cuda::launch(
			::kernels::test_shuffle_down<I>,
			launch_config,
			device_side_unshuffled.get(),
			device_side_shuffled.get(),
			delta);

		cuda::memory::copy(host_side_shuffled.data(), device_side_shuffled.get(), sizeof(host_side_shuffled));

		constexpr const auto print_results { false };
		auto found_discrepancy { false };
		for(auto i { 0 }; i < block_size; i++) {
			CHECK(host_side_shuffled[i] == host_side_expected_shuffled[i]);
			if (host_side_shuffled[i] != host_side_expected_shuffled[i]) {
				found_discrepancy = true;
				MESSAGE("index of discrepancy was: " << i);
			}
		}
		if (print_results) {
			if (found_discrepancy) {
				std::cout << "Unshuffled input:\n" << host_side_unshuffled << '\n';
				std::cout << "Input shuffled up with delta = " << delta <<  ":\n" << host_side_unshuffled << '\n';
				std::cout << "Expected shuffled up output :                   \n" << host_side_expected_shuffled << '\n';
			}
			else {
				std::cout << "No discrepancies for type = " << util::type_name<I>() << ", delta = " << delta << ".\n";
			}
		}
	}
}

TEST_CASE_TEMPLATE("xor", I, INTEGER_TYPES, FLOAT_TYPES ) //, ARRAY_TYPES_BY_SIZE)
{
	cuda::device_t<> device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	auto device_side_unshuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	auto device_side_shuffled { cuda::memory::device::make_unique<I[]>(device, block_size) };
	std::array<I, block_size> host_side_unshuffled;
	std::array<I, block_size> host_side_shuffled;

	std::iota(host_side_unshuffled.begin(),host_side_unshuffled.end(), 0);

	std::array<I, block_size> host_side_expected_shuffled;

	for(size_t mask_index { 0 }; mask_index < kat::warp_size; mask_index++) {

		// Note the mask can't have bits that aren't present in actual lane indices,
		// so the mask does nor exceed warp_size - 1

//		std::uniform_int_distribution<kat::lane_mask_t> distribution(kat::empty_warp_mask, kat::full_warp_mask);
//		// util::random::seed(std::time(0)); // seed with the current time
//		auto mask = util::random::sample_from(distribution);

		int mask = mask_index; // yes, just like that

		// std::cout << "Using mask " << std::hex << (unsigned) mask << std::dec << std::endl;

		for(std::size_t pos { 0 }; pos < host_side_expected_shuffled.size(); pos ++) {
			// Note: I wonder if it's a good idea not to use a typedef for lane indices.
			unsigned lane_index = pos % kat::warp_size;
			auto shuffle_origin_pos = (pos - lane_index) ^ (lane_index xor mask);
			host_side_expected_shuffled[pos] = host_side_unshuffled[shuffle_origin_pos];
			// std::cout << "pos = " << std::setw(2) << pos << ", host_side_expected_shuffled[" << std::setw(2) << pos << "] = " << std::setw(2) << host_side_expected_shuffled[pos] << std::endl;
		}

		cuda::memory::copy(device_side_unshuffled.get(), host_side_unshuffled.data(), sizeof(host_side_unshuffled));


		cuda::launch(
			::kernels::test_shuffle_xor<I>,
			launch_config,
			device_side_unshuffled.get(),
			device_side_shuffled.get(),
			mask);

		cuda::memory::copy(host_side_shuffled.data(), device_side_shuffled.get(), sizeof(host_side_shuffled));

		constexpr const auto print_results { false };
		auto found_discrepancy { false };
		for(auto i { 0 }; i < block_size; i++) {
			CHECK(host_side_shuffled[i] == host_side_expected_shuffled[i]);
			if (host_side_shuffled[i] != host_side_expected_shuffled[i]) {
				found_discrepancy = true;
				MESSAGE("index of discrepancy was: " << i);
			}
		}
		if (print_results) {
			if (found_discrepancy) {
				std::cout << "Unshuffled input:\n" << host_side_unshuffled << '\n';
				std::cout << "Input shuffled up with mask = " << std::hex << mask << std::dec <<  ":\n" << host_side_unshuffled << '\n';
				std::cout << "Expected shuffled up output :                   \n" << host_side_expected_shuffled << '\n';
			}
			else {
				std::cout << "No discrepancies for type = " << util::type_name<I>() << ", mask = " << std::hex << mask << std::dec <<  ".\n";
			}
		}
	}
}

#include <kat/undefine_specifiers.hpp>

} // TEST_SUITE("shuffle")
