#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/miscellany.cuh>
#include <limits>

/*

To test:


KAT_FD  void swap(T& x, T& y);

KAT_FD T* copy(
	T*        __restrict__  destination,
	const T*  __restrict__  source,
	size_t                  num_elements_to_copy)

constexpr KAT_FHD I num_warp_sizes_to_cover(I x)

 */

START_COUNTING_LINES(num_swap_checks);

template <typename T>
KAT_DEV bool single_swap_test(T x, T y)
{
	T x_ { x };
	T y_ { y };
	kat::swap<T>(x_, y_);
	return (x == y_) and (y == x_);
}

namespace kernels {


template <typename T>
__global__
void swap_tests(bool* results)
{
//	bool print_first_indices_for_each_function { false };
	auto i = 0;

//	auto maybe_print = [&](const char* section_title) {
//		if (print_first_indices_for_each_function) {
//			printf("%-30s tests start at index  %3d\n", section_title, i);
//		}
//	};
//
//	maybe_print("single_swap_test");
	results[i++] = single_swap_test<T>(T{0}, T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{2}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{2}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::min(), T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, std::numeric_limits<T>::min()); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::max(), T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, std::numeric_limits<T>::max()); COUNT_THIS_LINE

// #define num_swap_checks 11

}

} // namespace kernels

FINISH_COUNTING_LINES(num_swap_checks);

// TODO:
// * Test between_or_equal and strictly_between with differing types for all 3 arguments
// * Some floating-point tests
// * gcd tests with values of different types
// * Some tests with negative values

#define INSTANTIATE_CONSTEXPR_MATH_TEST(_tp) \
	compile_time_execution_results<_tp> UNIQUE_IDENTIFIER(test_struct_); \

#define INTEGER_TYPES \
	int8_t, int16_t, int32_t, int64_t, \
	uint8_t, uint16_t, uint32_t, uint64_t, \
	char, short, int, long, long long, \
	signed char, signed short, signed int, signed long, signed long long, \
	unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long



TEST_SUITE("miscellany") {

/*
TEST_CASE_TEMPLATE("test native-word copy", I, int8_t, int16_t, int32_t, uint32_t, int64_t, float, double )
{
	// Data arrays here
	cuda::device_t<> device { cuda::device::current::get() };
	auto block_size { 1 };
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	auto device_side_results { cuda::memory::device::make_unique<I[]>(device, num_swap_checks) };
	auto device_side_expected_results { cuda::memory::device::make_unique<I[]>(device, num_swap_checks) };
	auto host_side_results { std::unique_ptr<I[]>(new I[num_swap_checks]) };
	auto host_side_expected_results { std::unique_ptr<I[]>(new I[num_swap_checks]) };

	cuda::launch(
		kernels::try_out_integral_math_functions<I>,
		launch_config,
		device_side_results.get(), device_side_expected_results.get());

	cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(I) * num_swap_checks);
	cuda::memory::copy(host_side_expected_results.get(), device_side_expected_results.get(), sizeof(I) * num_swap_checks);

	for(auto i { 0 }; i < num_swap_checks; i++) {
		CHECK(host_side_results.get()[i] == host_side_expected_results.get()[i]);
		if (host_side_results.get()[i] != host_side_expected_results.get()[i]) {
			MESSAGE("index of failure was: " << i);
		}
	}
}
*/

// TODO: Consider using larger-than-64-bit types, classes, etc.
TEST_CASE_TEMPLATE("test swap", T, int8_t, int16_t, int32_t, uint32_t, int64_t, float, double )
{
	cuda::device_t<> device { cuda::device::current::get() };
	cuda::launch_configuration_t launch_config { cuda::grid::dimensions_t::point(), cuda::grid::dimensions_t::point() };
	auto device_side_results { cuda::memory::device::make_unique<bool[]>(device, num_swap_checks) };
	auto host_side_results { std::unique_ptr<bool[]>(new bool[num_swap_checks]) };

	cuda::launch(
		kernels::swap_tests<T>,
		launch_config,
		device_side_results.get());

	cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(bool) * num_swap_checks);

	for(auto i = 0 ; i < num_swap_checks; i++) {
		if (not host_side_results.get()[i]) {
			MESSAGE("The " << i << xth(i) << " swap check failed.");
		}
	}

}

//TEST_CASE_TEMPLATE("test num_warps_to_cover", I, uint8_t, uint16_t, uint32_t, uint64_t )
//{
//
//}

} // TEST_SUITE("miscellany")
