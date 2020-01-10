#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include "common.cuh"
// #include <kat/on_device/printing.cuh>
//#include "util/prettyprint.hpp"
#include "util/type_name.hpp"
//#include "util/random.hpp"
//#include "util/miscellany.cuh"
//#include "util/macro.h"


#include <doctest.h>
#include <cuda/api_wrappers.hpp>

constexpr const auto num_grid_blocks {  2 };
constexpr const auto block_size      {  3 };

namespace kernels {

template <typename T>
__global__ void print_stuff()
{
	printf("This is a plain printf() call.\n");
	// kat::sleep<kat::sleep_resolution::clock_cycles>(100);
	// thread_print("And this is another plain printf() call.\n");
	//thread_print("and this is a thread_print() call");
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


TEST_SUITE("printing") {

TEST_CASE_TEMPLATE("print_something", T, int)// INTEGER_TYPES)
{
	auto device { cuda::device::current::get() };
		// TODO: Test shuffles with non-full warps.
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	cuda::launch(::kernels::print_stuff<T>,	launch_config);
	cuda::outstanding_error::ensure_none();
	std::cout << "Have run the kernel for type name " << util::type_name<T>() << std::endl;

	// TODO: We could redirect the standard output stream into a buffer before launching the kernel,
	// then check the buffer contains what we want. However, this can probably be interfered with,
	// so I'm not sure it's a good idea even in principle.
	device.synchronize();
}

#include <kat/undefine_specifiers.hpp>

} // TEST_SUITE("printing")
