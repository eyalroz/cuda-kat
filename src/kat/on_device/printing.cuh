/**
 * @file on_device/printing.cuh
 *
 * @brief CUDA device-side functions for (formatted) printing from within kernels,
 * and debugging-related functions involving printing.
 *
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_PRINTING_CUH_
#define CUDA_KAT_ON_DEVICE_PRINTING_CUH_

#include <kat/on_device/grid_info.cuh>
#include <kat/on_device/miscellany.cuh>
#include <kat/on_device/builtins.cuh>

#include <math_functions.h>

// Necessary for printf()'ing in kernel code
#include <cstdio>


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {

namespace detail {
__fd__ unsigned num_digits_required_for(unsigned long long extremal_value)
{
	return ceilf(log10f(extremal_value));
}

template <typename T>
__fd__ unsigned get_bit(T x, unsigned bit_index) { return 0x1 & (x >> bit_index); }

template <typename T>
__fd__ const char* binary_representation (T x, char *target_buffer, unsigned field_width = 0)
{
	auto floor_log_of_x =
		  (CHAR_BIT * sizeof(T) - 1) - builtins::count_leading_zeros(x);

	unsigned num_bits = (x == 0) ? 0 : floor_log_of_x + 1;

	unsigned leading_zeros = max(field_width - num_bits, 0);
	#pragma unroll
	for(unsigned bit_index = 0; bit_index < leading_zeros; bit_index++) {
		target_buffer[bit_index] = '0';
	}

	#pragma unroll
	for(unsigned bit_index = 0; bit_index < num_bits; bit_index++) {
		target_buffer[leading_zeros + bit_index] = ((x & (1 << bit_index)) == 0) ? '0' : '1';
	}
	target_buffer[leading_zeros + num_bits] = '\0';
	return target_buffer;
}

__fd__ const char* true_or_false(bool x)
{
	return x ? "true" : "false";
}
__fd__ const char* yes_or_no(bool x)
{
	return x ? "yes" : "no";
}

constexpr __fd__ const char* ordinal_suffix(int n)
{
	return
		(n % 100 == 1 ? "st" :
		(n % 100 == 2 ? "nd" :
		(n % 100 == 3 ? "rd" :
		"th")));
}

} // namespace detail


///@cond
#define BIT_PRINTING_PATTERN_4BIT "%04u"
#define BIT_PRINTING_ARGUMENTS_4BIT(x)\
	(unsigned) detail::get_bit((x),  0) * 1000 + \
	(unsigned) detail::get_bit((x),  1) * 100 +  \
	(unsigned) detail::get_bit((x),  2) * 10 +   \
	(unsigned) detail::get_bit((x),  3) * 1

#define DOUBLE_LENGTH_PATTERN(single_length_pattern) single_length_pattern "'" single_length_pattern
#define DOUBLE_LENGTH_ARGUMENTS(x, single_length, single_length_arguments) \
	single_length_arguments(x), single_length_arguments(x >> single_length)

#define BIT_PRINTING_PATTERN_8BIT    DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_4BIT)
#define BIT_PRINTING_PATTERN_16BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_8BIT)
#define BIT_PRINTING_PATTERN_32BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_16BIT)
#define BIT_PRINTING_PATTERN_64BIT   DOUBLE_LENGTH_PATTERN(BIT_PRINTING_PATTERN_32BIT)

#define BIT_PRINTING_ARGUMENTS_8BIT(x)  DOUBLE_LENGTH_ARGUMENTS(x, 4,  BIT_PRINTING_ARGUMENTS_4BIT)
#define BIT_PRINTING_ARGUMENTS_16BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 8,  BIT_PRINTING_ARGUMENTS_8BIT)
#define BIT_PRINTING_ARGUMENTS_32BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 16, BIT_PRINTING_ARGUMENTS_16BIT)
#define BIT_PRINTING_ARGUMENTS_64BIT(x) DOUBLE_LENGTH_ARGUMENTS(x, 32, BIT_PRINTING_ARGUMENTS_32BIT)
///@endcond

#define printf_32_bits(x) \
printf( \
	"%04u'%04u'%04u'%04u'%04u'%04u'%04u'%04u\n", \
	detail::get_bit(x,  0) * 1000 + \
	detail::get_bit(x,  1) * 100 + \
	detail::get_bit(x,  2) * 10 + \
	detail::get_bit(x,  3) * 1, \
	detail::get_bit(x,  4) * 1000 + \
	detail::get_bit(x,  5) * 100 + \
	detail::get_bit(x,  6) * 10 + \
	detail::get_bit(x,  7) * 1, \
	detail::get_bit(x,  8) * 1000 + \
	detail::get_bit(x,  9) * 100 + \
	detail::get_bit(x, 10) * 10 + \
	detail::get_bit(x, 11) * 1, \
	detail::get_bit(x, 12) * 1000 + \
	detail::get_bit(x, 13) * 100 + \
	detail::get_bit(x, 14) * 10 + \
	detail::get_bit(x, 15) * 1, \
	detail::get_bit(x, 16) * 1000 + \
	detail::get_bit(x, 17) * 100 + \
	detail::get_bit(x, 18) * 10 + \
	detail::get_bit(x, 19) * 1, \
	detail::get_bit(x, 20) * 1000 + \
	detail::get_bit(x, 21) * 100 + \
	detail::get_bit(x, 22) * 10 + \
	detail::get_bit(x, 23) * 1, \
	detail::get_bit(x, 24) * 1000 + \
	detail::get_bit(x, 25) * 100 + \
	detail::get_bit(x, 26) * 10 + \
	detail::get_bit(x, 27) * 1, \
	detail::get_bit(x, 28) * 1000 + \
	detail::get_bit(x, 29) * 100 + \
	detail::get_bit(x, 30) * 10 + \
	detail::get_bit(x, 31) * 1 \
);

#define thread_printf_32_bits(x) \
thread_printf( \
	"%04u'%04u'%04u'%04u'%04u'%04u'%04u'%04u", \
	detail::get_bit(x,  0) * 1000 + \
	detail::get_bit(x,  1) * 100 + \
	detail::get_bit(x,  2) * 10 + \
	detail::get_bit(x,  3) * 1, \
	detail::get_bit(x,  4) * 1000 + \
	detail::get_bit(x,  5) * 100 + \
	detail::get_bit(x,  6) * 10 + \
	detail::get_bit(x,  7) * 1, \
	detail::get_bit(x,  8) * 1000 + \
	detail::get_bit(x,  9) * 100 + \
	detail::get_bit(x, 10) * 10 + \
	detail::get_bit(x, 11) * 1, \
	detail::get_bit(x, 12) * 1000 + \
	detail::get_bit(x, 13) * 100 + \
	detail::get_bit(x, 14) * 10 + \
	detail::get_bit(x, 15) * 1, \
	detail::get_bit(x, 16) * 1000 + \
	detail::get_bit(x, 17) * 100 + \
	detail::get_bit(x, 18) * 10 + \
	detail::get_bit(x, 19) * 1, \
	detail::get_bit(x, 20) * 1000 + \
	detail::get_bit(x, 21) * 100 + \
	detail::get_bit(x, 22) * 10 + \
	detail::get_bit(x, 23) * 1, \
	detail::get_bit(x, 24) * 1000 + \
	detail::get_bit(x, 25) * 100 + \
	detail::get_bit(x, 26) * 10 + \
	detail::get_bit(x, 27) * 1, \
	detail::get_bit(x, 28) * 1000 + \
	detail::get_bit(x, 29) * 100 + \
	detail::get_bit(x, 30) * 10 + \
	detail::get_bit(x, 31) * 1 \
);




namespace linear_grid {

/**
 * A wrapper for the printf function, which prefixes the printed string with
 * the thread's full identification: The In-block thread index, and a
 * block-warp-lane index triplet.
 *
 * Note that only a single printf() call is made for both the prefix id info and
 * the macros' arguments
 *
 * @note the parameters are the same as for @ref std::printf - a format string
 * with `%` tags, then one argument corresponding to each of the `%` tags.
 * @return same as @ref std::printf
 *
 * @note: I know it's a bit silly putting a macro in a namespace. For now, just assume
 * thread_printf, warp_printf etc. really only exist for linear grids
 */
#define thread_printf(format_str, ... )  \
	printf("T %0*u = (%0*u,%02u,%02u): " format_str "\n", \
		::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_threads() - 1llu)), \
		kat::linear_grid::grid_info::thread::global_index(), \
		::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1llu)), \
		blockIdx.x, \
		kat::linear_grid::grid_info::warp::index(), kat::grid_info::lane::index(), __VA_ARGS__);
#define thread_print(str)  \
	printf("T %0*u = (%0*u,%02u,%02u): %s\n", \
		::max(2,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_threads() - 1llu)), \
		kat::linear_grid::grid_info::thread::global_index(), \
		::max(2,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1llu)), \
		blockIdx.x, \
		kat::linear_grid::grid_info::warp::index(), kat::grid_info::lane::index(), str);
#define tprintf thread_printf
#define tprint thread_print

#define warp_printf(format_str, ... )  \
	do{ \
		if (grid_info::lane::is_first()) \
			printf("W %0*u = (%0*u,%02u): " format_str "\n", \
				::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_warps() - 1)), \
				kat::linear_grid::grid_info::warp::global_index(), \
				::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1)), \
				blockIdx.x, \
				kat::linear_grid::grid_info::warp::index(), __VA_ARGS__); \
	} while(0)
#define warp_print(str)  \
	do{ \
		if (grid_info::lane::is_first()) \
		printf("W %0*u = (%0*u,%02u): %s\n", \
			::max(2,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_warps() - 1)), \
			kat::linear_grid::grid_info::warp::global_index(), \
			::max(2,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1)), \
			blockIdx.x, \
			kat::linear_grid::grid_info::warp::index(), str); \
	} while(0)

#define block_printf(format_str, ... )  \
	do{ \
		if (kat::linear_grid::grid_info::thread::is_first_in_block()) \
			printf("B %0*u: " format_str "\n", \
				::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1)), \
				kat::linear_grid::grid_info::block::index(), __VA_ARGS__); \
	} while(0)
#define block_print(str)  \
	do{ \
		if (kat::linear_grid::grid_info::thread::is_first_in_block()) \
		printf("B %0*u: %s\n", \
			::max(2u,kat::detail::num_digits_required_for(kat::linear_grid::grid_info::grid::num_blocks() - 1)), \
				kat::linear_grid::grid_info::block::index(), str); \
	} while(0)
#define bprintf block_printf
#define bprint block_print

#define grid_printf(format_str, ... )  \
	do { \
		if (grid_info::thread::is_first_in_grid()) { \
		    printf("G " format_str "\n", __VA_ARGS__); \
		} \
	} while (false)
#define grid_print(str)  \
	do { \
		if (kat::linear_grid::grid_info::thread::global_index() == 0) { \
		    printf("G %s\n", str); \
		} \
	} while (false)

// Identification is 0-based!
inline __device__ void print_self_identification()
{
	printf("Thread %10d - %05d within block %05d - lane %02d of in-block warp %02d)\n",
		kat::linear_grid::grid_info::thread::global_index(),
		threadIdx.x, blockIdx.x, threadIdx.x % warpSize, threadIdx.x / warpSize);

}

#define identify_function() { \
	printf_once("Now executing function \"%s\"", __PRETTY_FUNCTION__); \
	__syncthreads(); \
}

} // namespace linear_grid

} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_PRINTING_CUH_
