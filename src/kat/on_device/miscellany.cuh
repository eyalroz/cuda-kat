/**
 * @file kat/on_device/miscellany.cuh
 *
 * @brief Miscellaneous functions provided by cudat-kat which are not a good
 * fit in any other header.
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
#define CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_

#include "common.cuh"

#include <type_traits>
#include <limits>
#include <cassert>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

/**
 * @brief Swap two values on the device-side, in-place.
 *
 * @note A (CUDA, or any other) compiler will often not actually
 * emit any code when this function is used. Instead, it will use
 * one argument instead of the other in later code, i.e. "swap"
 * them in its own internal figuring.
 */
template <typename T>
KAT_FD  void swap(T& x, T& y)  {
   T _x = x;
   T _y = y;
   x = _y;
   y = _x;
}

/**
 * Copies some data from one location to another - using the native register
 * size for individual elements on CUDA GPUs, i.e. sizeof(int) = 4
 *
 * @note CUDA's own general-purpose memcpy() takes void pointers and uses a u8 (byte)
 * LD-ST loop. See: @url https://godbolt.org/z/9ChTPM ; this LD-ST's using the native
 * register size, 4 bytes, if possible.
 *
 * @note this function assumes appropriate alignment.
 *
 * @note Instead of using this function, you're probably better off using a warp-level
 * or block-level primitive for copying data.
 *
 * @param destination Destination of the copy. Must have at least
 * 4 * {@p num_elements_to_copy} bytes allocated.
 * @param source The beginning of the memory region from which to copy.
 * There must be 4 * {@p num_elements_to_copy} bytes readable starting with
 * this address.
 * @param num_elements_to_copy the number of elements of data to copy - not their
 * total size in bytes!
 * @return the destination pointer
 */
template <typename T>
KAT_FD T* copy(
	T*        __restrict__  destination,
	const T*  __restrict__  source,
	size_t                  num_elements_to_copy)
{
	if (sizeof(T) % sizeof(kat::native_word_t) == 0) {
		while (num_elements_to_copy-- > 0) {
			*(destination++) = *(source++);
		}
	}
	else {
		// This emits a u8 load-store loop, which we want to avoid
		memcpy(destination, source, sizeof(T) * num_elements_to_copy);
	}
	return destination;
}

/**
 * @brief Return the number of full warps in a linear grid
 * which would, overall, contain at least a given number of threads.
 *
 * @note This comes in handy more times than you must expect even in device-side code.
 *
 * @note the reason this function is defined directly rather than using
 * the functions in math or constexpr_math is that bit-counting is
 * either slow in run-time on the GPUwhen you use the constexpr way of
 * doing it, or not constexpr if you use the GPU-side population count
 * instruction.
 */
template <typename I>
constexpr KAT_FHD I num_warp_sizes_to_cover(I number_of_threads)
{
	static_assert(std::is_integral<I>::value, "Number of threads specified using a non-integral type");
	enum : I { mask = (warp_size - 1) };
	enum : I { log_warp_size = 5 } ;
	return (number_of_threads >> log_warp_size) + ((number_of_threads & mask) != 0);
}

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
