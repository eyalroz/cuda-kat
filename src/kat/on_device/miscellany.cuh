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
#include <kat/define_specifiers.hpp>
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
__fd__  void swap(T& x, T& y)  {
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
__fd__ T* copy(
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

template <typename I>
constexpr __fhd__ I num_warp_sizes_to_cover(I x)
{
    enum : I { log_Warp_size = 5 };
    return (x >> log_warp_size) + ((x & (warp_size-1)) > 0);
}

} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
