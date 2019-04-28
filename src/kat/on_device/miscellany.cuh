#pragma once
#ifndef CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
#define CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_

#include "common.cuh"

#include <type_traits>

#include <kat/define_specifiers.hpp>

template <typename T>
__fd__  void swap(T& x, T& y) {
   T _x = x;
   T _y = y;
   x = _y;
   y = _x;
}

/**
 * Copies some data from one location to antoher - using the native register
 * size for individual elements on CUDA GPUs, i.e. sizeof(int) = 4
 *
 * @todo This is untested.
 *
 * @note unlike memcpy, the size of copied data is num_elements_to_copy times the
 * element size, i.e. 4*num_elements_to_copy.
 *
 * @note CUDA's own general-purpose memcpy() takes void pointers and uses a byte
 *  LD-ST loop; this  LD-ST's the native register size, 4 bytes.
 *
 * @note this function entirely ignores alignment.
 *
 * @param destination Destination of the copy. Must have at least
 * 4 * {@ref num_elements_to_copy} bytes allocated.
 * @param source The beginning of the memory region from which to copy.
 * There must be 4 * {@ref num_elements_to_copy} bytes readable starting with
 * this address.
 * @param num_elements_to_copy the number of int-sized elements of data to copy
 * (regardless of whether the data really consists of integers
 * @return the destination pointer
 */
template <typename T = unsigned int>
__fd__
std::enable_if<sizeof(T) == sizeof(int), T> * copy(
	T*        __restrict__  destination,
	const T*  __restrict__  source,
	size_t                  num_elements_to_copy)
{
	while (num_elements_to_copy-- > 0) {
		*(destination++) = *(source++);
	}
	return destination;
}

/**
 * A replacement for CUDA's iteration-per-byte-loop memcpy()
 * implemntation: This accounts for alignment (although not of
 * incompatible alignment between source and thread); and uses
 * register-sized quanta for copying as much as possible.
 *
 * @note not tested
 *
 * @param destination A location to which to copy data.
 * @param source location from which to copy data
 * @param size number of bytes to copy
 */
template <typename T>
__fd__
void* long_memcpy(
	void*        __restrict__  destination,
	const void*  __restrict__  source,
	size_t                     size)
{
	static_assert(sizeof(int) == 4, "Expecting sizeof(int) to be 4");
	static_assert(sizeof(void*) == sizeof(unsigned long long), "Expecting pointers to be unsigned long long");

	auto misalignment = (unsigned long long) destination % 4;
	// Assumes the source and destination have the same alignment modulo sizeof(int) = 4
	switch((unsigned long long) destination % 4) {
	case 1: *((char* )destination) = *((const char* )source); break;
	case 2: *((short*)destination) = *((const short*)source); break;
	case 3: *((char3*)destination) = *((const char3*)source); break;
	}
	size -= misalignment;

	// These int pointers will be aligned at sizeof(int)
	int*       destination_int = ((int*)       ((char*)      destination - misalignment)) + 1;
	const int* source_int      = ((const int*) ((const char*)source      - misalignment)) + 1;
	auto num_ints_to_copy = size >> 4;
	while (num_ints_to_copy-- > 0) {
		*(destination_int++) = *(source_int++);
	}
	switch(size % 4) {
	case 1: *((char* )destination_int) = *((const char* )source_int); break;
	case 2: *((short*)destination_int) = *((const short*)source_int); break;
	case 3: *((char3*)destination_int) = *((const char3*)source_int); break;
	}
	return destination;
}

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
