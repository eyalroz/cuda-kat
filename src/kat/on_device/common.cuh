/**
 * @file Some basic type and constant definitions used by all device-side CUDA-related code
 * in this directory; possibly imported from `cuda-api-wrappers`.
 *
 */
#pragma once
#ifndef CUDA_DEVICE_SIDE_COMMON_CUH_
#define CUDA_DEVICE_SIDE_COMMON_CUH_

//#if defined(HAVE_CUDA_API_WRAPPERS) || (defined(CUDA_API_WRAPPERS_TYPES_HPP_) && defined(CUDA_API_WRAPPERS_CONSTANTS_HPP_))
#if 0

#include <cuda/api/constants.hpp>
#include <cuda/api/types.hpp>

// Device-side code doesn't use the cuda:: prefix
using cuda::log_warp_size;
using cuda::warp_size;
using cuda::grid_dimension_t;
using cuda::dimensions_t;
using cuda::native_word_t;

#else

#include <type_traits>

/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the numbers of blocks per grid is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 *
 * @note All three dimensions in dim3 are of the same type as dim3::x
 */
using grid_dimension_t = decltype(dim3::x);

/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the number of threads per block is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 */
using grid_block_dimension_t  = grid_dimension_t;

using native_word_t = unsigned; // TODO: Make this uint32_t perhaps?
enum : native_word_t { warp_size = 32 };
enum : native_word_t { log_warp_size = 5 };

#endif

#include <kat/define_specifiers.hpp>

constexpr __fhd__ bool operator==(const uint3& lhs, const uint3& rhs)
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}

constexpr __fhd__ bool operator==(const int3& lhs, const int3& rhs)
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}

#include <kat/undefine_specifiers.hpp>


#endif // CUDA_DEVICE_SIDE_COMMON_CUH_
