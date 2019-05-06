/**
 * @file on_device/common.cuh Some basic type and constant definitions used by all device-side CUDA-related code
 * in this directory; possibly imported from `cuda-api-wrappers`.
 *
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_COMMON_CUH_
#define CUDA_KAT_ON_DEVICE_COMMON_CUH_

#include <type_traits>
#include <climits> // for CHAR_BIT

namespace kat {

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

/**
 * The number bits in the representation of a value of type T.
 *
 * @note with this variant, you'll need to manually specify the type.
 */
template <typename T>
constexpr std::size_t size_in_bits() { return sizeof(T) * CHAR_BIT; }

/**
 * The number bits in the representation of a value of type T
 *
 * @note with this variant, the type will be deduced from the
 * object you pass.
 */
template <typename T>
constexpr std::size_t size_in_bits(const T&) { return sizeof(T) * CHAR_BIT; }


} // namespace kat

#include <kat/define_specifiers.hpp>

//constexpr __fhd__ bool operator==(const dim3& lhs, const dim3& rhs)
//{
//	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
//}

#include <kat/undefine_specifiers.hpp>


#endif // CUDA_KAT_ON_DEVICE_COMMON_CUH_
