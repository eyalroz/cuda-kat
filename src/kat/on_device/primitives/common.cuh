/**
 * @file on_device/primitives/common.cuh
 *
 * @brief Some common definitions for all on-device computational primitives (grid,
 * block, warp, thread and lane).
 */

#ifndef CUDA_KAT_ON_DEVICE_PRIMITIVES_COMMON_CUH_
#define CUDA_KAT_ON_DEVICE_PRIMITIVES_COMMON_CUH_

#include <cuda/api/types.hpp>
#include <kat/on_device/miscellany.cuh>
#include <kat/on_device/math.cuh>
#ifdef DEBUG
#include <kat/on_device/printing.cuh>
#endif

#include <type_traits>

namespace cuda {
namespace primitives {

enum inclusivity_t : bool {
	Exclusive = false,
	Inclusive = true
};

namespace detail {

/**
 * In a "full warp write", we want [1] each lane to write an integral number
 * native words (at the moment and for the foreseeable future, 4-byte integers).
 * At the same time, the lane writes complete elements of type T, not arbitrary
 * sequences of `sizeof(native_word_t)`, hence this definition.
 *
 * @todo: Can't we assume that T is a POD type, and just have lanes not write
 * complete T's?
 */
template <typename T>
struct elements_per_lane_in_full_warp_write {
    enum { value = sizeof(native_word_t) / constexpr_::gcd<unsigned>(sizeof(native_word_t),sizeof(T)) };
};
} // namespace detail

} // namespace primitives
} // namespace cuda

#endif // CUDA_KAT_ON_DEVICE_PRIMITIVES_COMMON_CUH_
