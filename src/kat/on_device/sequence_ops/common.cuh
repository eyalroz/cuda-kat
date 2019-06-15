/**
 * @file on_device/sequence_ops/common.cuh
 *
 * @brief Some common definitions for all on-device collaborative sequence operations
 */

#ifndef CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_COMMON_CUH_
#define CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_COMMON_CUH_

#include <kat/on_device/common.cuh>
#include <kat/on_device/math.cuh>

namespace kat {
namespace collaboration {

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

} // namespace collaboration
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_COMMON_CUH_
