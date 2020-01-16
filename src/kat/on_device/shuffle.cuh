/**
 * @file shuffle.cuh Templated warp-shuffle operation variants
 */

/*
 * Originally based on Bryan Catanzaro's CUDA generics
 * https://github.com/bryancatanzaro/generics/
 * Downloaded on: 2016-04-16
 * ... but reimplemented by Eyal Rozenberg, CWI Amsterdam
 */

#pragma once
#ifndef CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_
#define CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_

#include <kat/on_device/common.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

// The functions here can be used to shuffle types as large as you like. Of course,
// if they're not plain-old-data, shuffle at your peril.

/**
 * @brief Have each lane in a warp get a value from an (arbitrary) other lane.
 *
 * @tparam T the type of datum to be shared with other lane(s); may be of
 * arbitrary size, but (at least for now) must be plain-old-data.
 *
 * @param t Each lane shares own value, which other lanes can choose
 * to receive.
 * @param source_lane The lane whose value the current lane wants to get
 * @return the @p t value of @p source_lane
 */
template<typename T> KAT_FD T shuffle_arbitrary(const T& t, int source_lane);

/**
 * @param t Each lane shares own value, which a lane with a higher index
 * will get.
 * @param delta The difference in lane index to the source lane of the new
 * value, i.e. a lane with index i gets the new value from lane i + delta.
 * @return The @p t value of the lane with index @p delta less than the calling
 * lane's; a lane with a high index, above warp_size - @p delta, has its own @p t
 * returned unchanged.
 */
template<typename T> KAT_FD T shuffle_down(const T& t, unsigned int delta);

/**
 * @tparam T the type of datum to be shared with other lane(s); may be of
 * arbitrary size, but (at least for now) must be plain-old-data.
 *
 * @param t Each lane shares own value, which a lane with a lower index
 * will get.
 * @param delta The difference in lane index to the source lane of the new
 * value, i.e. a lane with index i gets the new value from the lane of index
 * i - delta.
 * @return The @p t value of the lane with index @p delta less than the calling
 * lane's; a lane with a low index, under @p delta, has its own @p t returned
 * unchanged.
 */
template<typename T> KAT_FD T shuffle_up(const T& t, unsigned int delta);

/**
 * @brief Have pairs of lanes exchange a value, with the pairing performed
 * by XORing bits of the lane index.
 *
 * @tparam T the type of datum to be shared with other lane(s); may be of
 * arbitrary size, but (at least for now) must be plain-old-data.
 *
 * @param t The value to exchange with a counterpart lane
 * @param mask Determines how lanes will be paired: The lane with index i
 * is paired with the lane with index i ^ mask.
 * @return The @p t value of the paired lane
 */
template<typename T> KAT_FD T shuffle_xor(const T& t, int mask);

} // namespace kat

#include "detail/shuffle.cuh"

#endif // CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_
