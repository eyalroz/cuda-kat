/**
 * @file on_device/collaboration/grid.cuh
 *
 * @brief CUDA device computation grid-level primitives, i.e. those involving
 * interaction of threads from different blocks in the grid
 *
 */

#pragma once
#ifndef CUDA_KAT_GRID_COLLABORATIVE_SEQUENCE_OPS_CUH_
#define CUDA_KAT_GRID_COLLABORATIVE_SEQUENCE_OPS_CUH_

#include "common.cuh"
#include <kat/on_device/collaboration/grid.cuh>
#include <kat/on_device/sequence_ops/warp.cuh>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace collaborative {
namespace warp_to_grid {

/**
 * Used by multiple warps, in multiple blocks with each warp having
 * a bunch of data it has obtained and all warps' data must be
 * chained into a global-memory vector - with no gaps and no
 * overwriting (but not necessarily in the order of warps, just any
 * order.)
 *
 * @note if the input is not 32-byte (sometimes 128-byte )-aligned,
 * and more importantly, the output is not 128-byte-aligned,
 * performance will likely degrade due to the need to execute a pair
 * of memory transactions for every single 32 x 4 byte write.
 *
 * @note this must be called by complete warps, with all lanes
 * active and participating. But it does _not_ - for the time
 * being - have to called by complete blocks.
 *
 * @tparam T the type of data elements being copied
 * @tparam Size must fit any index used into the input or output array;
 * for the general case it would be 64-bit, but this is
 * usable also for when you need 32-bit work (e.g. a 32-bit length
 * output variable).
 * @param global_output
 * @param global_output_length
 * @param fragment_to_append
 * @param fragment_length
 */
template <typename T, typename Size = size_t>
KAT_FD void collaborative_append_to_global_memory(
	T*     __restrict__  global_output,
	Size*  __restrict__  global_output_length,
	T*     __restrict__  fragment_to_append,
	Size   __restrict__  fragment_length)
{
	using namespace grid_info;
	Size previous_output_size = thread::is_first_in_warp() ?
		atomic::add(global_output_length, fragment_length) : 0;
	Size offset_to_start_writing_at = collaborative::warp::get_from_first_lane(
		previous_output_size);

	// Now the (0-based) positions
	// previous_output_size ... previous_output_size + fragment_length - 1
	// are reserved by this warp; nobody else will write there and we don't need
	// any more atomics

	enum : bool { may_have_slack = true };

	if (detail::elements_per_lane_in_full_warp_write<T>::value > 1) {
		// We don't have a version of copy which handles unaligned destinations, so
		warp::detail::naive_copy(global_output + offset_to_start_writing_at,
			fragment_to_append, fragment_length);
	}
	else {
		warp::copy_n<T, Size,  may_have_slack>(
			global_output + offset_to_start_writing_at,
			fragment_to_append, fragment_length);
	}
}

} // namespace warp_to_grid
} // namespace collaborative
} // namespace kat

#endif // CUDA_KAT_GRID_COLLABORATIVE_SEQUENCE_OPS_CUH_
