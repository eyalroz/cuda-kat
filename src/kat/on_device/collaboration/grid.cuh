/**
 * @file on_device/collaboration/grid.cuh
 *
 * @brief CUDA device computation grid-level primitives, i.e. those involving
 * interaction of threads from different blocks in the grid
 *
 */

#pragma once
#ifndef CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_GRID_CUH_
#define CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_GRID_CUH_

#include "warp.cuh"

#include <kat/on_device/common.cuh>
#include <kat/on_device/math.cuh>
#include <kat/on_device/grid_info.cuh>

#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace linear_grid {
namespace collaborative {
namespace grid {

// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::index()` without prefixing that with anything.

namespace grid   = kat::linear_grid::grid_info::grid;
namespace block  = kat::linear_grid::grid_info::block;
namespace warp   = kat::linear_grid::grid_info::warp;
namespace thread = kat::linear_grid::grid_info::thread;
namespace lane   = kat::linear_grid::grid_info::lane;

/**
 * Have all kernel threads perform some action over the linear range
 * of 0..length-1, at strides equal to the grid length, i.e. a thread
 * with index i_t in block with index i_b, where block lengths are n_b,
 * will perform the action on elements i_t, i_t + n_b, i_t + 2*n_b, and
 * so on.
 *
 * Thus, if in the following chart the rectangles represent
 * consecutive segments of n_b integers, the numbers
 * indicate which blocks work on which elements in "grid stride":
 *
 *   -------------------------------------------------------
 *  |   1   |  222  |  333  |   1   |  222  |  333  |   1   |
 *  |  11   | 2   2 | 3   3 |  11   | 2   2 | 3   3 |  11   |
 *  |   1   |     2 |    3  |   1   |     2 |    3  |   1   |
 *  |   1   |  222  |    3  |   1   |  222  |    3  |   1   |
 *  |   1   | 2     | 3   3 |   1   | 2     | 3   3 |   1   |
 *  |  111  | 22222 |  333  |  111  | 22222 |  333  |  111  |
 *   -------------------------------------------------------
 *
 * (the grid is 3 blocks' worth, so block 1 strides 3 blocks
 * from one sequence of indices it processes to the next.)
 * This is unlike `at_block_stride()`, for which instead
 * of 1, 2, 3, 1, 2, 3, 1 we would have 1, 1, 1, 2, 2, 2, 3
 * (or 1, 1, 2, 2, 3, 3, 4 if the grid has 4 blocks).
 *
 * @note assumes the number of grid threads is fixed (does that
 * always hold? even with dynamic parallelism?)
 *
 * @param length The length of the range (of integers) on which to act
 * @param f The callable to call for each element of the sequence.
 */
template <typename Function, typename Size = size_t>
KAT_FD void at_grid_stride(Size length, const Function& f)
{
	auto num_grid_threads = grid::num_threads();
	for(promoted_size_t<Size> pos = thread::global_id();
		pos < length;
		pos += num_grid_threads)
	{
		f(pos);
	}
}

namespace warp_per_input_element {

/**
 * A variant of the one-position-per-thread applicator,
 * `collaborative::grid::at_grid_stride()`: Here each warp works on one
 * input position, advancing by 'grid stride' in the sense of total
 * warps in the grid.
 *
 * @note it is assumed the grid only has fully-active warps; any
 * possibly-inactive threads are not given consideration.
 *
 * @note This version of `at_grid_stride` is specific to linear grids,
 * even though the text of its code looks the same as that of
 * @ref kat::grid_info::collaborative::warp::at_grid_stride .
 *
 * @param length The length of the range of positions on which to act
 * @param f The callable for warps to use each position in the sequence
 */
template <typename Function, typename Size = unsigned>
KAT_FD void at_grid_stride(Size length, const Function& f)
{
	auto num_warps_in_grid = grid_info::grid::num_warps();
	for(// _not_ the global thread index! - one element per warp
		promoted_size_t<Size> pos = grid_info::warp::global_id();
		pos < length;
		pos += num_warps_in_grid)
	{
		f(pos);
	}
}


} // namespace warp_per_input_element


/**
 * Have all grid threads perform some action over the linear range
 * of 0..length-1, with each thread acting on a fixed number of items
 * (@p the serialization_factor) at at stride of the block length,
 * i.e. a thread with index i_t in
 * block with index i_b, where block lengths are n_b,
 * will perform the action on elements
 *
 *  n_b * i_b      * serialization_factor + i_t,
 * (n_b * i_b + 1) * serialization_factor + i_t,
 * (n_b * i_b + 2) * serialization_factor + i_t,
 *
 * and so on. For lengths which are not divisible by n_b *
 * serialization_factor, threads in the last block will
 * work on less items.
 *
 * Thus, if in the following chart the rectangles represent
 * consecutive segments of n_b integers, the numbers
 * indicate which blocks work on which elements in "block stride":
 *
 *   -------------------------------------------------------
 *  |   1   |   1   |  222  |  222  |  333  |  333  |    4  |
 *  |  11   |  11   | 2   2 | 2   2 | 3   3 | 3   3 |   44  |
 *  |   1   |   1   |     2 |     2 |    3  |    3  |  4 4  |
 *  |   1   |   1   |  222  |  222  |    3  |    3  | 4  4  |
 *  |   1   |   1   | 2     | 2     | 3   3 | 3   3 | 44444 |
 *  |  111  |  111  | 22222 | 22222 |  333  |  333  |    4  |
 *   -------------------------------------------------------
 *
 * (A block strides from one blocks' worth of indices to the next.)
 * This is unlike `at_grid_stride()`, for which instead
 * of 1, 1, 2, 2, 3, 3, 4 we would have 1, 2, 3, 1, 2, 3, 1 (if the
 * grid has 3 blocks) or 1, 2, 3, 4, 1, 2 (if the grid has 4 blocks).
 *
 * @note Theoretically, the @param serialization_factor value could be
 * computed by this function itself. This is avoided, assuming that's
 * been take care of before. Specifically, we assume that the
 * @param serialization_factor is no higher than it absolutely
 * must be.
 *
 * @note There's a block-level variant of this primitive, but there -
 * each block applies f to the _same_ range of elements, rather than
 * covering part of a larger range.
 *
 * @note This implementation does not handle cases of overflow of
 * the @tparam Size type, e.g. if your Size is uint32_t and @param
 * length is close to 2^32 - 1, the function may fail.
 *
 * @note There's a tricky tradeoff here between avoiding per-iteration
 * checks for whether we're past the end, and avoiding too many
 * initial checks. Two of the the template parameters help up avoid
 * this tradeoff in certain cases by not having to check explicitly
 * for things.
 *
 *
 * @param length The length of the range (of integers) on which to act
 * @param serialization_factor the number of elements each thread is to
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 *
 */
template <
	typename Function,
	typename Size = size_t,
	bool AssumeLengthIsDivisibleByBlockSize = false,
	bool GridMayFullyCoverLength = true,
	typename SerializationFactor = unsigned>
KAT_FD void at_block_stride(
	Size                 length,
	const Function&      f,
	SerializationFactor  serialization_factor)
{
	auto block_length = block::length();
	auto num_elements_to_process_by_each_block = serialization_factor * block_length;
	Size block_start_pos = num_elements_to_process_by_each_block * block::index();
	Size pos = block_start_pos + thread::index();
	if (pos >= length) { return; }
	auto in_last_acting_block = (block_start_pos + num_elements_to_process_by_each_block >= length);
		// Note: Be careful about overflow in this last line, if block_start_pos is close
		// to the maximum value of Size.

	if (in_last_acting_block) {
		#pragma unroll
		for(; pos < length; pos += block_length) {
			f(pos);
		}
		return;
	}
	// If we're not in the last block which needs to take any action, we assume that we'll perform
	// full iterations and don't need to check for overstepping any bounds
	#pragma unroll
	for(SerializationFactor i = 0; i < serialization_factor; i++, pos += block_length) {
		f(pos);
	}
}

} // namespace grid
} // namespace collaborative
} // namespace linear_grid
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_GRID_CUH_
