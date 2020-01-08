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

#include <type_traits>


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {
namespace collaborative {
namespace grid {
namespace linear {

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
__fd__ void at_grid_stride(Size length, const Function& f)
{
	auto num_grid_threads = grid::num_threads();
	for(promoted_size_t<Size> pos = thread::global_index();
		pos < length;
		pos += num_grid_threads)
	{
		f(pos);
	}
}

/**
 * Have all grid threads perform some action over the linear range
 * of 0..length-1, with each thread acting on a fixed number of items
 * (@p the SerializationFactor) at at stride of the block length,
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
 *
 * @note There's a block-level variant of this primitive, but there -
 * each block applies f to the _same_ range of elements, rather than
 * covering part of a larger range.
 *
 * @note Does not handle cases of overflow, i.e. if @param length is
 * very close to the maximum possible value for @tparam Size, this
 * may fail.
 *
 * @note The current implementation avoids an extra condition at each
 * iteration - but at the price of divergence in the last warp; the
 * other trade-off is sometimes more appropriate
 *
 * @param length The length of the range (of integers) on which to act
 * @param serialization_factor the number of elements each thread is to
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 */
template <typename Function, typename Size = size_t, typename SerializationFactor = unsigned>
__fd__ void at_block_stride(
	Size length, const Function& f, SerializationFactor serialization_factor = 1)
{
	Size pos = thread::block_stride_start_position(serialization_factor);
	auto block_length = block::length();
	if (pos + block_length * (serialization_factor - 1) < length) {
		// The usual case, which occurs for most blocks in the grid
		#pragma unroll
		for(SerializationFactor i = 0; i < serialization_factor; i++) {
			f(pos);
			pos += block_length;
		}
	}
	else {
		// We're at the blocks at the end of the grid. In this case, we know we'll
		// be stopped not by getting to the serialization_factor+1'th iteration,
		// but by getting to the end of the range on which we work
		#pragma unroll
		for(; pos < length; pos += block_length) { f(pos); }
	}
}

} // namespace linear
} // namespace grid

namespace block_to_grid {

/**
 * Accumulates the result of some computation from all the blocks into a single,
 * global (=grid-level) scalar - without writes getting lost due to races etc.
 *
 * @note It is necessarily that at least the first thread in every block calls
 * this function, with the appropriate value, otherwise it will fail. Other threads
 * may either call it or fail to call it, and the value they pass is disregarded.
 *
 * @param accumulator The target in global memory into which block results are
 * accumulated. Typically one should care to initialize it somehow before this
 * primitive is used (probably before the whole kernel is invoked).
 * @param block_value The result of some block-specific computation (which would be
 * different for threads of different blocks of course)
 */
template <typename BinaryOp>
__fd__ void accumulation_to_scalar(
	typename BinaryOp::result_type*          accumulator,
	typename BinaryOp::second_argument_type  block_value)
{
	// TODO: Is it really a good idea to "hammer" the single accumulator
	// from all blocks, ostensibly at once? While it's true that
	// at every cycle, at most one block per SM will dispatch its
	// atomic instruction, but that's still up to 30 of these on
	// a Pascal Titan card, per cycle - which is a lot.
	if (grid_info::thread::is_first_in_block()) {
		typename BinaryOp::accumulator::atomic atomic_accumulation_op;
		atomic_accumulation_op(*accumulator, block_value);
	}
}


} // namespace block_to_grid
} // namespace collaborative
} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_SEQUENCE_OPS_GRID_CUH_
