/**
 * @file on_device/collaboration/block.cuh
 *
 * @brief CUDA device computation warp-level primitives, i.e. those involving
 * interaction of many/all of each blocks's lanes, but _no_ inter-block
 * interaction.
 *
 * @todo Some of these assume linear grids, others do not - sort them out
 *
 */

#pragma once
#ifndef BLOCK_LEVEL_PRIMITIVES_CUH_
#define BLOCK_LEVEL_PRIMITIVES_CUH_

#include <kat/on_device/collaboration/warp.cuh>
#include <kat/on_device/shared_memory/basic.cuh>
#include <kat/on_device/common.cuh>
#include <kat/on_device/math.cuh>

#include <type_traits>

///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {
namespace linear_grid {
namespace collaborative {
namespace block {

///@cond
// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::index()` without prefixing that with anything.

namespace grid   = grid_info::grid;
namespace block  = grid_info::block;
namespace warp   = grid_info::warp;
namespace thread = grid_info::thread;
namespace lane   = grid_info::lane;

///@endcond

/*
 * TODO: Implement
 * __fd__  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

/**
 * Have all threads in (one/some/all) blocks perform some action over the
 * linear range of 0..length-1 - the same range for each block.
 *
 * @param length The length of the range (of integers) on which to act
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 */
template <typename Function, typename Size = size_t>
__fd__ void at_block_stride(Size length, const Function& f)
{
	auto block_length = block::length();
	#pragma unroll
	for(promoted_size_t<Size> pos = thread::index_in_block();
		pos < length;
		pos += block_length)
	{
		f(pos);
	}
}

// TODO: Perhaps make this a "warp to block" primitive?
/**
 * Makes one element of type T for each warp, which was previously
 * only visible to that warp (or rather not known to be otherwise) - visible to,
 * shared with, the entire block, via shared memory.
 *
 * @param datum a warp-specific (but not thread-specific) piece of data,
 * one for each warp, which is to be shared with the whole block
 * @param where_to_make_available the various warp-specific data will be
 * stored here by warp index
 * @param writing_lane_index which lane in each warp should perform write operations
 */
template <typename T, bool Synchronize = true>
__fd__ void share_warp_datum_with_whole_block(
	const T&         datum,
	T* __restrict__  where_to_make_available,
	unsigned         writing_lane_index = 0)
{
	if (lane::index() == writing_lane_index) {
		where_to_make_available[warp::index()] = datum;
	}
	if (Synchronize) __syncthreads();
}

__fd__ void barrier() { __syncthreads(); }

/**
 * @brief Execute some code exactly once per block of threads
 *
 * @note This macro is a kind of ugly and non-robust way of achieving
 * the stated effect; but a safe way to do so would be much slower.
 *
 * @note You must make sure that first thread of the block hits this
 * instruction to get the desired effect...
 */
#define once_per_block  if (thread::is_first_in_block())

/**
 * @brief have all block threads obtain a value held by just
 * one of the threads (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for the "broadcast" by the thread holding
 * the relevant value
 */
template <typename T>
__fd__ T get_from_thread(const T& value, unsigned source_thread_index)
{
	__shared__ static T tmp;
	if (thread::index_in_block() == source_thread_index) {
		tmp = value;
	}
	__syncthreads();
	return tmp;
}

/**
 * @brief have all block threads obtain a value held by the first
 * thread in the block (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for "broadcasting" the value
 */
template <typename T>
__fd__ T get_from_first_thread(const T& value)
{
	return get_from_thread(value, 0);
}

} // namespace block
} // namespace collaborative
} // namespace linear_grid

namespace collaborative {

namespace block {

///@cond
// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::index()` without prefixing that with anything.

namespace grid   = grid_info::grid;
namespace block  = grid_info::block;
namespace warp   = grid_info::warp;
namespace thread = grid_info::thread;
namespace lane   = grid_info::lane;

///@endcond

/*
 * TODO: Implement
 * __fd__  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

// TODO: Perhaps make this a "warp to block" primitive?
/**
 * Makes one element of type T for each warp, which was previously
 * only visible to that warp (or rather not known to be otherwise) - visible to,
 * shared with, the entire block, via shared memory.
 *
 * @param datum a warp-specific (but not thread-specific) piece of data,
 * one for each warp, which is to be shared with the whole block
 * @param where_to_make_available the various warp-specific data will be
 * stored here by warp index
 * @param writing_lane_index which lane in each warp should perform write operations
 */
template <typename T, bool Synchronize = true>
__fd__ void share_warp_datum_with_whole_block(
	const T& datum,
	T* __restrict__ where_to_make_available,
	unsigned writing_lane_index = 0)
{
	if (lane::index() == writing_lane_index) {
		where_to_make_available[warp::index()] = datum;
	}
	if (Synchronize) __syncthreads();
}

__fd__ void barrier() { __syncthreads(); }


/**
 * @brief have all block threads obtain a value held by just
 * one of the threads (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for the "broadcast" by the thread holding
 * the relevant value
 */
template <typename T>
__fd__ T get_from_thread(const T& value, unsigned source_thread_index)
{
	__shared__ static T tmp;
	if (thread::index_in_block() == source_thread_index) {
		tmp = value;
	}
	__syncthreads();
	return tmp;
}

/**
 * @brief have all block threads obtain a value held by the first
 * thread in the block (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for "broadcasting" the value
 */
template <typename T>
__fd__ T get_from_first_thread(const T& value)
{
	return get_from_thread(value, 0);
}

} // namespace block
} // namespace collaborative
} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // BLOCK_LEVEL_PRIMITIVES_CUH_
