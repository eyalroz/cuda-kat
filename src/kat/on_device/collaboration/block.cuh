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
#include <kat/on_device/grid_info.cuh>

#include <type_traits>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

namespace collaborative {

namespace block {

///@cond
// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::id()` without prefixing that with anything.

namespace grid   = grid_info::grid;
namespace block  = grid_info::block;
namespace warp   = grid_info::warp;
namespace thread = grid_info::thread;
namespace lane   = grid_info::lane;

///@endcond

/*
 * TODO: Implement
 * KAT_FD  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

// TODO: Perhaps make this a "warp to block" primitive?
/**
 * @brief Share one element of type T for each warp with the entire block -
 * using a single array in shared memory for all shared values.
 *
 * @param datum a warp-specific (but not thread-specific) piece of data,
 * one for each warp, which is to be shared with the whole block
 * @param where_to_make_available the various warp-specific data will be
 * stored here by warp index
 * @param writing_lane_id which lane in each warp should perform write operations
 */
template <typename T, bool Synchronize = true>
KAT_FD void share_per_warp_data(
	T                 datum,
	T*  __restrict__  where_to_make_available,
	unsigned          writing_lane_id)
{
	if (lane::index() == writing_lane_id) {
		where_to_make_available[warp::id()] = datum;
	}
	if (Synchronize) __syncthreads();
}

/**
 * @brief A variant of @ref share_per_warp_data , with the writing lane index
 * being decided dynamically in each lane based on who's actually active.
 */
template <typename T, bool Synchronize = true>
KAT_FD void share_per_warp_data(
	T                 datum,
	T*  __restrict__  where_to_make_available)
{
	return share_per_warp_data(
		std::forward<T>(datum),
		where_to_make_available,
		collaborative::warp::select_leader_lane());
}

KAT_FD void barrier() { __syncthreads(); }


/**
 * @brief have all block threads obtain a value held by just
 * one of the threads (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @p value For most threads can be any junk T (even uninitialized);
 * for the thread at position @p source_thread - this is the value
 * to be shared with the rest of the block.
 * @p source_thread_position (multi-dimensional) position of the thread
 * from which to obtain the value.
 *
 * @return the referenced value of @p value of the thread at @p
 * source_thread_position.
 *
 * @note uses shared memory for the "broadcast" by the thread holding
 * the relevant value
 */
template <typename T, bool Synchronize = true, unsigned Dimensionality = 3>
KAT_FD T get_from_thread(const T& value, kat::position_t source_thread_position)
{
	using decayed_type = typename std::decay<T>::type;
	__shared__ static decayed_type tmp;
	if (kat::equals<Dimensionality>(threadIdx, source_thread_position)) {
		tmp = value;
	}
	if (Synchronize) { __syncthreads(); }
	return tmp;
}

/**
 * @brief have all block threads obtain a value held by the first
 * thread in the block (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for "broadcasting" the value
 */
template <typename T, bool Synchronize = true>
KAT_FD T get_from_first_thread(T&& value)
{
	return get_from_thread<T, Synchronize>(value, dim3{0,0,0} );
}

} // namespace block
} // namespace collaborative

namespace linear_grid {
namespace collaborative {
namespace block {

///@cond
// If we want to refer to other collaboration primitives, we'll make those references explicit;
// but we do want to be able to say `warp::id()` without prefixing that with anything.

namespace grid   = grid_info::grid;
namespace block  = grid_info::block;
namespace warp   = grid_info::warp;
namespace thread = grid_info::thread;
namespace lane   = grid_info::lane;

///@endcond

/*
 * TODO: Implement
 * KAT_FD  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

/**
 * Have all threads in (one/some/all) blocks perform some action over the
 * linear range of 0..length-1 - the same range for each block.
 *
 * @note This function semi-assumes the block size is a multiple of
 * the warp size; otherwise, it should still work, but - it'll be slow(ish).
 *
 * @param length The length of the range (of integers) on which to act
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 */
template <typename Function, typename Size = size_t>
KAT_FD void at_block_stride(Size length, const Function& f)
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
 * @brief Share one element of type T for each warp with the entire block -
 * using a single array in shared memory for all shared values.
 *
 * @param datum a warp-specific (but not thread-specific) piece of data,
 * one for each warp, which is to be shared with the whole block
 * @param where_to_make_available the various warp-specific data will be
 * stored here by warp index
 * @param writing_lane_id which lane in each warp should perform write operations
 *
 * @note if different threads in a warp have different values, behavior is
 * not guaranteed.
 */
template <typename T, bool Synchronize = true>
KAT_FD void share_per_warp_data(
	T                 datum,
	T*  __restrict__  where_to_make_available,
	unsigned          writing_lane_id)
{
	if (lane::index() == writing_lane_id) {
		where_to_make_available[warp::id()] = datum;
	}
	if (Synchronize) __syncthreads();
}

/**
 * @brief A variant of @ref share_per_warp_data , with the writing lane index
 * being decided dynamically in each lane based on who's actually active.
 */
template <typename T, bool Synchronize = true>
KAT_FD void share_per_warp_data(
	T                 datum,
	T*  __restrict__  where_to_make_available)
{
	return share_per_warp_data(
		std::forward<T>(datum),
		where_to_make_available,
		::kat::collaborative::warp::select_leader_lane());
}


KAT_FD void barrier() { __syncthreads(); }

/**
 * @brief have all block threads obtain a value held by just
 * one of the threads (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for the "broadcast" by the thread holding
 * the relevant value
 */
template <typename T, bool Synchronize = true>
KAT_FD T get_from_thread(T&& value, unsigned source_thread_id)
{
	using decayed_type = typename std::decay<T>::type;
	__shared__ static decayed_type tmp;
	if (thread::id_in_block() == source_thread_id) {
		tmp = value;
	}
	if (Synchronize) { __syncthreads(); }
	return tmp;
}

/**
 * @brief have all block threads obtain a value held by the first
 * thread in the block (and likely not otherwise easily accessible
 * to the rest of the block's threads).
 *
 * @note uses shared memory for "broadcasting" the value
 */
template <typename T, bool Synchronize = true>
KAT_FD T get_from_first_thread(T&& value)
{
	return get_from_thread<T, Synchronize>(value, 0u);
}

} // namespace block
} // namespace collaborative
} // namespace linear_grid


} // namespace kat

#endif // BLOCK_LEVEL_PRIMITIVES_CUH_
