/**
 * @file grid_info.cuh
 *
 * @brief Information regarindg the current kernel launch grid and positions within it
 *
 * @note Currently, CUDA does not allows more than 2^31 threads in a launch grid, hence many functions
 * here return unsigned / dimension_t
 *
 * @todo Consider converting the `unsigned` return types to `native_word_t`
 */

#pragma once
#ifndef CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_
#define CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_

#include <kat/on_device/common.cuh>

#include <kat/define_specifiers.hpp>

//#if defined(HAVE_CUDA_API_WRAPPERS) || (defined(CUDA_API_WRAPPERS_TYPES_HPP_) && defined(CUDA_API_WRAPPERS_CONSTANTS_HPP_))
#if 0
using cuda::dimensions_t;
#else
/**
 * A richer (kind-of-a-)wrapper for CUDA's @ref dim3 class, used
 * to specify dimensions for blocks and grid (up to 3 dimensions).
 *
 * @note Unfortunately, dim3 does not have constexpr methods -
 * preventing us from having constexpr methods here.
 */
struct dimensions_t // this almost-inherits dim3
{
    grid_dimension_t x, y, z;
    constexpr __hd__ dimensions_t(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
    : x(x_), y(y_), z(z_) {}

    __hd__ constexpr dimensions_t(const uint3& v) : dimensions_t(v.x, v.y, v.z) { }
    __hd__ constexpr dimensions_t(const dim3& dims) : dimensions_t(dims.x, dims.y, dims.z) { }

    __hd__ constexpr operator uint3(void) const { return { x, y, z }; }

    // This _should_ have been constexpr, but nVIDIA have not marked the dim3 constructors
    // as constexpr, so it isn't
    __hd__ operator dim3(void) const { return { x, y, z }; }

    __fhd__ constexpr size_t volume() const { return (size_t) x * y * z; }
    __fhd__ constexpr bool empty() const {  return volume() == 0; }
    __fhd__ constexpr unsigned char dimensionality() const
    {
        return ((z > 1) + (y > 1) + (x > 1)) * (!empty());
    }
};
#endif

namespace detail {

// Note: This can very well overflow, but for CUDA upto 9.0,
// in practice - it can't
template <typename Size = unsigned>
__fd__ Size row_major_linearization(dimensions_t position, dimensions_t dims)
{
	return
		((dims.z == 1) ? 0 : (position.z * dims.x * dims.y)) +
		((dims.y == 1) ? 0 : (position.y * dims.x)) +
		position.x;
}

} // namespace detail

/**
 *
 ************************************************************************
 * Convenience one-liners relating to grid dimensions, indices within
 * the grid, block or warp, lane functionality etc.
 ************************************************************************
 */


// TODO: Perhaps have functions for strided copy in and out

namespace grid_info {

namespace grid {

__fd__ dimensions_t dimensions()            { return gridDim; }
__fd__ size_t       num_blocks()            { return dimensions().volume(); }
__fd__ dimensions_t first_block_position()  { return dimensions_t{0, 0, 0}; }
__fd__ dimensions_t last_block_position()   { return dimensions_t{gridDim.x - 1, gridDim.y - 1, gridDim.z - 1}; }

} // namespace grid

namespace block {

__fd__ dimensions_t dimensions()            { return blockDim; }
__fd__ dimensions_t position_in_grid()      { return blockIdx; }
__fd__ bool         is_first_in_grid()      { return blockIdx == grid::first_block_position(); };
__fd__ bool         is_last_in_grid()       { return blockIdx == grid::last_block_position(); };
__fd__ dimensions_t row_major_index()       { return detail::row_major_linearization(blockIdx, blockDim); }
__fd__ grid_block_dimension_t
                    size()                  { return dimensions().volume(); }
__fd__ grid_block_dimension_t
                    num_warps()             { return (block::size() + warp_size - 1) / warp_size; }
__fd__ grid_block_dimension_t
                    num_full_warps()        { return block::size() / warp_size; }
__fd__ dimensions_t first_thread_position() { return dimensions_t{0, 0, 0}; }
__fd__ dimensions_t last_thread_position()  { return dimensions_t{blockDim.x - 1, blockDim.y - 1, blockDim.z - 1}; }

} // namespace block

namespace grid {

__fd__ unsigned     num_warps()             { return num_blocks() * block::num_warps(); }
__fd__ unsigned     num_threads()           { return num_blocks() * block::size(); }
__fd__ unsigned     total_size()            { return num_threads(); }

} // namespace grid

namespace warp {

enum : unsigned { first_lane = 0, last_lane = warp_size - 1 };

__fd__ unsigned size()   { return warp_size; }
__fd__ unsigned length() { return warp_size; }

} // namespace warp

namespace half_warp {

enum { size = warp_size / 2 };

} // namespace warp

namespace thread {

__fd__ dimensions_t position()            { return threadIdx; }
__fd__ dimensions_t position_in_block()   { return threadIdx; }

__fd__ bool         is_first_in_block()   { return position_in_block() == block::first_thread_position();    };
__fd__ bool         is_last_in_block()    { return position_in_block() == block::last_thread_position();     };
__fd__ bool         is_first_in_grid()    { return block::is_first_in_grid() && thread::is_first_in_block(); }
__fd__ bool         is_last_in_grid()     { return block::is_last_in_grid() && thread::is_last_in_block();   }


/**
 * @param thread_index the 3-dimensional version of the thread index within its grid block
 * @return The 1-d index of the specified thread within the entire grid, when it's
 * flattened so that threads with identical z and y axis coordinates form a contiguous
 * sequence
 */
__fd__ unsigned row_major_index(dimensions_t thread_position_in_block)
{
	return detail::row_major_linearization<unsigned>(thread_position_in_block, block::dimensions());
}
__fd__ unsigned row_major_index()         { return row_major_index(position_in_block()); }


__fd__ unsigned row_major_index_in_grid(uint3 block_index, uint3 thread_index)
{
	return
		detail::row_major_linearization<unsigned>(block::position_in_grid(), block::dimensions()) +
		detail::row_major_linearization<unsigned>(thread_index, block::dimensions());
}

__fd__ unsigned row_major_index_in_grid()
{
	return row_major_index_in_grid(block::position_in_grid(), position());
}

} // namespace thread

namespace lane {

__fd__ unsigned index(unsigned thread_index)
{
	// we could use a special register:
	//
	//   return builtins::lane_index();
	//
	// but apparently, retrieving a special register takes a good
	// number of clock cycles (why?!), so in practice, this is
	// probably faster:
	enum { lane_id_mask = warp_size - 1 };
	return thread_index & lane_id_mask;
}

__fd__ unsigned index()                   { return index(threadIdx.x); }
__fd__ unsigned index_in_warp()           { return index(); }
__fd__ unsigned is_first()                { return index_in_warp() == warp::first_lane; }
__fd__ unsigned is_last()                 { return index_in_warp() == warp::last_lane; }

__fd__ unsigned index_in_half_warp(unsigned thread_or_lane_index)
{
	enum { half_warp_index_mask = half_warp::size - 1 };
	return thread_or_lane_index & half_warp_index_mask;
}

__fd__ unsigned index_in_half_warp()      { return index_in_half_warp(threadIdx.x); }
__fd__ unsigned is_in_first_half_warp()   { return index_in_warp() < half_warp::size; }
__fd__ unsigned is_in_second_half_warp()  { return index_in_warp() >= half_warp::size; }


} // namespace lane

namespace thread {

__fd__ bool is_first_in_warp()            { return lane::index() == warp::first_lane; }
__fd__ bool is_last_in_warp()             { return lane::index_in_warp() == warp::last_lane; }

} // namespace thread


// I couldn't use '1d, 2d, 3d since those aren't valid identifiers...
namespace linear {

namespace grid {

__fd__ unsigned num_blocks()              { return gridDim.x; }
__fd__ unsigned index_of_first_block()    { return 0; }
__fd__ unsigned index_of_last_block()     { return num_blocks() - 1; }

} // namespace grid

namespace block {

__fd__ unsigned index()                   { return blockIdx.x; }
__fd__ unsigned index_in_grid()           { return index(); }
__fd__ bool     is_first_in_grid()        { return block::index_in_grid() == grid::index_of_first_block(); }
__fd__ bool     is_last_in_grid()         { return index() == grid::index_of_last_block(); }
__fd__ unsigned size()                    { return blockDim.x; }
__fd__ unsigned length()                  { return blockDim.x; }
__fd__ unsigned num_full_warps()          { return blockDim.x >> log_warp_size; }
__fd__ unsigned num_threads()             { return blockDim.x; }
__fd__ unsigned index_of_first_thread()   { return 0; }
__fd__ unsigned index_of_last_thread()    { return num_threads() - 1; }
__fd__ unsigned first_thread_position()   { return index_of_first_thread(); }
__fd__ unsigned last_thread_position()    { return index_of_last_thread(); }

__fd__ unsigned num_warps()
{
	// This equals div_rounding_up(blockDim.x, warp_size) and can't overflow
	return (size() + warp_size - 1) >> log_warp_size;
}
__fd__ unsigned index_of_first_warp()     { return 0; }
__fd__ unsigned index_of_last_warp()      { return num_warps() - 1; }
__fd__ unsigned index_of_last_full_warp() { return num_full_warps() - 1; }

} // namespace block

namespace grid {

__fd__ unsigned num_threads()             { return num_blocks() * block::size(); }
__fd__ unsigned total_size()              { return num_threads(); }
__fd__ unsigned num_warps()               { return num_blocks() * block::num_warps(); }
__fd__ unsigned num_warps_per_block()     { return block::num_warps(); }

} // namespace grid

namespace thread_block = block;

namespace thread {

__fd__ unsigned index()             { return threadIdx.x; }
__fd__ unsigned index_in_block()    { return index(); }
__fd__ bool     is_first_in_block() { return index_in_block() == block::first_thread_position(); }
__fd__ bool     is_last_in_block()  { return index_in_block() == block::last_thread_position(); }

__fd__ dimensions_t position()            { return index(); }
__fd__ dimensions_t position_in_block()   { return index(); }

__fd__ bool     is_first_in_grid()    { return block::is_first_in_grid() && thread::is_first_in_block(); }
__fd__ bool     is_last_in_grid()     { return block::is_last_in_grid() && thread::is_last_in_block();   }



/**
 * Returns the global index of the thread - not within the block (the work group), but
 * considering all threads for the current kernel together - assuming a one-dimensional
 * grid.
 */
__fd__ unsigned index_in_grid(grid_dimension_t block_index, grid_dimension_t thread_index)
{
	return thread_index + block_index * block::size();
}

__fd__ unsigned index_in_grid()     { return index_in_grid(block::index(), index()); }
__fd__ unsigned global_index()      { return index_in_grid(); }


/**
 * Use this for kernels in a 1-dimensional (linear) grid, in which each block of K
 * threads handles K * serialization_factor consecutive elements. That's pretty
 * common... (?)
 *
 * Anyway, each individual thread accesses data with a stride of K.
 *
 * @param serialization_factor The number of elements each thread would access
 * @return the initial position for a given thread
 */
__fd__ unsigned block_stride_start_position(unsigned serialization_factor = 1)
{
	return index() + serialization_factor * block::index() * block::length();
}

} // namespace thread

namespace lane = ::grid_info::lane;

namespace thread {

// TODO: Move this out of linear
__fd__ bool is_first_in_warp() { return lane::index() == warp::first_lane; }
__fd__ bool is_last_in_warp()  { return lane::index_in_warp() == warp::last_lane; }

} // namespace thread


namespace warp {

using grid_info::warp::first_lane;
using grid_info::warp::last_lane;

/**
 * Returns the global index of the warp the calling thread is in - not within the block
 * (the work group), but considering all blocks for the current kernel together -
 * assuming a one-dimensional grid.
 */
__fd__ unsigned global_index() { return thread::global_index() >> log_warp_size; }

__fd__ unsigned index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

__fd__ unsigned global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index() & lane_index_mask;
}

__fd__ unsigned index_in_grid()      { return warp::global_index(); }
__fd__ unsigned index_of_first_lane_in_grid()
                                     { return warp::global_index_of_first_lane(); }
__fd__ unsigned int index()          { return thread::index() >> log_warp_size; }
__fd__ unsigned int index_in_block() { return warp::index(); }
__fd__ bool is_first_in_block()      { return warp::index_in_block() == block::index_of_first_warp(); }
__fd__ bool is_last_in_block()       { return warp::index_in_block() == block::index_of_last_warp(); }
__fd__ bool is_first_in_grid()       { return warp::is_first_in_block() && block::is_first_in_grid(); }
__fd__ bool is_last_in_grid()        { return warp::is_last_in_block() && block::is_last_in_grid(); }

} // namespace warp

} // namespace linear

} // namespace grid_info

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_
