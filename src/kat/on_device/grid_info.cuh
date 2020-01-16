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

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

#include <cassert>

namespace kat {

/**
 * A richer (kind-of-a-)wrapper for CUDA's `dim3` class, used
 * to specify dimensions for blocks and grid (up to 3 dimensions).
 *
 * @note same as `cuda::dimensions_t` from the cuda-api-wrappers library...
 *
 * @todo consider templating this on the number of dimensions.
 */
struct dimensions_t // this almost-inherits dim3
{
    grid_dimension_t x, y, z;
    constexpr KAT_FHD dimensions_t(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) noexcept
    : x(x_), y(y_), z(z_) {}

    constexpr KAT_FHD dimensions_t(const uint3& v) noexcept : dimensions_t(v.x, v.y, v.z) { }
    constexpr KAT_FHD dimensions_t(const dim3& dims) noexcept : dimensions_t(dims.x, dims.y, dims.z) { }

    constexpr KAT_FHD operator uint3(void) const noexcept { return { x, y, z }; }

    // This _should_ have been constexpr, but nVIDIA have not marked the dim3 constructors
    // as constexpr, so it isn't
    KAT_FHD operator dim3(void) const noexcept { return { x, y, z }; }

    constexpr KAT_FHD size_t volume() const noexcept { return (size_t) x * y * z; }
    constexpr KAT_FHD bool empty() const noexcept { return  (x == 0) or (y == 0) or (z == 0); }

	/**
	 * @brief The number of actual dimensions (i.e. dimensions/axes with more than a single value)
     */
    constexpr KAT_FHD unsigned dimensionality() const noexcept
    {
        return empty() ? 0 : ((z > 1) + (y > 1) + (x > 1));
    }
};


constexpr KAT_FHD bool operator==(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}

namespace detail {

/*// Note: This can very well overflow, but for CUDA upto 9.0,
// in practice - it can't
template <typename Size = unsigned>
KAT_FD Size row_major_linearization(dimensions_t position, dimensions_t dims)
{
	return
		((dims.z == 1) ? 0 : (position.z * dims.x * dims.y)) +
		((dims.y == 1) ? 0 : (position.y * dims.x)) +
		position.x;
}*/

template <unsigned NumDimensions = 3, typename Size = unsigned>
KAT_FD Size row_major_linearization(uint3 position, dimensions_t dims)
{
#pragma push
#pragma diag_suppress = code_is_unreachable
	switch(NumDimensions) {
	case 0: return 0; break;
	case 1: return position.x; break;
	case 2: return position.x + position.y * dims.x; break;
	case 3: return position.x + position.y * dims.x + position.z * dims.x * dims.y; break;
	}
	assert(false and "can't get here.");
	return {};
#pragma pop
}

} // namespace detail

/**
 * @return true if no non-trivial dimensions follow trivial dimensions
 *
 * @note Assumes non-empty dimensions!
 */
KAT_FD bool dimensionality_is_canonical(dimensions_t dims)
{
#if __cplusplus >= 201402L
	assert(not dims.empty());
#endif
	return
		(dims.x > 1 or (dims.y == 1 and dims.z == 1)) and
		(dims.y > 1 or dims.z == 1);
}

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

/**
 * @note These are the dimensions of the grid over blocks; the blocks may have additional "dimensions" relative to threads.
 */
KAT_FD dimensions_t dimensions_in_blocks()  { return gridDim; }
KAT_FD size_t       num_blocks()            { return dimensions_in_blocks().volume(); }
KAT_FD dimensions_t first_block_position()  { return dimensions_t{0, 0, 0}; }
KAT_FD dimensions_t last_block_position()   { return dimensions_t{gridDim.x - 1, gridDim.y - 1, gridDim.z - 1}; }

/**
 * Determines whether the grid's non-trivial dimensions - in blocks and in threads - are on the x axis only.
 *
 * @note One could consider y-only or z-only dimensions as linear; this definition was chosen for convenience
 * (and performance) and is used throughout this library
 */
KAT_FD bool         is_linear()
{
	return gridDim.y == 1 and gridDim.z == 1 and blockDim.y == 1 and blockDim.z == 1;
}

/**
 * @note These are the dimensions of the grid in terms of threads. This means that a grid can have less blocks (or
 * even one block) in each dimension, but each block many have multiple threads, contributing to the overall dimension.
 */
KAT_FD dimensions_t dimensions_in_threads()     { return dimensions_t{ gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z }; }


} // namespace grid

namespace block {

KAT_FD dimensions_t dimensions()            { return blockDim; }
KAT_FD dimensions_t position_in_grid()      { return blockIdx; }
KAT_FD bool         is_first_in_grid()      { return blockIdx == grid::first_block_position(); };
KAT_FD bool         is_last_in_grid()       { return blockIdx == grid::last_block_position(); };
template <unsigned NumDimensions = 3>
KAT_FD dimensions_t index()                 { return detail::row_major_linearization<NumDimensions>(position_in_grid(), grid::dimensions_in_blocks()); }
KAT_FD grid_block_dimension_t
                    size()                  { return dimensions().volume(); }
KAT_FD grid_block_dimension_t
                    num_full_warps()        { return block::size() / warp_size; }
KAT_FD dimensions_t first_thread_position() { return dimensions_t{0, 0, 0}; }
KAT_FD dimensions_t last_thread_position()  { return dimensions_t{blockDim.x - 1, blockDim.y - 1, blockDim.z - 1}; }

KAT_FD grid_block_dimension_t num_warps()
{
	return (block::size() + warp_size - 1) >> log_warp_size;
		// While this form of rounded-up-division may generally overflow, that's not possible
		// here, since CUDA block size is capped at 1024 as of 2019, and is unlikely to get close
		// to the maximum integer value.
}

KAT_FD grid_block_dimension_t
                    index_of_first_warp()   { return 0; }
KAT_FD grid_block_dimension_t
                    index_of_last_warp()    { return num_warps() - 1; }
KAT_FD bool         is_linear()             { return dimensions().y == 1 and dimensions().z == 1; }


} // namespace block

namespace thread_block = block;

namespace grid {

KAT_FD unsigned     num_warps()             { return num_blocks() * block::num_warps(); }
KAT_FD unsigned     num_threads()           { return num_blocks() * block::size(); }
KAT_FD unsigned     total_size()            { return num_threads(); }
KAT_FD unsigned     num_warps_per_block()   { return block::num_warps(); }

} // namespace grid

namespace warp {

enum : unsigned { first_lane = 0, last_lane = warp_size - 1 };

KAT_FD unsigned size()   { return warp_size; }
KAT_FD unsigned length() { return warp_size; }

} // namespace warp

namespace thread {

KAT_FD uint3    position()            { return threadIdx; }
KAT_FD uint3    position_in_block()   { return threadIdx; }

KAT_FD bool     is_first_in_block()   { return position_in_block() == block::first_thread_position();    };
KAT_FD bool     is_last_in_block()    { return position_in_block() == block::last_thread_position();     };
KAT_FD bool     is_first_in_grid()    { return block::is_first_in_grid() and thread::is_first_in_block(); }
KAT_FD bool     is_last_in_grid()     { return block::is_last_in_grid() and thread::is_last_in_block();   }


/**
 *
 * @param thread_position_in_block the NumDimensions-dimensional version of the thread index within its grid block,
 * represented in the 3-dimensional dimensions_t structure
 * @return The 1-d index of the specified thread within the entire grid, when it's
 * flattened so that threads with identical z and y axis coordinates form a contiguous
 * sequence
 */
template <unsigned NumDimensions = 3>
KAT_FD unsigned  index(dimensions_t thread_position_in_block)
{
	return detail::row_major_linearization<NumDimensions, unsigned>(thread_position_in_block, block::dimensions());
}
template <unsigned NumDimensions = 3>
KAT_FD unsigned  index_in_block()      { return index(thread::position_in_block()); }

template <unsigned NumDimensions = 3>
KAT_FD unsigned  index()               { return index_in_block(); }

template <unsigned NumDimensions = 3>
KAT_FD unsigned  index_in_grid(uint3 block_position_in_grid, uint3 thread_index)
{

	return
		detail::row_major_linearization<NumDimensions, unsigned>(block_position_in_grid, grid::dimensions_in_blocks()) +
		detail::row_major_linearization<NumDimensions, unsigned>(thread_index, block::dimensions());
}

template <unsigned NumDimensions = 3>
KAT_FD unsigned  index_in_grid()
{
	return index_in_grid<NumDimensions>(block::position_in_grid(), thread::position_in_block());
}
template <unsigned NumDimensions = 3>
KAT_FD unsigned global_index()      { return index_in_grid<NumDimensions>(); }


} // namespace thread

namespace warp {

template <unsigned NumDimensions = 3>
KAT_FD unsigned index()          { return grid_info::thread::index<NumDimensions>() / warp_size; }
template <unsigned NumDimensions = 3>
KAT_FD unsigned index_in_grid()  { return grid_info::thread::index_in_grid<NumDimensions>() / warp_size; }

KAT_FD unsigned global_index()   { return index_in_grid(); }
KAT_FD unsigned index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

KAT_FD unsigned index_in_block_of_first_lane() { return index_of_first_lane(); }

KAT_FD unsigned global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index() & lane_index_mask;
}

KAT_FD unsigned index_in_grid_of_first_lane()
                                     { return warp::global_index_of_first_lane(); }
KAT_FD unsigned int index_in_block() { return thread::index() >> log_warp_size; }
KAT_FD unsigned int index()          { return warp::index(); }
KAT_FD bool is_first_in_block()      { return warp::index_in_block() == block::index_of_first_warp(); }
KAT_FD bool is_last_in_block()       { return warp::index_in_block() == block::index_of_last_warp(); }
KAT_FD bool is_first_in_grid()       { return warp::is_first_in_block() and block::is_first_in_grid(); }
KAT_FD bool is_last_in_grid()        { return warp::is_last_in_block() and block::is_last_in_grid(); }


} // namespace warp

namespace lane {

enum { half_warp_size = warp_size / 2 };


KAT_FD unsigned index(unsigned thread_index)
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

KAT_FD unsigned index()                   { return index(threadIdx.x); }
KAT_FD unsigned index_in_warp()           { return index(); }
KAT_FD unsigned is_first()                { return index_in_warp() == warp::first_lane; }
KAT_FD unsigned is_last()                 { return index_in_warp() == warp::last_lane; }

KAT_FD unsigned index_in_half_warp(unsigned thread_or_lane_index)
{
	enum { half_warp_index_mask = half_warp_size - 1 };
	return thread_or_lane_index & half_warp_index_mask;
}

KAT_FD unsigned index_in_half_warp()      { return index_in_half_warp(threadIdx.x); }
KAT_FD unsigned is_in_first_half_warp()   { return index_in_warp() < half_warp_size; }
KAT_FD unsigned is_in_second_half_warp()  { return index_in_warp() >= half_warp_size; }


} // namespace lane

namespace thread {

KAT_FD bool     is_first_in_warp()    { return lane::index() == warp::first_lane; }
KAT_FD bool     is_last_in_warp()     { return lane::index_in_warp() == warp::last_lane; }

} // namespace thread

} // namespace grid_info

// I couldn't use '1d, 2d, 3d since those aren't valid identifiers...
namespace linear_grid {

namespace grid_info {

namespace grid {

// TODO: Should we use the same return types as for the non-linear case?
// For now, we don't, relying on the implicit convertibility of the
// return types here to the general-case ones. But some of the types
// are admittedly a bit fudged.

KAT_FD decltype(gridDim.x) dimensions_in_blocks()    { return gridDim.x; }
KAT_FD grid_dimension_t  num_blocks()              { return gridDim.x; }
KAT_FD grid_dimension_t  index_of_first_block()    { return 0; }
KAT_FD grid_dimension_t  index_of_last_block()     { return num_blocks() - 1; }
KAT_FD grid_dimension_t  first_block_position()    { return index_of_first_block(); }
KAT_FD grid_dimension_t  first_last_position()     { return index_of_last_block(); }


} // namespace grid

namespace block {

using kat::grid_info::block::dimensions;
KAT_FD grid_block_dimension_t  index()                   { return blockIdx.x; }
KAT_FD unsigned                index_in_grid()           { return index(); }
KAT_FD grid_block_dimension_t  position_in_grid()        { return index_in_grid(); }
KAT_FD bool                    is_first_in_grid()        { return block::index_in_grid() == grid::index_of_first_block(); }
KAT_FD bool                    is_last_in_grid()         { return index() == grid::index_of_last_block(); }
KAT_FD grid_block_dimension_t  length()                  { return blockDim.x; }
KAT_FD grid_block_dimension_t  size()                    { return length(); }
KAT_FD grid_block_dimension_t  num_threads()             { return length(); }
KAT_FD unsigned                num_full_warps()          { return length() >> log_warp_size; }
KAT_FD unsigned                index_of_first_thread()   { return 0; }
KAT_FD unsigned                index_of_last_thread()    { return num_threads() - 1; }
KAT_FD unsigned                first_thread_position()   { return index_of_first_thread(); }
KAT_FD unsigned                last_thread_position()    { return index_of_last_thread(); }
KAT_FD unsigned num_warps()
{
	return (block::size() + warp_size - 1) >> log_warp_size;
		// While this form of rounded-up-division may generally overflow, that's not possible
		// here, since CUDA block size is capped at 1024 as of 2019, and is unlikely to get close
		// to the maximum integer value.
}
KAT_FD grid_block_dimension_t  index_of_first_warp()     { return 0; }
KAT_FD grid_block_dimension_t  index_of_last_warp()      { return num_warps() - 1; }
KAT_FD grid_block_dimension_t  index_of_last_full_warp() { return num_full_warps() - 1; }
KAT_FD bool                    is_linear()               { return true; }

/**
 * @note These are the dimensions of the grid in terms of threads. This means that a grid can have less blocks (or
 * even one block) in each dimension, but each block many have multiple threads, contributing to the overall dimension.
 */
KAT_FD dimensions_t dimensions_in_threads()     { return dimensions_t{ gridDim.x * blockDim.x }; }

} // namespace block

namespace thread_block = block;

namespace grid {

KAT_FD unsigned     num_warps()             { return num_blocks() * block::num_warps(); }
KAT_FD unsigned     num_threads()           { return num_blocks() * block::size(); }
KAT_FD unsigned     total_size()            { return num_threads(); }
KAT_FD unsigned     num_warps_per_block()   { return block::num_warps(); }

} // namespace grid

namespace warp {

using kat::grid_info::warp::first_lane;
using kat::grid_info::warp::last_lane;
using kat::grid_info::warp::size;
using kat::grid_info::warp::length;

}

namespace thread {

KAT_FD grid_block_dimension_t  index(uint3 position_in_block)
                                                   { return position_in_block.x; }
KAT_FD grid_block_dimension_t  index_in_block()    { return threadIdx.x; }
KAT_FD grid_block_dimension_t  index()             { return index_in_block(); }

KAT_FD grid_block_dimension_t  position()          { return index_in_block(); }
KAT_FD grid_block_dimension_t  position_in_block() { return index_in_block(); }

KAT_FD bool                    is_first_in_block() { return index_in_block() == block::first_thread_position(); }
KAT_FD bool                    is_last_in_block()  { return index_in_block() == block::last_thread_position(); }

KAT_FD bool                    is_first_in_grid()  { return block::is_first_in_grid() and thread::is_first_in_block(); }
KAT_FD bool                    is_last_in_grid()   { return block::is_last_in_grid() and thread::is_last_in_block();   }

using ::kat::grid_info::thread::is_first_in_warp;
using ::kat::grid_info::thread::is_last_in_warp;


/**
 * Returns the global index of the thread - not within the block (the work group), but
 * considering all threads for the current kernel together - assuming a one-dimensional
 * grid.
 */
KAT_FD unsigned index_in_grid(grid_dimension_t block_index, grid_dimension_t thread_index)
{
	return thread_index + block_index * block::size();
}

KAT_FD unsigned index_in_grid()     { return index_in_grid(block::index(), index()); }
KAT_FD unsigned global_index()      { return index_in_grid(); }


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
KAT_FD unsigned block_stride_start_position(unsigned serialization_factor = 1)
{
	return index() + serialization_factor * block::index() * block::length();
}

} // namespace thread

namespace warp {

KAT_FD grid_block_dimension_t index()          { return thread::index_in_block() >> log_warp_size; }
KAT_FD unsigned               index_in_grid()  { return thread::index_in_grid() >> log_warp_size; }
KAT_FD unsigned               global_index()   { return index_in_grid(); }

KAT_FD unsigned index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

KAT_FD unsigned index_in_block_of_first_lane() { return index_of_first_lane(); }

KAT_FD unsigned global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index() & lane_index_mask;
}
KAT_FD unsigned index_in_grid_of_first_lane()  { return warp::global_index_of_first_lane(); }
KAT_FD unsigned int index_in_block()           { return warp::index(); }
KAT_FD bool is_first_in_block()                { return warp::index_in_block() == block::index_of_first_warp(); }
KAT_FD bool is_last_in_block()                 { return warp::index_in_block() == block::index_of_last_warp(); }
KAT_FD bool is_first_in_grid()                 { return warp::is_first_in_block() and block::is_first_in_grid(); }
KAT_FD bool is_last_in_grid()                  { return warp::is_last_in_block() and block::is_last_in_grid(); }

} // namespace warp


namespace lane = ::kat::grid_info::lane;

} // namespace grid_info

} // namespace linear_grid

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_
