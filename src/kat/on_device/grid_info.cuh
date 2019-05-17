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
#include <kat/define_specifiers.hpp>
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
    constexpr __fhd__ dimensions_t(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) noexcept
    : x(x_), y(y_), z(z_) {}

    constexpr __fhd__ dimensions_t(const uint3& v) noexcept : dimensions_t(v.x, v.y, v.z) { }
    constexpr __fhd__ dimensions_t(const dim3& dims) noexcept : dimensions_t(dims.x, dims.y, dims.z) { }

    constexpr __fhd__ operator uint3(void) const noexcept { return { x, y, z }; }

    // This _should_ have been constexpr, but nVIDIA have not marked the dim3 constructors
    // as constexpr, so it isn't
    __fhd__ operator dim3(void) const noexcept { return { x, y, z }; }

    constexpr __fhd__ size_t volume() const noexcept { return (size_t) x * y * z; }
    constexpr __fhd__ bool empty() const noexcept {  return volume() == 0; }
    constexpr __fhd__ unsigned char dimensionality() const noexcept
    {
        return ((z > 1) + (y > 1) + (x > 1)) * (!empty());
    }
};


constexpr __fhd__ bool operator==(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}

namespace detail {

/*// Note: This can very well overflow, but for CUDA upto 9.0,
// in practice - it can't
template <typename Size = unsigned>
__fd__ Size row_major_linearization(dimensions_t position, dimensions_t dims)
{
	return
		((dims.z == 1) ? 0 : (position.z * dims.x * dims.y)) +
		((dims.y == 1) ? 0 : (position.y * dims.x)) +
		position.x;
}*/

template <unsigned NumDimensions = 3, typename Size = unsigned>
__fd__ Size row_major_linearization(uint3 position, dimensions_t dims)
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
__fd__ unsigned     dimensionality()        { return dimensions().dimensionality(); }
__fd__ size_t       num_blocks()            { return dimensions().volume(); }
__fd__ dimensions_t first_block_position()  { return dimensions_t{0, 0, 0}; }
__fd__ dimensions_t last_block_position()   { return dimensions_t{gridDim.x - 1, gridDim.y - 1, gridDim.z - 1}; }

} // namespace grid

namespace block {

__fd__ dimensions_t dimensions()            { return blockDim; }
__fd__ dimensions_t position_in_grid()      { return blockIdx; }
__fd__ bool         is_first_in_grid()      { return blockIdx == grid::first_block_position(); };
__fd__ bool         is_last_in_grid()       { return blockIdx == grid::last_block_position(); };
template <unsigned NumDimensions = 3>
__fd__ dimensions_t index()                 { return detail::row_major_linearization<NumDimensions>(position_in_grid(), grid::dimensions()); }
__fd__ grid_block_dimension_t
                    size()                  { return dimensions().volume(); }
__fd__ grid_block_dimension_t
                    num_full_warps()        { return block::size() / warp_size; }
__fd__ dimensions_t first_thread_position() { return dimensions_t{0, 0, 0}; }
__fd__ dimensions_t last_thread_position()  { return dimensions_t{blockDim.x - 1, blockDim.y - 1, blockDim.z - 1}; }

__fd__ grid_block_dimension_t num_warps()
{
	return (block::size() + warp_size - 1) >> log_warp_size;
		// While this form of rounded-up-division may generally overflow, that's not possible
		// here, since CUDA block size is capped at 1024 as of 2019, and is unlikely to get close
		// to the maximum integer value.
}

__fd__ grid_block_dimension_t
                    index_of_first_warp()   { return 0; }
__fd__ grid_block_dimension_t
                    index_of_last_warp()    { return num_warps() - 1; }


} // namespace block

namespace thread_block = block;

namespace grid {

__fd__ unsigned     num_warps()             { return num_blocks() * block::num_warps(); }
__fd__ unsigned     num_threads()           { return num_blocks() * block::size(); }
__fd__ unsigned     total_size()            { return num_threads(); }
__fd__ unsigned     num_warps_per_block()   { return block::num_warps(); }

} // namespace grid

namespace warp {

enum : unsigned { first_lane = 0, last_lane = warp_size - 1 };

__fd__ unsigned size()   { return warp_size; }
__fd__ unsigned length() { return warp_size; }

} // namespace warp

namespace thread {

__fd__ uint3    position()            { return threadIdx; }
__fd__ uint3    position_in_block()   { return threadIdx; }

__fd__ bool     is_first_in_block()   { return position_in_block() == block::first_thread_position();    };
__fd__ bool     is_last_in_block()    { return position_in_block() == block::last_thread_position();     };
__fd__ bool     is_first_in_grid()    { return block::is_first_in_grid() and thread::is_first_in_block(); }
__fd__ bool     is_last_in_grid()     { return block::is_last_in_grid() and thread::is_last_in_block();   }


/**
 *
 * @param thread_position_in_block the NumDimensions-dimensional version of the thread index within its grid block,
 * represented in the 3-dimensional dimensions_t structure
 * @return The 1-d index of the specified thread within the entire grid, when it's
 * flattened so that threads with identical z and y axis coordinates form a contiguous
 * sequence
 */
template <unsigned NumDimensions = 3>
__fd__ unsigned  index(dimensions_t thread_position_in_block)
{
	return detail::row_major_linearization<NumDimensions, unsigned>(thread_position_in_block, block::dimensions());
}
template <unsigned NumDimensions = 3>
__fd__ unsigned  index_in_block()      { return index(thread::position_in_block()); }

template <unsigned NumDimensions = 3>
__fd__ unsigned  index()               { return index_in_block(); }

template <unsigned NumDimensions = 3>
__fd__ unsigned  index_in_grid(uint3 block_position_in_grid, uint3 thread_index)
{

	return
		detail::row_major_linearization<NumDimensions, unsigned>(block_position_in_grid, grid::dimensions()) +
		detail::row_major_linearization<NumDimensions, unsigned>(thread_index, block::dimensions());
}

template <unsigned NumDimensions = 3>
__fd__ unsigned  index_in_grid()
{
	return index_in_grid<NumDimensions>(block::position_in_grid(), thread::position_in_block());
}
template <unsigned NumDimensions = 3>
__fd__ unsigned global_index()      { return index_in_grid<NumDimensions>(); }


} // namespace thread

namespace warp {

template <unsigned NumDimensions = 3>
__fd__ unsigned index()          { return grid_info::thread::index<NumDimensions>() / warp_size; }
template <unsigned NumDimensions = 3>
__fd__ unsigned index_in_grid()  { return grid_info::thread::index_in_grid<NumDimensions>() / warp_size; }

__fd__ unsigned global_index()   { return index_in_grid(); }
__fd__ unsigned index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

__fd__ unsigned index_in_block_of_first_lane() { return index_of_first_lane(); }

__fd__ unsigned global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index() & lane_index_mask;
}

__fd__ unsigned index_in_grid_of_first_lane()
                                     { return warp::global_index_of_first_lane(); }
__fd__ unsigned int index_in_block() { return thread::index() >> log_warp_size; }
__fd__ unsigned int index()          { return warp::index(); }
__fd__ bool is_first_in_block()      { return warp::index_in_block() == block::index_of_first_warp(); }
__fd__ bool is_last_in_block()       { return warp::index_in_block() == block::index_of_last_warp(); }
__fd__ bool is_first_in_grid()       { return warp::is_first_in_block() and block::is_first_in_grid(); }
__fd__ bool is_last_in_grid()        { return warp::is_last_in_block() and block::is_last_in_grid(); }


} // namespace warp

namespace lane {

enum { half_warp_size = warp_size / 2 };


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
	enum { half_warp_index_mask = half_warp_size - 1 };
	return thread_or_lane_index & half_warp_index_mask;
}

__fd__ unsigned index_in_half_warp()      { return index_in_half_warp(threadIdx.x); }
__fd__ unsigned is_in_first_half_warp()   { return index_in_warp() < half_warp_size; }
__fd__ unsigned is_in_second_half_warp()  { return index_in_warp() >= half_warp_size; }


} // namespace lane

namespace thread {

__fd__ bool     is_first_in_warp()    { return lane::index() == warp::first_lane; }
__fd__ bool     is_last_in_warp()     { return lane::index_in_warp() == warp::last_lane; }

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

using kat::grid_info::grid::dimensions;

__fd__ unsigned          dimensionality()          { return gridDim.x > 0 ? 1 : 0; }
__fd__ grid_dimension_t  num_blocks()              { return gridDim.x; }
__fd__ grid_dimension_t  index_of_first_block()    { return 0; }
__fd__ grid_dimension_t  index_of_last_block()     { return num_blocks() - 1; }
__fd__ grid_dimension_t  first_block_position()    { return index_of_first_block(); }
__fd__ grid_dimension_t  first_last_position()     { return index_of_last_block(); }


} // namespace grid

namespace block {

__fd__ grid_block_dimension_t  index()                   { return blockIdx.x; }
__fd__ unsigned                index_in_grid()           { return index(); }
__fd__ grid_block_dimension_t  position_in_grid()        { return index_in_grid(); }
__fd__ bool                    is_first_in_grid()        { return block::index_in_grid() == grid::index_of_first_block(); }
__fd__ bool                    is_last_in_grid()         { return index() == grid::index_of_last_block(); }
__fd__ grid_block_dimension_t  length()                  { return blockDim.x; }
__fd__ grid_block_dimension_t  size()                    { return length(); }
__fd__ grid_block_dimension_t  num_threads()             { return length(); }
__fd__ unsigned                num_full_warps()          { return length() >> log_warp_size; }
__fd__ unsigned                index_of_first_thread()   { return 0; }
__fd__ unsigned                index_of_last_thread()    { return num_threads() - 1; }
__fd__ unsigned                first_thread_position()   { return index_of_first_thread(); }
__fd__ unsigned                last_thread_position()    { return index_of_last_thread(); }
__fd__ unsigned num_warps()
{
	return (block::size() + warp_size - 1) >> log_warp_size;
		// While this form of rounded-up-division may generally overflow, that's not possible
		// here, since CUDA block size is capped at 1024 as of 2019, and is unlikely to get close
		// to the maximum integer value.
}
__fd__ grid_block_dimension_t  index_of_first_warp()     { return 0; }
__fd__ grid_block_dimension_t  index_of_last_warp()      { return num_warps() - 1; }
__fd__ grid_block_dimension_t  index_of_last_full_warp() { return num_full_warps() - 1; }

} // namespace block

namespace thread_block = block;

namespace grid {

__fd__ unsigned     num_warps()             { return num_blocks() * block::num_warps(); }
__fd__ unsigned     num_threads()           { return num_blocks() * block::size(); }
__fd__ unsigned     total_size()            { return num_threads(); }
__fd__ unsigned     num_warps_per_block()   { return block::num_warps(); }

} // namespace grid

namespace warp {

using kat::grid_info::warp::first_lane;
using kat::grid_info::warp::last_lane;
using kat::grid_info::warp::size;
using kat::grid_info::warp::length;

}

namespace thread {

__fd__ grid_block_dimension_t  index(uint3 position_in_block)
                                                   { return position_in_block.x; }
__fd__ grid_block_dimension_t  index_in_block()    { return threadIdx.x; }
__fd__ grid_block_dimension_t  index()             { return index_in_block(); }

__fd__ grid_block_dimension_t  position()          { return index_in_block(); }
__fd__ grid_block_dimension_t  position_in_block() { return index_in_block(); }

__fd__ bool                    is_first_in_block() { return index_in_block() == block::first_thread_position(); }
__fd__ bool                    is_last_in_block()  { return index_in_block() == block::last_thread_position(); }

__fd__ bool                    is_first_in_grid()  { return block::is_first_in_grid() and thread::is_first_in_block(); }
__fd__ bool                    is_last_in_grid()   { return block::is_last_in_grid() and thread::is_last_in_block();   }

using ::kat::grid_info::thread::is_first_in_warp;
using ::kat::grid_info::thread::is_last_in_warp;


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

namespace warp {

__fd__ grid_block_dimension_t index()          { return thread::index_in_block() >> log_warp_size; }
__fd__ unsigned               index_in_grid()  { return thread::index_in_grid() >> log_warp_size; }
__fd__ unsigned               global_index()   { return index_in_grid(); }

__fd__ unsigned index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::index_in_block() & lane_index_mask;
}

__fd__ unsigned index_in_block_of_first_lane() { return index_of_first_lane(); }

__fd__ unsigned global_index_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_index() & lane_index_mask;
}
__fd__ unsigned index_in_grid_of_first_lane()  { return warp::global_index_of_first_lane(); }
__fd__ unsigned int index_in_block()           { return warp::index(); }
__fd__ bool is_first_in_block()                { return warp::index_in_block() == block::index_of_first_warp(); }
__fd__ bool is_last_in_block()                 { return warp::index_in_block() == block::index_of_last_warp(); }
__fd__ bool is_first_in_grid()                 { return warp::is_first_in_block() and block::is_first_in_grid(); }
__fd__ bool is_last_in_grid()                  { return warp::is_last_in_block() and block::is_last_in_grid(); }

} // namespace warp


namespace lane = ::kat::grid_info::lane;

} // namespace grid_info

} // namespace linear_grid

} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_
