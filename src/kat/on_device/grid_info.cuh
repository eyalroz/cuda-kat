/**
 * @file grid_info.cuh
 *
 * @brief Information regarding the current kernel's launch grid and the calling
 * thread's positions within it.
 *
 * @note Currently, CUDA does not allows more than 2^31 threads in a launch grid, hence many functions
 * here return unsigned / dimension_t
 *
 * @note CUDA uses the term "index" for the 3-dimensional position within a block or a grid, and the
 * term "id" for the position in a linearization of that block or grid. This is somewhat confusing, as
 * the _switching_ of those terms would be more in line with their literal, dictionary definitions;
 * however - for consistency, the functions in this file maintain the same convention. So,
 *
 *    IF YOU WANT TO GET   --== A SINGLE NUMBER ==-- ,  USE  ::id()       METHODS<br>
 *    IF YOU WANT TO GET   --== 3-D COORDINATES ==-- ,  USE  ::index()    METHODS
 *
 * however, the above applies only for 3-D grids. When the grid is linear, we drop the distinction
 * between "index" and "id". Also, warp ID's are always linearized, since they don't respect the
 * multi-dimensional structure.
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

    constexpr KAT_FHD dimensions_t(uint3 v) noexcept : dimensions_t(v.x, v.y, v.z) { }
    constexpr KAT_FHD dimensions_t(dim3 dims) noexcept : dimensions_t(dims.x, dims.y, dims.z) { }

    constexpr KAT_FHD operator uint3(void) const noexcept { return { x, y, z }; }

    // This _should_ have been constexpr, but nVIDIA have not marked the dim3 constructors
    // as constexpr, so it isn't
    KAT_FHD operator dim3(void) const noexcept { return { x, y, z }; }

    constexpr KAT_FHD unsigned volume() const noexcept { return x * y * z; }
    	// TODO: Do we need this to be size_t?
    constexpr KAT_FHD bool empty() const noexcept { return  (x == 0) or (y == 0) or (z == 0); }

	/**
	 * @brief The number of actual dimensions (i.e. dimensions/axes with more than a single value)
     */
    constexpr KAT_FHD unsigned dimensionality() const noexcept
	{
        return empty() ? 0 : ((z > 1) + (y > 1) + (x > 1));
	}
};

template <unsigned Dimensionality = 3>
KAT_FD unsigned size(dimensions_t dims)
{
	switch (Dimensionality) {
	case 0: return 1;
	case 1: return dims.x;
	case 2: return dims.x * dims.y;
	case 3:
	default: return dims.volume();
	}
}

/**
 * A position within a 3-dimensional grid or block.
 *
 * @note all coordinates are non-negative - positions are taken from the "corner", not the center.
 */
using position_t = uint3;


constexpr KAT_FHD bool operator==(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return static_cast<uint3>(lhs) == static_cast<uint3>(rhs);
}

/**
 * A dimensions-conscious version of operator==
 */
template <unsigned Dimensionality = 3>
constexpr KAT_FHD bool equals(const uint3& lhs, const uint3& rhs) noexcept
{
	return
		((Dimensionality < 1) or (lhs.x == rhs.x)) and
		((Dimensionality < 2) or (lhs.y == rhs.y)) and
		((Dimensionality < 3) or (lhs.z == rhs.z));
}

template <unsigned Dimensionality = 3>
constexpr KAT_FHD bool equals(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return equals<Dimensionality>(static_cast<uint3>(lhs), static_cast<uint3>(rhs));
}

/**
 * A dimensions-conscious version of operator<
 */
template <unsigned Dimensionality = 3>
constexpr KAT_FHD bool less_than(const uint3& lhs, const uint3& rhs) noexcept
{
	return
		( (Dimensionality < 1) or (lhs.x < rhs.x) ) and
		( (Dimensionality < 2) or ((lhs.x == rhs.x) and (lhs.y < rhs.y)) ) and
		( (Dimensionality < 3) or ((lhs.x == rhs.x) and (lhs.y == rhs.y) and (lhs.z < rhs.z)) );
}

template <unsigned Dimensionality = 3>
constexpr KAT_FHD bool less_than(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return less_than<Dimensionality>(static_cast<uint3>(lhs), static_cast<uint3>(rhs));
}

namespace detail {

template <unsigned Dimensionality = 3, typename Size = unsigned>
KAT_FHD Size row_major_linearization(position_t position, dimensions_t dims)
{
	// If you're wondering why this doesn't use a switch statement - that's
	// due to an (apparent) NVCC bug, complaining about "unreachable statements"
	// which _are_ reachable for different template parameters.
	if (Dimensionality == 0) { return 0; }
	else if (Dimensionality == 1) { return position.x; }
	else if (Dimensionality == 2) { return position.x + position.y * dims.x; }
	else if (Dimensionality == 3) { return position.x + position.y * dims.x + position.z * dims.x * dims.y; }
	else return {};
}

} // namespace detail

/**
 * @brief Determines whether a dimensions specification follows CUDA's
 * convention of having non-trivial dimensions first.
 *
 * @param[in] dims A dimensions specification. Assumed to not be "empty",
 * i.e. assumed to have a value of at least 1 in every axis.
 *
 * @return true if no non-trivial dimensions follow trivial dimensions
 */
constexpr KAT_FHD bool dimensionality_is_canonical(dimensions_t dims)
{
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

namespace detail {

template <unsigned Dimensionality = 3>
KAT_FD position_t   last_position_for(dimensions_t dims)
{
	return {
		(Dimensionality < 1) ? 0 : dims.x - 1,
		(Dimensionality < 2) ? 0 : dims.y - 1,
		(Dimensionality < 3) ? 0 : dims.z - 1
	};
}

template <unsigned Dimensionality = 3>
KAT_FD position_t   first_position() { return { 0, 0, 0 }; }

} // namespace detail

namespace grid {

/**
 * @note These are the dimensions of the grid over blocks; the blocks may have additional "dimensions" relative to threads.
 */
KAT_FD dimensions_t dimensions_in_blocks()  { return gridDim; }
template <unsigned Dimensionality = 3>
KAT_FD unsigned     num_blocks()            { return size(dimensions_in_blocks()); }
KAT_FD position_t   first_block_position()  { return dimensions_t{0, 0, 0}; }

template <unsigned Dimensionality = 3>
KAT_FD position_t   last_block_position()   { return detail::last_position_for(gridDim); }


/**
 * @note These are the dimensions of the grid in terms of threads. This means that a grid can have less blocks (or
 * even one block) in each dimension, but each block many have multiple threads, contributing to the overall dimension.
 */
template <unsigned Dimensionality = 3>
KAT_FD dimensions_t dimensions_in_threads()
{
	switch (Dimensionality) {
	case 0: return { 1, 1, 1 };
	case 1: return { gridDim.x * blockDim.x, 1, 1 };
	case 2: return { gridDim.x * blockDim.x, gridDim.y * blockDim.y, 1 };
	case 3:
	default: return { gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z };
	}
}

} // namespace grid

namespace block {

KAT_FD dimensions_t dimensions()            { return blockDim; }
KAT_FD position_t   position_in_grid()      { return blockIdx; }
/**
 * @note Remember a thread's index is a multi-dimensional entity, not a single linear value
 */
KAT_FD position_t   index()                 { return position_in_grid(); }
template <unsigned Dimensionality = 3>
KAT_FD bool         is_first_in_grid()
{
	return equals<Dimensionality>(block::position_in_grid(), grid::first_block_position() );
};

template <unsigned Dimensionality = 3>
KAT_FD bool         is_last_in_grid()
{
	return equals<Dimensionality>(block::position_in_grid(), grid::last_block_position() );
};

/**
 * @brief Produces the linearization of a block's index in the grid.
 *
 * @note Remember a thread's index is a multi-dimensional entity, not a single linear value. The id is
 * the linearization of the index
 */
template <unsigned Dimensionality = 3>
KAT_FD grid_dimension_t id_in_grid()
{
	return kat::detail::row_major_linearization<Dimensionality>(
		position_in_grid(), grid::dimensions_in_blocks());
}

template <unsigned Dimensionality = 3>
KAT_FD grid_dimension_t id()                { return id_in_grid(); }

template <unsigned Dimensionality = 3>
KAT_FD grid_block_dimension_t size()        { return size(dimensions()); }

KAT_FD position_t   first_thread_position() { return position_t{0, 0, 0}; }
template <unsigned Dimensionality = 3>
KAT_FD position_t   last_thread_position()  { return grid_info::detail::last_position_for(blockDim); }

template <unsigned Dimensionality = 3>
KAT_FD grid_block_dimension_t
                    num_full_warps()        { return block::size<Dimensionality>() / warp_size; }
template <unsigned Dimensionality = 3>
KAT_FD grid_block_dimension_t num_warps()
{
	return (block::size<Dimensionality>() + warp_size - 1) >> log_warp_size;
		// While this form of rounded-up-division may generally overflow, that's not possible
		// here, since CUDA block size is capped at 1024 as of 2019, and is unlikely to get close
		// to the maximum integer value.
}

KAT_FD grid_block_dimension_t
                    id_of_first_warp()      { return 0; }
KAT_FD position_t   index_of_first_warp()   { return {0, 0, 0}; }
KAT_FD grid_block_dimension_t
                    id_of_last_warp()       { return num_warps() - 1; }
/**
 * @note assumes linear kernels only use the x dimension - which is a reasonable assumptions,
 * since the y and z dimensions are limited in extent by CUDA.
 */
KAT_FD bool         is_linear()             { return block::dimensions().y == 1 and block::dimensions().z == 1; }

} // namespace block

namespace thread_block = block;

namespace grid {

/**
 * Determines whether the grid's non-trivial dimensions - in blocks and in threads - are on the x axis only.
 *
 * @note One could consider y-only or z-only dimensions as linear; this definition was chosen for convenience
 * (and performance) and is used throughout this library
 */
KAT_FD bool         is_linear()
{
	return gridDim.y == 1 and gridDim.z == 1 and grid_info::block::is_linear();
}

// TODO: Consider templatizing this on the dimensions too
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned     num_warps()             { return num_blocks<OuterDimensionality>() * block::num_warps<InnerDimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned     num_threads()           { return num_blocks<OuterDimensionality>() * block::size<InnerDimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned     total_size()            { return num_threads<OuterDimensionality, InnerDimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned     size_in_threads()       { return num_threads<OuterDimensionality, InnerDimensionality>(); }
template <unsigned Dimensionality = 3>
KAT_FD unsigned     num_warps_per_block()   { return block::num_warps<Dimensionality>(); }

} // namespace grid

namespace warp {

enum : unsigned { first_lane = 0, last_lane = warp_size - 1 };

KAT_FD unsigned     size()   { return warp_size; }
KAT_FD unsigned     length() { return warp_size; }

} // namespace warp

namespace thread {

// TODO: Should we avoid reading alll of threadIdx and only take some of its fields?
// The compiler might not optimize the read away.
KAT_FD position_t   position_in_block()    { return threadIdx;  }
KAT_FD position_t   position()             { return position_in_block();  }
KAT_FD position_t   index_in_block()       { return position(); }
KAT_FD position_t   index()                { return position(); }

template <unsigned Dimensionality = 3>
KAT_FD bool         is_first_in_block()    { return equals<Dimensionality>(position(), block::first_thread_position()); }

template <unsigned Dimensionality = 3>
KAT_FD bool         is_last_in_block()     { return equals<Dimensionality>(position(), block::last_thread_position<Dimensionality>()); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD bool         is_first_in_grid()     { return block::is_first_in_grid<OuterDimensionality>() and thread::is_first_in_block<InnerDimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD bool         is_last_in_grid()      { return block::is_last_in_grid<OuterDimensionality>() and thread::is_last_in_block<InnerDimensionality>();   }


/**
 * @brief Linearizes of a thread's position within its block.
 *
 * @param thread_position_in_block the Dimensionality-dimensional version of the thread index within its grid block,
 * represented in the 3-dimensional dimensions_t structure
 * @return The 1-d index of the specified thread within its block, when it's
 * flattened so that threads with identical z and y axis coordinates form a contiguous
 * sequence
 */
template <unsigned Dimensionality = 3>
KAT_FD unsigned    id_in_block(position_t thread_position_in_block)
{
	return kat::detail::row_major_linearization<Dimensionality, unsigned>(thread_position_in_block, block::dimensions());
}
template <unsigned Dimensionality = 3>
KAT_FD unsigned    id_in_block()           { return id_in_block<Dimensionality>(thread::position_in_block()); }
template <unsigned Dimensionality = 3>
KAT_FD unsigned    id()                    { return id_in_block<Dimensionality>(); }

template <unsigned Dimensionality = 3>
KAT_FD position_t  position_in_grid(position_t block_position_in_grid, position_t thread_position_in_block)
{
	return {
		(Dimensionality < 1) ? 0 : (block_position_in_grid.x * blockDim.x + thread_position_in_block.x),
		(Dimensionality < 2) ? 0 : (block_position_in_grid.y * blockDim.y + thread_position_in_block.y),
		(Dimensionality < 3) ? 0 : (block_position_in_grid.z * blockDim.z + thread_position_in_block.z),
	};
}

template <unsigned OuterDimensionality = 3>
KAT_FD unsigned id_in_grid(unsigned block_id_in_grid, unsigned thread_id_in_block)
{
	return thread_id_in_block + block::size<OuterDimensionality>() * block_id_in_grid;
}

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned id_in_grid(position_t block_position_in_grid, position_t thread_position_in_block)
{
	return thread::id_in_grid<OuterDimensionality>(
		thread::id_in_block<InnerDimensionality>(thread_position_in_block),
		block::id_in_grid<OuterDimensionality>(thread_position_in_block)
		);
}

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned id_in_grid()
{
	return thread::id_in_grid<OuterDimensionality>(
		block::id_in_grid<OuterDimensionality>(),
		thread::id_in_block<InnerDimensionality>());
}

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD position_t  position_in_grid()
{
	constexpr const unsigned overall_dimensionality =
		(OuterDimensionality < InnerDimensionality) ? InnerDimensionality : OuterDimensionality;
	return thread::position_in_grid<overall_dimensionality>(
		block::position_in_grid(), thread::position_in_block());
}

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD position_t  index_in_grid()         { return position_in_grid<OuterDimensionality, InnerDimensionality>(); }

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned    global_id()             { return id_in_grid<OuterDimensionality, InnerDimensionality>(); }

template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD position_t  global_index()          { return position_in_grid<OuterDimensionality, InnerDimensionality>(); }


} // namespace thread

namespace warp {

template <unsigned Dimensionality = 3>
KAT_FD unsigned    id_in_block()           { return grid_info::thread::id_in_block<Dimensionality>() / warp_size; }
template <unsigned Dimensionality = 3>
KAT_FD unsigned    index_in_block()        { return id_in_block<Dimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned    id_in_grid()            { return grid_info::thread::id_in_grid<OuterDimensionality, InnerDimensionality>() / warp_size; }
template <unsigned Dimensionality>
KAT_FD unsigned    index()                 { return index_in_block<Dimensionality>(); }
template <unsigned OuterDimensionality = 3, unsigned InnerDimensionality = 3>
KAT_FD unsigned    global_id()             { return id_in_grid<OuterDimensionality, InnerDimensionality>(); }
template <unsigned Dimensionality = 3>
KAT_FD unsigned    id_of_first_lane()   {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::id_in_block<Dimensionality>() & lane_index_mask;
}

template <unsigned Dimensionality = 3>
KAT_FD unsigned    index_in_block_of_first_lane()
                                           { return id_of_first_lane<Dimensionality>(); }

template <unsigned Dimensionality = 3>
KAT_FD unsigned    global_id_of_first_lane() {
	constexpr const auto lane_index_mask = warp_size - 1;
	return thread::global_id<Dimensionality>() & lane_index_mask;
}

template <unsigned Dimensionality = 3>
KAT_FD unsigned    index_in_grid_of_first_lane()
{
	return warp::global_id_of_first_lane<Dimensionality>();
}

template <unsigned Dimensionality = 3>
KAT_FD unsigned    id()                    { return id_in_block<Dimensionality>();                             }

template <unsigned Dimensionality = 3>
KAT_FD bool        is_first_in_block()
{
	return warp::id_in_block<Dimensionality>() == block::id_of_first_warp();
}
template <unsigned Dimensionality = 3>
KAT_FD bool        is_last_in_block()
{
	return warp::id_in_block() == block::id_of_last_warp();
}
template <unsigned Dimensionality = 3>
KAT_FD bool        is_first_in_grid()
{
	return warp::is_first_in_block() and block::is_first_in_grid();
}
template <unsigned Dimensionality = 3>
KAT_FD bool        is_last_in_grid()
{
	return warp::is_last_in_block() and block::is_last_in_grid();
}


} // namespace warp

namespace lane {

enum { half_warp_size = warp_size / 2 };


template <unsigned Dimensionality = 3>
KAT_FD unsigned id(position_t thread_position)
{
	// we could use a special register:
	//
	//   return builtins::lane_index();
	//
	// but apparently, retrieving a special register takes a good
	// number of clock cycles (why?!), so in practice, this might be
	// faster:
	constexpr const auto lane_id_mask = warp_size - 1;
	return thread::id_in_block<Dimensionality>(thread_position) & lane_id_mask;
	// ... but it's less obvious than the linear grid case, where
	// no linearization is required.
}

template <unsigned Dimensionality = 3>
KAT_FD unsigned id_in_warp()        { return id<Dimensionality>(threadIdx);                    }
template <unsigned Dimensionality = 3>
KAT_FD unsigned id()                { return id_in_warp<Dimensionality>();                     }
template <unsigned Dimensionality = 3>
KAT_FD unsigned index()             { return id<Dimensionality>();                             }
template <unsigned Dimensionality = 3>
KAT_FD unsigned index_in_warp()     { return id<Dimensionality>();                             }
template <unsigned Dimensionality = 3>
KAT_FD unsigned is_first()          { return id_in_warp<Dimensionality>() == warp::first_lane; }
template <unsigned Dimensionality = 3>
KAT_FD unsigned is_last()           { return id_in_warp<Dimensionality>() == warp::last_lane;  }

} // namespace lane

namespace thread {

template <unsigned Dimensionality = 3>
KAT_FD bool     is_first_in_warp()  { return lane::id<Dimensionality>() == warp::first_lane; }
template <unsigned Dimensionality = 3>
KAT_FD bool     is_last_in_warp()   { return lane::id<Dimensionality>() == warp::last_lane;  }

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

KAT_FD grid_dimension_t  num_blocks()              { return gridDim.x; }
KAT_FD grid_dimension_t  dimensions_in_blocks()    { return num_blocks(); }
KAT_FD grid_dimension_t  index_of_first_block()    { return 0; }
KAT_FD grid_dimension_t  index_of_last_block()     { return num_blocks() - 1; }
KAT_FD grid_dimension_t  first_block_position()    { return index_of_first_block(); }
KAT_FD grid_dimension_t  first_last_position()     { return index_of_last_block(); }

} // namespace grid

namespace block {

using kat::grid_info::block::dimensions;
KAT_FD unsigned                index_in_grid()           { return blockIdx.x; }
KAT_FD grid_block_dimension_t  index()                   { return index_in_grid(); }
KAT_FD unsigned                id_in_grid()              { return index_in_grid(); }
KAT_FD grid_block_dimension_t  id()                      { return id_in_grid(); }
KAT_FD grid_block_dimension_t  position_in_grid()        { return index_in_grid(); }
KAT_FD bool                    is_first_in_grid()        { return block::id_in_grid() == grid::index_of_first_block(); }
KAT_FD bool                    is_last_in_grid()         { return id() == grid::index_of_last_block(); }
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

KAT_FD grid_block_dimension_t  index_in_block(uint3 position_in_block)
                                                   { return position_in_block.x; }
KAT_FD grid_block_dimension_t  index_in_block()    { return index_in_block(threadIdx); }
KAT_FD grid_block_dimension_t  id_in_block(uint3 position_in_block)
                                                   { return index_in_block(position_in_block); }
KAT_FD grid_block_dimension_t  index()             { return index_in_block(); }
KAT_FD grid_block_dimension_t  id_in_block()       { return index_in_block(); }
KAT_FD grid_block_dimension_t  id()                { return id_in_block(); }

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

KAT_FD unsigned id_in_grid(grid_dimension_t block_index, grid_dimension_t thread_index)
{
	return index_in_grid(block_index, thread_index);
}

KAT_FD unsigned index_in_grid()     { return index_in_grid(block::index(), index()); }
KAT_FD unsigned id_in_grid()        { return index_in_grid(); }
KAT_FD unsigned global_index()      { return index_in_grid(); }
KAT_FD unsigned global_id()         { return index_in_grid(); }


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

KAT_FD grid_block_dimension_t index_in_block() { return thread::index_in_block() >> log_warp_size; }
KAT_FD grid_block_dimension_t index()          { return index_in_block(); }
KAT_FD grid_block_dimension_t id_in_block()    { return index_in_block(); }
KAT_FD grid_block_dimension_t id()             { return id_in_block(); }
KAT_FD unsigned               index_in_grid()  { return thread::index_in_grid() >> log_warp_size; }
KAT_FD unsigned               id_in_grid()     { return index_in_grid(); }
KAT_FD unsigned               global_index()   { return index_in_grid(); }
KAT_FD unsigned               global_id()      { return id_in_grid(); }

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
KAT_FD bool is_first_in_block()                { return warp::index_in_block() == block::index_of_first_warp(); }
KAT_FD bool is_last_in_block()                 { return warp::index_in_block() == block::index_of_last_warp(); }
KAT_FD bool is_first_in_grid()                 { return warp::is_first_in_block() and block::is_first_in_grid(); }
KAT_FD bool is_last_in_grid()                  { return warp::is_last_in_block() and block::is_last_in_grid(); }

} // namespace warp


namespace lane {

// Note: Warps are strictly one-dimensional entities,
// so within a warp, a lane's ID and its index are one and the
// same thing. However... because we use thread indices to
// obtain the lane index rather than the special register for warps,
// directly - we have to separate the code for the linear-grid and
// non-linear-grid cases.

enum { half_warp_size = kat::grid_info::lane::half_warp_size };

KAT_FD unsigned id(unsigned thread_index)
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

KAT_FD unsigned id_in_warp()              { return id(threadIdx.x); }
KAT_FD unsigned id()                      { return id_in_warp(); }
KAT_FD unsigned index()                   { return id(); }
KAT_FD unsigned index_in_warp()           { return id(); }
KAT_FD unsigned is_first()                { return id_in_warp() == warp::first_lane; }
KAT_FD unsigned is_last()                 { return id_in_warp() == warp::last_lane; }


KAT_FHD unsigned id_in_half_warp(unsigned thread_or_lane_index)
{
	enum { half_warp_index_mask = half_warp_size - 1 };
	return thread_or_lane_index & half_warp_index_mask;
}

KAT_FD unsigned id_in_half_warp()         { return id_in_half_warp(threadIdx.x); }
KAT_FD unsigned index_in_half_warp()      { return id_in_half_warp(threadIdx.x); }
KAT_FD unsigned is_in_first_half_warp()   { return id_in_warp() < half_warp_size; }
KAT_FD unsigned is_in_second_half_warp()  { return id_in_warp() >= half_warp_size; }

} // namespace lane

} // namespace grid_info

} // namespace linear_grid

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_GRID_INFO_CUH_
