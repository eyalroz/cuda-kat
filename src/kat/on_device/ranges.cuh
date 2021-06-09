/**
 * @file on_device/ranges.cuh
 *
 * @brief Ranges for warps, blocks and grids to collaboratively iterate ranges of indices/integers.
 *
 */
#ifndef CUDA_KAT_ON_DEVICE_RANGES_CUH_
#define CUDA_KAT_ON_DEVICE_RANGES_CUH_

#include <kat/on_device/grid_info.cuh>
#include <kat/ranges.hpp>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace ranges {

/**
 * @brief Same as @ref kat::linear_grid::ranges::warp_stride, but for use in
 * non-linear grids.
 *
 * @note warp-striding is relevant for non-linear-grid cases, since warps are
 * in themselves linear even if the block/grid overall aren't.
 */
template <typename Size>
KAT_DEV kat::ranges::strided<Size> warp_stride(Size length)
{
	constexpr const auto stride = warp_size;
	auto begin = lane::id();
	return ::kat::ranges::strided<promoted_size_t<Size>>(begin, length, stride);
}

} // namespace ranges


namespace linear_grid {
namespace ranges {

/**
 * @brief A named constructor idiom for iteration ranges - to be used in ranged-for
 * loops by a warp to collaboratively iterate a sequence of indices.
 *
 * @tparam Size A type large enough to represent @p length
 * @param length The number of indices to iterate
 * @return the constructed range
 *
 * @note See @see collaborative::linear_grid::warp::at_warp_stride . Specifically,
 *
 *    for(auto pos : kat::ranges::warp_stride(my_length) {
 *        foo(pos);
 *    }
 *
 * will have each warp run `foo()` in lockstep for indices 0..31, then 32..64 etc.
 */
template <typename Size>
KAT_DEV kat::ranges::strided<Size> warp_stride(Size length)
{
	constexpr const auto stride = warp_size;
	auto begin = lane::id();
	return ::kat::ranges::strided<promoted_size_t<Size>>(begin, length, stride);
}


/**
 * @brief A named constructor idiom for ranged for loops which can be
 * used to have complete blocks iterate a sequence of indices.
 *
 * @tparam Size A type large enough to represent @p length
 * @param length The number of indices to iterate
 * @return the constructed range
 *
 * @note Usage: See @see collaborative::linear_grid::block::at_block_stride .
 */
template <typename Size>
KAT_DEV kat::ranges::strided<Size> block_stride(Size length)
{
	const auto stride = linear_grid::block::size();
	const auto begin = thread::id_in_block();
	return ::kat::ranges::strided<promoted_size_t<Size>>(begin, length, stride);
}

/**
 * @brief A named constructor idiom for ranged for loops which can be
 * used to have entire grids iterate a sequence of indices.
 *
 * @tparam Size A type large enough to represent @p length
 * @param length The number of indices to iterate
 * @return the constructed range
 *
 * @note Usage: See @see collaborative::linear_grid::grid::at_grid_stride .
 */
template <typename Size>
KAT_DEV kat::ranges::strided<Size> grid_stride(Size length)
{
	const auto stride = grid::total_size();
	const auto begin = thread::global_id();
	return ::kat::ranges::strided<promoted_size_t<Size>>(begin, length, stride);
}

namespace warp_per_input_element {

template <typename Size>
KAT_DEV kat::ranges::strided<Size> grid_stride(Size length)
{
	const auto stride = grid::num_warps();
	const auto begin = warp::global_id();
	return ::kat::ranges::strided<promoted_size_t<Size>>(begin, length, stride);
}

} // namespace warp_per_input_element

} // namespace ranges
} // namespace linear_grid
} // namespace kat


#endif /* CUDA_KAT_ON_DEVICE_RANGES_CUH_ */
