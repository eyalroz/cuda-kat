#pragma once
#ifndef CUDA_KAT_OSTREAM_PREFIX_GENERATORS_CUH_
#define CUDA_KAT_OSTREAM_PREFIX_GENERATORS_CUH_

#include <kat/on_device/streams/printfing_ostream.cuh>
#include <kat/on_device/grid_info.cuh>
#include <kat/on_device/math.cuh>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

namespace detail {
KAT_DEV unsigned num_digits_required_for(unsigned long long extremal_value)
{
	return ceilf(log10f(extremal_value));
}

} // namespace detail

namespace linear_grid {

namespace prefix_generators {

template <printfing_ostream::resolution IdentityResolution>
KAT_DEV void self_identify(kat::stringstream& ss);

// Prefix will look like (example for thread 34):
//
// "T 34 = (00,01,02) "
//
// ... since 34 is the third thread (index 2) in the second warp (index 1) in
// the first block.
//
template <>
KAT_DEV void self_identify<printfing_ostream::resolution::thread>(kat::stringstream& ss)
{
	namespace gi = ::kat::linear_grid::grid_info;

	const auto global_thread_id_width = detail::num_digits_required_for(gi::grid::num_threads() - 1);
	const auto block_id_width         = detail::num_digits_required_for(gi::grid::num_blocks() - 1);
	const auto warp_id_width          = detail::num_digits_required_for(gi::grid::num_warps_per_block() - 1);
	const auto lane_id_width          = 2; // ceilf(log10(warp_size - 1))
	constexpr const auto fill_char = '0';

	ss
		<< "T " << strf::right(gi::thread::global_id(), global_thread_id_width, fill_char)
		<< " = (" << strf::right(gi::block::id_in_grid(), block_id_width, fill_char )
		<< ',' << strf::right(gi::warp::id_in_block(), warp_id_width, fill_char)
		<< ',' << strf::right(gi::lane::id(), lane_id_width, fill_char)
		<< ") ";
}


// Prefix will look like (example for thread 1025 and block size 512):
//
// "W 32 = (02,00) "
//
// ... since thread 1025 overall is the second thread in the third block (block index 2), and thus in the first warp (warp index 0)
//
template <>
KAT_DEV void self_identify<printfing_ostream::resolution::warp>(kat::stringstream& ss)
{
	namespace gi = ::kat::linear_grid::grid_info;

	auto global_warp_id_width = detail::num_digits_required_for(gi::grid::num_warps() - 1);
	auto warp_id_width        = detail::num_digits_required_for(gi::grid::num_warps_per_block() - 1);
	auto block_id_width       = detail::num_digits_required_for(gi::grid::num_blocks() - 1);
	constexpr const auto fill_char = '0';
	ss
		<< "W " << strf::right(gi::warp::id_in_grid(), global_warp_id_width, fill_char)
		<< " = (" << strf::right(gi::block::id_in_grid(), block_id_width, fill_char)
		<< ',' << strf::right(gi::warp::id_in_block(), warp_id_width, fill_char)
		<< ") ";
}

// Prefix will look like (example for thread 1025 and block size 512):
//
// "B 2 "
//
// ... since thread 1025 is  in the 3rd block and block indices are 0-based
//
template <>
KAT_DEV void self_identify<printfing_ostream::resolution::block>(kat::stringstream& ss)
{
	namespace gi = ::kat::linear_grid::grid_info;

	const unsigned block_id_width = detail::num_digits_required_for(gi::grid::num_blocks() - 1);
	constexpr const auto fill_char = '0';
	ss << "B " << strf::right(gi::block::id_in_grid(), block_id_width, fill_char) << " : ";
}

template <>
KAT_DEV void self_identify<printfing_ostream::resolution::grid>(kat::stringstream& ss)
{
	ss << "G ";
}


} // namespace prefix_generators

namespace manipulators {

KAT_DEV printfing_ostream& identify( kat::printfing_ostream& os )
{
	using namespace kat::manipulators;
	prefix_generator_type gen;
	switch(os.printing_resolution()) {
	case printfing_ostream::resolution::thread : gen = prefix_generators::self_identify< printfing_ostream::resolution::thread >; break;
	case printfing_ostream::resolution::warp   : gen = prefix_generators::self_identify< printfing_ostream::resolution::warp   >; break;
	case printfing_ostream::resolution::block  : gen = prefix_generators::self_identify< printfing_ostream::resolution::block  >; break;
	case printfing_ostream::resolution::grid   : gen = prefix_generators::self_identify< printfing_ostream::resolution::grid   >; break;
	}
	return os.set_prefix_generator(gen);
}
} // namespace manipulators

} // namespace linear_grid

} // namespace kat

#endif // CUDA_KAT_OSTREAM_PREFIX_GENERATORS_CUH_
