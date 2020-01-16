/**
 * @file on_device/printfing_ostream.cuh
 *
 * @brief CUDA device-side functions for a C++-standard-library-like stream
 * whose output (eventually) gets printed using CUDA's device-side printf().
 *
 */
#pragma once
#ifndef CUDA_KAT_ON_PRINTFING_STREAM_CUH_
#define CUDA_KAT_ON_PRINTFING_STREAM_CUH_

#include <kat/on_device/streams/stringstream.hpp>
#include <kat/on_device/grid_info.cuh>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

namespace manipulators {

using prefix_generator_type = void (*)(kat::stringstream&);
	// TODO: Make it into a function with an std-string-like outputm, when we
	// have an std-string like class.

KAT_DEV auto prefix(prefix_generator_type gen);

}


class printfing_ostream
{
	static constexpr const std::size_t cout_initial_buffer_size { 1 << 8 };

public:
	enum class resolution { thread, warp, block, grid };

	KAT_DEV printfing_ostream(std::size_t initial_buffer_size = cout_initial_buffer_size) : main_buffer(initial_buffer_size) { }
	STRF_HD printfing_ostream(printfing_ostream&& other) : main_buffer(other.main_buffer) { }
	STRF_HD printfing_ostream(const printfing_ostream& other) : main_buffer(other.main_buffer) { }
    STRF_HD ~printfing_ostream();

    // Note: You can also use strf::flush if that exists
	KAT_DEV void flush()
	{
		if (not newline_on_flush and main_buffer.tellp() == 0) {
			// Note: Returning even though the string here might end up being
			// non-empty due to the prefix generation.
			return;
		}

		if (not should_act_for_resolution(printing_resolution)) {
				return;
		}
		if (use_prefix) {
			prefix_generator(prefix);
			printf(newline_on_flush ? "%*s%*s\n" : "%*s%*s",
				prefix.tellp(), prefix.c_str(),
				main_buffer.tellp(), main_buffer.c_str());
		}
		else {
			printf(newline_on_flush ? "%*s\n" : "%*s",
				main_buffer.tellp(), main_buffer.c_str()
			);
		}
		main_buffer.seekp(0);
	}

protected:
	static bool KAT_DEV should_act_for_resolution(resolution r) {
		// TODO: It might be a better idea to check which threads in the warp/block are still active
		// rather than assuming the first one is.
		switch(r) {
		case resolution::thread: return true;
		case resolution::warp:   return grid_info::thread::is_first_in_warp();
		case resolution::block:  return grid_info::thread::is_first_in_block();
		case resolution::grid:   return grid_info::thread::is_first_in_grid();
		default: return false; // but can't get here
		}
	}

public:
	template <typename T>
	KAT_DEV printfing_ostream& operator<<(const T& arg)
	{
		if (not should_act_for_resolution(printing_resolution)) { return *this; }
		strf::print_preview<false, false> no_preview;
		strf::make_printer<char>(strf::rank<5>(), strf::pack(), no_preview, arg).print_to(main_buffer);
		return *this;
	}

	// Manipulators are a clever, but confusing, idea from the C++ standard library's
	// IO streams: They're functions which manipulate streams, but can also be made
	// to manipulatethem by being sent to them using the << operator - which instead
	// of actually adding any data to the stream, invokes the manipulator function.
	//
	using manipulator = kat::printfing_ostream& ( kat::printfing_ostream& );

	KAT_DEV printfing_ostream& no_prefix()
	{
		use_prefix = false;
		prefix_generator = nullptr;
		return *this;
	}

	KAT_DEV printfing_ostream& set_prefix_generator(manipulators::prefix_generator_type gen)
	{
		use_prefix = true;
		prefix_generator = gen;
		return *this;
	}

	KAT_DEV printfing_ostream& no_newline_on_flush()
	{
		newline_on_flush = false;
		return *this;
	}

	KAT_DEV printfing_ostream& append_newline_on_flush()
	{
		newline_on_flush = true;
		return *this;
	}

	// Also clears the prefix - as that's assumed to have been resolution-related
	KAT_DEV printfing_ostream& set_resolution(resolution new_resolution)
	{
		main_buffer.clear();
		no_prefix();
		printing_resolution = new_resolution;
		return *this;
	}

protected:
	kat::stringstream main_buffer;
	kat::stringstream prefix { 100 }; // { 0 };
		// no prefix by default, so why bother allocating a buffer?
		// TODO: Make this an optional along with the prefix_generator

	bool flush_on_destruction { true };
	bool newline_on_flush { false };

	// We may want to prefix out printing with a string which the code using cout has not explicitly specified
	// beforehand. For example: An identifier of the current thread or warp.

	bool use_prefix { false };
		// TODO: Make this into a kat::optional<kat::stringstream> when we have an optional class,
		// and perhaps simply
	manipulators::prefix_generator_type prefix_generator { nullptr };


	// By default, all grid threads print; but we may want a printing only once per each warp, or block etc;
	// that resolution is controlled by this variable.

	resolution printing_resolution { resolution::thread };

};


namespace manipulators {
KAT_FD kat::printfing_ostream& flush( kat::printfing_ostream& os ) { os.flush(); return os; }
KAT_FD kat::printfing_ostream& endl( kat::printfing_ostream& os ) { os << '\n'; os.flush(); return os; }
KAT_FD kat::printfing_ostream& no_prefix( kat::printfing_ostream& os ) { return os.no_prefix(); }
KAT_FD kat::printfing_ostream& no_newline_on_flush( kat::printfing_ostream& os ) { return os.no_newline_on_flush(); }
KAT_FD kat::printfing_ostream& newline_on_flush( kat::printfing_ostream& os ) { return os.append_newline_on_flush(); }

} // manipulators


// This is defined only with __CUDA_ARCH__, since the implementation is actually device-only,
// referring to this->flush(), which can only really run on the device. We could, instead,
// make printfing_ostream::flush() be an STRF_HD (host-and-device) function, which simply
// fails on the host side, but that would be too much of a lie.
#ifdef __CUDA_ARCH__

STRF_HD printfing_ostream::~printfing_ostream()
{
	this->flush();
}

#endif

template <>
KAT_DEV printfing_ostream& printfing_ostream::operator<< <printfing_ostream::manipulator>(
	printfing_ostream::manipulator& manip)
{
	return manip(*this);
}

namespace manipulators {
KAT_DEV auto prefix(prefix_generator_type gen) {
	return [gen](kat::printfing_ostream& os) { return os.set_prefix_generator(gen); };
}
} // namespace manipulators

// This conditional compilation segment is necessary because NVCC (10.x) will not accept a
// reference-to/address-of a device function except in device function bodies, or when __CUDA_ARCH__
// is defined. In other words: we have a "device-side 'using' statement" here, followed by the
// operator<<() function which makes use of it.
#ifdef __CUDA_ARCH__
namespace manipulators {
using prefix_setting_manipulator_type = std::result_of< decltype(&prefix)(prefix_generator_type) >::type;
} // namespace manipulators

KAT_DEV printfing_ostream& operator<< (printfing_ostream& os, manipulators::prefix_setting_manipulator_type manip)
{
	// std::basic_ostream<manipulators::prefix_setting_manipulator_type> x;
	manip(os);
	return os;
}
#endif

namespace manipulators {
KAT_DEV auto resolution(printfing_ostream::resolution new_resolution) {
	return [new_resolution](kat::printfing_ostream& os) { return os.set_resolution(new_resolution); };
}
} // namespace manipulators

#ifdef __CUDA_ARCH__
namespace manipulators {
using resolution_setting_manipulator_type = std::result_of< decltype(&resolution)(printfing_ostream::resolution) >::type;
} // namespace manipulators

KAT_DEV printfing_ostream& operator<< (printfing_ostream& os, manipulators::resolution_setting_manipulator_type manip)
{
	manip(os);
	return os;
}
#endif


using manipulators::flush;

} // namespace kat

#endif // CUDA_KAT_ON_PRINTFING_STREAM_CUH_
