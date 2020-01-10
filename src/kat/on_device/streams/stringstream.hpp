/**
 * @file on_device/string_stream.cuh
 *
 * @brief A string stream class for CUDA device-side code (usable by individual threads).
 *
 * @note This class will likely be rather slow in use: Its code is entirely serial, and it
 * uses occasional dynamic memory allocations. You are advised to use it mostly for debugging
 * purposes.
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_STRINGSTREAM_CUH_
#define CUDA_KAT_ON_DEVICE_STRINGSTREAM_CUH_

#include <kat/on_device/builtins.cuh>

#include <strf.hpp>

// Necessary for printf()'ing in kernel code
#include <cstdio>
// #include <cstdarg>

///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {

namespace detail {

template <typename T>
__device__ T* safe_malloc(std::size_t size)
{
	auto p = malloc(size);
	if (p == nullptr) {
		asm("trap;");
	}
	return static_cast<T*>(p);
}

}

/**
 * An std::stringstream-like class into which one can add formatted
 * data using `my_stringstream << my_datum`. It won't accept std::ios
 * flags - since we can't depend on host-side-only IOS code - but it
 * will accept a bunch of strf equivalents. See:
 *
 * @url https://robhz786.github.io/strf/doc/quick_reference.html#format_functions
 *
 * for a list of format functions.
 *
 * @note This class owns its buffer.
 * @note nothing is dynamically allocated if the length is 0
 */
class stringstream: public ::strf::basic_outbuf<char>
{
public:
	using char_type = char;
	// no traits_type - this is not implemented by strf
	// using int_type = int; // really
	// using off_type = std::off_t; // really?
	using pos_type = std::size_t;

protected:
	STRF_HD stringstream(char_type* initial_buffer, std::size_t initial_buffer_size) :
		buffer_size(initial_buffer_size),
		buffer(initial_buffer),
		strf::basic_outbuf<char_type>(initial_buffer, buffer_size)
	{
	}

public:
	STRF_HD stringstream(std::size_t initial_buffer_size);

	STRF_HD stringstream(stringstream&& other) : strf::basic_outbuf<char_type>(other.buffer, other.buffer_size)
    {
		if (buffer != nullptr) {
			free(buffer);
		}
		buffer = other.buffer;
		buffer_size = other.buffer_size;
	    other.buffer = nullptr;
	    other.buffer_size = 0;
	}

	STRF_HD stringstream(const stringstream& other)	: stringstream(other.buffer_size)
	{
		memcpy(buffer, other.buffer, sizeof(char_type) * (buffer_size + 1));
	}

	STRF_HD ~stringstream()
	{
		if (buffer != nullptr) {
			free(buffer);
		}
	}

	STRF_HD void recycle() override;

	__device__ void clear()
	{
		if (buffer != nullptr) {
			*pos() = '\0';
		}
	}

	__device__ void flush() { clear(); }

	// should be able to produce an std-string-like proxy supporting a c_str() method, rather
	// than providing a c_str() directly.

	__device__ const char* c_str()
	{
		flush();
		return buffer;
	}

	__device__ pos_type tellp() const { return pos() - buffer; }
	__device__ bool empty() const { return tellp() == 0; }
		// std::stringstream's don't have this
	__device__ stringstream& seekp(pos_type pos) { set_pos(buffer + pos); return *this; }

	__device__ std::size_t capacity() const { return buffer_size; } // perhaps there's something else we can use instead?

	// TO implement (maybe):
	//
	// seekp
	// tellp
	// put
	// write
	// swap <- NO.
	//
	// good
	// eof
	// fail
	// bad
	// operator!
	// operator bool
	// rdstate
	// setstate
	// copyfmt
	// fill
	// exceptions <- No exception support; but might still implement this
	// imbue <- No locale support
	// tie
	// narrow <- No locale support
	// widen <- No locale support
	//
	// flags
	// setf
	// unsetf
	// precision
	// width
	// imbue <- No locale support
	// getloc <- No locale support
	// xalloc, iword, pword <- Not relevant on the device side, I think.
	// register_callback <- ??
	// sync_with_stdio <- No
	//



protected:
	// TODO: Write and use a device-side unique_ptr class and use kat::unique_ptr<char_type>
	// instead of these two variables
	std::size_t buffer_size; // not including space for a trailing '\0'.
	char_type* buffer;
};

#ifdef __CUDA_ARCH__

STRF_HD stringstream::stringstream(std::size_t initial_buffer_size)
    : stringstream(
    	initial_buffer_size == 0 ? nullptr : detail::safe_malloc<char_type>(initial_buffer_size + 1),
        initial_buffer_size)
{
}

__device__ void stringstream::recycle()
{
	// printf("recycle()! ... from size buffer_size =  %llu" , buffer_size);
	std::size_t used_size = (buffer_size == 0) ? 0 : (this->pos() - buffer);
	// a postcondition of recycle() is that at least so much free space is available.
	auto new_buffer_size = builtins::maximum(
		buffer_size * 2,
		used_size + strf::min_size_after_recycle<char_type>());
	auto new_buff = detail::safe_malloc<char_type>(new_buffer_size + 1);
	if (buffer != nullptr) {
		memcpy(new_buff, buffer, sizeof(char_type) * used_size);
		free(buffer);
	}
	this->set_pos(new_buff + used_size);
	this->set_end(new_buff + new_buffer_size);
	buffer = new_buff;
}
#endif


template <typename T>
__device__  stringstream& operator<<(stringstream& out, const T& arg)
{
	if (out.capacity() == 0) {
		// We should not need to do the following. However, for some reason, make_printer(...).print_to(out)
		// will fail on empty (nullptr) buffers; so we might end up "recycle()ing" more than once for the sam
		// streaming operation.
		out.recycle();
	}

	// TODO:
	// 1. Can `no_preview` be made constant?
	// 2. Can't we target a specific overload rather than play with ranks?
	auto no_preview = ::strf::print_preview<false, false>{};
	::strf::make_printer<char>(
		::strf::rank<5>(),
			// strf::rank is a method for controlling matching within the overload set:
			// rank objects have no members, it's only about their type. Higher rank objects can
			// match lower-rank objects (i.e. match functions in the overload sets expecting lower-rank
			// objects), which means they have access to more of the overload sets. If we create
			// a lower-rank object here we will only be able to match a few overload set members.
		::strf::pack(),
			// not modifying any facets such as digit grouping or digit separator
		no_preview,
			// Don't know what this means actually
		arg
		).print_to(out);

	// Note: This function doesn't actually rely on out being a stringstream; any
	// ostream-like class would do. But for now, we don't have any ostreams other
	// than the stringstream, so we'll leave it this way. Later, with could either
	// have an intermediate class, or wrap basic_outbuf with an ostream class
	// without a buffer, or just call basic_outbuf an ostream

	return out;
}

} // namespace kat

///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_STRINGSTREAM_CUH_

