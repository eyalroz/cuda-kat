/**
 * @file kat/on_device/miscellany.cuh
 *
 * @brief Miscellaneous functions provided by cuda-kat which are not a good
 * fit in any other header.
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
#define CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_

#include "common.cuh"
#include <kat/detail/pointers.cuh>

#include <type_traits>
#include <limits>
#include <cassert>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

namespace detail {

template <bool Signed, std::size_t NumBits> struct integer_type_struct;
template <> struct integer_type_struct<false,  8> { using type = std::uint8_t;  };
template <> struct integer_type_struct<false, 16> { using type = std::uint16_t; };
template <> struct integer_type_struct<false, 32> { using type = std::uint32_t; };
template <> struct integer_type_struct<false, 64> { using type = std::uint64_t; };
template <> struct integer_type_struct<true,   8> { using type = std::int8_t;   };
template <> struct integer_type_struct<true,  16> { using type = std::int16_t;  };
template <> struct integer_type_struct<true,  32> { using type = std::int32_t;  };
template <> struct integer_type_struct<true,  64> { using type = std::int64_t;  };

// TODO: Consider pushing these types upwards into kat:: proper.

/**
 * A templating by size of the signed integer types
 */
template <std::size_t NumBits>
using int_t = typename detail::integer_type_struct<true, NumBits>::type;

/**
 * A templating by size of the unsigned integer types
 */
template <std::size_t NumBits>
using uint_t = typename detail::integer_type_struct<false, NumBits>::type;


/**
 * @note Assumes num_elements_to_copy > 0 and the same misalignment of the source
 * and destination w.r.t. native words.
 */
KAT_FD void copy(
	uint32_t*        __restrict__  destination,
	const uint32_t*  __restrict__  source,
	std::size_t                    num_elements_to_copy)
{
	while (num_elements_to_copy-- > 0) {
		*(destination++) = *(source++);
	}
}

/**
 * @note Assumes num_elements_to_copy > 0 and the same misalignment of the source
 * and destination w.r.t. native words.
 */
KAT_FD void copy(
	uint16_t*        __restrict__  destination,
	const uint16_t*  __restrict__  source,
	std::size_t                    num_elements_to_copy)
{
	bool got_non_word_head = not is_aligned<native_word_t>(destination);
	if (got_non_word_head) {
		*(destination++) = *(source++);
		num_elements_to_copy--;
	}
	auto num_words_to_copy =
		num_elements_to_copy / ((sizeof(native_word_t) / sizeof(uint16_t)));
		// ... so, half as many words as elements;
	detail::copy(
		reinterpret_cast<native_word_t*>(destination),
		reinterpret_cast<const native_word_t*>(source),
		num_words_to_copy
	);
	bool got_non_word_tail = not is_aligned<native_word_t>(destination + num_elements_to_copy);
	if (got_non_word_tail) {
		destination[num_elements_to_copy - 1] = source[num_elements_to_copy - 1];
	}
}

/**
 * @note Assumes num_elements_to_copy > 0 and the same misalignment of the source
 * and destination w.r.t. native words.
 */
KAT_FD void copy(
	uint8_t*        __restrict__  destination,
	const uint8_t*  __restrict__  source,
	std::size_t                   num_elements_to_copy)
{
	// TODO: Improve this implementation to use native-word copies as much as possible, just like the 2-byte case
	if (num_elements_to_copy > 0) {
		::memcpy(destination, source, num_elements_to_copy * sizeof(uint8_t));
	}
}

} // namespace detail

/**
 * Copies some data from one location to another - using the native register
 * size for individual elements on CUDA GPUs, i.e. sizeof(int) = 4
 *
 * @note CUDA's own general-purpose memcpy() takes void pointers and uses a u8 (byte)
 * LD-ST loop. See: @url https://godbolt.org/z/9ChTPM ; this LD-ST's using the native
 * register size, 4 bytes, if possible.
 *
 * @note this function assumes appropriate alignment.
 *
 * @note Instead of using this function, you're probably better off using a warp-level
 * or block-level primitive for copying data.
 *
 * @param destination Destination of the copy. Must have at least
 * 4 (@p num_elements_to_copy} bytes allocated. Data must be self-aligned, i.e. the
 * numeric value of this parameter must be divisible by sizeof(T).
 * @param source The beginning of the memory region from which to copy.
 * There must be sizeof(T) * {@p num_elements_to_copy} bytes readable starting with
 * this address. Data must be self-aligned, i.e. the numeric value of this parameter
 * must be divisible by sizeof(T).
 * @param num_elements_to_copy the number of elements of data to copy - not their
 * total size in bytes!
 * @return the destination pointer
 */
template <typename T, bool AssumeSameAlignmentWithinWord = false>
KAT_FD T* copy(
	T*        __restrict__  destination,
	const T*  __restrict__  source,
	std::size_t             num_elements_to_copy)
{
	// This function uses the native word size explicitly in a few places, so:
	static_assert(sizeof(native_word_t) == sizeof(uint32_t), "unexpected size of native word");

	if (not std::is_trivially_copyable<T>::value) {
		// Can't optimize, must use T::operator=.
		for(std::size_t i = 0; i < num_elements_to_copy; i++) {
			destination[i] = source[i];
		}
		return destination;
	}

	if (not AssumeSameAlignmentWithinWord) {
		auto source_misalignent_in_bytes = detail::misalignment_extent<native_word_t>(source);
		auto destination_misalignent_in_bytes = detail::misalignment_extent<native_word_t>(destination);
		if (source_misalignent_in_bytes != destination_misalignent_in_bytes) {
			// Since the alignments don't match, any read-and-write operation pair
			// will be unaligned - unless we work on individual bytes.

			if (num_elements_to_copy > 0) {
				::memcpy(destination, source, num_elements_to_copy * sizeof(T));
			}
			return destination;
			// ... but actually the above claim is not true, for the case of 2-byte size_mod;
			// if the alignments are, say, 0 and 2 or 3 and 1 then we can at least use a
			// loop over 2-byte copying. TODO: Implement that.
		}
	}

	if (num_elements_to_copy == 0) {
		return destination;
	}

	constexpr const auto size_mod_in_bytes { sizeof(T) % sizeof(native_word_t) };
	constexpr const auto size_gcd_of_T_and_native_word {
		size_mod_in_bytes  == 4 ? 0 : (size_mod_in_bytes == 2 ? 2 : 1)
	};
	using copy_unit_type = detail::uint_t<size_gcd_of_T_and_native_word * CHAR_BIT>;
	auto num_copy_unit_elements_to_copy = num_elements_to_copy * sizeof(T) / sizeof(copy_unit_type);

	detail::copy(
		reinterpret_cast<copy_unit_type*>(destination),
		reinterpret_cast<const copy_unit_type*>(source),
		num_copy_unit_elements_to_copy
	);
	return destination;
}

/**
 * @brief Return the number of full warps in a linear grid
 * which would, overall, contain at least a given number of threads.
 *
 * @note This comes in handy more times than you must expect even in device-side code.
 *
 * @note the reason this function is defined directly rather than using
 * the functions in math or constexpr_math is that bit-counting is
 * either slow in run-time on the GPUwhen you use the constexpr way of
 * doing it, or not constexpr if you use the GPU-side population count
 * instruction.
 */
template <typename I>
constexpr KAT_FHD I num_warp_sizes_to_cover(I number_of_threads)
{
	static_assert(std::is_integral<I>::value, "Number of threads specified using a non-integral type");
	enum : I { mask = (warp_size - 1) };
	enum : I { log_warp_size = 5 } ;
	return (number_of_threads >> log_warp_size) + ((number_of_threads & mask) != 0);
}

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_MISCELLANY_CUH_
