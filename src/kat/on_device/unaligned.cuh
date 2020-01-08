/**
 * @file on_device/time.cuh
 *
 * @brief CUDA device-side functions related to making accesses to unaligned
 * global memory.
 */

#pragma once
#ifndef CUDA_KAT_ON_DEVICE_UNALIGNED_CUH_
#define CUDA_KAT_ON_DEVICE_UNALIGNED_CUH_

#include <kat/on_device/ptx.cuh>
#include <kat/detail/pointers.cuh>

#include <cstdint>
#include <type_traits>

#warning "This header has not been in use for a long time - use at your own peril (or rather - run some tests)"


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {

/**
 * Read a value from a potentially-unaligned location in memory
 *
 *
 * @param ptr address from which to read the value - which, unlike
 * a regular memory read in CUDA, does not need to be a multiple of
 * sizeof(T)
 * @return the T value whose bytes start at @p ptr
 *
 * @note this is a safe-but-possibly-slower variant, which will never try
 * to read past the 32-bit naturally-aligned word containing
 * the last relevant byte of data
 */
//template <typename T>
//__fd__ T read_unaligned(const T* __restrict__ ptr);


/**
 * Read a value from a potentially-unaligned location in memory,
 * assuming we can read past it
 *
 *
 * @param ptr address from which to read the value - which, unlike
 * a regular memory read in CUDA, does not need to be a multiple of 4
 * @return the value starting at @p ptr
 *
 * @note this is an unsafe variant, which may read past the end
 * of an array
 */template <typename T>
__fd__ T read_unaligned_unsafe(const T* __restrict__ ptr);


 template <>
__fd__ uint32_t read_unaligned_unsafe<uint32_t>(const uint32_t* __restrict__ ptr)
{
	/*
	 * So here's what the memory looks like:
	 *
	 * ... --------+-------------+-------------+-------- ...
	 * ...         | b0 b1 b2 b3 | b4 b5 b6 b7 |         ...
	 * ... --------+-------------+-------------+-------- ...
	 * ---           first_value   second_value
	 *
	 * these are two consecutive 4-byte (=32-bit) words in global device memory.
	 * Now, @p ptr points has the address of one of b0, b1, b2 or b3 -
	 * depending on its lower 2 bits.
	 *
	 * CUDA PTX has the facility to obtain the appropriate 4-byte value
	 * (which could be b0 through b3, or b1 through b4 etc.)
	 * when provided with the two aligned values and the address.
	 *
	 * The unsafe part here is that if @p ptr points to b0, then
	 * reading second_value may take us past the end of whatever memory
	 * the calling code has allocated (while if @p ptr points further on
	 * it's safe to assume the allocation covers everything up to b7).
	 */
	auto aligned_ptr  = cuda::align_down<uint32_t>(ptr);
	auto first_value  = *aligned_ptr;
	auto second_value = *(aligned_ptr + 1);

	return ptx::prmt_forward_4_extract(first_value, second_value, (uint32_t) cuda::address_as_number(ptr));
}

// From here on - untested!

template <>
__fd__ uint64_t read_unaligned_unsafe<uint64_t>(const uint64_t* __restrict__ ptr)
{
	auto aligned_word_ptr = reinterpret_cast<const uint32_t *>(cuda::align_down<uint64_t>(ptr));
	uint32_t aligned_native_words[3] =
		{ aligned_word_ptr[0], aligned_word_ptr[1], aligned_word_ptr[2] };
	auto prmt_control_bits = (uint32_t) cuda::address_as_number(ptr);
	return
		  (uint64_t) ptx::prmt_forward_4_extract(aligned_native_words[0], aligned_native_words[1], prmt_control_bits)
		| ((uint64_t) ptx::prmt_forward_4_extract(aligned_native_words[1], aligned_native_words[2], prmt_control_bits)) << 32;
}

/**
 *
 * @tparam T data type to read - must be trivial/POD, but can have arbitrary size
 * @tparam R data type in which to return the read result; we won't force this to
 * be the same as T in case that's an annoying type, e.g. for T being a 3-byte
 * integer we'd probably want a 4-byte integer, the more natural type for a GPU
 * @param ptr starting address of a type-T value in global device memory (possibly
 * in other spaces?)
 * @return the T value at @p ptr, within an R value
 */
template <typename T>
__fd__ T read_unaligned_unsafe(const T* __restrict__ ptr)
{
	static_assert(sizeof(T) <= 8, "Unsupported size");
	static_assert(std::is_trivial<T>::value,
		"Unaligned reading only implemented for \"trivial\" types");

	if (detail::has_nice_simple_size<T>()) {
		switch(sizeof(T)) {
		case 1:
			uint8_t v = read_unaligned_unsafe<uint8_t >(reinterpret_cast<const uint8_t*>(ptr));
			return reinterpret_cast<T>(v);
		case 2:
			v = read_unaligned_unsafe<uint16_t>(reinterpret_cast<const uint16_t*>(ptr));
			return reinterpret_cast<T>(v);
		case 4:
			v = read_unaligned_unsafe<uint32_t>(reinterpret_cast<const uint32_t*>(ptr));
			return reinterpret_cast<T>(v);
		case 8:
			v = read_unaligned_unsafe<uint64_t>(reinterpret_cast<const uint64_t*>(ptr));
			return reinterpret_cast<T>(v);
		}
		static_assert(false, "Unaligned reads of a single value only supported for types of sizes 1,2,4,8");
	}
	else {
		auto aligned_ptr  = cuda::align_down<T>(ptr);
		enum {
			native_word_size = sizeof(uint32_t),
			full_native_words_per_value = sizeof(T) < native_word_size ? 1 : sizeof(T) / native_word_size,
			native_words_per_unaligned_value = full_native_words_per_value + 1,
		};
		uint32_t aligned_native_words[native_words_per_unaligned_value];
		#pragma unroll
		for(int i = 0; i < native_words_per_unaligned_value; i++) {
			aligned_native_words[i] = aligned_ptr[i];
		}
		T result;
		auto prmt_control_bits = (uint32_t) cuda::address_as_number(ptr);
		if (sizeof(T) == 4) {
			return ptx::prmt_forward_4_extract(
				aligned_native_words[0], aligned_native_words[1], prmt_control_bits);
		}

		#pragma unroll
		for(int i = 0; i < full_native_words_per_value; i++) {
			reinterpret_cast<uint32_t *>(&result)[i] =
				ptx::prmt_forward_4_extract(
					aligned_native_words[i],
					aligned_native_words[i+1],
					prmt_control_bits);
		}
		if (sizeof(T) % native_word_size > 0) {
			auto final_word =
				ptx::prmt_forward_4_extract(
					aligned_native_words[full_native_words_per_value-1],
					aligned_native_words[full_native_words_per_value],
					prmt_control_bits);

			auto end_of_full_words = reinterpret_cast<uint32_t *>(&result)[full_native_words_per_value];
			switch (sizeof(T) % native_word_size > 0) {
			case 3:
				reinterpret_cast<uint16_t *>(end_of_full_words)[0] = cuda::lowest_k_bits(final_word, 16);
				reinterpret_cast<uint8_t  *>(end_of_full_words)[2] = cuda::bit_subsequence(final_word, 16, 8);
				break;
			case 2:
				reinterpret_cast<uint16_t &>(end_of_full_words) = cuda::lowest_k_bits(final_word, 16);
				break;
			case 1:
				reinterpret_cast<uint8_t  &>(end_of_full_words) = cuda::lowest_k_bits(final_word, 8);
				break;
			}
		}
		return result;
	}
}

__fd__ uint8_t read_unaligned_unsafe(const uint8_t* __restrict__ ptr)
{
	return *ptr;
}

__fd__ uint16_t read_unaligned_unsafe(const uint16_t* __restrict__ ptr)
{
	auto byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
	return
		((uint16_t) byte_ptr[0]) +
		((uint16_t) byte_ptr[1]) << cuda::bits_per_byte;
}

__fd__ uint8_t read_unaligned(const uint8_t* __restrict__ ptr)
{
	return *ptr;
}

__fd__ uint16_t read_unaligned(const uint16_t* __restrict__ ptr)
{
	return cuda::is_aligned(ptr) ? *ptr : read_unaligned_unsafe<uint16_t>(ptr);
}

__fd__ uint32_t read_unaligned(const uint32_t* __restrict__ ptr)
{
	return cuda::is_aligned(ptr) ? *ptr : read_unaligned_unsafe<uint32_t>(ptr);
}

/*
template <typename T, std::enable_if<sizeof(T), sizeof(uint32_t)>::type = 0>
__fd__ T read_unaligned<T>(const T* __restrict__ ptr)
{
	auto reinterpreted_ptr = reinterpret_cast<const uint32_t *>(ptr);
	auto v =
		cuda::is_aligned(ptr) ? *reinterpreted_ptr : read_unaligned_unsafe<uint32_t>(reinterpreted_ptr);
	return reinterpret_cast<T&>(v);
}
*/

__fd__ uint64_t read_unaligned(const uint64_t* __restrict__ ptr)
{
	return cuda::is_aligned(ptr) ? *ptr : read_unaligned_unsafe<uint64_t>(ptr);
}

/*
template <typename T>
__fd__ T read_unaligned(const T* __restrict__ ptr)
{
	static_assert(sizeof(T) <= 8, "Unsupported size");

	if (sizeof(T) == 4) {
		auto uptr = reinterpret_cast<const uint32_t*>(ptr);
		auto uv = read_unaligned<uint32_t>(uptr);
		return reinterpret_cast<T>(uv);
	}
	else if (sizeof(T) == 8) {
		return reinterpret_cast<T>(read_unaligned<uint64_t>(reinterpret_cast<const uint64_t*>(ptr)));
	}

	enum { single_read_size = round_down_to_power_of_2(sizeof(T), 4) }; // i.e. either 4 or 8
	auto start_offset_in_bytes_inside_first_read =
		(cuda::address_as_number(ptr) & ~((cuda::address_t) (single_read_size - 1)));
	if (start_offset_in_bytes_inside_first_read + sizeof(T) <= single_read_size) {
		return (single_read_size == 4) ?
			T{bit_subsequence(* reinterpret_cast<const uint32_t*>(ptr),
				start_offset_in_bytes_inside_first_read * cuda::bits_per_byte,
				cuda::size_in_bits<T>::value)} :
			T{bit_subsequence(* reinterpret_cast<const uint64_t*>(ptr),
				start_offset_in_bytes_inside_first_read * cuda::bits_per_byte,
				cuda::size_in_bits<T>::value)};

	}
	// At this point we are _certain_ we'll need data that' past the first
	// uint32_t or uint64_t as the case may be
	// TODO: ... but make sure that read_unaligned_unsafe only reads as much
	// extra as is
	return read_unaligned_unsafe(ptr);
}
*/

} // namespace kat


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond


#endif // CUDA_KAT_ON_DEVICE_UNALIGNED_CUH_
