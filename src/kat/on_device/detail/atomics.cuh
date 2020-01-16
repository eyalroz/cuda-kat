/*
 *
 * Copyright (c) 2018, Eyal Rozenberg, CWI Amsterdam
 * Copyright (c) 2019, Eyal Rozenberg <eyalroz@technion.ac.il>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of CWI Amsterdam nor the names of its contributors may
 *    be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_ATOMICS_DETAIL_CUH_
#define CUDA_KAT_ON_DEVICE_ATOMICS_DETAIL_CUH_

#include "atomics/missing_in_cuda.cuh"

#include <kat/on_device/common.cuh>
#include <kat/on_device/builtins.cuh>
#include <device_atomic_functions.h>
#if (CUDART_VERSION >= 8000)
#include <sm_60_atomic_functions.h>
#include <cuda_fp16.h>
#endif
#include <kat/detail/pointers.cuh>

#include <functional>
#include <type_traits>
#include <climits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace atomic {
namespace detail {

template <unsigned NBytes> struct uint_helper;

template<> struct uint_helper<1> { using type = uint8_t;  };
template<> struct uint_helper<2> { using type = uint16_t; };
template<> struct uint_helper<4> { using type = uint32_t; };
template<> struct uint_helper<8> { using type = uint64_t; };

template <unsigned NBytes>
using uint_t = typename uint_helper<NBytes>::type;

/**
 * @brief extracts a smaller-size value using a contiguous sequence
 * of bytes representing a value of another, larger type.
 *
 * @note This may not work for any pair of types; but should work
 * for integral types of sizes which are a power of 2.
 */
template <typename Smaller, typename Larger, typename Offset>
KAT_FHD Smaller extract_value(Larger larger_value ,Offset offset_into_larger_value)
{
	static_assert(std::is_trivial<Smaller>::value, "Cannot extract non-trivially-copyable values");
	static_assert(std::is_trivial<Larger>::value, "Cannot extract from non-trivially-copyable values");
	unsigned char* smaller_value_ptr = reinterpret_cast<unsigned char*>(larger_value) + offset_into_larger_value;
		// In C++17 this could have been std::byte I suppose
	return reinterpret_cast<const Smaller&>(smaller_value_ptr);
}


template <typename T>
struct tag { };

template <typename U, typename I>
constexpr U KAT_FHD ones(I num_ones) { return (U{1} << num_ones) - I{1}; }

template <typename Larger, typename Smaller, typename Offset>
uint_t<sizeof(Larger)> KAT_FHD bitmask_for_value_at_offset(Offset offset_into_larger_value)
{
    using larger_uint_type      = uint_t<sizeof(Larger)>;

    constexpr const auto unshifted_ones { ones<larger_uint_type>(sizeof(Smaller) * CHAR_BIT) };
    auto shift_amount = offset_into_larger_value * CHAR_BIT;
    return unshifted_ones << shift_amount;
}

template <typename Larger, typename Smaller, typename Offset>
uint_t<sizeof(Larger)> KAT_FHD bitmask_for_padding_before_value_at_offset(Offset offset_into_larger_value)
{
    return ~bitmask_for_value_at_offset(offset_into_larger_value);
}

template <typename Larger, typename Smaller, typename Offset>
Larger KAT_FHD generate_value_at_offset(Smaller small_value, Offset offset_into_larger_value)
{
    static_assert(sizeof(Larger) > sizeof(Smaller), "invalid sizes");
    static_assert(std::is_trivial<Larger>::value, "Larger must be a trivial type");
    static_assert(std::is_trivial<Smaller>::value, "Smaller must be a trivial type");
    auto shift_amount = offset_into_larger_value * CHAR_BIT;
    using smaller_uint = uint_t<sizeof(Smaller)>;
    using larger_uint = uint_t<sizeof(Larger)>;

    auto small_value_as_uint { reinterpret<smaller_uint>(small_value) };
    auto result_as_uint = larger_uint{small_value_as_uint} << shift_amount;
    return reinterpret<Larger>(result_as_uint);
}

template <typename Larger, typename Smaller, typename Offset>
Larger KAT_FHD replace_bytes_at_offset(Larger larger_value, Smaller replacement, Offset offset_into_larger_value)
{
    auto clearing_mask { ~bitmask_for_value_at_offset<Larger, Smaller>(offset_into_larger_value) };
    return (larger_value & clearing_mask ) | generate_value_at_offset<Larger>(replacement, offset_into_larger_value);
    	// TODO: Some Larger types may not admit bitwise-and'ing, so we'd have to reinterpret them as larger_uint_type
}

namespace implementation {

template <typename T>
KAT_FD T compare_and_swap(std::true_type, T* __restrict__ address, T compare, T val)
{
	static_assert(sizeof(T) == sizeof(int) or sizeof(T) == sizeof(long long int),
		"Cannot compare_and_swap directly with the requested type");

	// This switch is necessary due to atomicCAS being defined
	// only for a very small selection of types - int and unsigned long long
	switch(sizeof(T)) {
	case sizeof(int): {
		int int_result = atomicCAS(
			reinterpret_cast<int*      >(address),
			reinterpret_cast<const int&>(compare),
			reinterpret_cast<const int&>(val)
		);
		return reinterpret<T>(int_result);
	}
	case sizeof(long long int): {
		long long int llint_result = atomicCAS(
			reinterpret_cast<unsigned long long*      >(address),
			reinterpret_cast<const unsigned long long&>(compare),
			reinterpret_cast<const unsigned long long&>(val)
		);
		return reinterpret<T>(llint_result);
	}
	default: return T(); // should not be able to get here
	}
}


template <typename T>
KAT_FD T compare_and_swap(std::false_type, T* __restrict__ address, T compare, T val)
{
	// The idea is to apply compare-and-swap on more memory than the T type actually takes
	// up. However, we can't just have that larger stretch of memory start at `address`, since
	// NVIDIA GPUs require operands to machine instructions to be naturally-aligned (e.g.
	// a 4-byte-sized operand must start at an address being a multiple of 4). That means we
	// could theoretically have "slack" bytes we use in the compare-and-swap both before and
	// after the value we're actually interested in. We'll have to "contend" for atomic access
	// to both the proper bytes of interest and the slack bytes, together.

	static_assert (not std::is_const<T>::value, "Can't compare-and-swap with a const value");
	static_assert (sizeof(T) <= sizeof(long long int), "Type size too large for atomic compare-and-swap");
	using casable_type = typename std::conditional<sizeof(T) < sizeof(int), int, unsigned long long int>::type;

	casable_type* casable = kat::detail::align_down<casable_type>(address);
	const auto offset_into_casable_type { kat::detail::address_difference(address, casable) };
	const auto mask_for_value_at_offset { bitmask_for_value_at_offset<casable_type, T>(offset_into_casable_type) };
		// The on-bits are those which are _not_ used for the T-type value within the casable value at address
	auto value_at_offset { generate_value_at_offset<casable_type>(val, offset_into_casable_type) };
		// ... with 0's in all other bits

	auto last_known_casable_value { *casable };
	auto casable_value_to_swap_in { replace_bytes_at_offset(last_known_casable_value, val, offset_into_casable_type) };
	casable_type expected_value_at_addr = last_known_casable_value;
	bool value_of_interest_changed;
	do {
		last_known_casable_value = atomic::compare_and_swap<casable_type>(
			casable, expected_value_at_addr, casable_value_to_swap_in);

		auto cas_did_swap = (last_known_casable_value == value_at_offset);
		if (cas_did_swap) { return compare; }

		auto value_of_interest_at_address { extract_value<T>(last_known_casable_value, offset_into_casable_type) };
		if (value_of_interest_at_address != compare) {
			return value_of_interest_at_address;
		}
		// At this point, it must be the case that the padding bytes changed. That means
		// we still need to try to perform the CASing - but we need to update the padding
		// so that we don't switch it back.
		expected_value_at_addr |= (last_known_casable_value & ~mask_for_value_at_offset);
	} while (true);
}

} // namespace implementation
} // namespace detail

template <typename T>
KAT_FD T compare_and_swap(T* address, T compare, T val)
{
	static_assert(sizeof(T) <= sizeof(long long int),
		"nVIDIA GPUs do not support atomic operations on data larger than long long int");

	// CUDA PTX (or at least, the CUDA on-device API) cannot compare-and-swap all fundamental
	// numeric C/C++ types; and some types it can CAS, but only if you pass them as another
	// type of the same size. Thus we need to choose between an implementation which
	// compares-and-swaps directly, and another which uses a larger value that is supported.

	constexpr bool can_cas_directly =
		(sizeof(T) == sizeof(int)) or (sizeof(T) == sizeof(unsigned long long int));

	return detail::implementation::compare_and_swap<T>(
		std::integral_constant<bool, can_cas_directly>{},
		address, compare, val);
}


namespace detail {
namespace implementation {

template <typename UnaryFunction, typename T>
KAT_FD T apply(std::true_type, UnaryFunction f, T* __restrict__ address)
{
	T newest_value_found_at_addr { *address };
	T value_expected_at_addr;
	do {
		value_expected_at_addr = newest_value_found_at_addr;
		auto value_to_set = f(value_expected_at_addr);
		newest_value_found_at_addr = detail::implementation::compare_and_swap<T>(std::true_type{}, address, value_expected_at_addr, value_to_set);
		if (newest_value_found_at_addr != value_expected_at_addr) {
		}
	} while(newest_value_found_at_addr != value_expected_at_addr);
	return newest_value_found_at_addr;
}

template <typename UnaryFunction, typename T>
KAT_FD T apply(std::false_type, UnaryFunction f, T* __restrict__ address)
{
	// Similar to the no-primitive-available compare_and_swap, except that we don't have a "compare"
	// value, i.e. we don't condition our action on a specific existing value - we'll go with whatever
	// is at the address

	static_assert (not std::is_const<T>::value, "Can't compare-and-swap with a const value");
	static_assert (sizeof(T) <= sizeof(long long int), "Type size too large for atomic compare-and-swap");
	using casable_type = typename std::conditional<sizeof(T) < sizeof(int), int, unsigned long long int>::type;

	casable_type* casable = kat::detail::align_down<casable_type>(address);
	const auto offset_into_casable_type { kat::detail::address_difference(address, casable) };
	// const auto mask_for_value_at_offset { bitmask_for_value_at_offset<casable_type, T>(offset_into_casable_type) };
		// The on-bits are those which are _not_ used for the T-type value within the casable value at address
	// auto value_at_offset { generate_value_at_offset<casable_type>(val, offset_into_casable_type) };
		// ... with 0's in all other bits

	casable_type extracted_from_known_casable;
	bool cas_did_swap;
	auto known_value_of_casable { *casable };
	while (true) {
		extracted_from_known_casable = extract_value<T>(known_value_of_casable, offset_into_casable_type);
		casable_type casable_value_to_swap_in =
			replace_bytes_at_offset(known_value_of_casable, f(extracted_from_known_casable), offset_into_casable_type);
		auto new_known_value_of_casable = atomic::compare_and_swap<casable_type>(
			casable, known_value_of_casable, casable_value_to_swap_in);
		cas_did_swap = (new_known_value_of_casable == known_value_of_casable);
		if (cas_did_swap) {
			return extracted_from_known_casable;
		}
		known_value_of_casable = new_known_value_of_casable;
	};
}

} // namespace implementation
} // namespace detail

template <typename UnaryFunction, typename T>
KAT_FD T apply(UnaryFunction f, T* __restrict__ address)
{
	static_assert(sizeof(T) <= sizeof(long long int),
		"nVIDIA GPUs do not support atomic operations on data larger than long long int");

	// We can't really apply atomically; we can only apply f to a copy - which means
	// we can't avoid a loop. But we can still benefit from an atomic CAS if it's _directly_
	// available to us.

	constexpr bool can_cas_directly =
		(sizeof(T) == sizeof(int)) or (sizeof(T) == sizeof(unsigned long long int));

	return detail::implementation::apply<UnaryFunction, T>(
		std::integral_constant<bool, can_cas_directly>{}, f, address);
}

template <typename Function, typename T, typename... Ts>
KAT_FD T apply(
	Function                f,
	T*       __restrict__   address,
	const Ts...             xs)
{
	auto uf = [&](T existing_value) -> T { return f(existing_value, xs...); };
	static_assert(std::is_same<decltype(uf(*address)), T>::value,
		"The function to apply must return the same type it takes");
	return apply(uf, address);
}

namespace detail {
namespace implementation {

template <typename T>
KAT_FD T increment(
	std::true_type,
	T*  address,
	T   wraparound_value)
{
	static_assert(sizeof(T) == sizeof(unsigned int), "invalid type");
	return (T) (atomicInc(
		reinterpret_cast<unsigned int*      >(address),
		reinterpret_cast<const unsigned int&>(wraparound_value)
	));
}

template <typename T>
KAT_FD T increment(
	std::false_type,
	T*  address,
	T   wraparound_value)
{
	auto do_increment = [](T existing_value, T wraparound) -> T { return existing_value >= wraparound ? T{0} : T{existing_value+1}; };
	return atomic::apply(do_increment, address, wraparound_value);
}

} // namespace implementation
} // namespace detail

template <typename T>
KAT_FD T increment(
	T*  address,
	T   wraparound_value)
{
	constexpr bool can_act_directly = (sizeof(T) == sizeof(unsigned int));

	return detail::implementation::increment<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, wraparound_value);
}

namespace detail {
namespace implementation {

template <typename T>
KAT_FD T decrement (
	std::true_type,
	T*  address,
	T   wraparound_value)
{
	static_assert(sizeof(T) == sizeof(unsigned int), "invalid type");
	return (T) (atomicDec(
		reinterpret_cast<unsigned int*      >(address),
		reinterpret_cast<const unsigned int&>(wraparound_value)
	));
}

template <typename T>
KAT_FD T decrement (
	std::false_type,
	T*  address,
	T   wraparound_value)
{
	auto do_decrement =
		[](auto existing_value, auto wraparound) -> T {
			return
				((existing_value <= 0) or (existing_value >= wraparound)) ?
				wraparound - 1 :
				existing_value - 1;
		};
	return apply(do_decrement, address, wraparound_value);
}

} // namespace implementation
} // namespace detail

template <typename T>
KAT_FD T decrement (
	T*  address,
	T   wraparound_value)
{
	constexpr bool can_act_directly = (sizeof(T) == sizeof(unsigned int));

	return detail::implementation::decrement<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, wraparound_value);
}

namespace detail {
namespace implementation {

template <typename T>
KAT_FD T add(std::true_type, T* address, T val)
{
	return ::atomicAdd(address, val);
}

template <typename T>
KAT_FD T add(std::false_type, T* address, T val)
{
	auto do_addition = [](T existing_value, T x) -> T { return static_cast<T>(existing_value + x); };
	return kat::atomic::apply(do_addition, address, val);
}

// TODO: Double-check (no pun intended) that pre-Pascal cards, or Pascal-and-onwards
// cards running CC 3.0 code, handle doubles properly - I have my doubts.


template <typename T>
KAT_FD T subtract(std::true_type, T*  address, T val)
{
	return ::atomicSub(address, val);
}

template <typename T>
KAT_FD T subtract(std::false_type, T* address, T val)
{
	auto do_subtraction = [](T existing_value, T x) -> T { return static_cast<T>(existing_value - x); };
	return kat::atomic::apply(do_subtraction, address, val);
}

template <typename T>
KAT_FD T exchange(std::true_type, T*  address, T val)
{
	// Note: We know there are implementations available for int, unsigned, unsigned long long and float;
	// but the only thing we should really care about here is the size.
	static_assert(sizeof(unsigned) == sizeof(float), "Unexpected fundamental type sizes");

	switch(sizeof(T)) {
	case sizeof(unsigned): {
		unsigned previous_value =
			::atomicExch(reinterpret_cast<unsigned*>(address), reinterpret<unsigned>(val));
		return reinterpret<T>(previous_value);
	}
	case sizeof(unsigned long long): {
		unsigned long long previous_value =
			::atomicExch(reinterpret_cast<unsigned long long*>(address), reinterpret<unsigned long long>(val));
		return reinterpret<T>(previous_value);
	}
	default: // should not be able to get here
		return T{};
	}
}

template <typename T>
KAT_FD T exchange(std::false_type, T* address, T val)
{
	auto do_exchange = [](T existing_value, T x) -> T { return x; };
	return kat::atomic::apply(do_exchange, address, val);
}

template <typename T>
KAT_FD T min(std::true_type, T*  address, T val)
{
	return ::atomicMin(address, val);
}

template <typename T>
KAT_FD T min(std::false_type, T* address, T val)
{
	auto do_min = [](T existing_value, T x) -> T { return builtins::minimum(existing_value, x); };
	return kat::atomic::apply(do_min, address, val);
}

template <typename T>
KAT_FD T max(std::true_type, T*  address, T val)
{
	return ::atomicMax(address, val);
}

template <typename T>
KAT_FD T max(std::false_type, T* address, T val)
{
	auto do_max = [](T existing_value, T x) -> T { return builtins::maximum(existing_value, x); };
	return kat::atomic::apply(do_max, address, val);
}

template <typename T>
KAT_FD T bitwise_and(std::false_type, T* address, T val)
{
	auto do_and = [](T existing_value, T x) -> T { return existing_value & x; };
	return kat::atomic::apply(do_and, address, val);
}

template <typename T>
KAT_FD T bitwise_or(std::false_type, T* address, T val)
{
	auto do_or = [](T existing_value, T x) -> T { return existing_value | x; };
	return kat::atomic::apply(do_or, address, val);
}

template <typename T>
KAT_FD T bitwise_xor(std::false_type, T* address, T val)
{
	auto do_xor = [](T existing_value, T x) -> T { return existing_value ^ x; };
	return kat::atomic::apply(do_xor, address, val);
}

// TODO: Double-check (no pun intended) that pre-Pascal cards, or Pascal-and-onwards
// cards running CC 3.0 code, handle doubles properly - I have my doubts.

} // namespace implementation
} // namespace detail

template <typename T>
KAT_FD T add(T*  address, T val)
{
	constexpr bool can_act_directly =
		   std::is_same< T,int                >::value
		or std::is_same< T,long int           >::value
		or std::is_same< T,long long int      >::value
		or std::is_same< T,unsigned           >::value
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
		or std::is_same< T,float              >::value
#if __CUDA_ARCH__ >= 600
		or std::is_same< T,half               >::value
		or std::is_same< T,double             >::value
#endif
		;
	return detail::implementation::add<T>(
		std::integral_constant<bool, can_act_directly>{},
		//std::integral_constant<bool, false>{},
		address, val);
}

template <typename T>
KAT_FD T subtract (T*  address, T val)
{
	constexpr bool can_act_directly =
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
		;
	return detail::implementation::subtract<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T exchange (T*  address, T val)
{
	constexpr bool can_act_directly = (sizeof(T) == 4) or (sizeof(T) == 8);
	return detail::implementation::exchange<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T min (T*  address, T val)
{
	constexpr bool can_act_directly =
#if CUDA_ARCH >= 320
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
#if  CUDA_ARCH >= 350
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
#endif
#else
		false
#endif
		;
	return detail::implementation::min<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T max (T*  address, T val)
{
	constexpr bool can_act_directly =
#if CUDA_ARCH >= 320
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
#if  CUDA_ARCH >= 350
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
#endif
		#else
		false
#endif
		;
	return detail::implementation::max<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T bitwise_and (T*  address, T val)
{
	constexpr bool can_act_directly =
#if CUDA_ARCH >= 320
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
#if  CUDA_ARCH >= 350
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
#endif
#else
		false
#endif
		;
	return detail::implementation::bitwise_and<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T bitwise_or (T*  address, T val)
{
	constexpr bool can_act_directly =
#if CUDA_ARCH >= 320
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
#if  CUDA_ARCH >= 350
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
#endif
#else
		false
#endif
		;
	return detail::implementation::bitwise_or<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

template <typename T>
KAT_FD T bitwise_xor (T*  address, T val)
{
	constexpr bool can_act_directly =
#if CUDA_ARCH >= 320
		   std::is_same< T,int                >::value
		or std::is_same< T,unsigned           >::value
#if  CUDA_ARCH >= 350
		or std::is_same< T,unsigned long      >::value
		or std::is_same< T,unsigned long long >::value
#endif
#else
		false
#endif
		;
	return detail::implementation::bitwise_xor<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, val);
}

// ... and now for some ops for which there are absolutely no primitives,
// so that they can only be implemented using apply()

template <typename T>
KAT_FD T logical_and(T* address, T val)
{
	auto do_logical_and = [](T existing_value, T x) -> T { return existing_value and x; };
	return kat::atomic::apply(do_logical_and, address, val);
}

template <typename T>
KAT_FD T logical_or(T* address, T val)
{
	auto do_logical_or = [](T existing_value, T x) -> T { return existing_value or x; };
	return kat::atomic::apply(do_logical_or, address, val);
}

template <typename T>
KAT_FD T logical_xor(std::false_type, T* address, T val)
{
	auto do_logical_xor = [](T existing_value, T x) -> T { return (existing_value and not x) or (not existing_value and x); };
	return kat::atomic::apply(do_logical_xor, address, val);
}

template <typename T>
KAT_FD T logical_not(std::false_type, T* address)
{
	auto do_logical_not = [](T existing_value) -> T { return not existing_value; };
	return kat::atomic::apply(do_logical_not, address);
}

template <typename T>
KAT_FD T bitwise_not (T* address)
{
	constexpr const T all_ones { ~T{0} };
	return bitwise_xor(address, all_ones);
}

template <typename T>
KAT_FD T set_bit (T* address, native_word_t bit_index)
{
	auto f = [](T existing_value, native_word_t x) -> T { return existing_value | 1 << x; };
	return apply(f, address, bit_index);
}

template <typename T>
KAT_FD T unset_bit (T* address, native_word_t bit_index)
{
	auto f = [](T existing_value, native_word_t x) -> T { return existing_value & ~(1 << x); };
	return apply(f, address, bit_index);
}

} // namespace atomic
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_DETAIL_CUH_
