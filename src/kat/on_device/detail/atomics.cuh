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

#include <kat/on_device/common.cuh>
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
#include <kat/define_specifiers.hpp>
///@endcond

// Annoyingly, CUDA - upto and including version 10.0 - provide atomic
// operation wrappers for unsigned int and unsigned long long int, but
// not for the in-between type of unsigned long int. So - we
// have to make our own. Don't worry about the condition checks -
// they are optimized away at compile time and the result is basically
// a single PTX instruction
__fd__ unsigned long int atomicAdd(unsigned long int *address, unsigned long int val)
{
	static_assert(
		sizeof (unsigned long int) == sizeof(unsigned long long int) or
		sizeof (unsigned long int) == sizeof(unsigned int),
		"Unexpected size of unsigned long int");
    if (sizeof (unsigned long int) == sizeof(unsigned long long int)) {
        return ::atomicAdd(reinterpret_cast<unsigned long long int*>(address), val);
    }
    else {
        // It holds that sizeof (unsigned long int) == sizeof(unsigned int)
        return  ::atomicAdd(reinterpret_cast<unsigned int*>(address), val);
    }
}

__fd__ long int atomicAdd(long* address, long val)
{
	return atomicAdd(reinterpret_cast<unsigned long*>(address), static_cast<unsigned long>(val));
}

__fd__ long long int atomicAdd(long long* address, long long val)
{
	return ::atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
}


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


// TODO: Unify this with reinterpret()
template<typename Interpreted, typename Original>
__fhd__ Interpreted reinterpret2(Original& x)
{
	return *reinterpret_cast<Interpreted*>(&x);
}

/**
 * @brief extracts a smaller-size value using a contiguous sequence
 * of bytes representing a value of another, larger type.
 *
 * @note This may not work for any pair of types; but should work
 * for integral types of sizes which are a power of 2.
 */
template <typename Smaller, typename Larger, typename Offset>
__fhd__ Smaller extract_value(Larger larger_value ,Offset offset_into_larger_value)
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
constexpr U __fhd__ ones(I num_ones) { return (U{1} << num_ones) - I{1}; }

template <typename Larger, typename Smaller, typename Offset>
uint_t<sizeof(Larger)> __fhd__ bitmask_for_value_at_offset(Offset offset_into_larger_value)
{
    using larger_uint_type      = uint_t<sizeof(Larger)>;

    constexpr const auto unshifted_ones { ones<larger_uint_type>(sizeof(Smaller) * CHAR_BIT) };
//    printf("unshifted_ones (should be %d ones): %X\n", (unsigned) (sizeof(Smaller) * CHAR_BIT), (unsigned int) unshifted_ones);
    auto shift_amount = offset_into_larger_value * CHAR_BIT;
    return unshifted_ones << shift_amount;
}

template <typename Larger, typename Smaller, typename Offset>
uint_t<sizeof(Larger)> __fhd__ bitmask_for_padding_before_value_at_offset(Offset offset_into_larger_value)
{
    return ~bitmask_for_value_at_offset(offset_into_larger_value);
}

template <typename Larger, typename Smaller, typename Offset>
Larger __fhd__ generate_value_at_offset(Smaller small_value, Offset offset_into_larger_value)
{
    static_assert(sizeof(Larger) > sizeof(Smaller), "invalid sizes");
    static_assert(std::is_trivial<Larger>::value, "Larger must be a trivial type");
    static_assert(std::is_trivial<Smaller>::value, "Smaller must be a trivial type");
    auto shift_amount = offset_into_larger_value * CHAR_BIT;
    using smaller_uint = uint_t<sizeof(Smaller)>;
    using larger_uint = uint_t<sizeof(Larger)>;

    auto small_value_as_uint { reinterpret2<smaller_uint>(small_value) };
    auto result_as_uint = larger_uint{small_value_as_uint} << shift_amount;
    return reinterpret2<Larger>(result_as_uint);
}

template <typename Larger, typename Smaller, typename Offset>
Larger __fhd__ replace_bytes_at_offset(Larger larger_value, Smaller replacement, Offset offset_into_larger_value)
{
    auto clearing_mask { ~bitmask_for_value_at_offset<Larger, Smaller>(offset_into_larger_value) };
//    printf("mask is %X\n", (unsigned) clearing_mask);
//    printf("(larger_value & mask ) = %d , generate_value_at_offset<Larger>(...) = %d\n",
//    	(larger_value & ~mask ), generate_value_at_offset<Larger>(replacement, offset_into_larger_value));
    return (larger_value & clearing_mask ) | generate_value_at_offset<Larger>(replacement, offset_into_larger_value);
    	// TODO: Some Larger types may not admit bitwise-and'ing, so we'd have to reinterpret them as larger_uint_type
}

/*template <typename T, typename CASable>
__fd__ T compare_and_swap(std::true_type, T* __restrict__ address, T compare, T val);

template <>
__fd__ T compare_and_swap<double, double>(std::true_type, double* __restrict__ address, double compare, double val)
{
	static_assert(sizeof(double) == sizeof(long long), "Unexpected sizes");
	printf("In compare_and_swap with a primitive\n");
	return __longlong_as_double(
		atomicCAS(
			reinterpret_cast<unsigned long long*>(address),
			__double_as_longlong(compare),
			__double_as_longlong(val)
		)
	);
}*/

/*
template <typename T>
__fd__ T compare_and_swap<T, int>(
	std::true_type,
	T* __restrict__ address,
	T compare,
	T val)
{
	static_assert(sizeof(T) == sizeof(int), "Unexpected sizes");
	printf("In compare_and_swap with a primitive\n");
	return reinterpret_cast<T>(atomicCAS(
			reinterpret_cast<unsigned long long*      >(address),
			reinterpret_cast<const unsigned long long&>(compare),
			reinterpret_cast<const unsigned long long&>(val)
		));
}

*/

template <typename T>
__fd__ T compare_and_swap(std::true_type, T* __restrict__ address, T compare, T val)
{
	static_assert(sizeof(T) == sizeof(int) or sizeof(T) == sizeof(long long int),
		"Cannot compare_and_swap directly with the requested type");

	printf("In compare_and_swap with a primitive\n");


	// This switch is necessary due to atomicCAS being defined
	// only for a very small selection of types - int and unsigned long long
	switch(sizeof(T)) {
	case sizeof(int): {
		int int_result = atomicCAS(
			reinterpret_cast<int*      >(address),
			reinterpret_cast<const int&>(compare),
			reinterpret_cast<const int&>(val)
		);
		return reinterpret_cast<const T&>(int_result);
	}
	case sizeof(long long int): {
		long long int llint_result = atomicCAS(
			reinterpret_cast<unsigned long long*      >(address),
			reinterpret_cast<const unsigned long long&>(compare),
			reinterpret_cast<const unsigned long long&>(val)
		);
		return reinterpret_cast<const T&>(llint_result);
	}
	default: return T(); // should not be able to get here
	}
}


/*
template <typename T, typename CASCapable>
__fd__ T non_atomic_compare_and_swap(
	CASCapable* __restrict__  address_for_builtin_cas,
	CASCapable                expected_previous_value,
	std::ptrdiff_t            offset_into_cas_capable,
	const T& compare,
	const T& val,
	cas_builtin_type last_known_casable_value = *casable)
{
	using cas_builtin_type = decltype(*address_for_builtin_cas);

	auto expected_value_at_addr = last_known_casable_value;
	auto value_to_swap_in { replace_bytes_at_offset(expected_value_at_addr, val, address_difference) };
		// TODO: I'm worried about wasting instructions here for a conversion of address_difference to size_t

	auto last_known_casable_value = atomic::compare_and_swap<cas_builtin_type>(
		casable, expected_value_at_addr, value_to_swap_in);
	return last_known_casable_value;
};

// So, nothing actually uses this one, it's just a demonstration. The "inner" function
// will actually get used.
template <typename T>
__fd__ T non_atomic_compare_and_swap(T* __restrict__ address, const T& compare, const T& val)
{
	using cas_builtin_type = typename std::conditional<sizeof(T) < sizeof(int), int, unsigned long long int>::type;

	cas_builtin_type expected_value_at_addr = last_known_casable_value;
	auto value_to_swap_in { replace_bytes_at_offset(expected_value_at_addr, val, address_difference) };

	cas_builtin_type last_known_casable_value = atomic::compare_and_swap<cas_builtin_type>(
		casable, expected_value_at_addr, value_to_swap_in);
	return last_known_casable_value;
};
*/

template <typename T>
__fd__ T compare_and_swap(std::false_type, T* __restrict__ address, T compare, T val)
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

	printf("In compare_and_swap without a primitive\n");

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

} // namespace detail

template <typename T>
__fd__ T compare_and_swap(T* address, T compare, T val)
{
	static_assert(sizeof(T) <= sizeof(long long int),
		"nVIDIA GPUs do not support atomic operations on data larger than long long int");

	// CUDA PTX (or at least, the CUDA on-device API) cannot compare-and-swap all fundamental
	// numeric C/C++ types; and some types it can CAS, but only if you pass them as another
	// type of the same size. Thus we need to choose between an implementation which
	// compares-and-swaps directly, and another which uses a larger value that is supported.

	constexpr bool can_cas_directly =
		(sizeof(T) == sizeof(int)) or (sizeof(T) == sizeof(unsigned long long int));
//	printf("can_cas_directly ? %s\n", (can_cas_directly ? "yes" : "no"));

	return detail::compare_and_swap<T>(
		std::integral_constant<bool, can_cas_directly>{},
		address, compare, val);
}


namespace detail {

template <typename UnaryFunction, typename T>
__fd__ T apply(std::true_type, UnaryFunction f, T* __restrict__ address)
{
	printf("apply uniform function with direct CASing\n");
	T newest_value_found_at_addr { *address };
	printf("Initially, seeing %f at address\n", newest_value_found_at_addr);
	T value_expected_at_addr;
	do {
		value_expected_at_addr = newest_value_found_at_addr;
		auto value_to_set = f(value_expected_at_addr);
		printf("Expecting %f at address, trying to set to %f\n", value_expected_at_addr, value_to_set);
		newest_value_found_at_addr = detail::compare_and_swap<T>(std::true_type{}, address, value_expected_at_addr, value_to_set);
		if (newest_value_found_at_addr != value_expected_at_addr) {
			printf("Found %f rather than the expected %f !\n", newest_value_found_at_addr, value_expected_at_addr);
		}
	} while(newest_value_found_at_addr != value_expected_at_addr);
	printf("CAS succeeded - have set to %f.\n", f(value_expected_at_addr));
	return newest_value_found_at_addr;
}

template <typename UnaryFunction, typename T>
__fd__ T apply(std::false_type, UnaryFunction f, T* __restrict__ address)
{
	printf("apply uniform function without direct CASing\n");
	// Similar to the no-primitive-available compare_and_swap, except that we don't have a "compare"
	// value, i.e. we don't condition our action on a specific existing value - we'll go with whatever
	// is at the address

	static_assert (not std::is_const<T>::value, "Can't compare-and-swap with a const value");
	static_assert (sizeof(T) <= sizeof(long long int), "Type size too large for atomic compare-and-swap");
	using casable_type = typename std::conditional<sizeof(T) < sizeof(int), int, unsigned long long int>::type;

	casable_type* casable = kat::detail::align_down<casable_type>(address);
	const auto offset_into_casable_type { kat::detail::address_difference(address, casable) };
//	printf("address = %p , casable  = %p of size %u which %s unsigned, offset into casable = %u\n",
//		address, casable, (unsigned) sizeof(casable_type), (std::is_unsigned<casable_type>::value ? "is" : "isn't"), (unsigned) offset_into_casable_type );
	// const auto mask_for_value_at_offset { bitmask_for_value_at_offset<casable_type, T>(offset_into_casable_type) };
		// The on-bits are those which are _not_ used for the T-type value within the casable value at address
	// auto value_at_offset { generate_value_at_offset<casable_type>(val, offset_into_casable_type) };
		// ... with 0's in all other bits

	casable_type extracted_from_known_casable;
	bool cas_did_swap;
	auto known_value_of_casable { *casable };
//	printf("known_value_of_casable is %d\n", (int) known_value_of_casable);
	while (true) {
		extracted_from_known_casable = extract_value<T>(known_value_of_casable, offset_into_casable_type);
//		printf("extracted_from_known_casable: %hd; f(extracted_from_known_casable) = %hd\n", (short int) extracted_from_known_casable, (short int) f(extracted_from_known_casable) );
		casable_type casable_value_to_swap_in =
			replace_bytes_at_offset(known_value_of_casable, f(extracted_from_known_casable), offset_into_casable_type);
//		printf("casable_value_to_swap_in: %d\n", (int) casable_value_to_swap_in);
		auto new_known_value_of_casable = atomic::compare_and_swap<casable_type>(
			casable, known_value_of_casable, casable_value_to_swap_in);
		cas_did_swap = (new_known_value_of_casable == known_value_of_casable);
		if (cas_did_swap) {
//			printf("CAS swapped %d with %d\n", (int) known_value_of_casable, (int) casable_value_to_swap_in );
			return extracted_from_known_casable;
		}
//		else { printf("CAS didn't swap!\n"); }
		known_value_of_casable = new_known_value_of_casable;
	};
}

} // namespace detail

template <typename UnaryFunction, typename T>
__fd__ T apply(UnaryFunction f, T* __restrict__ address)
{
	static_assert(sizeof(T) <= sizeof(long long int),
		"nVIDIA GPUs do not support atomic operations on data larger than long long int");

	// We can't really apply atomically; we can only apply f to a copy - which means
	// we can't avoid a loop. But we can still benefit from an atomic CAS if it's _directly_
	// available to us.

	constexpr bool can_cas_directly =
		(sizeof(T) == sizeof(int)) or (sizeof(T) == sizeof(unsigned long long int));

	return detail::apply<UnaryFunction, T>(
		std::integral_constant<bool, can_cas_directly>{}, f, address);
}

template <typename Function, typename T, typename... Ts>
__fd__ T apply(
	Function                f,
	T*       __restrict__   address,
	const Ts...             xs)
{
	auto uf = [&](T x) { return f(x, xs...); };
	static_assert(std::is_same<decltype(uf(*address)), T>::value,
		"The function to apply must return the same type it takes");
	return apply(uf, address);
}


namespace detail {

/**
 * Use CUDA intrinsics where possible and relevant to reinterpret the bits
 * of values of different types
 *
 * @param x[in]  the value to reinterpret. No references please!
 * @return the reinterpreted value
 */
template <typename ToInterpret, typename Interpreted>
__fd__  Interpreted reinterpret(
	typename std::enable_if<
		!std::is_same<
			typename std::decay<ToInterpret>::type, // I actually just don't want references here
			typename std::decay<Interpreted>::type>::value && // I actually just don't want references here
		sizeof(ToInterpret) == sizeof(Interpreted), ToInterpret>::type x)
{
	return x;
}

template<> __fd__ double reinterpret<long long int, double>(long long int x) { return __longlong_as_double(x); }
template<> __fd__ long long int reinterpret<double, long long int>(double x) { return __double_as_longlong(x); }

template<> __fd__ double reinterpret<unsigned long long int, double>(unsigned long long int x) { return __longlong_as_double(x); }
template<> __fd__ unsigned long long int reinterpret<double, unsigned long long int>(double x) { return __double_as_longlong(x); }

template<> __fd__ float reinterpret<int, float>(int x) { return __int_as_float(x); }
template<> __fd__ int reinterpret<float, int>(float x) { return __float_as_int(x); }

} // namespace detail

template <typename T>
__fd__ T bitwise_or (T* address, T val)
{
	auto f = [](T x, T y) { return x | y; };
	return apply<T, T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_and (T* address, T val)
{
	auto f = [](T x, T y) { return x & y; };
	return apply<T, T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_xor (T* address, T val)
{
	auto f = [](T x, T y) { return x ^ y; };
	return apply<T, T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_not (T* address)
{
	auto f = [](T x) { return ~x; };
	return apply<T, T>(f, address);
}

template <typename T>
__fd__ T set_bit (T* address, native_word_t bit_index)
{
	auto f = [](T x, native_word_t y) { return x | 1 << y; };
	return apply(f, address, bit_index);
}
template <typename T>
__fd__ T unset_bit (T* address, native_word_t bit_index)
{
	auto f = [](T x, native_word_t y) { return x & ~(1 << y); };
	return apply(f, address, bit_index);
}

// This next #if-#endif block is intended to supply us with 64-bits of some
// atomic primitives only available with compute capability 3.2 or higher
// (which is not currently a requirement. CC 3.0 sort of is.)
#if __CUDA_ARCH__ < 320

template <typename T>
__fd__ T atomicMin (T* address, T val)
{
	auto f = [](T x, T y) { return x < y ? x : y; };
	return apply(f, address, val);
}

template <typename T>
__fd__ T atomicMax (T* address, T val)
{
	auto f = [](T x, T y) { return x > y ? x : y; };
	return apply(f, address, val);
}

template <typename T>
__fd__ T atomicAnd (T* address, T val)
{
	auto f = [](T x, T y) { return x and y; };
	return apply(f, address, val);
}

template <typename T>
__fd__ T atomicOr (T* address, T val)
{
	auto f = [](T x, T y) { return x or y; };
	return apply(f, address, val);
}

template <typename T>
__fd__ T atomicXor (T* address, T val)
{
	auto f = [](T x, T y) { return (x and not y) || (y and not x); };
	return apply(f, address, val);
}

#else

template <typename T>
__fd__ T atomicMin(T* address, T val)
{
	return ::atomicMin(address, val);
}

template <typename T>
__fd__ T atomicMax(T* address, T val)
{
	return ::atomicMax(address, val);
}

template<> __fd__ unsigned long atomicMin<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long val)
{
	return ::atomicMin(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

template<> __fd__ unsigned long atomicMax<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long val)
{
	return ::atomicMax(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

#endif /* __CUDA_ARCH__ >= 320 */

namespace detail {

template <typename T>
__fd__ T increment_impl  (
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
__fd__ T increment_impl  (
	std::false_type,
	T*  address,
	T   wraparound_value)
{
	auto do_increment = [](T x, T wraparound) { return x >= wraparound ? T{0} : T{x+1}; };
	return atomic::apply(do_increment, address, wraparound_value);
}

} // namespace detail

template <typename T>
__fd__ T increment  (
	T*  address,
	T   wraparound_value)
{
	constexpr bool can_act_directly = (sizeof(T) == sizeof(unsigned int));

	return detail::increment_impl<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, wraparound_value);
}

namespace detail {

template <typename T>
__fd__ T decrement_impl (
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
__fd__ T decrement_impl (
	std::false_type,
	T*  address,
	T   wraparound_value)
{
	auto do_decrement = [](auto x, auto wraparound) { return ((x <= 0) or (x >= wraparound)) ? wraparound - 1 : x - 1; };
	return apply(do_decrement, address, wraparound_value);
}

} // namespace detail

template <typename T>
__fd__ T decrement (
	T*  address,
	T   wraparound_value)
{
	constexpr bool can_act_directly = (sizeof(T) == sizeof(unsigned int));

	return detail::decrement_impl<T>(
		std::integral_constant<bool, can_act_directly>{},
		address, wraparound_value);
}

namespace detail {

template <typename T>
__fd__ T add_impl  (
	std::true_type,
	T*  address,
	T   val)
{
	static_assert(
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
		,
		"invalid type");
	printf("add_impl - can act directly\n");
	return ::atomicAdd(address, val);
}

template <typename T>
__fd__ T add_impl  (
	std::false_type, // can't atomic-add directly, need to use compare-and-swap
	T*  address,
	T   val)
{
	printf("add_impl - can't act directly\n");
	auto do_addition = [](T x, T y) { return static_cast<T>(x + y); };
	return kat::atomic::apply(do_addition, address, val);
}

// TODO: Double-check (no pun intended) that pre-Pascal cards, or Pascal-and-onwards
// cards running CC 3.0 code, handle doubles properly - I have my doubts.

} // namespace detail

template <typename T>
__fd__ T add (
	T*  address,
	T   val)
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
	return detail::add_impl<T>(
		std::integral_constant<bool, can_act_directly>{},
		//std::integral_constant<bool, false>{},
		address, val);
}

template <typename T>  __fd__ T subtract   (T* address, T val)  { return atomicSub (address, val); }
template <typename T>  __fd__ T exchange   (T* address, T val)  { return atomicExch(address, val); }
template <typename T>  __fd__ T min        (T* address, T val)  { return atomicMin (address, val); }
template <typename T>  __fd__ T max        (T* address, T val)  { return atomicMax (address, val); }
template <typename T>  __fd__ T logical_and(T* address, T val)  { return atomicAnd (address, val); }
template <typename T>  __fd__ T logical_or (T* address, T val)  { return atomicOr  (address, val); }
template <typename T>  __fd__ T logical_xor(T* address, T val)  { return atomicXor (address, val); }

} // namespace atomic
} // namespace kat



///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

///@endcond

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_DETAIL_CUH_
