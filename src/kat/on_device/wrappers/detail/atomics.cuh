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

#include <kat/define_specifiers.hpp>

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

template <unsigned NBytes> struct uint;

template<> struct uint<1> { using type = uint8_t;  };
template<> struct uint<2> { using type = uint16_t; };
template<> struct uint<4> { using type = uint32_t; };
template<> struct uint<8> { using type = uint64_t; };

template <typename T1, typename T2>
T1 replace_bytes_at_offset(T1 larger_value ,T2 replacement, std::size_t offset_into_larger_value)
{
    static_assert(sizeof(T1) > sizeof(T2), "invalid sizes");
    static_assert(std::is_trivial<T1>::value, "T1 must be a trivial type");
    static_assert(std::is_trivial<T2>::value, "T2 must be a trivial type");
    auto shift_amount = offset_into_larger_value * CHAR_BIT;
    using larger_uint_type      = uint<sizeof(larger_value)>;
    using replacement_uint_type = uint<sizeof(replacement)>;
    auto&       v1_as_uint = *reinterpret_cast<larger_uint_type*>(&larger_value);
    const auto& v2_as_uint = *reinterpret_cast<replacement_uint_type*>(&replacement);

    auto mask = ~( (larger_uint_type{1} << (sizeof(T2) * CHAR_BIT) - 1) << shift_amount);
    auto shifted_replacement = larger_uint_type{v2_as_uint} <<  shift_amount;
    return (larger_value & mask ) | shifted_replacement;
}


template <typename T>
__fd__ T compare_and_swap(std::true_type, T* __restrict__ address, const T& compare, const T& val)
{
	// This switch is necessary due to atomicCAS being defined
	// only for a very small selection of types - int and unsigned long long
	switch(sizeof(T)) {
	case sizeof(int):
			return (T) (atomicCAS(
				reinterpret_cast<int*      >(address),
				reinterpret_cast<const int&>(compare),
				reinterpret_cast<const int&>(val)
			));
	case sizeof(long long int):
		return (T) (atomicCAS(
				reinterpret_cast<unsigned long long*      >(address),
				reinterpret_cast<const unsigned long long&>(compare),
				reinterpret_cast<const unsigned long long&>(val)
			));
	default: return T(); // should not be able to get here
	}
}

template <typename T>
__fd__ T compare_and_swap(std::false_type, T* __restrict__ address, const T& compare, const T& val)
{
	// The idea is to apply compare-and-swap on more memory than the T type actually takes
	// up. However, we can't just have that larger stretch of memory start at `address`, since
	// NVIDIA GPUs require operands to machine instructions to be naturally-aligned (e.g.
	// a 4-byte-sized operand must start at an address being a multiple of 4). That means we
	// could theoretically have "slack" bytes we use in the compare-and-swap both before and
	// after the value we're actually interested in. We'll have to "contend" for atomic access
	// to both the proper bytes of interest and the slack bytes, together.

	static_assert (sizeof(T) <= sizeof(long long int), "Type size too large for atomic compare-and-swap");
	using cas_builtin_type = typename std::conditional<sizeof(T) < sizeof(int), int, unsigned long long int>::type;

	auto address_for_builtin = kat::detail::align_down(address);
	auto address_difference = raw_address_difference(address, address_for_builtin);
	cas_builtin_type actual_value_at_addr = *address_for_builtin;
	cas_builtin_type expected_value_at_addr;
	do {
		auto expected_value_at_addr { actual_value_at_addr };
		auto larger_value_to_set { replace_bytes_at_offset(expected_value_at_addr, val, address_difference) };
		actual_value_at_addr = atomic::compare_and_swap<cas_builtin_type>(
			address_for_builtin, expected_value_at_addr, larger_value_to_set);
	} while (actual_value_at_addr != expected_value_at_addr);
	return reinterpret_cast<const T&>(reinterpret_cast<char*>(&actual_value_at_addr) + address_difference);
}

} // namespace detail

template <typename T>  __fd__ T compare_and_swap(
    T* __restrict__  address,
    const T&         compare,
    const T&         val)
{
	static_assert(sizeof(T) <= sizeof(long long int),
		"nVIDIA GPUs do not support atomic operations on data larger than long long int");

	constexpr bool cuda_has_builtin_with_appropriate_size =
		sizeof(T) == sizeof(int) or sizeof(T) == sizeof(unsigned long long int);

	return detail::compare_and_swap<T>(
		std::integral_constant<bool, cuda_has_builtin_with_appropriate_size>{},
		address, compare, val);
}


/**
 * Use atomic compare-and-swap to apply a unary function to some value,
 * replacing it at its memory location with the result before anything
 * else changes it.
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename UnaryFunction>
__fd__ T apply_atomically(UnaryFunction f, T* __restrict__ address)
{
	auto actual_previous_value = *address;
	T expected_previous_value;
	do {
		expected_previous_value = actual_previous_value;
		T prospective_new_value = f(expected_previous_value);
		actual_previous_value = atomic::compare_and_swap(address, expected_previous_value,
			prospective_new_value);
	} while (actual_previous_value != expected_previous_value);
	return actual_previous_value;
}

/**
 * Use atomic compare-and-swap to apply a binary function to two values,
 * replacing the first at its memory location with the result before anything
 * else changes it.
 *
 * @todo Take a template parameter pack and multiple arguments to f
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename BinaryFunction>
__fd__ T apply_atomically(
	BinaryFunction   f,
	T* __restrict__  address,
	const T&         rhs)
{
	auto uf = [&](const T& lhs) { return f(lhs, rhs); };
	return apply_atomically(uf, address);
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

// TODO: Consider making apply_atomically take functors,
// including the functors header and having using statements here
// instead of actual definitions
template <typename T>
__fd__ T bitwise_or (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x | y; };
	return apply_atomically<T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_and (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x & y; };
	return apply_atomically<T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_xor (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x ^ y; };
	return apply_atomically<T>(f, address, val);
}

template <typename T>
__fd__ T bitwise_not (T* __restrict__ address)
{
	auto f = [](const T& x) { return ~x; };
	return apply_atomically<T>(f, address);
}

template <typename T>
__fd__ T set_bit (T* __restrict__ address, const unsigned bit_index)
{
	auto f = [](const T& x, const unsigned y) { return x | 1 << y; };
	return apply_atomically<T>(f, address, bit_index);
}
template <typename T>
__fd__ T unset_bit (T* __restrict__ address, const unsigned bit_index)
{
	auto f = [](const T& x, const unsigned y) { return x & ~(1 << y); };
	return apply_atomically(f, address, bit_index);
}


// This next #if-#endif block is intended to supply us with 64-bits of some
// atomic primitives only available with compute capability 3.2 or higher
// (which is not currently a requirement. CC 3.0 sort of is.)
#if __CUDA_ARCH__ < 320

template <typename T>
__fd__ T atomicMin (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x < y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicMax (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x > y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicAnd (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x && y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicOr (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return x || y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicXor (T* __restrict__ address, T val)
{
	auto f = [](const T& x, const T& y) { return (x && !y) || (y && !x); };
	return apply_atomically(f, address, val);
}

#else

template <typename T>
__fd__ T atomicMin(T* __restrict__ address, T val)
{
	return ::atomicMin(address, val);
}

template <typename T>
__fd__ T atomicMax(T* __restrict__ address, T val)
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
	T* __restrict__  address,
	T                wraparound_value)
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
	T* __restrict__  address,
	T                wraparound_value)
{
	auto do_increment = [&](auto x) { return x + wraparound_value;};
	return apply_atomically(do_increment, address);
}

} // namespace detail

template <typename T>
__fd__ T increment  (
	T* __restrict__  address,
	T                wraparound_value)
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
	T* __restrict__  address,
	T                wraparound_value)
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
	T* __restrict__  address,
	T                wraparound_value)
{
	auto do_decrement = [&](auto x) { return x - wraparound_value;};
	return apply_atomically(do_decrement, address);
}

} // namespace detail

template <typename T>
__fd__ T decrement (
	T* __restrict__  address,
	T                wraparound_value)
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
	T* __restrict__  address,
	T                val)
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
	return ::atomicAdd(address, val);
}

template <typename T>
__fd__ T add_impl  (
	std::false_type,
	T* __restrict__  address,
	T                val)
{
	auto do_addition = [](T x, T val) { return x + val; };
	return apply_atomically(do_addition, address, val);
}

// TODO: Double-check (no pun intended) that pre-Pascal cards, or Pascal-and-onwards
// cards running CC 3.0 code, handle doubles properly - I have my doubts.

} // namespace detail

template <typename T>
__fd__ T add (
	T* __restrict__  address,
	T                val)
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


template <typename T>  __fd__ T subtract   (T* __restrict__ address, T val)  { return atomicSub(address, val);  }
template <typename T>  __fd__ T exchange   (T* __restrict__ address, T val)  { return atomicExch(address, val); }
template <typename T>  __fd__ T min        (T* __restrict__ address, T val)  { return atomicMin(address, val);  }
template <typename T>  __fd__ T max        (T* __restrict__ address, T val)  { return atomicMax(address, val);  }
template <typename T>  __fd__ T logical_and(T* __restrict__ address, T val)  { return atomicAnd(address, val);  }
template <typename T>  __fd__ T logical_or (T* __restrict__ address, T val)  { return atomicOr(address, val);   }
template <typename T>  __fd__ T logical_xor(T* __restrict__ address, T val)  { return atomicXor(address, val);  }

} // namespace atomic
} // namespace kat

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_DETAIL_CUH_
