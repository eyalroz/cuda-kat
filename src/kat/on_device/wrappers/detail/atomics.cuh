/**
 * @file Wrappers for CUDA atomic operations - implementation details
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
#ifndef CUDA_ON_DEVICE_ATOMICS_DETAIL_CUH_
#define CUDA_ON_DEVICE_ATOMICS_DETAIL_CUH_

#include <cuda/api/constants.hpp>
#include <device_atomic_functions.h>

#include <functional>
#include <type_traits>

#include <kat/define_specifiers.hpp>

// Annoyingly, CUDA - upto and including version 10.0 - provide atomic
// operation wrappers for unsigned int and unsigned long long int, but
// not for the in-between type of unsigned long int. So - we
// have to make our own. Don't worry about the condition checks -
// they are optimized away at compile time and the result is basically
// a single PTX instruction
__fd__ unsigned long int atomicAdd(unsigned long int *address, unsigned long int val)
{
    if (sizeof (unsigned long int) == sizeof(unsigned long long int)) {
        return atomicAdd(reinterpret_cast<unsigned long long int*>(address), val);
    }
    else if (sizeof (unsigned long int) == sizeof(unsigned int)) {
        return  atomicAdd(reinterpret_cast<unsigned int*>(address), val);
    }
    else return 0;
}

namespace atomic {

template <typename T>  __fd__ T compare_and_swap(
	typename std::enable_if<
		sizeof(T) == sizeof(int) or sizeof(T) == sizeof(long long int), T
	>::type	* __restrict__ address,
	const T& compare,
	const T& val)
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

// The default (which should be 32-or-less-bit types
template <typename T, typename = void>
struct add_impl {
	__fd__ T operator()(T*  __restrict__ address, const T& val) const
	{
		return atomicAdd(address, val);
	}
};

template <typename T>
struct add_impl<T,
	typename std::enable_if<
		!std::is_same<T, unsigned long long int>::value &&
		sizeof(T) == sizeof(unsigned long long int)
	>::type> {
	using surrogate_t = unsigned long long int;

	__fd__ T operator()(T*  __restrict__ address, const T& val) const
	{
		auto address_ = reinterpret_cast<surrogate_t*>(address);

		// TODO: Use apply_atomically

		surrogate_t previous_ = *address_;
		surrogate_t expected_previous_;
		do {
			expected_previous_ = previous_;
			T updated_value = reinterpret<surrogate_t, T>(previous_) + val;
			previous_ = atomicCAS(address_, expected_previous_,
				reinterpret<T, surrogate_t>(updated_value));
		} while (expected_previous_ != previous_);
		T rv = reinterpret<surrogate_t, T>(previous_);
		return rv;
	}
};

} // namespace detail

template <typename T>
__fd__ T add(T* __restrict__ address, const T& val)
{
	return detail::add_impl<T>()(address, val);
}


// TODO: Consider making apply_atomically take functors,
// including the functors header and having using statements here
// instead of actual definitions
template <typename T>
__fd__ T bitwise_or (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x | y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T bitwise_and (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x & y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T bitwise_xor (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x ^ y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T bitwise_not (T* __restrict__ address)
{
	auto f = [](const T& x) { return ~x; };
	return apply_atomically(f, address);
}

template <typename T>
__fd__ T set_bit (T* __restrict__ address, const unsigned bit_index)
{
	auto f = [](const T& x, const unsigned y) { return x | 1 << y; };
	return apply_atomically(f, address, bit_index);
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
__fd__ T atomicMin (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x < y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicMax (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x > y ? x : y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicAnd (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x && y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicOr (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return x || y; };
	return apply_atomically(f, address, val);
}

template <typename T>
__fd__ T atomicXor (T* __restrict__ address, const T& val)
{
	auto f = [](const T& x, const T& y) { return (x && !y) || (y && !x); };
	return apply_atomically(f, address, val);
}

#else

template <typename T>
__fd__ T atomicMin(
	T* __restrict__ address, const T& val)
{
	return ::atomicMin(address, val);
}

template <typename T>
__fd__ T atomicMax(
	T* __restrict__ address, const T& val)
{
	return ::atomicMax(address, val);
}

template<> __fd__ unsigned long atomicMin<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long& val)
{
	return ::atomicMin(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

template<> __fd__ unsigned long atomicMax<unsigned long>(
	unsigned long* __restrict__ address, const unsigned long& val)
{
	return ::atomicMax(
		reinterpret_cast<unsigned long long*>(address),
		reinterpret_cast<const unsigned long long&>(val)
	);
}

#endif /* __CUDA_ARCH__ >= 320 */

template <typename T>
__fd__ T increment  (
	T* __restrict__  address,
	const T&         wraparound_value)
{
	return atomicInc(address, wraparound_value);
}

template <typename T>  __fd__ T decrement  (
	T* __restrict__  address,
	const T&         wraparound_value )
{
	return atomicDec(address, wraparound_value);
}

template <typename T>  __fd__ T subtract   (T* __restrict__ address, const T& val)  { return atomicSub(address, val);  }
template <typename T>  __fd__ T exchange   (T* __restrict__ address, const T& val)  { return atomicExch(address, val); }
template <typename T>  __fd__ T min        (T* __restrict__ address, const T& val)  { return atomicMin(address, val);  }
template <typename T>  __fd__ T max        (T* __restrict__ address, const T& val)  { return atomicMax(address, val);  }
template <typename T>  __fd__ T logical_and(T* __restrict__ address, const T& val)  { return atomicAnd(address, val);  }
template <typename T>  __fd__ T logical_or (T* __restrict__ address, const T& val)  { return atomicOr(address, val);   }
template <typename T>  __fd__ T logical_xor(T* __restrict__ address, const T& val)  { return atomicXor(address, val);  }

} // namespace atomic

#include <kat/undefine_specifiers.hpp>

#endif /* CUDA_ON_DEVICE_ATOMICS_DETAIL_CUH_ */
