/**
 * @file on_device/math.cuh
 *
 * @brief Mathematical function definitions for our troubled world
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_MATH_CUH_
#define CUDA_KAT_ON_DEVICE_MATH_CUH_

// Though this is an include file, it should not be visible outside
// the op store's implementation
#include "common.cuh"
#include "constexpr_math.cuh"
#include <kat/on_device/wrappers/builtins.cuh>
	// for absolute_value(), sum_of_absolute_differences(), minimum(), maximum() etc...


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

#include <type_traits>

namespace kat {

/**
 * @brief compute the base-two logarithm of a number known to be a power of 2.
 *
 * @note Yes, this is trivial to do, but:
 *   1. This says _what_ you're doing, not _how_ you do it (e.g. left-shifting
 *      bits and such)
 *   2. There's a device-side optimization here (which isn't constexpr)
 *
 * @param p an integral power of 2
 * @return the exponent l such than 2^l equals p
 */
template <typename I>
__fd__ unsigned log2_of_power_of_2(I p)
{
	static_assert(std::is_integral<I>::value, "Only supported for integers");
	// Remember 0 is _not_ a power of 2.
	return  builtins::population_count(p - 1);
}

/**
 * A variant of `div_rounding_up` (which you can find in `constexpr_math.cuh`),
 * which has (non-constexpr, unfortunately) optimizations based on the knowledge
 * the divisor is a power of 2
 *
 * @return The smallest multiple of divisor above dividend / divisor
 */
template <typename T, typename S>
__fd__ T div_by_power_of_2_rounding_up(const T& dividend, const S& divisor)
{
	auto mask = divisor - 1; // Remember: 0 is _not_ a power of 2
	auto log_2_of_divisor = log2_of_power_of_2(divisor);
	auto correction_for_rounding_up = ((dividend & mask) + mask) >> log_2_of_divisor;

	return (dividend >> log_2_of_divisor) + correction_for_rounding_up;
}


template <typename I, typename P>
constexpr __fd__ I div_by_power_of_2(I dividend, P power_of_2)
{
	return dividend >> log2_of_power_of_2(power_of_2);
}



#if __cplusplus < 201402L
/**
 * @brief compute the greatest common divisor (gcd) of two values.
 *
 * @param u One integral value (prefer making this the larger one)
 * @param v Another integral value (prefer making this the smaller one)
 * @return the largest I value d such that d divides @p u and d divides @p v.
 */
template <typename T>
constexpr __fd__ T gcd(T u, T v)
{
	static_assert(std::is_integral<I>::value, "Only supported for integers");
	while (v != 0) {
		T r = u % v;
		u = v;
		v = r;
	}
	return u;
}
// ... and for C++14, this is a constexpr_ implementation, and we don't need to redo it here
#endif

/**
 * @brief compute the least common multiple (LCM) of two integer values
 *
 * @tparam I an integral (or integral-number-like) type
 *
 * @param u One of the numbers which the result must divide
 * @param v Another one of the numbers which the result must divide
 * @return The highest I value which divides both @p u and @p v.
 */
template <typename I>
__fd__ I lcm(I u, I v)
{
	static_assert(std::is_integral<I>::value, "Only supported for integers at the moment");
	return (u / gcd(u,v)) * v;
}

/**
 * @brief compute the (integral) base-two logarithm of a number
 *
 * @note Yes, this is trivial to do, but:
 *   1. This says _what_ you're doing, not _how_ you do it (e.g. left-shifting
 *      bits and such)
 *   2. There's a device-side optimization here (which isn't constexpr)
 *
 * @param x a non-negative value
 * @return floor(log2(x)), i.e. the least exponent l such than 2^l >= x
 */
template <typename I>
__fd__ unsigned log2(I x) {
	assert(x > 0);
	return (CHAR_BIT * sizeof(I) - 1) - builtins::count_leading_zeros(x);
}

} // namespace kat

///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_MATH_CUH_
