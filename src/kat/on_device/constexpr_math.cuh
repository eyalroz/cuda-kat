/**
 * @file on_device/constexpr_math.cuh
 *
 * @brief mathematical functions (mostly super-simple ones) implemented using
 * compile-time-executable code. Some of these are also the reasonable runtime-
 * executable version, some aren't; the former appear outside of any namespace,
 * the latter have their own namespace (although maybe they shouldn't)
 */
#pragma once
#ifndef CUDA_ON_DEVICE_CONSTEXPR_MATH_CUH_
#define CUDA_ON_DEVICE_CONSTEXPR_MATH_CUH_

#include "common.cuh" // for warp_size

#include <type_traits>

#include <kat/define_specifiers.hpp>

template <typename T>
__fhd__ constexpr bool is_power_of_2(T val) { return (val & (val-1)) == 0; }
	// Yes, this works: Only if val had exactly one 1 bit will subtracting 1 switch
	// all of its 1 bits.


template <typename T>
constexpr inline T& modular_inc(T& x, T modulus) { return (x+1) % modulus; }

template <typename T>
constexpr inline T& modular_dec(T& x, T modulus) { return (x-1) % modulus; }

namespace detail {

template <typename T>
constexpr T ipow(T base, unsigned exponent, T coefficient) {
	return exponent == 0 ? coefficient :
		ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
}

} // namespace detail

template <typename I>
constexpr I ipow(I base, unsigned exponent)
{
	return detail::ipow(base, exponent, 1);
}

template <typename I, typename I2>
__fhd__ constexpr I div_rounding_up_unsafe(I dividend, const I2 divisor)
{
	return (dividend + divisor - 1) / divisor;
}

template <typename I, typename I2>
__fhd__ constexpr I div_rounding_up_safe(I dividend, const I2 divisor)
{
	return (dividend / divisor) + !!(dividend % divisor);
}

template <typename I, typename I2>
__fhd__ constexpr I round_down(const I x, const I2 y)
{
	return x - x%y;
}

/**
 * @note Don't use this with negative values.
 */
template <typename I>
__fhd__ constexpr I round_down_to_warp_size(I x)
{
	return x & ~(warp_size - 1);
}

/**
 * @note implemented in an unsafe way - will overflow for values close
 * to the maximum
 */
template <typename I, typename I2 = I>
__fhd__ constexpr I round_up_unsafe(I x, I2 y)
{
	return round_down(x+y-1, y);
}

template <typename I, typename I2 = I>
__fhd__ constexpr I round_down_to_power_of_2(I x, I2 power_of_2)
{
	return (x & ~(I{power_of_2} - 1));
}

/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
__fhd__ constexpr I round_up_to_power_of_2_unsafe(I x, I2 power_of_2)
{
	return round_down_to_power_of_2 (x + I{power_of_2} - 1, power_of_2);
}

/**
 * @note careful, this may overflow!
 */
template <typename I>
__fhd__ constexpr I round_up_to_full_warps_unsafe(I x) {
	return round_up_to_power_of_2_unsafe<I, native_word_t>(x, warp_size);
}

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool between_or_equal(const T& x, const Lower& l, const Upper& u) { return (l <= x) && (x <= u); }

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool strictly_between(const T& x, const Lower& l, const Upper& u) { return (l < x) && (x < u); }

#if __cplusplus >= 201402L
template <typename T>
constexpr __fd__ T gcd(T u, T v)
{
    while (v != 0) {
        T r = u % v;
        u = v;
        v = r;
    }
    return u;
}
#endif

namespace constexpr_ {

template <typename I>
constexpr __fhd__ int log2(I val)
{
	return val ? 1 + log2(val >> 1) : -1;
}

template <typename I, I Divisor>
__fhd__ constexpr I div_by_fixed_power_of_2(I dividend)
{
	return dividend >> log2(Divisor);
}

template <typename S, typename T = S>
__fhd__ constexpr typename std::common_type<S,T>::type gcd(S u, T v)
{
	return (v == 0) ? u : gcd(v, u % v);
}

template <typename S, typename T = S>
__fhd__ constexpr typename std::common_type<S,T>::type lcm(S u, T v)
{
	using result_type = typename std::common_type<S,T>::type;
	return ((result_type) u / gcd(u,v)) * v;
}


namespace detail {
template <typename T>
__fhd__ constexpr T sqrt_helper(T x, T low, T high)
{
	// this ugly macro cant be replaced by a lambda
	// or the use of temporary variable, as in C++11, a constexpr
	// function must have a single statement
#define sqrt_HELPER_MID ((low + high + 1) / 2)
	return low == high ?
		low :
		((x / sqrt_HELPER_MID < sqrt_HELPER_MID) ?
			sqrt_helper(x, low, sqrt_HELPER_MID - 1) :
			sqrt_helper(x, sqrt_HELPER_MID, high));
#undef sqrt_HELPER_MID
}

} // namespace detail

template <typename T>
__fhd__ constexpr T sqrt(T& x)
{
  return detail::sqrt_helper(x, 0, x / 2 + 1);
}

} // namespace constexpr_


template <typename I>
__fhd__ constexpr I div_by_power_of_2(I dividend, I divisor)
{
	return dividend >> log2_of_power_of_2(divisor);
}

template <typename I, I Divisor>
__fhd__ constexpr I div_by_fixed_power_of_2_rounding_up(I dividend)
{
/*
	// C++14 and later:

	constexpr auto log_2_of_divisor = constexpr_::log2(Divisor);
	constexpr auto mask = Divisor - 1;
	auto correction_for_rounding_up = ((dividend & mask) + mask) >> log_2_of_divisor;

	return (dividend >> log_2_of_divisor) + correction_for_rounding_up;
*/
	// single-statement C++11 version
	return (dividend >> constexpr_::log2(Divisor)) +
		(((dividend & (Divisor - 1)) + (Divisor - 1)) >> constexpr_::log2(Divisor));
}

template <typename I>
__fhd__ constexpr I num_warp_sizes_to_cover(I x)
{
	return constexpr_::div_by_fixed_power_of_2<I, warp_size>(x) + ((x & (warp_size-1)) > 0);
}


template <typename I>
constexpr __fhd__ bool divides(I divisor, I dividend) {
	return dividend % divisor == 0;
}

template <typename I>
constexpr inline int is_divisible_by(I dividend, I divisor) {
	return divides(divisor, dividend);
}

template <typename I>
constexpr inline bool is_divisible_by_power_of_2(I dividend, I divisor) {
	return divisor & (dividend-1) == 0;
}


template <typename I>
constexpr __fhd__ bool is_odd(I x)  { return x & (I) 0x1 != 0; }
template <typename I>
constexpr __fhd__ bool is_even(I x) { return x & (I) 0x1 == 0; }


#include <kat/undefine_specifiers.hpp>

#endif /* CUDA_ON_DEVICE_CONSTEXPR_MATH_CUH_ */
