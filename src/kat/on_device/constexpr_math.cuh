/**
 * @file on_device/constexpr_math.cuh
 *
 * @brief mathematical functions (mostly super-simple ones) implemented using
 * compile-time-executable code. Some of these are also the reasonable runtime-
 * executable version, some aren't; the former appear outside of any namespace,
 * the latter have their own namespace (although maybe they shouldn't)
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_CONSTEXPR_MATH_CUH_
#define CUDA_KAT_ON_DEVICE_CONSTEXPR_MATH_CUH_

#include "common.cuh" // for warp_size

#include <cassert>
#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

template <typename T, typename Lower = T, typename Upper = T>
constexpr KAT_FHD bool between_or_equal(T x, Lower l, Upper u) noexcept { return (l <= x) && (x <= u); }

template <typename T, typename Lower = T, typename Upper = T>
constexpr KAT_FHD bool strictly_between(T x, Lower l, Upper u) noexcept { return (l < x) && (x < u); }


template <typename T>
constexpr KAT_FHD bool is_power_of_2(T val) noexcept { return (val & (val-1)) == 0; }
	// Yes, this works: Only if val had exactly one 1 bit will subtracting 1 switch
	// all of its 1 bits.

namespace detail {

template <typename T>
constexpr KAT_FHD T ipow(T base, unsigned exponent, T coefficient) noexcept {
	return exponent == 0 ? coefficient :
		ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
}

} // namespace detail

// #if __cplusplus >= 201402L
template <typename I>
constexpr KAT_FHD I ipow(I base, unsigned exponent) noexcept
{
	return detail::ipow(base, exponent, I{1});
}

template <typename I, typename I2>
constexpr KAT_FHD I div_rounding_up(I x, const I2 modulus) noexcept
{
	return (x / modulus) + !!(x % modulus);
}

template <typename I, typename I2>
constexpr KAT_FHD I round_down(const I x, const I2 modulus) noexcept
{
	return x - (x % modulus);
}

template <typename I, typename I2 = I>
constexpr KAT_FHD I round_down_to_power_of_2(I x, I2 power_of_2) noexcept
{
	return (x & ~(I{power_of_2} - 1));
}

/**
 * @note Don't use this with negative values.
 */
template <typename I>
constexpr KAT_FHD I round_down_to_full_warps(I x) noexcept
{
	return x & ~(warp_size - 1);
}

template <typename I, typename I2 = I>
constexpr KAT_FHD I round_up(I x, I2 y) noexcept
{
	return (x % y == 0) ? x : x + (y - x%y);
}

/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
constexpr KAT_FHD I round_up_to_power_of_2(I x, I2 power_of_2) noexcept
{
	return ((x & (power_of_2 - 1)) == 0) ? x : ((x & ~(power_of_2 - 1)) + power_of_2);
}


template <typename I>
constexpr KAT_FHD I round_up_to_full_warps(I x) noexcept {
	return round_up_to_power_of_2<I, native_word_t>(x, warp_size);
}


#if __cplusplus >= 201402L
template <typename S, typename T = S>
constexpr KAT_FHD typename std::common_type<S,T>::type gcd(S u, T v) noexcept
{
	// TODO: Shouldn't we first cast everything into the common type?
	while (v != 0) {
		auto remainder = u % v;
		u = v;
		v = remainder;
	}
	return u;
}
#else
template <typename S, typename T = S>
constexpr KAT_FHD typename std::common_type<S,T>::type gcd(S u, T v) noexcept
{
	return (v == 0) ? u : gcd<std::common_type<S,T>::type>(v, u % v);
}
#endif

template <typename S, typename T = S>
constexpr KAT_FHD typename std::common_type<S,T>::type lcm(S u, T v) noexcept
{
	using result_type = typename std::common_type<S,T>::type;
	return (result_type{u} / gcd(u,v)) * v;
}


template <typename I>
constexpr KAT_FHD bool divides(I non_zero_divisor, I dividend)  noexcept
{
#if __cplusplus >= 201402L
    assert(non_zero_divisor != 0);
#endif
	return (dividend % non_zero_divisor) == 0;
}

template <typename I>
constexpr KAT_FHD bool is_divisible_by(I dividend, I non_zero_divisor) noexcept
{
#if __cplusplus >= 201402L
    assert(non_zero_divisor != 0);
#endif
	return divides(non_zero_divisor, dividend);
}

template <typename I>
constexpr KAT_FHD I modulo_power_of_2(I x, I power_of_2_modulus) noexcept
{
#if __cplusplus >= 201402L
    assert(is_power_of_2(power_of_2_modulus));
#endif
	return x & (power_of_2_modulus - I{1});
}

template <typename I>
constexpr KAT_FHD bool power_of_2_divides(I power_of_2_divisor, I dividend) noexcept
{
#if __cplusplus >= 201402L
    assert(is_power_of_2(power_of_2_divisor));
#endif
	return modulo_power_of_2(dividend, power_of_2_divisor) == 0;
}

template <typename I>
constexpr KAT_FHD bool is_divisible_by_power_of_2(I dividend, I power_of_2_divisor) noexcept
{
#if __cplusplus >= 201402L
    assert(is_power_of_2(power_of_2_divisor));
#endif
	return power_of_2_divides(power_of_2_divisor, dividend);
}


template <typename I> constexpr KAT_FHD bool is_even(I x) noexcept { return power_of_2_divides(I{2}, x);     }
template <typename I> constexpr KAT_FHD bool is_odd (I x) noexcept { return not power_of_2_divides(I{2}, x); }

namespace detail {

// Assumes 0 <= x < modulus
template <typename I>
constexpr KAT_FHD I increment_modular_remainder(I modular_remainder, I modulus) noexcept
{
	return (modular_remainder == modulus - I{1}) ? I{0} : (modular_remainder + I{1});
}

// Assumes 0 <= x < modulus
template <typename I>
constexpr KAT_FHD I decrement_modular_remainder(I modular_remainder, I modulus) noexcept
{
	return (modular_remainder == 0) ? (modulus - I{1}) : (modular_remainder - I{1});
}

} // namespace detail

// Note: Safe but slow
template <typename I>
constexpr KAT_FHD I modular_increment(I x, I modulus) { return detail::increment_modular_remainder<I>(x % modulus, modulus); }

// Note: Safe but slow
template <typename I>
constexpr KAT_FHD I modular_decrement(I x, I modulus) { return detail::decrement_modular_remainder<I>(x % modulus, modulus); }



/**
 * Faster implementations of mathematical functions which can be incorrect for extremal or near-extremal values.
 */
namespace unsafe {

/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
constexpr KAT_FHD I round_up_to_power_of_2(I x, I2 power_of_2) noexcept
{
	return round_down_to_power_of_2 (x + I{power_of_2} - 1, power_of_2);
}

/**
 * @note careful, this may overflow!
 */
template <typename I>
constexpr KAT_FHD I round_up_to_full_warps(I x) noexcept {
	return unsafe::round_up_to_power_of_2<I, native_word_t>(x, warp_size);
}

/**
 * @note Will overflow when @p x is within @p modulus - 1 of the maximum
 * value of I1
 */
template <typename I1, typename I2>
constexpr KAT_FHD I1 div_rounding_up(I1 x, const I2 modulus) noexcept
{
	return ( x + I1{modulus} - I1{1} ) / I1{modulus};
}

/**
 * @note Will overflow when @p x is within @p y - 1 of the maximum
 * value of I1
 */
template <typename I1, typename I2 = I1>
constexpr KAT_FHD I1 round_up(I1 x, I2 y) noexcept
{
	return round_down(x + I1{y} - I1{1}, y);
}

template <typename I> constexpr inline I modular_increment(I x, I modulus) { return (x + I{1}) % modulus; }
template <typename I> constexpr inline I modular_decrement(I x, I modulus) { return (x + modulus - I{1}) % modulus; }


} // namespace unsafe


/**
 * @brief This namespace has functions whose constexpr (compile-time) implementation should _not_ be used at run-time
 */
namespace constexpr_ {

using kat::between_or_equal;
using kat::strictly_between;
using kat::is_power_of_2;
using kat::ipow;
using kat::div_rounding_up;
using kat::round_down;
using kat::round_down_to_power_of_2;
using kat::round_down_to_full_warps;
using kat::round_up;
using kat::round_up_to_power_of_2;
using kat::round_up_to_full_warps;
using kat::gcd;
using kat::lcm;
using kat::divides;
using kat::is_divisible_by;
using kat::modulo_power_of_2;
using kat::power_of_2_divides;
using kat::is_divisible_by_power_of_2;
using kat::is_even;
using kat::is_odd;
using kat::modular_increment;
using kat::modular_decrement;

namespace unsafe {

using kat::unsafe::round_up_to_power_of_2;
using kat::unsafe::round_up_to_full_warps;
using kat::unsafe::div_rounding_up;
using kat::unsafe::round_up;
using kat::unsafe::modular_increment;
using kat::unsafe::modular_decrement;

}  // namespace unsafe

template <typename I>
constexpr KAT_FHD int log2(I val) noexcept
{
	return val ? 1 + log2(val >> 1) : -1;
}

namespace detail {
template <typename T>
constexpr KAT_FHD T sqrt_helper(T x, T low, T high) noexcept
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
constexpr KAT_FHD T sqrt(T x) noexcept
{
	auto initial_high = x / T{2} + T{1};
	return detail::sqrt_helper<typename std::common_type<T, decltype(initial_high)>::type>(x, 0, initial_high);
}

#ifdef __GNUC__

template <typename I> constexpr inline I log2_of_power_of_2(I non_negative_power_of_2) noexcept
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(unsigned long long), "Unexpectedly large type");
#if __cplusplus >= 201402L
    assert(is_power_of_2(non_negative_power_of_2) and non_negative_power_of_2 >= 1);
#endif

	using cast_target_type = typename
		std::conditional<sizeof(I) <= sizeof(unsigned), unsigned,
			typename std::conditional<sizeof(I) <= sizeof(unsigned long),unsigned long, unsigned long long >::type
		>::type;
	return log2_of_power_of_2<cast_target_type>(static_cast<cast_target_type>(non_negative_power_of_2));
}
template <> constexpr inline unsigned           log2_of_power_of_2<unsigned          >(unsigned           non_negative_power_of_2) noexcept { return __builtin_ctz  (non_negative_power_of_2); }
template <> constexpr inline unsigned long      log2_of_power_of_2<unsigned long     >(unsigned long      non_negative_power_of_2) noexcept { return __builtin_ctzl (non_negative_power_of_2); }
template <> constexpr inline unsigned long long log2_of_power_of_2<unsigned long long>(unsigned long long non_negative_power_of_2) noexcept { return __builtin_ctzll(non_negative_power_of_2); }


template <typename I, typename P>
constexpr inline I div_by_power_of_2(I dividend, P power_of_2) noexcept
{
	return dividend >> log2_of_power_of_2(power_of_2);
}

#endif

} // namespace constexpr_

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_CONSTEXPR_MATH_CUH_
