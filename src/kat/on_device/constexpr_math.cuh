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

#include <type_traits>


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace kat {

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool between_or_equal(T x, Lower l, Upper u) noexcept { return (l <= x) && (x <= u); }

template <typename T, typename Lower = T, typename Upper = T>
constexpr inline bool strictly_between(T x, Lower l, Upper u) noexcept { return (l < x) && (x < u); }


template <typename T>
constexpr __fhd__ bool is_power_of_2(T val) noexcept { return (val & (val-1)) == 0; }
	// Yes, this works: Only if val had exactly one 1 bit will subtracting 1 switch
	// all of its 1 bits.

namespace detail {

template <typename T>
constexpr T ipow(T base, unsigned exponent, T coefficient) noexcept {
	return exponent == 0 ? coefficient :
		ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
}

} // namespace detail

// #if __cplusplus >= 201402L
template <typename I>
constexpr I ipow(I base, unsigned exponent) noexcept
{
	return detail::ipow(base, exponent, I{1});
}

template <typename I, typename I2>
constexpr __fhd__ I div_rounding_up_unsafe(I x, const I2 modulus) noexcept
{
	return (x + modulus - 1) / modulus;
}

template <typename I, typename I2>
constexpr __fhd__ I div_rounding_up_safe(I x, const I2 modulus) noexcept
{
	return (x / modulus) + !!(x % modulus);
}

template <typename I, typename I2>
constexpr __fhd__ I round_down(const I x, const I2 modulus) noexcept
{
	return x - (x % modulus);
}

/**
 * @note Don't use this with negative values.
 */
template <typename I>
constexpr __fhd__ I round_down_to_full_warps(I x) noexcept
{
	return x & ~(warp_size - 1);
}

/**
 * @note implemented in an unsafe way - will overflow for values close
 * to the maximum
 */
template <typename I, typename I2 = I>
constexpr __fhd__ I round_up_unsafe(I x, I2 y) noexcept
{
	return round_down(x+y-1, y);
}

template <typename I, typename I2 = I>
constexpr __fhd__ I round_up_safe(I x, I2 y) noexcept
{
	return (x % y == 0) ? x : x + (y - x%y);
}

template <typename I, typename I2 = I>
constexpr __fhd__ I round_down_to_power_of_2(I x, I2 power_of_2) noexcept
{
	return (x & ~(I{power_of_2} - 1));
}

/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
constexpr __fhd__ I round_up_to_power_of_2_unsafe(I x, I2 power_of_2) noexcept
{
	return round_down_to_power_of_2 (x + I{power_of_2} - 1, power_of_2);
}

/**
 * @note careful, this may overflow!
 */
template <typename I, typename I2 = I>
constexpr __fhd__ I round_up_to_power_of_2_safe(I x, I2 power_of_2) noexcept
{
	return ((x & (power_of_2 - 1)) == 0) ? x : ((x & ~(power_of_2 - 1)) + power_of_2);
}


/**
 * @note careful, this may overflow!
 */
template <typename I>
constexpr __fhd__ I round_up_to_full_warps_unsafe(I x) noexcept {
	return round_up_to_power_of_2_unsafe<I, native_word_t>(x, warp_size);
}

template <typename I>
constexpr __fhd__ I round_up_to_full_warps_safe(I x) noexcept {
	return round_up_to_power_of_2_safe<I, native_word_t>(x, warp_size);
}


#if __cplusplus >= 201402L
template <typename T>
constexpr __fhd__ T gcd(T u, T v) noexcept
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

namespace detail {

// Assumes 0 <= x < modulus
template <typename I>
constexpr inline I modular_inc(I x, I modulus) noexcept { return (x == modulus - I{1}) ? I{0} : (x + I{1}); }

// Assumes 0 <= x < modulus
template <typename I>
constexpr inline I modular_dec(I x, I modulus) noexcept { return (x == I{0}) ? (modulus - I{1}) : (x - I{1}); }

} // namespace detail

// Note: Safe but slow
template <typename I>
constexpr inline I modular_inc(I x, I modulus) { return detail::modular_inc<I>(x % modulus, modulus); }

// Note: Safe but slow
template <typename I>
constexpr inline I modular_dec(I x, I modulus) { return detail::modular_dec<I>(x % modulus, modulus); }


template <typename I>
constexpr __fhd__ int log2(I val) noexcept
{
	return val ? 1 + log2(val >> 1) : -1;
}

template <typename S, typename T = S>
constexpr __fhd__ typename std::common_type<S,T>::type gcd(S u, T v) noexcept
{
	return (v == 0) ? u : gcd(v, u % v);
}

template <typename S, typename T = S>
constexpr __fhd__ typename std::common_type<S,T>::type lcm(S u, T v) noexcept
{
	using result_type = typename std::common_type<S,T>::type;
	return ((result_type) u / gcd(u,v)) * v;
}


namespace detail {
template <typename T>
constexpr __fhd__ T sqrt_helper(T x, T low, T high) noexcept
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
constexpr __fhd__ T sqrt(T x) noexcept
{
	auto initial_high = x / T{2} + T{1};
	return detail::sqrt_helper<typename std::common_type<T, decltype(initial_high)>::type>(x, 0, initial_high);
}

// TODO: Need to implement the following for non-GNU-C-compatible compilers - without using the ctz builtins
#ifdef __GNUC__

template <typename I> constexpr inline I log2_of_power_of_2(I non_negative_power_of_2) noexcept
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(unsigned long long), "Unexpectedly large type");
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


/*

Perhaps use this instead?

constexpr unsigned __fh__
log2_of_power_of_2(unsigned non_negative_power_of_2) noexcept
{ return __builtin_ctz  (non_negative_power_of_2); }

static constexpr unsigned long __fh__
log2_of_power_of_2(unsigned long non_negative_power_of_2)  noexcept
{ return __builtin_ctzl (non_negative_power_of_2); }

static constexpr unsigned long long __fh__
log2_of_power_of_2(unsigned long long non_negative_power_of_2) noexcept
{ return __builtin_ctzll(non_negative_power_of_2); }

static constexpr int __fh__
log2_of_power_of_2(int non_negative_power_of_2) noexcept
{ return __builtin_ctz  (non_negative_power_of_2); }

static constexpr long __fh__
log2_of_power_of_2(long non_negative_power_of_2)  noexcept
{ return __builtin_ctzl (non_negative_power_of_2); }

static constexpr long long __fh__
log2_of_power_of_2(long long non_negative_power_of_2) noexcept
{ return __builtin_ctzll(non_negative_power_of_2); }

#include <cassert>

template <typename I, typename P>
constexpr I div_by_power_of_2(I dividend, P power_of_2_divisor) noexcept
{
#if __cplusplus >= 201402L
    assert((power_of_2_divisor & power_of_2_divisor - 1) == 0);
#endif
    return dividend >> log2_of_power_of_2(power_of_2_divisor);
}
*/


} // namespace constexpr_


template <typename I>
constexpr __fhd__ bool divides(I non_zero_divisor, I dividend)  noexcept
{
	return (dividend % non_zero_divisor) == 0;
}

template <typename I>
constexpr inline bool is_divisible_by(I dividend, I non_zero_divisor) noexcept
{
	return divides(non_zero_divisor, dividend);
}

template <typename I>
constexpr __fhd__ I modulo_power_of_2(I x, I power_of_2_modulus) noexcept
{
	return x & (power_of_2_modulus - I{1});
}

template <typename I>
constexpr __fhd__ bool power_of_2_divides(I power_of_2_divisor, I dividend) noexcept
{
	return modulo_power_of_2(dividend, power_of_2_divisor) == 0;
}

template <typename I>
constexpr __fhd__ bool is_divisible_by_power_of_2(I dividend, I power_of_2_divisor) noexcept
{
	return power_of_2_divides(power_of_2_divisor, dividend);
}


template <typename I> constexpr __fhd__ bool is_even(I x) noexcept { return power_of_2_divides(I{2}, x);     }
template <typename I> constexpr __fhd__ bool is_odd (I x) noexcept { return not power_of_2_divides(I{2}, x); }

} // namespace kat



///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif // CUDA_KAT_ON_DEVICE_CONSTEXPR_MATH_CUH_
