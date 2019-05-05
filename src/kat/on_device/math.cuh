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

#include <kat/define_specifiers.hpp>

namespace kat {

template <typename T, typename S>
__fd__ T div_by_power_of_2_rounding_up(const T& dividend, const S& divisor)
{
	auto mask = divisor - 1; // Remember: 0 is _not_ a power of 2
	auto log_2_of_divisor = builtins::population_count(mask);
	auto correction_for_rounding_up = ((dividend & mask) + mask) >> log_2_of_divisor;

	return (dividend >> log_2_of_divisor) + correction_for_rounding_up;
}

template <typename T>
__fd__ unsigned log2_of_power_of_2(T p)
{
	// Remember 0 is _not_ a power of 2.
	return  builtins::population_count(p - 1);
}

#if __cplusplus < 201402L
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

template <typename T>
__fd__ T lcm(T u, T v)
{
	return (u / gcd(u,v)) * v;
}

template <typename T>
__fd__ unsigned ilog2(std::enable_if<std::is_unsigned<T>::value, T> x) {
  return (CHAR_BIT * sizeof(T) - 1) - builtins::count_leading_zeros(x);
}

} // namespace kat

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_ON_DEVICE_MATH_CUH_
