/**
 * @file on_device/non-builtins.cuh
 *
 * @brief Namespace with uniform-naming scheme, templated-when-relevant,
 * wrappers of what could be (or should be) single PTX instruction - but
 * aren't.
 */
#ifndef CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_
#define CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_

#include <kat/on_device/builtins.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace non_builtins {

/**
 * @brief Determine the 1-based index of the first non-zero bit in the argument.
 *
 * @param x the value to be considered as a container of bits
 * @return If @p x is 0, returns 0; otherwise, returns the 1-based index of the
 * first non-zero bit in @p x
 */
template <typename I> KAT_FD int find_first_set(I x) = delete;
template <> KAT_FD int find_first_set< int                >(int x)                { return __ffs(x);   }
template <> KAT_FD int find_first_set< unsigned int       >(unsigned int x)       { return __ffs(x);   }
template <> KAT_FD int find_first_set< long long          >(long long x)          { return __ffsll(x); }
template <> KAT_FD int find_first_set< unsigned long long >(unsigned long long x) { return __ffsll(x); }

/**
 * @brief counts the number of initial zeros when considering the binary representation
 * of a number from least to most significant digit
 * @param x the number whose representation is to be counted
 * @return the number of initial zero bits before the first 1; if x is 0, -1 is returned
 */
template <typename I> KAT_FD int count_trailing_zeros(I x) { return find_first_set<I>(x) - 1; }

/**
 * @brief counts the number of initial zeros when considering the binary representation
 * of a number from most to least significant digit
 * @param x the number whose representation is to be counted
 * @return the counted number of 0 bits; if x is 0, 32 is returned
 */
template <typename I> KAT_FD int count_leading_zeros(I x)
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(long long), "Unexpectedly large type");

	using native_clz_type =
		typename std::conditional< sizeof(I) <= sizeof(int), int, long long >::type;
	enum : int { width_difference_in_bits = (sizeof(native_clz_type) - sizeof(I)) * CHAR_BIT };
	return builtins::count_leading_zeros<native_clz_type>(static_cast<native_clz_type>(x)) - width_difference_in_bits;
}


} // namespace non_builtins
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_
