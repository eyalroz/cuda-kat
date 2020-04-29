/**
 * @file on_device/non-builtins.cuh
 *
* @brief Templated, uniformly-named C++ functions wrapping what should
* have been single PTX - but aren't (in a dedicated `non_builtins` namespace).
*
* There are several functions one would expect would compile to single PTX
* instructions (Similar ones _do_ compile to single PTX instructions,
* and on the CPU, they themselves often translate to a single machine
* instruction) - but strangely, they do not. Implementations of such functions
* are found in this file rather than in @ref `on_device/builtins.cuh`; and they
* get a different namespace to avoid accidental confusion.
*
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
template <typename I> KAT_FD int find_first_set(I x)
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(long long), "Unexpectedly large type");

	using ffs_type = typename std::conditional< sizeof(I) <= sizeof(int), int, long long >::type;
	return find_first_set<ffs_type>(x);
}
template <> KAT_FD int find_first_set< int               >(int                x) { return __ffs(x);                       }
template <> KAT_FD int find_first_set< long long         >(long long          x) { return __ffsll(x);                     }

/**
 * @brief counts the number of initial zeros when considering the binary representation
 * of a number from least to most significant digit
 * 
 * @tparam FixSemanticsForZero the simpler implementation of this function uses the
 * @ref `find_first_set()` builtin. Unfortunately, that one returns -1 rather than 0
 * if no bits are set. Fixing this requires a couple of extra instructions. By default,
 * we'll use them, but one might be interested just skipping them and taking -1
 * instead of 32 (= warp_size) for the no-1's case.
 *
 * @param x the number whose binary representation is to be counted
 * @return the number of initial zero bits before the first 1; if x is 0, the full
 * number of bits is returned (or -1, depending on @tparam FixSemanticsForZero).
 */
template <typename I, bool FixSemanticsForZero = true>
KAT_FD int count_trailing_zeros(I x)
{
	if (FixSemanticsForZero and x == 0) {
		return size_in_bits<I>();
	}
	return find_first_set<I>(x) - 1;
}

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

/**
 * Performs a bitwise leftwards rotation of the bits in the binary representation of value
 *
 * @param value A value whose bits are to be rotated
 * @param shift_amount the number of bits to shift left by. Behavior for values about the
 * number of bits for T is undefined.
 * @return the left-rotation result, or something arbitrary for invalid @p shift_amount.
 *
 * @note a.k.a. `rol` or rotl` on many platforms.
 */
template <typename T>
__device__ T rotate_left(T value, int shift_amount) {
	static_assert(std::is_unsigned<T>::value, "Only unsigned integers are supported");
	return (std::is_same<T, uint32_t>::value) ?
		builtins::funnel_shift_left(value, value, shift_amount) :
		(value << shift_amount) | ( value >> (size_in_bits(value) - shift_amount) );
}

/**
 * Performs a bitwise rightwards rotation of the bits in the binary representation of value
 *
 * @param value A value whose bits are to be rotated
 * @param shift_amount the number of bits to shift left by. Behavior for values about the
 * number of bits for T is undefined.
 * @return the right-rotation result, or something arbitrary for invalid @p shift_amount.
 *
 * @note a.k.a. `ror` or rotr` on many platforms.
 */
template <typename T>
__device__ T rotate_right(T value, int shift_amount) {
	static_assert(std::is_unsigned<T>::value, "Only unsigned integers are supported");
	return (std::is_same<T, uint32_t>::value) ?
		builtins::funnel_shift_right(value, value, shift_amount) :
		(value >> shift_amount) | ( value << (size_in_bits(value)- shift_amount) );
}


} // namespace non_builtins
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_
