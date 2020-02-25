#ifndef CUDA_KAT_TEST_UTIL_CPU_BUILTIN_EQUIVALENTS_HPP_
#define CUDA_KAT_TEST_UTIL_CPU_BUILTIN_EQUIVALENTS_HPP_

#include <type_traits>


template <typename I>
constexpr inline I absolute_value(I x)
{
	static_assert(std::is_integral<I>::value, "Only to be used for integral types");
	return x > 0 ? x : I(-x);
}

template <> constexpr inline float  absolute_value<float >(float  x) { return std::abs(x); }
template <> constexpr inline double absolute_value<double>(double x) { return std::abs(x); }

namespace detail {
template <typename I>
constexpr inline std::make_unsigned_t<I> absolute_difference(std::false_type, I x, I y)
{
	// unsigned case
	return x < y ? y-x : x-y;
}

template <typename I>
constexpr inline std::make_unsigned_t<I> absolute_difference(std::true_type, I x, I y)
{
	// signed case

	auto have_same_sign = (x > 0) == (y > 0);
	if (have_same_sign) {
		return x < y ? y-x : x-y;
	}
	using uint_t = std::make_unsigned_t<I>;
	return x < y ?
		uint_t(-x) + uint_t(y) :
		uint_t(x) + uint_t(-y);
}

} // namespace detail

// This may be a a poor implementation, don't use it elsewhere
template <typename I>
constexpr inline std::make_unsigned_t<I> absolute_difference(I x, I y)
{
	static_assert(std::is_integral<I>::value, "Only to be used for integral types");
	using is_signed = std::integral_constant<bool, std::is_signed<I>::value>;
	return detail::absolute_difference<I>(is_signed{}, x, y);
}



template <typename I> int population_count(I x)
{
	static_assert(std::is_integral<I>::value, "Only integral types are supported");
	static_assert(sizeof(I) <= sizeof(unsigned long long), "Unexpectedly large type");

	using native_popc_type =
		typename std::conditional<
			sizeof(I) <= sizeof(unsigned),
			unsigned,
			unsigned long long
		>::type;
	return population_count<native_popc_type>(static_cast<native_popc_type>(x));
}

template <typename I> int population_count(I x);

template<> inline int population_count<unsigned>(unsigned x) { return __builtin_popcount(x); }
template<> inline int population_count<unsigned long>(unsigned long x) { return __builtin_popcountl(x); }
template<> inline int population_count<unsigned long long>(unsigned long long x) { return __builtin_popcountll(x); }

template <typename I> inline I bit_reverse(I x)
{
	static_assert(std::is_integral<I>::value and sizeof(I) <= 8, "bit_reverse is only available for integers with 64 bits or less");
	switch(sizeof(I)) {
	case 1:  return bit_reverse<uint8_t>(reinterpret_cast<uint8_t&>(x));
	case 2:  return bit_reverse<uint16_t>(reinterpret_cast<uint16_t&>(x));
	case 4:  return bit_reverse<uint32_t>(reinterpret_cast<uint32_t&>(x));
	default: return bit_reverse<uint64_t>(reinterpret_cast<uint64_t&>(x));
	}
}

template <>
inline uint8_t bit_reverse(uint8_t x)
{
	static unsigned char lookup[16] = {
		0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
		0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
	};

	// Reverse top half, reverse lower half, and swap them.
	return (lookup[x & 0b1111] << 4) | lookup[x >> 4];
}

template <>
inline uint16_t bit_reverse(uint16_t x)
{
	return (bit_reverse<uint8_t>(x & 0xFF) << 8) | bit_reverse<uint8_t>(x >> 8);
}


template <>
inline uint32_t bit_reverse(uint32_t x)
{
	return (bit_reverse<uint16_t>(x & 0xFFFF) << 16) | bit_reverse<uint16_t>(x >> 16);
}

template <>
inline uint64_t bit_reverse(uint64_t x)
{
	return (uint64_t{bit_reverse<uint32_t>(x & 0xFFFFFFFF)} << 32) | bit_reverse<uint32_t>(x >> 32);
}



#endif // CUDA_KAT_TEST_UTIL_CPU_BUILTIN_EQUIVALENTS_HPP_
