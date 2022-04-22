#ifndef CUDA_KAT_TEST_UTILS_PRINTING_HPP_
#define CUDA_KAT_TEST_UTILS_PRINTING_HPP_

#include <string>
#include <ostream>
#include <cuda/runtime_api.hpp>

namespace detail {
template <typename ToBeStreamed>
struct promoted_for_streaming { using type = ToBeStreamed; };
template<> struct promoted_for_streaming<char>{ using type = short; };
template<> struct promoted_for_streaming<signed char>{ using type = signed short; };
template<> struct promoted_for_streaming<unsigned char> { using type = unsigned short; };

} // namespace detail
/*
 * The following structs are used for streaming data to iostreams streams.
 * They have a tendency to try to outsmart you, e.g. w.r.t. char or unsigned
 *  char data - they assume you're really passing ISO-8859-1 code points
 *  rather than integral values, and will print accordingly. Using this
 *  generic promoter, you van avoid that.
 */
template <typename ToBeStreamed>
typename detail::promoted_for_streaming<ToBeStreamed>::type promote_for_streaming(const ToBeStreamed& tbs)
{
	return static_cast<typename detail::promoted_for_streaming<ToBeStreamed>::type>(tbs);
}

inline const char* ordinal_suffix(int n)
{
	static const char suffixes [4][5] = {"th", "st", "nd", "rd"};
	auto ord = n % 100;
	if (ord / 10 == 1) { ord = 0; }
	ord = ord % 10;
	return suffixes[ord > 3 ? 0 : ord];
}

// cuda-api-wrappers-related utilities

template <typename N = int>
inline std::string xth(N n) { return std::to_string(n) + ordinal_suffix(n); }

std::ostream& operator<<(std::ostream& os, cuda::grid::dimensions_t dims)
{
	return os << '(' << dims.x << "," << dims.y << "," << dims.z << ')';
}

std::ostream& operator<<(std::ostream& os, cuda::launch_configuration_t lc)
{
	return os
		<< "grid x block dimensions = " << lc.dimensions.grid << " x " << lc.dimensions.block << ", "
		<< lc.dynamic_shared_memory_size << " bytes dynamic shared memory" << '\n';
}

#ifdef __SIZEOF_INT128__

// always in hex!

std::ostream& operator<<(std::ostream& os, __uint128_t x)
{
	return os << "uint128_t{" << uint64_t(x >> 64) << uint64_t(x & ~uint64_t{0}) << '}';
}

std::ostream& operator<<(std::ostream& os, __int128_t x)
{
	auto sign = x < 0 ? '-' : ' ';
	auto magnitude = x < 0 ? -x : x ;
	return os << "int128_t{" << sign << uint64_t(magnitude >> 64) << uint64_t(magnitude & ~uint64_t{0}) << '}';
}
#endif


#endif // CUDA_KAT_TEST_UTILS_PRINTING_HPP_
