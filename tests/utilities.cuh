#ifndef TEST_UTILITIES_CUH_
#define TEST_UTILITIES_CUH_

#include <algorithm>
#include <climits>
#include <type_traits>
#include <iterator>

template <typename I>
I round_up(I x, I quantum) { return (x % quantum) ? (x + (quantum - (x % quantum))) : x; }

template <typename I>
I round_down(I x, I quantum) { return x - x % quantum; }

template <typename T, std::size_t Length>
std::size_t array_length(const T(&ref)[Length]) { return Length; }

// Should be constexpr - but only beginning in C++20
template< class InputIt>
bool all_of( InputIt first, InputIt last)
{
	static_assert(std::is_same<typename std::iterator_traits<InputIt>::value_type, bool>::value, "This function is intended for boolean-valued sequences only");
	return std::all_of(first, last, [](bool b) { return b; });
}

// Should be constexpr - but only beginning in C++20
template<class Container>
bool all_of(const Container& c)
{
	static_assert(std::is_same<typename Container::value_type, bool>::value, "This function is intended for boolean-valued sequences only");
	return std::all_of(std::cbegin(c), std::cend(c), [](bool b) { return b; });
}

#endif /* TEST_UTILITIES_CUH_ */
