#ifndef CUDA_KAT_TEST_MISC_UTILITIES_CUH_
#define CUDA_KAT_TEST_MISC_UTILITIES_CUH_

#include <cuda/api/types.hpp>
#include <doctest.h>

#include <algorithm>
#include <climits>
#include <type_traits>
#include <iterator>
#include <utility>

template <typename I>
constexpr inline I round_up(I x, I quantum) { return (x % quantum) ? (x + (quantum - (x % quantum))) : x; }

template <typename I>
constexpr inline I round_down(I x, I quantum) { return x - x % quantum; }

template <typename T, std::size_t Length>
constexpr inline std::size_t array_length(const T(&ref)[Length]) { return Length; }

// Should be constexpr - but only beginning in C++20
template< class InputIt>
bool inline all_of( InputIt first, InputIt last)
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

// Code for is_iterator lifted from:
// https://stackoverflow.com/a/12032923/1593077
template<typename T, typename = void>
struct is_iterator
{
   static constexpr bool value = false;
};

template<typename T>
struct is_iterator<T, typename std::enable_if<!std::is_same<typename std::iterator_traits<T>::value_type, void>::value>::type>
{
   static constexpr bool value = true;
};

/**
 * Use these next few types to make assertions regarding each member
 * of a template parameter pack, e.g.
 *
 *  static_assert(all_true<(Numbers == 0 || Numbers == 1)...>::value, "");
 *
 */
template<bool...> struct bool_pack;
template<bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template <typename T>
constexpr inline std::size_t size_in_bits() { return sizeof(T) * CHAR_BIT; }
template <typename T>
constexpr inline std::size_t size_in_bits(const T&) { return sizeof(T) * CHAR_BIT; }


/**
* Divides the left-hand-side by the right-hand-side, rounding up
* to an integral multiple of the right-hand-side, e.g. (9,5) -> 2 , (10,5) -> 2, (11,5) -> 3.
*
* @param dividend the number to divide
* @param divisor the number of by which to divide
* @return The least integer multiple of {@link divisor} which is greater-or-equal to
* the non-integral division dividend/divisor.
*
* @note sensitive to overflow, i.e. if dividend > std::numeric_limits<S>::max() - divisor,
* the result will be incorrect
*/
template <typename S, typename T>
constexpr inline S div_rounding_up(const S& dividend, const T& divisor) {
	return (dividend + divisor - 1) / divisor;
/*
	std::div_t div_result = std::div(dividend, divisor);
	return div_result.quot + !(!div_result.rem);
*/
}

// C++14 version of [[maybe_unused]] ...
template <typename T>
inline void ignore(T &&) { }

namespace doctest {

const char* current_test_name() { return doctest::detail::g_cs->currentTest->m_name; }

} // namespace doctest

// #ifdef __GNUC__
template <typename T>
[[gnu::warning("Artificial warning to print a type name - please ignore")]]
void print_type() noexcept { return; }

template <typename T>
[[gnu::warning("Artificial warning to print a type name - please ignore")]]
void print_type_of(T&& x) noexcept{ return; }
// #endif

#endif /* CUDA_KAT_TEST_MISC_UTILITIES_CUH_ */
