#ifndef CUDA_KAT_TEST_MISC_UTILITIES_CUH_
#define CUDA_KAT_TEST_MISC_UTILITIES_CUH_

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

#endif /* CUDA_KAT_TEST_MISC_UTILITIES_CUH_ */
