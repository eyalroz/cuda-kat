/**
 * @file <type_traits> constructs usable in C++11, which only
 * became available in C++14.
 */
#ifndef CUDA_KAT_CPP17_TYPE_TRAITS_HPP_
#define CUDA_KAT_CPP17_TYPE_TRAITS_HPP_

///@cond

#if __cplusplus < 201103L
#error "C++11 or newer is required to use this header"
#endif

#include <type_traits>

namespace kat {

#if  __cplusplus < 201702L
namespace detail {

struct is_nothrow_swappable_helper
{
	template<typename T>
	static std::integral_constant<bool, noexcept(swap(std::declval<T&>(), std::declval<T&>()))> dummy(int);

	template<typename>
	static std::false_type dummy(...);
};

} // namespace detail

template<typename T>
struct is_nothrow_swappable	: detail::is_nothrow_swappable_helper
{
	static constexpr const bool value = decltype(dummy<T>(0))::value;
};
#else
using std::is_nothrow_swappable;
#endif

} // namespace kat

///@endcond

#endif // CUDA_KAT_CPP17_TYPE_TRAITS_HPP_
