/**
 * @file kat/utility.hpp
 *
 * @brief An adaptation for host-and-device use of some
 * of the standard C++ library's `<utility>` code.
 */
#pragma once
#ifndef CUDA_KAT_UTILITY_HPP_
#define CUDA_KAT_UTILITY_HPP_

#include <kat/common.hpp>

#include <type_traits>
#include <utility> // Mainly so that KAT code can our header as a drop-in for <utility> itself

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond
#include <kat/detail/integer_sequence.hpp>


namespace kat {

template<typename T>
constexpr KAT_FHD typename std::remove_reference<T>::type&& move(T&& v) noexcept
{
	return static_cast<typename std::remove_reference<T>::type&&>(v);
}

template<typename T>
constexpr KAT_FHD T&& forward(typename std::remove_reference<T>::type& v) noexcept
{
	return static_cast<T&&>(v);
}

template<typename T>
constexpr KAT_FHD T&& forward(typename std::remove_reference<T>::type&& v) noexcept
{
	return static_cast<T&&>(v);
}

#if __cplusplus >= 201401L
template <typename T, typename U = T>
constexpr KAT_FHD auto exchange (T& x, U&& new_value) // TODO: A noexcept clause?
{
	auto old_value = move(x);
	x = forward<T>(new_value);
	return old_value;
}
#endif // __cplusplus >= 201401L

/**
 * @brief Swap two values on the device-side, in-place.
 *
 * @note A (CUDA, or any other) compiler will often not actually
 * emit any code when this function is used. Instead, it will use
 * one argument instead of the other in later code, i.e. "swap"
 * them in its own internal figuring.
 *
 * @note  Is this enough, without the multiple specializations for std::swap?
 * @todo How does EASTL swap work? Should I incorporate its specializations?
 *
 */
template <typename T>
KAT_FHD CONSTEXPR_SINCE_CPP_14 void swap( T& a, T& b )
	noexcept(
	    std::is_nothrow_move_constructible<T>::value &&
	    std::is_nothrow_move_assignable<T>::value
	)
{
	T tmp ( move(a) );
	a = move(b);
	b = move(tmp);
}

} // namespace kat

#endif // CUDA_KAT_UTILITY_HPP_
