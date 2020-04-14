/**
 * @file kat/common.hpp
 *
 * @brief Basic type and macro definitions used throughout the KAT library.
 */
#pragma once
#ifndef CUDA_KAT_COMMON_HPP_
#define CUDA_KAT_COMMON_HPP_

#include <cstddef> // for std::size_t
#include <type_traits>

#include <kat/detail/execution_space_specifiers.hpp>
#include <kat/detail/constexpr_by_cpp_version.hpp>

namespace kat {

/**
 * Used throughout the kat library for (non-negative) sizes and lengths
 * of containers, memory regions and so on - on both the host and the device
 * side.
 *
 * @note CUDA isn't explicit about this, but it also uses the standard library's
 * size_t occasionally.
 */
using size_t = std::size_t;

#if __cplusplus < 201703L

// Some C++17 type traits definable in C++11

template<typename ...               > struct conjunction : std::true_type {};
template<typename B                 > struct conjunction<B> : B {};
template<typename B, typename ... Bs> struct conjunction<B, Bs...> : std::conditional<bool(B::value), conjunction<Bs...>, B>::type {};

template<typename ...               > struct disjunction : std::true_type {};
template<typename B                 > struct disjunction<B> : B {};
template<typename B, typename ... Bs> struct disjunction<B, Bs...> : std::conditional<bool(B::value), disjunction<Bs...>, B>::type {};

template <bool B> using bool_constant = std::integral_constant<bool, B>;

template<typename B> struct negation : bool_constant<not bool(B::value)> {};

#else

template <typename ... Bs> using conjunction   = std::conjunction<Bs...>;
template <typename ... Bs> using disjunction   = std::disjunction<Bs...>;
template <bool B> using bool_constant = std::bool_constant<B>;
template <bool B> using negation      = std::negation<B>;


#endif


template<typename T, typename... Ts>
using is_any_of = disjunction<std::is_same<T, Ts>...>;

/*
template<typename T, typename First>
struct is_any_of<T, First>
	: std::is_same<T, First> {};

template<typename T, typename T1, typename T2>
struct is_any_of<T, T1, T2>
	: bool_constant<std::is_same<T, T1>::value or std::is_same<T, T2>::value> {};

template<typename T, typename T1, typename T2, typename T3>
struct is_any_of<T, T1, T2, T3>
	: bool_constant<std::is_same<T, T1>::value or std::is_same<T, T2>::value or std::is_same<T, T2>::value> {};

template<typename T, typename T1, typename T2, typename T3, typename... Rest>
struct is_any_of<T, T1, T2, T3, Rest...>
    : bool_constant<std::is_same<T, T1, T2, T3>::value or is_any_of<T, Rest...>::value> {};
*/

} // namespace kat

#endif // CUDA_KAT_COMMON_HPP_
