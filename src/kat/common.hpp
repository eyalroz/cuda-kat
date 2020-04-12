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

template<class ...> struct conjunction : std::true_type {};
template<class B> struct conjunction<B> : B {};
template<class B, class ... Bs>
struct conjunction<B, Bs...> : std::conditional<bool(B::value), conjunction<Bs...>, B>::type {};

template <bool B>
using bool_constant = std::integral_constant<bool, B>;

template<class B>
struct negation : kat::bool_constant<not bool(B::value)> {};

#else

template <class ... Bs> using conjunction   = std::conjunction<Bs...>;
template <bool B      > using bool_constant = std::bool_constant<B>;
template <bool B      > using negation      = std::negation<B>;


#endif

} // namespace kat

#endif // CUDA_KAT_COMMON_HPP_
