/**
 * @file kat/containers/detail/range_access.hpp
 *
 * @brief This file implements equivalents of std namespace methods
 * for accessing ranges, e.g. `std::begin()`, `std::end()` and the
 * like.
 */

// Copyright (C) 2010-2019 Free Software Foundation, Inc.
// Copyright (C) 2020 Eyal Rozenberg <eyalroz@technion.ac.il>
//
// This file is based on <array> from the GNU ISO C++ Library.  It is
// free software. Thus, you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.
//
// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation, i.e.:
//
// You may use this file as part of a free software library without
// restriction.  Specifically, if other files instantiate templates or
// use macros or inline functions from this file, or you compile this file
// and link it with other files to produce an executable, this file does
// not by itself cause the resulting executable to be covered by the GNU
// General Public License. This exception does not however invalidate any
// other reasons why the executable file might be covered by the GNU
// General Public License.
//
// A copy of the GNU General Public License and a copy of the GCC Runtime
// Library Exception are available at <http://www.gnu.org/licenses/>.

#ifndef CUDA_KAT_RANGE_ACCESS_HPP_
#define CUDA_KAT_RANGE_ACCESS_HPP_ 1

#include <kat/common.hpp>
#include <initializer_list>
// #include <iterator>

namespace kat {

  /**
   *  @brief  Return an iterator pointing to the first element of
   *          the container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    begin(Container& cont) -> decltype(cont.begin())
    { return cont.begin(); }

  /**
   *  @brief  Return an iterator pointing to the first element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    begin(const Container& cont) -> decltype(cont.begin())
    { return cont.begin(); }

  /**
   *  @brief  Return an iterator pointing to one past the last element of
   *          the container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    end(Container& cont) -> decltype(cont.end())
    { return cont.end(); }

  /**
   *  @brief  Return an iterator pointing to one past the last element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    end(const Container& cont) -> decltype(cont.end())
    { return cont.end(); }

  /**
   *  @brief  Return an iterator pointing to the first element of the array.
   *  @param  __arr  Array.
   */
  template<typename T, size_t NumElements>
    inline KAT_HD CONSTEXPR_SINCE_CPP_14 T*
    begin(T (&__arr)[NumElements])
    { return __arr; }

  /**
   *  @brief  Return an iterator pointing to one past the last element
   *          of the array.
   *  @param  __arr  Array.
   */
  template<typename T, size_t NumElements>
    inline KAT_HD CONSTEXPR_SINCE_CPP_14 T*
    end(T (&__arr)[NumElements])
    { return __arr + NumElements; }

#if __cplusplus >= 201402L

  template<typename T> class valarray;
  // These overloads must be declared for cbegin and cend to use them.
  template<typename T> T* begin(valarray<T>&);
  template<typename T> const T* begin(const valarray<T>&);
  template<typename T> T* end(valarray<T>&);
  template<typename T> const T* end(const valarray<T>&);

  /**
   *  @brief  Return an iterator pointing to the first element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD constexpr auto
    cbegin(const Container& cont) noexcept(noexcept(kat::begin(cont)))
      -> decltype(kat::begin(cont))
    { return kat::begin(cont); }

  /**
   *  @brief  Return an iterator pointing to one past the last element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD constexpr auto
    cend(const Container& cont) noexcept(noexcept(kat::end(cont)))
      -> decltype(kat::end(cont))
    { return kat::end(cont); }

  /**
   *  @brief  Return a reverse iterator pointing to the last element of
   *          the container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    rbegin(Container& cont) -> decltype(cont.rbegin())
    { return cont.rbegin(); }

  /**
   *  @brief  Return a reverse iterator pointing to the last element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    rbegin(const Container& cont) -> decltype(cont.rbegin())
    { return cont.rbegin(); }

  /**
   *  @brief  Return a reverse iterator pointing one past the first element of
   *          the container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    rend(Container& cont) -> decltype(cont.rend())
    { return cont.rend(); }

  /**
   *  @brief  Return a reverse iterator pointing one past the first element of
   *          the const container.
   *  @param  cont  Container.
   */
  template<typename Container>
    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
    rend(const Container& cont) -> decltype(cont.rend())
    { return cont.rend(); }

// Enable this when we have a kat::reverse_iterator (from bits/stl_iterator.h in libstdc++;
// note we've already adopted normal_iterator from there)
//  /**
//   *  @brief  Return a reverse iterator pointing to the last element of
//   *          the array.
//   *  @param  __arr  Array.
//   */
//  template<typename T, size_t NumElements>
//    inline KAT_HD CONSTEXPR_SINCE_CPP_17 reverse_iterator<T*>
//    rbegin(T (&__arr)[NumElements])
//    { return reverse_iterator<T*>(__arr + NumElements); }
//////
//////  /**
//////   *  @brief  Return a reverse iterator pointing one past the first element of
//////   *          the array.
//////   *  @param  __arr  Array.
//////   */
//////  template<typename T, size_t NumElements>
//////    inline KAT_HD CONSTEXPR_SINCE_CPP_17 reverse_iterator<T*>
//////    rend(T (&__arr)[NumElements])
//////    { return reverse_iterator<T*>(__arr); }
//////
//////  /**
//////   *  @brief  Return a reverse iterator pointing to the last element of
//////   *          the initializer_list.
//////   *  @param  init_list  initializer_list.
//////   */
//////  template<typename T>
//////    inline KAT_HD CONSTEXPR_SINCE_CPP_17 reverse_iterator<const T*>
//////    rbegin(initializer_list<T> init_list)
//////    { return reverse_iterator<const T*>(init_list.end()); }
////
////  /**
////   *  @brief  Return a reverse iterator pointing one past the first element of
////   *          the initializer_list.
////   *  @param  init_list  initializer_list.
////   */
////  template<typename T>
////    inline KAT_HD CONSTEXPR_SINCE_CPP_17 reverse_iterator<const T*>
////    rend(initializer_list<T> init_list)
////    { return reverse_iterator<const T*>(init_list.begin()); }
//
//  /**
//   *  @brief  Return a reverse iterator pointing to the last element of
//   *          the const container.
//   *  @param  cont  Container.
//   */
//  template<typename Container>
//    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
//    crbegin(const Container& cont) -> decltype(kat::rbegin(cont))
//    { return kat::rbegin(cont); }
//
//  /**
//   *  @brief  Return a reverse iterator pointing one past the first element of
//   *          the const container.
//   *  @param  cont  Container.
//   */
//  template<typename Container>
//    inline KAT_HD CONSTEXPR_SINCE_CPP_17 auto
//    crend(const Container& cont) -> decltype(kat::rend(cont))
//    { return kat::rend(cont); }

#endif // C++14

#if __cplusplus >= 201703L

  /**
   *  @brief  Return the size of a container.
   *  @param  cont  Container.
   */
  template <typename Container>
    constexpr auto
    size(const Container& cont) noexcept(noexcept(cont.size()))
    -> decltype(cont.size())
    { return cont.size(); }

  /**
   *  @brief  Return the size of an array.
   *  @param  __array  Array.
   */
  template <typename T, size_t NumElements>
    constexpr size_t
    size(const T (&/*__array*/)[NumElements]) noexcept
    { return NumElements; }

  /**
   *  @brief  Return whether a container is empty.
   *  @param  cont  Container.
   */
  template <typename Container>
    [[nodiscard]] constexpr auto
    empty(const Container& cont) noexcept(noexcept(cont.empty()))
    -> decltype(cont.empty())
    { return cont.empty(); }

  /**
   *  @brief  Return whether an array is empty (always false).
   *  @param  __array  Container.
   */
  template <typename T, size_t NumElements>
    [[nodiscard]] constexpr bool
    empty(const T (&/*__array*/)[NumElements]) noexcept
    { return false; }

  /**
   *  @brief  Return whether an initializer_list is empty.
   *  @param  init_list  Initializer list.
   */
  template <typename T>
    [[nodiscard]] constexpr bool
    empty(initializer_list<T> init_list) noexcept
    { return init_list.size() == 0;}

  /**
   *  @brief  Return the data pointer of a container.
   *  @param  cont  Container.
   */
  template <typename Container>
    constexpr auto
    data(Container& cont) noexcept(noexcept(cont.data()))
    -> decltype(cont.data())
    { return cont.data(); }

  /**
   *  @brief  Return the data pointer of a const container.
   *  @param  cont  Container.
   */
  template <typename Container>
    constexpr auto
    data(const Container& cont) noexcept(noexcept(cont.data()))
    -> decltype(cont.data())
    { return cont.data(); }

  /**
   *  @brief  Return the data pointer of an array.
   *  @param  __array  Array.
   */
  template <typename T, size_t NumElements>
    constexpr T*
    data(T (&__array)[NumElements]) noexcept
    { return __array; }

  /**
   *  @brief  Return the data pointer of an initializer list.
   *  @param  init_list  Initializer list.
   */
  template <typename T>
    constexpr const T*
    data(initializer_list<T> init_list) noexcept
    { return init_list.begin(); }

#endif // C++17

} // namespace kat

#endif // CUDA_KAT_RANGE_ACCESS_HPP_
