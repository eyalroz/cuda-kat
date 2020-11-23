/**
 * @file kat/containers/array.hpp
 *
 * @brief This file implements `kat::array`, an equivalent of C++11's
 * `std::array` which may be used both in host-side and CUDA-device-side
 * code, along with some supporting functions and overloaded operators
 * for that class.
 */

// Copyright (C) 2007, 2008, 2009, 2010 Free Software Foundation, Inc.
// Copyright (C) 2017, 2019 Eyal Rozenberg <eyalroz@technion.ac.il>
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

#pragma once
#ifndef CUDA_KAT_CONTAINERS_ARRAY_HPP_
#define CUDA_KAT_CONTAINERS_ARRAY_HPP_

#include <kat/common.hpp>
#include <kat/detail/range_access.hpp>
#include <kat/utility.hpp>

#include <cstddef>
#include <cstdlib>
#include <array>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <type_traits>

namespace kat {

///@cond
template<typename T, size_t NumElements>
struct array_traits
{
	typedef T type[NumElements];

	KAT_FHD static constexpr T& reference(const type& t, size_t n) noexcept { return const_cast<T&>(t[n]); }
	KAT_FHD static constexpr T* pointer(const type& t) noexcept             { return const_cast<T*>(t); }
};

template<typename T>
struct array_traits<T, 0>
{
	struct type { };

	KAT_FHD static constexpr T& reference(const type&, size_t) noexcept     { return *static_cast<T*>(nullptr); }
	KAT_FHD static constexpr T* pointer(const type&) noexcept               { return nullptr; }
};
///@endcond

  /**
   *  @brief A standard container for storing a fixed size sequence of elements,
   * based on @ref std::array - but fully GPU device-side enabled.
   *
   *  @tparam T              Type of individual elements
   *  @tparam NumElemements  Number of elements in array.
  */
template<typename T, size_t NumElements>
struct array
{
	typedef T                                       value_type;
	typedef value_type*                             pointer;
	typedef const value_type*                       const_pointer;
	typedef value_type&                             reference;
	typedef const value_type&                       const_reference;
	typedef value_type*                             iterator;
	typedef const value_type*                       const_iterator;
	typedef size_t                                  size_type;
	typedef std::ptrdiff_t                          difference_type;
	typedef std::reverse_iterator<iterator>         reverse_iterator;
	typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;

	// Support for zero-sized arrays mandatory.
	typedef array_traits<T, NumElements>            array_traits_type;

	typename array_traits_type::type   elements;

	// No explicit construct/copy/destroy for aggregate type.

	// DR 776.
	KAT_FHD CONSTEXPR_SINCE_CPP_14 void fill(const value_type& u)
	{
		//std::fill_n(begin(), size(), __u);
		for(size_type i = 0; i < NumElements; i++) {
			elements[i] = u;
		}
	}

	// Does the noexcept matter here?
	KAT_FHD CONSTEXPR_SINCE_CPP_14
	void
	swap(array& other) noexcept(noexcept(std::swap(std::declval<T&>(), std::declval<T&>())))
	{
		// std::swap_ranges(begin(), end(), other.begin());
		auto first1 = begin();
		auto last1 = end();
		auto first2 = other.begin();

		for (; first1 != last1; ++first1, (void)++first2)
			kat::swap(*first1, *first2);
	}

	// Iterators.

	KAT_FHD CONSTEXPR_SINCE_CPP_14 iterator begin() noexcept { return iterator(data()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_iterator begin() const noexcept { return const_iterator(data()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 iterator end() noexcept { return iterator(data() + NumElements); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_iterator end() const noexcept { return const_iterator(data() + NumElements); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_iterator cbegin() const noexcept	{ return const_iterator(data()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_iterator cend() const noexcept { return const_iterator(data() + NumElements); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
	KAT_FHD CONSTEXPR_SINCE_CPP_14 const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

	// Capacity.

	KAT_FHD constexpr size_type size() const noexcept { return NumElements; }
	KAT_FHD constexpr size_type max_size() const noexcept { return NumElements; }
	KAT_FHD constexpr bool empty() const noexcept { return size() == 0; }

	// Element access.

	KAT_FHD CONSTEXPR_SINCE_CPP_14 reference operator[](size_type n) noexcept { return array_traits_type::reference(elements, n); }

	KAT_FHD constexpr const_reference operator[](size_type n) const noexcept { return array_traits_type::reference(elements, n); }

	// Note: The bounds checking here is more violent than usual: It will
	// terminate the execution of the program / the kernel
	KAT_FHD CONSTEXPR_SINCE_CPP_17 reference at(size_type n) noexcept {
		if (n > NumElements) {
#if __CUDA_ARCH__
#if __cplusplus >= 202001L
			if (not std::is_constant_evaluated()) { asm("trap;"); }
#else
			assert(false);
#endif
			throw std::out_of_range("kat::array::at: index n exceeds number of elements");
#endif
		}
		return array_traits_type::reference(elements, n);
	}

	// Note: The bounds checking here is more violent than usual in a kernel
	// terminate the execution of the program / the kernel
	KAT_FHD CONSTEXPR_SINCE_CPP_17 const_reference at(size_type n) const
	{
		if (n > NumElements) {
#if __CUDA_ARCH__
#if __cplusplus >= 202001L
			if (not std::is_constant_evaluated()) { asm("trap;"); }
#else
			assert(false);
#endif
#else
			throw std::out_of_range("kat::array::at: index n exceeds number of elements");
#endif
		}
		return array_traits_type::reference(elements, n);
	}

	KAT_FHD CONSTEXPR_SINCE_CPP_14 reference front() noexcept
	{ return *begin(); }

	KAT_FHD constexpr const_reference front() const noexcept
	{ return array_traits_type::reference(elements, 0); }

	KAT_FHD reference back() noexcept
	{ return NumElements ? *(end() - 1) : *end(); }

	KAT_FHD constexpr const_reference back() const noexcept
	{
		return NumElements ?
			array_traits_type::reference(elements, NumElements - 1) :
			array_traits_type::reference(elements, 0);
	}

	KAT_FHD CONSTEXPR_SINCE_CPP_14 pointer data() noexcept { return array_traits_type::pointer(elements); }

	KAT_FHD constexpr const_pointer data() const noexcept { return array_traits_type::pointer(elements); }
};

#if __cpp_deduction_guides >= 201606
  template<typename _Tp, typename... _Up>
    array(_Tp, _Up...)
      -> array<std::enable_if_t<(std::is_same_v<_Tp, _Up> && ...), _Tp>,
	       1 + sizeof...(_Up)>;
#endif

// Array comparisons.
template<typename T, size_t NumElements>
KAT_FHD CONSTEXPR_SINCE_CPP_17
bool operator==(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
#if __cplusplus >= 202001L
	// We can call it, since it's constexpr
	return std::equal(one.begin(), one.end(), two.begin());
#else
	auto first1 = one.cbegin();
	const auto last1 = one.cend();
	auto first2 = two.cbegin();
    for (; first1 != last1; ++first1, ++first2) {
        if (!(*first1 == *first2)) {
            return false;
        }
    }
    return true;
#endif
}

template<typename T, size_t NumElements>
KAT_FHD CONSTEXPR_SINCE_CPP_17 bool
operator!=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{ return !(one == two); }

template<typename T, size_t NumElements>
KAT_FHD bool CONSTEXPR_SINCE_CPP_17
operator<(const array<T, NumElements>& a, const array<T, NumElements>& b)
{
#if __cplusplus >= 202001L
	// We can call it, since it's constexpr
	return std::lexicographical_compare(a.cbegin(), a.cend(), b.cbegin(), b.cend());
#else
	auto first1 = a.cbegin();
	const auto last1 = a.cend();
	auto first2 = b.cbegin();
	const auto last2 = b.cend();
    for ( ; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2 ) {
        if (*first1 < *first2) return true;
        if (*first2 < *first1) return false;
    }
    return (first1 == last1) && (first2 != last2);
#endif
}

template<typename T, size_t NumElements>
KAT_FHD bool CONSTEXPR_SINCE_CPP_17
operator>(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return two < one;
}

template<typename T, size_t NumElements>
KAT_FHD bool CONSTEXPR_SINCE_CPP_17
operator<=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return !(one > two);
}

template<typename T, size_t NumElements>
KAT_FHD bool CONSTEXPR_SINCE_CPP_17
operator>=(const array<T, NumElements>& one, const array<T, NumElements>& two)
{
	return !(one < two);
}

// Specialized algorithms.
// #if __cplusplus >= 201712L
//template<typename T, size_t NumElements, typename = std::enable_if_t<std::is_swappable<T>::value>>
//#else
template<typename T, size_t NumElements>
// #endif
KAT_FHD CONSTEXPR_SINCE_CPP_14 void swap(array<T, NumElements>& one, array<T, NumElements>& two)
noexcept(noexcept(one.swap(two)))
{
	one.swap(two);
}

template<size_t Integer, typename T, size_t NumElements>
KAT_FHD constexpr T& get(array<T, NumElements>& arr) noexcept
{
	static_assert(Integer < NumElements, "index is out of bounds");
	return array_traits<T, NumElements>::reference(arr.elements, Integer);
}

template<size_t Integer, typename T, size_t NumElements>
KAT_FHD constexpr T&& get(array<T, NumElements>&& arr) noexcept
{
#ifndef KAT_DEFINE_MOVE_AND_FORWARD
	using std::move;
#endif
	static_assert(Integer < NumElements, "index is out of bounds");
	return move(get<Integer>(arr));
}

template<size_t Integer, typename T, size_t NumElements>
KAT_FHD constexpr const T& get(const array<T, NumElements>& arr) noexcept
{
	static_assert(Integer < NumElements, "index is out of bounds");
	return array_traits<T, NumElements>::reference(arr.elements, Integer);
}

} // namespace kat

///@cond

namespace kat {

// kat::tuple interface to class template array.

template<typename T> class tuple_size;

template<typename T, size_t NumElements>
struct tuple_size<array<T, NumElements>> : public std::integral_constant<size_t, NumElements> { };

template<size_t Integer, typename T> class tuple_element;

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = T;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, const array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = const T;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, volatile array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = volatile T;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, const volatile array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = const volatile T;
};

#if __cplusplus >= 201402L
#ifndef _KAT_TUPLE_ELEMENT_T
template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;
#define _KAT_TUPLE_ELEMENT_T
#endif
#endif

///@endcond

} // namespace kat

namespace std {

// std::tuple interface to class template kat::array.


template<typename T> class tuple_size;

template<typename T, size_t NumElements>
struct tuple_size<kat::array<T, NumElements>> : public integral_constant<size_t, NumElements> { };

template<size_t Integer, typename T> class tuple_element;

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, kat::array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	typedef T type;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, const kat::array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = const T;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, volatile kat::array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = volatile T;
};

template<size_t Integer, typename T, size_t NumElements>
struct tuple_element<Integer, const volatile kat::array<T, NumElements>>
{
	static_assert(Integer < NumElements, "index is out of bounds");
	using type = const volatile T;
};

} // namespace std

///@endcond

#endif // CUDA_KAT_CONTAINERS_ARRAY_HPP_
