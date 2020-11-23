/**
 * @file kat/containers/detail/normal_iterator.hpp
 *
 *  @brief This file implements normal_iterator and its supporting
 *  functions and overloaded operators.
 */

// Copyright (C) 2001-2020 Free Software Foundation, Inc.
// Copyright (C) 2020 Eyal Rozenberg <eyalroz@technion.ac.il>
//
// This file is based on an except of <bits/stl_iterator.h> from the GNU
// ISO C++ Library.  It is free software. Thus, you can redistribute it
// and/or modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or (at
// your option) any later version.
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


/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996-1998
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

#ifndef KAT_DETAIL_NORMAL_ITERATOR_HPP_
#define KAT_DETAIL_NORMAL_ITERATOR_HPP_

#include <type_traits>

#if __cplusplus > 201703L
# include <compare>
# include <new>
#endif

#ifndef KAT_CXX20_CONSTEXPR
#if __cplusplus >= 202001L
#define KAT_CXX20_CONSTEXPR constexpr
#else
#define KAT_CXX20_CONSTEXPR constexpr
#endif
#endif

namespace kat {
namespace detail {

  // This iterator adapter is @a normal in the sense that it does not
  // change the semantics of any of the operators of its iterator
  // parameter.  Its primary purpose is to convert an iterator that is
  // not a class, e.g. a pointer, into an iterator that is a class.
  // The Container parameter exists solely so that different containers
  // using this template can instantiate different types, even if the
  // Iterator parameter is the same.
template<typename Iterator, typename Container>
class normal_iterator {
protected:
	Iterator current;

	typedef std::iterator_traits<Iterator> traits_type;

public:
	typedef Iterator iterator_type;
	typedef typename traits_type::iterator_category iterator_category;
	typedef typename traits_type::value_type value_type;
	typedef typename traits_type::difference_type difference_type;
	typedef typename traits_type::reference reference;
	typedef typename traits_type::pointer pointer;

#if __cplusplus > 201703L && __cpp_lib_concepts
	using iterator_concept = std::__detail::__iter_concept<Iterator>;
#endif

	constexpr normal_iterator() noexcept
	: current(Iterator()) {}

	explicit KAT_CXX20_CONSTEXPR
	normal_iterator(const Iterator& i) noexcept
	: current(i) {}

	// Allow iterator to const_iterator conversion
	template<typename Iter>
	KAT_CXX20_CONSTEXPR
	normal_iterator(const normal_iterator<Iter,
		typename std::enable_if<
		(std::is_same<Iter, typename Container::pointer>::value),
		Container>::type>& i) noexcept
	: current(i.base()) {}

	// Forward iterator requirements
	KAT_CXX20_CONSTEXPR
	reference
	operator*() const noexcept
	{	return *current;}

	KAT_CXX20_CONSTEXPR
	pointer
	operator->() const noexcept
	{	return current;}

	KAT_CXX20_CONSTEXPR
	normal_iterator&
	operator++() noexcept
	{
		++current;
		return *this;
	}

	KAT_CXX20_CONSTEXPR
	normal_iterator
	operator++(int) noexcept
	{	return normal_iterator(current++);}

	// Bidirectional iterator requirements
	KAT_CXX20_CONSTEXPR
	normal_iterator&
	operator--() noexcept
	{
		--current;
		return *this;
	}

	KAT_CXX20_CONSTEXPR
	normal_iterator
	operator--(int) noexcept
	{	return normal_iterator(current--);}

	// Random access iterator requirements
	KAT_CXX20_CONSTEXPR
	reference
	operator[](difference_type n) const noexcept
	{	return current[n];}

	KAT_CXX20_CONSTEXPR
	normal_iterator&
	operator+=(difference_type n) noexcept
	{	current += n; return *this;}

	KAT_CXX20_CONSTEXPR
	normal_iterator
	operator+(difference_type n) const noexcept
	{	return normal_iterator(current + n);}

	KAT_CXX20_CONSTEXPR
	normal_iterator&
	operator-=(difference_type n) noexcept
	{	current -= n; return *this;}

	KAT_CXX20_CONSTEXPR
	normal_iterator
	operator-(difference_type n) const noexcept
	{	return normal_iterator(current - n);}

	KAT_CXX20_CONSTEXPR
	const Iterator&
	base() const noexcept
	{	return current;}
};

// Note: In what follows, the left- and right-hand-side iterators are
// allowed to vary in types (conceptually in cv-qualification) so that
// comparison between cv-qualified and non-cv-qualified iterators be
// valid.  However, the greedy and unfriendly operators in std::rel_ops
// will make overload resolution ambiguous (when in scope) if we don't
// provide overloads whose operands are of the same type.  Can someone
// remind me what generic programming is about? -- Gaby

// Forward iterator requirements
template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator==(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() == rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator==(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() == rhs.base();
}

template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator!=(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() != rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator!=(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() != rhs.base();
}

// Random access iterator requirements
template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator<(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() < rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator<(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() < rhs.base();
}

template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator>(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() > rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator>(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() > rhs.base();
}

template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator<=(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() <= rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator<=(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() <= rhs.base();
}

template<typename IteratorL, typename IteratorR, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator>=(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
{
	return lhs.base() >= rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline bool operator>=(const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() >= rhs.base();
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// According to the resolution of DR179 not only the various comparison
// operators but also operator- must accept mixed iterator/const_iterator
// parameters.
template<typename IteratorL, typename IteratorR, typename Container>
#if __cplusplus >= 201103L
// DR 685.
KAT_CXX20_CONSTEXPR
inline auto operator-(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs) noexcept
	-> decltype(lhs.base() - rhs.base())
#else
inline typename normal_iterator<IteratorL, Container>::difference_type
operator-(const normal_iterator<IteratorL, Container>& lhs,
	const normal_iterator<IteratorR, Container>& rhs)
#endif
{
	return lhs.base() - rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline typename normal_iterator<Iterator, Container>::difference_type operator-(
	const normal_iterator<Iterator, Container>& lhs, const normal_iterator<Iterator, Container>& rhs) noexcept
{
	return lhs.base() - rhs.base();
}

template<typename Iterator, typename Container>
KAT_CXX20_CONSTEXPR
inline normal_iterator<Iterator, Container> operator+(typename normal_iterator<Iterator, Container>::difference_type n,
	const normal_iterator<Iterator, Container>& i) noexcept
{
	return normal_iterator<Iterator, Container>(i.base() + n);
}

} // namespace detail
} // namespace kat

#endif // KAT_DETAIL_NORMAL_ITERATOR_HPP_
