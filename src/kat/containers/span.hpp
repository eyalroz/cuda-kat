/**
 * @file kat/containers/span.hpp
 *
 * @brief This file implements `kat::span` an equivalent of C++'s
 * `std::span` which may be used both in host-side and CUDA-device-side
 * code, along with some supporting functions and overloaded operators
 * for that class.
 */

// Copyright (C) 2019-2020 Free Software Foundation, Inc.
// Copyright (C) 2020 Eyal Rozenberg <eyalroz@technion.ac.il>
//
// This file is based on <span> from the GNU ISO C++ Library.  It is
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

#ifndef KAT_CONTAINERS_SPAN_HPP_
#define KAT_CONTAINERS_SPAN_HPP_

#include <type_traits>
#include <iterator>
#include <array>
#include <cassert>
#include <kat/containers/array.hpp>
#include "detail/normal_iterator.hpp"
#include <kat/detail/execution_space_specifiers.hpp>
#include <kat/detail/range_access.hpp>


#if __cplusplus >= 202001L
// #include <bits/range_access.h> <- TODO: What replaces this?
#endif

namespace kat {

#if __cplusplus >= 201703L
inline 
#endif
constexpr const std::size_t dynamic_extent = static_cast<std::size_t>(-1);

template<typename Type, std::size_t Extent>
class span;

namespace detail {

template <typename T>
#if __cplusplus >= 202001L
using iter_reference_t = std::iter_reference_t<T>;
#else
using iter_reference_t = decltype(*std::declval<T&>());
#endif

template<typename T>
struct is_span : std::false_type {};

template<typename T, std::size_t Num>
struct is_span<kat::span<T, Num>> : std::true_type {};

#if __cplusplus >= 202001L
template<typename T, std::size_t Num>
struct is_span<std::span<T, Num>> : std::true_type {};

template<typename T, std::size_t Num>
struct is_span<std::span<T, Num>> : std::true_type {};
#endif
// TODO: Entries here for gsl::span

/*

    template<typename T>
      struct __is_std_array : std::false_type { };


    template<typename T, std::size_t Num>
      struct __is_std_array<_GLIBCXX_STD_C::array<T, Num>> : std::true_type { };
*/

template<std::size_t Extent>
class extent_storage {
public:
	constexpr KAT_HD extent_storage(std::size_t) noexcept
	{
	}

	static constexpr KAT_HD std::size_t extent() noexcept
	{
		return Extent;
	}
};

template<>
class extent_storage<dynamic_extent> {
public:
	constexpr KAT_HD extent_storage(std::size_t extent) noexcept
	: extent_value(extent)
	{}

	constexpr KAT_HD std::size_t
	extent() const noexcept
	{
		return this->extent_value;
	}

private:
	std::size_t extent_value;
};

} // namespace detail

template<typename Type, std::size_t Extent = dynamic_extent>
class span {

public:
	// TODO: These two methods should be private (and are, in the libstdc++ sources.
	// Unfortunately it gives me trouble when accessing them during construction of a length-0 span in some cases
	template<std::size_t Offset, std::size_t Count, typename = typename std::enable_if<Count != dynamic_extent>::type>
	static constexpr KAT_HD std::size_t S_subspan_extent()
	{
		static_assert(Count != dynamic_extent, "This implementation should not have been instantiated for Count == dynamic_extent...");
		return Count;
	}

	template<std::size_t Offset>
	static constexpr KAT_HD std::size_t S_subspan_extent<Offset, dynamic_extent>()
	{
	#if __cplusplus >= 201703L
		if constexpr (extent != dynamic_extent)
	#else
		if (extent != dynamic_extent)
	#endif
			return Extent - Offset;
		else
			return dynamic_extent;

	}

private:
      // TODO: Where does __is_array_convertible come from?? I can't use it for now,
      // so the criteria here become trivial before C++2020

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 3255. span's array constructor is too strict
      template<typename T, std::size_t ArrayExtent>
	using is_compatible_array = std::integral_constant<bool,
	  true
	  and (Extent == dynamic_extent or ArrayExtent == Extent)
	  and (std::is_const<Type>::value or not std::is_const<T>::value)
#if false
      and __is_array_convertible<Type, T>
#endif
      >;

    template<typename Iter, typename Ref = detail::iter_reference_t<Iter>>
	using is_compatible_iterator = std::integral_constant<bool,
		true
#if __cplusplus >= 202001L
#if false
		and contiguous_iterator<Iter>::value  // where does this come from?
#endif
	    and std::is_lvalue_reference<detail::iter_reference_t<Iter>>::value,
	    and std::is_same<std::iter_value_t<Iter>, std::remove_cvref_t<Ref>>::value,
#endif
#if false
	    and __is_array_convertible<Type, std::remove_reference_t<Ref>>::value
#endif
	  >;

#if __cplusplus >= 202001L
    template<typename Range>
	using __is_compatible_range
	  = is_compatible_iterator<ranges::iterator_t<Range>>;
#endif

public:
	// member types
	using value_type             = typename std::remove_cv<Type>::type;
	using element_type           = Type;
	using size_type              = std::size_t;
	using reference              = element_type&;
	using const_reference        = const element_type&;
	using pointer                = Type*;
	using const_pointer          = const Type*;

	using iterator               = kat::detail::normal_iterator<pointer, span>;
	using const_iterator         = kat::detail::normal_iterator<const_pointer, span>;
	using reverse_iterator       = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;
	using difference_type        = std::ptrdiff_t;

	// member constants
#if __cplusplus >= 201703L
	inline
#endif
	static constexpr const std::size_t extent = Extent;

	// constructors

	constexpr KAT_HD span() noexcept
#if __cplusplus >= 202001L
	requires ((Extent + 1u) <= 1u)
#endif
	: extent_(0), ptr_(nullptr)
	{
#if __cplusplus < 202001L
		static_assert((Extent + 1u) <= 1u, "Invalid Extent value - for this constructor, can be either dynamic_extent or 0");
#endif
	}

	constexpr KAT_HD
	span(const span&) noexcept = default;

#if __cplusplus >= 202001L
	template<typename T, std::size_t ArrayExtent>
	requires (is_compatible_array<T, ArrayExtent>::value)
#else
	template<typename T, std::size_t ArrayExtent, typename = typename std::enable_if<is_compatible_array<T, ArrayExtent>::value>::type >
#endif
	constexpr KAT_HD
	span(T (&arr)[ArrayExtent]) noexcept
	: span(static_cast<pointer>(arr), ArrayExtent)
	{
//		here
		static_assert(is_compatible_array<T, ArrayExtent>::value, "Attempt to construct a span with an incompatible raw array");
	}

#if __cplusplus >= 202001L
	template<typename T, std::size_t ArrayExtent>
	requires (is_compatible_array<T, ArrayExtent>::value)
#else
	template<typename T, std::size_t ArrayExtent, typename = typename std::enable_if<is_compatible_array<T, ArrayExtent>::value>::type >
#endif
	constexpr KAT_HD
	span(kat::array<T, ArrayExtent>& arr) noexcept
	: span(static_cast<pointer>(arr.data()), ArrayExtent)
	{
		static_assert(is_compatible_array<T, ArrayExtent>::value, "Attempt to construct a span with an incompatible kat::array");
	}

#if __cplusplus >= 202001L
	template<typename T, std::size_t ArrayExtent>
	requires (is_compatible_array<T, ArrayExtent>::value)
#else
	template<typename T, std::size_t ArrayExtent, typename = typename std::enable_if<is_compatible_array<const T, ArrayExtent>::value>::type >
#endif
	constexpr KAT_HD span(const kat::array<T, ArrayExtent>& arr) noexcept
	: span(static_cast<pointer>(arr.data()), ArrayExtent)
	{
		static_assert(is_compatible_array<T, ArrayExtent>::value, "Attempt to construct a span with an incompatible const kat::array");
	}

#if __cplusplus >= 202001L
	template<typename T, std::size_t ArrayExtent>
	requires (is_compatible_array<T, ArrayExtent>::value)
#else
	template<typename T, std::size_t ArrayExtent, typename = typename std::enable_if<is_compatible_array<T, ArrayExtent>::value>::type >
#endif
	constexpr KAT_HOST span(std::array<T, ArrayExtent>& arr) noexcept
	: span(static_cast<pointer>(arr.data()), ArrayExtent)
	{
		static_assert(is_compatible_array<T, ArrayExtent>::value, "Constructing a span with an incompatible std::array");
	}

#if __cplusplus >= 202001L
	template<typename T, std::size_t ArrayExtent>
	requires (is_compatible_array<T, ArrayExtent>::value)
#else
	template<typename T, std::size_t ArrayExtent, typename = typename std::enable_if<is_compatible_array<const T, ArrayExtent>::value>::type >
#endif
	constexpr KAT_HOST span(const std::array<T, ArrayExtent>& arr) noexcept
	: span(static_cast<pointer>(arr.data()), ArrayExtent)
	{
		static_assert(is_compatible_array<T, ArrayExtent>::value, "Constructing a span with an incompatible const std::array");
	}



public:
	// Note: Currently, we can't construct spans from Containers (except through ranges in C++20)

#if __cplusplus >= 202001L
	template<ranges::contiguous_range Range>
	requires (Extent == dynamic_extent)
	  and (!detail::is_span<std::remove_cvref_t<Range>>::value)
	  and (!detail::__is_std_array<std::remove_cvref_t<Range>>::value)
	  and (!is_array_v<std::remove_reference_t<Range>>)
	  and (__is_compatible_range<Range>::value)
	constexpr KAT_HD
	span(Range&& range)
	noexcept(noexcept(ranges::data(range))
		  and noexcept(ranges::size(range)))
	: span(ranges::data(range), ranges::size(range))
	{ }
#endif



#if __cplusplus >= 202001L
	template<contiguous_iterator ContiguousIterator,
	sized_sentinel_for<ContiguousIterator> Sentinel>
	  requires (is_compatible_iterator<ContiguousIterator>::value)
	    and (!is_convertible_v<Sentinel, size_type>)
	constexpr KAT_HD
	span(ContiguousIterator first, Sentinel last)
	noexcept(noexcept(last - first))
	: extent_(static_cast<size_type>(last - first)),
	  ptr_(std::to_address(first))
	{
	  if (Extent != dynamic_extent)
	    assert((last - first) == Extent);
	}
#else
    constexpr KAT_HD span(pointer first, pointer last) noexcept
        : extent_(static_cast<size_type>(last - first)),
    	  ptr_(first)
    { }
#endif

#if __cplusplus >= 202001L
	template<contiguous_iterator ContiguousIterator>
	requires (is_compatible_iterator<ContiguousIterator>::value)
	constexpr KAT_HD
	span(ContiguousIterator first, size_type count)
	noexcept
	: extent_(count), ptr_(std::to_address(first))
	{ assert(Extent == dynamic_extent or count == Extent); }
#else
    constexpr KAT_HD span(pointer first, std::size_t count) noexcept
    	: extent_(count), ptr_(first)
    { }
#endif


      template<typename OType, std::size_t OExtent>
#if __cplusplus >= 2017031L
	requires (Extent == dynamic_extent or Extent == OExtent)
	  and (__is_array_convertible<Type, OType>::value)
#endif
	constexpr KAT_HD
	span(const span<OType, OExtent>& s) noexcept
	: extent_(s.size()), ptr_(s.data())
	{
#if __cplusplus < 2017031L
    	static_assert(Extent == dynamic_extent or Extent == OExtent, "Invalid extent");
    	// what about is_array_convertible? :-(
#endif
	}

	// assignment

	constexpr KAT_HD span&
	operator=(const span&) noexcept = default;

	// observers

	constexpr KAT_HD size_type
	size() const noexcept
	{	return extent_.extent();}

	constexpr KAT_HD size_type
	size_bytes() const noexcept
	{	return extent_.extent() * sizeof(element_type);}

	[[nodiscard]] constexpr KAT_HD bool
	empty() const noexcept
	{	return size() == 0;}

	// element access

	constexpr KAT_HD reference
	front() const noexcept
	{
		static_assert(extent != 0, "Zero span extent");
		assert(!empty());
		return *ptr_;
	}

	constexpr KAT_HD reference
	back() const noexcept
	{
		static_assert(extent != 0, "Zero span extent");
		assert(!empty());
		return *(ptr_ + (size() - 1));
	}

	constexpr KAT_HD reference
	operator[](size_type idx) const noexcept
	{
		static_assert(extent != 0, "Zero span extent");
		assert(idx < size());
		return *(ptr_ + idx);
	}

	constexpr KAT_HD pointer
	data() const noexcept
	{	return ptr_;}

	// iterator support

	constexpr KAT_HD iterator
	begin() const noexcept
	{	return iterator(ptr_);}

	constexpr KAT_HD const_iterator
	cbegin() const noexcept
	{	return const_iterator(ptr_);}

	constexpr KAT_HD iterator
	end() const noexcept
	{	return iterator(ptr_ + size());}

	constexpr KAT_HD const_iterator
	cend() const noexcept
	{	return const_iterator(ptr_ + size());}

	constexpr KAT_HD reverse_iterator
	rbegin() const noexcept
	{	return reverse_iterator(this->end());}

	constexpr KAT_HD const_reverse_iterator
	crbegin() const noexcept
	{	return const_reverse_iterator(this->cend());}

	constexpr KAT_HD reverse_iterator
	rend() const noexcept
	{	return reverse_iterator(this->begin());}

	constexpr KAT_HD const_reverse_iterator
	crend() const noexcept
	{	return const_reverse_iterator(this->cbegin());}

	// subviews

	template<std::size_t Count>
	constexpr KAT_HD span<element_type, Count>
	first() const noexcept
	{
	#if __cplusplus >= 201703L
		if constexpr (Extent == dynamic_extent) {
			if constexpr (Count > 0) {
				assert(Count <= size());
			}
		}
		else
		static_assert(Count <= extent);
	#else
		if (Extent == dynamic_extent) {
			assert ((Count < size() or Count == size()) and "Span fixed element count exceeds its size");
		}
		else {
			assert ((size() < Extent or size() == Extent ) and "Span size exceeds its extent");
		}
	#endif
		return { data(), Count };
	}

	constexpr KAT_HD span<element_type, dynamic_extent>
	first(size_type count) const noexcept
	{
		assert(count <= size());
		return { data(), count };
	}

	template<std::size_t Count>
	constexpr KAT_HD span<element_type, Count>
	last() const noexcept
	{
	#if __cplusplus >= 201703L
		if constexpr (Extent == dynamic_extent) {
			if constexpr (Count > 0) {
				assert(Count <= size());
			}
		}
		else
		static_assert(Count <= extent);
	#else
		if (Extent == dynamic_extent) {
			assert ((Count < size() or Count == size()) and "Span fixed element count exceeds its size");
		}
		else {
			assert ((size() < Extent  or size() == Extent) and "Span size exceeds its extent");
		}
	#endif
		return {data() + (size() - Count), Count};
	}

	constexpr KAT_HD span<element_type, dynamic_extent>
	last(size_type count) const noexcept
	{
		assert(count <= size());
		return {data() + (size() - count), count};
	}

	template<std::size_t Offset, std::size_t Count = dynamic_extent>
	constexpr KAT_HD auto
	subspan() const noexcept
	-> span<element_type, S_subspan_extent<Offset, Count>()>
	{
	#if __cplusplus >= 201703L
		if constexpr (Extent == dynamic_extent) {
			if constexpr (Offset > 0) {
				assert(Offset <= size());
			}
		}
		else
		static_assert(Offset <= extent);
	#else
		if (Extent == dynamic_extent) {
			assert((Offset < size() or Offset == size()) and "Subspan offset exceeds span size");
		}
		else{
			assert((Offset < Extent or Offset == Extent) and "Subspan offset exceeds fixed span extent");
		}
	#endif

	#if __cplusplus >= 201703L
		if constexpr (Extent == dynamic_extent)
		{
			if constexpr (Count > 0) {
				assert(Count <= size());
				assert(Count <= (size() - Offset));
			}
		}
		else
		{
			static_assert(Count <= extent);
			static_assert(Count <= (extent - Offset));
		}
	#else
		if (Extent == dynamic_extent) {
			// Note: not using <= comparisons in the assertions to avoid compiler warnings
			assert((size() > Count or size() == Count ) and "Count exceeds span size");
			assert( (Count < (size() - Offset) or Count == (size() - Offset)) and "Subspan ends past the span's end");
		}
		else {
			assert((Count <= extent) and "Count exceeds span extent");
			assert(Count <= (extent - Offset) and  "Subspan ends past the span's extent");
		}
	#endif

		return {data() + Offset, Count};
	}

	template<std::size_t Offset>
	constexpr KAT_HD auto
	subspan<Offset, dynamic_extent>() const noexcept
	-> span<element_type, S_subspan_extent<Offset, dynamic_extent>()>
	{
	#if __cplusplus >= 201703L
		if constexpr (Extent == dynamic_extent)
			assert(Offset <= size());
		else
			static_assert(Offset <= extent);
	#else
		if (Extent == dynamic_extent) {
			assert(Offset <= size() and "Subspan offset exceeds span size");
		}
		else {
			assert((Offset <= Extent) and  "Subspan offset exceeds fixed span extent");
		}
	#endif
		return {data() + Offset, size() - Offset};
	}

	constexpr KAT_HD span<element_type, dynamic_extent>
	subspan(size_type offset, size_type count = dynamic_extent) const
	noexcept
	{
		assert(offset <= size());
		if (count == dynamic_extent) {
			count = size() - offset;
		}
		else
		{
			assert(count <= size());
			assert(offset + count <= size());
		}
		return {data() + offset, count};
	}

private:
#if __cplusplus >= 202001L
	[[no_unique_address]]
#endif
	 detail::extent_storage<extent> extent_;
	pointer ptr_;
}; // class span

#if __cplusplus < 201703L
template<typename Type, std::size_t Extent>
constexpr const std::size_t span<Type, Extent>::extent;
#endif


#if __cplusplus >= 201703L
  // deduction guides
  template<typename Type, std::size_t ArrayExtent>
    span(Type(&)[ArrayExtent]) -> span<Type, ArrayExtent>;

  template<typename Type, std::size_t ArrayExtent>
    span(kat::array<Type, ArrayExtent>&) -> span<Type, ArrayExtent>;

  template<typename Type, std::size_t ArrayExtent>
    span(std::array<Type, ArrayExtent>&) -> span<Type, ArrayExtent>;

  template<typename Type, std::size_t ArrayExtent>
    span(const kat::array<Type, ArrayExtent>&)
      -> span<const Type, ArrayExtent>;

  template<typename Type, std::size_t ArrayExtent>
    span(const std::array<Type, ArrayExtent>&)
      -> span<const Type, ArrayExtent>;

// These two definitions  probably requires the ranges library
//
//  template<contiguous_iterator Iter, typename Sentinel>
//    span(Iter, Sentinel)
//      -> span<std::remove_reference_t<ranges::range_reference_t<Iter>>>;
//
//  template<typename Range>
//    span(Range &&)
//      -> span<std::remove_reference_t<ranges::range_reference_t<Range&>>>;
#endif

#if __cplusplus >= 201703L
template<typename Type, std::size_t Extent>
inline KAT_HD span<const std::byte, Extent == dynamic_extent ? dynamic_extent : Extent * sizeof(Type)> as_bytes(
	span<Type, Extent> sp) noexcept
{
	return {reinterpret_cast<const std::byte*>(sp.data()), sp.size_bytes()};
}

template<typename Type, std::size_t Extent>
inline KAT_HD span<std::byte, Extent == dynamic_extent ? dynamic_extent : Extent * sizeof(Type)> as_writable_bytes(
	span<Type, Extent> sp) noexcept
{
	return {reinterpret_cast<std::byte*>(sp.data()), sp.size_bytes()};
}
#endif

// The following code is useful for pre-C++20 code, where
// the span CTAD is unavailable

// TODO: Fixed-extent make_span's

template<class Type>
constexpr KAT_HD
span<Type> make_span(Type* first, Type* last)
{
	return span<Type>(first, last);
}

template<class Type>
constexpr KAT_HD
span<Type> make_span(Type* ptr, std::size_t count)
{
	return span<Type>(ptr, count);
}

template<class Type, std::size_t Extent>
constexpr KAT_HD
span<Type, Extent> make_span(Type (&arr)[Extent])
{
	return span<Type, Extent>(arr);
}

template<class Type, std::size_t Extent>
constexpr KAT_HD
span<const Type, Extent>
make_span(const kat::array<Type, Extent>& arr)
{
	return span<const Type, Extent>(arr);
}

template<class Type, std::size_t Extent>
constexpr KAT_HD
span<Type, Extent>
make_span(kat::array<Type, Extent>& arr)
{
	return span<Type, Extent>(arr);
}

template<class Container>
constexpr KAT_HD
span<typename Container::value_type>
make_span(Container &container)
{
	return span<typename Container::value_type>(container);
}

template<class Container>
constexpr KAT_HD
span<const typename Container::value_type>
make_span(const Container &container)
{
	return span<const typename Container::value_type>(container);
}

template<class Pointer>
constexpr KAT_HD
span<typename Pointer::element_type>
make_span(Pointer& container, std::size_t count)
{
	return span<typename Pointer::element_type>(container, count);
}

template<class Pointer>
constexpr KAT_HD
span<typename Pointer::element_type>
make_span(Pointer& container)
{
	return span<typename Pointer::element_type>(container);
}

} // namespace kat

namespace std {

// tuple helpers
template<std::size_t Index, typename Type, std::size_t Extent>
constexpr KAT_HD Type&
get(kat::span<Type, Extent> sp) noexcept
{
	static_assert(Extent != kat::dynamic_extent and Index < Extent,
		"get<I> can only be used with a span of non-dynamic (fixed) extent");
	return sp[Index];
}

template<typename T> struct tuple_size;
template<std::size_t i, typename T> struct tuple_element;

template<typename Type, std::size_t Extent>
struct tuple_size<kat::span<Type, Extent>> : public std::integral_constant<std::size_t, Extent> {
	static_assert(Extent != kat::dynamic_extent, "tuple_size can only "
		"be used with a span of non-dynamic (fixed) extent");
};

template<std::size_t Index, typename Type, std::size_t Extent>
struct tuple_element<Index, kat::span<Type, Extent>> {
	static_assert(Extent != kat::dynamic_extent, "tuple_element can only "
		"be used with a span of non-dynamic (fixed) extent");
	static_assert(Index < Extent, "Index is less than Extent");
	using type = Type;
};

#if __cplusplus >= 202001L
namespace ranges
{
	template<typename> extern inline const bool enable_safe_range;
	// Opt-in to safe_range concept
	template<typename _ElementType, std::size_t Extent>
	inline constexpr KAT_HD bool
	enable_safe_range<kat::span<_ElementType, Extent>> = true;
}
#endif

} // namespace std

#endif // KAT_CONTAINERS_SPAN_HPP_
