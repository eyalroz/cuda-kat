/**
 * @file kat/containers/tuple.hpp
 *
 * @brief This file implements `kat::tuple`, an equivalent of C++11's
 * `std::tuple` which may be used both in host-side and CUDA-device-side
 * code.
 */


//
// Original code Copyright (c) Electronic Arts Inc. All rights reserved
// Modifications Copyright (c) 2020 Eyal Rozenberg.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Note: Retrieved from https://github.com/electronicarts/EASTL/ , master branch,
// on 2020-03-06.

#ifndef CUDA_KAT_TUPLE_HPP_
#define CUDA_KAT_TUPLE_HPP_

#include <kat/common.hpp>
#include <kat/utility.hpp>
#include <kat/reference_wrapper.hpp>

#include <functional>
#include <type_traits>
#include <tuple>

// Forward declarations - originally from EASTL/detail/tuple_fwd_decls.h
namespace kat {

template <typename... T>
class tuple;

template <typename Tuple>
class tuple_size;

template <size_t I, typename Tuple>
class tuple_element;

template <size_t I, typename Tuple>
using tuple_element_t = typename tuple_element<I, Tuple>::type;

// const typename for tuple_element_t, for when tuple or tuple_impl cannot itself be const
template <size_t I, typename Tuple>
using const_tuple_element_t = typename std::conditional<
	std::is_lvalue_reference<tuple_element_t<I, Tuple>>::value,
	typename std::add_lvalue_reference<const typename std::remove_reference<tuple_element_t<I, Tuple>>::type >,
						 const tuple_element_t<I, Tuple>
					>::type;

// get
template <size_t I, typename... Ts_>
KAT_HD tuple_element_t<I, tuple<Ts_...>>& get(tuple<Ts_...>& t);

template <size_t I, typename... Ts_>
KAT_HD const_tuple_element_t<I, tuple<Ts_...>>& get(const tuple<Ts_...>& t);

template <size_t I, typename... Ts_>
KAT_HD tuple_element_t<I, tuple<Ts_...>>&& get(tuple<Ts_...>&& t);

template <typename T, typename... ts_>
KAT_HD T& get(tuple<ts_...>& t);

template <typename T, typename... ts_>
KAT_HD const T& get(const tuple<ts_...>& t);

template <typename T, typename... ts_>
KAT_HD T&& get(tuple<ts_...>&& t);

} // namespace kat


//EA_DISABLE_VC_WARNING(4623) // warning C4623: default constructor was implicitly defined as deleted
//EA_DISABLE_VC_WARNING(4625) // warning C4625: copy constructor was implicitly defined as deleted
//EA_DISABLE_VC_WARNING(4510) // warning C4510: default constructor could not be generated

namespace kat
{
// non-recursive tuple implementation based on libc++ tuple implementation and description at
// http://mitchnull.blogspot.ca/2012/06/c11-tuple-implementation-details-part-1.html

// tuple_types helper
template <typename... Ts> struct tuple_types {};

// tuple_size helper
template <typename T> class tuple_size {};
template <typename T> class tuple_size<const T>          : public tuple_size<T> {};
template <typename T> class tuple_size<volatile T>       : public tuple_size<T> {};
template <typename T> class tuple_size<const volatile T> : public tuple_size<T> {};

template <typename... Ts> class tuple_size<tuple_types<Ts...>> : public std::integral_constant<size_t, sizeof...(Ts)> {};
template <typename... Ts> class tuple_size<tuple<Ts...>>      : public std::integral_constant<size_t, sizeof...(Ts)> {};

template <typename... Ts> class tuple_size<std::tuple<Ts...>>  : public std::integral_constant<size_t, sizeof...(Ts)> {};

// Originally from EASTL's tuple implementation in their <utility> header

template <typename T1, typename T2>
class tuple_size<std::pair<T1, T2>> : public std::integral_constant<size_t, 2> {};

template <typename T1, typename T2>
class tuple_size<const std::pair<T1, T2>> : public std::integral_constant<size_t, 2> {};

#if __cplusplus >= 201703L
	template <class T>
	constexpr size_t tuple_size_v = tuple_size<T>::value;
#endif

namespace detail
{
	template <typename TupleIndices, typename... Ts>
	struct tuple_impl;
} // namespace detail

template <typename Indices, typename... Ts>
class tuple_size<detail::tuple_impl<Indices, Ts...>> : public std::integral_constant<size_t, sizeof...(Ts)> { };

// tuple_element helper to be able to isolate a type given an index
template <size_t I, typename T>
class tuple_element { };

template <size_t I>
class tuple_element<I, tuple_types<>>
{
public:
	static_assert(I != I, "tuple_element index out of range");
};

template <typename H, typename... Ts>
class tuple_element<0, tuple_types<H, Ts...>>
{
public:
	using type = H;
};

template <size_t I, typename H, typename... Ts>
class tuple_element<I, tuple_types<H, Ts...>>
{
public:
	using type = tuple_element_t<I - 1, tuple_types<Ts...>>;
};

// specialization for tuple
template <size_t I, typename... Ts>
class tuple_element<I, tuple<Ts...>>
{
public:
	using type = tuple_element_t<I, tuple_types<Ts...>>;
};

// tuple-std-tuple compatibility adaptation of the above
template <size_t I, typename... Ts>
class tuple_element<I, std::tuple<Ts...>>
{
public:
	using type = typename std::tuple_element<I, std::tuple<Ts...>>::type;
};

template <size_t I, typename... Ts>
class tuple_element<I, const tuple<Ts...>>
{
public:
	using type = typename std::add_const<tuple_element_t<I, tuple_types<Ts...>>>::type;
};

// tuple-std-tuple compatibility adaptation of the above
template <size_t I, typename... Ts>
class tuple_element<I, const std::tuple<Ts...>>
{
public:
	using type = typename std::tuple_element<I, const std::tuple<Ts...>>::type;
};

template <size_t I, typename... Ts>
class tuple_element<I, volatile tuple<Ts...>>
{
public:
	using type = typename std::add_volatile<tuple_element_t<I, tuple_types<Ts...>>>::type;
};

// tuple-std-tuple compatibility adaptation of the above
template <size_t I, typename... Ts>
class tuple_element<I, volatile std::tuple<Ts...>>
{
public:
	using type = typename std::tuple_element<I, volatile std::tuple<Ts...>>::type;
};


template <size_t I, typename... Ts>
class tuple_element<I, const volatile tuple<Ts...>>
{
public:
	using type = typename std::add_cv<tuple_element_t<I, tuple_types<Ts...>>>::type;
};

// tuple-std-tuple compatibility adaptation of the above
template <size_t I, typename... Ts>
class tuple_element<I, const volatile std::tuple<Ts...>>
{
public:
	using type = typename std::tuple_element<I, const volatile std::tuple<Ts...>>::type;
};

// specialization for tuple_impl
template <size_t I, typename Indices, typename... Ts>
class tuple_element<I, detail::tuple_impl<Indices, Ts...>> : public tuple_element<I, tuple<Ts...>>
{
};

template <size_t I, typename Indices, typename... Ts>
class tuple_element<I, const detail::tuple_impl<Indices, Ts...>> : public tuple_element<I, const tuple<Ts...>>
{
};

template <size_t I, typename Indices, typename... Ts>
class tuple_element<I, volatile detail::tuple_impl<Indices, Ts...>> : public tuple_element<I, volatile tuple<Ts...>>
{
};

template <size_t I, typename Indices, typename... Ts>
class tuple_element<I, const volatile detail::tuple_impl<Indices, Ts...>> : public tuple_element<
																				 I, const volatile tuple<Ts...>>
{
};

// Originally from EASTL's tuple implementation in their <utility> header

template<typename T1, typename T2>
class tuple_element<0, std::pair<T1, T2>> {
public:
	typedef T1 type;
};

template<typename T1, typename T2>
class tuple_element<1, std::pair<T1, T2>> {
public:
	typedef T2 type;
};

template<typename T1, typename T2>
class tuple_element<0, const std::pair<T1, T2>> {
public:
	typedef const T1 type;
};

template<typename T1, typename T2>
class tuple_element<1, const std::pair<T1, T2>> {
public:
	typedef const T2 type;
};

// attempt to isolate index given a type
template <typename T, typename Tuple>
struct tuple_index
{
};

template <typename T>
struct tuple_index<T, tuple_types<>>
{
	typedef void DuplicateTypeCheck;
	tuple_index() = delete; // tuple_index should only be used for compile-time assistance, and never be instantiated
	static const size_t index = 0;
};

template <typename T, typename... TsRest>
struct tuple_index<T, tuple_types<T, TsRest...>>
{
	typedef int DuplicateTypeCheck;
	// after finding type T in the list of types, try to find type T in TsRest.
	// If we stumble back into this version of tuple_index, i.e. type T appears twice in the list of types, then DuplicateTypeCheck will be of type int, and the static_assert will fail.
	// If we don't, then we'll go through the version of tuple_index above, where all of the types have been exhausted, and DuplicateTypeCheck will be void.
	static_assert(std::is_void<typename tuple_index<T, tuple_types<TsRest...>>::DuplicateTypeCheck>::value, "duplicate type T in tuple_vector::get<T>(); unique types must be provided in declaration, or only use get<size_t>()");

	static const size_t index = 0;
};

template <typename T, typename TsHead, typename... TsRest>
struct tuple_index<T, tuple_types<TsHead, TsRest...>>
{
	typedef typename tuple_index<T, tuple_types<TsRest...>>::DuplicateTypeCheck DuplicateTypeCheck;
	static const size_t index = tuple_index<T, tuple_types<TsRest...>>::index + 1;
};

template <typename T, typename Indices, typename... Ts>
struct tuple_index<T, detail::tuple_impl<Indices, Ts...>> : public tuple_index<T, tuple_types<Ts...>>
{
};


namespace detail {

// swallow
//
// Provides a vessel to expand variadic packs.
//
template <typename... Ts>
constexpr KAT_HD void swallow(Ts&&...) {}


// tuple_leaf
//
template <size_t I, typename ValueType, bool IsEmpty = std::is_empty<ValueType>::value>
class tuple_leaf;


template <size_t I, typename ValueType, bool IsEmpty>
CONSTEXPR_SINCE_CPP_14 inline KAT_HD void swap(
	tuple_leaf<I, ValueType, IsEmpty>& a,
	tuple_leaf<I, ValueType, IsEmpty>& b
	) noexcept(noexcept(kat::swap(a.get_internal(),b.get_internal())))
{
	kat::swap(a.get_internal(), b.get_internal());
}

template <size_t I, typename ValueType, bool IsEmpty>
class tuple_leaf
{
public:
	KAT_HD tuple_leaf() : mValue() {}
	KAT_HD tuple_leaf(const tuple_leaf&) = default; // TODO: This might not work
	KAT_HD tuple_leaf& operator=(const tuple_leaf&) = delete;

	// We shouldn't need this explicit constructor as it should be handled by the template below but OSX clang
	// is_constructible type trait incorrectly gives false for is_constructible<T&&, T&&>::value
	KAT_HD explicit tuple_leaf(ValueType&& v) : mValue(std::move(v)) {}

	template <typename T, typename = typename std::enable_if<std::is_constructible<ValueType, T&&>::value>::type>
	KAT_HD explicit tuple_leaf(T&& t)
		: mValue(std::forward<T>(t))
	{
	}

	template <typename T>
	KAT_HD explicit tuple_leaf(const tuple_leaf<I, T>& t)
		: mValue(t.get_internal())
	{
	}

	template <typename T>
	KAT_HD tuple_leaf& operator=(T&& t)
	{
		mValue = std::forward<T>(t);
		return *this;
	}

	KAT_HD int swap(tuple_leaf& t)
	{
		kat::detail::swap(*this, t);
		return 0;
	}

	KAT_HD ValueType& get_internal() { return mValue; }
	KAT_HD const ValueType& get_internal() const { return mValue; }

private:
	ValueType mValue;
};

// tuple_leaf: Specialize for when ValueType is a reference
template <size_t I, typename ValueType, bool IsEmpty>
class tuple_leaf<I, ValueType&, IsEmpty>
{
public:
	tuple_leaf(const tuple_leaf&) = default;
	tuple_leaf& operator=(const tuple_leaf&) = delete;

	template <typename T, typename = typename std::enable_if<std::is_constructible<ValueType, T&&>::value>::type>
	KAT_HD explicit tuple_leaf(T&& t)
		: mValue(std::forward<T>(t))
	{
	}

	KAT_HD explicit tuple_leaf(ValueType& t) : mValue(t)
	{
	}

	template <typename T>
	KAT_HD explicit tuple_leaf(const tuple_leaf<I, T>& t)
		: mValue(t.get_internal())
	{
	}

	template <typename T>
	KAT_HD tuple_leaf& operator=(T&& t)
	{
		mValue = std::forward<T>(t);
		return *this;
	}

	KAT_HD int swap(tuple_leaf& t)
	{
		kat::detail::swap(*this, t);
		return 0;
	}

	KAT_HD ValueType& get_internal() { return mValue; }
	KAT_HD const ValueType& get_internal() const { return mValue; }

private:
	ValueType& mValue;
};

// tuple_leaf: partial specialization for when we can use the Empty Base Class Optimization
template <size_t I, typename ValueType>
class tuple_leaf<I, ValueType, true> : private ValueType
{
public:
	// std::true_type / std::false_type constructors for case where ValueType is default constructible and should be value
	// initialized and case where it is not
	tuple_leaf(const tuple_leaf&) = default;

	template <typename T, typename = typename std::enable_if<std::is_constructible<ValueType, T&&>::value>::type>
	KAT_HD explicit tuple_leaf(T&& t)
		: ValueType(std::forward<T>(t))
	{
	}

	template <typename T>
	KAT_HD explicit tuple_leaf(const tuple_leaf<I, T>& t)
		: ValueType(t.get_internal())
	{
	}

	template <typename T>
	KAT_HD tuple_leaf& operator=(T&& t)
	{
		ValueType::operator=(std::forward<T>(t));
		return *this;
	}

	KAT_HD int swap(tuple_leaf& t)
	{
		kat::detail::swap(*this, t);
		return 0;
	}

	KAT_HD ValueType& get_internal() { return static_cast<ValueType&>(*this); }
	KAT_HD const ValueType& get_internal() const { return static_cast<const ValueType&>(*this); }

private:
	KAT_HD tuple_leaf& operator=(const tuple_leaf&) = delete;
};



// make_tuple_types
//
//
template <typename tuple_types, typename Tuple, size_t Start, size_t End>
struct make_tuple_types_impl;

template <typename... Types, typename Tuple, size_t Start, size_t End>
struct make_tuple_types_impl<tuple_types<Types...>, Tuple, Start, End>
{
	typedef typename std::remove_reference<Tuple>::type tuple_type;
	typedef typename make_tuple_types_impl<
		tuple_types<Types..., typename std::conditional<std::is_lvalue_reference<Tuple>::value,
												  // append ref if Tuple is ref
												  tuple_element_t<Start, tuple_type>&,
												  // append non-ref otherwise
												  tuple_element_t<Start, tuple_type>>::type>,
		Tuple, Start + 1, End>::type type;
};

template <typename... Types, typename Tuple, size_t End>
struct make_tuple_types_impl<tuple_types<Types...>, Tuple, End, End>
{
	typedef tuple_types<Types...> type;
};

template <typename Tuple>
using make_tuple_types_t = typename make_tuple_types_impl<tuple_types<>, Tuple, 0,
													 tuple_size<typename std::remove_reference<Tuple>::type>::value>::type;


// tuple_impl
//
//
template <size_t I, typename Indices, typename... Ts>
inline KAT_HD tuple_element_t<I, tuple_impl<Indices, Ts...>>& get(tuple_impl<Indices, Ts...>& t);

template <size_t I, typename Indices, typename... Ts>
inline KAT_HD const_tuple_element_t<I, tuple_impl<Indices, Ts...>>& get(const tuple_impl<Indices, Ts...>& t);

template <size_t I, typename Indices, typename... Ts>
KAT_HD tuple_element_t<I, tuple_impl<Indices, Ts...>>&& get(tuple_impl<Indices, Ts...>&& t);

template <typename T, typename Indices, typename... Ts>
KAT_HD T& get(tuple_impl<Indices, Ts...>& t);

template <typename T, typename Indices, typename... Ts>
KAT_HD const T& get(const tuple_impl<Indices, Ts...>& t);

template <typename T, typename Indices, typename... Ts>
KAT_HD T&& get(tuple_impl<Indices, Ts...>&& t);

template <size_t... Indices, typename... Ts>
struct tuple_impl<std::integer_sequence<size_t, Indices...>, Ts...> : public tuple_leaf<Indices, Ts>...
{
	tuple_impl() = default; // TODO: Probably won't work

	// index_sequence changed to integer_sequence due to issues described below in VS2015 CTP 6.
	// https://connect.microsoft.com/VisualStudio/feedback/details/1126958/error-in-template-parameter-pack-expansion-of-std-index-sequence
	//
	template <typename... Us, typename... ValueTypes>
	explicit KAT_HD tuple_impl(std::integer_sequence<size_t, Indices...>, tuple_types<Us...>, ValueTypes&&... values)
		: tuple_leaf<Indices, Ts>(std::forward<ValueTypes>(values))...
	{
	}

	template <typename OtherTuple>
	KAT_HD tuple_impl(OtherTuple&& t)
		: tuple_leaf<Indices, Ts>(std::forward<tuple_element_t<Indices, make_tuple_types_t<OtherTuple>>>(get<Indices>(t)))...
	{
	}

	template <typename OtherTuple>
	KAT_HD tuple_impl& operator=(OtherTuple&& t)
	{
		swallow(tuple_leaf<Indices, Ts>::operator=(
			std::forward<tuple_element_t<Indices, make_tuple_types_t<OtherTuple>>>(get<Indices>(t)))...);
		return *this;
	}

	KAT_HD tuple_impl& operator=(const tuple_impl& t)
	{
		swallow(tuple_leaf<Indices, Ts>::operator=(static_cast<const tuple_leaf<Indices, Ts>&>(t).get_internal())...);
		return *this;
	}

	KAT_HD void swap(tuple_impl& t) { swallow(tuple_leaf<Indices, Ts>::swap(static_cast<tuple_leaf<Indices, Ts>&>(t))...); }

};

template <size_t I, typename Indices, typename... Ts>
inline KAT_HD tuple_element_t<I, tuple_impl<Indices, Ts...>>& get(tuple_impl<Indices, Ts...>& t)
{
	typedef tuple_element_t<I, tuple_impl<Indices, Ts...>> Type;
	return static_cast<detail::tuple_leaf<I, Type>&>(t).get_internal();
}

template <size_t I, typename Indices, typename... Ts>
inline KAT_HD const_tuple_element_t<I, tuple_impl<Indices, Ts...>>& get(const tuple_impl<Indices, Ts...>& t)
{
	typedef tuple_element_t<I, tuple_impl<Indices, Ts...>> Type;
	return static_cast<const detail::tuple_leaf<I, Type>&>(t).get_internal();
}

template <size_t I, typename Indices, typename... Ts>
inline KAT_HD tuple_element_t<I, tuple_impl<Indices, Ts...>>&& get(tuple_impl<Indices, Ts...>&& t)
{
	typedef tuple_element_t<I, tuple_impl<Indices, Ts...>> Type;
	return static_cast<Type&&>(static_cast<detail::tuple_leaf<I, Type>&>(t).get_internal());
}

template <typename T, typename Indices, typename... Ts>
KAT_HD T& get(tuple_impl<Indices, Ts...>& t)
{
	typedef tuple_index<T, tuple_impl<Indices, Ts...>> Index;
	return static_cast<detail::tuple_leaf<Index::index, T>&>(t).get_internal();
}

template <typename T, typename Indices, typename... Ts>
inline KAT_HD const T& get(const tuple_impl<Indices, Ts...>& t)
{
	typedef tuple_index<T, tuple_impl<Indices, Ts...>> Index;
	return static_cast<const detail::tuple_leaf<Index::index, T>&>(t).get_internal();
}

template <typename T, typename Indices, typename... Ts>
inline KAT_HD T&& get(tuple_impl<Indices, Ts...>&& t)
{
	typedef tuple_index<T, tuple_impl<Indices, Ts...>> Index;
	return static_cast<T&&>(static_cast<detail::tuple_leaf<Index::index, T>&>(t).get_internal());
}

template <size_t... Indices, typename... Ts>
inline KAT_HOST std::tuple<Ts...> as_std_tuple(tuple_impl<std::integer_sequence<size_t, Indices...>, Ts...> ti);

// tuple_like
//
// type-trait that determines if a type is an eastl::tuple or an eastl::pair.
//
// TODO: Do we really need these for anything?
template <typename T> struct tuple_like                   : public std::false_type {};
template <typename T> struct tuple_like<const T>          : public tuple_like<T> {};
template <typename T> struct tuple_like<volatile T>       : public tuple_like<T> {};
template <typename T> struct tuple_like<const volatile T> : public tuple_like<T> {};

template <typename... Ts>
struct tuple_like<tuple<Ts...>> : public std::true_type {};

// tuple-std-tuple compatibility adaptation of the above
template <typename... Ts>
struct tuple_like<std::tuple<Ts...>> : public std::true_type {};

template <typename First, typename Second>
struct tuple_like<std::pair<First, Second>> : public std::true_type {};

// kat::pair ?
//template <typename First, typename Second>
//struct tuple_like<kat::pair<First, Second>> : public std::true_type {};


// tuple_convertible
//
//
//
template <bool IsSameSize, typename From, typename To>
struct tuple_convertible_impl : public std::false_type
{
};

template <typename... FromTypes, typename... ToTypes>
struct tuple_convertible_impl<true, tuple_types<FromTypes...>,	tuple_types<ToTypes...>>
	: public kat::bool_constant<conjunction<std::is_convertible<FromTypes, ToTypes>...>::value>
{
};

template <typename From, typename To,
		  bool = tuple_like<typename std::remove_reference<From>::type>::value,
		  bool = tuple_like<typename std::remove_reference<To>::type>::value>
struct tuple_convertible : public std::false_type
{
};

template <typename From, typename To>
struct tuple_convertible<From, To, true, true>
	: public tuple_convertible_impl<tuple_size<typename std::remove_reference<From>::type>::value ==
			tuple_size<typename std::remove_reference<To>::type>::value,
			make_tuple_types_t<From>, make_tuple_types_t<To>>
{
};


// tuple_assignable
//
//
//
template <bool IsSameSize, typename Target, typename From>
struct tuple_assignable_impl : public std::false_type
{
};

template <typename... TargetTypes, typename... FromTypes>
struct tuple_assignable_impl<true, tuple_types<TargetTypes...>, tuple_types<FromTypes...>>
	: public bool_constant<conjunction<std::is_assignable<TargetTypes, FromTypes>...>::value>
{
};

template <typename Target, typename From,
		  bool = tuple_like<typename std::remove_reference<Target>::type>::value,
		  bool = tuple_like<typename std::remove_reference<From>::type>::value>
struct tuple_assignable : public std::false_type
{
};

template <typename Target, typename From>
struct tuple_assignable<Target, From, true, true>
	: public tuple_assignable_impl<
		tuple_size<typename std::remove_reference<Target>::type>::value ==
		tuple_size<typename std::remove_reference<From>::type>::value,
		make_tuple_types_t<Target>, make_tuple_types_t<From>>
{
	using parent_type = tuple_assignable_impl<
		tuple_size<typename std::remove_reference<Target>::type>::value ==
		tuple_size<typename std::remove_reference<From>::type>::value,
		make_tuple_types_t<Target>, make_tuple_types_t<From>>;
};


// tuple_implicitly_convertible and tuple_explicitly_convertible
//
// helpers for constraining conditionally-explicit ctors
//
template <bool IsSameSize, typename TargetType, typename... FromTypes>
struct tuple_implicitly_convertible_impl : public std::false_type
{
};


template <typename... TargetTypes, typename... FromTypes>
struct tuple_implicitly_convertible_impl<true, tuple_types<TargetTypes...>, FromTypes...>
	: public conjunction<
	std::is_constructible<TargetTypes, FromTypes>...,
	std::is_convertible<FromTypes, TargetTypes>...>
{
};

template <typename TargetTupleType, typename... FromTypes>
struct tuple_implicitly_convertible
	: public tuple_implicitly_convertible_impl<
	tuple_size<TargetTupleType>::value == sizeof...(FromTypes),
	make_tuple_types_t<TargetTupleType>, FromTypes...>::type
{
};

template<typename TargetTupleType, typename... FromTypes>
using tuple_implicitly_convertible_t = std::enable_if_t<tuple_implicitly_convertible<TargetTupleType, FromTypes...>::value, bool>;

template <bool IsSameSize, typename TargetType, typename... FromTypes>
struct tuple_explicitly_convertible_impl : public std::false_type
{
};

template <typename... TargetTypes, typename... FromTypes>
struct tuple_explicitly_convertible_impl<true, tuple_types<TargetTypes...>, FromTypes...>
	: public conjunction<
		std::is_constructible<TargetTypes, FromTypes>...,
		negation<conjunction<std::is_convertible<FromTypes, TargetTypes>...>>>
{
};

template <typename TargetTupleType, typename... FromTypes>
struct tuple_explicitly_convertible
	: public tuple_explicitly_convertible_impl<
	tuple_size<TargetTupleType>::value == sizeof...(FromTypes),
	make_tuple_types_t<TargetTupleType>, FromTypes...>::type
{
};

template<typename TargetTupleType, typename... FromTypes>
using tuple_explicitly_convertible_t = std::enable_if_t<tuple_explicitly_convertible<TargetTupleType, FromTypes...>::value, bool>;


// tuple_equal
//
//
//
template <size_t I>
struct tuple_equal
{
	template <typename Tuple1, typename Tuple2>
	KAT_HD bool operator()(const Tuple1& t1, const Tuple2& t2)
	{
		static_assert(tuple_size<Tuple1>::value == tuple_size<Tuple2>::value, "comparing tuples of different sizes.");
		return tuple_equal<I - 1>()(t1, t2) && get<I - 1>(t1) == get<I - 1>(t2);
	}
};

template <>
struct tuple_equal<0>
{
	template <typename Tuple1, typename Tuple2>
	KAT_HD bool operator()(const Tuple1&, const Tuple2&)
	{
		return true;
	}
};


// tuple_less
//
//
//
template <size_t I>
struct tuple_less
{
	template <typename Tuple1, typename Tuple2>
	KAT_HD bool operator()(const Tuple1& t1, const Tuple2& t2)
	{
		static_assert(tuple_size<Tuple1>::value == tuple_size<Tuple2>::value, "comparing tuples of different sizes.");
		return tuple_less<I - 1>()(t1, t2) || (!tuple_less<I - 1>()(t2, t1) && get<I - 1>(t1) < get<I - 1>(t2));
	}
};

template <>
struct tuple_less<0>
{
	template <typename Tuple1, typename Tuple2>
	KAT_HD bool operator()(const Tuple1&, const Tuple2&)
	{
		return false;
	}
};


// MakeTupleReturnImpl
//
//
//
template <typename T> struct MakeTupleReturnImpl                       { typedef T type; };
template <typename T> struct MakeTupleReturnImpl<reference_wrapper<T>> { typedef T& type; };

template <typename T>
using make_tuple_return_t = typename MakeTupleReturnImpl<typename std::decay<T>::type>::type;


// tuple_cat helpers
//
//
//

// tuple_cat_2_impl
template <typename Tuple1, typename Is1, typename Tuple2, typename Is2>
struct tuple_cat_2_impl;

template <typename... T1s, size_t... I1s, typename... T2s, size_t... I2s>
struct tuple_cat_2_impl<tuple<T1s...>, index_sequence<I1s...>, tuple<T2s...>, index_sequence<I2s...>>
{
	using result_type = tuple<T1s..., T2s...>;

	template <typename Tuple1, typename Tuple2>
	static KAT_HD result_type do_cat_2(Tuple1&& t1, Tuple2&& t2)
	{
		return result_type(get<I1s>(std::forward<Tuple1>(t1))..., get<I2s>(std::forward<Tuple2>(t2))...);
	}
};

// tuple_cat_2
template <typename Tuple1, typename Tuple2>
struct tuple_cat_2 { };

template <typename... T1s, typename... T2s>
struct tuple_cat_2<tuple<T1s...>, tuple<T2s...>>
{
	using Is1        = make_index_sequence<sizeof...(T1s)>;
	using Is2        = make_index_sequence<sizeof...(T2s)>;
	using tci_type = tuple_cat_2_impl<tuple<T1s...>, Is1, tuple<T2s...>, Is2>;
	using result_type = typename tci_type::result_type;

	template <typename Tuple1, typename Tuple2>
	static inline KAT_HD result_type do_cat_2(Tuple1&& t1, Tuple2&& t2)
	{
		return tci_type::do_cat_2(std::forward<Tuple1>(t1), std::forward<Tuple2>(t2));
	}
};

// tuple_cat
template <typename... Tuples>
struct tuple_cat;

template <typename Tuple1, typename Tuple2, typename... TuplesRest>
struct tuple_cat<Tuple1, Tuple2, TuplesRest...>
{
	using first_result_type = typename tuple_cat_2<Tuple1, Tuple2>::result_type;
	using result_type      = typename tuple_cat<first_result_type, TuplesRest...>::result_type;

	template <typename TupleArg1, typename TupleArg2, typename... TupleArgsRest>
	static inline KAT_HD result_type do_cat(TupleArg1&& t1, TupleArg2&& t2, TupleArgsRest&&... ts)
	{
		return tuple_cat<first_result_type, TuplesRest...>::do_cat(
			tuple_cat_2<TupleArg1, TupleArg2>::do_cat_2(std::forward<TupleArg1>(t1), std::forward<TupleArg2>(t2)),
			std::forward<TupleArgsRest>(ts)...);
	}
};

template <typename Tuple1, typename Tuple2>
struct tuple_cat<Tuple1, Tuple2>
{
	using tc2_type = tuple_cat_2<Tuple1, typename std::remove_reference<Tuple2>::type>;
	using result_type = typename tc2_type::result_type;

	template <typename TupleArg1, typename TupleArg2>
	static KAT_HD result_type do_cat(TupleArg1&& t1, TupleArg2&& t2)
	{
		return tc2_type::do_cat_2(std::forward<TupleArg1>(t1), std::forward<TupleArg2>(t2));
	}
};

}  // namespace detail



/**
 * @brief A container of heterogenenous-type values, of a fixed-size,
 * for use in host-side and CUDA-device-side code.
 *
 * see https://en.cppreference.com/w/cpp/utility/tuple
 */
template <typename T, typename... Ts>
class tuple<T, Ts...>
{
public:
	tuple() = default; // TODO: This won't work

	template <typename T2 = T,
		detail::tuple_implicitly_convertible_t<tuple, const T2&, const Ts&...> = 0>
	constexpr KAT_HD tuple(const T& t, const Ts&... ts)
		: impl_(make_index_sequence<sizeof...(Ts) + 1>{}, detail::make_tuple_types_t<tuple>{}, t, ts...)
	{
	}

	template <typename T2 = T,
		detail::tuple_explicitly_convertible_t<tuple, const T2&, const Ts&...> = 0>
	explicit constexpr KAT_HD tuple(const T& t, const Ts&... ts)
		: impl_(make_index_sequence<sizeof...(Ts) + 1>{}, detail::make_tuple_types_t<tuple>{}, t, ts...)
	{
	}

	template <typename U, typename... Us,
		detail::tuple_implicitly_convertible_t<tuple, U, Us...> = 0>
		constexpr KAT_HD tuple(U&& u, Us&&... us)
		: impl_(make_index_sequence<sizeof...(Us) + 1>{}, detail::make_tuple_types_t<tuple>{}, std::forward<U>(u),
			std::forward<Us>(us)...)
	{
	}

	template <typename U, typename... Us,
		detail::tuple_explicitly_convertible_t<tuple, U, Us...> = 0>
		explicit constexpr KAT_HD tuple(U&& u, Us&&... us)
		: impl_(make_index_sequence<sizeof...(Us) + 1>{}, detail::make_tuple_types_t<tuple>{}, std::forward<U>(u),
			std::forward<Us>(us)...)
	{
	}

	template <typename OtherTuple,
			  typename std::enable_if<detail::tuple_convertible<OtherTuple, tuple>::value, bool>::type = false>
	KAT_HD tuple(OtherTuple&& t)
		: impl_(std::forward<OtherTuple>(t))
	{
	}

	template <typename OtherTuple,
			  typename std::enable_if<detail::tuple_assignable<tuple, OtherTuple>::value, bool>::type = false>
	KAT_HD tuple& operator=(OtherTuple&& t)
	{
		impl_.operator=(std::forward<OtherTuple>(t));
		return *this;
	}

	KAT_HD void swap(tuple& t) { impl_.swap(t.impl_); }

	// TODO: Perhaps make this explicit?
	operator std::tuple<T, Ts...>() const {
		return detail::as_std_tuple(impl_);
	}

private:
	typedef detail::tuple_impl<kat::make_index_sequence<sizeof...(Ts) + 1>, T, Ts...> impl_type;
	impl_type impl_;

	template <size_t I, typename... Ts_>
	friend KAT_HD tuple_element_t<I, tuple<Ts_...>>& get(tuple<Ts_...>& t);

	template <size_t I, typename... Ts_>
	friend KAT_HD const_tuple_element_t<I, tuple<Ts_...>>& get(const tuple<Ts_...>& t);

	template <size_t I, typename... Ts_>
	friend KAT_HD tuple_element_t<I, tuple<Ts_...>>&& get(tuple<Ts_...>&& t);

	template <typename T_, typename... ts_>
	friend KAT_HD T_& get(tuple<ts_...>& t);

	template <typename T_, typename... ts_>
	friend KAT_HD const T_& get(const tuple<ts_...>& t);

	template <typename T_, typename... ts_>
	friend KAT_HD T_&& get(tuple<ts_...>&& t);
};

// template specialization for an empty tuple
template <>
class tuple<>
{
public:
	KAT_HD void swap(tuple&) {}

	// TODO: Perhaps make this explicit?
	operator std::tuple<>() const
	{
		return std::tuple<>();
	}
};

template <size_t I, typename... Ts>
inline KAT_HD tuple_element_t<I, tuple<Ts...>>& get(tuple<Ts...>& t)
{
	return get<I>(t.impl_);
}

template <size_t I, typename... Ts>
inline KAT_HD const_tuple_element_t<I, tuple<Ts...>>& get(const tuple<Ts...>& t)
{
	return get<I>(t.impl_);
}

template <size_t I, typename... Ts>
inline KAT_HD tuple_element_t<I, tuple<Ts...>>&& get(tuple<Ts...>&& t)
{
	return get<I>(std::move(t.impl_));
}

template <typename T, typename... Ts>
inline KAT_HD T& get(tuple<Ts...>& t)
{
	return get<T>(t.impl_);
}

template <typename T, typename... Ts>
inline KAT_HD const T& get(const tuple<Ts...>& t)
{
	return get<T>(t.impl_);
}

template <typename T, typename... Ts>
inline KAT_HD T&& get(tuple<Ts...>&& t)
{
	return get<T>(std::move(t.impl_));
}

template <typename... Ts>
inline KAT_HD void swap(tuple<Ts...>& a, tuple<Ts...>& b)
{
	a.swap(b);
}


// tuple operators
//
//
template <typename... T1s, typename... T2s>
inline KAT_HD bool operator==(const tuple<T1s...>& t1, const tuple<T2s...>& t2)
{
	return detail::tuple_equal<sizeof...(T1s)>()(t1, t2);
}

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s>
KAT_HOST bool operator==(const tuple<T1s...>& t1, const std::tuple<T2s...>& t2)
{
	return detail::tuple_equal<sizeof...(T1s)>()(t1, tuple<T2s...>{t2});
}

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s>
KAT_HOST bool operator==(const std::tuple<T1s...>& t1, const tuple<T2s...>& t2)
{
	return detail::tuple_equal<sizeof...(T1s)>()(t1, tuple<T2s...>{t2});
}



template <typename... T1s, typename... T2s>
inline KAT_HD bool operator<(const tuple<T1s...>& t1, const tuple<T2s...>& t2)
{
	return detail::tuple_less<sizeof...(T1s)>()(t1, t2);
}

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s>
KAT_HOST bool operator<(const tuple<T1s...>& t1, const std::tuple<T2s...>& t2)
{
	return detail::tuple_less<sizeof...(T1s)>()(t1, t2);
}

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s>
KAT_HOST bool operator<(const std::tuple<T1s...>& t1, const tuple<T2s...>& t2)
{
	return detail::tuple_less<sizeof...(T1s)>()(t1, t2);
}

template <typename... T1s, typename... T2s> inline KAT_HD bool operator!=(const tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t1 == t2); }
template <typename... T1s, typename... T2s> inline KAT_HD bool operator> (const tuple<T1s...>& t1, const tuple<T2s...>& t2) { return t2 < t1; }
template <typename... T1s, typename... T2s> inline KAT_HD bool operator<=(const tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t2 < t1); }
template <typename... T1s, typename... T2s> inline KAT_HD bool operator>=(const tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t1 < t2); }

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator!=(const tuple<T1s...>& t1, const std::tuple<T2s...>& t2) { return !(t1 == t2); }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator> (const tuple<T1s...>& t1, const std::tuple<T2s...>& t2) { return t2 < t1; }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator<=(const tuple<T1s...>& t1, const std::tuple<T2s...>& t2) { return !(t2 < t1); }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator>=(const tuple<T1s...>& t1, const std::tuple<T2s...>& t2) { return !(t1 < t2); }

// tuple-std-tuple adaptation of the above
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator!=(const std::tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t1 == t2); }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator> (const std::tuple<T1s...>& t1, const tuple<T2s...>& t2) { return t2 < t1; }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator<=(const std::tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t2 < t1); }
template <typename... T1s, typename... T2s> inline KAT_HOST bool operator>=(const std::tuple<T1s...>& t1, const tuple<T2s...>& t2) { return !(t1 < t2); }


// tuple_cat
//
//
template <typename... Tuples>
inline KAT_HD typename detail::tuple_cat<Tuples...>::result_type tuple_cat(Tuples&&... ts)
{
	return detail::tuple_cat<Tuples...>::do_cat(std::forward<Tuples>(ts)...);
}


// make_tuple
//
//
template <typename... Ts>
inline KAT_HD constexpr tuple<detail::make_tuple_return_t<Ts>...> make_tuple(Ts&&... values)
{
	return tuple<detail::make_tuple_return_t<Ts>...>(std::forward<Ts>(values)...);
}


// forward_as_tuple
//
//
template <typename... Ts>
inline KAT_HD constexpr tuple<Ts&&...> forward_as_tuple(Ts&&... ts) noexcept
{
	return tuple<Ts&&...>(std::forward<Ts&&>(ts)...);
}

namespace detail {

template <size_t... Indices, typename... Ts>
inline KAT_HOST std::tuple<Ts...> as_std_tuple(tuple_impl<std::integer_sequence<size_t, Indices...>, Ts...> ti)
{
	return std::make_tuple(get<Indices>(ti)...);
}

} // namespace detail

namespace detail {
// ignore
//
// An object of unspecified type such that any value can be assigned to it with no effect.
//
// https://en.cppreference.com/w/cpp/utility/tuple/ignore
//
template <class U>
struct ignore_t
{
    template <class T> KAT_HD constexpr const ignore_t& operator=(T&&) const { return *this; }
};
} // namespace detail

namespace {
// Note: This will probably fail in device side code...
static constexpr const detail::ignore_t<unsigned char> ignore { };
// So you might want to use:
// __device__ static const detail::ignore_t<unsigned char> device_ignore { };
// but we won't enable that here - for now
}

/**
 * @brief Creates a tuple of lvalue references to its arguments or instances of kat::ignore.
 *
 * @see https://en.cppreference.com/w/cpp/utility/tuple/tie
 */
template <typename... Ts>
inline KAT_HD constexpr tuple<Ts&...> tie(Ts&... ts) noexcept
{
	return tuple<Ts&...>(ts...);
}

#if __cplusplus >=  201703L

// TODO: Implement a kat::invoke (and perhaps a kat::apply) function
#if FALSE

// apply
//
// Invoke a callable object using a tuple to supply the arguments.
//
// http://en.cppreference.com/w/cpp/utility/apply
//
namespace detail
{

template <class F, class Tuple, size_t... I>
constexpr KAT_HD decltype(auto) apply_impl(F&& f, Tuple&& t, index_sequence<I...>)
{
#ifndef KAT_DEFINE_MOVE_AND_FORWARD
	using std::forward;
#endif
	// TODO: Will std::invoke really work here? I doubt it.
	return std::invoke(forward<F>(f), get<I>(forward<Tuple>(t))...);
}

} // namespace detail

template <class F, class Tuple>
constexpr KAT_HD decltype(auto) apply(F&& f, Tuple&& t)
{
#ifndef KAT_DEFINE_MOVE_AND_FORWARD
	using std::forward;
#endif
	return detail::apply_impl(
		forward<F>(f), forward<Tuple>(t),
		make_index_sequence<tuple_size_v<typename std::remove_reference<Tuple>::type>>{});
}

#endif

#endif // no kat::invoke implementation

}  // namespace kat


///////////////////////////////////////////////////////////////
// C++17 structured bindings support for kat::tuple
//
#if __cplusplus >=  201703L

#include <tuple>
namespace std {

// NOTE(rparolin): Some platform implementations didn't check the standard specification and implemented the
// "tuple_size" and "tuple_element" primary template with as a struct.  The standard specifies they are
// implemented with the class keyword so we provide the template specializations as a class and disable the
// generated warning.

// EA_DISABLE_CLANG_WARNING(-Wmismatched-tags)

template <class... Ts>
class tuple_size<::kat::tuple<Ts...>> : public std::integral_constant<size_t, sizeof...(Ts)>
{
};

template <size_t I, class... Ts>
class tuple_element<I, ::kat::tuple<Ts...>> : public ::kat::tuple_element<I, ::kat::tuple<Ts...>>
{
};

//	EA_RESTORE_CLANG_WARNING()

} // namespace std

#endif // __cplusplus >=  201703L

#endif  // CUDA_KAT_TUPLE_HPP_
