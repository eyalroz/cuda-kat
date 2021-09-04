/**
 * @file kat/pair.hpp
 *
 * @brief This file implements `kat::par`, an equivalent of C++'s pair class
 * which may be used both in host-side and CUDA-device-side code, along with
 * some supporting functions and overloaded operators for that class.
 */

#ifndef CUDA_KAT_PAIR_HPP_
#define CUDA_KAT_PAIR_HPP_

#include <kat/common.hpp>
#include <kat/detail/range_access.hpp>
#include <kat/detail/constexpr_by_cpp_version.hpp>
#include <kat/detail/cpp14_type_traits.hpp>
#include <kat/detail/cpp17_type_traits.hpp>
#include <kat/detail/integer_sequence.hpp>
#include <kat/utility.hpp>

#include <type_traits>

namespace kat {

namespace detail {

// TODO: Move these upwards if we find more constructs requiring reference wrapper removal

// Helper which adds a reference to a type when given a reference_wrapper of that type.
template <typename T> struct remove_reference_wrapper                                    { typedef T  type; };
template <typename T> struct remove_reference_wrapper< std::reference_wrapper<T>       > { typedef T& type; };
template <typename T> struct remove_reference_wrapper< const std::reference_wrapper<T> > { typedef T& type; };

}

/**
 *  @brief A class compatible (*) with std::pair - but fully GPU device-side enabled.
 *
 *  @note More methods are marked as constexpr, when possible, than in the standard.
 */
template <typename T1, typename T2>
struct pair {

	using first_type = T1;
	using second_type = T2;
	using this_type = pair<T1, T2>;

	T1 first;
	T2 second;

public: // constructors
	template <typename TT1 = T1, typename TT2 = T2,
		class = kat::enable_if_t<std::is_default_constructible<TT1>::value and std::is_default_constructible<TT2>::value>>
	KAT_HD constexpr pair() : first(), second() { }

	// Note: Dropped EASTL pair's constructor which only takes the first element

	template <typename TT1 = T1, typename TT2 = T2,
		class = kat::enable_if_t<std::is_copy_constructible<TT1>::vale and std::is_copy_constructible<TT2>::value >>
	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(const T1& x, const T2& y)
		: first(x), second(y) { }

	// Consider using the C++23-style ctor.
	template <typename U, typename V,
		typename = kat::enable_if_t<std::is_convertible<U, T1>::value and std::is_convertible<V, T2>::value>>
	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(U&& u, V&& v)
		: first(std::forward<U>(u)), second(std::forward<V>(v)) { }

	template <typename U, typename V,
		class = kat::enable_if_t<std::is_convertible<const U&, T1>::value and std::is_convertible<const V&, T2>::value>>
	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(const pair<U, V>& p)
		: first(p.first), second(p.second) { }

	template <typename U, typename V,
		typename = kat::enable_if_t<std::is_convertible<U, T1>::value and std::is_convertible<V, T2>::value>>
		KAT_HD CONSTEXPR_SINCE_CPP_14 pair(pair<U, V>&& p)
		: first(std::forward<U>(p.first)), second(std::forward<V>(p.second)) { }

//		template <typename U, typename = kat::enable_if_t<std::is_convertible<U, T1>::value>>
//	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(U&& x, const T2& y)
//		: first(std::forward<U>(x)), second(y) { }
//
//		template <typename V, typename = kat::enable_if_t<std::is_convertible<V, T2>::value>>
//		KAT_HD CONSTEXPR_SINCE_CPP_14 pair(const T1& x, V&& y)
//		: first(x), second(std::forward<V>(y)) { }
		
	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(pair&& p) = default;
	KAT_HD CONSTEXPR_SINCE_CPP_14 pair(const pair&) = default;


/* TODO: Support these constructors

	// Initializes first with arguments of types Args1... obtained by forwarding the elements of first_args and
	// initializes second with arguments of types Args2... obtained by forwarding the elements of second_args.
	template <class... Args1, class... Args2,
		typename = kat::enable_if_t<std::is_constructible<first_type, Args1&&...>::value and
		std::is_constructible<second_type, Args2&&...>::value > >
	pair(eastl::piecewise_construct_t pwc, eastl::tuple<Args1...> first_args, eastl::tuple<Args2...> second_args)
		: pair(pwc, std::move(first_args), std::move(second_args),
			eastl::make_index_sequence<sizeof...(Args1)>(),
			eastl::make_index_sequence<sizeof...(Args2)>()) { }

private:
	// NOTE(rparolin): Internal constructor used to expand the index_sequence required to expand the tuple elements.
	template <class... Args1, class... Args2, size_t... I1, size_t... I2>
	pair(eastl::piecewise_construct_t,
		 eastl::tuple<Args1...> first_args,
		 eastl::tuple<Args2...> second_args,
		 eastl::index_sequence<I1...>,
		 eastl::index_sequence<I2...>)
		 : first(std::forward<Args1>(eastl::get<I1>(first_args))...)
		 , second(std::forward<Args2>(eastl::get<I2>(second_args))...) { }
*/
public:
	CONSTEXPR_SINCE_CPP_14 pair& operator=(const pair& p)
		noexcept(std::is_nothrow_copy_assignable<T1>::value&& std::is_nothrow_copy_assignable<T2>::value)
	{
		first = p.first;
		second = p.second;
		return *this;
	}

	template <typename U, typename V,
		typename = kat::enable_if_t<std::is_convertible<U, T1>::value and std::is_convertible<V, T2>::value>>
	CONSTEXPR_SINCE_CPP_14 pair& operator=(const pair<U, V>& p)
	{
		first = p.first;
		second = p.second;
		return *this;
	}

	CONSTEXPR_SINCE_CPP_14 pair& operator=(pair&& p)
		noexcept(std::is_nothrow_move_assignable<T1>::value and std::is_nothrow_move_assignable<T2>::value)
	{
		first = std::forward<T1>(p.first);
		second = std::forward<T2>(p.second);
		return *this;
	}

	template <typename U, typename V,
		typename = kat::enable_if_t<std::is_convertible<U, T1>::value and std::is_convertible<V, T2>::value>>
	CONSTEXPR_SINCE_CPP_14 pair& operator=(pair<U, V>&& p)
	{
		first = std::forward<U>(p.first);
		second = std::forward<V>(p.second);
		return *this;
	}

	CONSTEXPR_SINCE_CPP_14 void swap(pair& p)
	noexcept(kat::is_nothrow_swappable<T1>::value and kat::is_nothrow_swappable<T2>::value)
	{
		kat::swap(&first, &p.first);
		kat::swap(&second, &p.second);
	}
};


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator==(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	return ((a.first == b.first) and (a.second == b.second));
}


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator<(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	// Note that we use only operator < in this expression. Otherwise we could
	// use the simpler: return (a.m1 == b.m1) ? (a.m2 < b.m2) : (a.m1 < b.m1);
	// The user can write a specialization for this operator to get around this
	// in cases where the highest performance is required.
	return ((a.first < b.first) or (not (b.first < a.first) and (a.second < b.second)));
}


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator!=(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	return not (a == b);
}


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator>(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	return b < a;
}


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator>=(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	return not (a < b);
}


template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline bool operator<=(const pair<T1, T2>& a, const pair<T1, T2>& b)
{
	return not (b < a);
}

///////////////////////////////////////////////////////////////////////
/// make_pair
///
/// make_pair is
/// 
/// Note: You don't usually need to use make_pair in order to make a pair. 
/// The following code is equivalent, and the latter avoids one more level of inlining:
///     return make_pair(charPtr, charPtr);
///     return pair<char*, char*>(charPtr, charPtr);

/**
 * @brief Creates a std::pair object, deducing the target type from the types of arguments.
 */
template <typename T1, typename T2>
KAT_HD CONSTEXPR_SINCE_CPP_14 inline
pair<typename detail::remove_reference_wrapper<typename std::decay<T1>::type>::type,
typename detail::remove_reference_wrapper<typename std::decay<T2>::type>::type>
make_pair(T1&& a, T2&& b)
{
	typedef typename detail::remove_reference_wrapper<typename std::decay<T1>::type>::type T1Type;
	typedef typename detail::remove_reference_wrapper<typename std::decay<T2>::type>::type T2Type;

	return kat::pair<T1Type, T2Type>(std::forward<T1>(a), std::forward<T2>(b));
}

/* Enable this for tuples
template <typename T1, typename T2>
class tuple_size<pair<T1, T2>> : public integral_constant<size_t, 2>
{
};

template <typename T1, typename T2>
class tuple_size<const pair<T1, T2>> : public integral_constant<size_t, 2>
{
};

template <typename T1, typename T2>
class tuple_element<0, pair<T1, T2>>
{
public:
	using type = T1;
};

template <typename T1, typename T2>
class tuple_element<1, pair<T1, T2>>
{
public:
	using type = T1;
};

template <typename T1, typename T2>
class tuple_element<0, const pair<T1, T2>>
{
public:
	using type = const T1;
};

template <typename T1, typename T2>
class tuple_element<1, const pair<T1, T2>>
{
public:
	using type = const T2;
};

template <size_t I>
struct GetPair;

template <>
struct GetPair<0> {
	template <typename T1, typename T2>
	static KAT_HD constexpr T1& getInternal(pair<T1, T2>& p)
	{
		return p.first;
	}

	template <typename T1, typename T2>
	static KAT_HD constexpr const T1& getInternal(const pair<T1, T2>& p)
	{
		return p.first;
	}

	template <typename T1, typename T2>
	static KAT_HD constexpr T1&& getInternal(pair<T1, T2>&& p)
	{
		return std::forward<T1>(p.first);
	}
};

template <>
struct GetPair<1> {
	template <typename T1, typename T2>
	static KAT_HD constexpr T2& getInternal(pair<T1, T2>& p)
	{
		return p.second;
	}

	template <typename T1, typename T2>
	static KAT_HD constexpr const T2& getInternal(const pair<T1, T2>& p)
	{
		return p.second;
	}

	template <typename T1, typename T2>
	static KAT_HD constexpr T2&& getInternal(pair<T1, T2>&& p)
	{
		return std::forward<T2>(p.second);
	}
};

template <size_t I, typename T1, typename T2>
tuple_element_t<I, pair<T1, T2>>& get(pair<T1, T2>& p)
{
	return GetPair<I>::getInternal(p);
}

template <size_t I, typename T1, typename T2>
const tuple_element_t<I, pair<T1, T2>>& get(const pair<T1, T2>& p)
{
	return GetPair<I>::getInternal(p);
}

template <size_t I, typename T1, typename T2>
tuple_element_t<I, pair<T1, T2>>&& get(pair<T1, T2>&& p)
{
	return GetPair<I>::getInternal(std::move(p));
}
*/

}  // namespace kat

// C++17 structured bindings support

#include <tuple>
namespace std {
// NOTE(rparolin): Some platform implementations didn't check the standard specification and implemented the
// "tuple_size" and "tuple_element" primary template with as a struct.  The standard specifies they are
// implemented with the class keyword so we provide the template specializations as a class and disable the
// generated warning.
// EA_DISABLE_CLANG_WARNING(-Wmismatched-tags)

template <class... Ts>
class tuple_size<::kat::pair<Ts...>> : public ::std::integral_constant<size_t, sizeof...(Ts)>
{
};

template <size_t I, class... Ts>
class tuple_element<I, ::kat::pair<Ts...>> : public ::std::tuple_element<I, ::std::pair<Ts...>>
{
};

} // namespace std

#endif // CUDA_KAT_PAIR_HPP_
