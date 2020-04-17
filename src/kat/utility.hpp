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

#ifdef KAT_DEFINE_MOVE_AND_FORWARD
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
#endif

#if __cplusplus >= 201401L
template <typename T, typename U = T>
constexpr KAT_FHD auto exchange (T& x, U&& new_value) // TODO: A noexcept clause?
{
#ifndef KAT_DEFINE_MOVE_AND_FORWARD
	using std::move;
	using std::forward;
#endif
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
 * @note Some kat types overload this default implementation.
 *
 */
template <typename T>
KAT_FHD CONSTEXPR_SINCE_CPP_14 void swap( T& a, T& b )
	noexcept(
	    std::is_nothrow_move_constructible<T>::value &&
	    std::is_nothrow_move_assignable<T>::value
	)
{
#ifndef KAT_DEFINE_MOVE_AND_FORWARD
	using std::move;
#endif
	T tmp ( move(a) );
	a = move(b);
	b = move(tmp);
}

namespace detail {

template<class T>
struct addr_impl_ref
{
	T& v_;

	KAT_FHD addr_impl_ref( T& v ): v_( v ) {}
	KAT_FHD operator T& () const { return v_; }

private:
	KAT_FHD addr_impl_ref & operator=(const addr_impl_ref &);
};

template<class T>
struct addressof_impl
{
	static KAT_FHD  T* f( T& v, long ) {
		return reinterpret_cast<T*>(
			&const_cast<char&>(reinterpret_cast<const volatile char &>(v)));
	}

	static KAT_FHD T* f( T* v, int ) { return v; }
};

} // namespace detail

template<class T>
KAT_FHD T* addressof( T& v ) {
	// Note the complex implementation details are due to some objects
	// overloading their & operator
	return detail::addressof_impl<T>::f( detail::addr_impl_ref<T>( v ), 0 );
}

template <class T>
const KAT_FHD T* addressof(const T&&) = delete;

} // namespace kat

#endif // CUDA_KAT_UTILITY_HPP_
