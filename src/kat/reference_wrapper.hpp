/**
 * @file kat/reference_wrapper.hpp
 *
 * @brief This file implements `kat::reference_wrapper`, an equivalent of
 * C++11's `std::reference_wrapper` which may be used both in host-side and
 * CUDA-device-side code.
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
// on 2020-03-11.


#ifndef CUDA_KAT_REFERENCE_WRAPPER_HPP_
#define CUDA_KAT_REFERENCE_WRAPPER_HPP_

#include <kat/common.hpp>
#include <kat/utility.hpp>
#include <type_traits>

namespace kat {

/// reference_wrapper
template <typename T>
class reference_wrapper
{
public:
	typedef T type;

	KAT_HD reference_wrapper(T&) noexcept;
	KAT_HD reference_wrapper(T&&) = delete;
	KAT_HD reference_wrapper(const reference_wrapper<T>& x) noexcept;

	KAT_HD reference_wrapper& operator=(const reference_wrapper<T>& x) noexcept;

	KAT_HD operator T& () const noexcept;
	KAT_HD T& get() const noexcept;

	template <typename... ArgTypes>
	KAT_HD typename std::result_of<T&(ArgTypes&&...)>::type operator() (ArgTypes&&...) const;

private:
	T* val;
};

template <typename T>
KAT_HD reference_wrapper<T>::reference_wrapper(T &v) noexcept
// Originally, EASTL has:
//
// : val(addressof(v))
//
// here. But we can't use std::addressof, since it is not accessible in device-side code;
// and we don't have the utility functions in <memory> implemented in device-and-host versions.
// So - we'll just inline an implementation of std::addressof() here instead
	: val(
		reinterpret_cast<T*>(
			&const_cast<char&>(
				reinterpret_cast<const volatile char&>(v)
			)
		)
	)
{}

template <typename T>
KAT_HD reference_wrapper<T>::reference_wrapper(const reference_wrapper<T>& other) noexcept
	: val(other.val)
{}

template <typename T>
KAT_HD reference_wrapper<T>& reference_wrapper<T>::operator=(const reference_wrapper<T>& other) noexcept
{
	val = other.val;
	return *this;
}

template <typename T>
KAT_HD reference_wrapper<T>::operator T&() const noexcept
{
	return *val;
}

template <typename T>
KAT_HD T& reference_wrapper<T>::get() const noexcept
{
	return *val;
}

template <typename T>
template <typename... ArgTypes>
KAT_HD typename std::result_of<T&(ArgTypes&&...)>::type reference_wrapper<T>::operator() (ArgTypes&&... args) const
{
	//	return std::invoke(*val, std::forward<ArgTypes>(args)...);
	return *val(std::forward<ArgTypes>(args)...);
}

// reference_wrapper-specific utilties
template <typename T>
KAT_HD reference_wrapper<T> ref(T& t) noexcept
{
	return kat::reference_wrapper<T>(t);
}

template <typename T>
KAT_HD void ref(const T&&) = delete;

template <typename T>
KAT_HD reference_wrapper<T> ref(reference_wrapper<T>t) noexcept
{
	return kat::ref(t.get());
}

template <typename T>
KAT_HD reference_wrapper<const T> cref(const T& t) noexcept
{
	return kat::reference_wrapper<const T>(t);
}

template <typename T>
KAT_HD void cref(const T&&) = delete;

template <typename T>
KAT_HD reference_wrapper<const T> cref(reference_wrapper<T> t) noexcept
{
	return kat::cref(t.get());
}


// reference_wrapper-specific type traits
template <typename T>
struct is_reference_wrapper_helper
	: public std::false_type {};

template <typename T>
struct is_reference_wrapper_helper<kat::reference_wrapper<T> >
	: public std::true_type {};

template <typename T>
struct is_reference_wrapper
	: public kat::is_reference_wrapper_helper<typename std::remove_cv<T>::type> {};


// Helper which adds a reference to a type when given a reference_wrapper of that type.
template <typename T>
struct remove_reference_wrapper
	{ typedef T type; };

template <typename T>
struct remove_reference_wrapper< kat::reference_wrapper<T> >
	{ typedef T& type; };

template <typename T>
struct remove_reference_wrapper< const kat::reference_wrapper<T> >
	{ typedef T& type; };

/*
// reference_wrapper specializations of invoke
// These have to come after reference_wrapper is defined, but reference_wrapper needs to have a
// definition of invoke, so these specializations need to come after everything else has been defined.
template <typename R, typename C, typename T, typename... Args>
auto invoke_impl(R (C::*func)(Args...), T&& obj, Args&&... args) ->
	typename std::enable_if<is_reference_wrapper<typename std::remove_reference<T>::type>::value,
					   decltype((obj.get().*func)(std::forward<Args>(args)...))>::type
{
	return (obj.get().*func)(std::forward<Args>(args)...);
}

template <typename M, typename C, typename T>
auto invoke_impl(M(C::*member), T&& obj) ->
	typename enable_if<is_reference_wrapper<typename remove_reference<T>::type>::value,
					   decltype(obj.get().*member)>::type
{
	return obj.get().*member;
}
*/

} // namespace kat

#endif // CUDA_KAT_REFERENCE_WRAPPER_HPP_
