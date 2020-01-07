#pragma once
#ifndef UTIL_TYPE_NAME_HPP_
#define UTIL_TYPE_NAME_HPP_

#include "poor_mans_constexpr_string.hpp"
#include <type_traits>
#include <typeinfo>
#include <iostream>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <string>
#include <sstream>
#include <memory>


///@cond
#include <kat/define_specifiers.hpp>
///@endcond

namespace util {

template <class T>
CONSTEXPR14_TN __hd__ constexpr_string type_name()
{
#ifdef __clang__
    constexpr_string p = __PRETTY_FUNCTION__;
    return constexpr_string(p.data() + 31, p.size() - 31 - 1);
//#elif defined(__CUDA_ARCH__)
//    constexpr_string p = __PRETTY_FUNCTION__;
//    return constexpr_string(p.data(), p.size());
#elif defined(__CUDACC__)
    constexpr_string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402 || defined(__CUDA_ARCH__)
    return constexpr_string(p.data() + 51, p.size() - 51 - 1); // 50 is the length of util::constexpr_string util::type_name() [with T =
#  else
    return constexpr_string(p.data() + 61, p.size() - 61 - 1); // 50 is the length of constexpr util::constexpr_string util::type_name() [with T =
#  endif
#elif defined(__GNUC__)
    constexpr_string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return constexpr_string(p.data() + 36, p.size() - 36 - 1);
#  else
    return constexpr_string(p.data() + 46, p.size() - 46 - 1);
#  endif
#elif defined(_MSC_VER)
    constexpr_string p = __FUNCSIG__;
    return constexpr_string(p.data() + 38, p.size() - 38 - 7);
#endif
}

/*template <class T>
CONSTEXPR14_TN __fhd__ constexpr_string type_name(T&&)
{
	return type_name<T>();
}

template <class T>
CONSTEXPR14_TN __fhd__ constexpr_string type_name(const T&)
{
	return type_name<T>();
}*/


/**
 * A function for obtaining the string name
 * of a type, using that actual type at compile-time.
 * (The function might have been constexpr, but I doubt
 * so much is acceptable at compile time.) This is an
 * alternative to using type_info<T>.name() which also
 * preserves CV qualifiers (const, volatile, reference,
 *  rvalue-reference)
 *
 * The code was copied from this StackOverflow answer:
 *  http://stackoverflow.com/a/20170989/1593077
 * due to Howard Hinnant
 * ... with some slight modifications by Eyal Rozenberg
 */


template <typename T, bool WithCVCorrections = false>
std::string type_name_()
{
	typedef typename std::remove_reference<T>::type TR;

	std::unique_ptr<char, void(*)(void*)> own(
#ifndef _MSC_VER
		abi::__cxa_demangle(typeid(TR).name(), nullptr,	nullptr, nullptr),
#else
		nullptr,
#endif
		std::free
	);
	std::string r = (own != nullptr) ? own.get() : typeid(TR).name();
	if (WithCVCorrections) {
		if (std::is_const<TR>::value)
			r += " const";
		if (std::is_volatile<TR>::value)
			r += " volatile";
		if (std::is_lvalue_reference<T>::value)
			r += "&";
		else if (std::is_rvalue_reference<T>::value)
			r += "&&";
	}
	return r;
}

/**
 * This is a convenience function, so that instead of
 *
 *   util::type_name<decltype(my_value)>()
 *
 * you could use:
 *
 *   util::type_name_of(my_value)
 *
 * @param v a value which is only passed to indicate a type
 * @return the string type name of typeof(v)
 */
template <typename T, bool WithCVCorrections = false>
std::string type_name_of(const T& v) { return util::type_name<T, WithCVCorrections>(); }


template <typename... Ts>
auto type_names_() -> decltype(std::make_tuple(type_name_<Ts>()...))
{ return std::make_tuple(type_name_<Ts>()...); }


/**
 * Removed the trailing template parameter listing from a type name, e.g.
 *
 *   foo<int> bar<plus<int>>
 *
 * becomes
 *
 *   foo<int> bar<plus<int>>
 *
 * This is not such useful function, as int bar<int>(double x) will
 * become int bar. So - fix it.
 *
 * @param type_name the name of a type, preferably obtained with
 * util::type_info
 * @return the template-less type name, or the original type name if
 * we could not find anything to remove (doesn't throw)
 */
inline std::string discard_template_parameters(const std::string& type_name)
{
	auto template_rbracket_pos = type_name.rfind('>');
	if (template_rbracket_pos == std::string::npos) {
		return type_name;
	}
	unsigned bracket_depth = 1;
	for (unsigned pos = template_rbracket_pos; pos > 0; pos++) {
		switch(type_name[pos]) {
		case '>': bracket_depth++; break;
		case '<': bracket_depth--; break;
		}
		if (bracket_depth == 0) return type_name.substr(0,pos);
	}
	return type_name;
}

} /* namespace util */


///@cond
#include <kat/undefine_specifiers.hpp>
///@endcond

#endif /* UTIL_TYPE_NAME_HPP_ */
