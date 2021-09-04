/**
 * @file <type_traits> constructs usable in C++11, which only
 * became available in C++14.
 */
#ifndef CUDA_KAT_CPP14_TYPE_TRAITS_HPP_
#define CUDA_KAT_CPP14_TYPE_TRAITS_HPP_

///@cond

#if __cplusplus < 201103L
#error "C++11 or newer is required to use this header"
#endif

#include <type_traits>

namespace kat {

#if  __cplusplus < 201402L
template <typename T> using add_const_t = typename add_const<T>::type;
template <typename T> using add_cv_t = typename add_cv<T>::type;
template <typename T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;
template <typename T> using add_pointer_t = typename add_pointer<T>::type;
template <typename T> using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;
template <typename T> using add_volatile_t = typename add_volatile<T>::type;
template <typename Length, typename Alignment> using aligned_storage_t = typename aligned_storage<Length, Alignment>::type;
template <typename Length, typename... ConstituentTypes> using aligned_union_t = typename aligned_union<Length, ConstituentTypes...>::type;
template <typename... Ts> using common_type_t = typename common_type<Ts...>::type;
template <typename Condition, typename OnTrue, typename OnFalse> using conditional_t = typename conditional<Condition, OnTrue, OnFalse>::type;
template <typename T> using decay_t = typename decay<T>::type;
template <typename Condition, typename T> using enable_if_t = typename enable_if<Condition, T>::type;
template <typename T> using make_signed_t = typename make_signed<T>::type;
template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;
template <typename T> using remove_all_extents_t = typename remove_all_extents<T>::type;
template <typename T> using remove_const_t = typename remove_const<T>::type;
template <typename T> using remove_cv_t = typename remove_cv<T>::type;
template <typename T> using remove_extent_t = typename remove_extent<T>::type;
template <typename T> using remove_pointer_t = typename remove_pointer<T>::type;
template <typename T> using remove_reference_t = typename remove_reference<T>::type;
template <typename T> using remove_volatile_t = typename remove_volatile<T>::type;
template <typename T> using result_of_t = typename result_of<T>::type;
template <typename T> using underlying_type_t = typename underlying_type<T>::type;
#else
using std::add_const_t;
using std::add_cv_t;
using std::add_lvalue_reference_t;
using std::add_pointer_t;
using std::add_rvalue_reference_t;
using std::add_volatile_t;
using std::aligned_storage_t;
using std::aligned_union_t;
using std::common_type_t;
using std::conditional_t;
using std::decay_t;
using std::enable_if_t;
using std::make_signed_t;
using std::make_unsigned_t;
using std::remove_all_extents_t;
using std::remove_const_t;
using std::remove_cv_t;
using std::remove_extent_t;
using std::remove_pointer_t;
using std::remove_reference_t;
using std::remove_volatile_t;
using std::result_of_t;
using std::underlying_type_t;
#endif

} // namespace kat

///@endcond

#endif // CUDA_KAT_CPP14_TYPE_TRAITS_HPP_
