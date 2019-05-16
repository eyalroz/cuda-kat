/**
 * @file on_device/non-builtins.cuh
 *
 * @brief Namespace with uniform-naming scheme, templated-when-relevant,
 * wrappers of what could be (or should be) single PTX instruction - but
 * aren't.
 */
#ifndef CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_
#define CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_

#include <kat/on_device/wrappers/builtins.cuh>

#include <kat/define_specifiers.hpp>

namespace kat {
namespace non_builtins {

/**
 * @brief Determine the 1-based index of the first non-zero bit in the argument.
 *
 * @param x the value to be considered as a container of bits
 * @return If @p x is 0, returns 0; otherwise, returns the 1-based index of the
 * first non-zero bit in @p x
 */
template <typename T> __fd__ int find_first_set(T x);
template <> __fd__ int find_first_set< int                >(int x)                { return __ffs(x);   }
template <> __fd__ int find_first_set< unsigned int       >(unsigned int x)       { return __ffs(x);   }
template <> __fd__ int find_first_set< long long          >(long long x)          { return __ffsll(x); }
template <> __fd__ int find_first_set< unsigned long long >(unsigned long long x) { return __ffsll(x); }

/**
 * @brief counts the number initial zeros when considering the binary representation
 * of a number from least to most significant digit
 * @param x the number whose representation is to be counted
 * @return the number of initial zero bits before the first 1; if x is 0, -1 is returned
 */
template <typename T> __fd__ int count_trailing_zeros(T x) { return find_first_set(x) - 1; }


} // namespace non_builtins
} // namespace kat

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_ON_DEVICE_NON_BUILTINS_CUH_
