/**
 * @file wrappers/atomics.cuh Type-generic wrappers for CUDA atomic operations
 */

#ifndef CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
#define CUDA_KAT_ON_DEVICE_ATOMICS_CUH_

#include <kat/define_specifiers.hpp>

namespace kat {
namespace atomic {

/*
 * TODO:
 * - Consider using non-const references instead of pointers (but make sure we get the same PTX)
 * - Consider creating an atomic<T> wrapper class which has these as operators.
 */
template <typename T>  __fd__ T add        (T* __restrict__ address, T val);
template <typename T>  __fd__ T subtract   (T* __restrict__ address, T val);
template <typename T>  __fd__ T increment  (T* __restrict__ address, T wraparound_value = T{1});
template <typename T>  __fd__ T decrement  (T* __restrict__ address, T wraparound_value = T{1});
template <typename T>  __fd__ T exchange   (T* __restrict__ address, T val);
template <typename T>  __fd__ T min        (T* __restrict__ address, T val);
template <typename T>  __fd__ T max        (T* __restrict__ address, T val);
template <typename T>  __fd__ T logical_and(T* __restrict__ address, T val);
template <typename T>  __fd__ T logical_or (T* __restrict__ address, T val);
template <typename T>  __fd__ T logical_xor(T* __restrict__ address, T val);
template <typename T>  __fd__ T bitwise_or (T* __restrict__ address, T val);
template <typename T>  __fd__ T bitwise_and(T* __restrict__ address, T val);
template <typename T>  __fd__ T bitwise_xor(T* __restrict__ address, T val);
template <typename T>  __fd__ T bitwise_not(T* __restrict__ address);
template <typename T>  __fd__ T set_bit    (T* __restrict__ address, const unsigned bit_index);
template <typename T>  __fd__ T unset_bit  (T* __restrict__ address, const unsigned bit_index);

// Note: We let this one take a const reference
template <typename T>  __fd__ T compare_and_swap(
    T* __restrict__  address,
    const T&         compare,
    const T&         val);


/**
 * Use atomic compare-and-swap to apply a unary function to some value,
 * replacing it at its memory location with the result before anything
 * else changes it.
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename UnaryFunction>
__fd__ T apply_atomically(UnaryFunction f, T* __restrict__ address);

/**
 * Use atomic compare-and-swap to apply a binary function to two values,
 * replacing the first at its memory location with the result before anything
 * else changes it.
 *
 * @todo Take a template parameter pack and multiple arguments to f
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename BinaryFunction>
__fd__ T apply_atomically(
	BinaryFunction   f,
	T* __restrict__  address,
	const T&         rhs);


} // namespace atomic
} // namespace kat

#include <kat/undefine_specifiers.hpp>
#include "detail/atomics.cuh"

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
