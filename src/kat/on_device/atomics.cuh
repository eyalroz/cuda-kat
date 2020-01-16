/**
 * @file atomics.cuh Type-generic wrappers for CUDA atomic operations
 *
 * @note nVIDIA makes a rather unfortunate and non-intuitive choice of parameter
 * names for its atomic functions, which - at least for now, and for the sake of
 * consistency - I adopt: they call a pointer an "address", and they call the
 * new value "val" even if there is another value to consider (e.g. atomicCAS).
 * Also, what's with the shorthand? Did you run out of disk space? :-(
 *
 * @note If you use an atomic function implemented here for a type size for
 * which CUDA doesn't support atomic primitives, you must have read and write
 * to the memory before it, so that a compare-and-swap on that memory and the
 * actual value of interest would be possible. This is a subtle point, since the
 * CUDA primitives themselves don't have any such requirements, nor do their
 * wrappers here; and - we can't afford to check (nor is it possible to check).
 */

#ifndef CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
#define CUDA_KAT_ON_DEVICE_ATOMICS_CUH_

#include <limits>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace atomic {

template <typename T>  KAT_FD T add        (T* address, T val);
template <typename T>  KAT_FD T subtract   (T* address, T val);
template <typename T>  KAT_FD T exchange   (T* address, T val);
template <typename T>  KAT_FD T min        (T* address, T val);
template <typename T>  KAT_FD T max        (T* address, T val);
template <typename T>  KAT_FD T logical_and(T* address, T val);
template <typename T>  KAT_FD T logical_or (T* address, T val);
template <typename T>  KAT_FD T logical_not(T* address);
template <typename T>  KAT_FD T logical_xor(T* address, T val);
template <typename T>  KAT_FD T bitwise_or (T* address, T val);
template <typename T>  KAT_FD T bitwise_and(T* address, T val);
template <typename T>  KAT_FD T bitwise_xor(T* address, T val);
template <typename T>  KAT_FD T bitwise_not(T* address);
template <typename T>  KAT_FD T set_bit    (T* address, native_word_t bit_index);
template <typename T>  KAT_FD T unset_bit  (T* address, native_word_t bit_index);
/**
 * @brief Increment the value at @p address by 1 - but if it reaches or surpasses @p wraparound_value, set it to 0.
 *
 * @note repeated invocations of this function will cycle through the range of values 0... @p wraparound_values - 1; thus
 * as long as the existing value is within that range, this is a simple incrementation modulo @p wraparound_value.
 */
template <typename T>  KAT_FD T increment  (T* address, T modulus = std::numeric_limits<T>::max());
/**
 * @brief Decrement the value at @p address by 1 - but if it reaches 0, or surpasses @p wraparound_value, it is set
 * to @p wrarparound_value - 1.
 *
 * @note repeated invocations of this function will cycle backwards through the range of values 0...
 * @p wraparound_values - 1; thus as long as the existing value is within that range, this is a simple decrementation
 * modulo @p wraparound_value.
 */
template <typename T>  KAT_FD T decrement  (T* address, T modulus = std::numeric_limits<T>::max());


// Note: We let this one take a const reference
template <typename T>  KAT_FD T compare_and_swap(
    T*       address,
    const T  compare,
    const T  val);


/**
 * Use atomic compare-and-swap to apply a unary function to some value,
 * replacing it at its memory location with the result before anything
 * else changes it.
 *
 * @return The new value which was stored in memory
 */
template <typename UnaryFunction, typename T>
KAT_FD T apply_atomically(UnaryFunction f, T* address);

/**
 * Use atomic compare-and-swap to apply a binary function to two values,
 * replacing the first at its memory location with the result before anything
 * else changes it.
 *
 * @return The new value which was stored in memory
 */
template <typename Function, typename T, typename... Ts>
KAT_FD T apply_atomically(
	Function                f,
	T*       __restrict__   address,
	const Ts...             xs);


} // namespace atomic
} // namespace kat

#include "detail/atomics.cuh"

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
