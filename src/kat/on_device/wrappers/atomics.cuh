/**
 * @file wrappers/atomics.cuh Type-generic wrappers for CUDA atomic operations
 */

#ifndef CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
#define CUDA_KAT_ON_DEVICE_ATOMICS_CUH_

#include <kat/define_specifiers.hpp>
#include <limits>

namespace kat {
namespace atomic {

template <typename T>  __fd__ T add        (T* address, T val);
template <typename T>  __fd__ T subtract   (T* address, T val);
template <typename T>  __fd__ T exchange   (T* address, const T& val);
template <typename T>  __fd__ T min        (T* address, T val);
template <typename T>  __fd__ T max        (T* address, T val);
template <typename T>  __fd__ T logical_and(T* address, T val);
template <typename T>  __fd__ T logical_or (T* address, T val);
template <typename T>  __fd__ T logical_xor(T* address, T val);
template <typename T>  __fd__ T bitwise_or (T* address, T val);
template <typename T>  __fd__ T bitwise_and(T* address, T val);
template <typename T>  __fd__ T bitwise_xor(T* address, T val);
template <typename T>  __fd__ T bitwise_not(T* address);
template <typename T>  __fd__ T set_bit    (T* address, native_word_t bit_index);
template <typename T>  __fd__ T unset_bit  (T* address, native_word_t bit_index);
/**
 * @brief Increment the value at @p address by 1 - but if it reaches or surpasses @p wraparound_value, set it to 0.
 *
 * @note repeated invocations of this function will cycle through the range of values 0... @p wraparound_values - 1; thus
 * as long as the existing value is within that range, this is a simple incrementation modulo @p wraparound_value.
 */
template <typename T>  __fd__ T increment  (T* address, T modulus = std::numeric_limits<T>::max());
/**
 * @brief Decrement the value at @p address by 1 - but if it reaches 0, or surpasses @p wraparound_value, it is set
 * to @p wrarparound_value - 1.
 *
 * @note repeated invocations of this function will cycle backwards through the range of values 0...
 * @p wraparound_values - 1; thus as long as the existing value is within that range, this is a simple decrementation
 * modulo @p wraparound_value.
 */
template <typename T>  __fd__ T decrement  (T* address, T modulus = std::numeric_limits<T>::max());


// Note: We let this one take a const reference
template <typename T>  __fd__ T compare_and_swap(
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
__fd__ T apply_atomically(UnaryFunction f, T* address);

/**
 * Use atomic compare-and-swap to apply a binary function to two values,
 * replacing the first at its memory location with the result before anything
 * else changes it.
 *
 * @todo Take a template parameter pack and multiple arguments to f
 *
 * @note using `__restrict__` here just to be on the safe side, in case
 * UnaryFunction ends up being some sort of pointer
 *
 * @return The new value which was stored in memory
 */
template <typename BinaryFunction, typename T>
__fd__ T apply_atomically(
	BinaryFunction         f,
	T*       __restrict__  address,
	const T  __restrict__  rhs);


} // namespace atomic
} // namespace kat

#include <kat/undefine_specifiers.hpp>
#include "detail/atomics.cuh"

#endif // CUDA_KAT_ON_DEVICE_ATOMICS_CUH_
