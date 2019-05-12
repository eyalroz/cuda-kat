/**
 * @file on_device/shared_memory/operations.cuh
 *
 * @brief Some basic operations on shared memory (using the library's general
 * computational primitives)
 *
 */

#pragma once
#ifndef CUDA_KAT_SHARED_MEMORY_OPS_CUH
#define CUDA_KAT_SHARED_MEMORY_OPS_CUH

#include <kat/on_device/shared_memory/basic.cuh>
#include <kat/on_device/sequence_ops/block.cuh>

#include <kat/define_specifiers.hpp>

namespace kat {
namespace shared_memory {
namespace dynamic {
namespace linear_grid {

/**
 * @brief Collaboratively fill the block's dynamic shared memory with a fixed
 * value, up to a certain point
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param value each element of the block's dynamic shared memory will be
 * set to this value
 * @param length the number of T elements to set to @p value
 */
template <typename T>
__fd__ void fill(
	const T&               value,
	shared_memory::size_t  length)
{
	T tmp = value;
	primitives::block::fill_n(shared_memory::dynamic::proxy<T>(), value, length);
}

/**
 * @brief Collaboratively fill the block's dynamic shared memory with a fixed value.
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param value each element of the block's dynamic shared memory will be
 * set to this value
 *
 * @note This variant of @ref fill() pays a small "penality" for determining
 * the size of the shared memory by itself, since it must access a
 * typically-unused special register for this purpose. If you can, prefer
 * passing a length yourself.
 */
template <typename T>
__fd__ void fill(const T& value)
{
	auto length = shared_memory::dynamic::size<T>();
	return fill(value, length);
}

/**
 * @brief Collaboratively zero-out the block's dynamic shared memory , up to a
 * certain point
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param length the number of T elements to set to zero
 */
template <typename T>
__fd__ void zero(kat::shared_memory::size_t length)
{
	return fill(T{0}, length);
}

/**
 * @brief Collaboratively zero-out the block's dynamic shared memory
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param length the number of T elements to set to zero
 */
template <typename T>
__fd__ void zero()
{
	auto length = shared_memory::dynamic::size<T>();
	return zero(length);
}


/**
 * Sets the (beginning of the dynamic) shared memory of the block
 * to a copy of some area of device memory.
 *
 * @param[in]  source Data in global memory (_not_ anywhere
 * else in shared memory! That breaks the {@code __restrict__}
 * restriction) which we wish to have in shared memory
 * @param[in] length length of the area to copy; must be
 * no larger than the available length (in T's) of shared
 * memory
 * @return the beginning of the block's shared memory -
 * which now contains a copy of the data at @p source.
 *
 * @note length is not checked to be valid - it is up to
 * the caller to refrain from trying to copy too much
 * into the shared memory; use
 */
template <typename T>
__fd__ T* __restrict__ set_to_copy_of(const T*  source, shared_memory::size_t length)
{
	T* __restrict__ data_in_shared_mem = shared_memory::dynamic::proxy<T>();
	primitives::block::copy(data_in_shared_mem, source, length);
	return data_in_shared_mem;
}

} // namespace linear_grid

/**
 * @brief Collaboratively fill the block's dynamic shared memory with a fixed
 * value, up to a certain point
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param value each element of the block's dynamic shared memory will be
 * set to this value
 * @param length the number of T elements to set to @p value
 *
 * @note Not implemented yet - need non-linear-grid variants of some of the block primtives.
 */
template <typename T>
__fd__ void fill(
	const T&               value,
	shared_memory::size_t  length);

/**
 * @brief Collaboratively fill the block's dynamic shared memory with a fixed value.
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param value each element of the block's dynamic shared memory will be
 * set to this value
 *
 * @note This variant of @ref fill() pays a small "penality" for determining
 * the size of the shared memory by itself, since it must access a
 * typically-unused special register for this purpose. If you can, prefer
 * passing a length yourself.
 */
template <typename T>
__fd__ void fill(const T& value)
{
	auto length = shared_memory::dynamic::size<T>();
	return fill(value, length);
}

/**
 * @brief Collaboratively zero-out the block's dynamic shared memory , up to a
 * certain point
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param length the number of T elements to set to zero
 */
template <typename T>
__fd__ void zero(kat::shared_memory::size_t length)
{
	return fill(T{0}, length);
}

/**
 * @brief Collaboratively zero-out the block's dynamic shared memory
 *
 * @tparam the element type which the block's shared memory is presumed to have
 * @param length the number of T elements to set to zero
 */
template <typename T>
__fd__ void zero()
{
	auto length = shared_memory::dynamic::size<T>();
	return zero(length);
}


/**
 * Sets the (beginning of the dynamic) shared memory of the block
 * to a copy of some area of device memory.
 *
 * @param[in]  source Data in global memory (_not_ anywhere
 * else in shared memory! That breaks the {@code __restrict__}
 * restriction) which we wish to have in shared memory
 * @param[in] length length of the area to copy; must be
 * no larger than the available length (in T's) of shared
 * memory
 * @return the beginning of the block's shared memory -
 * which now contains a copy of the data at @p source.
 *
 * @note length is not checked to be valid - it is up to
 * the caller to refrain from trying to copy too much
 * into the shared memory.
 *
 * @note Not implemented yet - need non-linear-grid variants of
 * some of the block primitives.
 */
template <typename T>
__fd__ T* __restrict__ set_to_copy_of(const T*  source, shared_memory::size_t length);


} // namespace dynamic
} // namespace shared_memory
} // namespace kat



#include <kat/undefine_specifiers.hpp>

#endif // CUDA_KAT_SHARED_MEMORY_OPS_CUH
