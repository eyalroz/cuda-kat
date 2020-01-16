/**
 * @file on_device/shared_memory/basic.cuh
 *
 * @brief Simpler / more basic utility code for working with shared memory,
 * not involving any actual computation.
 *
 */

#ifndef CUDA_KAT_ON_DEVICE_SHARED_MEMORY_BASIC_CUH_
#define CUDA_KAT_ON_DEVICE_SHARED_MEMORY_BASIC_CUH_

#include <kat/on_device/grid_info.cuh>
#include <kat/on_device/ptx.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace shared_memory {

using offset_t = int; // Perhaps make it an int32_t ?
using size_t = unsigned; // Should we make it signed like ssize_t ?

/**
 * @brief Obtain the total size in bytes of the (per-block) shared memory
 * for the running kernel - static + dynamic
 *
 * @note requires special register access which is not so cheap.
 *
 */
KAT_FD size_t size() {
	return ptx::special_registers::total_smem_size();
}

namespace static_ {

/**
 * @brief Obtain the size in bytes of the (per-block) static shared memory
 * for the running kernel.
 *
 * @note requires special register access which is not so cheap.
 */
KAT_FD size_t size() {
	return
		ptx::special_registers::total_smem_size() -
		ptx::special_registers::dynamic_smem_size();
}

} // namespace static_

namespace dynamic {

/**
 * @brief Obtain the size of the (per-block) dynamic shared_memory for
 * the running kernel
 *
 * @note without a template parameter, returns the size in bytes
 * @note requires special register access which is not so cheap.
 */
template <typename T = unsigned char>
KAT_FD size_t size() {
	return ptx::special_registers::dynamic_smem_size() / sizeof(T);
}

/**
 * This gadget is necessary for using dynamically-sized shared memory in
 * templated kernels (i.e. shared memory whose size is set by the launch
 * parameters rather than being fixed at compile time). Use of such
 * memory  requires a `__shared__ extern` unspecified-size array variable;
 * however, the way nvcc works, you cannot declare two such variables of
 * different types in your program - even if they're in different scopes.
 * That means we either need to have a different variable name for each
 * type (which would lead us into preprocessor macro hell), or - just
 * use the same type, and reintrepret according to the type we want...
 * which is what this gadget does.
 *
 * @note all threads would get the same address when calling this function,
 * so you would need to add different offsets for different threads if
 * you want a warp-specific or thread-specific pointer.
 *
 * @note see also https://stackoverflow.com/questions/27570552/
 */
template <typename T>
KAT_DEV T* proxy()
{
	// TODO: Do we need this alignment? Probably not
	extern __shared__ __align__(1024) unsigned char memory[];
	return reinterpret_cast<T*>(memory);
}

// TODO: It would be nice to get the shared memory as a span; but we
// don't currently have a span in this repository; and both std::span
// and GSL/span do not support CUDA.

/**
 * @note This namespace's contents is only relevant for linear grids
 */
namespace warp_specific {

/**
 * @brief Accesses the calling thread's warp-specific dynamic shared memory -
 * assuming the warps voluntarily divvy up the shared memory beyond some
 * point amongst themselves, using striding.
 *
 * The partitioning pattern is for each warp to get elements at a fixed
 * stride rather than a contiguous set of elements; this pattern ensures
 * that different warps are never in a bank conflict when accessing their
 * "private" shared memory - provided the number of warps divides 32, or is a
 * multiple of 32. The downside of this pattern is that different lanes accessing
 * different elements in a warp's shared memory will likely be in bank conflict
 * (and certainly be in conflict if there are 32 warps).
 *
 * @tparam T the element type assumed for all shared memory (or at least for
 * alignment and for the warp-specific shared memory)
 * @param base_offset How far into the block's overall shared memory to
 * start partitioning the memory into warp-specific sequences
 * @param num_elements_per_warp Size in elements of the area agreed to
 * be specific to each warp
 * @return Address of the first warp-specific element in shared memory
 */
template <typename T>
KAT_FD T* contiguous(unsigned num_elements_per_warp, offset_t base_offset = 0)
{
	return proxy<T>() + base_offset +
		num_elements_per_warp * linear_grid::grid_info::warp::index_in_block();
}

/**
 * @brief Accesses the calling thread's warp-specific dynamic shared memory -
 * assuming the warps voluntarily divvy up the shared memory beyond some
 * point amongst themselves into contiguous areas.
 *
 * The partitioning pattern is for each warp to get a contiguous sequence
 * of elements in memory.
 *
 * @tparam T the element type assumed for all shared memory (or at least for
 * alignment and for the warp-specific shared memory)
 * @param base_offset How far into the block's overall shared memory to
 * start partitioning the memory into warp-specific sequences
 * @return Address of the first warp-specific element in shared memory
 */
template <typename T>
KAT_FD T* strided(offset_t base_offset = 0)
{
	return proxy<T>() + base_offset + linear_grid::grid_info::warp::index_in_block();
}

} // namespace warp_specific

} // namespace dynamic
} // namespace shared_memory
} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_SHARED_MEMORY_BASIC_CUH_
