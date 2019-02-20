/**
 * @file on_device/shared_memory.cuh
 *
 * @brief Utility code for working with dynamic shared memory in CUDA device-side
 * code.
 */

/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2019, Eyal Rozenberg
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef CUDA_ON_DEVICE_SHARED_MEMORY_CUH_
#define CUDA_ON_DEVICE_SHARED_MEMORY_CUH_

#include <kat/on_device/grid_info.cuh>
#include <kat/on_device/ptx.cuh>

#include <kat/define_specifiers.hpp>

/**
 * @note Only regards dynamic shared memory.
 */
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
__fd__ size_t size() {
	return ptx::special_registers::total_smem_size();
}

namespace static_ {

/**
 * @brief Obtain the size in bytes of the (per-block) static shared memory
 * for the running kernel.
 *
 * @note requires special register access which is not so cheap.
 */
__fd__ size_t size() {
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
__fd__ size_t size() {
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
__device__ T* proxy()
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
__fd__ T* contiguous(unsigned num_elements_per_warp, offset_t base_offset = 0)
{
	return proxy<T>() + base_offset +
		num_elements_per_warp * grid_info::linear::warp::index_in_block();
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
__fd__ T* strided(offset_t base_offset = 0)
{
	return proxy<T>() + base_offset + grid_info::linear::warp::index_in_block();
}

} // namespace dynamic


} // namespace linear_grid

} // namespace shared_memory

#include <kat/undefine_specifiers.hpp>

#endif // CUDA_ON_DEVICE_SHARED_MEMORY_CUH_
