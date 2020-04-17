/**
 * @file on_device/ptx.cuh
 *
 * @brief Wrapper functions for single PTX instructions --- using inline PTX
 * assembly --- which are not already available in the official CUDA includes
 *
 * CUDA provides many "intrinsics" functions, which wrap single PTX instructions,
 * e.g. `__ldg` or `__funnelshift_l` from `sm_32_intrinsics.h`. But - CUDA
 * doesn't provide such functions for all of the PTX instruction set. The
 * files included from this master-include contain such single-line assembly
 * wrapper functions for different categories of missing PTX instructions.
 *
 * @note Unlike @ref `on_device/builtins.cuh`, functions here are not
 * templated, and do not necessarily have the same name for different
 * parameter types. `on_device/builtins.cuh` functions do _use_ PTX wrapper
 * functions as their implementation.
 */

#pragma once
#ifndef CUDA_KAT_ON_DEVICE_PTX_CUH_
#define CUDA_KAT_ON_DEVICE_PTX_CUH_

#if ! __CUDA_ARCH__ >= 300
#error "This code can only target devices of compute capability 3.0 or higher."
#endif

namespace kat {

/**
 * @brief Code exposing CUDA's PTX intermediate representation instructions
 * to C++ code.
 *
 * With CUDA, device-side code is compiled from a C++-like language to an
 * intermediate representation (IR), which is not supported directly by any
 * GPU, but from which it is easy to compile.
 *
 * Occasionally, a developer wants to use a specific PTX instruction - e.g.
 * to optimize some code. CUDA's headers expose some of the opcodes for these
 * instructions - but not all of them. Also, the exposed instructions are
 * not templated on the arguments - while PTX instructions _are_ thus
 * templated. These two gaps are filled by this library.
 */
namespace ptx { }

} // namespace kat

#include "ptx/special_registers.cuh"
#include "ptx/miscellany.cuh"
#include "ptx/video_instructions.cuh"

#endif // CUDA_KAT_ON_DEVICE_PTX_CUH_
