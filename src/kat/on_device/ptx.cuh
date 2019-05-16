/**
 * @file on_device/ptx.cuh
 *
 * @brief an include-of-includes of all functions whose implementation uses
 * inline PTX (CUDA's inline IR, using `inline asm` in the source code).
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
