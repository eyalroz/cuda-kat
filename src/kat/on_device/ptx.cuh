/**
 * @file on_device/ptx.cuh
 *
 * @brief an include-of-includes of all functions whose implementation uses
 * inline PTX (CUDA's inline IR, using `inline asm` in the source code).
 */

#pragma once
#ifndef CUDA_ON_DEVICE_PTX_CUH_
#define CUDA_ON_DEVICE_PTX_CUH_

#if ! __CUDA_ARCH__ >= 300
#error "This code can only target devices of compute capability 3.0 or higher."
#endif

#include "ptx/special_registers.cuh"
#include "ptx/miscellany.cuh"
#include "ptx/video_instructions.cuh"

#endif /* CUDA_ON_DEVICE_PTX_CUH_ */
