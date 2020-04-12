/**
 * @file ptx/video_instructions.cuh Non-templated wrappers for PTX "video"
 * instructions, which nVIDIA does not provide wrappers for through the CUDA
 * `<device_functions.h>` header
 *
 * "Video" instructions are not really about video (although they're probably used
 * for video somehow). Essentially they're instructions which combine another
 * operation, and another operand, after the main one; additionally, they offer
 * variants with all sorts of saturation, wraparound, sign-extension and similar
 * bells and whistles.
 *
 * These instructions (at least, the "scalar" ones) are:
 *
 *
 *  vadd       - addition
 *  vsub       - subtraction
 *  vabsdiff   - absolute difference
 *  vmin       - minimum
 *  vmax       - maximum
 *  vshl       - shift left
 *  vshr       - shift right
 *  vmad       - multiply-and-add
 *  vset       - equality check
 *
 * For now, we won't implement most of these instructions, and even for the ones
 * we do implement - we'll only choose some of the variants.
 */
#pragma once
#ifndef CUDA_KAT_PTX_VIDEO_INSTRUCTIONS_CUH_
#define CUDA_KAT_PTX_VIDEO_INSTRUCTIONS_CUH_

#include "detail/define_macros.cuh"
#include <kat/on_device/common.cuh>
#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace ptx {

/**
 * @brief bit shift, then apply a binary operator.
 *
 */
#define DEFINE_SHIFT_AND_OP(direction, second_op) \
KAT_FD uint32_t \
vsh##direction##_##second_op ( \
	uint32_t x, \
	uint32_t shift_amount, \
	uint32_t extra_operand) \
{ \
	uint32_t ret; \
	asm ("vsh" PTX_STRINGIFY(direction) ".u32.u32.u32.clamp." PTX_STRINGIFY(second_op) " %0, %1, %2, %3;" \
		: "=r"(ret)  \
		: "r"(x) \
		, "r"(shift_amount) \
		, "r"(extra_operand) \
	); \
	return ret; \
}

DEFINE_SHIFT_AND_OP(l,add) // vshl_add
DEFINE_SHIFT_AND_OP(l,min) // vshl_min
DEFINE_SHIFT_AND_OP(l,max) // vshl_max
DEFINE_SHIFT_AND_OP(r,add) // vshr_add
DEFINE_SHIFT_AND_OP(r,min) // vshr_min
DEFINE_SHIFT_AND_OP(r,max) // vshr_max


} // namespace ptx
} // namespace kat


#include "detail/undefine_macros.cuh"

#endif // CUDA_KAT_PTX_VIDEO_INSTRUCTIONS_CUH_
