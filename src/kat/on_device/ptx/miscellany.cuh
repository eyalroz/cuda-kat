/**
 * @file ptx/miscellany.cuh Non-templated wrappers for PTX instructions, which nVIDIA
 * does not provide wrappers for through the CUDA `<device_functions.h>` header.
 */
#pragma once
#ifndef CUDA_KAT_ON_DEVICE_PTX_MISCELLANY_CUH_
#define CUDA_KAT_ON_DEVICE_PTX_MISCELLANY_CUH_

#include "detail/define_macros.cuh"
#include <kat/on_device/common.cuh>

#include <cstdint>
#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

namespace ptx {

/**
 * @brief Load data through the read-only data cache
 *
 * @note See the <a href="http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ldg-function">relevant section</a>
 * of the CUDA C Programming guide for details on using this instruction for loading.
 *
 * @param ptr The global memory location from which to load
 * @return the value at location @p ptr , loaded through the read-only data cache rather than
 * through the usual (read-write) caches
 */
template <typename T>
KAT_FD T ldg(const T* ptr)
{
#if __CUDA_ARCH__ >= 320
	return __ldg(ptr);
#else
	return *ptr; // maybe we should ld.cg or ld.cs here?
#endif
}

/**
 * See <a href="http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep">relevant section</a>
 * of the CUDA PTX reference for details on these instructions.
 */
#define DEFINE_IS_IN_MEMORY_SPACE(_which_space) \
KAT_FD int32_t is_in_ ## _which_space ## _memory (const void *ptr) \
{ \
	int32_t result; \
	asm ("{\n\t" \
		".reg .pred p;\n\t" \
		"isspacep." PTX_STRINGIFY(_which_space) " p, %1;\n\t" \
		"selp.b32 %0, 1, 0, p;\n\t" \
		"}" \
		: "=r"(result) : PTR_SIZE_CONSTRAINT(ptr)); \
	return result; \
}

DEFINE_IS_IN_MEMORY_SPACE(const)  // is_in_const_memory
DEFINE_IS_IN_MEMORY_SPACE(global) // is_in_global_memory
DEFINE_IS_IN_MEMORY_SPACE(local)  // is_in_local_memory
DEFINE_IS_IN_MEMORY_SPACE(shared) // is_in_shared_memory

#undef DEFINE_IS_IN_MEMORY_SPACE

/*
 * @brief Find the last non-sign bit in a signed or an unsigned integer value
 *
 * @note See <a href="http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind">relevant section</a>
 * of the CUDA PTX reference for details on this instruction.
 *
 * @param val the value in which to find non-sign bits
 * @return the bit index (counting from least significant bit being 0) of the first
 * bit which is 0 if @p val is positive, or of the first bit which is 1 if @p val is negative. If @p val has only
 * sign bits (i.e. if it's 0 or if its type is signed and its bits are all 1) - the value 0xFFFFFFFF (-1) is returned
 */

#define DEFINE_BFIND(ptx_value_type) \
KAT_FD uint32_t \
bfind(CPP_TYPE_BY_PTX_TYPE(ptx_value_type) val) \
{ \
	uint32_t ret;  \
	asm ( \
		"bfind." PTX_STRINGIFY(ptx_value_type) " %0, %1;" \
		: "=r"(ret) : SIZE_CONSTRAINT(ptx_value_type) (val)); \
	return ret; \
}

DEFINE_BFIND(s32) // bfind
DEFINE_BFIND(s64) // bfind
DEFINE_BFIND(u32) // bfind
DEFINE_BFIND(u64) // bfind

#undef DEFINE_BFIND

#define DEFINE_PRMT_WITH_MODE(selection_mode_name, selection_mode) \
KAT_FD  uint32_t prmt_ ## selection_mode_name (uint32_t first, uint32_t second, uint32_t control_bits) \
{ \
	uint32_t result; \
	asm("prmt.b32." PTX_STRINGIFY(selection_mode) " %0, %1, %2, %3;" \
		: "=r"(result) : "r"(first), "r"(second), "r"(control_bits)); \
	return result; \
}

/*
 * See:
 * @url http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt
 * for information about these instructions
 */
DEFINE_PRMT_WITH_MODE( forward_4_extract,  f4e  ) // prmt_forward_4_extract
DEFINE_PRMT_WITH_MODE( backward_4_extract, b4e  ) // prmt_backward_4_extract
DEFINE_PRMT_WITH_MODE( replicate_8,        rc8  ) // prmt_replicate_8
DEFINE_PRMT_WITH_MODE( replicate_16,       rc16 ) // prmt_replicate_16
DEFINE_PRMT_WITH_MODE( edge_clam_left,     ecl  ) // prmt_edge_clam_left
DEFINE_PRMT_WITH_MODE( edge_clam_right,    ecl  ) // prmt_edge_clam_right

/**
 * @brief Aborts execution (of the entire kernel grid) and generates an interrupt to the host CPU.
 */
KAT_FD  void trap()
{
	asm("trap;");
}

/**
 * Ends execution of the current thread of this kernel/grid
 */
KAT_FD void exit()
{
	asm("exit;");
}


/**
 * @brief See: <a href="http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt">relevant section</a>
 * of the CUDA PTX reference for an explanation of what this does exactly
 *
 * @param first           a first value from which to potentially use bytes
 * @param second          a second value from which to potentially use bytes
 * @param byte_selectors  a packing of 4 selector structures; each selector structure
 *                        is 3 bits specifying which of the input bytes are to be used (as there are 8
 *                        bytes overall in @p first and @p second ), and another bit specifying if it's an
 *                        actual copy of a byte, or instead whether the sign of the byte (intrepeted as
 *                        an int8_t) should be replicated to fill the target byte.
 * @return the four bytes of first and/or second, or replicated signs thereof, indicated by the byte selectors
 *
 * @note Only the lower 16 bits of byte_selectors are used.
 * @note "prmt" stands for "permute"
 */
KAT_FD uint32_t prmt(uint32_t first, uint32_t second, uint32_t byte_selectors)
{
	uint32_t result;
	asm("prmt.b32 %0, %1, %2, %3;"
		: "=r"(result) : "r"(first), "r"(second), "r"(byte_selectors));
	return result;
}


/**
 * @brief Extracts the bits with 0-based indices start_pos...start_pos+length-1, counting
 * from least to most significant, from a bit field field. Has sign extension semantics
 * for signed inputs which are bit tricky, see in the PTX ISA guide:
 *
 * http://docs.nvidia.com/cuda/parallel-thread-execution/index.html
 *
 * TODO: CUB 1.5.2's BFE wrapper seems kind of fishy. Why does Duane Merill not use PTX for extraction from 64-bit fields?
 * I'll take a different route.
 */
#define DEFINE_BFE(ptx_value_type) \
KAT_FD CPP_TYPE_BY_PTX_TYPE(ptx_value_type) \
bfe( \
	CPP_TYPE_BY_PTX_TYPE(ptx_value_type) bits, \
	uint32_t start_position, \
	uint32_t num_bits) \
{ \
	CPP_TYPE_BY_PTX_TYPE(ptx_value_type) extracted_bits;  \
	asm ( \
		"bfe." PTX_STRINGIFY(ptx_value_type) " %0, %1, %2, %3;" \
		: "=" SIZE_CONSTRAINT(ptx_value_type) (extracted_bits) \
		: SIZE_CONSTRAINT(ptx_value_type) (bits) \
		, "r" (start_position) \
		, "r" (num_bits) \
	);\
	return extracted_bits; \
}

DEFINE_BFE(s32) // bfe
DEFINE_BFE(s64) // bfe
DEFINE_BFE(u32) // bfe
DEFINE_BFE(u64) // bfe

#undef DEFINE_BFE

KAT_FD uint32_t
bfi(
	uint32_t  bits_to_insert,
	uint32_t  existing_bit_field,
	uint32_t  start_position,
	uint32_t  num_bits)
{
	uint32_t ret;
	asm (
		"bfi.b32 %0, %1, %2, %3, %4;"
		: "=r"(ret)
		: "r"(bits_to_insert)
		, "r"(existing_bit_field)
		, "r"(start_position)
		, "r"(num_bits)
	);
	return ret;
}

KAT_FD uint64_t
bfi(
	uint64_t  bits_to_insert,
	uint64_t  existing_bit_field,
	uint32_t  start_position,
	uint32_t  num_bits)
{
	uint64_t ret;
	asm (
		"bfi.b64 %0, %1, %2, %3, %4;"
		: "=l"(ret)
		: "l"(bits_to_insert)
		, "l"(existing_bit_field)
		, "r"(start_position)
		, "r"(num_bits)
	);
	return ret;
}

} // namespace ptx
} // namespace kat


#include "detail/undefine_macros.cuh"

#endif // CUDA_KAT_ON_DEVICE_PTX_MISCELLANY_CUH_

