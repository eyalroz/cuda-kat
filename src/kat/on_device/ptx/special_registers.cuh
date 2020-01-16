/**
 * @file kat/on_device/ptx/special_registers.cuh
 *
 * @brief PTX instruction wrapper functions for accessing special on-GPU-core registers.
 */
#pragma once
#ifndef CUDA_KAT_PTX_SPECIAL_REGISTERS_CUH_
#define CUDA_KAT_PTX_SPECIAL_REGISTERS_CUH_

#include "detail/define_macros.cuh"

namespace kat {
namespace ptx {

/**
 * @brief Wrappers for instructions obtaining the value of one of the special hardware registers on nVIDIA GPUs.
 *
 * See the <a href="http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers">relevant section</a>
 * of the PTX instruction set guide for more details.
 */
namespace special_registers {


#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, ptx_value_type) \
KAT_FD CPP_TYPE_BY_PTX_TYPE(ptx_value_type) special_register_name() \
{ \
	CPP_TYPE_BY_PTX_TYPE(ptx_value_type) ret;  \
	asm volatile ("mov." PTX_STRINGIFY(ptx_value_type) "%0, %" PTX_STRINGIFY(special_register_name) ";" : "=" SIZE_CONSTRAINT(ptx_value_type) (ret)); \
	return ret; \
} \

DEFINE_SPECIAL_REGISTER_GETTER( laneid,             u32); // PTX 1.3
DEFINE_SPECIAL_REGISTER_GETTER( gridid,             u64); // PTX 3.0
DEFINE_SPECIAL_REGISTER_GETTER( smid,               u32); // PTX 1.3
DEFINE_SPECIAL_REGISTER_GETTER( nsmid,              u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( clock,              u32); // PTX 1.0
DEFINE_SPECIAL_REGISTER_GETTER( clock_hi,           u32); // PTX 5.0
DEFINE_SPECIAL_REGISTER_GETTER( clock64,            u64); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( globaltimer_hi,     u32); // PTX 3.1
DEFINE_SPECIAL_REGISTER_GETTER( globaltimer_lo,     u32); // PTX 3.1
DEFINE_SPECIAL_REGISTER_GETTER( globaltimer,        u64); // PTX 3.1
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_lt,        u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_le,        u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_eq,        u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_ge,        u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( lanemask_gt,        u32); // PTX 2.0
DEFINE_SPECIAL_REGISTER_GETTER( dynamic_smem_size,  u32); // PTX 4.1
DEFINE_SPECIAL_REGISTER_GETTER( total_smem_size,    u32); // PTX 4.1

#undef DEFINE_SPECIAL_REGISTER_GETTER


/*
 * Not defining getters for:
 *
 * %tid                      - available as threadIdx
 * %ntid                     - available as blockDim
 * %warpid                   - not interesting
 * %nwarpid                  - not interesting
 * %ctaid                    - available as blockId
 * %nctaid                   - available as gridDim
 * %pm0, ..., %pm7           - not interesting, for now (performance monitoring)
 * %pm0_64, ..., %pm7_64     - not interesting, for now (performance monitoring)
 * %envreg0, ..., %envreg31  - not interesting, for now
 */


} // namespace special_registers

} // namespace ptx

} // namespace kat

#include "detail/undefine_macros.cuh"


#endif // CUDA_KAT_PTX_SPECIAL_REGISTERS_CUH_
