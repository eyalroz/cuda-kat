/*
 * Notes:
 *
 * - Prefer including ../ptx.cuh rather than this file directly
 * - Including this file "pollutes" the rest of your code with preprocessor
 *   includes You may not want. To get rid of them, include
 *   ptx_clear_macros.cuh afterwards
 */

#ifndef PTX_UTILITY_MACROS_DEFINED
#define PTX_UTILITY_MACROS_DEFINED

#include <cstdint>     // for uintXX_t types

#define PTX_STRINGIFY(_q) #_q

#define SIZE_CONSTRAINT_s16 "h"
#define SIZE_CONSTRAINT_u16 "h"
#define SIZE_CONSTRAINT_s32 "r"
#define SIZE_CONSTRAINT_u32 "r"
#define SIZE_CONSTRAINT_s64 "l"
#define SIZE_CONSTRAINT_u64 "l"
#define SIZE_CONSTRAINT_f32 "f"
#define SIZE_CONSTRAINT_f64 "d"

/*
 *  In PTX inline assembly, every variable name must be preceded by a string indicating its size.
 *  Why, would you ask - if the variable _has_ a size which the compiler knows? Just because.
 *  This maps your PTX-style type to its appropriate size indicator string
 */
#define SIZE_CONSTRAINT(ptx_value_type) SIZE_CONSTRAINT_ ## ptx_value_type

/*
 * Always use this as (part of) the
 * constraint string for pointer arguments to PTX inline assembly instructions
 * (see http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints)
 */
#if defined(_WIN64) || defined(__LP64__)
#define PTR_SIZE_CONSTRAINT SIZE_CONSTRAINT(u64)
#else
#define PTR_SIZE_CONSTRAINT SIZE_CONSTRAINT(u32)
#endif

#define CPP_TYPE_BY_PTX_TYPE_s16 int16_t
#define CPP_TYPE_BY_PTX_TYPE_s32 int32_t
#define CPP_TYPE_BY_PTX_TYPE_s64 int64_t
#define CPP_TYPE_BY_PTX_TYPE_u16 uint16_t
#define CPP_TYPE_BY_PTX_TYPE_u32 uint32_t
#define CPP_TYPE_BY_PTX_TYPE_u64 uint64_t
#define CPP_TYPE_BY_PTX_TYPE_f32 float
#define CPP_TYPE_BY_PTX_TYPE_f64 double

/*
 * In our PTX wrapper, we need to declare function parameters and local variables
 * basedon PTX-style types; this is a mechanism for obtaining the corresponding C++
 * type (at the preprocessor level).
 */
#define CPP_TYPE_BY_PTX_TYPE(ptx_value_type) CPP_TYPE_BY_PTX_TYPE_ ## ptx_value_type

#endif /* PTX_UTILITY_MACROS_DEFINED */

