/*
 * Notes:
 *
 * - Try not to use this file directly, but rather just ../ptx.cuh
 * - If you are using this file directly - include it after having
 *   included utilities.cuh - and any other include files which
 *   uses these utilities.
 */

#ifdef PTX_UTILITY_MACROS_DEFINED

#undef PTR_SIZE_CONSTRAINT
#undef CPLUSPLUS_VARIABLE_TYPE
#undef CPLUSPLUS_VARIABLE_TYPE_u16
#undef CPLUSPLUS_VARIABLE_TYPE_u32
#undef CPLUSPLUS_VARIABLE_TYPE_u64
#undef CPLUSPLUS_VARIABLE_TYPE_f32
#undef CPLUSPLUS_VARIABLE_TYPE_f64
#undef SIZE_CONSTRAINT
#undef SIZE_CONSTRAINT_u16
#undef SIZE_CONSTRAINT_u32
#undef SIZE_CONSTRAINT_u64
#undef SIZE_CONSTRAINT_f32
#undef SIZE_CONSTRAINT_f64

#undef PTX_STRINGIFY

#undef PTX_UTILITY_MACROS_DEFINED

#endif // PTX_UTILITY_MACROS_DEFINED
