/**
 * @file define_specifiers.hpp
 *
 * @brief A "preprocessor utility" header for defining declaration specifier shorthands,
 * used throughout the cuda-kat library for brevity.
 *
 * @note see @ref undefine_specifiers.hpp for the undefining macros - used to avoid
 * "polluting the macro namespace" after cuda-kat functions have been defined.
 *
 * @note This is not intended merely for in-library usage - it is rather useful for
 * other code as well.
 */
#ifdef __CUDACC__

#ifndef __fd__
#define __fd__  __forceinline__ __device__
#endif

#ifndef __fh__
#define __fh__  __forceinline__ __host__
#endif

#ifndef __fhd__
#define __fhd__ __forceinline__ __host__ __device__
#endif

#ifndef __hd__
#define __hd__ __host__ __device__
#endif

#else // __CUDACC__

#ifndef __fd__
#define __fd__ inline
#endif

#ifndef __fh__
#define __fh__ inline
#endif

#ifndef __fhd__
#define __fhd__ inline
#endif

#ifndef __hd__
#define __hd__
#endif

#endif // __CUDACC__
