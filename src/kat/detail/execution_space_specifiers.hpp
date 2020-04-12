/**
 * @file execution_space_specifiers.hpp
 *
 * @brief Some functions need a specification of their appropriate execution space w.r.t. the
 * CUDA device-vs-host-side side, as well as their inlining requirement. For brevity,
 * we introduce shorthands for these.
 */

#ifndef EXECUTION_SPACE_SPECIFIERS_HPP_
#define EXECUTION_SPACE_SPECIFIERS_HPP_

///@cond

#ifdef __CUDACC__

#ifndef KAT_FD
#define KAT_FD  __forceinline__ __device__
#endif

#ifndef KAT_FH
#define KAT_FH  __forceinline__ __host__
#endif

#ifndef KAT_FHD
#define KAT_FHD __forceinline__ __host__ __device__
#endif

#ifndef KAT_ID
#define KAT_ID  inline __device__
#endif

#ifndef KAT_IH
#define KAT_IH  inline __host__
#endif

#ifndef KAT_IHD
#define KAT_IHD inline __host__ __device__
#endif

#ifndef KAT_HD
#define KAT_HD __host__ __device__
#endif

#ifndef KAT_DEV
#define KAT_DEV __device__
#endif

#ifndef KAT_HOST
#define KAT_HOST __host__
#endif

#else // __CUDACC__

#ifndef KAT_FD
#define KAT_FD inline
#endif

#ifndef KAT_FH
#define KAT_FH inline
#endif

#ifndef KAT_FHD
#define KAT_FHD inline
#endif

#ifndef KAT_ID
#define KAT_ID inline
#endif

#ifndef KAT_IH
#define KAT_IH inline
#endif

#ifndef KAT_IHD
#define KAT_IHD inline
#endif

#ifndef KAT_HD
#define KAT_HD
#endif

#ifndef KAT_DEV
#define KAT_DEV
#endif

#ifndef KAT_HOST
#define KAT_HOST
#endif

#endif // __CUDACC__

///@endcond


#endif // EXECUTION_SPACE_SPECIFIERS_HPP_
