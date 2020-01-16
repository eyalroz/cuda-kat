/*
 *
 * Copyright (c) 2019, Eyal Rozenberg <eyalroz@technion.ac.il>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of CWI Amsterdam nor the names of its contributors may
 *    be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
#pragma once
#ifndef CUDA_CUDA_KAT_ON_DEVICE_ATOMICS_MISSING_FROM_CUDA_CUH_
#define CUDA_CUDA_KAT_ON_DEVICE_ATOMICS_MISSING_FROM_CUDA_CUH_

#include <device_atomic_functions.h>

static_assert(sizeof (unsigned long int) == sizeof(unsigned long long int), "Unexpected size of unsigned long int");

// Annoyingly, CUDA - upto and including version 10.0 - provide atomic
// operation wrappers for unsigned int and unsigned long long int, but
// not for the in-between type of unsigned long int. So - we
// have to make our own.
//
// TODO: On CUDA devices, sizeof(long) is 8, like sizeof(long long). However,
// that's not true on Windows host-side code. Need to double check this discrepancy
// doesn't mess this code's correctness up somehow.

#define CUDA_KAT_DEFINE_MISSING_ATOMIC(arg_type, op) \
KAT_FD arg_type atomic ## op(arg_type *address, arg_type val) \
{ \
	return ::atomicAdd(reinterpret_cast<unsigned long long int*>(address), reinterpret_cast<arg_type&>(val)); \
}

#define CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(op) \
CUDA_KAT_DEFINE_MISSING_ATOMIC(unsigned long, op) \
CUDA_KAT_DEFINE_MISSING_ATOMIC(long, op) \
CUDA_KAT_DEFINE_MISSING_ATOMIC(long long, op)

CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(Add)
#if CUDA_ARCH >= 320
CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(And)
CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(Or)
CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(Xor)
CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(Min)
CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP(Max)
#endif

#undef CUDA_KAT_DEFINE_MISSING_ATOMICS_FOR_OP
#undef CUDA_KAT_DEFINE_MISSING_ATOMIC

#endif // CUDA_CUDA_KAT_ON_DEVICE_ATOMICS_MISSING_FROM_CUDA_CUH_
