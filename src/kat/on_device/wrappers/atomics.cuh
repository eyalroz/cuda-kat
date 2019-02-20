/**
 * @file Wrappers for CUDA atomic operations
 *
 * Copyright (c) 2018, Eyal Rozenberg, CWI Amsterdam
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
#ifndef CUDA_ON_DEVICE_ATOMICS_CUH_
#define CUDA_ON_DEVICE_ATOMICS_CUH_

#include <kat/define_specifiers.hpp>

namespace atomic {

/*
 * TODO:
 * - Consider using non-const references instead of pointers (but make sure we get the same PTX)
 * - Consider creating an atomic<T> wrapper class which has these as operators.
 */
template <typename T>  __fd__ T add        (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T subtract   (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T increment  (T* __restrict__ address, const T& wraparound_value);
template <typename T>  __fd__ T decrement  (T* __restrict__ address, const T& wraparound_value);
template <typename T>  __fd__ T exchange   (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T min        (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T max        (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T logical_and(T* __restrict__ address, const T& val);
template <typename T>  __fd__ T logical_or (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T logical_xor(T* __restrict__ address, const T& val);
template <typename T>  __fd__ T bitwise_or (T* __restrict__ address, const T& val);
template <typename T>  __fd__ T bitwise_and(T* __restrict__ address, const T& val);
template <typename T>  __fd__ T bitwise_xor(T* __restrict__ address, const T& val);
template <typename T>  __fd__ T bitwise_not(T* __restrict__ address);
template <typename T>  __fd__ T set_bit    (T* __restrict__ address, const unsigned bit_index);
template <typename T>  __fd__ T unset_bit  (T* __restrict__ address, const unsigned bit_index);

/**
 * Use atomic compare-and-swap to apply a unary function to some value,
 * replacing it at its memory location with the result before anything
 * else changes it.
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename UnaryFunction>
__fd__ T apply_atomically(UnaryFunction f, T* __restrict__ address);

/**
 * Use atomic compare-and-swap to apply a binary function to two values,
 * replacing the first at its memory location with the result before anything
 * else changes it.
 *
 * @todo Take a template parameter pack and multiple arguments to f
 *
 * @return The new value which was stored in memory
 */
template <typename T, typename BinaryFunction>
__fd__ T apply_atomically(
	BinaryFunction   f,
	T* __restrict__  address,
	const T&         rhs);


} // namespace atomic

#include <kat/undefine_specifiers.hpp>
#include "detail/atomics.cuh"

#endif /* CUDA_ON_DEVICE_ATOMICS_CUH_ */
