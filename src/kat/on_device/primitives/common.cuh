/**
 * @file on_device/primitives/common.cuh
 *
 * @brief Some common definitions for all on-device computational primitives (grid,
 * block, warp, thread and lane).
 */

/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, Eyal Rozenberg and CWI Amsterdam
 * Copyright (c) 2019, Eyal Rozenberg
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_
#define CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_

#include <cuda/api/types.hpp>
#include <kat/on_device/miscellany.cuh>
#include <kat/on_device/math.cuh>
#ifdef DEBUG
#include <kat/on_device/printing.cuh>
#endif

#include <type_traits>

namespace cuda {
namespace primitives {

enum inclusivity_t : bool {
	Exclusive = false,
	Inclusive = true
};

namespace detail {

/**
 * In a "full warp write", we want [1] each lane to write an integral number
 * native words (at the moment and for the foreseeable future, 4-byte integers).
 * At the same time, the lane writes complete elements of type T, not arbitrary
 * sequences of `sizeof(native_word_t)`, hence this definition.
 *
 * @todo: Can't we assume that T is a POD type, and just have lanes not write
 * complete T's?
 */
template <typename T>
struct elements_per_lane_in_full_warp_write {
    enum { value = sizeof(native_word_t) / constexpr_::gcd<unsigned>(sizeof(native_word_t),sizeof(T)) };
};
} // namespace detail

} // namespace primitives
} // namespace cuda

#endif /* CUDA_ON_DEVICE_PRIMITIVES_COMMON_CUH_ */
