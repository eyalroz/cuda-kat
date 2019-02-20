/**
 * @file on_device/time.cuh
 *
 * @brief CUDA device-side functions having to do with timing and the hardware clock.
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
#ifndef CUDA_ON_DEVICE_TIME_CUH_
#define CUDA_ON_DEVICE_TIME_CUH_

#include <kat/define_specifiers.hpp>

enum class sleep_resolution { clock_cycles, nanoseconds };

using clock_value_t = long long;
	// CUDA uses a signed type for clock values - for some unknown reason; See the declaration of clock64()

namespace detail {

template <sleep_resolution Resolution>
struct sleep_unit;

template  struct sleep_unit<sleep_resolution::clock_cycles> { using type = clock_value_t; };
template  struct sleep_unit<sleep_resolution::nanoseconds > { using type = unsigned int; };
	// Why unsigned int? See the declaration of nanosleep()...

} // namespace detail

template <sleep_resolution Resolution>
using sleep_unit_t = typename detail::sleep_unit<Resolutiom>::type;


/**
 * @brief Have the calling warp busy-sleep for (at least) a certain
 * number of clock cycles.
 *
 * @note In 2017, a typical GPU clock cycle is around 1 ns (i.e. 1 GHz frequency).
 *
 */
template <sleep_resolution Resolution>
__fd__ void sleep(sleep_unit_t<Resolution> num_cycles);

template
__fd__ void sleep<sleep_resolution::clock_cycles>(
	sleep_unit_t<sleep_resolution::clock_cycles> num_cycles)
{
	// The clock64() function returns an SM-specific clock ticks value,
	// which occasionally gets reset. Even if it were not reset, it would
	// only wrap around in 300 years or so since it began ticking, which is
	// why there's no need to check for wrap-around.
	// Also, it seems this code is not optimized-away despite not having
	// any obvious side effects.
	clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; }
    while (cycles_elapsed < num_cycles);
}

#if __CUDA_ARCH__ >= 700

template
__fd__ void sleep<sleep_resolution::clock_cycles>(
	sleep_unit_t<sleep_resolution::clock_cycles> num_cycles)
{
	__nanosleep(unsigned int ns);
}

#endif // __CUDA_ARCH__ >= 700

#include <kat/undefine_specifiers.hpp>

#endif /* CUDA_ON_DEVICE_TIME_CUH_ */
