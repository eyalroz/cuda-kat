/**
 * @file wrappers/shuffle.cuh Templated warp-shuffle operation variants
 */

/*
 * Originally based on Bryan Catanzaro's CUDA generics
 * https://github.com/bryancatanzaro/generics/
 * Downloaded on: 2016-04-16
 * ... but reimplemented by Eyal Rozenberg, CWI Amsterdam
 */

#pragma once
#ifndef CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_
#define CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_

#include <kat/on_device/common.cuh>

#include <kat/define_specifiers.hpp>

// The functions here can be used to shuffle types as large as you like. Of course,
// if they're not plain-old-data, shuffle at your peril.

template<typename T> __fd__ T shuffle_arbitrary(const T& t, const int& source_lane);
template<typename T> __fd__ T shuffle_down(const T& t, const unsigned int& delta);
template<typename T> __fd__ T shuffle_up(const T& t, const unsigned int& delta);
template<typename T> __fd__ T shuffle_xor(const T& t, const int& lane_mask);

#include <kat/undefine_specifiers.hpp>
#include "detail/shuffle.cuh"

#endif /* CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_CUH_ */
