/**
 * @file Templated warp-shuffle operation variants
 *
 * Originally based on Bryan Catanzaro's CUDA generics
 * https://github.com/bryancatanzaro/generics/
 * Downloaded on: 2016-04-16
 * ... but reimplemented by Eyal Rozenberg, CWI Amsterdam
 */

#pragma once
#ifndef CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_
#define CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_

#include <kat/on_device/common.cuh>
#include <sm_30_intrinsics.h>
#include <vector_types.h>
#include <kat/containers/array.hpp>
#include <cuda/api/types.hpp>
#include <kat/on_device/wrappers/builtins.cuh>

#include <kat/define_specifiers.hpp>

namespace detail {

template<typename InputIterator, typename OutputIterator, class UnaryOperation>
__fd__ OutputIterator transform(
	InputIterator input_it, InputIterator input_sentinel,
	OutputIterator output_it, UnaryOperation unary_op)
{
    while (input_it != input_sentinel ) {
    	*output_it = unary_op(*input_it);
    	++input_it, ++output_it;
    }
    return output_it;
}

template<int s>
__fd__ static void shuffle_arbitrary(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const int                   source_lane)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&source_lane](const int& x) { return ::builtins::warp::shuffle::arbitrary(x, source_lane); });
}

template<int s>
__fd__
static void shuffle_down(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const unsigned int          delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return ::builtins::warp::shuffle::down(x, delta); });
}

template<int s>
__fd__ static void shuffle_up(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const unsigned int          delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return ::builtins::warp::shuffle::up(x, delta); });
}

template<int s>
__fd__ static void shuffle_xor(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const int                   lane_mask)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&lane_mask](const int& x) { return ::builtins::warp::shuffle::xor_(x, lane_mask); });
}


} // namespace detail

template<typename T>
__fd__ T shuffle_arbitrary(const T& t, const int& source_lane) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<kat::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const kat::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_arbitrary<num_int_shuffles>(t_as_array, as_array, source_lane);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::builtins::warp::shuffle::arbitrary(surrogate, source_lane, warp_size);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__fd__ T shuffle_down(const T& t, const unsigned int& delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<kat::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const kat::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_down<num_int_shuffles>(t_as_array, as_array, delta);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::builtins::warp::shuffle::down(surrogate, delta, warp_size);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__fd__ T shuffle_up(const T& t, const unsigned int& delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<kat::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const kat::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_up<num_int_shuffles>(t_as_array, as_array, delta);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::builtins::warp::shuffle::up(surrogate, delta, warp_size);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

template<typename T>
__fd__ T shuffle_xor(const T& t, const int& lane_mask) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
    constexpr auto num_int_shuffles = sizeof(T)/sizeof(int);
    if (num_int_shuffles > 0) {
    	auto& as_array = reinterpret_cast<kat::array<int, num_int_shuffles>&>(result);
    	auto& t_as_array = reinterpret_cast<const kat::array<int, num_int_shuffles>&>(t);
    	detail::shuffle_xor<num_int_shuffles>(t_as_array, as_array, lane_mask);
    }
    constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
    if (sub_int_remainder_size > 0) {
    	int surrogate;
    	const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
    	surrogate = ::builtins::warp::shuffle::xor_(surrogate, lane_mask, warp_size);
    	int& result_remainder = *(reinterpret_cast<int*>(&result) + num_int_shuffles);
    	switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
    return result;
}

#include <kat/undefine_specifiers.hpp>

#endif /* CUDA_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_ */
