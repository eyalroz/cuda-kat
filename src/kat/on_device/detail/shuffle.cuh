#pragma once
#ifndef CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_
#define CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_

#include <kat/on_device/common.cuh>
#include <sm_30_intrinsics.h>
#include <vector_types.h>
#include <kat/containers/array.hpp>
#include <kat/on_device/builtins.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace detail {

template<typename InputIterator, typename OutputIterator, class UnaryOperation>
KAT_FD OutputIterator transform(
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
KAT_FD static void shuffle_arbitrary(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const int                  source_lane)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&source_lane](const int& x) { return builtins::warp::shuffle::arbitrary(x, source_lane); });
}

template<int s>
KAT_FD
static void shuffle_down(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const unsigned int         delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return builtins::warp::shuffle::down(x, delta); });
}

template<int s>
KAT_FD static void shuffle_up(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const unsigned int         delta)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&delta](const int& x) { return builtins::warp::shuffle::up(x, delta); });
}

template<int s>
KAT_FD static void shuffle_xor(
	const kat::array<int, s>&  in,
	kat::array<int, s>&        result,
	const int                  mask)
{
	transform(
		in.begin(), in.end(), result.begin(),
		[&mask](const int& x) { return builtins::warp::shuffle::xor_(x, mask); });
}

} // namespace detail

template<typename T>
KAT_FD T shuffle_arbitrary(const T& t, int source_lane) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
	constexpr auto num_full_int_shuffles = sizeof(T)/sizeof(int);
	if (num_full_int_shuffles > 0) {
	auto& as_array = reinterpret_cast<kat::array<int, num_full_int_shuffles>&>(result);
	auto& t_as_array = reinterpret_cast<const kat::array<int, num_full_int_shuffles>&>(t);
	detail::shuffle_arbitrary<num_full_int_shuffles>(t_as_array, as_array, source_lane);
	}
	constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
	if (sub_int_remainder_size > 0) {
		int surrogate;
		const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
		surrogate = builtins::warp::shuffle::arbitrary(surrogate, source_lane, warp_size);
		int& result_remainder = *(reinterpret_cast<int*>(&result) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
	return result;
}

template<typename T>
KAT_FD T shuffle_down(const T& t, unsigned int delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
	constexpr auto num_full_int_shuffles = sizeof(T)/sizeof(int);
	if (num_full_int_shuffles > 0) {
	auto& as_array = reinterpret_cast<kat::array<int, num_full_int_shuffles>&>(result);
	auto& t_as_array = reinterpret_cast<const kat::array<int, num_full_int_shuffles>&>(t);
	detail::shuffle_down<num_full_int_shuffles>(t_as_array, as_array, delta);
	}
	constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
	if (sub_int_remainder_size > 0) {
		int surrogate;
		const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
		surrogate = builtins::warp::shuffle::down(surrogate, delta, warp_size);
		int& result_remainder = *(reinterpret_cast<int*>(&result) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
	return result;
}

template<typename T>
KAT_FD T shuffle_up(const T& t, unsigned int delta) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
	constexpr auto num_full_int_shuffles = sizeof(T)/sizeof(int);
	if (num_full_int_shuffles > 0) {
		auto& as_array = reinterpret_cast<kat::array<int, num_full_int_shuffles>&>(result);
		auto& t_as_array = reinterpret_cast<const kat::array<int, num_full_int_shuffles>&>(t);
		detail::shuffle_up<num_full_int_shuffles>(t_as_array, as_array, delta);
	}
	constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
	if (sub_int_remainder_size > 0) {
		int surrogate;
		const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
		surrogate = builtins::warp::shuffle::up(surrogate, delta, warp_size);
		int& result_remainder = *(reinterpret_cast<int*>(&result) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
	return result;
}

template<typename T>
KAT_FD T shuffle_xor(const T& t, int mask) {
	static_assert(sizeof(int) == 4 && sizeof(short) == 2 && sizeof(char) == 1, "Sizes sanity check failed");
	T result;
	constexpr auto num_full_int_shuffles = sizeof(T)/sizeof(int);
	if (num_full_int_shuffles > 0) {
		auto& as_array = reinterpret_cast<kat::array<int, num_full_int_shuffles>&>(result);
		auto& t_as_array = reinterpret_cast<const kat::array<int, num_full_int_shuffles>&>(t);
		detail::shuffle_xor<num_full_int_shuffles>(t_as_array, as_array, mask);
	}
	constexpr auto sub_int_remainder_size = sizeof(T) % sizeof(int);
	if (sub_int_remainder_size > 0) {
		int surrogate;
		const int& t_remainder = *(reinterpret_cast<const int*>(&t) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (surrogate) = reinterpret_cast<const char&> (t_remainder); break;
		case 2: reinterpret_cast<short&>(surrogate) = reinterpret_cast<const short&>(t_remainder); break;
		case 3: reinterpret_cast<char3&>(surrogate) = reinterpret_cast<const char3&>(t_remainder); break;
		}
		surrogate = builtins::warp::shuffle::xor_(surrogate, mask, warp_size);
		int& result_remainder = *(reinterpret_cast<int*>(&result) + num_full_int_shuffles);
		switch(sub_int_remainder_size) {
		case 1: reinterpret_cast<char&> (result_remainder) = reinterpret_cast<char&> (surrogate); break;
		case 2: reinterpret_cast<short&>(result_remainder) = reinterpret_cast<short&>(surrogate); break;
		case 3: reinterpret_cast<char3&>(result_remainder) = reinterpret_cast<char3&>(surrogate); break;
		}
	}
	return result;
}

} // namespace kat

#endif // CUDA_KAT_ON_DEVICE_TEMPLATED_SHUFFLE_DETAIL_CUH_
