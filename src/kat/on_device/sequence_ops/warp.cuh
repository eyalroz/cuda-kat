/**
 * @file on_device/sequence_ops/warp.cuh
 *
 * @brief GPU device-side versions of `std::algorithm`-like functions, with
 * warp-level collaboration, i.e. different CUDA warps act independently, but
 * all lanes in each warp collaborate on the same task.
 *
 * @note Most functions actually in `std::algorithm` are still missing;
 * see <a href="https://en.cppreference.com/w/cpp/algorithm">the cppreference page</a>
 * for `<algorithm>` for a full list of those.
 *
 * @note some functions here are not actually in `std::algorithm` but might as
 * well have been, e.g. `memzero()` which is like `std::memset()` with 0.
 *
 * @note This is the most-divergent version of std-algorithm-like functions, i.e.
 * don't go looking for thread-level implementations (which would, in fact,
 * be the same as a straightforward CPU-side implementation of `std::algorithm`);
 * if you find yourself needing them, it's possible - perhaps likely - that you're
 * doing something wrong.
 *
 * @todo Some inclusions in the warp-primitives might only be relevant to the
 * functions here; double-check.
 */

#pragma once
#ifndef CUDA_KAT_WARP_COLLABORATIVE_SEQUENCE_OPS_CUH_
#define CUDA_KAT_WARP_COLLABORATIVE_SEQUENCE_OPS_CUH_

#include "common.cuh"
#include <kat/on_device/collaboration/warp.cuh>

#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace collaborative {
namespace warp {

namespace detail {

template<typename LHS, typename RHS = LHS, typename Result = LHS>
struct plus {
	using first_argument_type = LHS;
	using second_argument_type = RHS;
	using result_type = Result;

	KAT_FHD Result operator() (const LHS& x, const RHS& y) const noexcept { return x + y; }
	struct accumulator {
		KAT_FHD Result operator()(
			typename std::enable_if<std::is_same<LHS, RHS>::value, Result>::type& x,
			const RHS& y) const noexcept { return x += y; }
		struct atomic {
#ifdef __CUDA_ARCH__
			KAT_FD Result operator()(
				typename std::enable_if<std::is_same<LHS,RHS>::value, Result>::type& x,
				const RHS& y) const noexcept { return kat::atomic::add(&x,y); }
#endif // __CUDA_ARCH__
		};
		KAT_FHD static Result neutral_value() noexcept { return 0; };
	};
	KAT_FHD static Result neutral_value() noexcept { return 0; };
};

} // namespace detail

/**
 * Performs a reduction (e.g. a summation or a multiplication) of all elements passed into
 * the function by the threads of a block - but with each thread ending up with the reduction
 * result for all threads upto itself.
 *
 * @note What about inclusivity?
 *
 * @todo offer both an inclusive and an exclusive versionn
 */
template<typename T, typename AccumulationOp>
KAT_FD T reduce(T value, AccumulationOp op)
{
	auto partial_result { value };
	for (int shuffle_mask = warp_size/2; shuffle_mask > 0; shuffle_mask >>= 1)
		op(partial_result, shuffle_xor(partial_result, shuffle_mask));
	return partial_result;
}

template <typename T>
KAT_FD T sum(T value)
{
	const auto plus = [](T& x, T y) { x += y; };
	return reduce(value, plus);
}


template <
	typename T,
	typename AccumulationOp,
	inclusivity_t Inclusivity = inclusivity_t::Inclusive
>
KAT_FD T scan(T value, AccumulationOp op)
{
	constexpr const T neutral_value {};

	T x;

	if (Inclusivity == inclusivity_t::Exclusive) {
		T preshuffled = shuffle_up(value, 1);
		// ... and now we can pretend to be doing an inclusive shuffle
		x = lane::is_first() ? neutral_value : preshuffled;
	}
	else { x = value; }

	// x of lane i holds the reduction of values of
	// the lanes i - 2*(offset) ... i - offset , and we've already
	// taken care of the iteration for offset = 0 , above.
	#pragma unroll
	for (int offset = 1; offset < warp_size; offset <<= 1) {
		T shuffled = shuffle_up(x, offset);
		if(lane::id() >= offset) { op(x, shuffled); }
	}
	return x;
}


// TODO: Need to implement a scan-and-reduce warp primitive

template <
	typename T,
	inclusivity_t Inclusivity = inclusivity_t::Inclusive
>
KAT_FD T prefix_sum(T value)
{
	const auto plus = [](T& x, T y) { x += y; };
	return scan<T, decltype(plus), Inclusivity>(value, plus);
}

template <typename T>
KAT_FD T exclusive_prefix_sum(T value)
{
	return prefix_sum<T, inclusivity_t::Exclusive>(value);
}


//--------------------------------------------------

template <typename RandomAccessIterator, typename Size, typename T>
KAT_FD void fill_n(RandomAccessIterator start, Size count, const T& value)
{
	for(auto pos : ranges::warp_stride(count)) {
		start[pos] = value;
	};
}

template <typename RandomAccessIterator, typename T, typename Size = decltype(std::declval<RandomAccessIterator>() - std::declval<RandomAccessIterator>())>
KAT_FD void fill(RandomAccessIterator start, RandomAccessIterator end, const T& value)
{
    Size count = end - start;
    return fill_n(start, count, value);
}

template <typename RandomAccessIterator, typename Size>
KAT_FD void memzero_n(RandomAccessIterator start, Size count)
{
	using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    return fill_n(start, count, value_type{0});
}

template <typename RandomAccessIterator, typename Size = decltype(std::declval<RandomAccessIterator>() - std::declval<RandomAccessIterator>())>
KAT_FD void memzero(RandomAccessIterator start, RandomAccessIterator end)
{
	auto count = end - start;
    return memzero_n(start, count);
}

/**
 * @brief apply a transformation to each element of an array, placing the results
 * in another array.
 *
 * @param source The (block-common) origin of the data
 * @param target The (block-common) destination into which to write the converted elements
 * @param length The (block-common) number of elements available (for reading?] at the
 * source
 */
template <typename T, typename S, typename UnaryOperation, typename Size>
KAT_FD void transform_n(
	const S*  __restrict__  source,
	Size                    length,
	T*        __restrict__  target,
	UnaryOperation          unary_op)
{
	for(auto pos : ranges::warp_stride(length)) {
		target[pos] = unary_op(source[pos]);
	};
}

/**
 * @note Prefer `copy_n()`; this will force the size to `ptrdiff_t`, which unnecessarily large.
 */
template <typename S, typename T, typename UnaryOperation, typename Size = std::ptrdiff_t>
KAT_FD void transform(
	const S*  __restrict__  source_start,
	const S*  __restrict__  source_end,
	T*        __restrict__  target,
	UnaryOperation          unary_op)
{
	Size length = source_end - source_start;
	return transform_n(source_start, length, target, unary_op);
}

//-----------------------------------



/**
 * Have all warp threads collaborate in copying
 * data between two memory locations (possibly not in the same memory
 * space), while also converting types.
 *
 * @param target The destination into which to write the converted elements
 * @param source The origin of the data
 * @param length The number of elements available (for reading?] at the
 * source
 */
template <typename S, typename T, typename Size>
KAT_FD void cast_and_copy_n(
	const S*  __restrict__  source,
	Size                    length,
	T*        __restrict__  target)
{
	auto op = [](S x) -> T { return T(x);} ;
	return transform_n(source, length, target, op);
}

template <typename T, typename U, typename Size = std::ptrdiff_t>
KAT_FD void cast_and_copy(
	const U*  __restrict__  source_start,
	const U*  __restrict__  source_end,
	T*        __restrict__  target)
{
	Size length = source_end - source_start;
	return cast_and_copy_n(source_start, length, target);
}

namespace detail {

/**
 * A version of `kat::copy()` which ignores pointer alignment,
 * and the memory transaction size, simply making coalesced writes
 * of warp_size elements at a time (except for the last range)
 * @param target
 * @param source
 * @param length
 */
template <typename T, typename Size>
KAT_FD void naive_copy(
	const T*  __restrict__  source,
	Size                    length,
	T*        __restrict__  target)
{
	for(auto pos : ranges::warp_stride(length)) {
		target[pos] = source[pos];
	};
}

template <typename T> constexpr KAT_FHD  T clear_lower_bits(T x, unsigned k)
{
	return x & ~((1 << k) - 1);
}


} // namespace detail


/**
 * Has the warp copy data from one place to another
 *
 * @note if the input is not 32-byte (sometimes 128-byte )-aligned,
 * and more importantly, the output is not 128-byte-aligned,
 * performance will likely degrade due to the need to execute a pair
 * of memory transactions for every single 32 x 4 byte write.
 *
 * @tparam     T       type of the elements being copied
 * @tparam     Size    type of the length parameter
 * @tparam     MayHaveSlack
 *                     we "like" data whose size is a multiple of 4 bytes,
 *                     and can copy it faster. When this is true, we assume
 *                     the overall size of data to copy is a multiple of 4,
 *                     without taking the time to check. In the future the
 *                     semantics of this parameter will change to involve
 *                     alignment of the start and end addresses.
 * @param[out] target  starting address of the region of memory to copy into
 * @param[in]  source  starting address of the region of memory to copy from
 * @param[in]  length  number of elements (of type T) to copy
 */
template <typename T, typename Size, bool MayHaveSlack = true>
KAT_FD void copy_n(
	const T*  __restrict__  source,
	Size                    length,
	T*        __restrict__  target)
{
	using namespace linear_grid::grid_info;
	enum {
		elements_per_lane_in_full_warp_write =
			collaborative::detail::elements_per_lane_in_full_warp_write<T>::value
	};

	if ((elements_per_lane_in_full_warp_write == 1) or
	    not (sizeof(T) == 1 or sizeof(T) == 2 or sizeof(T) == 4 or sizeof(T) == 8) or
	    not std::is_trivially_copy_constructible<T>::value)
	{
		detail::naive_copy<T, Size>(source, length, target);
	}
	else {
		// elements_per_lane_in_full_warp_write is either 1, 2...
		kat::array<T, elements_per_lane_in_full_warp_write> buffer;
			// ... so this has either 1 or 2 elements and its overall size is 4

		promoted_size_t<Size> truncated_length = MayHaveSlack ?
			detail::clear_lower_bits(length, constexpr_::log2(elements_per_lane_in_full_warp_write)) :
			length;

		// TODO: Should I pragma-unroll this by a fixed amount? Should
		// I not specify an unroll at all?
		#pragma unroll
		for(promoted_size_t<Size> pos = lane::index() * elements_per_lane_in_full_warp_write;
			pos < truncated_length;
			pos += warp_size * elements_per_lane_in_full_warp_write)
		{
			* (reinterpret_cast<decltype(buffer) *>(target + pos)) =
				*( reinterpret_cast<const decltype(buffer) *>(source + pos) );
		}

		if (MayHaveSlack) {
			if (elements_per_lane_in_full_warp_write == 2) {
				// the slack must be exactly 1
				// Multiple writes to the same place are safe according to
				// the CUDA C Programming Guide v8 section G.3.2 Global Memory
				target[truncated_length] = source[truncated_length];
			}
			else {
				auto num_slack_elements = length - truncated_length;
				if (lane::index() < num_slack_elements) {
					auto pos = truncated_length + lane::index();
					target[pos] = source[pos];
				}
			}
		}
	}
}

template <typename T, bool MayHaveSlack = true, typename Size = std::ptrdiff_t>
KAT_FD void copy(
	const T*  __restrict__  source_start,
	const T*  __restrict__  source_end,
	T*        __restrict__  target_start)
{
	Size length = source_end - source_start;
	return copy_n(source_start, length, target_start);
}


/**
 * Use a lookup table to convert numeric indices to a sequence
 * of values of any type
 */
template <typename T, typename I, typename Size, typename U = T>
KAT_FD void lookup(
	T*       __restrict__  target,
	const U* __restrict__  lookup_table,
	const I* __restrict__  indices,
	Size                   num_indices)
{
	for(auto pos : ranges::warp_stride(num_indices)) {
		target[pos] = lookup_table[indices[pos]];
	};
}

/**
 * Perform an accumulation operation (e.g. addition) between equal-sized arrays -
 * with either regular or atomic semantics. Usable with memory locations which
 * the entire block has the same view of and accessibility to (mostly shared
 * and global, but not just those).
 *
 * @note
 * 1. Assumes a linear block.
 * 2. The operation is supposed to have the signature:
 *      WhateverWeDontCare operation(D& accumulator_element, S value)
 *    otherwise it might be a no-op here.
 * 3. If you're having multiple blocks calling this function with the same
 *    destination, it will have to be atomic (as you cannot guarantee these blocks will
 *    not execute simultaneously, either on different multiprocessors or on the same
 *    multiprocessor). Also, if you want to use a global-mem source, you will
 *    need to pass this function block-specific offsets; remember it is not
 *    a kernel!
 *
 * @tparam D Destination data type
 * @tparam S Source data type
 * @tparam AccumulatingOperation Typically, one of the 'accumulator' substructures of
 * the functors in liftedfunctions.hpp ; but it may very well be an accumulator::atomic
 * substructure
 * @tparam Size ... so that you don't have to decide whether you want to specify your
 * number of elements as an int, uint, long long int, ulong long etc.
 * @param[inout] destination The array into which we accumulate; holds existing data
 * and is not simply overwritten.
 * @param[in] source The array of partial data to integrate via accumulation.
 * @param[in] length the length in elements of @p destination and @p source
 *
 * @todo consider taking a GSL-span-like parameter isntead of a ptr+length
 *
 * @todo Some inclusions in the block-primitives might only be relevant to the
 * functions here; double-check.
 *
 * @todo consider using elementwise_apply for this.
 *
 */
template <typename D, typename RandomAccessIterator, typename AccumulatingOperation, typename Size>
KAT_FD void elementwise_accumulate_n(
	AccumulatingOperation              op,
	D*                   __restrict__  destination,
	RandomAccessIterator __restrict__  source,
	Size                               length)
{
	for(auto pos : ranges::warp_stride(length)) {
		op(destination[pos], source[pos]);
	};
}

template <typename D, typename RandomAccessIterator, typename AccumulatingOperation, typename Size = std::ptrdiff_t>
KAT_FD void elementwise_accumulate(
	AccumulatingOperation              op,
	D*                   __restrict__  destination,
	RandomAccessIterator __restrict__  source_start,
	RandomAccessIterator __restrict__  source_end)
{
	elementwise_accumulate_n(op, destination, source_start,
		// kat::distance(source_start, source_end)
		source_end - source_start
		);
}

} // namespace warp
} // namespace collaborative
} // namespace kat

#endif // CUDA_KAT_WARP_COLLABORATIVE_SEQUENCE_OPS_CUH_
