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
 * Performs a reduction (e.g. a summation of multiplication) of all elements passed into
 * the function by the threads of a block.
 *
 * @note This ignores overflow! Make sure you use a roomy type. Alternatively, you
 * could implement a two-type version which takes input in one type and works on a
 * bigger one.
 */
template <typename ReductionOp>
KAT_FD typename ReductionOp::result_type reduce(typename ReductionOp::first_argument_type value)
{
	static_assert(std::is_same<
		typename ReductionOp::first_argument_type,
		typename ReductionOp::second_argument_type>::value, "A reduction operation "
			"must have the same types for its LHS and RHS");
	static_assert(std::is_same<
		typename ReductionOp::first_argument_type,
		typename ReductionOp::result_type>::value, "The warp shuffle primitive "
			"can only be applied with a reduction op having the same result and "
			"argument types");
	typename ReductionOp::accumulator op_acc;
	// Let's cross our fingers and hope this next variable gets optimized
	// away with return-value optimization (and that the same register is used
	// for everything in the case of
	//
	//   x = reduce<ReductionOp>(x);
	//
	for (int shuffle_mask = warp_size/2; shuffle_mask > 0; shuffle_mask >>= 1)
		op_acc(value, shuffle_xor(value, shuffle_mask));
	return value;
}

template <typename Datum>
KAT_FD Datum sum(Datum value)
{
	return reduce<detail::plus<Datum>>(value);
}

/**
 * Performs a reduction (e.g. a summation of multiplication) of all elements passed into
 * the function by the threads of a block - but with each thread ending up with the reduction
 * result for all threads upto itself.
 *
 * @note What about inclusivity?
 *
 * @todo offer both an inclusive and an exclusive versionn
 */
template <
	typename ReductionOp,
	typename InputDatum,
	inclusivity_t Inclusivity = inclusivity_t::Inclusive>
KAT_FD typename ReductionOp::result_type scan(InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	typename ReductionOp::accumulator acc_op;

	result_type x;

	if (Inclusivity == inclusivity_t::Exclusive) {
		InputDatum preshuffled = shuffle_up(value, 1);
		// ... and now we can pretend to be doing an inclusive shuffle
		x = lane::is_first() ? op.neutral_value() : preshuffled;
	}
	else { x = value; }

	// x of lane i holds the reduction of values of
	// the lanes i - 2*(offset) ... i - offset , and initially offset = 0
	#pragma unroll
	for (int offset = 1; offset < warp_size; offset <<= 1) {
		result_type shuffled = shuffle_up(x, offset);
		if(lane::index() >= offset) { acc_op(x, shuffled); }
	}
	return x;
}

// TODO: Need to implement a scan-and-reduce warp primitive

template <typename T, inclusivity_t Inclusivity = inclusivity_t::Inclusive, typename Result = T>
KAT_FD T prefix_sum(T value)
{
	return scan<detail::plus<Result>, T, Inclusivity>(value);
}

template <typename T, typename Result = T>
KAT_FD T exclusive_prefix_sum(T value)
{
	return prefix_sum<T, inclusivity_t::Exclusive, Result>(value);
}

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
template <typename T, typename U, typename Size>
KAT_FD void cast_and_copy_n(
	T*        __restrict__  target,
	const U*  __restrict__  source,
	Size                    length)
{
	using namespace linear_grid::grid_info;
	// sometimes this next loop can be unrolled (when length is known
	// at compile time; since the function is inlined)
	#pragma unroll
	for(promoted_size_t<Size> pos = lane::index(); pos < length; pos += warp_size) {
		target[pos] = source[pos];
	}
}

template <typename T, typename U>
KAT_FD void cast_and_copy(
	T*        __restrict__  target,
	const U*  __restrict__  source_start,
	const U*  __restrict__  source_end)
{
	auto length = source_end - source_start;
	return cast_and_copy_n<T, U, decltype(length)>(target, source_start, source_end - source_start);
}
/*
template <typename T>
KAT_FD void single_write(T* __restrict__  target, T&& x)
{
	target[lane::index()] = std::forward(x);
}
*/

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
	T*        __restrict__  target,
	const T*  __restrict__  source,
	Size                    length)
{
	#pragma unroll
	for(promoted_size_t<Size> pos = lane::index(); pos < length; pos += warp_size)
	{
		target[pos] = source[pos];
	}
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
	T*        __restrict__  target,
	const T*  __restrict__  source,
	Size                    length)
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
		detail::naive_copy(target, source, length);
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

template <typename T, bool MayHaveSlack = true>
KAT_FD void copy(
	const T*  __restrict__  source_start,
	const T*  __restrict__  source_end,
	T*        __restrict__  target_start)
{
	auto length = source_end - source_start;
	return copy_n<T, sizeof(length), MayHaveSlack>(target_start, source_start, length);
}

// TODO: Check whether writing this with a forward iterator and std::advance
// yields the same PTX code (in which case we'll prefer that)
template <typename RandomAccessIterator, typename Size, typename T>
inline KAT_DEV void fill_n(RandomAccessIterator start, Size count, const T& value)
{
	T tmp = value;
	for(promoted_size_t<Size> index = lane::index();
		index < count;
		index += warp_size)
	{
		start[index] = tmp;
	}
}

template <typename ForwardIterator, typename T>
inline KAT_DEV void fill(ForwardIterator start, ForwardIterator end, const T& value)
{
    const T tmp = value;
	auto iter = start + lane::index();
    for (; iter < end; iter += warp_size)
    {
		*iter = tmp;
    }
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
	using namespace linear_grid::grid_info;
	#pragma unroll
	for(promoted_size_t<Size> pos = lane::index(); pos < num_indices; pos += warp_size) {
		target[pos] = lookup_table[indices[pos]];
	}
}

/**
 * @note If you call this for multiple warps and the same destination,
 * you'd better use an atomic accumulator op...
 */

template <typename D, typename S, typename AccumulatingOperation, typename Size>
KAT_FD void elementwise_accumulate(
	D*       __restrict__  destination,
	const S* __restrict__  source,
	Size                   length)
{
	AccumulatingOperation op;
	for(promoted_size_t<Size> pos = lane::index(); pos < length; pos += warp_size) {
		op(destination[pos], source[pos]);
	}
}

/**
 * A variant of elementwise_accumulate for when the length <= warp_size,
 * in which case each thread will get just one source element (possibly
 * a junk or default-constructed element) to work with
 *
 * @note If you call this for multiple warps and the same destination,
 * you'd better use an atomic accumulator op...
 */
template <typename D, typename S, typename AccumulatingOperation, typename Size>
KAT_FD void elementwise_accumulate(
	D*       __restrict__  destination,
	const S&               source_element,
	Size                   length)
{
	AccumulatingOperation op;
	if (lane::index() < length) {
		op(destination[lane::index()], source_element);
	}
}

} // namespace warp
} // namespace collaborative
} // namespace kat

#endif // CUDA_KAT_WARP_COLLABORATIVE_SEQUENCE_OPS_CUH_
