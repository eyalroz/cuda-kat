/**
 * @file on_device/sequence_ops/block.cuh
 *
 * @brief GPU device-side versions of `std::algorithm`-like functions, with
 * block-level collaboration, i.e. different CUDA blocks act independently, but
 * all lanes in each warp collaborate on the same task.
 *
 * @note Most functions actually in `std::algorithm` are still missing;
 * see the <a href="https://en.cppreference.com/w/cpp/algorithm">`algorithm` page</a>
 * on cppreference.com for a full list of those.
 *
 * @note some functions here are not actually in `std::algorithm` but might as
 * well have been, e.g. `memzero()` which is like `std::memset()` with 0.
 */

#pragma once
#ifndef CUDA_KAT_BLOCK_COLLABORATIVE_SEQUENCE_OPS_CUH_
#define CUDA_KAT_BLOCK_COLLABORATIVE_SEQUENCE_OPS_CUH_

#include "common.cuh"
#include <kat/on_device/collaboration/warp.cuh>
#include <kat/on_device/collaboration/block.cuh>
#include <kat/on_device/sequence_ops/warp.cuh>
#include <kat/on_device/shuffle.cuh>
#include <kat/on_device/ranges.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace linear_grid {

using kat::inclusivity_t;

namespace block {

/*
 * TODO: Implement
 * KAT_FD  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * KAT_FD  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

// TODO: Currently kind-of assuming linear grids in several places in this file

// TODO: Check whether writing this with a forward iterator and std::advance
// yields the same PTX code (in which case we'll prefer that)
template <typename RandomAccessIterator, typename Size, typename T>
KAT_FD void fill_n(RandomAccessIterator start, Size count, const T& value)
{
	for(auto pos : ranges::block_stride(count)) {
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
	for(auto pos : ranges::block_stride(length)) {
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

/**
 * Have all warp threads collaborate in copying
 * data between two memory locations (possibly not in the same memory
 * space), while also converting types.
 *
 * @param target The (block-common) destination into which to write the converted elements
 * @param source The (block-common) origin of the data
 * @param length The (block-common) number of elements available (for reading?] at the
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

template <typename S, typename T, typename Size = std::ptrdiff_t>
KAT_FD void cast_and_copy(
	const S*  __restrict__  source_start,
	const S*  __restrict__  source_end,
	T*        __restrict__  target)
{
	Size length = source_end - source_start;
	return cast_and_copy_n(source_start, length, target);
}


/**
 * @brief block-collaboratively copy data between stretches of memory
 *
 * @param source (block-common) location from which to copy data
 * @param target (block-common) location into which to copy the first element
 * @param length number of elements at @p source to copy
 */
template <typename T, typename Size>
KAT_FD void copy_n(
	const T*  __restrict__  source,
	Size                    length,
	T*        __restrict__  target)
{
	for(auto pos : ranges::block_stride(length)) {
		target[pos] = source[pos];
	};
}

/**
 * @brief block-collaboratively copy data between stretches of memory
 *
 * @param source_start (block-common) location of the first data element to copy
 * @param source_end (block-common) location past the last data element to copy
 * @param target (block-common) location into which to copy the first element
 *
 * @note Prefer `copy_n()`; this will force the size to `ptrdiff_t`, which unnecessarily large.
 */
template <typename T, typename Size = std::ptrdiff_t>
KAT_FD void copy(
	const T*  __restrict__  source_start,
	const T*  __restrict__  source_end,
	T*        __restrict__  target)
{
	Size length = source_end - source_start;
	return copy_n(source_start, length, target);
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
	for(auto pos : ranges::block_stride(num_indices)) {
		target[pos] = lookup_table[indices[pos]];
	};
}


// TODO: Consider replacing the following with a functor on GSL-style array spans

namespace detail {

template<class Op> struct accumulator_op_return_type_helper : accumulator_op_return_type_helper<decltype(&Op::operator())> {};
template<class Op> struct accumulator_op_return_type_helper<Op(Op&)> { using type = Op; };
template<class Op> struct accumulator_op_return_type_helper<Op(Op&) const> { using type = Op; };
template<class Op> struct accumulator_op_return_type_helper<Op(*)(Op&)> { using type = Op; };
template<class C, class M> struct accumulator_op_return_type_helper<M (C::*)> : accumulator_op_return_type_helper<M> {};

template <typename Op>
using accumulator_op_return_type_t = typename accumulator_op_return_type_helper<Op>::type;

}

/**
 * @brief Perform a reduction over a block's worth of data with a specific
 * (asymmetric) accumulation operation, and maintaing the input element type.
 *
 * @param value each thread's contribution to the reduction
 * @param op the accumulation operator - it must have the appropriate `operator()`,
 * i.e. with signature `T AccumulationOp::operator()(T&, T)`. It does not have
 * to have any other members or types defined (so a lambda works fine).
 *
 * @return for threads in the first warp of the block - the reduction result over
 * all @p value elements of all block threads; for other threads - the
 * result is undefined, in case @tparam AllThreadsObtainResult is false,
 * or like the first warp if AllThreadsObtainResult is true
 *
 * @note This _should_ work without full block participation, but it
 * does need full warp participation, i.e. each warp either participates fully
 * or not at all.
 *
 * @note One might wonder: "Why insist on the same type for the result and the
 * input?" - well, that is not necessary. However, separating the types would
 * require additional template or parameter information: Two operators (if not
 * more), and a decision at what point we switch to the result type - immediately,
 * after at most k operations, above the warp level. This also makes it
 * nearly impossible to write "simple" calls to reduce - with a value and
 * a single lambda. We may at some point define a structure for setting these
 * parameters, which will put some onus on the user code, but allow for
 * this flexibility. Poke the library author/contributors about this.
 *
 * @tparam AllThreadsObtainResult when true, all threads in a block will
 * return the reduction result; otherwise, only the first warp of the block
 * is guaranteed to return the actual reduction result.
 */
template<
	typename T,
	typename AccumulationOp,
	bool AllThreadsObtainResult = false>
KAT_DEV T reduce(T value, AccumulationOp op)
{
	namespace gi = kat::linear_grid;
	constexpr const T neutral_value {};

	static __shared__ T warp_reductions[warp_size];

	auto intra_warp_result = kat::warp::reduce<T, AccumulationOp>(value, op);

	block::share_per_warp_data(intra_warp_result, warp_reductions, kat::warp::first_lane);

	// Note: assuming here that there are at most 32 warps per block;
	// if/when this changes, more warps may need to be involved in this second
	// phase

	if (not AllThreadsObtainResult) {
		// We currently only guarantee the first thread has the final result,
		// which is what allows most threads to return already:
		if (not kat::warp::is_first_in_block()) { return neutral_value; }
	}

	block::barrier(); // Perhaps we can do with something weaker here?

	// shared memory now holds all intra-warp reduction results

	// read from shared memory only if that warp actually existed
	auto other_warp_result  = (lane::id() < gi::block::num_warps()) ?
		warp_reductions[lane::id()] : neutral_value;

	return kat::warp::reduce<T, AccumulationOp>(other_warp_result, op);
		// TODO: Would it perhaps be faster to have only one warp compute this,
		// and then use get_from_first_thread() ?
}

template<
	typename T,
	bool AllThreadsObtainResult = false>
KAT_DEV T sum(T value)
{
	auto plus = [](T& x, T y) { x += y; };
	return reduce<T, decltype(plus), AllThreadsObtainResult>(value, plus);
}

/**
 *
 * @note Supports only full-warps, and you should probably have
 * the entire block participate.
 *
 * @param scratch
 * @param value
 * @return
 */
template <
	typename T,
	typename AccumulationOp,
	bool Inclusivity = inclusivity_t::Inclusive
>
KAT_DEV T scan(T value, AccumulationOp op, T* __restrict__ scratch)
{
	constexpr const T neutral_value {};
	auto intra_warp_inclusive_scan_result =	kat::warp::scan<
		T, AccumulationOp, inclusivity_t::Inclusive >(value, op);

	auto last_active_lane_id =
		// (AssumeFullWarps or not warp::is_last_in_block()) ?
		kat::warp::last_lane
		// : warp::last_active_lane_index()
		;

	// Note: At the moment, we assume the block is not made up of full warps,
	// as otherwise the last active lane may not be the last one - so no lane
	// will write to shared memory. However, the assumptions is actually earlier,
	// since our warp scan also assumes the participation of the full warp.

	block::share_per_warp_data(
		intra_warp_inclusive_scan_result, scratch, last_active_lane_id);
		// The last active lane writes, because only it has the whole warp's reduction value

	block::barrier();

	// scratch buffer now holds all full-warp _reductions_;

	if (kat::warp::is_first_in_block()) {
		// Note that for a block with less than warp_size warps, some of the lanes
		// here will read junk data from the scratch area; but that's not a problem,
		// since these values will not effect the scan results of previous lanes,
		// and hence not affect any of the existing warps later on when they rely
		// on what the first warp computes here.
		auto warp_reductions_scan_result =
			kat::warp::scan<T, AccumulationOp, inclusivity_t::Exclusive>(
				scratch[lane::id()], op);
		scratch[lane::id()] = warp_reductions_scan_result;
	}

	block::barrier();

	auto r = scratch[kat::warp::id()];
	T intra_warp_scan_result;
	if (Inclusivity == inclusivity_t::Inclusive) {
		intra_warp_scan_result = intra_warp_inclusive_scan_result;
	}
	else {
		auto shuffled = shuffle_up(intra_warp_inclusive_scan_result, 1);
		intra_warp_scan_result = lane::is_first() ? neutral_value : shuffled;
	}
	op(r, intra_warp_scan_result);
	return r;
}

template <
	typename T,
	typename AccumulationOp,
	bool Inclusivity = inclusivity_t::Inclusive
>
KAT_DEV T scan(T value, AccumulationOp op)
{
	// Note the assumption there can no than warp_size warps per block
	static __shared__ T scratch[warp_size];
	return scan<T, AccumulationOp, Inclusivity>(value, op, scratch);
}


/**
 * Perform both a block-level scan and a block-level reduction,
 * with each thread having the results of both.
 *
 * @note implementation relies on the details of the implementation
 * of the scan primitive, above.
 *
 * @todo consider returning a pair rather than using non-const references
 * @todo lots of code duplication with just-scan
 * @todo add a bool template param allowing the code to assume the block is
 * full (this saves a few ops)
 *
 *
 * @param scratch An area of memory in which this primitive can use for
 * inter-warp communication (as warps cannot communicate directly). It
 * must have at least warp_size elements allocated
 * (i.e. sizeof(ReductionOp::result_type)*warp_size bytes
 * @param value Each thread provides its input value, and the scan is applied
 * to them all as though they were in some input array
 * @param scan_result the result of applying a scan to all threads' input values,
 * in order of the thread indices
 * @param reduction_result the result of reducing all threads' input values
 */
template <
	typename T,
	typename AccumulationOp,
	bool Inclusivity = inclusivity_t::Inclusive
>
 KAT_DEV void scan_and_reduce(
	 T* __restrict__ scratch, // must have as many elements as there are warps
	 T               value,
	 AccumulationOp  op,
	 T&              scan_result,
	 T&              reduction_result)
{
	constexpr const T neutral_value {};

	auto intra_warp_inclusive_scan_result = kat::warp::scan<
		T, AccumulationOp, inclusivity_t::Inclusive>(value, op);

	auto last_active_lane_id =
		// (AssumeFullWarps or not warp::is_last_in_block()) ?
		kat::warp::last_lane
		// : warp::last_active_lane_index()
		;

	// Note: At the moment, we assume the block is not made up of full warps,
	// as otherwise the last active lane may not be the last one - so no lane
	// will write to shared memory. However, the assumptions is actually earlier,
	// since our warp scan also assumes the participation of the full warp.

	block::share_per_warp_data(
		intra_warp_inclusive_scan_result, scratch, last_active_lane_id);
		// The last active lane writes, because only it has the whole warp's reduction value

	// scratch[i] now contains the reduction result of the data of all threads in
	// the i'th warp of this block

	auto num_warps = kat::block::num_warps();
	auto partial_reduction_result = scratch[num_warps - 1];
		// We're keeping this single-warp reduction result, since it will soon
		// be overwritten

	if (kat::warp::is_first_in_block()) {
		// Note that for a block with less than warp_size warps, some of the lanes
		// here will read junk data from the scratch area; but that's not a problem,
		// since these values will not effect the scan results of previous lanes,
		// and hence not affect any of the existing warps later on when they rely
		// on what the first warp computes here.
		auto other_warp_reduction_result = scratch[lane::id()];
		auto warp_reductions_scan_result = kat::warp::scan<
			T, AccumulationOp, inclusivity_t::Exclusive>(
				other_warp_reduction_result, op);
		scratch[lane::id()] = warp_reductions_scan_result;
	}

	block::barrier();

	// scratch[i] now contains the reduction result of the data of all threads in
	// warps 0 ... i-1 of this block

	op(partial_reduction_result, scratch[num_warps - 1]);
		// We had kept the last warp's reduction result, now we've taken
		// the other warps into account as well

	auto partial_scan_result = scratch[kat::warp::id()]; // only a partial result for now

	// To finalize the computation, we now account for the requested scan inclusivity

	T intra_warp_scan_result;
	if (Inclusivity == inclusivity_t::Inclusive) {
		intra_warp_scan_result = intra_warp_inclusive_scan_result;
	}
	else {
		// Note: We don't have a de-accumulation operator.
		// TODO: Version of this function taking a de-accumulation operator
		// which avoid this shuffle
		T shuffled = shuffle_up(intra_warp_inclusive_scan_result, 1);
		intra_warp_scan_result = lane::is_first() ? neutral_value : shuffled;
	}
	op(partial_scan_result, intra_warp_scan_result);

	reduction_result = partial_reduction_result;
	scan_result = partial_scan_result;
}

template <
	typename T,
	typename AccumulationOp,
	bool Inclusivity = inclusivity_t::Inclusive
>
 KAT_DEV void scan_and_reduce(
	 T               value,
	 AccumulationOp  op,
	 T&              scan_result,
	 T&              reduction_result)
{
	// Note the assumption there can no than warp_size warps per block
	static __shared__ T scratch[warp_size];
	scan_and_reduce<T, AccumulationOp, Inclusivity>(
		scratch, value, op, scan_result, reduction_result);
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
	for(auto pos : ranges::block_stride(length)) {
		op(destination[pos], source[pos]);
	}
}

template <typename D, typename RandomAccessIterator, typename AccumulatingOperation, typename Size = std::ptrdiff_t>
KAT_FD void elementwise_accumulate(
	AccumulatingOperation              op,
	D*                   __restrict__  destination,
	RandomAccessIterator __restrict__  source_start,
	RandomAccessIterator __restrict__  source_end)
{
	elementwise_accumulate_n(op, destination, source_start, source_end - source_start);
}

template <typename Operation, typename Size, typename ResultDatum, typename... Args>
KAT_FD void elementwise_apply(
	ResultDatum*     __restrict__  results,
	Size                           length,
	Operation                      op,
	const Args* __restrict__ ...   arguments)
{
	for(auto pos : ranges::block_stride(length)) {
		return op(arguments[pos]...);
	};
}


} // namespace block
} // namespace linear_grid
} // namespace kat

#endif // CUDA_KAT_BLOCK_COLLABORATIVE_SEQUENCE_OPS_CUH_
