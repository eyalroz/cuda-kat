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
#include <kat/on_device/sequence_ops/warp.cuh>
#include <kat/on_device/collaboration/block.cuh>
#include <kat/on_device/sequence_ops/warp.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {
namespace linear_grid {
namespace collaborative {

using kat::collaborative::inclusivity_t;

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
KAT_FD void fill_n(RandomAccessIterator start, Size count, T value)
{
	auto f = [=](promoted_size_t<Size> pos) {
		start[pos] = value;
	};
	at_block_stride(count, f);
}

template <typename RandomAccessIterator, typename T>
KAT_FD void fill(RandomAccessIterator start, RandomAccessIterator end, const T& value)
{
    auto count = end - start;
    return fill_n(start, count, value);
}

template <typename RandomAccessIterator, typename Size>
KAT_FD void memzero_n(RandomAccessIterator start, Size count)
{
    return fill_n(start, count, 0);
}

template <typename RandomAccessIterator, typename Size>
KAT_FD void memzero(RandomAccessIterator start, RandomAccessIterator end)
{
	auto count = end - start;
    return fill_n(start, count, 0);
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
	T*        __restrict__  target,
	Size                    length,
	UnaryOperation          op)
{
	auto f = [=](promoted_size_t<Size> pos) {
		target[pos] = op(source[pos]);
	};
	at_block_stride(length, f);
}

/**
 * @note Prefer `copy_n()`; this will force the size to `ptrdiff_t`, which unnecessarily large.
 */
template <typename S, typename T, typename UnaryOperation, typename Size>
KAT_FD void transform(
	const S*  __restrict__  source_start,
	const S*  __restrict__  source_end,
	T*        __restrict__  target,
	UnaryOperation          op)
{
	auto length = source_end - source_start;
	return transform_n(source_start, target, op, length);
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
	T*        __restrict__  target,
	Size                    length)
{
	auto op = [](S x) { return T{x};} ;
	return transform_n(source, target, length, op);
}

template <typename S, typename T, typename Size>
KAT_FD void cast_and_copy_n(
	const S*  __restrict__  source_start,
	const S*  __restrict__  source_end,
	T*        __restrict__  target)
{
	auto length = source_end - source_start;
	return cast_and_copy_n(source_start, target, length);
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
	T*        __restrict__  target,
	Size                    length)
{
	auto f = [=](promoted_size_t<Size> pos) {
		target[pos] = source[pos];
	};
	at_block_stride(length, f);
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
template <typename T>
KAT_FD void copy(
	const T*  __restrict__  source_start,
	const T*  __restrict__  source_end,
	T*        __restrict__  target)
{
	auto length = source_end - source_start;
	return copy_n(source_start, target, length);
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
	auto f = [=](promoted_size_t<Size> pos) {
		target[pos] = lookup_table[indices[pos]];
	};
	at_block_stride(num_indices, f);
}


// TODO: Consider replacing the following with a functor on GSL-style array spans


/**
 * @brief Perform a reduction over a block's worth of data with a specific
 * binary reduction operation (e.g. sum, or maximum, etc.).
 *
 * @param value each thread's contribution to the reduction
 * @return for the first thread of the warp - the reduction result over
 * all @p value elements of all block threads; for other threads - the
 * result is undefined
 *
 * @note all threads must participate in this primitive; consider
 * supporting partial participation
 *
 */
template<
	typename ReductionOp,
	typename InputDatum,
	bool AllThreadsObtainResult = false>
KAT_DEV typename ReductionOp::result_type reduce(InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	static __shared__ result_type warp_reductions[warp_size];

	result_type intra_warp_result = kat::collaborative::warp::reduce<ReductionOp>(static_cast<result_type>(value));
	kat::linear_grid::collaborative::block::share_warp_datum_with_whole_block(intra_warp_result, warp_reductions);

	// Note: assuming here that there are at most 32 warps per block;
	// if/when this changes, more warps may need to be involved in this second
	// phase

	if (!AllThreadsObtainResult) {
		// We currently only guarantee the first thread has the final result,
		// which is what allows most threads to return already:
		if (!linear_grid::grid_info::warp::is_first_in_block()) { return op.neutral_value(); }
	}

	__syncthreads(); // Perhaps a block fence is enough here?

	// shared memory now holds all intra-warp reduction results

	// read from shared memory only if that warp actually existed
	result_type other_warp_result  =
		(grid_info::lane::index() < linear_grid::grid_info::block::num_warps()) ?
		warp_reductions[grid_info::lane::index()] : op.neutral_value();

	return kat::collaborative::warp::reduce<ReductionOp>(other_warp_result);
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
template<
	typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive>
 KAT_DEV typename ReductionOp::result_type scan(
	 typename ReductionOp::result_type* __restrict__ scratch, // must have warp_size element allocated
	 InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;

	result_type intra_warp_inclusive_scan_result =
		kat::collaborative::warp::scan<ReductionOp, InputDatum, inclusivity_t::Inclusive>(static_cast<result_type>(value));

	collaborative::block::share_warp_datum_with_whole_block(
		intra_warp_inclusive_scan_result, scratch, grid_info::warp::last_lane);
		// Note: if the block is not made up of full warps, this will fail,
		// since the last warp will not have a lane to do the writing

	__syncthreads();

	// shared memory now holds all full-warp _reductions_;

	if (warp::is_first_in_block()) {
		// Note that for a block with less than warp_size warps, some of the lanes
		// here will read junk data from the scratch area; but that's not a problem,
		// since these values will not effect the scan results of previous lanes,
		// and hence not affect any of the existing warps later on when they rely
		// on what the first warp computes here.
		auto warp_reductions_scan_result =
			kat::collaborative::warp::scan<ReductionOp, result_type, inclusivity_t::Exclusive>(
				scratch[lane::index()]);
		scratch[lane::index()] = warp_reductions_scan_result;
	}
	__syncthreads();


	auto inclusive_reduction_result_upto_previous_warp = scratch[warp::index()];
	result_type intra_warp_scan_result;
	if (Inclusivity == inclusivity_t::Inclusive) {
		intra_warp_scan_result = intra_warp_inclusive_scan_result;
	}
	else {
		result_type shuffled = shuffle_up(intra_warp_inclusive_scan_result, 1);
		intra_warp_scan_result =
			lane::is_first() ? op.neutral_value() : shuffled;
	}
	return op(inclusive_reduction_result_upto_previous_warp, intra_warp_scan_result);
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
template<
	typename ReductionOp,
	typename InputDatum,
	bool Inclusivity = inclusivity_t::Inclusive>
 KAT_DEV void scan_and_reduce(
	 typename ReductionOp::result_type* __restrict__ scratch, // must have warp_size element allocated
	 InputDatum                                      value,
	 typename ReductionOp::result_type&              scan_result,
	 typename ReductionOp::result_type&              reduction_result)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	typename ReductionOp::accumulator acc_op;

	result_type intra_warp_inclusive_scan_result =
		kat::collaborative::warp::scan<ReductionOp, InputDatum, inclusivity_t::Inclusive>(
			static_cast<result_type>(value));

	collaborative::block::share_warp_datum_with_whole_block(
		intra_warp_inclusive_scan_result, scratch, grid_info::warp::last_lane);
		// Note: if the block is not made up of full warps, this will fail,
		// since the last warp will not have a lane to do the writing


	// scratch[i] now contains the reduction result of the data of all threads in
	// the i'th warp of this block

	auto num_warps = block::num_warps();
	reduction_result = scratch[num_warps - 1];

	if (warp::is_first_in_block()) {
		// Note that for a block with less than warp_size warps, some of the lanes
		// here will read junk data from the scratch area; but that's not a problem,
		// since these values will not effect the scan results of previous lanes,
		// and hence not affect any of the existing warps later on when they rely
		// on what the first warp computes here.
		auto warp_reductions_scan_result =
			kat::collaborative::warp::scan<ReductionOp, result_type, inclusivity_t::Exclusive>(
				scratch[lane::index()]);
		scratch[lane::index()] = warp_reductions_scan_result;
	}
	__syncthreads();

	// scratch[i] now contains the reduction result of the data of all threads in
	// warps 0 ... i-1 of this block

	acc_op(reduction_result, scratch[num_warps - 1]);

	auto inclusive_reduction_result_upto_previous_warp = scratch[warp::index()];

	// To finalize the computation, we now account for the requested scan inclusivity

	result_type intra_warp_scan_result;
	if (Inclusivity == inclusivity_t::Inclusive) {
		intra_warp_scan_result = intra_warp_inclusive_scan_result;
	}
	else {
		result_type shuffled = shuffle_up(intra_warp_inclusive_scan_result, 1);
		intra_warp_scan_result =
			lane::is_first() ? op.neutral_value() : shuffled;
	}
	scan_result = op(inclusive_reduction_result_upto_previous_warp, intra_warp_scan_result);
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
template <typename D, typename S, typename AccumulatingOperation, typename Size>
KAT_FD void elementwise_accumulate(
	D*       __restrict__  destination,
	const S* __restrict__  source,
	Size                   length)
{
	AccumulatingOperation op;
	for(promoted_size_t<Size> pos = thread::index(); pos < length; pos += block::size()) {
		op(destination[pos], source[pos]);
	}
}

template <typename Operation, typename Size, typename ResultDatum, typename... Args>
KAT_FD void elementwise_apply(
	ResultDatum*     __restrict__  results,
	Size                           length,
	Operation                      op,
	const Args* __restrict__ ...   arguments)
{
	auto f = [&](promoted_size_t<Size> pos) {
		return op(arguments[pos]...);
	};
	at_block_stride(length, f);
}


} // namespace block
} // namespace collaborative
} // namespace linear_grid
} // namespace kat

#endif // CUDA_KAT_BLOCK_COLLABORATIVE_SEQUENCE_OPS_CUH_
