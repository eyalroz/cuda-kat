/**
 * @file on_device/primitives/block.cuh
 *
 * @brief CUDA device computation warp-level primitives, i.e. those involving
 * interaction of many/all of each blocks's lanes, but _no_ inter-block
 * interaction.
 *
 * @todo Some of these assume linear grids, others do not - sort them out
 *
 */

#pragma once
#ifndef BLOCK_LEVEL_PRIMITIVES_CUH_
#define BLOCK_LEVEL_PRIMITIVES_CUH_

#include <kat/on_device/primitives/common.cuh>
#include <kat/on_device/primitives/warp.cuh>
#include <kat/on_device/miscellany.cuh>
#include <kat/on_device/shared_memory.cuh>

#include <kat/define_specifiers.hpp>

namespace kat {

namespace primitives {

namespace block {

// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::index()` without prefixing that with anything.

namespace grid   = ::grid_info::linear::grid;
namespace block  = ::grid_info::linear::block;
namespace warp   = ::grid_info::linear::warp;
namespace thread = ::grid_info::linear::thread;
namespace lane   = ::grid_info::linear::lane;

/*
 * TODO: Implement
 * __fd__  unsigned all_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned none_satisfy(unsigned int predicate, unsigned* scratch_area);
 * __fd__  unsigned some_satisfy(unsigned int predicate, unsigned* scratch_area);
 *
 * at the block level
 *
 */

// TODO: Currently kind-of assuming linear grids in several places in this file

// TODO: Check whether writing this with a forward iterator and std::advance
// yields the same PTX code (in which case we'll prefer that)
template <typename RandomAccessIterator, typename Size, typename T>
__fd__ void fill_n(RandomAccessIterator start, Size count, const T& value)
{
	T tmp = value;
	for(promoted_size_t<Size> index = thread::index();
		index < count;
		index += block::size())
	{
		start[index] = tmp;
	}
}

template <typename ForwardIterator, typename T>
__fd__ void fill(ForwardIterator start, ForwardIterator end, const T& value)
{
    const T tmp = value;
	auto iter = start + thread::index();
    for (; iter < end; iter += block::length())
    {
    	*iter = tmp;
    }
}

template <typename ForwardIterator, typename Size>
__fd__ void memzero(ForwardIterator start, Size count)
{
    return fill_n(start, count, 0);
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
__fd__ void cast_and_copy(
	T*        __restrict__  target,
	const U*  __restrict__  source,
	Size                    length)
{
	using namespace grid_info::linear;
	#pragma unroll
	for(promoted_size_t<Size> pos = thread::index_in_block(); pos < length; pos += block::size()) {
		target[pos] = source[pos];
	}
}

/**
 * Same as {@ref cast_copy}, except that no casting is done
 *
template <typename T, typename Size>
__fd__ void copy(
	T*        __restrict__  target,
	const T*  __restrict__  source,
	Size                    length)
{
	return cast_and_copy<T, T, Size>(target, source, length);
}
*/

/**
 * Same as {@ref cast_copy}, except that no casting is done
 */
template <typename T, typename Size>
__fd__ void copy_n(
	T*        __restrict__  target,
	const T*  __restrict__  source,
	Size                    length)
{
	return cast_and_copy<T, T, Size>(target, source, length);
}



/**
 * Use a lookup table to convert numeric indices to a sequence
 * of values of any type
 */
template <typename T, typename I, typename Size, typename U = T>
__fd__ void lookup(
	T*       __restrict__  target,
	const U* __restrict__  lookup_table,
	const I* __restrict__  indices,
	Size                   num_indices)
{
	auto f = [&](promoted_size_t<Size> pos) {
		target[pos] = lookup_table[indices[pos]];
	};
	at_block_stride(num_indices, f);
}


// TODO: Consider replacing the following with a functor on GSL-style array spans

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
 * @param [inout] destination The array into which we accumulate; holds existing data
 * and is not simply overwritten.
 * @param [in] source The array of partial data to integrate via accumulation.
 * @param [in] num_elements the length in elements of {@ref destination} and {@ref source}
 */
template <typename D, typename S, typename AccumulatingOperation, typename Size>
__fd__ void elementwise_accumulate(
	D*       __restrict__  destination,
	const S* __restrict__  source,
	Size                   length)
{
	AccumulatingOperation op;
	for(int pos = thread::index(); pos < length; pos += block::size()) {
		op(destination[pos], source[pos]);
	}
}

// TODO: Make this a variadic template
template <typename ResultDatum, typename LHSDatum, typename RHSDatum, typename Operation, typename Size>
__fd__ void elementwise_apply(
	ResultDatum*     __restrict__  results,
	const LHSDatum*  __restrict__  lhs,
	const RHSDatum*  __restrict__  rhs,
	Size                           length)
{
	using namespace grid_info::linear;
	Operation op;
	for(promoted_size_t<Size> pos = thread::index(); pos < length; pos += block::size()) {
		results[pos] = op(lhs[pos], rhs[pos]);
	}
}


/**
 * Have all threads in (one/some/all) blocks perform some action over the
 * linear range of 0..length-1 - the same range for each block.
 *
 * @param length The length of the range (of integers) on which to act
 * handle (serially)
 * @param f The callable to execute for each element of the sequence.
 */
template <typename Function, typename Size = size_t>
__fd__ void at_block_stride(Size length, const Function& f)
{
	#pragma unroll
	for(promoted_size_t<Size> pos = thread::index_in_block();
		pos < length;
		pos += block::length())
	{
		f(pos);
	}
}

// TODO: Perhaps make this a "warp to block" primitive?
/**
 * Makes one element of type T for each warp, which was previously
 * only visible to that warp (or rather not known to be otherwise) - visible to,
 * shared with, the entire block, via shared memory.
 *
 * @param datum a warp-specific (but not thread-specific) piece of data,
 * one for each warp, which is to be shared with the whole block
 * @param where_to_make_available the various warp-specific data will be
 * stored here by warp index
 * @param writing_lane_index which lane in each warp should perform write operations
 */
template <typename T, bool Synchronize = true>
__fd__ void share_warp_datum_with_whole_block(
	const T& datum,
	T* __restrict__ where_to_make_available,
	unsigned writing_lane_index = 0)
{
	if (lane::index() == writing_lane_index) {
		where_to_make_available[warp::index()] = datum;
	}
	if (Synchronize) __syncthreads();
}

/**
 * Perform a reduction over a block's worth of data with a specific
 * binary reduction operation (e.g. sum, or maximum, etc.)
 *
 * @note all threads must participate in this primitive; consider
 * supporting partial participation
 *
 *
 * @param value each thread's contribution to the reduction
 * @return for the first thread of the warp - the reduction result over
 * all @ref value elements of all block threads; for other threads - the
 * result is undefined
 */
template<
	typename ReductionOp,
	typename InputDatum,
	bool AllThreadsObtainResult = false>
 __inline__ __device__ typename ReductionOp::result_type reduce(InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	static __shared__ result_type warp_reductions[warp_size];

	result_type intra_warp_result = primitives::warp::reduce<ReductionOp>(static_cast<result_type>(value));
	primitives::block::share_warp_datum_with_whole_block(intra_warp_result, warp_reductions);

	// Note: assuming here that there are at most 32 warps per block;
	// if/when this changes, more warps may need to be involved in this second
	// phase

	if (!AllThreadsObtainResult) {
		// We currently only guarantee the first thread has the final result,
		// which is what allows most threads to return already:
		if (!warp::is_first_in_block()) { return op.neutral_value(); }
	}

	__syncthreads(); // Perhaps a block fence is enough here?

	// shared memory now holds all intra-warp reduction results

	// read from shared memory only if that warp actually existed
	result_type other_warp_result  =
		(lane::index() < block::num_warps()) ?
		warp_reductions[lane::index()] : op.neutral_value();

	return primitives::warp::reduce<ReductionOp>(other_warp_result);
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
 __inline__ __device__ typename ReductionOp::result_type scan(
	 typename ReductionOp::result_type* __restrict__ scratch, // must have warp_size element allocated
	 InputDatum value)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;

	result_type intra_warp_inclusive_scan_result =
		primitives::warp::scan<ReductionOp, InputDatum, inclusivity_t::Inclusive>(static_cast<result_type>(value));

	primitives::block::share_warp_datum_with_whole_block(
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
			primitives::warp::scan<ReductionOp, result_type, inclusivity_t::Exclusive>(
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
 __inline__ __device__ void scan_and_reduce(
	 typename ReductionOp::result_type* __restrict__ scratch, // must have warp_size element allocated
	 InputDatum                                      value,
	 typename ReductionOp::result_type&              scan_result,
	 typename ReductionOp::result_type&              reduction_result)
{
	using result_type = typename ReductionOp::result_type;
	ReductionOp op;
	typename ReductionOp::accumulator acc_op;

	result_type intra_warp_inclusive_scan_result =
		primitives::warp::scan<ReductionOp, InputDatum, inclusivity_t::Inclusive>(
			static_cast<result_type>(value));

	primitives::block::share_warp_datum_with_whole_block(
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
			primitives::warp::scan<ReductionOp, result_type, inclusivity_t::Exclusive>(
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

__fd__ void barrier() { __syncthreads(); }


// A bit ugly, but...
// Note: Make sure the first block thread has not diverged/exited
#define once_per_block if (thread::is_first_in_block())


/**
 * Note: Untested.
 */
template <typename T>
__fd__ T get_from_thread(const T& value, unsigned source_thread_index)
{
	__shared__ static T tmp;
	if (thread::index_in_block() == source_thread_index) {
		tmp = value;
	}
	__syncthreads();
	return tmp;
}

template <typename T>
__fd__ T get_from_first_thread(const T& value)
{
	return get_from_thread(value, 0);
}


namespace dynamic_shared_memory {


template <typename T>
__fd__ void fill(
	const T&                 value,
	shared_memory::size_t    length)
{
	T tmp = value;
	fill_n(shared_memory::dynamic::proxy<T>(), value, length);
}

/**
 * @note This variant of @ref fill_shared_memory() is slightly slower,
 * since it must access a typically-unused special register to determine
 * the shared memory size. If you can, prefer passing the count yourself.
 */
template <typename T>
__fd__ void fill(const T& value)
{
	auto length = ::shared_memory::dynamic::size<T>();
	return fill(value, length);
}

template <typename T>
__fd__ void zero(shared_memory::size_t length)
{
	return fill(T{0}, length);
}

template <typename T>
__fd__ void zero()
{
	auto length = ::shared_memory::dynamic::size<T>();
	return zero(length);
}


/**
 * Sets the (beginning of the dynamic) shared memory of the block
 * to a copy of some area of device memory.
 *
 * @param[in]  source Data in global memory (_not_ anywhere
 * else in shared memory! That breaks the {@code __restrict__}
 * restriction) which we wish to have in shared memory
 * @param[in] length length of the area to copy; must be
 * no larger than the available length (in T's) of shared
 * memory
 * @return the beginning of the block's shared memory -
 * which now contains a copy of the data at @p source.
 *
 * @note length is not checked to be valid - it is up to
 * the caller to refrain from trying to copy too much
 * into the shared memory; use
 */
template <typename T, typename Size>
__fd__ T* __restrict__ set_to_copy_of(
	const T*  __restrict__  source,
	Size                    length)
{
	T* __restrict__ data_in_shared_mem = shared_memory::dynamic::proxy<T>();
	copy<T, Size>(data_in_shared_mem, source, length);
	return data_in_shared_mem;
}

} // namespace dyanmic_shared_memory


} // namespace block
} // namespace primitives
} // namespace kat

#include <kat/undefine_specifiers.hpp>

#endif // BLOCK_LEVEL_PRIMITIVES_CUH_
