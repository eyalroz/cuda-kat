/**
 * @file on_device/collaboration/warp.cuh
 *
 * @brief CUDA device computation warp-level primitives, i.e. those involving
 * interaction of most/all of each warp's lanes, but _no_ inter-warp interaction.
 *
 * @todo
 * 1. Some of these assume linear grids, others do not - sort them out.
 * 2. Use a lane index type
 */

#pragma once
#ifndef CUDA_KAT_WARP_LEVEL_PRIMITIVES_CUH_
#define CUDA_KAT_WARP_LEVEL_PRIMITIVES_CUH_

#include "../grid_info.cuh"

#include <kat/on_device/shuffle.cuh>
#include <kat/on_device/builtins.cuh>
#include <kat/on_device/atomics.cuh>
#include <kat/on_device/non-builtins.cuh>
#include <kat/on_device/ptx.cuh>
#include <kat/on_device/math.cuh>
#include <kat/on_device/common.cuh>

#include <type_traits>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

// Lane-mask-related functions.
// TODO: Refactor these out of here
KAT_FD unsigned num_lanes_in(lane_mask_t mask)
{
	// Note the type cast from signed to unsigned
	return builtins::population_count(mask);
}

/**
 * @brief Determines which lane is the first within a lane mask
 * (considered in LSB-to-MSB order)
 *
 * @tparam ReturnWarpSizeForEmptyMask when set to true, the
 * semantics of this function will be consistent for empty
 * (all-zero) lane masks, in that the "first lane" inside
 * the mask will be one past the last lane - like in
 * the @ref `last_lane_in()` function
 *
 * @return index of the first 1-bit in the warp-size-bit mask;
 * if no lanes have a corresponding 1-bit, -1 or 32 (warp_size)
 * is returned, depending on @tparam ReturnWarpSizeForEmptyMask
 */
template <bool ReturnWarpSizeForEmptyMask = true>
KAT_FD int first_lane_in(lane_mask_t mask)
{
	return non_builtins::count_trailing_zeros<lane_mask_t, ReturnWarpSizeForEmptyMask>(mask);
}

/**
 * @brief Determines which lane is the first within a lane mask
 * (considered in LSB-to-MSB order)
 *
 * @return index of the first 1-bit in the warp-size-bit mask;
 * if no lanes have a corresponding 1-bit, 32 is returned;
 */
KAT_FD int last_lane_in(lane_mask_t mask)
{
	return builtins::count_leading_zeros(mask);
}


namespace collaborative {
namespace warp {

// If we want to refer to other primitives, we'll make those references explicit;
// but we do want to be able to say `warp::id()` without prefixing that with anything.

//namespace grid   = grid_info::grid;
//namespace block  = grid_info::block;
//namespace warp   = grid_info::warp;
//namespace thread = grid_info::thread;
namespace lane   = grid_info::lane;

// lane conditions
// ----------------------------

/**
 * Checks whether a condition holds for an entire warp of threads
 *
 * @param condition A boolean value (passed as an integer
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if condition is non-zero for all threads
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  bool all_lanes_satisfy(int condition)
#else
KAT_FD  bool all_lanes_satisfy(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	return builtins::warp::all_lanes_satisfy(condition);
#else
	return builtins::warp::all_lanes_satisfy(condition, lane_mask);
#endif
}


/**
 * Checks whether a condition holds for none of the threads in a warp
 *
 * @param condition A boolean value (passed as an integer
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if condition is zero for all threads
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  bool no_lanes_satisfy(int condition)
#else
KAT_FD  bool no_lanes_satisfy(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	return all_lanes_satisfy(not condition);
#else
	return all_lanes_satisfy(not condition, lane_mask);
#endif
}

/**
 * Checks whether a condition holds for an entire warp of threads
 *
 * @param condition A boolean value (passed as an integer
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if condition is non-zero for all threads
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  bool all_lanes_agree_on(int condition)
#else
KAT_FD  bool all_lanes_agree_on(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	auto ballot_results = builtins::warp::ballot(condition);
#else
	auto ballot_results = builtins::warp::ballot(condition, lane_mask);
#endif

	return
		    ballot_results == 0  // none satisfy the condition
		or ~ballot_results == 0; // all satisfy the condition);
}


/**
 * Checks whether a condition holds for at least one of the threads
 * in a warp
 *
 * @param condition A boolean value (passed as an integer
 * since that's what nVIDIA GPUs actually check with the HW instruction
 * @return true if condition is non-zero for at least one thread
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  bool some_lanes_satisfy(int condition)
#else
KAT_FD  bool some_lanes_satisfy(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	return !no_lanes_satisfy(condition);
#else
	return !no_lanes_satisfy(condition, lane_mask);
#endif
}

/**
 * Count the lanes in a warp for which some condition holds
 *
 * @param condition the condition value for each lane (true if non-zero)
 * @return the number of threads in the warp whose @p condition is true (non-zero)
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD native_word_t num_lanes_satisfying(int condition)
#else
KAT_FD native_word_t num_lanes_satisfying(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	return num_lanes_in(builtins::warp::ballot(condition));
#else
	return num_lanes_in(builtins::warp::ballot(condition, lane_mask));
#endif
}

/**
 * Count the lanes in a warp which have the same condition value as the calling lane
 *
 * @param condition the condition value for each lane (true if non-zero)
 * @return the number of threads in the warp whose @p condition is the same value
 * as the calling lane (including the calling lane itself)
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  native_word_t num_lanes_agreeing_on(int condition)
#else
KAT_FD  native_word_t num_lanes_agreeing_on(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
	auto satisfying =
#if (__CUDACC_VER_MAJOR__ < 9)
		num_lanes_satisfying(condition);
#else
		num_lanes_satisfying(condition, lane_mask);
#endif
	return condition ? satisfying : warp_size - satisfying;
}

/**
 * Check whether a condition holds for most lanes in a warp
 *
 * @param condition A boolean value (passed as an integer
 * since that's what nVIDIA GPUs actually check with the HW instruction
 */
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD  bool majority_vote(int condition)
#else
KAT_FD  bool majority_vote(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	return num_lanes_satisfying(condition) > (warp_size / 2);
#else
	return num_lanes_satisfying(condition, lane_mask) > (num_lanes_in(lane_mask) / 2);
#endif
}

// --------------------------------------------------

#if !defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700
/**
 * Compares values provided by each lane in a warp, checking for uniqueness
 *
 * @param value to check for matches; each lane brings its own
 * @param lane_mask the lanes participating in the uniqueness check (other
 * lanes' values are not considered); should be uniform among participating
 * lanes
 * @return true if the lane this value provided had no matches among the values
 * provided by other lanes
 */
#if (__CUDACC_VER_MAJOR__ < 9)
template <typename T> KAT_FD bool in_unique_lane_with(T value)
#else
template <typename T> KAT_FD bool in_unique_lane_with(T value, lane_mask_t lane_mask = full_warp_mask)
#endif
{
	auto self_lane_mask = (1 << lane::id());
		// Note we're _not_ using the PTX builtin for obtaining the self lane mask from a special
		// regiater - since that would probably be much slower.

	// Note: This assumes a lane's bit is always on in the result of get_matching_lanes();
	// this must indeed be the case, because The PTX spec demands that the calling lane be
	// part of its own masked lanes; see:
	//
	// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync
	//
#if (__CUDACC_VER_MAJOR__ < 9)
	return builtins::warp::get_matching_lanes(value) == self_lane_mask;
#else
	return builtins::warp::get_matching_lanes(value, lane_mask) == self_lane_mask;
#endif
}
#endif //  !defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700

/*
 *
 * Notes:
 *
 * 1. This relies on the shuffle generics! Without them it'll fail for > 32-bit types.
 * 2.Let's consider replacing this function with some proxy to enable syntax such as
 *
 *    warp_choose(my_var).from_lane(some_lane_id)
 * 3. The position parameter is a signed int because that's what the shuffle functions
 *    take; although perhaps we should make it a native_word_t?
 */
template <typename T>
KAT_FD T get_from_lane(T value, int source_lane)
{
	return shuffle_arbitrary(value, source_lane);
}

template <typename T>
KAT_FD T get_from_first_lane(T value)
{
	return get_from_lane(value, grid_info::warp::first_lane);
}

template <typename T>
KAT_FD T get_from_last_lane(T value)
{
	return get_from_lane(value, grid_info::warp::last_lane);
}

/**
 * Determines which is the first lane within a warp which satisfies some
 * boolean condition
 *
 * @param[in] condition an essentially-boolena value representing whether
 * or not the calling thread (lane) satisfies the condition; typed
 * as an int since that's what CUDA primitives take mostly.
 *
 * @return index of the first lane in the warp for which condition is non-zero.
 * If no lane has non-zero condition, then the result is either undefined,
 * or if @tparam DefinedResultOnNoSatisfaction  was set - 32 (warp_size).
 */
template <bool DefinedResultOnNoSatisfaction = true>
#if (__CUDACC_VER_MAJOR__ < 9)
KAT_FD native_word_t first_lane_satisfying(int condition)
#else
KAT_FD native_word_t first_lane_satisfying(int condition, lane_mask_t lane_mask = full_warp_mask)
#endif
{
#if (__CUDACC_VER_MAJOR__ < 9)
	auto ballot_results = builtins::warp::ballot(condition);
#else
	auto ballot_results = builtins::warp::ballot(condition, lane_mask);
#endif
	return first_lane_in<DefinedResultOnNoSatisfaction>(ballot_results);
}

KAT_FD lane_mask_t get_active_lanes()
{
	return __activemask();
}

KAT_FD unsigned num_active_lanes()
{
	return num_lanes_in(get_active_lanes());
}

namespace detail {

// Note: Return value is unspecified for empty lane masks
template <bool PreferFirstLane = true>
KAT_FD unsigned select_leader_lane_among(unsigned lanes_mask)
{
	return PreferFirstLane ?
		first_lane_in(lanes_mask) :	last_lane_in(lanes_mask);
}

template <bool PreferFirstLane = true>
KAT_FD bool am_leader_lane(unsigned active_lanes_mask)
{
	// This code hints that leadership is automatic given who's active - and in fact, it is,
	// despite the name "select leader lane". TODO: Perhaps we should rename that function, then?
	return select_leader_lane_among<PreferFirstLane>(active_lanes_mask) == lane::id();
}

KAT_FD unsigned lane_index_among(lane_mask_t mask)
{
	return num_lanes_in(ptx::special_registers::lanemask_lt() & mask);
	// Note: If we know we're in a linear grid, it may be faster to use threadIdx.x which
	// is probably already known:
	//
	// return num_lanes_in(builtins::bit_field::extract_bits(mask, threadIdx.x % kat::warp_size));
}


} // namespace detail

/**
 * This is a mechanism for making exactly one lane act instead of the whole warp,
 * which supports the case of some threads being inactive (e.g. having exited).
 *
 * @tparam PreferFirstLane if true, the first lane will be the acting leader
 * whenever it is active - at the cost of making this functions slightly more
 * expensive (an extra subtraction instruction)
 * @return Iif PreferFirstLane is true, and the first lane is active,
 * then 0; the index of some active lane otherwise.
 */
template <bool PreferFirstLane = true>
KAT_FD unsigned select_leader_lane()
{
	return detail::select_leader_lane_among<PreferFirstLane>( get_active_lanes() );
}

/**
 * This applies the leader lane selection mechanism to obtain a condition
 * for being the leader lane.
 *
 * @tparam PreferFirstLane if true, the first lane will be the acting leader
 * whenever it is active - at the cost of making this functions slightly more
 * expensive (an extra subtraction instruction)
 * @return 1 for exactly one of the active lanes in each warp, 0 for all others
 */
template <bool PreferFirstLane = true>
KAT_FD bool am_leader_lane()
{
	return detail::am_leader_lane<PreferFirstLane>( get_active_lanes() );
}

template <typename Function>
KAT_FD typename std::result_of<Function()>::type have_a_single_lane_compute(
	Function f, unsigned designated_computing_lane)
{
	typename std::result_of<Function()>::type result;
	if (lane::id() == designated_computing_lane) { result = f(); }
	return get_from_lane(result, designated_computing_lane);
}

template <typename Function, bool PreferFirstLane = true>
KAT_FD typename std::result_of<Function()>::type have_a_single_lane_compute(Function f)
{
	unsigned computing_lane { select_leader_lane<PreferFirstLane>() };
	return have_a_single_lane_compute(f, computing_lane);
}

template <typename Function>
KAT_FD typename std::result_of<Function()>::type have_first_lane_compute(Function f)
{
	return have_a_single_lane_compute<Function>(f, grid_info::warp::first_lane);
}

template <typename Function>
KAT_FD typename std::result_of<Function()>::type have_last_lane_compute(Function f)
{
	return have_a_single_lane_compute<Function>(f, grid_info::warp::last_lane);
}

KAT_FD unsigned index_among_active_lanes()
{
	return detail::lane_index_among( get_active_lanes() );
}

KAT_FD unsigned last_active_lane_index()
{
	return detail::lane_index_among( get_active_lanes() );
}

// TODO: Consider implementing have_first_active_lane_compute and
// have_last_active_lane_compute

/**
 * When every (active) lane in a warp needs to increment a counter,
 * use this function to avoid all of them having to execute atomics;
 * each active lane gets the "previous value" as though they had all
 * incremented in order.
 *
 * @note It's not clear to me how better this is from just using atomics
 * and being done with it
 *
 * @todo extend this to other atomic operations
 */
template <typename T>
KAT_FD T active_lanes_atomically_increment(T* counter)
{
	auto lanes_mask = get_active_lanes();
	auto active_lane_count = num_lanes_in(lanes_mask);
	auto perform_all_increments = [counter, active_lane_count]() -> T {
		return atomic::add<T>(counter, active_lane_count);
	};
	auto value_before_all_lane_increments =
		have_a_single_lane_compute(perform_all_increments,
			detail::select_leader_lane_among(lanes_mask));

	// the return value simulates the case of every lane having done its
	// own atomic increment
	return value_before_all_lane_increments +
		detail::lane_index_among(lanes_mask);
}


template <typename Function, typename Size = unsigned>
KAT_FD void at_warp_stride(Size length, Function f)
{
	// If the length is known at compile-time, perhaps this loop can be unrolled
	#pragma unroll
	for(promoted_size_t<Size> pos = lane::id();
		pos < length;
		pos += warp_size)
	{
		f(pos);
	}
}

} // namespace warp
} // namespace collaborative

namespace linear_grid {
namespace collaborative {
namespace warp {

/**
 * A structure for warp-level search results. Semantically,
 * it should have been std::optional-like, but that might
 * incur too much overhead, so we're just encoding the
 * 'empty' indication using the result fields.
 */
template <typename T>
struct search_result_t {
	native_word_t lane_index;
	T value;
	KAT_FHD bool is_set() const { return lane_index < warp_size; }
	KAT_FHD void unset() { lane_index = warp_size; }
	KAT_FHD bool operator==(const search_result_t<T>& other)
	{
		return
			(lane_index == other.lane_index)
			and ( (not is_set() ) or (value == other.value) );
	}
};

/**
 * Have each lane search for its own value of interest within the
 * sorted sequence of single values provided by all the warp lanes.
 *
 * @note The amount of time this function takes is very much
 * data-dependent!
 *
 * @note this function assumes all warp lanes are active.
 *
 * @todo Does it matter if the _needles_, as opposed to the
 * _hay straws_, are sorted? I wonder.
 *
 * @todo consider specializing for non-full warps
 *
 * @todo Specialize for smaller and larger data types: For
 * larger ones, compare 4-byte parts of the datum separately
 * (assuming @tparam T is bitwise-comparable); for smaller
 * ones, consider having lanes collect multiple data and pass
 * it on to other lanes, which can then spare some shuffling.
 *
 * @param lane_needle the value the current lane wants to search
 * for
 * @param lane_hay_straw the warp_size hay "straws" passed by
 * the lanes make up the entire "haystack" we search in. They
 * _must_ be known to be in order, i < j => straw of lane i <
 * straw of lane j. They are accessed using intra-warp shuffling,
 * solely.
 * @return For lane i, the index of the first lane j with
 * straw-of-j > needle-of-i, along with straw-of-j; if there is no such
 * lane j, warp_size is returned as the lane index and an arbitrary
 * value is returned as the result.
 */
template <typename T, bool AssumeNeedlesAreSorted = false>
KAT_FD search_result_t<T> multisearch(const T& lane_needle, const T& lane_hay_straw)
{
	search_result_t<T> result;
	result.unset();

	struct {
		unsigned lower, upper; // lower  is inclusive, upper is exclusive
	} bounds;
	if (lane_needle <= lane_hay_straw) {
		bounds.lower = grid_info::warp::first_lane;
		bounds.upper = grid_info::lane::id();
	}
	else {
		bounds.lower = grid_info::lane::id() + 1;
		bounds.upper = warp_size;
	}
	enum : unsigned { cutoff_to_linear_search = 6 };
		// is 6 a good choice for a cutoff? should it depend on the microarch?

	while (bounds.upper - bounds.lower >= cutoff_to_linear_search) {
		unsigned mid = (bounds.lower + bounds.upper) / 2;
		auto mid_lane_hay_straw = shuffle_arbitrary(lane_needle, mid);
		if (lane_needle <= mid_lane_hay_straw) { bounds.lower = mid + 1; }
		else { bounds.upper = mid; }
	}
	for(unsigned lane = bounds.lower; lane < bounds.upper; lane++) {
		auto hay_straw = shuffle_arbitrary(lane_needle, lane);
		if (not result.is_set() and hay_straw > lane_needle) {
			result = { lane, hay_straw };
		}
		// Note: We don't break from this loop even if we've found
		// our result - as we still need to participate in shuffles
	}
	return result;
}

template <typename Function, typename Size = unsigned>
KAT_FD void at_warp_stride(Size length, Function f)
{
	// If the length is known at compile-time, perhaps this loop can be unrolled
	#pragma unroll
	for(promoted_size_t<Size> pos = linear_grid::grid_info::lane::id();
		pos < length;
		pos += warp_size)
	{
		f(pos);
	}
}

namespace detail {

/**
 * Sometimes you have a run-time-determined range of indices
 * for which you need to compute some predicate, but you have
 * it constrained to be a nice round value, say a multiple of
 * warp_size * warp_size - or even a constant multiple, but by
 * means of, say, a configuration file. Now, the compiler will
 * not be able to figure out this is the case, and will compile
 * in the condition checks and the code for handling slack -
 * which you don't actually need. To prevent that and cater
 * to my OCD, you can use a template parameter to indicate how
 * nicely-behaving your input length is.
 */
enum predicate_computation_length_slack_t {
	has_no_slack,                //!< Length known to be multiple of warp_size * warp_size
	may_have_full_warps_of_slack,//!< Length only known to be multiple of warp_size
	may_have_arbitrary_slack     //!< Length may have any value, make no assumptions about it
};

} // namespace detail

/**
 * @brief Arrange the computation of a predicate by a warp so that
 * both reads and writes are coalesced and divergence is minimized
 *
 * This relies on the fact that there are as many threads in a warp as
 * there are bits in the native register size (a 4-byte unsigned int
 * has 32 bits, and warp size is 32). This means that the whole warp
 * computing the predicate once (i.e. for 32 elements), the results in
 * bits exactly suffice for a single write of a 4-byte value by a
 * single thread. Since we want coalesced writes, of 128 bytes at a
 * time, we need that many results for each of the lanes in a warp,
 * i.e. we need to have 32 times all-warp computations of the
 * predicate, with every consecutive 32 providing the write value
 * to one of the lanes. That's how this function arranges the
 * computation.
 *
 * @note The predicate is passed no input data other than the index
 * within the range 0..@p length - 1; if you need such input,
 * pass a closure (e.g. a lambda which captures the relevant data).
 *
 * @note the last bits of the last bit container - those which are
 * beyond @param length bits overall - are slack bits; it is assumed
 * we're allowed to write anything to them.
 *
 * @note There is no inter-warp collaboration here; the outputs
 * may be completely disjoint; and any warp's range's slack is
 * separate. See @ref kat::linear_grid::collabopration::grid::compute_predicate_at_warp_stride
 * for the case of a single joint range to cover.
 *
 * @param computed_predicate the result of the computation of the
 * predicate for each of the indices in the range
 * @param length The number of elements for which to compute the
 * predicate, which is also the number of bits (not the size in
 * bytes) of @p computed_predicate
 * @param predicate
 */
template <
	typename  Predicate,
	typename  Size  = native_word_t,
	detail::predicate_computation_length_slack_t PossibilityOfSlack = detail::may_have_arbitrary_slack
		// set Index to something smaller than Size if you have a size that's
		// something like 1 << sizeof(uint32_t), and then you have to use uint64_t as Size
> KAT_FD void compute_predicate_at_warp_stride(
	unsigned*         computed_predicate,
	Predicate&        predicate,
	Size              length)
{
	static_assert(warp_size == size_in_bits<native_word_t>(),
		"The assumption of having as many threads in a warp "
		"as there are bits in the native register size - "
		"doesn't hold; you can't use this function.");

	// The are three ways of proceeding with the computations, by decreasing preference:
	//
	// 1. Full-warp coalesced reads, full-warp coalesced writes (in units of warp_size^2)
	// 2. Full-warp coalesced reads, non-coalesced writes (in units of warp_size)
	// 3. Non-coalesced reads, non-coalesced writes (in units of 1)
	//
	// Using compile-time logic we'll try to completely avoid any consideration of
	// cases 2 and 3 when they're not absolutely necessary; otherwise, we'll do most of
	// the work in case 1

	promoted_size_t<Size> full_warp_reads_output_length = length >> log_warp_size;
	auto full_warp_writes_output_length = (PossibilityOfSlack == detail::has_no_slack) ?
		full_warp_reads_output_length :
		round_down_to_full_warps(full_warp_reads_output_length);
	const auto lane_index = grid_info::lane::id();


	promoted_size_t<Size> input_pos = lane_index;

	// This is the finger-licking-good part :-)

	promoted_size_t<Size> output_pos; // We'll need this after the loop as well.

	for (output_pos = lane_index;
	     output_pos < full_warp_writes_output_length;
	     output_pos += warp_size)
	{
		native_word_t warp_results_write_buffer;
		#pragma unroll
		for(native_word_t writing_lane = 0;
		    writing_lane < warp_size;
		    writing_lane++, input_pos += warp_size)
		{
			auto thread_result = predicate(input_pos);
			auto warp_results = builtins::warp::ballot(thread_result);
			if (lane_index == writing_lane) { warp_results_write_buffer = warp_results; }
		}
		computed_predicate[output_pos] = warp_results_write_buffer;
	}

	// ... and all the rest is ugly but necessary

	if (PossibilityOfSlack != detail::has_no_slack) {
		// In this case, the output length is not known to be a multiple
		// of 1024 = 32 * 32 = warp_size * warp_size
		//
		// Note that we're continuing to advance our input and output
		// position variables

		promoted_size_t<Size> full_warp_reads_output_slack_length =
			full_warp_reads_output_length - full_warp_writes_output_length;
		native_word_t warp_results_write_buffer;
		if (full_warp_reads_output_slack_length > 0) {
			for (native_word_t writing_lane = 0;
				 writing_lane < full_warp_reads_output_slack_length;
				 writing_lane++, input_pos += warp_size)
			{
				auto thread_result = predicate(input_pos);
				auto warp_results = builtins::warp::ballot(thread_result);
				if (lane_index == writing_lane) {
					warp_results_write_buffer = warp_results;
				}
			}
		}
		native_word_t num_writing_lanes = full_warp_reads_output_slack_length;
		if (PossibilityOfSlack == detail::may_have_arbitrary_slack) {
			native_word_t input_slack_length = length % warp_size; // let's hope this gets optimized...
			if (input_slack_length > 0) {
				auto thread_result = (input_pos < length) ? predicate(input_pos) : false;
				auto warp_results = builtins::warp::ballot(thread_result);
				if (lane_index == num_writing_lanes) { warp_results_write_buffer = warp_results; }
				num_writing_lanes++;
			}
		}
		// Note it could theoretically be the case that num_writing_lanes is 0
		if (lane_index < num_writing_lanes) {
			computed_predicate[output_pos] = warp_results_write_buffer;
		}
	}
}


} // namespace warp
} // namespace collaborative
} // namespace linear_grid
} // namespace kat

#endif // CUDA_KAT_WARP_LEVEL_PRIMITIVES_CUH_
