/**
 * @file on_device/builtins.cuh
 *
 * @brief C++ wrappers for single PTX instructions (in the `builtins` namespace).
 *
 * @note
 * 1. This obviously doesn't include those built-ins which are inherent
 * operators in C++ as a language, i.e. % + / * - << >> and so on.
 * 2. PTX collaboration/single instructions don't always translate to single
 * SASS (GPU assembly) instructions - as PTX is an intermediate
 * representation (IR) common to multiple GPU microarchitectures.
 * 3. No function here performs any computation other beyond a single PTX
 * instruction; non-built-in operations belong in other files. But
 * this is only _almost_true. The functions here do have:
 *    3.1 Type casts
 *    3.2 Substitutions of a constant, for an instruction parameter,
 *        especially via default arguments.
 * 4. The templated builtins are only available for a _subset_ of the
 *    fundamental C++ types (and never for aggregate types); other files
 *    utilize these actual built-ins to generalize them to a richer set
 *    of types.
 */
#ifndef CUDA_KAT_ON_DEVICE_BUILTINS_CUH_
#define CUDA_KAT_ON_DEVICE_BUILTINS_CUH_

#include <kat/on_device/common.cuh>


///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

namespace kat {

/**
 * @brief Uniform-naming scheme, templated-when-relevant wrappers of single PTX instruction
 *
 * @note should contain wrappers for all instructions which are not trivially
 * producible with simple C++ code (e.g. no add or subtract)
 */
namespace builtins {

// Arithmetic
// --------------------------------------------

/**
 * When multiplying two n-bit numbers, the result may take up to 2n bits.
 * without upcasting, the value of x * y is the lower n bits of the result;
 * this lets you get the upper bits, without performing a 2n-by-2n multiplication
 */
template <typename I> KAT_FD  I multiplication_high_bits(I x, I y);

/**
 * Division which becomes faster and less precise than regular "/",
 * when --use-fast-math is specified; otherwise it's the same as regular "/".
 */
template <typename F> KAT_FD F divide(F dividend, F divisor);
template <typename T> KAT_FD T absolute_value(T x);
template <typename T> KAT_FD T minimum(T x, T y) = delete; // don't worry, it's not really deleted for all types
template <typename T> KAT_FD T maximum(T x, T y) = delete; // don't worry, it's not really deleted for all types

/**
 * @brief Computes @p addend + |@p x- @p y| .
 *
 * See the <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad">relevant section</a>
 * of the PTX ISA reference.
 */
template <typename I1, typename I2> KAT_FD I2 sum_with_absolute_difference(I1 x, I1 y, I2 addend);


// --------------------------------------------



// Bit and byte manipulation
// --------------------------------------------

template <typename I> KAT_FD int population_count(I x);
template <typename I> KAT_FD I bit_reverse(I x) = delete;

template <typename I> KAT_FD unsigned find_last_non_sign_bit(I x) = delete;
template <typename T> KAT_FD T load_global_with_non_coherent_cache(const T* ptr);
template <typename I> KAT_FD int count_leading_zeros(I x) = delete;


namespace bit_field {

/**
 * Extracts the bits with 0-based indices @p start_pos ... @p start_pos+ @p num_bits - 1, counting
 * from least to most significant, from a bit field field. Has sign extension semantics
 * for signed inputs which are bit tricky, see in the PTX ISA guide:
 *
 * http://docs.nvidia.com/cuda/parallel-thread-execution/index.html
 *
 * @todo CUB 1.5.2's BFE wrapper seems kind of fishy. Why does Duane Merill not use PTX for extraction from 64-bit fields?
 * For now only adopting his implementation for the 32-bit case.
 */
template <typename T> KAT_FD T extract(T bit_field, unsigned int start_pos, unsigned int num_bits);
template <typename T> KAT_FD T insert(T original_bit_field, T bits_to_insert, unsigned int start_pos, unsigned int num_bits);

// TODO: Implement these.
//template <typename BitField, typename T> KAT_FD T extract(BitField bit_field, unsigned int start_pos);
//template <typename BitField, typename T> KAT_FD BitField insert(BitField original_bit_field, T bits_to_insert, unsigned int start_pos);

} // namespace bit_field

template <typename T> KAT_FD T select_bytes(T x, T y, unsigned byte_selector);

/**
 * Use this to select which variant of the funnel shift intrinsic to use
 */
enum class funnel_shift_amount_resolution_mode_t {
	take_lower_bits,       //!< Shift by shift_amount & (size_in_bits<native_word_t> - 1)
	cap_at_full_word_size, //!< Shift by max(shift_amount, size_in_bits<native_word_t>)
};

/**
 * @brief Performs a right-shift on the combination of the two arguments
 * into a single, double-the-length, value
 *
 * @todo Perhaps make the "amount resolution mode" a template argument?
 *
 * @param low_word
 * @param high_word
 * @param shift_amount The number of bits to right-shift
 *
 * @tparam AmountResolutionMode shift_amount can have values which are
 * higher than the maximum possible number of bits to right-shift; this
 * indicates how to interpret such values. Hopefully this should be
 * gone when inlining
 *
 * @return the lower bits of the result
 */
template <
	typename T,
	funnel_shift_amount_resolution_mode_t AmountResolutionMode =
		funnel_shift_amount_resolution_mode_t::take_lower_bits
>
KAT_FD native_word_t funnel_shift(
	native_word_t  low_word,
	native_word_t  high_word,
	native_word_t  shift_amount);

// --------------------------------------------

/**
 * @brief compute the average of two values without needing special
 * accounting for overflow
 */
template <bool Signed, bool Rounded = false> KAT_FD
typename std::conditional<Signed, int, unsigned>::type average(
	typename std::conditional<Signed, int, unsigned>::type x,
	typename std::conditional<Signed, int, unsigned>::type y);



/**
 * Special register getter wrappers
 */
namespace special_registers {

KAT_FD unsigned           lane_index();
KAT_FD unsigned           symmetric_multiprocessor_index();
KAT_FD unsigned long long grid_index();
KAT_FD unsigned int       dynamic_shared_memory_size();
KAT_FD unsigned int       total_shared_memory_size();

} // namespace special_registers

namespace warp {

#if (__CUDACC_VER_MAJOR__ >= 9)
KAT_FD lane_mask_t ballot            (int condition, lane_mask_t lane_mask = full_warp_mask);
KAT_FD int         all_lanes_satisfy (int condition, lane_mask_t lane_mask = full_warp_mask);
KAT_FD int         some_lanes_satisfy(int condition, lane_mask_t lane_mask = full_warp_mask);
KAT_FD int         all_lanes_agree   (int condition, lane_mask_t lane_mask = full_warp_mask);
#else
KAT_FD lane_mask_t ballot            (int condition);
KAT_FD int         all_lanes_satisfy (int condition);
KAT_FD int         some_lanes_satisfy(int condition);
#endif

#if (__CUDACC_VER_MAJOR__ >= 9)
template <typename T> KAT_FD bool is_uniform_across_lanes(T value, lane_mask_t lane_mask = full_warp_mask);
template <typename T> KAT_FD bool is_uniform_across_warp(T value);
template <typename T> KAT_FD lane_mask_t matching_lanes(T value, lane_mask_t lanes = full_warp_mask);
#endif

namespace mask_of_lanes {

KAT_FD unsigned int preceding();
KAT_FD unsigned int preceding_and_self();
KAT_FD unsigned int self();
KAT_FD unsigned int succeeding_and_self();
KAT_FD unsigned int succeeding();

template <typename T> KAT_FD lane_mask_t matching_value(lane_mask_t lane_mask, T value);
template <typename T> KAT_FD lane_mask_t matching_value(T value);

} // namespace mask_of_lanes

namespace shuffle {

#if (__CUDACC_VER_MAJOR__ < 9)
template <typename T> KAT_FD T arbitrary(T x, int source_lane, int width = warp_size);
template <typename T> KAT_FD T down(T x, unsigned delta, int width = warp_size);
template <typename T> KAT_FD T up(T x, unsigned delta, int width = warp_size);
template <typename T> KAT_FD T xor_(T x, int lane_id_xoring_mask, int width = warp_size);
#else
template <typename T> KAT_FD T arbitrary(T x, int source_lane, int width = warp_size, lane_mask_t participants = full_warp_mask);
template <typename T> KAT_FD T down(T x, unsigned delta, int width = warp_size, lane_mask_t participants = full_warp_mask);
template <typename T> KAT_FD T up(T x, unsigned delta, int width = warp_size, lane_mask_t participants = full_warp_mask);
template <typename T> KAT_FD T xor_(T x, int lane_id_xoring_mask, int width = warp_size, lane_mask_t participants = full_warp_mask);
#endif
// Notes:
// 1. we have to use `xor_` here since `xor` is a reserved word
// 2. Why is lane_mask an `int` when bitmasks typically use unsigned types?
//    Because that's how nVIDIA's shuffle-xor signature expects it; probably
//    no good reason.
} // namespace shuffle

} // namespace warp


} // namespace builtins
} // namespace kat

#include "detail/builtins.cuh"

#endif // CUDA_KAT_ON_DEVICE_BUILTINS_CUH_
