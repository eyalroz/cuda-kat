/**
 * @file on_device/builtins.cuh
 *
 * @brief Templated, uniformly-named C++ functions wrapping single PTX
 * instructions (in a dedicated `builtins` namespace).
 *
 * CUDA provides C functions corresponding to many PTX instructions, which are
 * not otherwise easy, obvious or possible to generate with plain C or C++ code.
 * However - it doesn't provide such functions for all PTX instructions; nor
 * does it provide them in a type-generic way, for use in templated C++ code.
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
 * 5. This file (and its implementation) has _no_ PTX code. PTX instructions
 *    are wrapped in functions under the `ptx/` directory, which are not
 *    templated.
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

/**
 * @brief clamps the input value to the unit segment [0.0,+1.0].
 *
 * @note behavior undefined for nan/infinity/etc.
 *
 * @return max(0.0,min(1.0,x))
 */
template <typename F> KAT_FD F clamp_to_unit_segment(F x);

template <typename T> KAT_FD T absolute_value(T x)
{
	static_assert(std::is_unsigned<T>::value,
		"There is no generic implementation of absolute value for signed types, only for a few specific ones");
	return x;
}
template <typename T> KAT_FD T minimum(T x, T y) = delete; // don't worry, it's not really deleted for all types
template <typename T> KAT_FD T maximum(T x, T y) = delete; // don't worry, it's not really deleted for all types

/**
 * @brief Computes @p addend + |@p x- @p y| .
 *
 * See the <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad">relevant section</a>
 * of the PTX ISA reference.
 *
 * @note The addend and the result are always unsigned, but of the same size as @p x and @p y .
 */
template <typename I>
KAT_FD typename std::make_unsigned<I>::type
sum_with_absolute_difference(I x, I y, typename std::make_unsigned<I>::type addend);


// --------------------------------------------



// Bit and byte manipulation
// --------------------------------------------

template <typename I> KAT_FD int population_count(I x);
template <typename I> KAT_FD I bit_reverse(I x) = delete;

/**
 * @brief Find the most-significant, i.e. leading, bit that's different
 * from the input's sign bit.
 *
 * @return for unsigned types, 0-based index of the last 1 bit, starting
 * from the LSB towards the MSB; for signed integers it's the same if their
 * sign bit (their MSB) is 0, and the index of the last 0 bit if the sign
 * bit is 1.
 */
template <typename I> KAT_FD unsigned find_leading_non_sign_bit(I x) = delete;
#if __CUDA_ARCH__ >= 320
template <typename T> KAT_FD T load_global_with_non_coherent_cache(const T* ptr);
#endif

/**
 * @brief Return the number of bits, beginning from the least-significant,
 * which are all 0 ("leading" zeros)
 *
 * @return The number of leading zeros, between 0 and the size of I in bits.
 */
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
 *
 * @note This method is more "strict" in its specialization that others.
 */
template <typename I> KAT_FD I extract_bits(I bit_field, unsigned int start_pos, unsigned int num_bits) = delete;
template <typename I> KAT_FD I replace_bits(I original_bit_field, I bits_to_insert, unsigned int start_pos, unsigned int num_bits) = delete;

} // namespace bit_field

/**
 * @brief See: <a href="http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt">relevant section</a>
 * of the CUDA PTX reference for an explanation of what this does exactly
 *
 * @param first           a first value from which to potentially use bytes
 * @param second          a second value from which to potentially use bytes
 * @param byte_selectors  a packing of 4 selector structures; each selector structure
 *                        is 3 bits specifying which of the input bytes are to be used (as there are 8
 *                        bytes overall in @p first and @p second ), and another bit specifying if it's an
 *                        actual copy of a byte, or instead whether the sign of the byte (intrepeted as
 *                        an int8_t) should be replicated to fill the target byte.
 * @return the four bytes of first and/or second, or replicated signs thereof, indicated by the byte selectors
 *
 *@note If you don't use the sign-related bits, you could call this function "gather bytes" or "select bytes"
 *
 */
KAT_FD unsigned permute_bytes(unsigned first, unsigned second, unsigned byte_selectors);

/**
 * Use this to select which variant of the funnel shift intrinsic to use
 */
enum class funnel_shift_amount_resolution_mode_t {
	take_lower_bits_of_amount,  //!< Shift by shift_amount & (size_in_bits<native_word_t> - 1)
	cap_at_full_word_size,      //!< Shift by max(shift_amount, size_in_bits<native_word_t>)
};

/**
 * @brief Performs a right-shift on the combination of the two arguments
 * into a single, double-the-length, value
 *
 * @param low_word the lower 32-bit word
 * @param high_word the higher/upper 32-bit word
 * @param shift_amount The number of bits to right-shift
 *
 * @tparam AmountResolutionMode shift_amount can have values which are
 * higher than the maximum possible number of bits to right-shift; this
 * indicates how to interpret such values.
 *
 * @return the lower word (lower 32 bits) of the result
 */
template <
	funnel_shift_amount_resolution_mode_t AmountResolutionMode =
		funnel_shift_amount_resolution_mode_t::cap_at_full_word_size
>
KAT_FD uint32_t funnel_shift_right(
	uint32_t  low_word,
	uint32_t  high_word,
	uint32_t  shift_amount);

/**
 * @brief Performs a left-shift on the combination of the two arguments
 * into a single, double-the-length, value
 *
 * @param low_word the lower 32-bit word
 * @param high_word the higher/upper 32-bit word
 * @param shift_amount The number of bits to left-shift
 *
 * @tparam AmountResolutionMode shift_amount can have values which are
 * higher than the maximum possible number of bits to right-shift; this
 * indicates how to interpret such values.
 *
 * @return the higher world (higher 32 bits) of the result
 */
template <
	funnel_shift_amount_resolution_mode_t AmountResolutionMode =
		funnel_shift_amount_resolution_mode_t::cap_at_full_word_size
>
KAT_FD uint32_t funnel_shift_left(
	uint32_t  low_word,
	uint32_t  high_word,
	uint32_t  shift_amount);


// --------------------------------------------

/**
 * @brief compute the average of two integer values without needing special
 * accounting for overflow - rounding down
 */
template <typename I> I KAT_FD average(I x, I y) = delete; // don't worry, it's not really deleted for all types

/**
 * @brief compute the average of two values without needing special
 * accounting for overflow - rounding up
 *
 * @note  ignoring type limits, average_rounded_up(x,y) = floor ((x + y + 1 ) / 2)
 */
template <typename I> I KAT_FD average_rounded_up(I x, I y) = delete; // don't worry, it's not really deleted for all types

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
KAT_FD int         any_lanes_satisfy (int condition, lane_mask_t lane_mask = full_warp_mask);
KAT_FD int         all_lanes_agree   (int condition, lane_mask_t lane_mask = full_warp_mask);
	// Note: all_lanes_agree has the same semantics as all_lanes_

#else
KAT_FD lane_mask_t ballot            (int condition);
KAT_FD int         all_lanes_satisfy (int condition);
KAT_FD int         any_lanes_satisfy (int condition);
#endif

#if (__CUDACC_VER_MAJOR__ >= 9)
#if ! defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700

template <typename T> KAT_FD lane_mask_t propagate_mask_if_lanes_agree(T value, lane_mask_t lane_mask);
template <typename T> KAT_FD lane_mask_t propagate_mask_if_warp_agrees(T value);
template <typename T> KAT_FD lane_mask_t get_matching_lanes(T value, lane_mask_t lanes = full_warp_mask);

#endif
#endif

namespace mask_of_lanes {

KAT_FD unsigned int preceding();
KAT_FD unsigned int preceding_and_self();
KAT_FD unsigned int self();
KAT_FD unsigned int succeeding_and_self();
KAT_FD unsigned int succeeding();

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
