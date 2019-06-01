#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "macro.h"
#include "common.cuh"
#include "utilities.cuh"
//#include "../src/kat/on_device/constexpr_math.cuh"
#include <kat/on_device/constexpr_math.cuh>
//#include "../external/doctest/doctest.h"
#include <limits>

namespace kernels {
} // namespace kernels

namespace kce = kat::constexpr_;

// TODO: What about invalid arguments?

template <typename I>
struct compile_time_execution_results {


    static_assert(kce::strictly_between<I>( I{   0 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   1 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   4 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   5 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   6 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   8 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{   9 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{  10 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{  11 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");
    static_assert(kce::strictly_between<I>( I{ 123 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::strictly_between");

    static_assert(kce::between_or_equal<I>( I{   1 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{   4 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{   5 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{   6 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{   8 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{   9 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{  10 }, I{  5 }, I{  10 } ) == true,  "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{  11 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::between_or_equal");
    static_assert(kce::between_or_equal<I>( I{ 123 }, I{  5 }, I{  10 } ) == false, "kat::constexpr_::between_or_equal");

    static_assert(kce::is_power_of_2<I>(I{ 1}) == true,  "kat::constexpr_::is_power_of_2( 1) error");
    static_assert(kce::is_power_of_2<I>(I{ 2}) == true,  "kat::constexpr_::is_power_of_2( 2) error");
    static_assert(kce::is_power_of_2<I>(I{ 4}) == true,  "kat::constexpr_::is_power_of_2( 4) error");
    static_assert(kce::is_power_of_2<I>(I{ 7}) == false, "kat::constexpr_::is_power_of_2( 7) error");
    static_assert(kce::is_power_of_2<I>(I{32}) == true,  "kat::constexpr_::is_power_of_2(32) error");
    static_assert(kce::is_power_of_2<I>(I{33}) == false, "kat::constexpr_::is_power_of_2(33) error");

    static_assert(kce::modular_increment<I>(I{ 0}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 1}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 0}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 1}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 2}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 3}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_increment error");
    static_assert(kce::modular_increment<I>(I{ 4}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_increment error");

    static_assert(kce::modular_decrement<I>(I{ 0}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 1}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 0}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 1}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 2}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 3}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_decrement error");
    static_assert(kce::modular_decrement<I>(I{ 4}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_decrement error");

    static_assert(kce::ipow<I>(I{ 0 },   1 ) == I{  0 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 0 },   2 ) == I{  0 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 0 }, 100 ) == I{  0 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 1 },   0 ) == I{  1 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 1 },   1 ) == I{  1 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 1 },   2 ) == I{  1 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 1 }, 100 ) == I{  1 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 3 },   0 ) == I{  1 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 3 },   1 ) == I{  3 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 3 },   2 ) == I{  9 }, "kat::constexpr_::ipow error");
    static_assert(kce::ipow<I>(I{ 3 },   4 ) == I{ 81 }, "kat::constexpr_::ipow error");

    static_assert(kce::unsafe::div_rounding_up<I>( I{   0 }, I{   1 } ) == I{   0 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{   0 }, I{   2 } ) == I{   0 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{   0 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{   1 }, I{   2 } ) == I{   1 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{ 122 }, I{ 123 } ) == I{   1 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{ 123 }, I{ 123 } ) == I{   1 }, "kat::constexpr_::unsafe::div_rounding_up error");
    static_assert(kce::unsafe::div_rounding_up<I>( I{ 124 }, I{ 123 } ) == I{   2 }, "kat::constexpr_::unsafe::div_rounding_up error");

    static_assert(kce::div_rounding_up<I>( I{   0 }, I{   1 } ) == I{   0 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{   0 }, I{   2 } ) == I{   0 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{   0 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{   1 }, I{   2 } ) == I{   1 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{ 122 }, I{ 123 } ) == I{   1 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{ 123 }, I{ 123 } ) == I{   1 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{ 124 }, I{ 123 } ) == I{   2 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( I{ 124 }, I{ 123 } ) == I{   2 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( std::numeric_limits<I>::max()    , std::numeric_limits<I>::max() - 1 ) == I{   2 }, "kat::constexpr_::div_rounding_up error");
    static_assert(kce::div_rounding_up<I>( std::numeric_limits<I>::max() - 1, std::numeric_limits<I>::max()     ) == I{   1 }, "kat::constexpr_::div_rounding_up error");

    static_assert(kce::round_down<I>( I{   0 }, I{   2 } ) == I{   0 }, "kat::constexpr_::round_down error");
    static_assert(kce::round_down<I>( I{   0 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::round_down error");
    static_assert(kce::round_down<I>( I{   1 }, I{   2 } ) == I{   0 }, "kat::constexpr_::round_down error");
    static_assert(kce::round_down<I>( I{ 122 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::round_down error");
    static_assert(kce::round_down<I>( I{ 123 }, I{ 123 } ) == I{ 123 }, "kat::constexpr_::round_down error");
    static_assert(kce::round_down<I>( I{ 124 }, I{ 123 } ) == I{ 123 }, "kat::constexpr_::round_down error");

    static_assert(kce::round_down_to_full_warps<I>( I{   0 } ) == I{  0 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{   1 } ) == I{  0 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{   8 } ) == I{  0 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{  16 } ) == I{  0 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{  31 } ) == I{  0 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{  32 } ) == I{ 32 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{  33 } ) == I{ 32 }, "kat::constexpr_::round_down_to_full_warps error");
    static_assert(kce::round_down_to_full_warps<I>( I{ 125 } ) == I{ 96 }, "kat::constexpr_::round_down_to_full_warps error");

    // TODO: Consider testing rounding-up with negative dividends

    static_assert(kce::unsafe::round_up<I>( I{   0 }, I{   1 } ) == I{   0 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{   0 }, I{   2 } ) == I{   0 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{   0 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{   1 }, I{   2 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{  63 }, I{  64 } ) == I{  64 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{  64 }, I{  64 } ) == I{  64 }, "kat::constexpr_::unsafe::round_up error");
    static_assert(kce::unsafe::round_up<I>( I{  65 }, I{  32 } ) == I{  96 }, "kat::constexpr_::unsafe::round_up error");

    static_assert(kce::round_up<I>( I{   0 }, I{   1 } ) == I{   0 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{   0 }, I{   2 } ) == I{   0 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{   0 }, I{ 123 } ) == I{   0 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{   1 }, I{   2 } ) == I{   2 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{  63 }, I{  64 } ) == I{  64 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{  64 }, I{  64 } ) == I{  64 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( I{  65 }, I{  32 } ) == I{  96 }, "kat::constexpr_::round_up error");
    static_assert(kce::round_up<I>( std::numeric_limits<I>::max() - 1, std::numeric_limits<I>::max() ) == I{ std::numeric_limits<I>::max() }, "kat::constexpr_::round_up error");

    static_assert(kce::round_down_to_power_of_2<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   2 }, I{   1 } ) == I{   2 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   3 }, I{   1 } ) == I{   3 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   4 }, I{   1 } ) == I{   4 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{ 123 }, I{   1 } ) == I{ 123 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   1 }, I{   2 } ) == I{   0 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   2 }, I{   2 } ) == I{   2 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   3 }, I{   2 } ) == I{   2 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{   4 }, I{   2 } ) == I{   4 }, "kat::constexpr_::round_down_to_power_of_2 error");
    static_assert(kce::round_down_to_power_of_2<I>( I{ 123 }, I{   2 } ) == I{ 122 }, "kat::constexpr_::round_down_to_power_of_2 error");

    static_assert(kce::round_up_to_power_of_2<I>( I{   1 }, I{   1 } ) == I{   1 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   2 }, I{   1 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   3 }, I{   1 } ) == I{   3 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   4 }, I{   1 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{ 123 }, I{   1 } ) == I{ 123 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   1 }, I{   2 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   2 }, I{   2 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   3 }, I{   2 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{   4 }, I{   2 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::round_up_to_power_of_2<I>( I{ 123 }, I{   2 } ) == I{ 124 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");

    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  1 }, I{  1 } ) == I{   1 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  2 }, I{  1 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  3 }, I{  1 } ) == I{   3 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  4 }, I{  1 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{ 23 }, I{  1 } ) == I{  23 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  1 }, I{  2 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  2 }, I{  2 } ) == I{   2 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  3 }, I{  2 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{  4 }, I{  2 } ) == I{   4 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");
    static_assert(kce::unsafe::round_up_to_power_of_2<I>( I{ 63 }, I{  2 } ) == I{  64 }, "kat::constexpr_::unsafe::round_up_to_power_of_2 error");

    static_assert(kce::round_up_to_full_warps<I>( I{   0 } ) == I{  0 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{   1 } ) == I{ 32 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{   8 } ) == I{ 32 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{  16 } ) == I{ 32 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{  31 } ) == I{ 32 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{  32 } ) == I{ 32 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{  33 } ) == I{ 64 }, "kat::constexpr_::round_up_to_full_warps error");
    static_assert(kce::round_up_to_full_warps<I>( I{  63 } ) == I{ 64 }, "kat::constexpr_::round_up_to_full_warps error");

#if __cplusplus >= 201402L
    static_assert(kce::gcd<I>( I{   1 }, I{   1 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   2 }, I{   1 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   1 }, I{   2 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   2 }, I{   2 } ) == I{  2 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   8 }, I{   4 } ) == I{  4 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   4 }, I{   8 } ) == I{  4 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  10 }, I{   6 } ) == I{  2 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{ 120 }, I{  70 } ) == I{ 10 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  70 }, I{ 120 } ) == I{ 10 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  97 }, I{ 120 } ) == I{  1 }, "kat::constexpr_::gcd error");
#endif

    static_assert(kce::gcd<I>( I{   1 }, I{   1 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   2 }, I{   1 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   1 }, I{   2 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   2 }, I{   2 } ) == I{  2 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   5 }, I{   3 } ) == I{  1 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   8 }, I{   4 } ) == I{  4 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{   4 }, I{   8 } ) == I{  4 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  10 }, I{   6 } ) == I{  2 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{ 120 }, I{  70 } ) == I{ 10 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  70 }, I{ 120 } ) == I{ 10 }, "kat::constexpr_::gcd error");
    static_assert(kce::gcd<I>( I{  97 }, I{ 120 } ) == I{  1 }, "kat::constexpr_::gcd error");

    static_assert(kce::lcm<I>( I{   1 }, I{   1 } ) == I{  1 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   2 }, I{   1 } ) == I{  2 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   1 }, I{   2 } ) == I{  2 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   2 }, I{   2 } ) == I{  2 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   5 }, I{   3 } ) == I{ 15 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   8 }, I{   4 } ) == I{  8 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{   4 }, I{   8 } ) == I{  8 }, "kat::constexpr_::lcm error");
    static_assert(kce::lcm<I>( I{  10 }, I{   6 } ) == I{ 30 }, "kat::constexpr_::lcm error");

    static_assert(kce::is_even<I>( I{   0 } ) == true,  "kat::constexpr_::is_even error");
    static_assert(kce::is_even<I>( I{   1 } ) == false, "kat::constexpr_::is_even error");
    static_assert(kce::is_even<I>( I{   2 } ) == true,  "kat::constexpr_::is_even error");
    static_assert(kce::is_even<I>( I{   3 } ) == false, "kat::constexpr_::is_even error");
    static_assert(kce::is_even<I>( I{ 123 } ) == false, "kat::constexpr_::is_even error");
    static_assert(kce::is_even<I>( I{ 124 } ) == true,  "kat::constexpr_::is_even error");

    static_assert(kce::is_odd<I>( I{   0 } ) == false, "kat::constexpr_::is_odd error");
    static_assert(kce::is_odd<I>( I{   1 } ) == true,  "kat::constexpr_::is_odd error");
    static_assert(kce::is_odd<I>( I{   2 } ) == false, "kat::constexpr_::is_odd error");
    static_assert(kce::is_odd<I>( I{   3 } ) == true,  "kat::constexpr_::is_odd error");
    static_assert(kce::is_odd<I>( I{ 123 } ) == true,  "kat::constexpr_::is_odd error");
    static_assert(kce::is_odd<I>( I{ 124 } ) == false, "kat::constexpr_::is_odd error");

    static_assert(kce::log2<I>( I{   1 } ) == 0, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   2 } ) == 1, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   3 } ) == 1, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   4 } ) == 2, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   6 } ) == 2, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   7 } ) == 2, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{   8 } ) == 3, "kat::constexpr_::log2 error");
    static_assert(kce::log2<I>( I{ 127 } ) == 6, "kat::constexpr_::log2 error");

    static_assert(kce::sqrt<I>( I{   0 } ) ==  0, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   1 } ) ==  1, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   2 } ) ==  1, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   3 } ) ==  1, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   4 } ) ==  2, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   5 } ) ==  2, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{   9 } ) ==  3, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{  10 } ) ==  3, "kat::constexpr_::sqrt error");
    static_assert(kce::sqrt<I>( I{ 127 } ) == 11, "kat::constexpr_::sqrt error");

    static_assert(kce::div_by_power_of_2<I>( I{   0 }, I {  1 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   1 }, I {  1 }) == I{   1 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{ 111 }, I {  1 }) == I{ 111 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   0 }, I {  2 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   1 }, I {  2 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   2 }, I {  2 }) == I{   1 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   3 }, I {  2 }) == I{   1 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   4 }, I {  2 }) == I{   2 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{ 111 }, I {  2 }) == I{  55 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   0 }, I { 16 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{   1 }, I { 16 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{  15 }, I { 16 }) == I{   0 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{  16 }, I { 16 }) == I{   1 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{  17 }, I { 16 }) == I{   1 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{  32 }, I { 16 }) == I{   2 }, "kat::constexpr_::div_by_power_of_2 error");
    static_assert(kce::div_by_power_of_2<I>( I{ 111 }, I { 16 }) == I{   6 }, "kat::constexpr_::div_by_power_of_2 error");

    static_assert(kce::divides<I>( I{   1 }, I{   0 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   2 }, I{   0 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   3 }, I{   0 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   1 }, I{   1 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   2 }, I{   1 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   3 }, I{   1 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   1 }, I{   2 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   2 }, I{   2 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   3 }, I{   2 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   4 }, I{   2 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   6 }, I{   9 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   9 }, I{   6 } ) == false, "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{   4 }, I{  24 } ) == true,  "kat::constexpr_::divides error");
    static_assert(kce::divides<I>( I{  24 }, I{   4 } ) == false, "kat::constexpr_::divides error");

    static_assert(kce::is_divisible_by<I>( I{   0 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   0 }, I{   2 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   0 }, I{   3 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   1 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   1 }, I{   2 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   1 }, I{   3 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   2 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   2 }, I{   2 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   2 }, I{   3 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   2 }, I{   4 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   9 }, I{   6 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   6 }, I{   9 } ) == false, "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{  24 }, I{   4 } ) == true,  "kat::constexpr_::is_divisible_by error");
    static_assert(kce::is_divisible_by<I>( I{   4 }, I{  24 } ) == false, "kat::constexpr_::is_divisible_by error");

    static_assert(kce::is_divisible_by_power_of_2<I>( I{   0 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   0 }, I{   2 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   1 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   1 }, I{   2 } ) == false, "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   2 }, I{   1 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   2 }, I{   2 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{   2 }, I{   4 } ) == false, "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{  24 }, I{   4 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{  72 }, I{  16 } ) == false, "kat::constexpr_::is_divisible_by_power_of_2 error");
    static_assert(kce::is_divisible_by_power_of_2<I>( I{  64 }, I{  16 } ) == true,  "kat::constexpr_::is_divisible_by_power_of_2 error");

    static_assert(kce::power_of_2_divides<I>( I{   1 }, I{   0 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   2 }, I{   0 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   1 }, I{   1 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   2 }, I{   1 } ) == false, "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   1 }, I{   2 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   2 }, I{   2 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   4 }, I{   2 } ) == false, "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{   4 }, I{  24 } ) == true,  "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{  16 }, I{  72 } ) == false, "kat::constexpr_::power_of_2_divides error");
    static_assert(kce::power_of_2_divides<I>( I{  16 }, I{  64 } ) == true,  "kat::constexpr_::power_of_2_divides error");

    static_assert(kce::log2_of_power_of_2<I>( I{  1 } ) == I { 0 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{  2 } ) == I { 1 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{  4 } ) == I { 2 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{  8 } ) == I { 3 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{ 16 } ) == I { 4 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{ 32 } ) == I { 5 }, "kat::constexpr_::log2_of_power_of_2");
    static_assert(kce::log2_of_power_of_2<I>( I{ 64 } ) == I { 6 }, "kat::constexpr_::log2_of_power_of_2");


    static_assert(kce::modulo_power_of_2<I>( I{   0 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   1 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   2 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   3 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   4 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   5 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{  63 }, I{   1 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   0 }, I{   2 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   1 }, I{   2 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   2 }, I{   2 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   3 }, I{   2 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   4 }, I{   2 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   5 }, I{   2 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{  63 }, I{   2 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   0 }, I{   4 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   1 }, I{   4 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   2 }, I{   4 } ) == I{ 2 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   3 }, I{   4 } ) == I{ 3 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   4 }, I{   4 } ) == I{ 0 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{   5 }, I{   4 } ) == I{ 1 },  "kat::constexpr_::modulo_power_of_2 error");
    static_assert(kce::modulo_power_of_2<I>( I{  63 }, I{   4 } ) == I{ 3 },  "kat::constexpr_::modulo_power_of_2 error");

};

// TODO:
// * Test between_or_equal and strictly_between with differing types for all 3 arguments
// * Some floating-point tests
// * gcd tests with values of different types
// * Some tests with negative values

#define INSTANTIATE_CONSTEXPR_MATH_TEST(_tp) \
	compile_time_execution_results<_tp> UNIQUE_IDENTIFIER(test_struct_); \

MAP(INSTANTIATE_CONSTEXPR_MATH_TEST,
	int8_t, int16_t, int32_t, int64_t,
	uint8_t, uint16_t, uint32_t, uint64_t,
	char, short, int, long, long long,
	signed char, signed short, signed int, signed long, signed long long,
	unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long);



TEST_SUITE("constexpr_math") {

TEST_CASE_TEMPLATE("run-time on-host", T, int32_t, int64_t, float, double)
{
    (void) 0; // Don't need to do anything
}

} // TEST_SUITE("constexpr_math")
