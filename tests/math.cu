#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/math.cuh>
#include <kat/on_device/printing.cuh>

namespace kernels {



template <typename I>
__global__ void try_out_integral_math_functions(I* results, I* __restrict expected)
{
	size_t i { 0 };
	bool print_first_indices_for_each_function { false };

	auto maybe_print = [&](const char* section_title) {
		if (print_first_indices_for_each_function) {
			printf("%-30s tests start at index  %3d\n", section_title, i);
		}
	};

	results[i] = kat::strictly_between<I>( I{   0 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{   1 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{   4 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{   5 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{   6 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::strictly_between<I>( I{   8 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::strictly_between<I>( I{   9 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::strictly_between<I>( I{  10 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{  11 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::strictly_between<I>( I{ 123 }, I{  5 }, I{  10 } ); expected[i++] = false;

	maybe_print("between_or_equal");
	results[i] = kat::between_or_equal<I>( I{   1 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::between_or_equal<I>( I{   4 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::between_or_equal<I>( I{   5 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::between_or_equal<I>( I{   6 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::between_or_equal<I>( I{   8 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::between_or_equal<I>( I{   9 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::between_or_equal<I>( I{  10 }, I{  5 }, I{  10 } ); expected[i++] = true;
	results[i] = kat::between_or_equal<I>( I{  11 }, I{  5 }, I{  10 } ); expected[i++] = false;
	results[i] = kat::between_or_equal<I>( I{ 123 }, I{  5 }, I{  10 } ); expected[i++] = false;

	maybe_print("is_power_of_2");
	results[i] = kat::is_power_of_2<I>(I{ 1}); expected[i++] = true;
	results[i] = kat::is_power_of_2<I>(I{ 2}); expected[i++] = true;
	results[i] = kat::is_power_of_2<I>(I{ 4}); expected[i++] = true;
	results[i] = kat::is_power_of_2<I>(I{ 7}); expected[i++] = false;
	results[i] = kat::is_power_of_2<I>(I{32}); expected[i++] = true;
	results[i] = kat::is_power_of_2<I>(I{33}); expected[i++] = false;

	maybe_print("modular_increment");
	results[i] = kat::modular_increment<I>(I{ 0}, I{ 1}); expected[i++] = I{ 0 };
	results[i] = kat::modular_increment<I>(I{ 1}, I{ 1}); expected[i++] = I{ 0 };
	results[i] = kat::modular_increment<I>(I{ 0}, I{ 3}); expected[i++] = I{ 1 };
	results[i] = kat::modular_increment<I>(I{ 1}, I{ 3}); expected[i++] = I{ 2 };
	results[i] = kat::modular_increment<I>(I{ 2}, I{ 3}); expected[i++] = I{ 0 };
	results[i] = kat::modular_increment<I>(I{ 3}, I{ 3}); expected[i++] = I{ 1 };
	results[i] = kat::modular_increment<I>(I{ 4}, I{ 3}); expected[i++] = I{ 2 };

	maybe_print("modular_decrement");
	results[i] = kat::modular_decrement<I>(I{ 0}, I{ 1}); expected[i++] = I{ 0 };
	results[i] = kat::modular_decrement<I>(I{ 1}, I{ 1}); expected[i++] = I{ 0 };
	results[i] = kat::modular_decrement<I>(I{ 0}, I{ 3}); expected[i++] = I{ 2 };
	results[i] = kat::modular_decrement<I>(I{ 1}, I{ 3}); expected[i++] = I{ 0 };
	results[i] = kat::modular_decrement<I>(I{ 2}, I{ 3}); expected[i++] = I{ 1 };
	results[i] = kat::modular_decrement<I>(I{ 3}, I{ 3}); expected[i++] = I{ 2 };
	results[i] = kat::modular_decrement<I>(I{ 4}, I{ 3}); expected[i++] = I{ 0 };

	maybe_print("ipow");
	results[i] = kat::ipow<I>(I{ 0 },   1 ); expected[i++] = I{  0 };
	results[i] = kat::ipow<I>(I{ 0 },   2 ); expected[i++] = I{  0 };
	results[i] = kat::ipow<I>(I{ 0 }, 100 ); expected[i++] = I{  0 };
	results[i] = kat::ipow<I>(I{ 1 },   0 ); expected[i++] = I{  1 };
	results[i] = kat::ipow<I>(I{ 1 },   1 ); expected[i++] = I{  1 };
	results[i] = kat::ipow<I>(I{ 1 },   2 ); expected[i++] = I{  1 };
	results[i] = kat::ipow<I>(I{ 1 }, 100 ); expected[i++] = I{  1 };
	results[i] = kat::ipow<I>(I{ 3 },   0 ); expected[i++] = I{  1 };
	results[i] = kat::ipow<I>(I{ 3 },   1 ); expected[i++] = I{  3 };
	results[i] = kat::ipow<I>(I{ 3 },   2 ); expected[i++] = I{  9 };
	results[i] = kat::ipow<I>(I{ 3 },   4 ); expected[i++] = I{ 81 };

	maybe_print("unsafe div_rounding_up");
	results[i] = kat::unsafe::div_rounding_up<I>( I{   0 }, I{   1 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{   0 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{   0 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{   1 }, I{   1 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{   1 }, I{   2 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{ 122 }, I{ 123 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{ 123 }, I{ 123 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::div_rounding_up<I>( I{ 124 }, I{ 123 } ); expected[i++] = I{   2 };

	maybe_print("div_rounding_up");
	results[i] = kat::div_rounding_up<I>( I{   0 }, I{   1 } ); expected[i++] = I{   0 };
	results[i] = kat::div_rounding_up<I>( I{   0 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::div_rounding_up<I>( I{   0 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::div_rounding_up<I>( I{   1 }, I{   1 } ); expected[i++] = I{   1 };
	results[i] = kat::div_rounding_up<I>( I{   1 }, I{   2 } ); expected[i++] = I{   1 };
	results[i] = kat::div_rounding_up<I>( I{ 122 }, I{ 123 } ); expected[i++] = I{   1 };
	results[i] = kat::div_rounding_up<I>( I{ 123 }, I{ 123 } ); expected[i++] = I{   1 };
	results[i] = kat::div_rounding_up<I>( I{ 124 }, I{ 123 } ); expected[i++] = I{   2 };
	results[i] = kat::div_rounding_up<I>( I{ 124 }, I{ 123 } ); expected[i++] = I{   2 };
	results[i] = kat::div_rounding_up<I>( std::numeric_limits<I>::max()    , std::numeric_limits<I>::max() - 1 ); expected[i++] = I{   2 };
	results[i] = kat::div_rounding_up<I>( std::numeric_limits<I>::max() - 1, std::numeric_limits<I>::max()     ); expected[i++] = I{   1 };

	maybe_print("round_down");
	results[i] = kat::round_down<I>( I{   0 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::round_down<I>( I{   0 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::round_down<I>( I{   1 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::round_down<I>( I{ 122 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::round_down<I>( I{ 123 }, I{ 123 } ); expected[i++] = I{ 123 };
	results[i] = kat::round_down<I>( I{ 124 }, I{ 123 } ); expected[i++] = I{ 123 };

	maybe_print("round_down_to_full_warps");
	results[i] = kat::round_down_to_full_warps<I>( I{   0 } ); expected[i++] = I{  0 };
	results[i] = kat::round_down_to_full_warps<I>( I{   1 } ); expected[i++] = I{  0 };
	results[i] = kat::round_down_to_full_warps<I>( I{   8 } ); expected[i++] = I{  0 };
	results[i] = kat::round_down_to_full_warps<I>( I{  16 } ); expected[i++] = I{  0 };
	results[i] = kat::round_down_to_full_warps<I>( I{  31 } ); expected[i++] = I{  0 };
	results[i] = kat::round_down_to_full_warps<I>( I{  32 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_down_to_full_warps<I>( I{  33 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_down_to_full_warps<I>( I{ 125 } ); expected[i++] = I{ 96 };

	// TODO: Consider testing rounding-up with negative dividends

	maybe_print("unsafe round_up");
	results[i] = kat::unsafe::round_up<I>( I{   0 }, I{   1 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::round_up<I>( I{   0 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::round_up<I>( I{   0 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::unsafe::round_up<I>( I{   1 }, I{   1 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::round_up<I>( I{   1 }, I{   2 } ); expected[i++] = I{   2 };
	results[i] = kat::unsafe::round_up<I>( I{  63 }, I{  64 } ); expected[i++] = I{  64 };
	results[i] = kat::unsafe::round_up<I>( I{  64 }, I{  64 } ); expected[i++] = I{  64 };
	results[i] = kat::unsafe::round_up<I>( I{  65 }, I{  32 } ); expected[i++] = I{  96 };

	maybe_print("round_up");
	results[i] = kat::round_up<I>( I{   0 }, I{   1 } ); expected[i++] = I{   0 };
	results[i] = kat::round_up<I>( I{   0 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::round_up<I>( I{   0 }, I{ 123 } ); expected[i++] = I{   0 };
	results[i] = kat::round_up<I>( I{   1 }, I{   1 } ); expected[i++] = I{   1 };
	results[i] = kat::round_up<I>( I{   1 }, I{   2 } ); expected[i++] = I{   2 };
	results[i] = kat::round_up<I>( I{  63 }, I{  64 } ); expected[i++] = I{  64 };
	results[i] = kat::round_up<I>( I{  64 }, I{  64 } ); expected[i++] = I{  64 };
	results[i] = kat::round_up<I>( I{  65 }, I{  32 } ); expected[i++] = I{  96 };
	results[i] = kat::round_up<I>( std::numeric_limits<I>::max() - 1, std::numeric_limits<I>::max() ); expected[i++] = I{ std::numeric_limits<I>::max() };

	maybe_print("round_down_to_power_of_2");
	results[i] = kat::round_down_to_power_of_2<I>( I{   1 }, I{   1 } ); expected[i++] = I{   1 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   2 }, I{   1 } ); expected[i++] = I{   2 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   3 }, I{   1 } ); expected[i++] = I{   3 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   4 }, I{   1 } ); expected[i++] = I{   4 };
	results[i] = kat::round_down_to_power_of_2<I>( I{ 123 }, I{   1 } ); expected[i++] = I{ 123 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   1 }, I{   2 } ); expected[i++] = I{   0 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   2 }, I{   2 } ); expected[i++] = I{   2 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   3 }, I{   2 } ); expected[i++] = I{   2 };
	results[i] = kat::round_down_to_power_of_2<I>( I{   4 }, I{   2 } ); expected[i++] = I{   4 };
	results[i] = kat::round_down_to_power_of_2<I>( I{ 123 }, I{   2 } ); expected[i++] = I{ 122 };

	maybe_print("round_up_to_power_of_2");
	results[i] = kat::round_up_to_power_of_2<I>( I{  1 }, I{  1 } ); expected[i++] = I{   1 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  2 }, I{  1 } ); expected[i++] = I{   2 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  3 }, I{  1 } ); expected[i++] = I{   3 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  4 }, I{  1 } ); expected[i++] = I{   4 };
	results[i] = kat::round_up_to_power_of_2<I>( I{ 23 }, I{  1 } ); expected[i++] = I{  23 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  1 }, I{  2 } ); expected[i++] = I{   2 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  2 }, I{  2 } ); expected[i++] = I{   2 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  3 }, I{  2 } ); expected[i++] = I{   4 };
	results[i] = kat::round_up_to_power_of_2<I>( I{  4 }, I{  2 } ); expected[i++] = I{   4 };
	results[i] = kat::round_up_to_power_of_2<I>( I{ 63 }, I{  2 } ); expected[i++] = I{  64 };

	maybe_print("unsafe round_up_to_power_of_2");
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  1 }, I{  1 } ); expected[i++] = I{   1 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  2 }, I{  1 } ); expected[i++] = I{   2 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  3 }, I{  1 } ); expected[i++] = I{   3 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  4 }, I{  1 } ); expected[i++] = I{   4 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{ 23 }, I{  1 } ); expected[i++] = I{  23 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  1 }, I{  2 } ); expected[i++] = I{   2 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  2 }, I{  2 } ); expected[i++] = I{   2 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  3 }, I{  2 } ); expected[i++] = I{   4 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{  4 }, I{  2 } ); expected[i++] = I{   4 };
	results[i] = kat::unsafe::round_up_to_power_of_2<I>( I{ 63 }, I{  2 } ); expected[i++] = I{  64 };

	maybe_print("round_up_to_full_warps");
	results[i] = kat::round_up_to_full_warps<I>( I{   0 } ); expected[i++] = I{  0 };
	results[i] = kat::round_up_to_full_warps<I>( I{   1 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_up_to_full_warps<I>( I{   8 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_up_to_full_warps<I>( I{  16 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_up_to_full_warps<I>( I{  31 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_up_to_full_warps<I>( I{  32 } ); expected[i++] = I{ 32 };
	results[i] = kat::round_up_to_full_warps<I>( I{  33 } ); expected[i++] = I{ 64 };
	results[i] = kat::round_up_to_full_warps<I>( I{  63 } ); expected[i++] = I{ 64 };

	maybe_print("gcd");
	results[i] = kat::gcd<I>( I{   1 }, I{   1 } ); expected[i++] = I{  1 };
	results[i] = kat::gcd<I>( I{   2 }, I{   1 } ); expected[i++] = I{  1 };
	results[i] = kat::gcd<I>( I{   1 }, I{   2 } ); expected[i++] = I{  1 };
	results[i] = kat::gcd<I>( I{   2 }, I{   2 } ); expected[i++] = I{  2 };
	results[i] = kat::gcd<I>( I{   8 }, I{   4 } ); expected[i++] = I{  4 };
	results[i] = kat::gcd<I>( I{   4 }, I{   8 } ); expected[i++] = I{  4 };
	results[i] = kat::gcd<I>( I{  10 }, I{   6 } ); expected[i++] = I{  2 };
	results[i] = kat::gcd<I>( I{ 120 }, I{  70 } ); expected[i++] = I{ 10 };
	results[i] = kat::gcd<I>( I{  70 }, I{ 120 } ); expected[i++] = I{ 10 };
	results[i] = kat::gcd<I>( I{  97 }, I{ 120 } ); expected[i++] = I{  1 };

	maybe_print("lcm");
	results[i] = kat::lcm<I>( I{   1 }, I{   1 } ); expected[i++] = I{  1 };
	results[i] = kat::lcm<I>( I{   2 }, I{   1 } ); expected[i++] = I{  2 };
	results[i] = kat::lcm<I>( I{   1 }, I{   2 } ); expected[i++] = I{  2 };
	results[i] = kat::lcm<I>( I{   2 }, I{   2 } ); expected[i++] = I{  2 };
	results[i] = kat::lcm<I>( I{   5 }, I{   3 } ); expected[i++] = I{ 15 };
	results[i] = kat::lcm<I>( I{   8 }, I{   4 } ); expected[i++] = I{  8 };
	results[i] = kat::lcm<I>( I{   4 }, I{   8 } ); expected[i++] = I{  8 };
	results[i] = kat::lcm<I>( I{  10 }, I{   6 } ); expected[i++] = I{ 30 };

	maybe_print("is_even");
	results[i] = kat::is_even<I>( I{   0 } ); expected[i++] = true;
	results[i] = kat::is_even<I>( I{   1 } ); expected[i++] = false;
	results[i] = kat::is_even<I>( I{   2 } ); expected[i++] = true;
	results[i] = kat::is_even<I>( I{   3 } ); expected[i++] = false;
	results[i] = kat::is_even<I>( I{ 123 } ); expected[i++] = false;
	results[i] = kat::is_even<I>( I{ 124 } ); expected[i++] = true;

	maybe_print("is_odd");
	results[i] = kat::is_odd<I>( I{   0 } ); expected[i++] = false;
	results[i] = kat::is_odd<I>( I{   1 } ); expected[i++] = true;
	results[i] = kat::is_odd<I>( I{   2 } ); expected[i++] = false;
	results[i] = kat::is_odd<I>( I{   3 } ); expected[i++] = true;
	results[i] = kat::is_odd<I>( I{ 123 } ); expected[i++] = true;
	results[i] = kat::is_odd<I>( I{ 124 } ); expected[i++] = false;

	maybe_print("log2");
	results[i] = kat::log2<I>( I{   1 } ); expected[i++] = 0;
	results[i] = kat::log2<I>( I{   2 } ); expected[i++] = 1;
	results[i] = kat::log2<I>( I{   3 } ); expected[i++] = 1;
	results[i] = kat::log2<I>( I{   4 } ); expected[i++] = 2;
	results[i] = kat::log2<I>( I{   6 } ); expected[i++] = 2;
	results[i] = kat::log2<I>( I{   7 } ); expected[i++] = 2;
	results[i] = kat::log2<I>( I{   8 } ); expected[i++] = 3;
	results[i] = kat::log2<I>( I{ 127 } ); expected[i++] = 6;

//	We don't have a goot integer sqrt() implementation to offer here. Perhaps
//	we could offer something based on casting to float?
//
//	results[i] = kat::sqrt<I>( I{   0 } ); expected[i++] =  0;
//	results[i] = kat::sqrt<I>( I{   1 } ); expected[i++] =  1;
//	results[i] = kat::sqrt<I>( I{   2 } ); expected[i++] =  1;
//	results[i] = kat::sqrt<I>( I{   3 } ); expected[i++] =  1;
//	results[i] = kat::sqrt<I>( I{   4 } ); expected[i++] =  2;
//	results[i] = kat::sqrt<I>( I{   5 } ); expected[i++] =  2;
//	results[i] = kat::sqrt<I>( I{   9 } ); expected[i++] =  3;
//	results[i] = kat::sqrt<I>( I{  10 } ); expected[i++] =  3;
//	results[i] = kat::sqrt<I>( I{ 127 } ); expected[i++] = 11;

	maybe_print("div_by_power_of_2");
	results[i] = kat::div_by_power_of_2<I>( I{   0 }, I {  1 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{   1 }, I {  1 }); expected[i++] = I{   1 };
	results[i] = kat::div_by_power_of_2<I>( I{ 111 }, I {  1 }); expected[i++] = I{ 111 };
	results[i] = kat::div_by_power_of_2<I>( I{   0 }, I {  2 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{   1 }, I {  2 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{   2 }, I {  2 }); expected[i++] = I{   1 };
	results[i] = kat::div_by_power_of_2<I>( I{   3 }, I {  2 }); expected[i++] = I{   1 };
	results[i] = kat::div_by_power_of_2<I>( I{   4 }, I {  2 }); expected[i++] = I{   2 };
	results[i] = kat::div_by_power_of_2<I>( I{ 111 }, I {  2 }); expected[i++] = I{  55 };
	results[i] = kat::div_by_power_of_2<I>( I{   0 }, I { 16 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{   1 }, I { 16 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{  15 }, I { 16 }); expected[i++] = I{   0 };
	results[i] = kat::div_by_power_of_2<I>( I{  16 }, I { 16 }); expected[i++] = I{   1 };
	results[i] = kat::div_by_power_of_2<I>( I{  17 }, I { 16 }); expected[i++] = I{   1 };
	results[i] = kat::div_by_power_of_2<I>( I{  32 }, I { 16 }); expected[i++] = I{   2 };
	results[i] = kat::div_by_power_of_2<I>( I{ 111 }, I { 16 }); expected[i++] = I{   6 };

	maybe_print("divides");
	results[i] = kat::divides<I>( I{   1 }, I{   0 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   2 }, I{   0 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   3 }, I{   0 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   1 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   2 }, I{   1 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   3 }, I{   1 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   1 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   2 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{   3 }, I{   2 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   4 }, I{   2 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   6 }, I{   9 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   9 }, I{   6 } ); expected[i++] = false;
	results[i] = kat::divides<I>( I{   4 }, I{  24 } ); expected[i++] = true;
	results[i] = kat::divides<I>( I{  24 }, I{   4 } ); expected[i++] = false;

	maybe_print("is_divisible_by");
	results[i] = kat::is_divisible_by<I>( I{   0 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   0 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   0 }, I{   3 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   1 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   1 }, I{   2 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{   1 }, I{   3 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{   2 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   2 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   2 }, I{   3 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{   2 }, I{   4 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{   9 }, I{   6 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{   6 }, I{   9 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by<I>( I{  24 }, I{   4 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by<I>( I{   4 }, I{  24 } ); expected[i++] = false;

	maybe_print("is_divisible_by_power_of_2");
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   0 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   0 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   1 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   1 }, I{   2 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   2 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   2 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{   2 }, I{   4 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{  24 }, I{   4 } ); expected[i++] = true;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{  72 }, I{  16 } ); expected[i++] = false;
	results[i] = kat::is_divisible_by_power_of_2<I>( I{  64 }, I{  16 } ); expected[i++] = true;

	maybe_print("power_of_2_divides");
	results[i] = kat::power_of_2_divides<I>( I{   1 }, I{   0 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{   2 }, I{   0 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{   1 }, I{   1 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{   2 }, I{   1 } ); expected[i++] = false;
	results[i] = kat::power_of_2_divides<I>( I{   1 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{   2 }, I{   2 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{   4 }, I{   2 } ); expected[i++] = false;
	results[i] = kat::power_of_2_divides<I>( I{   4 }, I{  24 } ); expected[i++] = true;
	results[i] = kat::power_of_2_divides<I>( I{  16 }, I{  72 } ); expected[i++] = false;
	results[i] = kat::power_of_2_divides<I>( I{  16 }, I{  64 } ); expected[i++] = true;

	maybe_print("log2_of_power_of_2");
	results[i] = kat::log2_of_power_of_2<I>( I{  1 } ); expected[i++] = I{ 0 };
	results[i] = kat::log2_of_power_of_2<I>( I{  2 } ); expected[i++] = I{ 1 };
	results[i] = kat::log2_of_power_of_2<I>( I{  4 } ); expected[i++] = I{ 2 };
	results[i] = kat::log2_of_power_of_2<I>( I{  8 } ); expected[i++] = I{ 3 };
	results[i] = kat::log2_of_power_of_2<I>( I{ 16 } ); expected[i++] = I{ 4 };
	results[i] = kat::log2_of_power_of_2<I>( I{ 32 } ); expected[i++] = I{ 5 };
	results[i] = kat::log2_of_power_of_2<I>( I{ 64 } ); expected[i++] = I{ 6 };

	maybe_print("modulo_power_of_2");
	results[i] = kat::modulo_power_of_2<I>( I{   0 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   1 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   2 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   3 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   4 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   5 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{  63 }, I{   1 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   0 }, I{   2 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   1 }, I{   2 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{   2 }, I{   2 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   3 }, I{   2 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{   4 }, I{   2 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   5 }, I{   2 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{  63 }, I{   2 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{   0 }, I{   4 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   1 }, I{   4 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{   2 }, I{   4 } ); expected[i++] = I{ 2 };
	results[i] = kat::modulo_power_of_2<I>( I{   3 }, I{   4 } ); expected[i++] = I{ 3 };
	results[i] = kat::modulo_power_of_2<I>( I{   4 }, I{   4 } ); expected[i++] = I{ 0 };
	results[i] = kat::modulo_power_of_2<I>( I{   5 }, I{   4 } ); expected[i++] = I{ 1 };
	results[i] = kat::modulo_power_of_2<I>( I{  63 }, I{   4 } ); expected[i++] = I{ 3 };

#define NUM_TEST_LINES 268

}

} // namespace kernels

// TODO:
// * Test between_or_equal and strictly_between with differing types for all 3 arguments
// * Some floating-point tests
// * gcd tests with values of different types
// * Some tests with negative values

#define INSTANTIATE_CONSTEXPR_MATH_TEST(_tp) \
	compile_time_execution_results<_tp> UNIQUE_IDENTIFIER(test_struct_); \

#define INTEGER_TYPES \
	int8_t, int16_t, int32_t, int64_t, \
	uint8_t, uint16_t, uint32_t, uint64_t, \
	char, short, int, long, long long, \
	signed char, signed short, signed int, signed long, signed long long, \
	unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long



TEST_SUITE("math") {

TEST_CASE_TEMPLATE("run-time on-device", I, INTEGER_TYPES)
{
	cuda::device_t<> device { cuda::device::current::get() };
	auto block_size { 1 };
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	auto device_side_results { cuda::memory::device::make_unique<I[]>(device, NUM_TEST_LINES) };
	auto device_side_expected_results { cuda::memory::device::make_unique<I[]>(device, NUM_TEST_LINES) };
	auto host_side_results { std::unique_ptr<I[]>(new I[NUM_TEST_LINES]) };
	auto host_side_expected_results { std::unique_ptr<I[]>(new I[NUM_TEST_LINES]) };

	cuda::launch(
		kernels::try_out_integral_math_functions<I>,
		launch_config,
		device_side_results.get(), device_side_expected_results.get());

	cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(I) * NUM_TEST_LINES);
	cuda::memory::copy(host_side_expected_results.get(), device_side_expected_results.get(), sizeof(I) * NUM_TEST_LINES);

	for(auto i { 0 }; i < NUM_TEST_LINES; i++) {
		CHECK(host_side_results.get()[i] == host_side_expected_results.get()[i]);
		if (host_side_results.get()[i] != host_side_expected_results.get()[i]) {
			MESSAGE("index of failure was: " << i);
		}
	}
}

} // TEST_SUITE("math")
