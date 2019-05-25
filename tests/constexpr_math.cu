#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include "utilities.cuh"
//#include "../src/kat/on_device/constexpr_math.cuh"
#include <kat/on_device/constexpr_math.cuh>
//#include "../external/doctest/doctest.h"
#include <limits>

namespace kernels {
} // namespace kernels

template <typename I>
struct compile_time_execution_results {

// TODO: What about invalid arguments, e.g.

    static_assert(kat::is_power_of_2<I>(I{ 1}) == true,  "kat::is_power_of_2( 1) error");
    static_assert(kat::is_power_of_2<I>(I{ 2}) == true,  "kat::is_power_of_2( 2) error");
    static_assert(kat::is_power_of_2<I>(I{ 4}) == true,  "kat::is_power_of_2( 4) error");
    static_assert(kat::is_power_of_2<I>(I{ 7}) == false, "kat::is_power_of_2( 7) error");
    static_assert(kat::is_power_of_2<I>(I{32}) == true,  "kat::is_power_of_2(32) error");
    static_assert(kat::is_power_of_2<I>(I{33}) == false, "kat::is_power_of_2(33) error");

    static_assert(kat::constexpr_::modular_inc<I>(I{ 0}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 1}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 0}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 1}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 2}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 3}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_inc error");
    static_assert(kat::constexpr_::modular_inc<I>(I{ 4}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_inc error");

    static_assert(kat::constexpr_::modular_dec<I>(I{ 0}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 1}, I{ 1}) == I{ 0 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 0}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 1}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 2}, I{ 3}) == I{ 1 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 3}, I{ 3}) == I{ 2 }, "kat::constexpr_::modular_dec error");
    static_assert(kat::constexpr_::modular_dec<I>(I{ 4}, I{ 3}) == I{ 0 }, "kat::constexpr_::modular_dec error");

    static_assert(kat::ipow<I>(I{ 0 },   1 ) == I{  0 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 0 },   2 ) == I{  0 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 0 }, 100 ) == I{  0 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 1 },   0 ) == I{  1 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 1 },   1 ) == I{  1 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 1 },   2 ) == I{  1 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 1 }, 100 ) == I{  1 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 3 },   0 ) == I{  1 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 3 },   1 ) == I{  3 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 3 },   2 ) == I{  9 }, "kat::ipow error");
    static_assert(kat::ipow<I>(I{ 3 },   4 ) == I{ 81 }, "kat::ipow error");
};

compile_time_execution_results<int8_t  > s8;
compile_time_execution_results<int16_t > s16;
compile_time_execution_results<int32_t > s32;
compile_time_execution_results<int64_t > s64;
compile_time_execution_results<uint8_t > u8;
compile_time_execution_results<uint16_t> u16;
compile_time_execution_results<uint32_t> u32;
compile_time_execution_results<uint64_t> u64;

TEST_SUITE("constexpr_math") {
/*
is_power_of_2(T val)
constexpr inline T& modular_inc(T& x, T modulus)
constexpr inline T& modular_dec(T& x, T modulus)
constexpr I ipow(I base, unsigned exponent)

I div_rounding_up_unsafe(I dividend, const I2 divisor)
I div_rounding_up_safe(I dividend, const I2 divisor)
I round_down(const I x, const I2 y)
I round_down_to_warp_size(I x)
I round_up_unsafe(I x, I2 y)
I round_down_to_power_of_2(I x, I2 power_of_2)
I round_up_to_power_of_2_unsafe(I x, I2 power_of_2)
I round_up_to_full_warps_unsafe(I x)
constexpr inline bool between_or_equal(const T& x, const Lower& l, const Upper& u)
constexpr inline bool strictly_between(const T& x, const Lower& l, const Upper& u)
#if __cplusplus >= 201402L
T gcd(T u, T v)
#endif
int constexpr_::log2(I val)
I constexpr_::div_by_fixed_power_of_2(I dividend)
typename std::common_type<S,T>::type constexpr_::gcd(S u, T v)
typename std::common_type<S,T>::type constexpr_::lcm(S u, T v)
T constexpr_::sqrt(T& x)
I div_by_power_of_2(I dividend, I divisor)
I div_by_fixed_power_of_2_rounding_up(I dividend)
I num_warp_sizes_to_cover(I x)
bool divides(I divisor, I dividend)
constexpr inline int is_divisible_by(I dividend, I divisor)
constexpr inline bool is_divisible_by_power_of_2(I dividend, I divisor)
bool is_odd(I x)  { return x & I{0x1} != I{0}; }
bool is_even(I x) { return x & I{0x1} == I{0}; }
*/


TEST_CASE_TEMPLATE("run-time on-host", T, int32_t, int64_t, float, double)
{
    (void) 0; // Don't need to do anything

}
/*

TEST_CASE_TEMPLATE("run-time on-device", T, int32_t, int64_t, float, double)
{
*/
/*
    cuda::device_t<> device { cuda::device::current::get() };
    auto num_warps { 10 };
    auto num_blocks { 10 };
    auto block_size = num_warps * warp_size;
    auto launch_config { cuda::make_launch_config(num_blocks, block_size, 0 };
//    auto device_side_results { cuda::memory::device::make_unique<shmem_size_t[]>(device, num_warps) };
//    auto host_side_results { std::unique_ptr<shmem_size_t[]>(new shmem_size_t[num_warps]) };
    cuda::launch(kernels::test_math_function_at_run_time<T>, launch_config);
//    cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(shmem_size_t) * num_warps);
    CHECK_NOTHROW(device.synchronize());
// To do: Should we get the results back? probably not
*//*

}

TEST_CASE_TEMPLATE("compile-time on host", I, int32_t, int64_t)
{

}

TEST_CASE_TEMPLATE("compile-time on device", I, int32_t, int64_t)
{

}
*/

} // TEST_SUITE("constexpr_math")
