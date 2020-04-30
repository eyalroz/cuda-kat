#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"
#include "util/woodruff_int128_t.hpp"
#include "util/woodruff_uint128_t.hpp"
#include "util/cpu_builtin_equivalents.hpp"

#include <kat/on_device/builtins.cuh>
#include <kat/on_device/non-builtins.cuh>
#include <kat/on_device/collaboration/block.cuh>

using std::size_t;

#if __cplusplus < 201701L
#include <experimental/optional>
template <typename T>
using optional = std::experimental::optional<T>;
#else
template <typename T>
#include <optional>
using optional = std::optional<T>;
#endif

template <typename T>
const auto make_exact_comparison { optional<T>{} };

namespace device_function_ptrs {

#define PREPEND_TYPENAME_IDENTIFIER(t) , typename t

#define PREPARE_BUILTIN_INNER(subnamespace, builtin_function_basename, t1, t2) \
	template <t1> \
	struct builtin_function_basename { \
		static const void* const ptr; \
		static const char* const name; \
	}; \
    \
	template <t1> \
	const void* const builtin_function_basename<t2>::ptr { (void *) kat::subnamespace::builtin_function_basename<t2> }; \
	template <t1> \
	const char* const builtin_function_basename<t2>::name { STRINGIZE(kat::subnamespace::builtin_function_basename) }

#define COMMA ,
#define PREPARE_BUILTIN0(subnamespace, builtin_function_basename) \
	struct builtin_function_basename { \
		static const void* const ptr; \
		static const char* const name; \
	}; \
	\
	const void* const builtin_function_basename::ptr { (void *) kat::subnamespace::builtin_function_basename };  \
	const char* const builtin_function_basename::name { STRINGIZE(kat::subnamespace::builtin_function_basename) }

#define PREPARE_BUILTIN1(subnamespace, builtin_function_basename) PREPARE_BUILTIN_INNER(subnamespace, builtin_function_basename, typename T, T)
#define PREPARE_BUILTIN2(subnamespace, builtin_function_basename) PREPARE_BUILTIN_INNER(subnamespace, builtin_function_basename, typename T1 COMMA typename T2, T1 COMMA T2)
#define PREPARE_BUILTIN3(subnamespace, builtin_function_basename) PREPARE_BUILTIN_INNER(subnamespace, builtin_function_basename, typename T1 COMMA typename T2 COMMA typename T3, T1 COMMA T2 COMMA T3)

#define INSTANTIATE_BUILTIN_VIA_PTR(builtin_function_basename, ...) \
	template struct builtin_function_basename<__VA_ARGS__>



PREPARE_BUILTIN1(builtins, multiplication_high_bits);
PREPARE_BUILTIN1(builtins, divide);
PREPARE_BUILTIN1(builtins, absolute_value);
PREPARE_BUILTIN1(builtins, minimum);
PREPARE_BUILTIN1(builtins, maximum);
PREPARE_BUILTIN1(builtins, sum_with_absolute_difference);
PREPARE_BUILTIN1(builtins, population_count);
PREPARE_BUILTIN1(builtins, bit_reverse);
PREPARE_BUILTIN1(builtins, find_leading_non_sign_bit);
PREPARE_BUILTIN1(builtins::bit_field, extract_bits);
PREPARE_BUILTIN1(builtins::bit_field, replace_bits);
PREPARE_BUILTIN0(builtins, permute_bytes);
// This function is special, in that one of its template parameters is a value rather than a type.
PREPARE_BUILTIN_INNER(builtins, funnel_shift_right, kat::builtins::funnel_shift_amount_resolution_mode_t AmountResolutionMode, AmountResolutionMode);
PREPARE_BUILTIN_INNER(builtins, funnel_shift_left, kat::builtins::funnel_shift_amount_resolution_mode_t AmountResolutionMode, AmountResolutionMode);

// PREPARE_BUILTIN0(builtins, funnel_shift);
PREPARE_BUILTIN1(builtins, average);
PREPARE_BUILTIN1(builtins, average_rounded_up);

PREPARE_BUILTIN0(builtins::special_registers, lane_index);
PREPARE_BUILTIN0(builtins::special_registers, symmetric_multiprocessor_index);
PREPARE_BUILTIN0(builtins::special_registers, grid_index);
PREPARE_BUILTIN0(builtins::special_registers, dynamic_shared_memory_size);
PREPARE_BUILTIN0(builtins::special_registers, total_shared_memory_size);


#if (__CUDACC_VER_MAJOR__ >= 9)
// Note: These three ballot functions were available before CUDA 9, but for
// now we're only testing the CUDA 9 versions.
PREPARE_BUILTIN0(builtins::warp, ballot);
PREPARE_BUILTIN0(builtins::warp, all_lanes_satisfy);
PREPARE_BUILTIN0(builtins::warp, any_lanes_satisfy);

#if ! defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700
PREPARE_BUILTIN0(builtins::warp, all_lanes_agree);
PREPARE_BUILTIN1(builtins::warp, propagate_mask_if_lanes_agree);
PREPARE_BUILTIN1(builtins::warp, propagate_mask_if_warp_agrees);
PREPARE_BUILTIN1(builtins::warp, get_matching_lanes);
#endif

#endif

PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, preceding);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, preceding_and_self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, succeeding_and_self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, succeeding);

PREPARE_BUILTIN1(non_builtins, find_first_set);
PREPARE_BUILTIN1(non_builtins, count_trailing_zeros);
PREPARE_BUILTIN1(non_builtins, count_leading_zeros);

} // namespace builtin_device_function_ptrs

namespace kernels {

template <typename DeviceFunctionHook, typename R, typename... Is>
__global__ void execute_testcases(
//	F                           f,
	size_t                      num_checks,
	fake_bool* __restrict__     execution_complete,
	R*         __restrict__     results,
	const Is*  __restrict__ ... inputs
	)
{
	auto global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
	auto check_index = global_thread_index;
	if (check_index >= num_checks) { return; }
	using device_function_type = auto (Is...) -> R;
	auto f = (device_function_type*) DeviceFunctionHook::ptr;
	// printf("function ptr is at %p and name is %s\n", f, DeviceFunctionHook::name);
	results[check_index] = f(inputs[check_index]...);
		// It's up to the author of f to ensure there aren't any runtime errors...
		// Also, f should not use any shared memory
//	printf("Thread %3u = (%2u,%2u), result %x, value %d, mask %x, sizeof...(inputs) = %u\n",
//		(unsigned) i, (unsigned) i / 32 , (unsigned) i % 32, results[i], inputs[i]... , (unsigned) sizeof...(inputs));
	execution_complete[check_index] = true;
}

} // namespace kernels


namespace detail {
template <typename T>
struct multiplication_result_helper { };

template <>
struct multiplication_result_helper<unsigned>{
	static_assert(sizeof(unsigned) == 4, "Unexpected size");
	static_assert(sizeof(unsigned long long) == 8, "Unexpected size");
	using type = unsigned long long;
};

template <>
struct multiplication_result_helper<long long>{
	static_assert(sizeof(long long) == 8, "Unexpected size");
#ifdef __SIZEOF_INT128__
	using type = __int128_t;
#else
#warning "Untrustworthy 128-bit int implementation."
	using type = int128_t;
#endif
};

template <>
struct multiplication_result_helper<unsigned long long>{
	static_assert(sizeof(unsigned long long) == 8, "Unexpected size");
#ifdef __SIZEOF_INT128__
	using type = __uint128_t;
#else
#warning "Untrustworthy 128-bit uint implementation."
	using type = uint128_t;
#endif
};

} // namespace detail

template <typename T>
using multiplication_result_t = typename detail::multiplication_result_helper<T>::type;

template <typename I>
I multiplication_high_bits(I lhs, I rhs)
{
	multiplication_result_t<I> lhs_ { lhs }, rhs_ { rhs };
	auto m = lhs_ * rhs_;
	auto high_bits = m >> ((unsigned) size_in_bits<I>());
//	std::cout << std::hex << lhs << " * " << rhs_ << " = " << m << std::dec
//		<< "; I has " << size_in_bits<I>() << " and m has " << size_in_bits(m) << ". high bits: " << high_bits << " after cast " << I(high_bits) << "\n";
	return I(high_bits);
}

template <typename T> struct empty { };


template <typename T>
std::size_t set_width_for_up_to(T max)
{
//	assert(std::is_integral<I>::value, "Only integer types supported for now");
	std::stringstream ss;
	ss << std::dec << max;
	return ss.str().length();
}

bool check_execution_indicators(
	size_t                    num_checks,
	const char*               testcase_name,
	fake_bool*                execution_indicators)
{
	std::stringstream ss;
	auto index_width = set_width_for_up_to(num_checks);
	bool all_executed { true };

	// TODO: Consider using the maximum/minimum result values to set field widths.

	for(size_t i = 0; i < num_checks; i++) {
		ss.str("");
		ss << "Failed executing testcase " << (i+1) << " for " << testcase_name;
		auto failure_message { ss.str() };
		CHECK_MESSAGE(execution_indicators[i], failure_message);
		all_executed = all_executed and execution_indicators[i];
	}
	return all_executed;
}

namespace detail {

template <typename T>
T tolerance_gadget(std::true_type, optional<T> x) { return x.value(); }


template <typename T>
int tolerance_gadget(std::false_type, optional<T>) { return 0; }

} // namespace detail

template <typename T>
std::conditional_t<std::is_arithmetic<T>::value, T, int>  tolerance_gadget(optional<T> x)
{
	constexpr const auto is_arithmetic = std::is_arithmetic< std::decay_t<T> >::value;
	return detail::tolerance_gadget(std::integral_constant<bool, is_arithmetic>{}, x);
}


// TODO: Take iterator templates rather than pointers
template <typename R, typename F, typename... Is>
void check_results(
	size_t                    num_checks,
	const char*               testcase_name,
	// perhaps add another parameter for specific individual-check details?
	const R*  __restrict__    actual_results,
	F                         expected_result_retriever,
	optional<R>               comparison_tolerance_fraction,
	const Is* __restrict__... inputs)
{
	std::stringstream ss;
	auto index_width = set_width_for_up_to(num_checks);

	// TODO: Consider using the maximum/minimum result values to set field widths.

	for(size_t i = 0; i < num_checks; i++) {
		ss.str("");
		ss
			<< "Assertion " << std::setw(index_width) << (i+1) << " for testcase " << testcase_name
			// << " :\n"
			<< "(" << std::make_tuple(inputs[i]...) << ")"
		;
		auto mismatch_message { ss.str() };

		if (comparison_tolerance_fraction) {
			auto tolerance = tolerance_gadget(comparison_tolerance_fraction);
				// With C++17, we could just use if constexpr and never try to compare against
				// a non-arithmetic type
			CHECK_MESSAGE(actual_results[i] ==  doctest::Approx(expected_result_retriever(i)).epsilon(tolerance), mismatch_message);
		}
		else {
			CHECK_MESSAGE(actual_results[i] == expected_result_retriever(i), mismatch_message);
		}
	}
}

template <typename T>
struct tag { };

/**
 * @brief Executes a testcase intended to make certain checks using a GPU kernel
 * which produces the values to check for.
 *
 * @note The actual checks are eventually conducted on the host side, since doctest
 * code can't actually do anything useful on the GPU. So on the GPU side we "merely"
 * compute the values to check and let the test logic peform the actual comparison later
 * on.
 */
template <typename K, typename R, typename... Is, size_t... Indices>
auto execute_testcase_on_gpu(
	tag<R>,
	std::index_sequence<Indices...>,
	K                                testcase_kernel,
	const char*                      testcase_name,
	cuda::launch_configuration_t     launch_config,
	size_t                           num_checks,
	Is* __restrict__ ...             inputs)
{
	cuda::device_t device { cuda::device::current::get() };
	auto device_side_results { cuda::memory::device::make_unique<R[]>(device, num_checks) };
	cuda::memory::device::zero(device_side_results.get(), num_checks * sizeof(R)); // just to be on the safe side
	auto device_side_execution_indicators { cuda::memory::device::make_unique<fake_bool[]>(device, num_checks * sizeof(fake_bool)) };
	cuda::memory::device::zero(device_side_execution_indicators.get(), num_checks * sizeof(fake_bool)); // this is actually necessary
	auto host_side_results { std::vector<R>(num_checks) };
	auto host_side_execution_indicators { std::vector<fake_bool>(num_checks) };

	auto make_device_side_input = [&device, num_checks](auto input, size_t n) {
		using input_type = std::remove_reference_t<decltype(*input)>;
		auto device_side_input = cuda::memory::device::make_unique<input_type[]>(device, n);
		cuda::memory::copy(device_side_input.get(), input, num_checks * sizeof(input_type));
		return std::move(device_side_input);
	};
	auto device_side_inputs = std::make_tuple( make_device_side_input(inputs, num_checks)... );
	ignore(device_side_inputs); // for the case of no inputs

	cuda::launch(
		testcase_kernel,
		launch_config,
		num_checks,
		device_side_execution_indicators.get(),
		device_side_results.get(),
		std::get<Indices>(device_side_inputs).get()... );

	cuda::memory::copy(host_side_results.data(), device_side_results.get(), sizeof(R) * num_checks);
	cuda::memory::copy(host_side_execution_indicators.data(), device_side_execution_indicators.get(), sizeof(bool) * num_checks);

	check_execution_indicators(num_checks, testcase_name, host_side_execution_indicators.data());
	return host_side_results;
}

template <typename K, typename R, typename... Is, size_t... Indices>
void execute_testcase_on_gpu_and_check(
	std::index_sequence<Indices...>  is,
	const R* __restrict__            expected_results,
	K                                testcase_kernel,
	const char*                      testcase_name,
	cuda::launch_configuration_t     launch_config,
	size_t                           num_checks,
	optional<R>                      comparison_tolerance_fraction,
	Is* __restrict__ ...             inputs)
{
	auto host_side_results = execute_testcase_on_gpu(
		tag<R>{},
		is,
		testcase_kernel,
		testcase_name,
		launch_config,
		num_checks,
		inputs...);

	auto expected_result_retriever = [&](size_t pos) { return expected_results[pos]; };

	check_results (
		num_checks,
		testcase_name,
		// perhaps add another parameter for specific testcase details?
		host_side_results.data(),
		expected_result_retriever,
		comparison_tolerance_fraction,
		inputs...);
}




template <typename DeviceFunctionHook, typename R, typename... Is>
void execute_uniform_builtin_testcase_on_gpu_and_check(
	DeviceFunctionHook     dfh,
	const R* __restrict__  expected_results,
	size_t                 num_checks,
	optional<R>            comparison_tolerance_fraction,
	Is* __restrict__ ...   inputs)
{
	auto block_size { 128 };
	auto num_grid_blocks { div_rounding_up(num_checks, block_size) };
	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };

	auto host_side_results = execute_testcase_on_gpu(
		tag<R>{},
		typename std::make_index_sequence<sizeof...(Is)> {},
		kernels::execute_testcases<DeviceFunctionHook, R, Is...>,
		DeviceFunctionHook::name,
		launch_config,
		num_checks,
		inputs...
	);

	auto expected_result_retriever = [&](size_t pos) { return expected_results[pos]; };

	check_results (
		num_checks,
		DeviceFunctionHook::name,
		// perhaps add another parameter for specific testcase details?
		host_side_results.data(),
		expected_result_retriever,
		comparison_tolerance_fraction,
		inputs...);

}

template <typename DeviceFunctionHook, typename R, typename... Is>
void execute_non_uniform_builtin_testcase_on_gpu_and_check(
	DeviceFunctionHook             dfh,
	const R* __restrict__          expected_results,
	size_t                         num_checks,
	cuda::grid::dimension_t        num_grid_blocks,
	cuda::grid::block_dimension_t  block_size,
	optional<R>                    comparison_tolerance_fraction,
	Is* __restrict__ ...           inputs)
{
	auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
	// TODO: Should we check that num_checks is equal to the number of grid threads?

	auto host_side_results = execute_testcase_on_gpu(
		tag<R>{},
		typename std::make_index_sequence<sizeof...(Is)> {},
		kernels::execute_testcases<DeviceFunctionHook, R, Is...>,
		DeviceFunctionHook::name,
		launch_config,
		num_checks,
		inputs...
	);

	auto expected_result_retriever = [&](size_t pos) { return expected_results[pos]; };

	check_results (
		num_checks,
		DeviceFunctionHook::name,
		// perhaps add another parameter for specific testcase details?
		host_side_results.data(),
		expected_result_retriever,
		comparison_tolerance_fraction,
		inputs...);
}


// Builtins whose behavior is uniform across all grid threads, and does not depend on data held by other threads
TEST_SUITE("uniform builtins") {

// Note: Types for instantiation are chosen based on what's actually available in CUDA

TEST_CASE_TEMPLATE("multiplication high bits", I, unsigned, long long, unsigned long long)
{
	using result_type = I;

	std::vector<result_type> expected_results;
	std::vector<I> lhs;
	std::vector<I> rhs;

	auto add_check = [&](I x, I y) {
		lhs.emplace_back(x);
		rhs.emplace_back(y);
//		std::cout << "testcase " << expected_results.size() + 1 << ": ";
		auto result = multiplication_high_bits(x, y);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto almost_sqrt = (I{1} << half_num_bits) - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;

	// Yields 0
	add_check(0, 0);
	add_check(1, 0);
	add_check(0, 1);
	add_check(1, 1);
	add_check(almost_sqrt, almost_sqrt);

	// Yields 1
	add_check(mid_bit_on, mid_bit_on);

	// Yields 6
	add_check(mid_bit_on * 2, mid_bit_on * 3);
	add_check(mid_bit_on * 3, mid_bit_on * 2);

	// Depends...
	add_check(min, min);
	add_check(min, max);
	add_check(max, min);
	add_check(max, max);

	auto num_checks = expected_results.size();

//	std::cout << "function is at " << (void *)(kat::builtins::multiplication_high_bits<I>) << std::endl;


	execute_uniform_builtin_testcase_on_gpu_and_check(
			device_function_ptrs::multiplication_high_bits<I>{}, // kat::builtins::multiplication_high_bits<I>,
			expected_results.data(),
			num_checks, make_exact_comparison<result_type>,
			lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("minimum", T, int, unsigned int, long, unsigned long, long long, unsigned long long, float , double)
{
	using result_type = T;
	std::vector<result_type> expected_results;
	std::vector<T> lhs;
	std::vector<T> rhs;

	auto add_check = [&](T x, T y) {
		lhs.emplace_back(x);
		rhs.emplace_back(y);
		auto result = std::min<T>(x, y);
			// Note: This is not a trivial choice! The behavior in edge cases,
			// like or near-equality for floating-point types, is not the same
			// among any two implementations of a "minimum()" function.
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<T>::max();
	constexpr const auto min = std::numeric_limits<T>::min();
	constexpr const auto half_num_bits = size_in_bits<T>() / 2;
	constexpr const auto half_num_bits_max_bits = (T{ uint64_t{1} << half_num_bits}) - 1;
	constexpr const auto mid_bit_on = T{uint64_t{1} << half_num_bits};
		// Note that for floating-point types, bit-counting is not that meaningful

	add_check(0, 0);
	add_check(1, 0);
	add_check(0, 1);
	add_check(1, 1);
	add_check(half_num_bits_max_bits, half_num_bits_max_bits);
	add_check(mid_bit_on, mid_bit_on);
	add_check(mid_bit_on * 2, mid_bit_on * 3);
	add_check(mid_bit_on * 3, mid_bit_on * 2);
	add_check(min, min);
	add_check(min, max);
	add_check(max, min);
	add_check(max, max);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::minimum<T>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("maximum", T, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double)
{
	std::vector<T> expected_results;
	std::vector<T> lhs;
	std::vector<T> rhs;

	auto add_check = [&](T x, T y) {
		lhs.emplace_back(x);
		rhs.emplace_back(y);
		auto result = std::max<T>(x, y);
			// Note: This is not a trivial choice! The behavior in edge cases,
			// like or near-equality for floating-point types, is not the same
			// among any two implementations of a "minimum()" function.
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<T>::max();
	constexpr const auto min = std::numeric_limits<T>::min();
	constexpr const auto half_num_bits = size_in_bits<T>() / 2;
	constexpr const auto half_num_bits_max_bits = (T{ uint64_t{1} << half_num_bits}) - 1;
	constexpr const auto mid_bit_on = T{uint64_t{1} << half_num_bits};
		// Note that for floating-point types, bit-counting is not that meaningful

	add_check(0, 0);
	add_check(1, 0);
	add_check(0, 1);
	add_check(1, 1);
	add_check(half_num_bits_max_bits, half_num_bits_max_bits);
	add_check(mid_bit_on, mid_bit_on);
	add_check(mid_bit_on * 2, mid_bit_on * 3);
	add_check(mid_bit_on * 3, mid_bit_on * 2);
	add_check(min, min);
	add_check(min, max);
	add_check(max, min);
	add_check(max, max);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::maximum<T>{},
		expected_results.data(), num_checks,
		make_exact_comparison<T>,
		lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("absolute_value", T, int, long, long long, float, double, unsigned char, unsigned short, unsigned, unsigned long, unsigned long long)
{
	std::vector<T> expected_results;
	std::vector<T> values;

	auto add_check = [&](T x) {
		values.emplace_back(x);
		auto result = absolute_value(x);
		expected_results.emplace_back(result);
	};

	add_check(0);
	add_check(1);
	add_check(10);
	add_check(T(-1));
	add_check(T(-10));
	add_check(std::numeric_limits<T>::max());
	add_check(std::numeric_limits<T>::min());

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::absolute_value<T>{},
		expected_results.data(), num_checks,
		make_exact_comparison<T>,
		values.data());
}

TEST_CASE_TEMPLATE("divide", T, float, double)
{
	std::vector<T> expected_results;
	std::vector<T> dividends;
	std::vector<T> divisors;

	auto add_check = [&](T x, T y) {
		dividends.emplace_back(x);
		divisors.emplace_back(y);
		auto result = x / y;
			// Note: This is not a trivial choice - it depends on the exact floating-point
			// implementation on the CPU; and rounding choices...
		expected_results.emplace_back(result);
	};

	// constexpr const auto max = std::numeric_limits<T>::max();
	// constexpr const auto min = std::numeric_limits<T>::min();
	constexpr const auto infinity = std::numeric_limits<T>::infinity();

	// Should yield 0
	add_check(0, 1);
	add_check(0, 2);
	add_check(0, infinity);

	// This fails: We get nan's but should get 0
	// add_check(0, max);
	// add_check(0, min);


	add_check(T{0.5696892130}, T{0.0300253556});
	add_check(T{0.8300151169975111343}, T{0.99338683191717680375});

	// TODO: More testcases wouldn't hurt

	auto num_checks = expected_results.size();

	optional<T> comparison_tolerance_fraction { 1e-6 };
	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::divide<T>{},
		expected_results.data(), num_checks,
		comparison_tolerance_fraction,
		dividends.data(), divisors.data());
}

TEST_CASE_TEMPLATE("sum_with_absolute_difference", I, int16_t, int32_t ,int64_t, uint16_t, uint32_t, uint64_t)
{
	using uint_t = std::make_unsigned_t<I>;
	using result_type = uint_t;
	std::vector<result_type> expected_results;
	std::vector<uint_t> addends;
	std::vector<I> x_values;
	std::vector<I> y_values;

	auto add_check = [&](I x, I y, uint_t addend) {
		x_values.emplace_back(x);
		y_values.emplace_back(y);
		addends.emplace_back(addend);
//		std::cout << "Testcase " << (x_values.size()+1) << " : absolute_difference(x, y) = " << absolute_difference(x, y)
//			<< ", addend + I(absolute_difference(x, y)) = " << addend + I(absolute_difference(x, y)) << '\n';
		auto result = addend + I(absolute_difference(x, y));
			// The non-trivial choice here - conversion from the difference type to unsigned
		expected_results.emplace_back(result);
	};

	constexpr const auto max_uint = std::numeric_limits<uint_t>::max();
	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	// constexpr const auto half_num_bits_max_bits = (I{1} << half_num_bits) - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;

	// Should yield 0
	// ... but be careful - if you try to check some of these values in mid-flight
	// you might get stung by integer promotion. I know I have :-(
	add_check(I(0), I(0), 0);
	add_check(I(1), I(1), 0);
	add_check(I(min), I(min), 0);
	add_check(I(mid_bit_on), I(mid_bit_on), 0);
	add_check(I(1), I(0), max_uint);
	add_check(I(0), I(1), max_uint);

	// Should yield 1 << 15 int16_t, 1 << 16 for uint16_t, 0 otherwise
	add_check(I(max), I(0), 1);

	// Should yield 1 for unsigned, 0 for 32-bit types, maybe also for 64-bit types
	add_check(I(0), I(min), 1);

	// Should yield:
	// 1 << 16 for int16_t and uint16_t (I think)
	// 0 for int32_t and uint32_t
	// 0 for int64_t and uint64_t
	add_check(I(max), I(min), 1);

	// Should yield 123
	add_check(I(max), I(max), 123);
	add_check(I(min), I(min), 123);
	add_check(I(mid_bit_on), I(mid_bit_on), 123);

	// Should yield 2 * mid_bit_on + 1 for all
	add_check(I(mid_bit_on), I(-mid_bit_on), 1);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::sum_with_absolute_difference<I>{},
		expected_results.data(),
		num_checks,
		make_exact_comparison<result_type>,
		x_values.data(),
		y_values.data(),
		addends.data()
	);
}


TEST_CASE_TEMPLATE("population_count", I, uint8_t, uint16_t, uint32_t, uint64_t)
{
	using result_type = int;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x) {
		values.emplace_back(x);
		auto result = population_count<I>(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;


	add_check(0);
	add_check(1);
	add_check(2);
	add_check(3);
	add_check(4);
	add_check(8);
	add_check(16);
	add_check(31);
	add_check(32);
	add_check(33);
	add_check(max);
	add_check(mid_bit_on - 1);
	add_check(mid_bit_on);
	add_check(mid_bit_on + 1);
	add_check(max - 1);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::population_count<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}

TEST_CASE_TEMPLATE("bit_reverse", I, uint32_t, uint64_t, unsigned long)
{
	using result_type = I;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x) {
		values.emplace_back(x);
		auto result = bit_reverse<I>(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;


	add_check(0);
	add_check(0b1);
	add_check(0b10);
	add_check(0b11);
	add_check(0b101);
	add_check(mid_bit_on - 1);
	add_check(mid_bit_on);
	add_check(mid_bit_on + 1);
	add_check(~ (mid_bit_on - 1));
	add_check(~ (mid_bit_on));
	add_check(~ (mid_bit_on + 1));
	add_check(max - 1);
	add_check(max);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::bit_reverse<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}


TEST_CASE_TEMPLATE("find_leading_non_sign_bit", I, int, unsigned, long, unsigned long, long long, unsigned long long)
{
	using result_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x, result_type result) {
		values.emplace_back(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto mid_bit_index = half_num_bits - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;
	constexpr const auto no_nonsign_bits { std::numeric_limits<uint32_t>::max() };
	constexpr const auto msb_index { size_in_bits<I>() - 1 };


	add_check(0,                  no_nonsign_bits);
	add_check(0b1,                0);
	add_check(0b10,               1);
	add_check(0b11,               1);
	add_check(0b101,              2);
	add_check(mid_bit_on - 1,     half_num_bits - 1);
	add_check(mid_bit_on,         half_num_bits);
	add_check(mid_bit_on + 1,     half_num_bits);

	if (std::is_unsigned<I>::value) {
		add_check(I(~ (mid_bit_on - 1)), msb_index);
		add_check(I(~ (mid_bit_on)),     msb_index);
		add_check(I(~ (mid_bit_on + 1)), msb_index);
		add_check(max - 1,            msb_index);
		add_check(max,                msb_index);
	}
	else {
		add_check(I(-0b1),            no_nonsign_bits);
		add_check(I(-0b1010),         3);
		add_check(~ (mid_bit_on - 1), mid_bit_index);
		add_check(~ (mid_bit_on),     mid_bit_index + 1);
		add_check(~ (mid_bit_on + 1), mid_bit_index + 1);
		add_check(max - 1,            msb_index - 1);
		add_check(max,                msb_index - 1);
		add_check(min,                msb_index - 1);
		add_check(min + 1,            msb_index - 1);
		add_check(min + 2,            msb_index - 1);
	}

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::find_leading_non_sign_bit<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}

// Not testing ldg/load_global_with_non_coherent_cache here, since such a test is too dissimilar from the
// rest of the builtin tests.

TEST_CASE("select_bytes")
{
	using result_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<uint32_t> low_words;
	std::vector<uint32_t> high_words;
	std::vector<uint32_t> selectors_words;

	auto add_check = [&](uint32_t x, uint32_t y, uint32_t selectors, uint32_t result) {
		low_words.emplace_back(x);
		high_words.emplace_back(y);
		selectors_words.emplace_back(selectors);
		expected_results.emplace_back(result);
	};

	auto make_selector =
		[](
			unsigned first_byte,
			unsigned second_byte,
			unsigned third_byte,
			unsigned fourth_byte)
		{
			auto selectors_are_valid = (first_byte <= 0xF) and (second_byte <= 0xF) and (third_byte <= 0xF) and (fourth_byte <= 0xF);
			REQUIRE(selectors_are_valid); // { throw std::invalid_argument("Invalid byte selectors for PTX prmt"); }
			return
				   first_byte
				| (second_byte <<  4)
				| (third_byte  <<  8)
				| (fourth_byte << 12);
		};

	constexpr const auto replicate_sign  { 0b1000 };
	// constexpr const auto copy_value { 0b0000 };


	add_check(0x33221100, 0x77665544, make_selector(0,0,0,0), 0 );
	add_check(0x33221100, 0x77665544, make_selector(1,1,1,1), 0x11111111 );
	add_check(0x33221100, 0x77665544, make_selector(2,2,2,2), 0x22222222 );
	add_check(0x33221100, 0x77665544, make_selector(3,3,3,3), 0x33333333 );

	add_check(0x33221100, 0x77665544, make_selector(4,4,4,4), 0x44444444 );
	add_check(0x33221100, 0x77665544, make_selector(5,5,5,5), 0x55555555 );
	add_check(0x33221100, 0x77665544, make_selector(6,6,6,6), 0x66666666 );
	add_check(0x33221100, 0x77665544, make_selector(0,1,2,3), 0x33221100 );

	add_check(0x33221100, 0x77665544, make_selector(3,2,1,0), 0x00112233 );
	add_check(0x33221100, 0x77665544, make_selector(7,6,5,4), 0x44556677 );
	add_check(0x33221100, 0x77665544, make_selector(2,3,4,5), 0x55443322 );
	add_check(0x00000000, 0x00000000, make_selector(0 | replicate_sign,0 | replicate_sign,0 | replicate_sign,0 | replicate_sign), 0x0 );

	add_check(0xA0A0A000, 0xA0A0A0A0, make_selector(1 | replicate_sign,0, 0, 0), 0x000000FF );
	add_check(0xA0A0A0A0, 0xA0A0A0A0, make_selector(0 | replicate_sign,0 | replicate_sign,0 | replicate_sign,0 | replicate_sign), 0xFFFFFFFF );
	add_check(0xA0A0A0A0, 0xA0A0A0A0, make_selector(1 | replicate_sign,2 | replicate_sign,3 | replicate_sign,4 | replicate_sign), 0xFFFFFFFF );
	add_check(0x11111111, 0x11111111, make_selector(6, 7 | replicate_sign,1 | replicate_sign, 1 | replicate_sign), 0x00000011 );
	add_check(0x33221100, 0x77665544, make_selector(7,6 | replicate_sign,5 | replicate_sign,4), 0x44000077 );

	add_check(0x33221100, 0x77665544, make_selector(1,1 | replicate_sign,2,2 | replicate_sign), 0x00220011 );
	add_check(0x33F2F100, 0x77665544, make_selector(1,1 | replicate_sign,2,2 | replicate_sign), 0xFFF2FFF1 );

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::permute_bytes{},
		expected_results.data(),
		num_checks,
		make_exact_comparison<result_type>,
		low_words.data(),
		high_words.data(),
		selectors_words.data()
	);
}

TEST_CASE_TEMPLATE("count_leading_zeros", I, int32_t, uint32_t, int64_t, uint64_t)
{
	using result_type = int32_t;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x, result_type result) {
		values.emplace_back(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto num_all_bits = size_in_bits<I>();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
//	constexpr const auto mid_bit_index = half_num_bits - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;
//	constexpr const auto msb_index { size_in_bits<I>() - 1 };


	add_check(0,                  num_all_bits);
	add_check(0b1,                num_all_bits - 1);
	add_check(0b10,               num_all_bits - 2);
	add_check(0b11,               num_all_bits - 2);
	add_check(0b101,              num_all_bits - 3);

	add_check(mid_bit_on - 1,     half_num_bits);
	add_check(mid_bit_on,         half_num_bits - 1);
	add_check(mid_bit_on + 1,     half_num_bits - 1);
	add_check(~ (mid_bit_on - 1), 0);
	add_check(~ (mid_bit_on),     0);
	add_check(~ (mid_bit_on + 1), 0);

	if (std::is_unsigned<I>::value) {
		add_check(max - 1,            0);
		add_check(max,                0);
	}
	else {
		add_check(I(-1),              0);
		add_check(I(-20),             0);
		add_check(max - 1,            1);
		add_check(max,                1);
		add_check(min,                0);
		add_check(min + 1,            0);
	}

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::count_leading_zeros<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}

TEST_CASE_TEMPLATE("average", I, int, unsigned)
{
	using result_type = I;
	std::vector<result_type> expected_results;
	std::vector<I> lhs;
	std::vector<I> rhs;

	auto add_check = [&](I x, I y) {
		lhs.emplace_back(x);
		rhs.emplace_back(y);
		I result ( (int64_t{x} + int64_t{y}) / 2 );
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;


	add_check(0, 0);
	add_check(1, 0);
	add_check(1, 0);
	add_check(1, 1);
	add_check(mid_bit_on - 1, mid_bit_on - 1);
	add_check(mid_bit_on, mid_bit_on - 1);
	add_check(mid_bit_on - 1, mid_bit_on);
	add_check(mid_bit_on, mid_bit_on);
	add_check(mid_bit_on, mid_bit_on + 1);
	add_check(mid_bit_on + 1, mid_bit_on);
	add_check(mid_bit_on + 1, mid_bit_on + 1);
	add_check(max - 1, max - 1);
	add_check(max, max - 1);
	add_check(max - 1, max);
	add_check(max, max);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::average<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("average_rounded_up", I, int, unsigned)
{
	using result_type = I;
	std::vector<result_type> expected_results;
	std::vector<I> lhs;
	std::vector<I> rhs;

	auto add_check = [&](I x, I y) {
		lhs.emplace_back(x);
		rhs.emplace_back(y);
		I result ( ((int64_t{x} + int64_t{y}) + 1) / 2 );
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;


	add_check(0, 0);
	add_check(1, 0);
	add_check(1, 0);
	add_check(1, 1);
	add_check(mid_bit_on - 1, mid_bit_on - 1);
	add_check(mid_bit_on, mid_bit_on - 1);
	add_check(mid_bit_on - 1, mid_bit_on);
	add_check(mid_bit_on, mid_bit_on);
	add_check(mid_bit_on, mid_bit_on + 1);
	add_check(mid_bit_on + 1, mid_bit_on);
	add_check(mid_bit_on + 1, mid_bit_on + 1);
	add_check(max - 1, max - 1);
	add_check(max, max - 1);
	add_check(max - 1, max);
	add_check(max, max);

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::average_rounded_up<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		lhs.data(), rhs.data());
}

TEST_CASE("funnel_shift_right")
{
	using result_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<uint32_t> low_words;
	std::vector<uint32_t> high_words;
	std::vector<uint32_t> shift_amounts;

	auto add_check = [&](
		uint32_t low_word,
		uint32_t high_word,
		uint32_t shift_amount,
		result_type result)
	{
		low_words.emplace_back(low_word);
		high_words.emplace_back(high_word);
		shift_amounts.emplace_back(shift_amount);
		expected_results.emplace_back(result);
	};

	//            low          high          shift    result
	//            word         word          amount
	//           ----------------------------------------------------

	add_check(          ~0u,          0u,      0,         ~0u );
	add_check(       0xCA7u, 0xDEADBEEFu,      0,      0xCA7u );
	add_check(          ~0u,          0u,      5, 0x07FFFFFFu );
	add_check(          ~0u,      0b111u,      4, 0x7FFFFFFFu );
	add_check(           0u, 0xDEADBEEFu,     32, 0xDEADBEEFu );
	add_check(       0xCA7u, 0xDEADBEEFu,     32, 0xDEADBEEFu );
	add_check( 0xCA7u << 16, 0xDEADBEEFu,     16, 0xBEEF0CA7u );

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::funnel_shift_right<kat::builtins::funnel_shift_amount_resolution_mode_t::cap_at_full_word_size>{},
		expected_results.data(),
		num_checks,
		make_exact_comparison<result_type>,
		low_words.data(),
		high_words.data(),
		shift_amounts.data()
	);
}

TEST_CASE("funnel_shift_left")
{
	using result_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<uint32_t> low_words;
	std::vector<uint32_t> high_words;
	std::vector<uint32_t> shift_amounts;

	auto add_check = [&](
		uint32_t low_word,
		uint32_t high_word,
		uint32_t shift_amount,
		result_type result)
	{
		low_words.emplace_back(low_word);
		high_words.emplace_back(high_word);
		shift_amounts.emplace_back(shift_amount);
		expected_results.emplace_back(result);
	};

	//            low            high         shift    result
	//            word           word         amount
	//           ----------------------------------------------------

	add_check( 0u,            0xDEADBEEFu,      0, 0xDEADBEEFu );
	add_check( 0u,            0xDEADBEEFu,      4, 0xEADBEEF0u );
	add_check( 0u,            0xDEADBEEFu,     16, 0xBEEF0000u );
	add_check( 0x0ACEu << 16, 0xDEADBEEFu,     16, 0xBEEF0ACEu );
	add_check( 0xDEADBEEFu,   0u,              32, 0xDEADBEEFu );
	add_check( 0b10u,         ~0u,             31, (1 << 31) | 0b1 );

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::funnel_shift_left<kat::builtins::funnel_shift_amount_resolution_mode_t::cap_at_full_word_size>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		low_words.data(),
		high_words.data(),
		shift_amounts.data()
	);
}

} // TEST_SUITE("uniform builtins")

TEST_SUITE("uniform non-builtins") {

TEST_CASE_TEMPLATE("count_trailing_zeros", I, int, unsigned, long, unsigned long, long long, unsigned long long)
{
	using result_type = int32_t;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x, result_type result) {
		values.emplace_back(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto num_all_bits = size_in_bits<I>();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
//	constexpr const auto mid_bit_index = half_num_bits - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;
//	constexpr const auto msb_index { size_in_bits<I>() - 1 };


	add_check(0,                  size_in_bits<I>());
	add_check(0b1,                0);
	add_check(0b10,               1);
	add_check(0b11,               0);
	add_check(0b101,              0);

	add_check(mid_bit_on - 1,     0);
	add_check(mid_bit_on,         half_num_bits);
	add_check(mid_bit_on + 1,     0);
	add_check(~ (mid_bit_on - 1), half_num_bits);
	add_check(~ (mid_bit_on),     0);
	add_check(~ (mid_bit_on + 1), 1);

	if (std::is_unsigned<I>::value) {
		add_check(max - 1,            1);
		add_check(max,                0);
	}
	else {
		add_check(I(-1),              0);
		add_check(I(-20),             2);
		add_check(max - 1,            1);
		add_check(max,                0);
		add_check(min,                num_all_bits - 1);
		add_check(min + 1,            0);
	}

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::count_trailing_zeros<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}


TEST_CASE_TEMPLATE("find_first_set", I, int, unsigned, long, unsigned long, long long, unsigned long long)
{
	using result_type = int32_t;
	std::vector<result_type> expected_results;
	std::vector<I> values;

	auto add_check = [&](I x, result_type result) {
		values.emplace_back(x);
		expected_results.emplace_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto num_all_bits = size_in_bits<I>();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
//	constexpr const auto mid_bit_index = half_num_bits - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;
//	constexpr const auto msb_index { size_in_bits<I>() - 1 };


	add_check(0,                 -1 + 1);
	add_check(0b1,                0 + 1);
	add_check(0b10,               1 + 1);
	add_check(0b11,               0 + 1);
	add_check(0b101,              0 + 1);

	add_check(mid_bit_on - 1,     0 + 1);
	add_check(mid_bit_on,         half_num_bits + 1);
	add_check(mid_bit_on + 1,     0 + 1);
	add_check(~ (mid_bit_on - 1), half_num_bits + 1);
	add_check(~ (mid_bit_on),     0 + 1);
	add_check(~ (mid_bit_on + 1), 1 + 1);

	if (std::is_unsigned<I>::value) {
		add_check(max - 1,            1 + 1);
		add_check(max,                0 + 1);
	}
	else {
		add_check(I(-1),              0 + 1);
		add_check(I(-20),             2 + 1);
		add_check(max - 1,            1 + 1);
		add_check(max,                0 + 1);
		add_check(min,                num_all_bits - 1 + 1);
		add_check(min + 1,            0 + 1);
	}

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::find_first_set<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		values.data());
}

TEST_CASE_TEMPLATE("extract_bits", I, int32_t, uint32_t, int64_t, uint64_t)
	//int, unsigned int, long)// , unsigned long)//, long long, unsigned long long)
{
	using result_type = I;
	using bit_index_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<I> bit_fields;
	std::vector<bit_index_type> start_positions;
	std::vector<bit_index_type> numbers_of_bits;

	auto add_check = [&](
		I bits,
		bit_index_type start_pos,
		bit_index_type num_bits,
		result_type unsigned_result,
		std::make_signed_t<result_type> signed_result)
	{
		bit_fields.emplace_back(bits);
		start_positions.emplace_back(start_pos);
		numbers_of_bits.emplace_back(num_bits);
		expected_results.emplace_back(
			std::is_unsigned<I>::value ? unsigned_result : signed_result);
	};

	//           bit   start num   unsigned    signed
	//           field  pos  bits   result     result
	//           -------------------------------------

	add_check(0b0000, 0,   0,       0b0,       0b0);
	add_check(0b0000, 0,   1,       0b0,       0b0);
	add_check(0b0000, 0,   2,      0b00,       0b0);
	add_check(0b0000, 0,   3,     0b000,       0b0);
	add_check(0b0001, 0,   0,       0b0,       0b0);
	add_check(0b0001, 0,   1,       0b1,      -0b1);
	add_check(0b0001, 0,   2,      0b01,      0b01);
	add_check(0b0001, 0,   3,     0b001,     0b001);
	add_check(0b0101, 0,   0,       0b0,       0b0);
	add_check(0b0101, 0,   1,       0b1,      -0b1);
	add_check(0b0101, 0,   2,      0b01,      0b01);
	add_check(0b0101, 0,   3,     0b101,     -0b11);

	add_check(0b0000, 1,   0,       0b0,       0b0);
	add_check(0b0000, 1,   1,       0b0,       0b0);
	add_check(0b0000, 1,   2,      0b00,       0b0);
	add_check(0b0000, 1,   3,     0b000,       0b0);
	add_check(0b0001, 1,   0,       0b0,       0b0);
	add_check(0b0001, 1,   1,       0b0,       0b0);
	add_check(0b0001, 1,   2,      0b00,       0b0);
	add_check(0b0001, 1,   3,     0b000,       0b0);
	add_check(0b0101, 1,   0,       0b0,       0b0);
	add_check(0b0101, 1,   1,       0b0,       0b0);
	add_check(0b0101, 1,   2,      0b10,     -0b10);
	add_check(0b0101, 1,   3,     0b010,      0b10);

	if (std::is_signed<I>::value) {
		// TODO: signed testcases
	}

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::extract_bits<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		bit_fields.data(),
		start_positions.data(),
		numbers_of_bits.data()
	);
}

TEST_CASE_TEMPLATE("replace_bits", I, uint32_t, uint64_t)
{
	using result_type = I;
	using bit_index_type = uint32_t;
	std::vector<result_type> expected_results;
	std::vector<I> original_bit_fields;
	std::vector<I> bits_to_insert;
	std::vector<bit_index_type> start_positions;
	std::vector<bit_index_type> numbers_of_bits;

	auto add_check = [&](
		I original_bit_field,
		I bits_to_insert_into_this_field,
		bit_index_type start_pos,
		bit_index_type num_bits,
		result_type result)
	{
		original_bit_fields.emplace_back(original_bit_field);
		bits_to_insert.emplace_back(bits_to_insert_into_this_field);
		start_positions.emplace_back(start_pos);
		numbers_of_bits.emplace_back(num_bits);
		expected_results.emplace_back(result);
	};

	//           original    bits to  start  num   unsigned
	//           bit field    insert    pos   bits   result
	//           --------------------------------------

	add_check(     0,         1,      0,     1,   1);
	add_check(     0,         1,      1,     1,   2);
	add_check(     0,         1,      2,     1,   4);
	add_check(     0,         1,      3,     1,   8);
	add_check(     0,         1,      5,     1,   32);
	add_check(     0,         1,     11,     1,   2048);
	add_check(     0,         1,     31,     1,   (I(1) << 31) );

	if (std::is_same<I, uint64_t>::value) {

	add_check(     0,         1,     63,     1,   I(uint64_t{1} << 63));

	}

	add_check(0b1000000001, 0b1110011, 2, 6, 0b1011001101);
		// Note: only inserting 6 bits, even though the bits_to_insert value has
		// a non-zero 7'th bit

	auto num_checks = expected_results.size();

	execute_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::replace_bits<I>{},
		expected_results.data(), num_checks,
		make_exact_comparison<result_type>,
		bits_to_insert.data(),
		original_bit_fields.data(),
		start_positions.data(),
		numbers_of_bits.data()
	);
}

} // TEST_SUITE("uniform non-builtins")

// Builtins whose behavior is not uniform across all grid threads,
// or depends on the behavior/values held by other threads

TEST_SUITE("non-uniform builtins") {

TEST_CASE("lane_index")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable { return n++ % kat::warp_size; };

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::lane_index{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}


TEST_CASE("preceding_lanes_mask")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return (1u << lane_index) - 1;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };

	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::preceding{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}

TEST_CASE("preceding_and_self_lanes_mask")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return ((1u << lane_index) - 1) | (1u << lane_index);
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::preceding_and_self{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}

TEST_CASE("self_lane_mask")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return (1u << lane_index);
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::self{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}

TEST_CASE("succeeding_and_self_lanes_mask")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return ~((1u << lane_index) - 1);
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::succeeding_and_self{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}

TEST_CASE("succeeding_lanes_mask")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	// No arguments

	auto generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return ~((1u << lane_index) - 1) & ~(1u << lane_index);
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::succeeding{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>
	);
}

#if (__CUDACC_VER_MAJOR__ >= 9)


TEST_CASE("ballot")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 1 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	std::vector<int> values;
	std::vector<kat::lane_mask_t> lane_masks;

	// Our testcase will have two "parts", in each of the two blocks. In this
	// first block we'll use the full mask; in the second block we'll use
	// two different masks (but not warp-uniformly).

	auto value_generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return lane_index % 2;
	};

	std::generate_n(
		std::back_inserter(values),
		num_checks,
		value_generator
	);

	auto lane_mask_generator = [&, n = 0] () mutable {
		if (n < block_size) {
			n++;
			return kat::lane_mask_t{kat::full_warp_mask};
		}
		auto lane_index = n++ % kat::warp_size;
		constexpr const kat::lane_mask_t odd_lanes  = 0b0101'0101'0101'0101'0101'0101'0101'0101;
		constexpr const kat::lane_mask_t even_lanes = 0b1010'1010'1010'1010'1010'1010'1010'1010;
		kat::lane_mask_t self_mask = 1 << lane_index;
			// Warp voting, matching etc. instructions typically require each lane to have itself
			// included in the mask of relevant lanes
		return  self_mask |
			( (lane_index < kat::warp_size / 2) ? odd_lanes : even_lanes );
			// Note: the lane masks don't correspond to which lanes are looking at those lane masks
	};

	std::generate_n(
		std::back_inserter(lane_masks),
		num_checks,
		lane_mask_generator
	);

	auto result_generator = [&, n = 0] () mutable {
		auto lane_index = n % kat::warp_size;
		auto lane_value = values[n];
		auto lane_mask = lane_masks[n];
		kat::lane_mask_t ballot { 0 };
		for(auto i = 0; i < kat::warp_size; i++) {
			if (values[n - lane_index + i]) {
				ballot |= (1u << i);
			}
		}
		ballot &= lane_mask;
		n++;
		return ballot;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		result_generator
	);

	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::ballot{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>,
		values.data(),
		lane_masks.data()
	);
}

TEST_CASE("all_lanes_satisfy")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	std::vector<int> values;
	std::vector<kat::lane_mask_t> lane_masks;

	// Our testcase will have two "parts", in each of the two blocks. In this
	// first block we'll use the full mask; in the second block we'll use
	// two different masks (but not warp-uniformly).


	auto value_generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return lane_index % 2;
	};

	std::generate_n(
		std::back_inserter(values),
		num_checks,
		value_generator
	);

	auto lane_mask_generator = [&, n = 0] () mutable {
		if (n < block_size) { n++; return kat::lane_mask_t{kat::full_warp_mask}; }
		auto lane_index = n++ % kat::warp_size;
		constexpr const kat::lane_mask_t odd_lanes  = 0b0101'0101'0101'0101'0101'0101'0101'0101;
		constexpr const kat::lane_mask_t even_lanes = 0b1010'1010'1010'1010'1010'1010'1010'1010;
		kat::lane_mask_t self_mask = 1 << lane_index;
			// Warp voting, matching etc. instructions typically require each lane to have itself
			// included in the mask of relevant lanes
		return  self_mask |
			( (lane_index < kat::warp_size / 2) ? odd_lanes : even_lanes );
			// Note: the lane masks don't correspond to which lanes are looking at those lane masks
	};

	std::generate_n(
		std::back_inserter(lane_masks),
		num_checks,
		lane_mask_generator
	);

	auto result_generator = [&, n = 0] () mutable {
		auto lane_index = n % kat::warp_size;
		auto lane_value = values[n];
		auto lane_mask = lane_masks[n];
		kat::lane_mask_t ballot { 0 };
		for(auto i = 0; i < kat::warp_size; i++) {
			if (values[n - lane_index + i]) {
				ballot |= (1 << i);
			}
		}

		n++;
		return (ballot ^ ~lane_mask) == kat::full_warp_mask;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		result_generator
	);


	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::all_lanes_satisfy{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>,
		values.data(),
		lane_masks.data()
	);
}

TEST_CASE("any_lanes_satisfy")
{
	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	std::vector<int> values;
	std::vector<kat::lane_mask_t> lane_masks;

	// Our testcase will have two "parts", in each of the two blocks. In this
	// first block we'll use the full mask; in the second block we'll use
	// two different masks (but not warp-uniformly).


	auto value_generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return lane_index % 2;
	};

	std::generate_n(
		std::back_inserter(values),
		num_checks,
		value_generator
	);

	auto lane_mask_generator = [&, n = 0] () mutable {
		if (n < block_size) { n++; return kat::lane_mask_t{kat::full_warp_mask}; }
		auto lane_index = n++ % kat::warp_size;
		constexpr const kat::lane_mask_t odd_lanes  = 0b0101'0101'0101'0101'0101'0101'0101'0101;
		constexpr const kat::lane_mask_t even_lanes = 0b1010'1010'1010'1010'1010'1010'1010'1010;
		kat::lane_mask_t self_mask = 1 << lane_index;
			// Warp voting, matching etc. instructions typically require each lane to have itself
			// included in the mask of relevant lanes
		return  self_mask |
			( (lane_index < kat::warp_size / 2) ? odd_lanes : even_lanes );
			// Note: the lane masks don't correspond to which lanes are looking at those lane masks
	};

	std::generate_n(
		std::back_inserter(lane_masks),
		num_checks,
		lane_mask_generator
	);

	auto result_generator = [&, n = 0] () mutable {
		auto lane_index = n % kat::warp_size;
		auto lane_value = values[n];
		auto lane_mask = lane_masks[n];
		kat::lane_mask_t ballot { 0 };
		for(auto i = 0; i < kat::warp_size; i++) {
			if (values[n - lane_index + i]) {
				ballot |= (1 << i);
			}
		}
		n++;
		return (ballot & lane_mask) != 0;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		result_generator
	);


	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::any_lanes_satisfy{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>,
		values.data(),
		lane_masks.data()
	);
}

#if ! defined(__CUDA_ARCH__) or __CUDA_ARCH__ >= 700
TEST_CASE("all_lanes_agree")
{
	cuda::device_t device { cuda::device::current::get() };
	if (device.properties().compute_capability() < cuda::device::make_compute_capability(7,0)) {
		return;
	}

	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	std::vector<int> values;
	std::vector<kat::lane_mask_t> lane_masks;

	// Our testcase will have two "parts", in each of the two blocks. In this
	// first block we'll use the full mask; in the second block we'll use
	// two different masks (but not warp-uniformly).


	auto value_generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return lane_index % 2;
	};

	std::generate_n(
		std::back_inserter(values),
		num_checks,
		value_generator
	);

	auto lane_mask_generator = [&, n = 0] () mutable {
		if (n < block_size) { n++; return kat::lane_mask_t{kat::full_warp_mask}; }
		auto lane_index = n++ % kat::warp_size;
		constexpr const kat::lane_mask_t odd_lanes  = 0b0101'0101'0101'0101'0101'0101'0101'0101;
		constexpr const kat::lane_mask_t even_lanes = 0b1010'1010'1010'1010'1010'1010'1010'1010;
		kat::lane_mask_t self_mask = 1 << lane_index;
			// Warp voting, matching etc. instructions typically require each lane to have itself
			// included in the mask of relevant lanes
		return  self_mask |
			( (lane_index < kat::warp_size / 2) ? odd_lanes : even_lanes );
			// Note: the lane masks don't correspond to which lanes are looking at those lane masks
	};

	std::generate_n(
		std::back_inserter(lane_masks),
		num_checks,
		lane_mask_generator
	);

	auto result_generator = [&, n = 0] () mutable {
		auto lane_index = n % kat::warp_size;
		auto lane_value = values[n];
		auto lane_mask = lane_masks[n];
		kat::lane_mask_t ballot { 0 };
		for(auto i = 0; i < kat::warp_size; i++) {
			if (values[n - lane_index + i]) {
				ballot |= (1 << i);
			}
		}
		n++;
		return (ballot & lane_mask) != 0;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		result_generator
	);


	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::all_lanes_agree{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>,
		values.data(),
		lane_masks.data()
	);
}
#endif

TEST_CASE_TEMPLATE("get_matching_lanes", I, int, unsigned, long, unsigned long, long long, unsigned long long)
{
	cuda::device_t device { cuda::device::current::get() };
	if (device.properties().compute_capability() < cuda::device::make_compute_capability(7,0)) {
		return;
	}

	using result_type = uint32_t;
	auto block_size { kat::warp_size * 2 };
	auto num_grid_blocks { 2 };
	auto num_checks = block_size * num_grid_blocks; // one per thread

	std::vector<result_type> expected_results;
	std::vector<I> values;
	std::vector<kat::lane_mask_t> lane_masks;

	// Our testcase will have two "parts", in each of the two blocks. In this
	// first block we'll use the full mask; in the second block we'll use
	// two different masks (but not warp-uniformly).


	auto value_generator = [n = 0] () mutable {
		auto lane_index = n++ % kat::warp_size;
		return lane_index % 3;
	};

	std::generate_n(
		std::back_inserter(values),
		num_checks,
		value_generator
	);

	auto lane_mask_generator = [&, n = 0] () mutable {
		if (n < block_size) { n++; return kat::lane_mask_t{kat::full_warp_mask}; }
		auto lane_index = n++ % kat::warp_size;
		constexpr const kat::lane_mask_t odd_lanes  = 0b0101'0101'0101'0101'0101'0101'0101'0101;
		constexpr const kat::lane_mask_t even_lanes = 0b1010'1010'1010'1010'1010'1010'1010'1010;
		kat::lane_mask_t self_mask = 1 << lane_index;
			// Warp voting, matching etc. instructions typically require each lane to have itself
			// included in the mask of relevant lanes
		return  self_mask |
			( (lane_index < kat::warp_size / 2) ? odd_lanes : even_lanes );
			// Note: the lane masks don't correspond to which lanes are looking at those lane masks
	};

	std::generate_n(
		std::back_inserter(lane_masks),
		num_checks,
		lane_mask_generator
	);

	auto result_generator = [&, n = 0] () mutable {
		auto lane_index = n % kat::warp_size;
		auto lane_value = values[n];
		auto lane_mask = lane_masks[n];
		kat::lane_mask_t matches { 0 };
		for(auto i = 0; i < kat::warp_size; i++) {
			if (values[n - lane_index + i] == lane_value) {
				matches |= (1 << i);
			}
		}
		matches &= lane_mask;
		n++;
		return matches;
	};

	std::generate_n(
		std::back_inserter(expected_results),
		num_checks,
		result_generator
	);


	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	execute_non_uniform_builtin_testcase_on_gpu_and_check(
		device_function_ptrs::succeeding{},
		expected_results.data(),
		num_checks, num_grid_blocks, block_size,
		make_exact_comparison<result_type>,
		values.data(),
		lane_masks.data()
	);
}

// Note: Will not be testing the variants of these for versions of CUDA before 9.

#endif // CUDA 9

} // TEST_SUITE("non-uniform builtins")

/*

The following are a bit tricky to test - just need to check uniformity of results across the block or the grid?

unsigned           special_registers::symmetric_multiprocessor_index();
unsigned long long special_registers::grid_index();

The following are tested indirectly via shared-memory-related tests:

unsigned int       special_registers::dynamic_shared_memory_size();
unsigned int       special_registers::total_shared_memory_size();

*/
