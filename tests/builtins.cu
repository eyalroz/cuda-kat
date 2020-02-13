#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include "util/woodruff_int128_t.hpp"
#include "util/woodruff_uint128_t.hpp"
#include <kat/on_device/builtins.cuh>
#include <kat/on_device/non-builtins.cuh>
#include <limits>
#include <cuda/api_wrappers.hpp>

using std::size_t;
using fake_bool = int8_t; // so as not to have trouble with vector<bool>
static_assert(sizeof(bool) == sizeof(fake_bool), "unexpected size mismatch");

/*

To test:

T multiplication_high_bits(T x, T y);
F divide(F dividend, F divisor);
T absolute_value(T x);
T minimum(T x, T y) = delete; // don't worry, it's not really deleted for all types
T maximum(T x, T y) = delete; // don't worry, it's not really deleted for all types
template <typename T, typename S> S sum_with_absolute_difference(T x, T y, S addend);
int population_count(I x);
T bit_reverse(T x) = delete;

unsigned find_last_non_sign_bit(I x) = delete;
T load_global_with_non_coherent_cache(const T* ptr);
int count_leading_zeros(I x) = delete;
T extract(T bit_field, unsigned int start_pos, unsigned int num_bits);
T insert(T original_bit_field, T bits_to_insert, unsigned int start_pos, unsigned int num_bits);

T select_bytes(T x, T y, unsigned byte_selector);

native_word_t funnel_shift(native_word_t  low_word, native_word_t  high_word, native_word_t  shift_amount);

typename std::conditional<Signed, int, unsigned>::type average(
	typename std::conditional<Signed, int, unsigned>::type x,
	typename std::conditional<Signed, int, unsigned>::type y);

unsigned           special_registers::lane_index();
unsigned           special_registers::symmetric_multiprocessor_index();
unsigned long long special_registers::grid_index();
unsigned int       special_registers::dynamic_shared_memory_size();
unsigned int       special_registers::total_shared_memory_size();

} // namespace special_registers

#if (__CUDACC_VER_MAJOR__ >= 9)
lane_mask_t ballot            (int condition, lane_mask_t lane_mask = full_warp_mask);
int         all_lanes_satisfy (int condition, lane_mask_t lane_mask = full_warp_mask);
int         some_lanes_satisfy(int condition, lane_mask_t lane_mask = full_warp_mask);
int         all_lanes_agree   (int condition, lane_mask_t lane_mask = full_warp_mask);
#else
lane_mask_t ballot            (int condition);
int         all_lanes_satisfy (int condition);
int         some_lanes_satisfy(int condition);
#endif

#if (__CUDACC_VER_MAJOR__ >= 9)
bool is_uniform_across_lanes(T value, lane_mask_t lane_mask = full_warp_mask);
bool is_uniform_across_warp(T value);
lane_mask_t matching_lanes(T value, lane_mask_t lanes = full_warp_mask);
#endif

unsigned int mask_of_lanes::preceding();
unsigned int mask_of_lanes::preceding_and_self();
unsigned int mask_of_lanes::self();
unsigned int mask_of_lanes::succeeding_and_self();
unsigned int mask_of_lanes::succeeding();

lane_mask_t mask_of_lanes::matching_value(lane_mask_t lane_mask, T value);
lane_mask_t mask_of_lanes::matching_value(T value);
int find_first_set(I x);
int count_trailing_zeros(I x) { return find_first_set<I>(x) - 1; }
int count_leading_zeros(I x);

*/

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


//PREPARE_BUILTIN_INNER(builtins, multiplication_high_bits, T, typename T)

/*
#define PREPARE_BUILTIN(subnamespace, builtin_function_basename, t, ...) \
	template <typename t, MAP(PREPEND_TYPENAME_IDENTIFIER(__VA_ARGS__)> \
	struct builtin_function_basename { \
		static const void* const ptr; \
	}; \
    \
	template <t MAP(PREPEND_TYPENAME_IDENTIFIER, __VA_ARGS__)> \
	const void* const builtin_function_basename<t, __VA_ARGS__>::ptr { (void *) kat::subnamespace::builtin_function_basename<t, __VA_ARGS__> };
 */
#define INSTANTIATE_BUILTIN_VIA_PTR(builtin_function_basename, ...) \
	template struct builtin_function_basename<__VA_ARGS__>



PREPARE_BUILTIN1(builtins, multiplication_high_bits);
PREPARE_BUILTIN1(builtins, divide);
PREPARE_BUILTIN1(builtins, minimum);
PREPARE_BUILTIN1(builtins, maximum);
PREPARE_BUILTIN2(builtins, sum_with_absolute_difference);
PREPARE_BUILTIN1(builtins, population_count);
PREPARE_BUILTIN1(builtins, bit_reverse);
PREPARE_BUILTIN1(builtins, find_last_non_sign_bit);
PREPARE_BUILTIN1(builtins, load_global_with_non_coherent_cache);
PREPARE_BUILTIN1(builtins::bit_field, extract);
PREPARE_BUILTIN1(builtins::bit_field, insert);
PREPARE_BUILTIN1(builtins, select_bytes);
// This function is special, in that one of its template parameters is a value rather than a type.
PREPARE_BUILTIN_INNER(builtins, funnel_shift, typename T COMMA kat::builtins::funnel_shift_amount_resolution_mode_t AmountResolutionMode, T COMMA AmountResolutionMode);

PREPARE_BUILTIN1(builtins, average);
PREPARE_BUILTIN0(builtins::special_registers, lane_index);
PREPARE_BUILTIN0(builtins::special_registers, symmetric_multiprocessor_index);
PREPARE_BUILTIN0(builtins::special_registers, grid_index);
PREPARE_BUILTIN0(builtins::special_registers, dynamic_shared_memory_size);
PREPARE_BUILTIN0(builtins::special_registers, total_shared_memory_size);


PREPARE_BUILTIN0(builtins::warp, ballot);
PREPARE_BUILTIN0(builtins::warp, all_lanes_satisfy);
PREPARE_BUILTIN0(builtins::warp, any_lanes_satisfy);
#if (__CUDACC_VER_MAJOR__ >= 9)
PREPARE_BUILTIN0(builtins::warp, all_lanes_agree);
#endif


#if (__CUDACC_VER_MAJOR__ >= 9)
PREPARE_BUILTIN1(builtins::warp, is_uniform_across_lanes);
PREPARE_BUILTIN1(builtins::warp, is_uniform_across_warp);
PREPARE_BUILTIN1(builtins::warp, matching_lanes);
#endif

PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, preceding);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, preceding_and_self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, succeeding_and_self);
PREPARE_BUILTIN0(builtins::warp::mask_of_lanes, succeeding);

PREPARE_BUILTIN1(non_builtins, find_first_set);
PREPARE_BUILTIN1(non_builtins, count_trailing_zeros);
PREPARE_BUILTIN1(non_builtins, count_leading_zeros);

//INSTANTIATE_BUILTIN_VIA_PTR(multiplication_high_bits, unsigned);
//INSTANTIATE_BUILTIN_VIA_PTR(multiplication_high_bits, long long);
//INSTANTIATE_BUILTIN_VIA_PTR(multiplication_high_bits, unsigned long long);

} // namespace builtin_device_function_ptrs

namespace kernels {

template <typename DeviceFunctionHook, typename R, typename... Is>
__global__ void execute_testcases(
//	F                          f,
	size_t                     num_checks,
	fake_bool*   __restrict__     executed,
	R*        __restrict__     results,
	const Is* __restrict__ ... inputs
	)
{
	// static_assert(all_true<not std::is_reference<Is>::value...>::value, "The input types should be simple - no references please");
	auto i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_checks) { return; }
	using device_function_type = auto (Is...) -> R;
	auto f = // &kat::builtins::multiplication_high_bits<R>;
		(device_function_type*) DeviceFunctionHook::ptr;
//	printf("function ptr is at %p and name is %s\n", f, DeviceFunctionHook::name);
//	printf("i is %2d , num_checks is %2d, %u = high bits of %u * %u\n", (int) i, (int) num_checks, f(inputs[i]...), inputs[i]...);
	results[i] = f(inputs[i]...);
		// It's up to the author of f to ensure there aren't any runtime errors...
		// Also, f should not use any shared memory
	executed[i] = true;
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

// TODO: Take iterator templates rather than pointers
template <typename R, typename... Is>
void check_results(
	size_t                    num_checks,
	const char*               testcase_name,
	// perhaps add another parameter for specific individual-check details?
	const R*                  actual_results,
	fake_bool*                execution_indicators,
	const R*  __restrict__    expected_results,
	R                         comparison_tolerance_fraction,
	const Is* __restrict__... inputs)
{
	std::stringstream ss;
	auto index_width = set_width_for_up_to(num_checks);

	// TODO: Consider using the maximum/minimum result values to set field widths.

	for(size_t i = 0; i < num_checks; i++) {
		ss.str("");
		ss << "Failed executing testcase " << (i+1) << " for " << testcase_name;
		auto failure_message { ss.str() };
		CHECK_MESSAGE(execution_indicators[i], failure_message);
//		auto generate_mismatch_message = [&]()
		if (execution_indicators[i]) {
			ss.str("");
			ss
				<< "Assertion " << std::setw(index_width) << (i+1) << " for testcase" << testcase_name
				<< " :\n"
				<< "With inputs" << std::make_tuple(inputs[i]...)
				<< ", expected " << expected_results[i] << " but got " << actual_results[i];
			auto mismatch_message { ss.str() };
			if (comparison_tolerance_fraction == 0) {
				CHECK_MESSAGE(actual_results[i] == expected_results[i], mismatch_message);
			}
			else {
				CHECK_MESSAGE(actual_results[i] ==  doctest::Approx(expected_results[i]).epsilon(comparison_tolerance_fraction), mismatch_message);
			}
		}
	}
}

// TODO: Determine the result type of F...
template <typename DeviceFunctionHook, typename R, typename... Is, size_t... Indices>
void execute_testcase_on_gpu_(
	std::index_sequence<Indices...>,
	const R* __restrict__            expected_results,
	DeviceFunctionHook               dfh,
	size_t                           num_checks,
	R                                comparison_tolerance_fraction,
	Is* __restrict__ ...             inputs)
{
	cuda::device_t<> device { cuda::device::current::get() };
	auto block_size { 128 };
	auto num_grid_blocks { div_rounding_up(num_checks, block_size) };
	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
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

	// static_assert(not std::is_reference<F>::value, "F should not be a reference");

	//std::cout << "f is at " << (void *)(f) << std::endl;
//	std::cout << "function is at " << DeviceFunctionHook::ptr << " and name is " << DeviceFunctionHook::name << std::endl;

	cuda::launch(
		kernels::execute_testcases<DeviceFunctionHook, R, Is...>,
		launch_config,
		num_checks,
		device_side_execution_indicators.get(),
		device_side_results.get(),
		std::get<Indices>(device_side_inputs).get()... );

	cuda::memory::copy(host_side_results.data(), device_side_results.get(), sizeof(R) * num_checks);
	cuda::memory::copy(host_side_execution_indicators.data(), device_side_execution_indicators.get(), sizeof(bool) * num_checks);

	check_results (
		num_checks, DeviceFunctionHook::name,
		// perhaps add another parameter for specific testcase details?
		host_side_results.data(), host_side_execution_indicators.data(), expected_results, comparison_tolerance_fraction, inputs...);
}

template <typename DeviceFunctionHook, typename R, typename... Is>
void execute_testcase_on_gpu(
	DeviceFunctionHook     dfh,
	const R* __restrict__  expected_results,
	size_t                 num_checks,
	R                      comparison_tolerance_fraction,
	Is* __restrict__ ...   inputs)
{
	return execute_testcase_on_gpu_( // <R, Is...>(
		typename std::make_index_sequence<sizeof...(Is)> {},
		expected_results,
		dfh,
		num_checks,
		comparison_tolerance_fraction,
		inputs...
	);
}


TEST_SUITE("builtins (and non-builtins)") {

// Note: Types for instantiation are chosen based on what's actually available in CUDA

TEST_CASE_TEMPLATE("multiplication high bits", I, unsigned, long long, unsigned long long)
{
	std::vector<I> expected_results;
	std::vector<I> lhs;
	std::vector<I> rhs;

	auto add_testcase = [&](I x, I y) {
		lhs.push_back(x);
		rhs.push_back(y);
//		std::cout << "testcase " << expected_results.size() + 1 << ": ";
		auto result = multiplication_high_bits(x, y);
		expected_results.push_back(result);
	};

	constexpr const auto max = std::numeric_limits<I>::max();
	constexpr const auto min = std::numeric_limits<I>::min();
	constexpr const auto half_num_bits = size_in_bits<I>() / 2;
	constexpr const auto almost_sqrt = (I{1} << half_num_bits) - 1;
	constexpr const auto mid_bit_on = I{1} << half_num_bits;

	// Yields 0
	add_testcase(0, 0);
	add_testcase(1, 0);
	add_testcase(0, 1);
	add_testcase(1, 1);
	add_testcase(almost_sqrt, almost_sqrt);

	// Yields 1
	add_testcase(mid_bit_on, mid_bit_on);

	// Yields 6
	add_testcase(mid_bit_on * 2, mid_bit_on * 3);
	add_testcase(mid_bit_on * 3, mid_bit_on * 2);

	// Depends...
	add_testcase(min, min);
	add_testcase(min, max);
	add_testcase(max, min);
	add_testcase(max, max);

	auto num_checks = expected_results.size();

//	std::cout << "function is at " << (void *)(kat::builtins::multiplication_high_bits<I>) << std::endl;

	I comparison_tolerance_fraction { 0 };
	execute_testcase_on_gpu
			// <I, /*decltype(kat::builtins::multiplication_high_bits<I>),*/ I, I>
		(
			device_function_ptrs::multiplication_high_bits<I>{}, // kat::builtins::multiplication_high_bits<I>,
			expected_results.data(),
			num_checks, comparison_tolerance_fraction,
			lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("minimum", T, int, unsigned int, long, unsigned long, long long, unsigned long long, float , double)
{
	std::vector<T> expected_results;
	std::vector<T> lhs;
	std::vector<T> rhs;

	auto add_testcase = [&](T x, T y) {
		lhs.push_back(x);
		rhs.push_back(y);
		auto result = std::min<T>(x, y);
			// Note: This is not a trivial choice! The behavior in edge cases,
			// like or near-equality for floating-point types, is not the same
			// among any two implementations of a "minimum()" function.
		expected_results.push_back(result);
	};

	constexpr const auto max = std::numeric_limits<T>::max();
	constexpr const auto min = std::numeric_limits<T>::min();
	constexpr const auto half_num_bits = size_in_bits<T>() / 2;
	constexpr const auto half_num_bits_max_bits = (T{ uint64_t{1} << half_num_bits}) - 1;
	constexpr const auto mid_bit_on = T{uint64_t{1} << half_num_bits};
		// Note that for floating-point types, bit-counting is not that meaningful

	add_testcase(0, 0);
	add_testcase(1, 0);
	add_testcase(0, 1);
	add_testcase(1, 1);
	add_testcase(half_num_bits_max_bits, half_num_bits_max_bits);
	add_testcase(mid_bit_on, mid_bit_on);
	add_testcase(mid_bit_on * 2, mid_bit_on * 3);
	add_testcase(mid_bit_on * 3, mid_bit_on * 2);
	add_testcase(min, min);
	add_testcase(min, max);
	add_testcase(max, min);
	add_testcase(max, max);

	auto num_checks = expected_results.size();

	T comparison_tolerance_fraction { std::is_integral<T>::value ? T{0} : T{0} };
	execute_testcase_on_gpu(
		device_function_ptrs::minimum<T>{},
		expected_results.data(), num_checks,
		comparison_tolerance_fraction,
		lhs.data(), rhs.data());
}

TEST_CASE_TEMPLATE("maximum", T, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double)
{
	std::vector<T> expected_results;
	std::vector<T> lhs;
	std::vector<T> rhs;

	auto add_testcase = [&](T x, T y) {
		lhs.push_back(x);
		rhs.push_back(y);
		auto result = std::max<T>(x, y);
			// Note: This is not a trivial choice! The behavior in edge cases,
			// like or near-equality for floating-point types, is not the same
			// among any two implementations of a "minimum()" function.
		expected_results.push_back(result);
	};

	constexpr const auto max = std::numeric_limits<T>::max();
	constexpr const auto min = std::numeric_limits<T>::min();
	constexpr const auto half_num_bits = size_in_bits<T>() / 2;
	constexpr const auto half_num_bits_max_bits = (T{ uint64_t{1} << half_num_bits}) - 1;
	constexpr const auto mid_bit_on = T{uint64_t{1} << half_num_bits};
		// Note that for floating-point types, bit-counting is not that meaningful

	add_testcase(0, 0);
	add_testcase(1, 0);
	add_testcase(0, 1);
	add_testcase(1, 1);
	add_testcase(half_num_bits_max_bits, half_num_bits_max_bits);
	add_testcase(mid_bit_on, mid_bit_on);
	add_testcase(mid_bit_on * 2, mid_bit_on * 3);
	add_testcase(mid_bit_on * 3, mid_bit_on * 2);
	add_testcase(min, min);
	add_testcase(min, max);
	add_testcase(max, min);
	add_testcase(max, max);

	auto num_checks = expected_results.size();

	T comparison_tolerance_fraction { std::is_integral<T>::value ? T{0} : T{0} };
	execute_testcase_on_gpu(
		device_function_ptrs::maximum<T>{},
		expected_results.data(), num_checks,
		comparison_tolerance_fraction,
		lhs.data(), rhs.data());
}

} // TEST_SUITE("builtins (and non-builtins)")
