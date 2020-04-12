#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"

#include <kat/on_device/miscellany.cuh>
#include <kat/utility.hpp>

// #include <gsl/span>
#include <limits>

template <typename T>
KAT_DEV bool single_swap_test(T x, T y)
{
	T x_ { x };
	T y_ { y };
	kat::swap<T>(x_, y_);
	return (x == y_) and (y == x_);
}

struct copy_testcase_spec {
	std::size_t dst_offset;
	std::size_t src_offset;
	std::size_t num;
};

START_COUNTING_LINES(num_swap_checks);

namespace kernels {

template <typename T>
__global__
void swap_tests(bool* results)
{
//	bool print_first_indices_for_each_function { false };
	auto i = 0;

//	auto maybe_print = [&](const char* section_title) {
//		if (print_first_indices_for_each_function) {
//			printf("%-30s tests start at index  %3d\n", section_title, i);
//		}
//	};
//
//	maybe_print("single_swap_test");
	results[i++] = single_swap_test<T>(T{0}, T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{2}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{1}, T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{2}, T{1}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::min(), T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, std::numeric_limits<T>::min()); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(std::numeric_limits<T>::max(), T{0}); COUNT_THIS_LINE
	results[i++] = single_swap_test<T>(T{0}, std::numeric_limits<T>::max()); COUNT_THIS_LINE
}

} // namespace kernels

FINISH_COUNTING_LINES(num_swap_checks);

namespace kernels {

template <typename T>
__global__
void copy_tests(
	T*                         __restrict__ copy_destination_buffer,
	const T*                   __restrict__ test_data,
	const copy_testcase_spec*  __restrict__ test_cases,
	std::size_t                             num_test_cases)
{
	// Use the following two lines to limit the execution to a single
	// testcase and ignore all the others. They may fail, but if you want
	// to debug the run of a single one without interruption from
	// the others... this will be useful.
//	enum  { run_all_cases = -1 };
//	static const int single_testcase_to_run { 1 }; // 0-based!

	for(std::size_t i = 0; i < num_test_cases; i++) {
//		if ((single_testcase_to_run != run_all_cases) and (single_testcase_to_run != i)) { continue; }
		auto test_case = test_cases[i];
		kat::copy(
			copy_destination_buffer + test_case.dst_offset,
			test_data + test_case.src_offset,
			test_case.num
		);
	}
}

} // namespace kernels

// Note: Use CHECK_MESSAGE!


constexpr const std::size_t copy_alignment_quantum { 256 };

std::vector<copy_testcase_spec> make_copy_testcases()
{

	std::vector<copy_testcase_spec> testcase_specs;

	copy_testcase_spec tc;

	auto make_offset =
		[&](std::size_t extra_offset) {
			return copy_alignment_quantum * testcase_specs.size() + extra_offset;
		};

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 64; testcase_specs.push_back(tc);

	// ---

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 64; testcase_specs.push_back(tc);

	// ---

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 64; testcase_specs.push_back(tc);

	// ---

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 64; testcase_specs.push_back(tc);

	// ---

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(0); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(4); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(2); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(1); tc.num = 64; testcase_specs.push_back(tc);

	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  0; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  1; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  2; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  3; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num =  4; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 61; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 62; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 63; testcase_specs.push_back(tc);
	tc.dst_offset = make_offset(1); tc.src_offset = make_offset(3); tc.num = 64; testcase_specs.push_back(tc);

	return testcase_specs;
}

template <typename T, typename C>
std::vector<T> generate_copy_test_data(const C& testcases)
{
	auto buffer_length_in_elements = copy_alignment_quantum * sizeof(T) * testcases.size();
	std::vector<T> test_data;

	test_data.reserve(buffer_length_in_elements);
	util::random::insertion_generate_n(
		std::back_inserter(test_data),
		buffer_length_in_elements,
			// So, we don't need all of the data, but let's make this code simpler rather than generate less
		util::random::uniform_distribution<T>{}
	);

	// Enable the following for debugging - it will generate the testcase index
	// in the places to be copied, and 0 elsewhere

//	std::vector<T> test_data(buffer_length_in_elements);
//	std::fill(test_data.begin(), test_data.end(), 0);
//	for(auto it = testcases.begin(); it < testcases.end(); it++) {
//		auto idx = std::distance(testcases.begin(), it);
//		auto tc = *it;
//		std::fill_n(test_data.data() + tc.src_offset, tc.num, idx + 1);
//			// Note: There may be less possible values of idx in type T than testcases to run...
//	}

	return test_data;
}

TEST_SUITE("miscellany") {

// TODO: Consider using larger-than-64-bit types, classes, etc.
TEST_CASE_TEMPLATE("test swap", T, int8_t, int16_t, int32_t, int32_t, int64_t, float, double )
{
	cuda::device_t device { cuda::device::current::get() };
	cuda::launch_configuration_t launch_config { cuda::grid::dimensions_t::point(), cuda::grid::dimensions_t::point() };
	auto device_side_results { cuda::memory::device::make_unique<bool[]>(device, num_swap_checks) };
	auto host_side_results { std::unique_ptr<bool[]>(new bool[num_swap_checks]) };

	cuda::launch(
		kernels::swap_tests<T>,
		launch_config,
		device_side_results.get());

	cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(bool) * num_swap_checks);

	for(auto i = 0 ; i < num_swap_checks; i++) {
		if (not host_side_results.get()[i]) {
			MESSAGE("The " << i << xth(i) << " swap check failed.");
		}
	}

}

TEST_CASE_TEMPLATE("test num_warps_to_cover", I, int8_t, int16_t, int32_t, uint32_t, int64_t )
{
	using std::size_t;
	auto irrelevant = [](size_t x) { return x > std::numeric_limits<I>::max(); };
	using kat::num_warp_sizes_to_cover;
	size_t num;

	num =    0; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  0); }
	num =    1; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  1); }
	num =    2; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  1); }
	num =    3; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  1); }
	num =   31; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  1); }
	num =   32; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  1); }
	num =   33; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) ==  2); }
	num = 1023; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == 32); }
	num = 1024; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == 32); }
	num = 1025; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == 33); }

	num = (size_t{1} << 31) - 1; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == size_t{1} << 26);       }
	num =  size_t{1} << 31     ; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == size_t{1} << 26);       }
	num = (size_t{1} << 31) + 1; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == (size_t{1} << 26) + 1); }
	num = (size_t{1} << 32) - 1; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == size_t{1} << 27);       }
	num =  size_t{1} << 32     ; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == size_t{1} << 27);       }
	num = (size_t{1} << 32) + 1; if (not irrelevant(num)) { CHECK(num_warp_sizes_to_cover<I>(num) == (size_t{1} << 27) + 1); }
}

TEST_CASE_TEMPLATE("test native-word copy", T, uint8_t, int16_t, int32_t, uint32_t, int64_t ,float, double )
{
	std::size_t testcase_index { 0 };
	bool cuda_calls_complete { false };

	// TODO: Try this with element sizes which aren't powers of 2, and kat::array types

	auto host_side_testcases = make_copy_testcases();

	// Note: The testcases are assumed to have num + offset lower than the quantum

	auto buffer_length_in_elements = copy_alignment_quantum * sizeof(T) * host_side_testcases.size();

	auto host_side_test_data = generate_copy_test_data<T>(host_side_testcases);

	try {
		cuda::device_t device { cuda::device::current::get() };
		auto device_side_test_data { cuda::memory::device::make_unique<T[]>(device, buffer_length_in_elements) };
		auto device_side_copy_target { cuda::memory::device::make_unique<T[]>(device, buffer_length_in_elements) };
		auto host_side_copy_target { std::unique_ptr<T[]>(new T[buffer_length_in_elements]) };
		auto device_side_testcases { cuda::memory::device::make_unique<copy_testcase_spec[]>(device, host_side_testcases.size()) };

		auto launch_config { single_thread_launch_config() };
		cuda::memory::copy(device_side_test_data.get(), host_side_test_data.data(), sizeof(T) * buffer_length_in_elements);
		cuda::memory::device::zero(device_side_copy_target.get(), sizeof(T) * buffer_length_in_elements);
		cuda::memory::copy(device_side_testcases.get(), host_side_testcases.data(), sizeof(copy_testcase_spec) * host_side_testcases.size());

		cuda::launch(
			kernels::copy_tests<T>,
			launch_config,
			device_side_copy_target.get(),
			device_side_test_data.get(),
			device_side_testcases.get(),
			host_side_testcases.size()
		);

		cuda::outstanding_error::ensure_none();

		cuda::memory::copy(host_side_copy_target.get(), device_side_copy_target.get(), sizeof(T) * buffer_length_in_elements);

		cuda_calls_complete = true;

		std::stringstream ss;
		for(; testcase_index < host_side_testcases.size(); testcase_index++) {
			auto tc = host_side_testcases[testcase_index];
//			gsl::span<T> test_data(host_side_test_data.data() + tc.src_offset, tc.num);
//			gsl::span<T> copy_destination(host_side_copy_target.get() + tc.dst_offset, tc.num);
			std::vector<T> test_data{host_side_test_data.data() + tc.src_offset, host_side_test_data.data() + tc.src_offset + tc.num};
			std::vector<T> copy_destination{host_side_copy_target.get() + tc.dst_offset, host_side_copy_target.get() + tc.dst_offset + tc.num};

			auto first_mismatch = std::mismatch(test_data.begin(), test_data.end(), copy_destination.begin() );

			if (first_mismatch.first != test_data.end()) {

//				gsl::span<T> test_data_quantum(host_side_test_data.data() + testcase_index * copy_alignment_quantum, copy_alignment_quantum);
//				gsl::span<T> copy_destination_quantum(host_side_copy_target.get() + testcase_index * copy_alignment_quantum, copy_alignment_quantum);
//				std::cout << "Testcase " << std::setw(3) << (testcase_index+1) <<  " quantum within test data: ";
//				for(auto x : test_data_quantum) {
//					std::cout << promote_for_streaming(x) << ' ';
//				}
//				std::cout << '\n';
//				std::cout << "Testcase " << std::setw(3) << (testcase_index+1) << " quantum within copy buffer: ";
//				for(auto x : copy_destination_quantum) {
//					std::cout << promote_for_streaming(x) << ' ';
//				}
//				std::cout << "\n\n";

				ss.str(""); // clear the stream
				ss
					<< "Testcase " << std::setw(3) << (testcase_index+1) << " (1-based; "
					<< "src at +" << (tc.src_offset % copy_alignment_quantum) << ", "
					<< "dest at +" << (tc.dst_offset % copy_alignment_quantum) << ", "
					<< std::setw(3) << tc.num << " elements): "
					<< "at index " << std::setw(3) << (first_mismatch.first - test_data.begin()) << ", values "
					<< promote_for_streaming(*(first_mismatch.first)) << " != "
					<< promote_for_streaming(*(first_mismatch.second))
					;
			}
			std::string message = ss.str();
			CHECK_MESSAGE(first_mismatch.first == test_data.end(), message);
			// TODO: Ensure the rest of the alignment quantum in the copy destination is all 0's
		}
	} catch(std::exception& err) {
		std::stringstream message;
		message
			<< "An exception occurred "
			<< (cuda_calls_complete ? "after" : "before")
			<< " the copy, launch and copy back calls were completed, and before testcase "
			<< testcase_index << " was completed. The error was" << err.what();
		FAIL(message.str());
	}
}



} // TEST_SUITE("miscellany")
