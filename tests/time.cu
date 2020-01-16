#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include "common.cuh"
//#include "util/prettyprint.hpp"
#include "util/type_name.hpp"
//#include "util/random.hpp"
//#include "util/miscellany.cuh"
//#include "util/macro.h"
#include <kat/on_device/collaboration/block.cuh>
#include <kat/on_device/time.cuh>

#include <doctest.h>
#include <cuda/api_wrappers.hpp>

#include <algorithm>

constexpr const auto num_grid_blocks {  2 };
constexpr const auto block_size      {  kat::warp_size + 1 };

constexpr const auto sleep_elongation_multiplicative_factor { 10000 };
	// we want each sleep command to take a not-insignificant amount of time

constexpr const auto sleep_elongation_additive_factor { 10 };
	// we want each sleep command to take a not-insignificant amount of time


namespace kernels {

template <kat::sleep_resolution Resolution>
__global__ void measure_time_and_sleep(
	kat::clock_value_t* __restrict__  times_before_sleep,
	kat::clock_value_t* __restrict__  times_after_sleep,
	std::size_t                       total_num_threads
)
{
	auto global_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_thread_id >= total_num_threads) { return; }
	auto time_before_sleep = clock64();
	auto sleep_duration =
		(global_thread_id + sleep_elongation_additive_factor ) * sleep_elongation_multiplicative_factor;
	if (Resolution == kat::sleep_resolution::nanoseconds) {
#if __CUDA_ARCH__ >= 700
		kat::sleep<Resolution>(sleep_duration);
#else
		// we won't break the compilation; it's up to the host-side test code to not run this.
		asm("trap;");
#endif
	}
	else {
		kat::sleep<Resolution>(sleep_duration);
	}
	kat::collaborative::block::barrier();
	auto time_after_sleep = clock64();
	times_before_sleep[global_thread_id] = time_before_sleep;
	times_after_sleep[global_thread_id] = time_after_sleep;
//	thread_printf("Have slept for %u units. Time before sleep = %20lld, after = %20lld",
//		(unsigned) sleep_duration, time_before_sleep, time_after_sleep);
}

} // namespace kernels


template <typename T, T Value>
struct value_as_type {
	static constexpr const T value { Value };
};

TEST_SUITE("time") {

	TEST_CASE_TEMPLATE("measure_time_and_sleep", ResolutionValueAsType,
		value_as_type<kat::sleep_resolution, kat::sleep_resolution::clock_cycles>,
		value_as_type<kat::sleep_resolution, kat::sleep_resolution::nanoseconds>)
	{
		constexpr const kat::sleep_resolution resolution { ResolutionValueAsType::value };

		auto device { cuda::device::current::get() };
			// TODO: Test shuffles with non-full warps.
		if ((device.properties().compute_architecture().major < 7) and
			(resolution == kat::sleep_resolution::nanoseconds))
		{
			// nanosecond-resolution sleep is only supported starting from Volta/Turing
			return;
		}
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		std::size_t total_num_threads = launch_config.grid_dimensions.volume() * launch_config.block_dimensions.volume();
		auto times_before_sleep = cuda::memory::device::make_unique<kat::clock_value_t[]>(device.id(), total_num_threads);
		auto times_after_sleep = cuda::memory::device::make_unique<kat::clock_value_t[]>(device.id(), total_num_threads);
		auto kernel = ::kernels::measure_time_and_sleep<resolution>;
		cuda::launch(kernel, launch_config,
			times_before_sleep.get(), times_after_sleep.get(), total_num_threads);
		cuda::outstanding_error::ensure_none();
		auto host_times_before_sleep = std::make_unique<kat::clock_value_t[]>(total_num_threads);
		auto host_times_after_sleep = std::make_unique<kat::clock_value_t[]>(total_num_threads);
		cuda::memory::copy(host_times_before_sleep.get(), times_before_sleep.get(), total_num_threads * sizeof(kat::clock_value_t));
		cuda::memory::copy(host_times_after_sleep.get(), times_after_sleep.get(), total_num_threads * sizeof(kat::clock_value_t));

		device.synchronize();

		for(cuda::grid_block_dimension_t block_id = 0; block_id < num_grid_blocks; block_id++) {

			std::vector<kat::clock_value_t> block_times_before_sleep {
				host_times_before_sleep.get() + block_id * block_size,
				host_times_before_sleep.get() + (block_id+1) * block_size
			};
			std::vector<kat::clock_value_t> block_times_after_sleep {
				host_times_after_sleep.get() + block_id * block_size,
				host_times_after_sleep.get() + (block_id+1) * block_size
			};

//			std::cout << "Resolution: "
//				<< (resolution == kat::sleep_resolution::clock_cycles ? "clock_cycles" : "")
//				<< (resolution == kat::sleep_resolution::nanoseconds ? "nanoseconds" : "")
//				<< std::endl;

			for(cuda::grid_dimension_t thread_index = 0; thread_index < block_size; ++thread_index) {
				CHECK(block_times_before_sleep[thread_index] < block_times_after_sleep[thread_index]);
//				std::cout
//					<< "Block " << std::setw(4) << block_id << ", Thread " << std::setw(4) << thread_index << ": "
//					<< "Before sleep: " << std::setw(20) << tbs[thread_index] << ' '
//					<< "After sleep: "  << std::setw(20) << tas[thread_index] << std::endl;
			}

			auto max_time_before_sleep = *std::max_element(block_times_before_sleep.begin(), block_times_before_sleep.end());
			auto min_time_after_sleep = *std::min_element(block_times_after_sleep.begin(), block_times_after_sleep.end());
			CHECK_LT(max_time_before_sleep, min_time_after_sleep);

//			std::cout
//				<< " Max time before sleep: " << std::setw(20) << max_time_before_sleep
//				<< " Min time after sleep: " << std::setw(20) << min_time_after_sleep << std::endl;
		}
	}

} // TEST_SUITE("time")
