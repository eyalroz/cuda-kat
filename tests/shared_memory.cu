#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "common.cuh"

#include <kat/on_device/shared_memory.cuh>
#include <kat/on_device/wrappers/atomics.cuh>

using namespace kat;

using shmem_size_t = shared_memory::size_t;


struct sizes_t {
	shmem_size_t dynamic;
	shmem_size_t static_;
	shmem_size_t total;
};

namespace kernels {


template <shmem_size_t StaticSize>
__global__ void determine_sizes(sizes_t* results)
{
	static  __shared__  char arr[StaticSize];
	arr[0] = 0;
	arr[1] = arr[0];
	results->dynamic = shared_memory::dynamic::size();
	results->static_ = shared_memory::static_::size();
	results->total = shared_memory::size();
}

template <>
__global__ void determine_sizes<0>(sizes_t* results)
{
	results->dynamic = shared_memory::dynamic::size();
	results->static_ = shared_memory::static_::size();
	results->total = shared_memory::size();
}

template <typename I>
__global__ void check_overlap(shmem_size_t num_elements_per_warp, shmem_size_t* num_overlaps_encountered_by_warp)
{
	auto warp_shared_mem = shared_memory::dynamic::warp_specific::contiguous<I>(num_elements_per_warp);

	// Note: The rest of this kernel will use as little as possible kat functionality, so as not
	// to mix up testing different parts of the library. The price is some idiosyncracy. Also,
	// we'll let just a single thread of each warp act, so as not to worry about intra-warp collaboration.

	auto am_first_in_warp = (threadIdx.x % warp_size == 0);
	if (not am_first_in_warp) { return; }

	// clear the warp's shared memory
	for(shmem_size_t i = 0; i < num_elements_per_warp; i++) { warp_shared_mem[i] = I{0}; }
	__syncthreads();

	// touch every I-element in this warp's shared memory, in a way in which overlaps between warps'
	// shared memory stretches would be detected

	for(shmem_size_t i = 0; i < num_elements_per_warp; i++) {
		atomic::increment(&(warp_shared_mem[i]));
	}
	__syncthreads();

	// This could have been an std::count_if

	shmem_size_t num_overlaps_encountered { 0 };
	for(shmem_size_t i = 0; i < num_elements_per_warp; i++) {
		if (warp_shared_mem[i] != (I{1})) { num_overlaps_encountered++; }
	}
	auto warp_index = threadIdx.x / warp_size;
	num_overlaps_encountered_by_warp[warp_index] = num_overlaps_encountered;
}


} // namespace kernels


TEST_SUITE("shared_memory") {

TEST_CASE("correctly determining static and dynamic sizes")
{
	constexpr const shmem_size_t allocation_quantum { 256 };
		// It seems that static shared memory is allocated in quanta; and that the dynamic shared memory
		// can fill in the gap in the last quantum if necessary

	constexpr const shmem_size_t dynamic_shmem_sizes[] = { 0, 1, allocation_quantum, allocation_quantum+1 };
	constexpr const shmem_size_t used_static_shmem_sizes[] = { 1, allocation_quantum - 1, allocation_quantum, allocation_quantum + 1 };

//	Target architecture 	Shared memory allocation unit size
//	sm_2x 	128 bytes
//	sm_3x, sm_5x, sm_6x, sm_7x 	256 bytes


	auto device { cuda::device::current::get() };
	auto device_side_results { cuda::memory::device::make_unique<sizes_t>(device) };
	for (auto dynamic_shared_mem_size : dynamic_shmem_sizes) {
		// TODO: The following should really be a "for constexpr" - but that doesn't exist yet
		for (auto i = 0; i < array_length(used_static_shmem_sizes); i++) {
			auto launch_config { cuda::make_launch_config(1, 1, dynamic_shared_mem_size) };
			sizes_t host_side_results;
			switch(i) {
			case 0: cuda::launch(kernels::determine_sizes<used_static_shmem_sizes[0]>, launch_config, device_side_results.get()); break;
			case 1: cuda::launch(kernels::determine_sizes<used_static_shmem_sizes[1]>, launch_config, device_side_results.get()); break;
			case 2: cuda::launch(kernels::determine_sizes<used_static_shmem_sizes[2]>, launch_config, device_side_results.get()); break;
			case 3: cuda::launch(kernels::determine_sizes<used_static_shmem_sizes[3]>, launch_config, device_side_results.get()); break;
			}
			auto static_shared_mem_size { used_static_shmem_sizes[i] };
			auto aligned_total_size { round_up<shmem_size_t>(static_shared_mem_size + dynamic_shared_mem_size, allocation_quantum) };
			cuda::memory::copy(&host_side_results, device_side_results.get(), sizeof(sizes_t));
			CHECK(host_side_results.dynamic == dynamic_shared_mem_size);
// TODO: Figure out the exact rule for how much static shared memory is actually allocated. Apparently
// it depends on the existence of other kernels (???)
//			CHECK(host_side_results.static_ == aligned_total_size - dynamic_shared_mem_size);
//			CHECK(host_side_results.total   == aligned_total_size);
		}
	}
}

TEST_CASE_TEMPLATE("allocations of per-warp shared memory do not intersect", I, int32_t, int64_t)
{
	cuda::device_t<> device { cuda::device::current::get() };
	auto max_shared_mem = device.properties().sharedMemPerBlock;
	auto num_warps = device.properties().max_warps_per_block();
	shmem_size_t shared_mem_per_warp = max_shared_mem / num_warps;
	shmem_size_t num_shmem_elements_per_warp = shared_mem_per_warp / sizeof(I);
	auto block_size = num_warps * warp_size;
	auto launch_config { cuda::make_launch_config(1, block_size, num_shmem_elements_per_warp * sizeof(I) * num_warps) };
	auto device_side_results { cuda::memory::device::make_unique<shmem_size_t[]>(device, num_warps) };
	auto host_side_results { std::unique_ptr<shmem_size_t[]>(new shmem_size_t[num_warps]) };
	cuda::launch(kernels::check_overlap<I>, launch_config, num_shmem_elements_per_warp, device_side_results.get());
	cuda::memory::copy(host_side_results.get(), device_side_results.get(), sizeof(shmem_size_t) * num_warps);
	auto num_overlaps_found = std::accumulate(host_side_results.get(), host_side_results.get() + num_warps, 0);
	CHECK(num_overlaps_found == 0);
}

} // TEST_SUITE("shared_memory")
