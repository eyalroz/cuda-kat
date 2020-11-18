#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "util/type_name.hpp"
#include "util/miscellany.cuh"
#include <kat/on_device/streams/printfing_ostream.cuh>
#include <kat/on_device/collaboration/block.cuh>
#include <kat/on_device/time.cuh>

#include <doctest.h>
#include <cuda/api_wrappers.hpp>

constexpr const auto num_grid_blocks {  2 };
constexpr const auto block_size      {  2 };

constexpr const std::size_t stringstream_buffer_size { 50 };

KAT_DEV kat::stringstream& operator<<(kat::stringstream& os, const util::constexpr_string& arg)
{
	return os << strf::range<const char*>(arg.begin(), arg.end());
}

namespace kernels {


__global__ void stream_different_types_to_stringstream()
{
	char buff[40] = "char buffer original contents";
	kat::stringstream ss(stringstream_buffer_size);
	ss << "A string literal.\n";
	ss << "A single character - the letter a:       " << 'a' << "\n";
	ss << "An array of of characters on the stack:  \"" << buff << "\"\n";
	ss << "Positive signed int literal:             " << 123 << '\n';
	ss << "Negative signed int literal:             " << -456 << '\n';
	ss << "A float-type value (1/3):                " << ( ((float) 1) / 3 ) << '\n';
	ss << "A double-type value (1/3):               " << ( ((float) 1) / 3.0 ) << '\n';
	// This is not supported:
	// ss << "A non-latin, non-ASCII character: " << (char32_t) 'ת' << '\n';
    printf("The stringstream contents:\n%s", ss.c_str());
}

template <typename T>
__global__ void stream_to_stringstream_templated()
{
	kat::stringstream ss(stringstream_buffer_size);
	ss << "A default-initialized value of type T (" << util::type_name<T>() << "): " << T{} << '\n';
    printf("The stringstream contents:\n%s", ss.c_str());
}

__global__ void use_formatting_functions()
{
	kat::stringstream ss(stringstream_buffer_size);
	auto width = 8;
	ss << "No actual formatting, just wrapping in strf::fmt():\n";
	ss << strf::fmt(123) << '\n';
	ss << "Hexadecimal:\n";
	ss << strf::hex(123) << '\n';
	ss << "Octal:\n";
	ss << strf::oct(123) << '\n';
	ss << "Binary:\n";
	ss << strf::bin(123) << '\n';
	ss << "Set fill character, without setting width:\n";
	ss << strf::fmt(123).fill('0') << '\n';
	ss << "Set fill character, set width of " << width << ", right alignment; then a space and more text:\n";
	ss << (strf::fmt(123).fill('0') > width) << " and more text\n";
	ss << "Set fill character, set width of " << width << ", left alignment; then a space more text:\n";
	ss << (strf::fmt(123).fill('0') < width) << " and more text\n";
	ss << "Set fill character, set width of " << width << ", center alignment; then a space more text:\n";
	ss << (strf::fmt(123).fill('0') ^ width) << " and more text\n";
	ss << "Set fill character, set width of " << width << ", internal fill using hex; then a space more text:\n";
	ss << (strf::fmt(123).fill('0').hex() % width) << " and more text\n";
	// TODO: More strf formatting functions

	ss << "strf::right(0,2,'0') gives " << strf::right(0, 2, '0');

    printf("The stringstream contents: \"%s\"\n\n", ss.c_str());
}


__global__ void use_stringstream()
{
	kat::stringstream ss{stringstream_buffer_size}; // longer than some, but not all, of the strings we have here
	ss << "A string literal";
    printf("Position in the stream: %u. stream contents (enclosed in double-quotes): \"%s\"\n", (unsigned) ss.tellp(), ss.c_str());
    printf("Seeking to the beginning of the stream.\n");
    ss.seekp(0);
    printf("Position in the stream: %u. stream contents (enclosed in double-quotes, should be empty): \"%s\"\n", (unsigned) ss.tellp(), ss.c_str());

}

__global__ void use_zero_initialized_stringstream()
{
	kat::stringstream ss{0};
	ss << "A string literal";
    printf("Position in the stream: %u. stream contents (enclosed in double-quotes): \"%s\"\n", (unsigned) ss.tellp(), ss.c_str());
    printf("Seeking to the beginning of the stream.\n");
    ss.seekp(0);
    printf("Position in the stream: %u. stream contents (enclosed in double-quotes, should be empty): \"%s\"\n", (unsigned) ss.tellp(), ss.c_str());
}


__global__ void use_printfing_ostream()
{
	kat::printfing_ostream cout;
	cout << "String literal 1 with newline - to be printed on call of .flush() method\n";
	cout.flush();

	kat::collaborative::block::barrier();
	if (kat::linear_grid::grid_info::thread::index_in_block() == 0) {
		printf("All threads in block %d have flushed cout.\n", blockIdx.x);
	}
	cout << "String literal 2 with newline - to be printed on use of flush manipulator\n";
	cout << kat::flush;

	kat::collaborative::block::barrier();
	if (kat::linear_grid::grid_info::thread::index_in_block() == 0) {
		printf("All threads in block %d have streamed the flush manipulator to their cout.\n", blockIdx.x);
	}

	cout << "String literal 3 with newline - to be printed on destruction\n";
}

__global__ void printfing_ostream_settings()
{
	kat::printfing_ostream cout;
	namespace gi = kat::linear_grid::grid_info;
	cout << "Before any setting\n";
	cout.flush();

	// TODO: What if the text is big enough to cause recycling? That shouldn't matter, but we should try it.
	cout.append_newline_on_flush();
	cout << "SHOULD see \\n between threads' printouts of this sentence. ";
	cout.flush();
	cout.no_newline_on_flush();
	cout << "SHOULD NOT see \\n between threads' printouts of this sentence. ";
	cout.flush();

	if (kat::linear_grid::grid_info::thread::is_first_in_grid()) {
		cout << '\n';
		cout.flush();
	}
		// This will just add a newline after the long paragraph of many threads' non-newline-terminated strings.

	auto block_and_thread_gen = [](kat::stringstream& ss) {
		ss << "Block " << gi::block::index() << ", Thread " << gi::thread::index_in_block() << ": ";
	};

	cout.set_prefix_generator(block_and_thread_gen);
	cout << "A prefix with the thread and block number SHOULD appear before this sentence.\n";
	cout.flush();
	cout.no_prefix();
	cout << "A prefix with the thread and block number SHOULD NOT appear before this sentence.\n";
	cout.flush();

	// resolution and identification
}

__global__ void stream_manipulators_into_printfing_ostream()
{
	kat::printfing_ostream cout;
	using kat::flush;
	namespace gi = kat::linear_grid::grid_info;
	cout << "Before any setting\n" << flush;

	// TODO: What if the text is big enough to cause recycling? That shouldn't matter, but we should try it.
	cout << kat::manipulators::newline_on_flush
	     << "SHOULD see \\n between threads' printouts of this sentence. " << flush
	     << kat::manipulators::no_newline_on_flush
	     << "SHOULD NOT see \\n between threads' printouts of this sentence. " << flush;

	if (kat::linear_grid::grid_info::thread::is_first_in_grid()) {
		// This will just add a newline after the long paragraph of many threads' non-newline-terminated strings.
		cout << kat::manipulators::endl;
	}

	auto block_and_thread_gen = [](kat::stringstream& ss) {
		ss << "Block " << gi::block::index() << ", Thread " << gi::thread::index_in_block() << ": ";
	};

	cout << kat::manipulators::prefix(block_and_thread_gen)
	     << "A prefix with the thread and block number SHOULD appear before this sentence.\n" << flush
	     << kat::manipulators::no_prefix
	     << "A prefix with the thread and block number SHOULD NOT appear before this sentence.\n" << flush;

	// resolution and self-identification

//	cout << strf::join_right(15,'*')("joined right") << '\n' << flush;
}


__device__ const char* to_string(kat::printfing_ostream::resolution res)
{
	switch(res) {
	case kat::printfing_ostream::resolution::thread : return "thread";
	case kat::printfing_ostream::resolution::warp   : return "warp";
	case kat::printfing_ostream::resolution::block  : return "block";
	case kat::printfing_ostream::resolution::grid   : return "grid";
	}
	return nullptr;
}

__global__ void print_at_different_resolutions()
{
	kat::printfing_ostream cout;
	using kat::flush;
	namespace gi = kat::linear_grid::grid_info;

	cout << kat::manipulators::resolution(kat::printfing_ostream::resolution::grid);
	cout << "Printing at grid resolution. The printing thread is (" << blockIdx.x << "," << threadIdx.x << ")\n" << flush;



	cout << kat::manipulators::resolution(kat::printfing_ostream::resolution::warp);
	cout << "Printing at warp resolution. The printing thread is (" << blockIdx.x << "," << threadIdx.x << ")\n" << flush;

	cout << kat::manipulators::resolution(kat::printfing_ostream::resolution::thread);
	cout << "Printing at thread resolution. The printing thread is (" << blockIdx.x << "," << threadIdx.x << ")\n" << flush;
}

__device__ void self_identifying_printfing_ostream_for_resolution(kat::printfing_ostream& os, kat::printfing_ostream::resolution res)
{
	os
		<< kat::manipulators::resolution(res)
		<< kat::linear_grid::manipulators::identify
		<< "Printing to a self-identifying ostream with resolution "
		<< to_string(os.printing_resolution())
		<< kat::endl;
}

__global__ void self_identifying_printfing_ostream()
{
	kat::printfing_ostream cout;
	using kat::flush;
	namespace gi = kat::linear_grid::grid_info;

	self_identifying_printfing_ostream_for_resolution(cout, kat::printfing_ostream::resolution::grid);

	kat::sleep<kat::sleep_resolution::clock_cycles>(1e8);

	self_identifying_printfing_ostream_for_resolution(cout, kat::printfing_ostream::resolution::block);
	__syncthreads();
	self_identifying_printfing_ostream_for_resolution(cout, kat::printfing_ostream::resolution::warp);
	__syncthreads();
	self_identifying_printfing_ostream_for_resolution(cout, kat::printfing_ostream::resolution::thread);
	__syncthreads();
}


} // namespace kernels

TEST_SUITE("printing") {

	TEST_CASE("use_stringstream")// INTEGER_TYPES)
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::use_stringstream, launch_config);
		cuda::outstanding_error::ensure_none();

		// TODO: We could redirect the standard output stream into a buffer before launching the kernel,
		// then check the buffer contains what we want. However, this can probably be interfered with,
		// so I'm not sure it's a good idea even in principle.
		device.synchronize();
	}

	TEST_CASE("use_zero_initialized_stringstream")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::use_zero_initialized_stringstream, launch_config);
		cuda::outstanding_error::ensure_none();

		// TODO: We could redirect the standard output stream into a buffer before launching the kernel,
		// then check the buffer contains what we want. However, this can probably be interfered with,
		// so I'm not sure it's a good idea even in principle.
		device.synchronize();
	}

	TEST_CASE("stream_different_types_to_stringstream")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::stream_different_types_to_stringstream, launch_config);
		cuda::outstanding_error::ensure_none();

		// TODO: We could redirect the standard output stream into a buffer before launching the kernel,
		// then check the buffer contains what we want. However, this can probably be interfered with,
		// so I'm not sure it's a good idea even in principle.
		device.synchronize();
	}

	TEST_CASE_TEMPLATE("stream_to_stringstream_templated", T, long long int, short)
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::stream_to_stringstream_templated<T>, launch_config);
		cuda::outstanding_error::ensure_none();

		device.synchronize();
	}

	TEST_CASE("use_formatting_functions")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		cuda::launch(kernels::use_formatting_functions, single_thread_launch_config());
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}


	TEST_CASE("use_printfing_ostream")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::use_printfing_ostream, launch_config);
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}

	TEST_CASE("printfing_ostream_settings")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::printfing_ostream_settings, launch_config);
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}

	TEST_CASE("stream manipulators into printfing_ostream")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::stream_manipulators_into_printfing_ostream, launch_config);
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}

	TEST_CASE("print_at_different_resolutions")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::print_at_different_resolutions, launch_config);
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}

	TEST_CASE("self-identifying printfing_ostream")
	{
		auto device { cuda::device::current::get() };
		device.reset();
		auto launch_config { cuda::make_launch_config(num_grid_blocks, block_size) };
		cuda::launch(kernels::self_identifying_printfing_ostream, launch_config);
		cuda::outstanding_error::ensure_none();
		device.synchronize();
	}

} // TEST_SUITE("printing")
