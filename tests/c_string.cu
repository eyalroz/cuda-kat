#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common.cuh"
#include <kat/on_device/c_standard_library/string.cuh>

// TODO: Pass expected values to the device, and results back to the host for
// comparison. At the moment, errors only tell you which check failed, not
// what the actual and expected values were, nor what arguments the tested
// function was invoked with.

// Note:
// Testcases are adapted from those used in the Public-Domain C Library. See:
// https://rootdirectory.ddns.net/dokuwiki/doku.php?id=pdclib:start

constexpr const std::size_t max_num_checks_per_test { 100 };

namespace kernels {

__global__ void test_strcmp(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s1, const char* s2, bool (*predicate)(int)  ) {
		*(result++) =  predicate(kat::c_std_lib::strcmp(s1, s2));
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";
	const auto cmpabcde = "abcde";
	const auto cmpabcd_ = "abcd\xfc";
	const auto empty    = "";

	auto is_negative = [](int i) { return i < 0;  };
	auto is_positive = [](int i) { return i > 0;  };
	auto is_zero     = [](int i) { return i == 0; };

	constexpr int line_before_first_check = __LINE__;
	single_check(abcde, cmpabcde, is_zero     );
	single_check(abcde, abcdx,    is_negative );
	single_check(abcdx, abcde,    is_positive );
	single_check(empty, abcde,    is_negative );
	single_check(abcde, empty,    is_positive );
	single_check(abcde, cmpabcd_, is_negative );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strncmp(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s1, const char* s2, std::size_t n, bool (*predicate)(int)  ) {
		*(result++) =  predicate(kat::c_std_lib::strncmp(s1, s2, n));
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";
	const auto cmpabcde = "abcde";
	const auto cmpabcd_ = "abcd\xfc";
	const auto empty    = "";
    const auto x        = "x";

	auto is_negative = [](int i) { return i < 0;  };
	auto is_positive = [](int i) { return i > 0;  };
	auto is_zero     = [](int i) { return i == 0; };

	constexpr int line_before_first_check = __LINE__;
    single_check(abcde, cmpabcde,  5, is_zero);
    single_check(abcde, cmpabcde, 10, is_zero);
    single_check(abcde, abcdx,     5, is_negative);
    single_check(abcdx, abcde,     5, is_positive);
    single_check(empty, abcde,     5, is_negative);
    single_check(abcde, empty,     5, is_positive);
    single_check(abcde, abcdx,     4, is_zero);
    single_check(abcde, x,         0, is_zero);
    single_check(abcde, x,         1, is_negative);
    single_check(abcde, cmpabcd_, 10, is_negative);
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_memcmp(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const void* s1, const void* s2, std::size_t n, bool (*predicate)(int)  ) {
		*(result++) =  predicate(kat::c_std_lib::memcmp(s1, s2, n));
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";
    const auto xxxxx    = "xxxxx";

	auto is_negative = [](int i) { return i < 0;  };
	auto is_positive = [](int i) { return i > 0;  };
	auto is_zero     = [](int i) { return i == 0; };

	constexpr int line_before_first_check = __LINE__;
    single_check(abcde, abcdx, 5, is_negative);
    single_check(abcde, abcdx, 4, is_zero);
    single_check(abcdx, xxxxx, 0, is_zero);
    single_check(xxxxx, abcde, 1, is_positive);
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strcpy(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](char* dest, const char* src ) {
		auto ret = kat::c_std_lib::strcpy(dest, src);
		*(result++) = (ret == dest);
	};
	auto single_check_char_value = [&](const char* strcpy_dest, std::size_t pos, char expected_value) {
		*(result++) =  (strcpy_dest[pos] == expected_value);
	};

	const auto abcde    = "abcde";

    char s[] = "xxxxx";

	constexpr int line_before_first_check = __LINE__;
    single_check_invocation(s, ""   );
    single_check_char_value(s, 0, '\0' );
    single_check_char_value(s, 1, 'x'  );
    single_check_invocation(s, abcde);
    single_check_char_value(s, 0, 'a'  );
    single_check_char_value(s, 4, 'e'  );
    single_check_char_value(s, 5, '\0' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strncpy(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](char* dest, const char* src, std::size_t n ) {
		auto ret = kat::c_std_lib::strncpy(dest, src, n);
		*(result++) = (ret == dest);
	};
	auto single_check_char_value = [&](const char* strncpy_dest, std::size_t pos, char expected_value) {
		*(result++) =  (strncpy_dest[pos] == expected_value);
	};

	const auto abcde    = "abcde";

	char s[] = "xxxxxxx";

	constexpr int line_before_first_check = __LINE__;
    single_check_invocation( s, "", 1 );
    single_check_char_value( s, 0, '\0' );
    single_check_char_value( s, 1, 'x' );
    single_check_invocation( s, abcde, 6 );
    single_check_char_value( s, 0, 'a' );
    single_check_char_value( s, 4, 'e' );
    single_check_char_value( s, 5, '\0' );
    single_check_char_value( s, 6, 'x' );
    single_check_invocation( s, abcde, 7 );
    single_check_char_value( s, 6, '\0' );
    single_check_invocation( s, "xxxx", 3 );
    single_check_char_value( s, 0, 'x' );
    single_check_char_value( s, 2, 'x' );
    single_check_char_value( s, 3, 'd' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strlen(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, std::size_t expected) {
		*(result++) = (kat::c_std_lib::strlen(s) == expected);
	};

	constexpr int line_before_first_check = __LINE__;
	single_check( "abcde",  5 );
	single_check( "",       0 );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strcat(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](char* dest, const char* src ) {
		auto ret = kat::c_std_lib::strcat(dest, src);
		*(result++) = (ret == dest);
	};
	auto single_check_char_value = [&](const char* strcat_dest, std::size_t pos, char expected_value) {
		*(result++) =  (strcat_dest[pos] == expected_value);
	};

	const auto abcde = "abcde";
	const auto abcdx = "abcdx";

    char s[] = "xx\0xxxxxx";

	constexpr int line_before_first_check = __LINE__;
    single_check_invocation(s, abcde);
    single_check_char_value(s, 2, 'a' );
    single_check_char_value(s, 6, 'e' );
    single_check_char_value(s, 7, '\0' );
    single_check_char_value(s, 8, 'x' );
    s[0] = '\0'; single_check_invocation(s, abcdx);
    single_check_char_value(s, 4, 'x' );
    single_check_char_value(s, 5, '\0' );
    single_check_invocation(s, "\0");
    single_check_char_value(s, 5, '\0' );
    single_check_char_value(s, 6, 'e' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strncat(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](char* dest, const char* src, std::size_t n ) {
		auto ret = kat::c_std_lib::strncat(dest, src, n);
		*(result++) = (ret == dest);
	};
	auto single_check_char_value = [&](const char* strncat_dest, std::size_t pos, char expected_value) {
		*(result++) =  (strncat_dest[pos] == expected_value);
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";

    char s[] = "xx\0xxxxxx";

	constexpr int line_before_first_check = __LINE__;
    single_check_invocation(s, abcde, 10);
    single_check_char_value(s, 2, 'a'  );
    single_check_char_value(s, 6, 'e'  );
    single_check_char_value(s, 7, '\0' );
    single_check_char_value(s, 8, '\0' ); // Additional nulls must have been written, even beyond the end of the concatenation string
    s[0] = '\0'; single_check_invocation(s, abcdx, 10);
    single_check_char_value(s, 4, 'x'  );
    single_check_char_value(s, 5, '\0' );
    single_check_invocation(s, "\0", 10);
    single_check_char_value(s, 5, '\0' );
    single_check_char_value(s, 6, '\0' ); // Additional nulls must have been written, even beyond the end of the concatenation string
    single_check_invocation(s, abcde, 0);
    single_check_char_value(s, 4, 'x'  );
    single_check_char_value(s, 5, '\0' );
    single_check_char_value(s, 6, '\0' ); // Additional nulls must have been written, even beyond the end of the concatenation string
    single_check_invocation(s, abcde, 3);
    single_check_char_value(s, 5, 'a'  );
    single_check_char_value(s, 7, 'c'  );
    single_check_char_value(s, 8, '\0' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}


__global__ void test_memcpy(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](char* dest, const char* src, std::size_t n ) {
		auto ret = kat::c_std_lib::memcpy(dest, src, n);
		*(result++) = (ret == dest);
	};
	auto single_check_char_value = [&](const char* memcpy_dest, std::size_t pos, char expected_value) {
		*(result++) =  (memcpy_dest[pos] == expected_value);
	};

	const auto abcde    = "abcde";

	char s[] = "xxxxxxxxxxx";

	constexpr int line_before_first_check = __LINE__;
	single_check_invocation(s, abcde, 6);
	single_check_char_value(s,  4, 'e' );
	single_check_char_value(s,  5, '\0' );
	single_check_char_value(s,  6, 'x' );
	single_check_invocation(s + 5, abcde, 5);
	single_check_char_value(s,  9, 'e' );
	single_check_char_value(s, 10, 'x' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_memset(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check_invocation = [&](void* s, int c, std::size_t n ) {
		auto ret = kat::c_std_lib::memset(s, c, n);
		*(result++) = (ret == s);
	};
	auto single_check_char_value = [&](const char* memset_dest, std::size_t pos, char expected_value) {
		*(result++) =  (memset_dest[pos] == expected_value);
	};

	char s[] = "xxxxxxxxx";

	constexpr int line_before_first_check = __LINE__;
    single_check_invocation(s, 'o', 10);
    single_check_char_value(s, 0, 'o' );
    single_check_char_value(s, 9, 'o' );
    single_check_invocation(s, '_', 0);
    single_check_char_value(s, 0, 'o' );
    single_check_invocation(s, '_', 1);
    single_check_char_value(s, 0, '_' );
    single_check_invocation(s, '\xfd', 3);
    single_check_char_value(s, 2, '\xfd' );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_memchr(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, int c, std::size_t n, const char* expected ) {
		*(result++) = (kat::c_std_lib::memchr(s, c, n) == expected);
	};

	const auto abcde    = "abcde";

	constexpr int line_before_first_check = __LINE__;
	single_check(abcde, 'c',  5, &abcde[2] );
    single_check(abcde, 'a',  1, &abcde[0] );
    single_check(abcde, 'a',  0, nullptr   );
    single_check(abcde, '\0', 5, nullptr   );
    single_check(abcde, '\0', 6, &abcde[5] );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strchr(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, int c, const char* expected ) {
		*(result++) = (kat::c_std_lib::strchr(s, c) == expected);
	};

    const auto abccd    = "abccd";

	constexpr int line_before_first_check = __LINE__;
	single_check(abccd, 'x',  nullptr   );
	single_check(abccd, 'a',  &abccd[0] );
	single_check(abccd, 'd',  &abccd[4] );
	single_check(abccd, '\0', &abccd[5] );
	single_check(abccd, 'c',  &abccd[2] );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strrchr(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, int c, const char* expected ) {
		*(result++) = (kat::c_std_lib::strrchr(s, c) == expected);
	};

	const auto abcde    = "abcde";
    const auto abccd    = "abccd";

	constexpr int line_before_first_check = __LINE__;
	single_check(abcde, '\0', &abcde[5] );
	single_check(abcde, 'e',  &abcde[4] );
	single_check(abcde, 'a',  &abcde[0] );
	single_check(abccd, 'c',  &abccd[3] );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strpbrk(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, const char* accept, const char* expected ) {
		*(result++) = (kat::c_std_lib::strpbrk(s, accept) == expected);
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";

	constexpr int line_before_first_check = __LINE__;
    single_check(abcde, "x",   nullptr   );
    single_check(abcde, "xyz", nullptr   );
    single_check(abcdx, "x",   &abcdx[4] );
    single_check(abcdx, "xyz", &abcdx[4] );
    single_check(abcdx, "zyx", &abcdx[4] );
    single_check(abcde, "a",   &abcde[0] );
    single_check(abcde, "abc", &abcde[0] );
    single_check(abcde, "cba", &abcde[0] );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strspn(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, const char* accept, std::size_t expected ) {
		*(result++) = (kat::c_std_lib::strspn(s, accept) == expected);
	};

	const auto abcde    = "abcde";

	constexpr int line_before_first_check = __LINE__;
	single_check(abcde, "abc", 3 );
	single_check(abcde, "b",   0 );
	single_check(abcde, abcde, 5 );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strcspn(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* s, const char* reject, std::size_t expected ) {
		*(result++) = (kat::c_std_lib::strcspn(s, reject) == expected);
	};

	const auto abcde    = "abcde";
	const auto abcdx    = "abcdx";

	constexpr int line_before_first_check = __LINE__;
    single_check(abcde, "x",   5 );
    single_check(abcde, "xyz", 5 );
    single_check(abcde, "zyx", 5 );
    single_check(abcdx, "x",   4 );
    single_check(abcdx, "xyz", 4 );
    single_check(abcdx, "zyx", 4 );
    single_check(abcde, "a",   0 );
    single_check(abcde, "abc", 0 );
    single_check(abcde, "cba", 0 );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}


__global__ void test_strstr(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* haystack, const char* needle, const char* expected ) {
//		printf("Haystack: %s , Needle: %s , strstr pos: %d\n", haystack, needle, kat::c_std_lib::strstr(haystack, needle) == nullptr ? -1 : kat::c_std_lib::strstr(haystack, needle) - haystack);
		*(result++) = (kat::c_std_lib::strstr(haystack, needle) == expected);
	};

	char s[] = "abcabcabcdabcde";

	constexpr int line_before_first_check = __LINE__;
	single_check(s, "x",     nullptr );
	single_check(s, "xyz",   nullptr );
	single_check(s, "a",     &s[0]   );
	single_check(s, "abc",   &s[0]   );
	single_check(s, "abcd",  &s[6]   );
	single_check(s, "abcde", &s[10]  );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}

__global__ void test_strrstr(bool* results, std::size_t* num_checks)
{
	bool* result = results;

	auto single_check = [&](const char* haystack, const char* needle, const char* expected ) {
		*(result++) = (kat::c_std_lib::strrstr(haystack, needle) == expected);
	};

	const auto s = "abcabcabcdabcde";

	constexpr int line_before_first_check = __LINE__;
	single_check(s, "x",       nullptr );
	single_check(s, "xyz",     nullptr );
	single_check(s, "a",       &s[10]  );
	single_check(s, "abc",     &s[10]  );
	single_check(s, "abca",    &s[3]   );
	single_check(s, "abcab",   &s[3]   );
	single_check(s, "abcabca", &s[0]   );
    constexpr int line_after_last_check = __LINE__;
    *num_checks = line_after_last_check - line_before_first_check - 1;
}


} // namespace kernels

TEST_SUITE("c_string") {

using kernel_type = void (*)(bool*, std::size_t*);

void conduct_test(kernel_type kernel, const char* kernel_name)
{
	cuda::device_t<> device { cuda::device::current::get() };
	auto block_size { 1 };
	auto num_grid_blocks { 1 };
	auto launch_config { cuda::make_launch_config(block_size, num_grid_blocks) };
	auto device_side_results { cuda::memory::device::make_unique<bool[]>(device, max_num_checks_per_test) };
	auto device_side_num_checks { cuda::memory::device::make_unique<std::size_t>(device) };
	bool host_side_results[max_num_checks_per_test];
	std::size_t host_side_num_checks;

	cuda::launch(
		kernel,
		launch_config,
		device_side_results.get(), device_side_num_checks.get()
		);

	cuda::memory::copy(host_side_results, device_side_results.get(), sizeof(bool) * max_num_checks_per_test);
	cuda::memory::copy_single(&host_side_num_checks, device_side_num_checks.get());

	for(std::size_t i = 0; i < host_side_num_checks; i++) {
		CHECK(host_side_results[i] == true);
		if (not host_side_results[i]) {
			auto width_4 { std::setw(4) };
			auto i_plus_1 { i+1 };
			CHECK_MESSAGE(false, kernel_name << " check " << width_4 << i_plus_1 << " (1-based) of " << host_side_num_checks << " failed.");
		}
	}
}

TEST_CASE("strcmp" ) { conduct_test(kernels::test_strcmp,  "strcmp");  }
TEST_CASE("strncmp") { conduct_test(kernels::test_strncmp, "strncmp"); }
TEST_CASE("memcmp" ) { conduct_test(kernels::test_memcmp,  "memcmp");  }
TEST_CASE("strcpy" ) { conduct_test(kernels::test_strcpy,  "strcpy");  }
TEST_CASE("strncpy") { conduct_test(kernels::test_strncpy, "strncpy"); }
TEST_CASE("strlen" ) { conduct_test(kernels::test_strlen,  "strlen");  }
TEST_CASE("strcat" ) { conduct_test(kernels::test_strcat,  "strcat");  }
TEST_CASE("strncat") { conduct_test(kernels::test_strncat, "strncat"); }
TEST_CASE("memcpy" ) { conduct_test(kernels::test_memcpy,  "memcpy");  }
TEST_CASE("memset" ) { conduct_test(kernels::test_memset,  "memset");  }
TEST_CASE("memchr" ) { conduct_test(kernels::test_memchr,  "memchr");  }
TEST_CASE("strchr" ) { conduct_test(kernels::test_strchr,  "strchr");  }
TEST_CASE("strrchr") { conduct_test(kernels::test_strrchr, "strrchr"); }
TEST_CASE("strpbrk") { conduct_test(kernels::test_strpbrk, "strpbrk"); }
TEST_CASE("strspn" ) { conduct_test(kernels::test_strspn,  "strspn");  }
TEST_CASE("strcspn") { conduct_test(kernels::test_strcspn, "strcspn"); }
TEST_CASE("strstr" ) { conduct_test(kernels::test_strstr,  "strstr");  }
TEST_CASE("strrstr") { conduct_test(kernels::test_strrstr, "strrstr"); }


} // TEST_SUITE("c_string")
