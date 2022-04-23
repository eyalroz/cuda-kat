#include <kat/on_device/c_standard_library/printf.cuh>

#include <cstring>
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>
#include <limits>

#define TEST_WITH_NON_STANDARD_FORMAT_STRINGS         1
#define PRINTF_SUPPORT_DECIMAL_SPECIFIERS             1
#define PRINTF_SUPPORT_EXPONENTIAL_SPECIFIERS         1
#define PRINTF_SUPPORT_WRITEBACK_SPECIFIER            1
#define PRINTF_SUPPORT_MSVC_STYLE_INTEGER_SPECIFIERS  1
#define PRINTF_SUPPORT_LONG_LONG                      1
#define PRINTF_ALIAS_STANDARD_FUNCTION_NAMES          0

#define PRINTF_INTEGER_BUFFER_SIZE                   32
#define PRINTF_DECIMAL_BUFFER_SIZE                   32
#define PRINTF_DEFAULT_FLOAT_PRECISION                6
#define PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL        9



// Multi-compiler-compatible local warning suppression

#if defined(_MSC_VER)
    #define DISABLE_WARNING_PUSH           __pragma(warning( push ))
    #define DISABLE_WARNING_POP            __pragma(warning( pop ))
    #define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))

    // TODO: find the right warning number for this
    #define DISABLE_WARNING_PRINTF_FORMAT
    #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
    #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
    #define DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER

#elif defined(__NVCC__)
    #define DO_PRAGMA(X) _Pragma(#X)
    #define DISABLE_WARNING_PUSH           DO_PRAGMA(push)
    #define DISABLE_WARNING_POP            DO_PRAGMA(pop)
  #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
    #define DISABLE_WARNING(warning_code)  DO_PRAGMA(nv_diag_suppress warning_code)
  #else
    #define DISABLE_WARNING(warning_code)  DO_PRAGMA(diag_suppress warning_code)
  #endif

    #define DISABLE_WARNING_PRINTF_FORMAT             DISABLE_WARNING(bad_printf_format_string)
    #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
    #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
    #define DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER


#elif defined(__GNUC__) || defined(__clang__)
    #define DO_PRAGMA(X) _Pragma(#X)
    #define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
    #define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop)
    #define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)

    #define DISABLE_WARNING_PRINTF_FORMAT    DISABLE_WARNING(-Wformat)
    #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS DISABLE_WARNING(-Wformat-extra-args)
#if defined(__clang__)
    #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
    #define DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER DISABLE_WARNING(-Wformat-invalid-specifier)
#else
    #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW DISABLE_WARNING(-Wformat-overflow)
    #define DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER
#endif
#else
    #define DISABLE_WARNING_PUSH
    #define DISABLE_WARNING_POP
    #define DISABLE_WARNING_PRINTF_FORMAT
    #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
    #define DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER
#endif

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT
DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
DISABLE_WARNING_PRINTF_FORMAT_INVALID_SPECIFIER
#endif

#if defined(_MSC_VER)
DISABLE_WARNING(4996) // Discouragement of use of std::sprintf()
DISABLE_WARNING(4310) // Casting to smaller type
DISABLE_WARNING(4127) // Constant conditional expression
#endif

char* make_device_string(char const* s)
{
  if (s == nullptr) {
    return nullptr;
  }

  // Maybe it's _already_ a device string?

  cudaPointerAttributes attrs;
  auto error = cudaPointerGetAttributes (&attrs, s);
  if (error != cudaSuccess) {
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error)));
  }
  switch(attrs.type) {
    case cudaMemoryTypeUnregistered: // host mem, not registered with CUDA
    case cudaMemoryTypeHost : break;
    case cudaMemoryTypeDevice :
    case cudaMemoryTypeManaged :
      throw std::invalid_argument("Got a pointer which is already GPU device-side accessible");
    default:
      throw std::invalid_argument("Get a pointer to an unsupported/unregistered memory type");
  }

  size_t size = strlen(s) + 1;
  void* dsptr;
  cudaMalloc(&dsptr, size);
  cudaMemcpy(dsptr, s, size, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  return (char *) dsptr;
}

inline char* mkstr(char const* s) { return make_device_string(s); }

struct poor_mans_string_view {
  char* data;
  size_t size;
};

struct sv_and_pos { 
	const poor_mans_string_view sv; 
	size_t pos; 
};

__device__ void append_to_buffer(char c, void* type_erased_svnp)
{ 
  auto& svnp = *(static_cast<sv_and_pos*>(type_erased_svnp));
  if (svnp.pos < svnp.sv.size) {
    svnp.sv.data[svnp.pos++] = c; 
  }
}

// output function type
typedef void (*out_fct_type)(char character, void* arg);

enum class invokable {
  sprintf_, vsprintf_, snprintf_, vsnprintf_
};

__device__ __host__ char const* name(invokable inv) 
{
  switch(inv) {
  case invokable::sprintf_:   return "sprintf_";
  case invokable::snprintf_:  return "snprintf_";
  case invokable::vsprintf_:  return "vsprintf_";
  case invokable::vsnprintf_: return "vsnprintf_";
  }                     
  return "unknown";
}

__device__ int vsprintf_wrapper(char* buffer, char const* format, ...)
{
  va_list args;
  va_start(args, format);
  int ret = kat::vsprintf(buffer, format, args);
  va_end(args);
  return ret;
}

__device__ int vnsprintf_wrapper(char* buffer, size_t buffer_size, char const* format, ...)
{
  va_list args;
  va_start(args, format);
  int ret = kat::vsnprintf(buffer, buffer_size, format, args);
  va_end(args);
  return ret;
}

namespace kernels {

template <typename... Ts>
__global__ void 
invoke(
  int        * __restrict__  result, 
  invokable                  which, 
  char       * __restrict__  buffer, 
  size_t                     buffer_size, 
  char const * __restrict__  format, 
  Ts...                      args)
{
  switch(which) {
    // printf
    // vnprintf_with_scratch
    // nprintf_with_scratch
  case invokable::sprintf_:   *result = kat::sprintf(buffer, format, args...); break;
  case invokable::snprintf_:  *result = kat::snprintf(buffer, buffer_size, format, args...); break;
  case invokable::vsprintf_:  *result = vsprintf_wrapper(buffer, format, args...); break;
  case invokable::vsnprintf_: *result = vnsprintf_wrapper(buffer, buffer_size, format, args...); break;
  }
}

} // namespace kernels

template <typename... Ts>
int invoke_on_device(invokable which, char* buffer, size_t buffer_size, char const* format, Ts... args)
{
  char* buffer_d;
  int* result_d;
  int result;
  size_t format_size = strlen(format) + 1;
  cudaGetLastError(); // Clearing/ignoring earlier errors
  cudaMalloc(&result_d, sizeof(int));
  if (buffer != nullptr or buffer_size == 0) {
    cudaMalloc(&buffer_d, buffer_size);
    cudaMemcpy(buffer_d, buffer, buffer_size, cudaMemcpyDefault);
  } else {
	  buffer_d = nullptr;
  }
  char* format_d = make_device_string(format);
  // std::cout << "Copying done, now launching kernel." << std::endl;
  kernels::invoke<<<1, 1>>>(result_d, which, buffer_d, buffer_size, format_d, args...); // Note: No perfect forwarding.
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) { 
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); 
  }
  if (buffer != nullptr) {
    cudaMemcpy(buffer, buffer_d, buffer_size, cudaMemcpyDefault);
  }
  cudaMemcpy(&result, result_d, sizeof(int), cudaMemcpyDefault);
  cudaFree(buffer_d);
  cudaFree(format_d);
  cudaFree(result_d);
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) { 
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); 
  }
  return result;
}

constexpr const size_t base_buffer_size { 100 };


template <typename... Ts>
int invoke_on_device(invokable which, char* buffer, char const* format, Ts... args)
{
  return invoke_on_device<Ts...>(which, buffer, base_buffer_size, format, args...);
}

template <typename... Ts>
int printing_check(
  const char *expected,
  const char *,
  invokable invokable_printer,
  char *buffer,
  size_t buffer_size,
  const char *format,
  Ts &&... params)
{
  if (buffer == nullptr and expected != nullptr) {
    std::cerr << "Internal error: A null buffer is expected to become non-null" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto ret = invoke_on_device(invokable_printer, buffer, buffer_size, format, std::forward<Ts>(params)...);
  // std::cout << "invoked_on_device with format \"" << format << "\" done." << std::endl;
  if (buffer == nullptr) {
    return ret;
  }
  if (buffer_size != base_buffer_size) {
    buffer[base_buffer_size - 1] = '\0';
  }
  //  std::cout << "----\n";
  //  std::cout << "Resulting buffer contents: " << '"' << buffer << '"' << '\n';
  if (strncmp(buffer, expected, buffer_size) != 0) {
    buffer[strlen(expected)] = '\0';
    std::cerr << "Failed with printer " << name(invokable_printer) <<
	    " with format \"" << format << "\":\n"
		<< "Actual:   \"" << buffer   << "\"\n"
		<< "Expected: \"" << expected << "\"\n" << std::flush;
    exit(EXIT_FAILURE);
  }
  return ret;
}

template <typename... Ts>
void printing_and_ret_check(
  int expected_return_value,
  const char *expected,
  const char *,
  invokable invokable_printer,
  char *buffer,
  size_t buffer_size,
  const char *format,
  Ts &&... params)
{
    auto ret = printing_check(expected, nullptr, invokable_printer, buffer, buffer_size, format, std::forward<Ts>(params)...);
    if (ret != expected_return_value) {
      std::cerr << "Unexpected return value with printer " << name(invokable_printer) <<
      " and format \"" << format << "\":\n    Actual: " << ret << "\n    Expected: " <<
      expected_return_value << std::endl;
      exit(EXIT_FAILURE);
    }
}

// Note testing fctprintf() and vfctprintf()

#define STRINGIFY(_x) #_x

// You can add the following into PRINTING_CHECK_WITH_BUF_SIZE to keep
// track of which checks pass before a failure:
//
// printf("printing check at file %s line %d\n", __FILE__, (int) __LINE__); \

#define PRINTING_CHECK_WITH_BUF_SIZE(expected,dummy,printer,buffer,buffer_size,format,...) \
do {                                                                                       \
  printing_check(expected, STRINGIFY(dummy), invokable::printer, buffer, buffer_size, format, ##__VA_ARGS__); \
} while(false);

#define PRINTING_CHECK(expected,dummy,printer,buffer,format,...) \
PRINTING_CHECK_WITH_BUF_SIZE(expected, dummy, printer, buffer, base_buffer_size, format, ##__VA_ARGS__)

void testcase_snprintf() {
  char buffer[base_buffer_size];
  PRINTING_CHECK("-1000", ==, snprintf_, buffer, "%d", -1000);
  PRINTING_CHECK_WITH_BUF_SIZE("-1", ==, snprintf_, buffer, 3, "%d", -1000);
}

void testcase_vsprintf() {
  char buffer[base_buffer_size];
  PRINTING_CHECK("-1", ==, vsprintf_, buffer, "%d", -1 );
  PRINTING_CHECK("3 -1000 test", ==, vsprintf_, buffer, "%d %d %s", 3, -1000, mkstr("test") );
}

void testcase_vsnprintf() {
  char buffer[base_buffer_size];
  PRINTING_CHECK("-1", ==, vsnprintf_, buffer, "%d", -1);
  PRINTING_CHECK("3 -1000 test", ==, vsnprintf_, buffer, "%d %d %s", 3, -1000, mkstr("test"));
}

void testcase_simple_sprintf() {
  char buffer[base_buffer_size];
  memset(buffer, 0xCC, base_buffer_size);
  PRINTING_CHECK("42", ==, sprintf_, buffer, "%d", 42);
}


[[maybe_unused]] void testcase_brute_force_float() {
  char buffer[base_buffer_size];

  // brute force float
  bool fail = false;
  std::stringstream str;
  str.precision(5);
  for (float i = -100000; i < 100000; i += 1) {
   invoke_on_device(invokable::sprintf_, buffer, "%.5f", (double)(i / 10000));
    str.str("");
    str << std::fixed << i / 10000;
    fail = fail || !!strcmp(buffer, str.str().c_str());
  }
  if (fail) {
    std::cerr << "sprintf(\"" << "%.5f\" (double)(i / 10000)) failed." << std::endl;
    exit(EXIT_FAILURE);
  }


#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  // brute force exp
  str.setf(std::ios::scientific, std::ios::floatfield);
  for (float i = -1e20; i < (float) 1e20; i += (float) 1e15) {
   invoke_on_device(invokable::sprintf_, buffer, "%.5f", (double) i);
    str.str("");
    str << i;
    fail = fail || !!strcmp(buffer, str.str().c_str());
  }
  if (fail) {
    std::cerr << "sprintf(\"" << "%.5f\" (double) i) failed." << std::endl;
    exit(EXIT_FAILURE);
  }
#endif
}


#ifndef STRINGIFY
#define STRINGIFY(_x) #_x
#endif

#define PRINTF_TEST_CASE(unstringified_name) void testcase_ ## unstringified_name()
#define CHECK(...) \
do {               \
  if (!(__VA_ARGS__)) { \
    std::cerr << "Check failed at " << __FILE__ << ':' << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
} while(0)

#define CAPTURE_AND_PRINT(printer_, _buffer, format, ...) \
do {                                         \
  auto ret = invoke_on_device(invokable::printer_, _buffer, format, __VA_ARGS__); \
  if (not ret) {                         \
    std::cerr << "Failed at " << __FILE__ << ':' << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
} while(false);

#define CAPTURE_AND_PRINT_WITH_RETVAL(retval, printer_, ...) \
do { \
  retval = invoke_on_device(invokable::printer_, __VA_ARGS__); \
} while(false);


#include "printf_testcases.hpp"

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
DISABLE_WARNING_POP
#endif

int main()
{
  // testcases defined within this .cu file

//  testcase_fctprintf();
//  testcase_vfctprintf();
  testcase_snprintf();
  testcase_vsprintf();
  testcase_vsnprintf();
  testcase_simple_sprintf();
// Too long and costly for running on the GPU...
// testcase_brute_force_float();

  // testcases defined in the main testcases  header

  testcase_space_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_space_flag__non_standard_format();
#endif
  testcase_plus_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_plus_flag__non_standard_format();
#endif
  testcase_zero_flag();
  testcase_minus_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_minus_flag_and_non_standard_zero_modifier_for_integers();
#endif
  testcase_sharp_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_sharp_flag__non_standard_format();
#endif
#if PRINTF_SUPPORT_LONG_LONG
  testcase_sharp_flag_with_long_long();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_sharp_flag_with_long_long__non_standard_format();
#endif
#endif // PRINTF_SUPPORT_LONG_LONG
  testcase_specifier();
  testcase_width();
  testcase_width_20();
  testcase_width_asterisk_20();
  testcase_width_minus_20();
  testcase_width_0_minus_20();
  testcase_padding_20();
  testcase_padding_dot_20();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_padding_sharp_020__non_standard_format();
#endif
  testcase_padding_sharp_020();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_padding_sharp_20__non_standard_format();
#endif
  testcase_padding_sharp_20();
  testcase_padding_20_point_5();
  testcase_padding_negative_numbers();
#if PRINTF_SUPPORT_DECIMAL_SPECIFIERS || PRINTF_SUPPORT_EXPONENTIAL_SPECIFIERS
  testcase_float_padding_negative_numbers();
#endif
  testcase_length();
  testcase_length__non_standard_format();
#if PRINTF_SUPPORT_DECIMAL_SPECIFIERS || PRINTF_SUPPORT_EXPONENTIAL_SPECIFIERS
  testcase_infinity_and_not_a_number_values();
  testcase_floating_point_specifiers_with_31_to_32_bit_integer_values();
  testcase_tiny_floating_point_values();
  testcase_fallback_from_decimal_to_exponential();
  testcase_floating_point_specifiers_precision_and_flags();
#endif
  testcase_integer_types();
  testcase_types__non_standard_format();
  testcase_pointer();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_unknown_flag__non_standard_format();
#endif
  testcase_string_length();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_string_length__non_standard_format();
#endif
  testcase_buffer_length();
  testcase_misc();
  testcase_extremal_signed_integer_values();
  testcase_extremal_unsigned_integer_values();
}

