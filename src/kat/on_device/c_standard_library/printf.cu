/**
 * @author (c) Eyal Rozenberg <eyalroz1@gmx.com>
 *             2021-2022, Haifa, Palestine/Israel
 * @author (c) Marco Paland (info@paland.com)
 *             2014-2019, PALANDesign Hannover, Germany
 *
 * @note Others have made smaller contributions to this file: see the
 * contributors page at https://github.com/eyalroz/printf/graphs/contributors
 * or ask one of the authors. The original code for exponential specifiers was
 * contributed by Martijn Jasperse <m.jasperse@gmail.com>.
 *
 * @brief Small stand-alone implementation of the printf family of functions
 * (`(v)printf`, `(v)s(n)printf` etc., geared towards use on embedded systems with
 * a very limited resources.
 *
 * @note the implementations are thread-safe; re-entrant; use no functions from
 * the standard library; and do not dynamically allocate any memory.
 *
 * @license The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstdint>
#include <climits>
#include <cmath>
#include <cfloat>
#include <cstdio> // for CUDA's builtin printf()

#include <kat/on_device/c_standard_library/printf.cuh>

namespace kat {

namespace detail_ {
namespace printf {

enum {
  /// 'ntoa' conversion buffer size, this must be big enough to hold one converted
  /// numeric number including padded zeros (dynamically created on stack)
  integer_buffer_size = 32,

  // size of the fixed (on-stack) buffer for printing individual decimal numbers.
  // this must be big enough to hold one converted floating-point value including
  // padded zeros.
  decimal_buffer_size = 32,

  /// Default precision for the floating point conversion specifiers (the C standard sets this at 6)
  default_float_precision = 6,

  num_decimal_digits_in_int64_t = 18,
  max_supported_precision = num_decimal_digits_in_int64_t - 1,
};

// According to the C languages standard, printf() and related functions must be able to print any
// integral number in floating-point notation, regardless of length, when using the %f specifier -
// possibly hundreds of characters, potentially overflowing your buffers. In this implementation,
// all values beyond this threshold are switched to exponential notation.
constexpr const double float_notation_threshold = 1e9;

///////////////////////////////////////////////////////////////////////////////

namespace flags {

static_assert(sizeof(short) == 2, "Unexpected size of short");
static_assert(sizeof(int)   == 4, "Unexpected size of int");
static_assert(sizeof(long)  == 8, "Unexpected size of long");

enum : unsigned {
// internal flag definitions
  zeropad   = 1U << 0U,
  left      = 1U << 1U,
  plus      = 1U << 2U,
  space     = 1U << 3U,
  hash      = 1U << 4U,
  uppercase = 1U << 5U,
  char_     = 1U << 6U,
  short_    = 1U << 7U,
  int_      = 1U << 8U,
  long_     = 1U << 9U,
  long_long = 1U << 10U,
  precision = 1U << 11U,
  adapt_exp = 1U << 12U,
  pointer   = 1U << 13U, // Note: Similar, but not identical, effect as hash
  signed_   = 1U << 14U,
  int8      = char_,
  int16     = short_,
  int32     = int_,
  int64     = long_
};

} // namespace flags

typedef unsigned int flags_t;

namespace base {

enum {
  binary = 2,
  octal = 8,
  decimal = 10,
  hex = 16
};

} // namespace base

typedef uint8_t numeric_base_t;

typedef unsigned long long unsigned_value_t;
typedef long long signed_value_t;

// The printf()-family functions return an `int`; it is therefore
// unnecessary/inappropriate to use size_t - often larger than int
// in practice - for non-negative related values, such as widths,
// precisions, offsets into buffers used for printing and the sizes
// of these buffers. instead, we use:
typedef unsigned int printf_size_t;
enum {
  max_possible_buffer_size = INT_MAX
  // If we were to nitpick, this would actually be INT_MAX + 1,
  // since INT_MAX is the maximum return value, which excludes the
  // trailing '\0'.
};

namespace double_ {

static_assert(FLT_RADIX == 2, "Non-binary-radix floating-point types are unsupported.");
static_assert(DBL_MANT_DIG == 53, "Unsupported double type configuration");

typedef uint64_t uint_t;

enum {
  size_in_bits = 64,
  base_exponent = 1023,
  stored_mantissa_bits = DBL_MANT_DIG - 1,
};

enum : unsigned {
  exponent_mask = 0x7FFU
};

union with_bit_access {
  uint_t U;
  double F;

  static __device__ constexpr with_bit_access wrap(double x)
  {
    with_bit_access dwba = { .F = x };
    return dwba;
  }

  KAT_DEV constexpr __device__ int exp2() const
  {
    // The exponent in an IEEE-754 floating-point number occupies a contiguous
    // sequence of bits (e.g. 52..62 for 64-bit doubles), but with a non-trivial representation: An
    // unsigned offset from some negative value (with the extremal offset values reserved for
    // special use).
    return (int) ((U >> stored_mantissa_bits) & exponent_mask) - base_exponent;
  }

};

// Stores a fixed-precision representation of a double relative
// to a fixed precision (which cannot be determined by examining this structure)
struct components {
  int_fast64_t integral;
  int_fast64_t fractional;
  // ... truncation of the actual fractional part of the double value, scaled
  // by the precision value
  bool is_negative;
};

} // namespace double_

KAT_DEV static inline constexpr

int get_sign_bit(double x)
{
  // The sign is stored in the highest bit
  return (int) (double_::with_bit_access::wrap(x).U >> (double_::size_in_bits - 1));
}

KAT_DEV static inline int get_exp2(double x)
{
  return double_::with_bit_access::wrap(x).exp2();
}

template<typename T>
KAT_DEV constexpr T abs(T x)
{ return x > 0 ? x : -x; }

// Note in particular the behavior here on LONG_MIN or LLONG_MIN; it is valid
// and well-defined, but if you're not careful you can easily trigger undefined
// behavior with -LONG_MIN or -LLONG_MIN
template<typename T>
KAT_DEV constexpr unsigned_value_t abs_for_printing(T x)
{
  return x > 0 ? x : -(signed_value_t) x;
}

// wrapper (used as buffer) for output function type
//
// One of the following must hold:
// 1. max_chars is 0
// 2. buffer is non-null
// 3. function is non-null
//
// ... otherwise bad things will happen.
typedef struct {
  void (*function)(char c, void *extra_arg);

  void *extra_function_arg;
  char *buffer;
  printf_size_t pos;
  printf_size_t max_chars;
} output_gadget_t;

// Note: This function currently assumes it is not passed a '\0' char,
// or alternatively, that '\0' can be passed to the function in the output
// gadget. The former assumption holds within the printf library. It also
// assumes that the output gadget has been properly initialized.
__noinline__
KAT_DEV static inline void
putchar_via_gadget(output_gadget_t *gadget, char c)
{
  printf_size_t write_pos = gadget->pos++;
  // We're _always_ increasing pos, so as to count how may characters
  // _would_ have been written if not for the max_chars limitation
  if (write_pos >= gadget->max_chars) {
    return;
  }
  if (gadget->function != nullptr) {
    // No check for c == '\0' .
    gadget->function(c, gadget->extra_function_arg);
  } else {
    // it must be the case that gadget->buffer != nullptr , due to the constraint
    // on output_gadget_t ; and note we're relying on write_pos being non-negative.
    gadget->buffer[write_pos] = c;
  }
}

// Possibly-write the string-terminating '\0' character
KAT_DEV static inline void
append_termination_with_gadget(output_gadget_t *gadget)
{
  if (gadget->function != nullptr || gadget->max_chars == 0) {
    return;
  }
  if (gadget->buffer == nullptr) {
    return;
  }
  printf_size_t null_char_pos = gadget->pos < gadget->max_chars ? gadget->pos : gadget->max_chars - 1;
  gadget->buffer[null_char_pos] = '\0';
}

KAT_DEV static inline output_gadget_t

discarding_gadget()
{
  output_gadget_t gadget;
  gadget.function = nullptr;
  gadget.extra_function_arg = nullptr;
  gadget.buffer = nullptr;
  gadget.pos = 0;
  gadget.max_chars = 0;
  return gadget;
}

KAT_DEV static inline output_gadget_t

buffer_gadget(char *buffer, size_t buffer_size)
{
  printf_size_t usable_buffer_size = (buffer_size > max_possible_buffer_size) ?
                                     max_possible_buffer_size : (printf_size_t) buffer_size;
  output_gadget_t result = discarding_gadget();
  if (buffer != nullptr) {
    result.buffer = buffer;
    result.max_chars = usable_buffer_size;
  }
  return result;
}

// internal secure strlen
// @return The length of the string (excluding the terminating 0) limited by 'maxsize'
// @note strlen uses size_t, but wes only use this function with printf_size_t
// variables - hence the signature.
KAT_DEV static inline printf_size_t

strnlen_s_(const char *str, printf_size_t maxsize)
{
  const char *s;
  for (s = str; *s && maxsize--; ++s);
  return (printf_size_t) (s - str);
}

// internal test if char is a digit (0-9)
// @return true if char is a digit
KAT_DEV static inline constexpr

bool is_digit_(char ch)
{
  return (ch >= '0') && (ch <= '9');
}


// internal ASCII string to printf_size_t conversion
KAT_DEV static printf_size_t atou_(const char **str)
{
  printf_size_t i = 0U;
  while (is_digit_(**str)) {
    i = i * 10U + (printf_size_t) (*((*str)++) - '0');
  }
  return i;
}


// output the specified string in reverse, taking care of any zero-padding
KAT_DEV static void out_rev_(
  output_gadget_t * output,
  const char *buf,
  printf_size_t len,
  printf_size_t width,
  flags_t flags)
{
  const printf_size_t start_pos = output->pos;

  // pad spaces up to given width
  if (!(flags &flags::left) && !(flags &flags::zeropad)) {
    for (printf_size_t i = len; i<width; i++) {
      putchar_via_gadget(output, ' ');
    }
  }

  // reverse string
  while (len) {
    putchar_via_gadget(output, buf[--len]);
  }

  // append pad spaces up to given width
  if (flags &flags::left) {
    while (output->pos - start_pos<width) {
      putchar_via_gadget(output, ' ');
    }
  }
}


// Invoked by print_integer after the actual number has been printed, performing necessary
// work on the number's prefix (as the number is initially printed in reverse order)
KAT_DEV static void
print_integer_finalization(
output_gadget_t *__restrict__ output,
char *__restrict__ buf,
printf_size_t len,
bool negative,
numeric_base_t base,
printf_size_t precision,
printf_size_t width,
flags_t flags)
{
  printf_size_t unpadded_len = len;

  // pad with leading zeros
  {
    if (!(flags & flags::left)) {
      if (width && (flags & flags::zeropad) && (negative || (flags & (flags::plus | flags::space)))) {
        width--;
      }
      while ((flags & flags::zeropad) && (len < width) && (len < detail_::printf::integer_buffer_size)) {
        buf[len++] = '0';
      }
    }

    while ((len < precision) && (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = '0';
    }

    if (base == base::octal && (len > unpadded_len)) {
      // Since we've written some zeros, we've satisfied the alternative format leading space requirement
      flags &= ~flags::hash;
    }
  }

  // handle hash
  if (flags & (flags::hash | flags::pointer)) {
    if (!(flags & flags::precision) && len && ((len == precision) || (len == width))) {
      // Let's take back some padding digits to fit in what will eventually
      // be the format-specific prefix
      if (unpadded_len < len) {
        len--; // This should suffice for base::octal
      }
      if (len && (base == base::hex || base == base::binary) && (unpadded_len < len)) {
        len--; // ... and an extra one for 0x or 0b
      }
    }
    if ((base == base::hex) && !(flags & flags::uppercase) && (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'x';
    } else if ((base == base::hex) && (flags & flags::uppercase) && (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'X';
    } else if ((base == base::binary) && (len < detail_::printf::integer_buffer_size)) {
      buf[len++] = 'b';
    }
    if (len < detail_::printf::integer_buffer_size) {
      buf[len++] = '0';
    }
  }

  if (len < detail_::printf::integer_buffer_size) {
    if (negative) {
      buf[len++] = '-';
    } else if (flags & flags::plus) {
      buf[len++] = '+';  // ignore the space if the '+' exists
    } else if (flags & flags::space) {
      buf[len++] = ' ';
    }
  }

  out_rev_(output, buf, len, width, flags);
}

// An internal itoa-like function
KAT_DEV static void
print_integer(
output_gadget_t *output,
unsigned_value_t value,
bool negative, numeric_base_t base,
printf_size_t precision,
printf_size_t width,
flags_t flags)
{
  char buf[detail_::printf::integer_buffer_size];
  printf_size_t len = 0U;

  if (!value) {
    if (!(flags & flags::precision)) {
      buf[len++] = '0';
      flags &= ~flags::hash;
      // We drop this flag this since either the alternative and regular modes of the specifier
      // don't differ on 0 values, or (in the case of octal) we've already provided the special
      // handling for this mode.
    } else if (base == base::hex) {
      flags &= ~flags::hash;
      // We drop this flag this since either the alternative and regular modes of the specifier
      // don't differ on 0 values
    }
  } else {
    do {
      const char digit = (char) (value % base);
      buf[len++] = (char) (digit < 10 ? '0' + digit : (flags & flags::uppercase ? 'A' : 'a') + digit - 10);
      value /= base;
    } while (value && (len < detail_::printf::integer_buffer_size));
  }

  print_integer_finalization(output, buf, len, negative, base, precision, width, flags);
}


/**
 * Access a precomputed power of 10
 * @param e A value in the range 0...PRINTF_MAX_PRECOMPUTED_POWER_OF_10
 */
KAT_DEV double power_of_10(int e)
{
  // Note: We could have used constant memory, but it's probably
  // better to be slower and not mess with that limited resource.
  switch (e) {
    case 0:
      return 1e00;
    case 1:
      return 1e01;
    case 2:
      return 1e02;
    case 3:
      return 1e03;
    case 4:
      return 1e04;
    case 5:
      return 1e05;
    case 6:
      return 1e06;
    case 7:
      return 1e07;
    case 8:
      return 1e08;
    case 9:
      return 1e09;
    case 10:
      return 1e10;
    case 11:
      return 1e11;
    case 12:
      return 1e12;
    case 13:
      return 1e13;
    case 14:
      return 1e14;
    case 15:
      return 1e15;
    case 16:
      return 1e16;
    case 17:
      return 1e17;
  }
  return 1; // Shouldn't get here
}

// Break up a double number - which is known to be a finite non-negative number -
// into its base-10 parts: integral - before the decimal point, and fractional - after it.
// Taken the precision into account, but does not change it even internally.
KAT_DEV static double_::components

get_components(double number, printf_size_t precision)
{
  double_::components number_;
  number_.is_negative = get_sign_bit(number);
  double abs_number = (number_.is_negative) ? -number : number;
  number_.integral = (int_fast64_t) abs_number;
  double remainder = (abs_number - (double) number_.integral) * power_of_10((int) precision);
  number_.fractional = (int_fast64_t) remainder;

  remainder -= (double) number_.fractional;

  if (remainder > 0.5) {
    ++number_.fractional;
    // handle rollover, e.g. case 0.99 with precision 1 is 1.0
    if ((double) number_.fractional >= power_of_10((int) precision)) {
      number_.fractional = 0;
      ++number_.integral;
    }
  } else if ((remainder == 0.5) && ((number_.fractional == 0U) || (number_.fractional & 1U))) {
    // if halfway, round up if odd OR if last digit is 0
    ++number_.fractional;
  }

  if (precision == 0U) {
    remainder = abs_number - (double) number_.integral;
    if ((!(remainder < 0.5) || (remainder > 0.5)) && (number_.integral & 1)) {
      // exactly 0.5 and ODD, then round up
      // 1.5 -> 2, but 2.5 -> 2
      ++number_.integral;
    }
  }
  return number_;
}

struct scaling_factor {
  double raw_factor;
  bool multiply; // if true, need to multiply by raw_factor; otherwise need to divide by it
};

KAT_DEV static double apply_scaling(double num, scaling_factor normalization)
{
  return normalization.multiply ? num * normalization.raw_factor : num / normalization.raw_factor;
}

KAT_DEV static double unapply_scaling(double normalized, scaling_factor normalization)
{
  return normalization.multiply ? normalized / normalization.raw_factor : normalized * normalization.raw_factor;
}

KAT_DEV static scaling_factor
update_normalization(scaling_factor
sf,
double extra_multiplicative_factor
)
{
scaling_factor result;
// sf.multiply is false, i.e. we have factors in opposite directions

int factor_exp2 = get_exp2(sf.raw_factor);
int extra_factor_exp2 = get_exp2(extra_multiplicative_factor);

// Divide the larger-exponent raw raw_factor by the smaller
if (
abs(factor_exp2)
>
abs(extra_factor_exp2)
) {
result.
multiply = false;
result.
raw_factor = sf.raw_factor / extra_multiplicative_factor;
}
else {
result.
multiply = true;
result.
raw_factor = extra_multiplicative_factor / sf.raw_factor;
}
return
result;
}

KAT_DEV static double_::components

get_normalized_components(
bool negative,
printf_size_t precision,
double non_normalized,
scaling_factor normalization,
int floored_exp10)
{
  double_::components components;
  components.is_negative = negative;
  double scaled = apply_scaling(non_normalized, normalization);

  bool close_to_representation_extremum = ((-floored_exp10 + (int) precision) >= DBL_MAX_10_EXP - 1);
  if (close_to_representation_extremum) {
    // We can't have a normalization factor which also accounts for the precision, i.e. moves
    // some decimal digits into the mantissa, since it's unrepresentable, or nearly unrepresentable.
    // So, we'll give up early on getting extra precision...
    return get_components(negative ? -scaled : scaled, precision);
  }
  components.integral = (int_fast64_t) scaled;
  double remainder = non_normalized - unapply_scaling((double) components.integral, normalization);
  double prec_power_of_10 = power_of_10((int) precision);
  scaling_factor account_for_precision = update_normalization(normalization, prec_power_of_10);
  double scaled_remainder = apply_scaling(remainder, account_for_precision);
  double rounding_threshold = 0.5;

  components.fractional = (int_fast64_t) scaled_remainder; // when precision == 0, the assigned value should be 0
  scaled_remainder -= (double) components.fractional; //when precision == 0, this will not change scaled_remainder

  components.fractional += (scaled_remainder >= rounding_threshold);
  if (scaled_remainder == rounding_threshold) {
    // banker's rounding: Round towards the even number (making the mean error 0)
    components.fractional &= ~((int_fast64_t) 0x1);
  }
  // handle rollover, e.g. the case of 0.99 with precision 1 becoming (0,100),
  // and must then be corrected into (1, 0).
  // Note: for precision = 0, this will "translate" the rounding effect from
  // the fractional part to the integral part where it should actually be
  // felt (as prec_power_of_10 is 1)
  if ((double) components.fractional >= prec_power_of_10) {
    components.fractional = 0;
    ++components.integral;
  }
  return components;
}

KAT_DEV static void print_broken_up_decimal(
double_::components number_, output_gadget_t *output, printf_size_t precision,
printf_size_t width, flags_t flags, char *buf, printf_size_t len)
{
  if (precision != 0U) {
    // do fractional part, as an unsigned number

    printf_size_t count = precision;

    // %g/%G mandates we skip the trailing 0 digits...
    if ((flags & flags::adapt_exp) && !(flags & flags::hash) && (number_.fractional > 0)) {
      while (true) {
        int_fast64_t digit = number_.fractional % 10U;
        if (digit != 0) {
          break;
        }
        --count;
        number_.fractional /= 10U;

      }
      // ... and even the decimal point if there are no
      // non-zero fractional part digits (see below)
    }

    if (number_.fractional > 0 || !(flags & flags::adapt_exp) || (flags & flags::hash)) {
      while (len < decimal_buffer_size) {
        --count;
        buf[len++] = (char) ('0' + number_.fractional % 10U);
        if (!(number_.fractional /= 10U)) {
          break;
        }
      }
      // add extra 0s
      while ((len < decimal_buffer_size) && (count > 0U)) {
        buf[len++] = '0';
        --count;
      }
      if (len < decimal_buffer_size) {
        buf[len++] = '.';
      }
    }
  } else {
    if ((flags & flags::hash) && (len < decimal_buffer_size)) {
      buf[len++] = '.';
    }
  }

  // Write the integer part of the number (it comes after the fractional
  // since the character order is reversed)
  while (len < decimal_buffer_size) {
    buf[len++] = (char) ('0' + (number_.integral % 10));
    if (!(number_.integral /= 10)) {
      break;
    }
  }

  // pad leading zeros
  if (!(flags & flags::left) && (flags & flags::zeropad)) {
    if (width && (number_.is_negative || (flags & (flags::plus | flags::space)))) {
      width--;
    }
    while ((len < width) && (len < decimal_buffer_size)) {
      buf[len++] = '0';
    }
  }

  if (len < decimal_buffer_size) {
    if (number_.is_negative) {
      buf[len++] = '-';
    } else if (flags & flags::plus) {
      buf[len++] = '+';  // ignore the space if the '+' exists
    } else if (flags & flags::space) {
      buf[len++] = ' ';
    }
  }

  out_rev_(output, buf, len, width, flags);
}

// internal ftoa for fixed decimal floating point
KAT_DEV static void print_decimal_number(
output_gadget_t *__restrict__ output,
double number,
printf_size_t precision,
printf_size_t width,
flags_t flags,
char *__restrict__ buf,
printf_size_t len)
{
  double_::components value_ = get_components(number, precision);
  print_broken_up_decimal(value_, output, precision, width, flags, buf, len);
}

KAT_DEV static void print_exponential_number(
output_gadget_t *__restrict__ output,
double number,
printf_size_t precision,
printf_size_t width,
flags_t flags,
char *__restrict__ buf,
printf_size_t len)
{
  const bool negative = get_sign_bit(number);
  // This number will decrease gradually (by factors of 10) as we "extract" the exponent out of it
  double abs_number = negative ? -number : number;

  int floored_exp10;
  bool abs_exp10_covered_by_powers_table;
  scaling_factor normalization;

  // Determine the decimal exponent
  if (abs_number == 0.0) {
    // TODO: This is a special-case for 0.0 (and -0.0); but proper handling is required for denormals more generally.
    floored_exp10 = 0; // ... and no need to set a normalization factor or check the powers table
  } else {
    double exp10 = log10(abs_number);
    floored_exp10 = floor(exp10);
    double p10 = pow(10, floored_exp10);
    normalization.raw_factor = p10;
    abs_exp10_covered_by_powers_table = false;
  }

  // We now begin accounting for the widths of the two parts of our printed field:
  // the decimal part after decimal exponent extraction, and the base-10 exponent part.
  // For both of these, the value of 0 has a special meaning, but not the same one:
  // a 0 exponent-part width means "don't print the exponent"; a 0 decimal-part width
  // means "use as many characters as necessary".

  bool fall_back_to_decimal_only_mode = false;
  if (flags & flags::adapt_exp) {
    int required_significant_digits = (precision == 0) ? 1 : (int) precision;
    // Should we want to fall-back to "%f" mode, and only print the decimal part?
    fall_back_to_decimal_only_mode = (floored_exp10 >= -4 && floored_exp10 < required_significant_digits);
    // Now, let's adjust the precision
    // This also decided how we adjust the precision value - as in "%g" mode,
    // "precision" is the number of _significant digits_, and this is when we "translate"
    // the precision value to an actual number of decimal digits.
    int precision_ = fall_back_to_decimal_only_mode ?
                     (int) precision - 1 - floored_exp10 :
                     (int) precision -
                     1; // the presence of the exponent ensures only one significant digit comes before the decimal point
    precision = (precision_ > 0 ? (unsigned) precision_ : 0U);
    flags |= flags::precision;   // make sure print_broken_up_decimal respects our choice above
  }

  normalization.multiply = (floored_exp10 < 0 && abs_exp10_covered_by_powers_table);
  bool should_skip_normalization = (fall_back_to_decimal_only_mode || floored_exp10 == 0);
  double_::components decimal_part_components =
  should_skip_normalization ?
  get_components(negative ? -abs_number : abs_number, precision) :
  get_normalized_components(negative, precision, abs_number, normalization, floored_exp10);

  // Account for roll-over, e.g. rounding from 9.99 to 100.0 - which effects
  // the exponent and may require additional tweaking of the parts
  if (fall_back_to_decimal_only_mode) {
    if ((flags & flags::adapt_exp) && floored_exp10 >= -1 &&
        decimal_part_components.integral == power_of_10(floored_exp10 + 1)) {
      floored_exp10++; // Not strictly necessary, since floored_exp10 is no longer really used
      precision--;
      // ... and it should already be the case that decimal_part_components.fractional == 0
    }
    // TODO: What about rollover strictly within the fractional part?
  } else {
    if (decimal_part_components.integral >= 10) {
      floored_exp10++;
      decimal_part_components.integral = 1;
      decimal_part_components.fractional = 0;
    }
  }

  // the floored_exp10 format is "E%+03d" and largest possible floored_exp10 value for a 64-bit double
  // is "307" (for 2^1023), so we set aside 4-5 characters overall
  printf_size_t exp10_part_width = fall_back_to_decimal_only_mode ? 0U : (abs(floored_exp10) < 100) ? 4U : 5U;

  printf_size_t decimal_part_width =
  ((flags & flags::left) && exp10_part_width) ?
  // We're padding on the right, so the width constraint is the exponent part's
  // problem, not the decimal part's, so we'll use as many characters as we need:
  0U :
  // We're padding on the left; so the width constraint is the decimal part's
  // problem. Well, can both the decimal part and the exponent part fit within our overall width?
  ((width > exp10_part_width) ?
   // Yes, so we limit our decimal part's width.
   // (Note this is trivially valid even if we've fallen back to "%f" mode)
   width - exp10_part_width :
   // No; we just give up on any restriction on the decimal part and use as many
   // characters as we need
   0U);

  const printf_size_t printed_exponential_start_pos = output->pos;
  print_broken_up_decimal(decimal_part_components, output, precision, decimal_part_width, flags, buf, len);

  if (!fall_back_to_decimal_only_mode) {
    putchar_via_gadget(output, (flags & flags::uppercase) ? 'E' : 'e');
    print_integer(output,
                  abs_for_printing(floored_exp10),
                  floored_exp10 < 0, 10, 0, exp10_part_width - 1,
                  flags::zeropad | flags::plus);
    if (flags & flags::left) {
      // We need to right-pad with spaces to meet the width requirement
      while (output->pos - printed_exponential_start_pos < width) {
        putchar_via_gadget(output, ' ');
      }
    }
  }
}

KAT_DEV static void print_floating_point(
output_gadget_t *output,
double value,
printf_size_t precision,
printf_size_t width,
flags_t flags,
bool prefer_exponential)
{
  char buf[decimal_buffer_size];
  printf_size_t len = 0U;

  // test for special values
  if (value != value) {
    out_rev_(output, "nan", 3, width, flags);
    return;
  }
  if (value < -DBL_MAX) {
    out_rev_(output, "fni-", 4, width, flags);
    return;
  }
  if (value > DBL_MAX) {
    out_rev_(output, (flags & flags::plus) ? "fni+" : "fni", (flags & flags::plus) ? 4U : 3U, width, flags);
    return;
  }

  if (!prefer_exponential &&
      ((value > float_notation_threshold) || (value < -float_notation_threshold))) {
    // The required behavior of standard printf is to print _every_ integral-part digit -- which could mean
    // printing hundreds of characters, overflowing any fixed internal buffer and necessitating a more complicated
    // implementation.
    print_exponential_number(output, value, precision, width, flags, buf, len);
    return;
  }

  // set default precision, if not set explicitly
  if (!(flags & flags::precision)) {
    precision = default_float_precision;
  }

  // limit precision so that our integer holding the fractional part does not overflow
  while ((len < decimal_buffer_size) && (precision > max_supported_precision)) {
    buf[len++] = '0'; // This respects the precision in terms of result length only
    precision--;
  }

  if (prefer_exponential)
    print_exponential_number(output, value, precision, width, flags, buf, len);
  else
    print_decimal_number(output, value, precision, width, flags, buf, len);
}

// Advances the format pointer past the flags, and returns the parsed flags
// due to the characters passed
KAT_DEV static flags_t

parse_flags(const char **format)
{
  flags_t flags = 0U;
  do {
    switch (**format) {
      case '0':
        flags |= flags::zeropad;
        (*format)++;
        break;
      case '-':
        flags |= flags::left;
        (*format)++;
        break;
      case '+':
        flags |= flags::plus;
        (*format)++;
        break;
      case ' ':
        flags |= flags::space;
        (*format)++;
        break;
      case '#':
        flags |= flags::hash;
        (*format)++;
        break;
      default :
        return flags;
    }
  } while (true);
}

// internal vsnprintf - used for implementing _all library functions
// Note: We don't like the C standard's parameter names, so using more informative parameter names
// here instead.
KAT_DEV static int vsnprintf(output_gadget_t *output, const char *format, va_list args)
{
  // Note: The library only calls vsnprintf() with output->pos being 0. However, it is
  // possible to call this function with a non-zero pos value for some "remedial printing".

  while (*format) {
    // format specifier?  %[flags][width][.precision][length]
    if (*format != '%') {
      // no
      putchar_via_gadget(output, *format);
      format++;
      continue;
    } else {
      // yes, evaluate it
      format++;
    }

    flags_t flags = parse_flags(&format);

    // evaluate width field
    printf_size_t width = 0U;
    if (is_digit_(*format)) {
      width = (printf_size_t) atou_(&format);
    } else if (*format == '*') {
      const int w = va_arg(args, int);
      if (w < 0) {
        flags |= flags::left;    // reverse padding
        width = (printf_size_t) - w;
      } else {
        width = (printf_size_t) w;
      }
      format++;
    }

    // evaluate precision field
    printf_size_t precision = 0U;
    if (*format == '.') {
      flags |= flags::precision;
      format++;
      if (is_digit_(*format)) {
        precision = (printf_size_t) atou_(&format);
      } else if (*format == '*') {
        const int precision_ = va_arg(args, int);
        precision = precision_ > 0 ? (printf_size_t) precision_ : 0U;
        format++;
      }
    }

    // evaluate length field
    switch (*format) {
      case 'I' : {
        format++;
        // Greedily parse for size in bits: 8, 16, 32 or 64
        switch (*format) {
          case '8':
            flags |= flags::int8;
            format++;
            break;
          case '1':
            format++;
            if (*format == '6') {
              format++;
              flags |= flags::int16;
            }
            break;
          case '3':
            format++;
            if (*format == '2') {
              format++;
              flags |= flags::int32;
            }
            break;
          case '6':
            format++;
            if (*format == '4') {
              format++;
              flags |= flags::int64;
            }
            break;
          default:
            break;
        }
        break;
      }
      case 'l' :
        flags |= flags::long_;
        format++;
        if (*format == 'l') {
          flags |= flags::long_long;
          format++;
        }
        break;
      case 'h' :
        flags |= flags::short_;
        format++;
        if (*format == 'h') {
          flags |= flags::char_;
          format++;
        }
        break;
      case 't' :
      case 'j' :
      case 'z' :
        static_assert(sizeof(ptrdiff_t) == sizeof(long), "Unexpected sizeof(ptrdiff_t)");
        static_assert(sizeof(intmax_t) == sizeof(long), "Unexpected sizeof(intmax_t)");
        static_assert(sizeof(size_t) == sizeof(long), "Unexpected sizeof(size_t)");
        flags |= flags::long_;
        format++;
        break;
      default:
        break;
    }

    // evaluate specifier
    switch (*format) {
      case 'd' :
      case 'i' :
      case 'u' :
      case 'x' :
      case 'X' :
      case 'o' :
      case 'b' : {

        if (*format == 'd' || *format == 'i') {
          flags |= flags::signed_;
        }

        numeric_base_t base;
        if (*format == 'x' || *format == 'X') {
          base = base::hex;
        } else if (*format == 'o') {
          base = base::octal;
        } else if (*format == 'b') {
          base = base::binary;
        } else {
          base = base::decimal;
          flags &= ~flags::hash; // decimal integers have no alternative presentation
        }

        if (*format == 'X') {
          flags |= flags::uppercase;
        }

        format++;
        // ignore '0' flag when precision is given
        if (flags & flags::precision) {
          flags &= ~flags::zeropad;
        }

        if (flags & flags::signed_) {
          // A signed specifier: d, i or possibly I + bit size if enabled

          if (flags & flags::long_long) {
            const long long value = va_arg(args, long long);
            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
          } else if (flags & flags::long_) {
            const long value = va_arg(args, long);
            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
          } else {
            // We never try to interpret the argument as something potentially-smaller than int,
            // due to integer promotion rules: Even if the user passed a short int, short unsigned
            // etc. - these will come in after promotion, as int's (or unsigned for the case of
            // short unsigned when it has the same size as int)
            const int value =
            (flags & flags::char_) ? (signed char) va_arg(args, int) :
            (flags & flags::short_) ? (short int) va_arg(args, int) :
            va_arg(args, int);
            print_integer(output, abs_for_printing(value), value < 0, base, precision, width, flags);
          }
        } else {
          // An unsigned specifier: u, x, X, o, b

          flags &= ~(flags::plus | flags::space);

          if (flags & flags::long_long) {
            print_integer(output, (unsigned_value_t) va_arg(args, unsigned long long), false, base, precision, width,
                          flags);
          } else if (flags & flags::long_) {
            print_integer(output, (unsigned_value_t) va_arg(args, unsigned long), false, base, precision, width, flags);
          } else {
            const unsigned int value =
            (flags & flags::char_) ? (unsigned char) va_arg(args, unsigned int) :
            (flags & flags::short_) ? (unsigned short int) va_arg(args, unsigned int) :
            va_arg(args, unsigned int);
            print_integer(output, (unsigned_value_t) value, false, base, precision, width, flags);
          }
        }
        break;
      }

        enum : bool {
          prefer_decimal = false, prefer_exponential = true
        };

      case 'f' :
      case 'F' :
        if (*format == 'F') flags |= flags::uppercase;
        print_floating_point(output, va_arg(args, double), precision, width, flags, prefer_decimal);
        format++;
        break;
      case 'e':
      case 'E':
      case 'g':
      case 'G':
        if ((*format == 'g') || (*format == 'G')) flags |= flags::adapt_exp;
        if ((*format == 'E') || (*format == 'G')) flags |= flags::uppercase;
        print_floating_point(output, va_arg(args, double), precision, width, flags, prefer_exponential);
        format++;
        break;
      case 'c' : {
        printf_size_t l = 1U;
        // pre padding
        if (!(flags & flags::left)) {
          while (l++ < width) {
            putchar_via_gadget(output, ' ');
          }
        }
        // char output
        putchar_via_gadget(output, (char) va_arg(args, int));
        // post padding
        if (flags & flags::left) {
          while (l++ < width) {
            putchar_via_gadget(output, ' ');
          }
        }
        format++;
        break;
      }

      case 's' : {
        const char *p = va_arg(args, char * );
        if (p == nullptr) {
          out_rev_(output, ")llun(", 6, width, flags);
        } else {
          printf_size_t l = strnlen_s_(p, precision ? precision : max_possible_buffer_size);
          // pre padding
          if (flags & flags::precision) {
            l = (l < precision ? l : precision);
          }
          if (!(flags & flags::left)) {
            while (l++ < width) {
              putchar_via_gadget(output, ' ');
            }
          }
          // string output
          while ((*p != 0) && (!(flags & flags::precision) || precision)) {
            putchar_via_gadget(output, *(p++));
            --precision;
          }
          // post padding
          if (flags & flags::left) {
            while (l++ < width) {
              putchar_via_gadget(output, ' ');
            }
          }
        }
        format++;
        break;
      }

      case 'p' : {
        width = sizeof(void *) * 2U + 2; // 2 hex chars per byte + the "0x" prefix
        flags |= flags::zeropad | flags::pointer;
        uintptr_t value = (uintptr_t) va_arg(args, void * );
        (value == (uintptr_t) nullptr) ?
        out_rev_(output, ")lin(", 5, width, flags) :
        print_integer(output, (unsigned_value_t) value, false, base::hex, precision, width, flags);
        format++;
        break;
      }

      case '%' :
        putchar_via_gadget(output, '%');
        format++;
        break;

      case 'n' : {
        if (flags & flags::char_) *(va_arg(args, char * )) = (char) output->pos;
        else if (flags & flags::short_) *(va_arg(args, short * )) = (short) output->pos;
        else if (flags & flags::long_) *(va_arg(args, long * )) = (long) output->pos;
        else if (flags & flags::long_long) *(va_arg(args, long long * )) = (long long int) output->pos;
        else *(va_arg(args, int * )) = (int) output->pos;
        format++;
        break;
      }

      default :
        putchar_via_gadget(output, *format);
        format++;
        break;
    }
  }

  // termination
  append_termination_with_gadget(output);

  // return written chars without terminating \0
  return (int) output->pos;
}

} // namespace printf
} // namespace detail_

///////////////////////////////////////////////////////////////////////////////

KAT_DEV int vprintf(const char* format, va_list arg)
{
  detail_::printf::output_gadget_t gadget = detail_::printf::discarding_gadget();
  int ret = vsnprintf(&gadget, format, arg);
  if (ret < 0) {
    return ret;
  }
  size_t count = ret + 1;
  char* scratch = (char*) malloc(count);
  if (scratch == nullptr) {
    return -1;
  }
  ret = vsnprintf(scratch, count, format, arg);
  if (ret < 0) {
    free(scratch);
    return ret;
  }
  ret = printf("%s", scratch); // This is CUDA's builtin printf()
}

KAT_DEV int vsnprintf(char* s, size_t n, const char* format, va_list arg)
{
  detail_::printf::output_gadget_t gadget = detail_::printf::buffer_gadget(s, n);
  return detail_::printf::vsnprintf(&gadget, format, arg);
}

KAT_DEV int vsprintf(char* s, const char* format, va_list arg)
{
  return vsnprintf(s, detail_::printf::max_possible_buffer_size, format, arg);
}

KAT_DEV inline int vnprintf_with_scratch(char* scratch, size_t count, const char* format, va_list arg)
{
  const int ret = vsnprintf(scratch, count, format, arg);
  if (scratch == nullptr) {
    return ret;
  }
  if (ret > 0) {
    return printf("%s",scratch);
  }
};

KAT_DEV int printf(const char* format, ...)
{
  va_list args;
  va_start(args, format);
  const int ret = vprintf(format, args);
  va_end(args);
  return ret;
}

KAT_DEV int sprintf(char* s, const char* format, ...)
{
  va_list args;
  va_start(args, format);
  const int ret = vsprintf(s, format, args);
  va_end(args);
  return ret;
}

KAT_DEV int snprintf(char* s, size_t n, const char* format, ...)
{
  va_list args;
  va_start(args, format);
  const int ret = vsnprintf(s, n, format, args);
  va_end(args);
  return ret;
}

KAT_DEV int nprintf_with_scratch(char* scratch_buffer, size_t count, const char* format, ...)
{
  va_list args;
  va_start(args, format);
  return vnprintf_with_scratch(scratch_buffer, count, format, args);
}

} // namespace kat