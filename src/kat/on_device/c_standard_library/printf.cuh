/**
 * @author (c) Eyal Rozenberg <eyalroz1@gmx.com>
 *             2021-2022, Haifa, Palestine/Israel
 * @author (c) Marco Paland (info@paland.com)
 *             2014-2019, PALANDesign Hannover, Germany
 *
 * @note Others have made smaller contributions to this file: see the
 * contributors page at https://github.com/eyalroz/printf/graphs/contributors
 * or ask one of the authors.
 *
 * @brief An implementation of the printf family of functions (including 
 * (`(v)printf`, `(v)s(n)printf` etc.) for use with CUDA.
 *
 * @note These functions are not to be used for high-performance work. 
 * For high-performance work on strings in kernels, determine exactly what
 * formatting is necessary and write code to perform _only_ that - do not
 * use a kitchen-sink `printf()` like function with runtime parsing of a
 * format string.
 *
 * @note Unlike the original `printf` library (@see 
 * https://github.com/eyalroz/printf ), in this implementation we use CUDA
 * builtins, i.e. it is not entirely self-contained.
 *
 * @note This implementation includes a printf() differing from CUDA's `printf()`
 * builtin in two ways at least: (1.) It requires a scratch buffer and (2.) It
 * supports more of the C99 printf specification than CUDA's `printf()`, as well
 * as some common extensions
 *
 * @note the per-file license here differs from the whole-repository license.
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

#ifndef KAT_PRINTF_CUH_
#define KAT_PRINTF_CUH_

#include <kat/detail/execution_space_specifiers.hpp>
# include <cstdarg>

#ifdef __GNUC__
# define ATTR_PRINTF(one_based_format_index, first_arg) \
__attribute__((format(__printf__, (one_based_format_index), (first_arg))))
# define ATTR_VPRINTF(one_based_format_index) ATTR_PRINTF((one_based_format_index), 0)
#else
# define ATTR_PRINTF((one_based_format_index), (first_arg))
# define ATTR_VPRINTF(one_based_format_index)
#endif

namespace kat {

/**
 * An implementation of the C standard's printf/vprintf, via a self-allocated buffer, backed by
 * CUDA's `printf()`
 *
 * @note These functions will allocate some scratch memory to format a string into, which will then
 * be printed using CUDA's printf. This may be inconvenient or dangerous, so **use of these function
 * is _not_ recommended.** Prefer @ref printf_with_scratch or @ref vprintf_with_scratch instead.
 *
 * @param format A string specifying the format of the output, with %-marked specifiers of how to
 * interpret additional arguments.
 * @param arg Additional arguments to the function, one for each %-specifier in @p format string
 * @return The number of characters written to the output not counting the terminating null 
 *    character.
 */
///@{
KAT_DEV int printf(const char* format, ...)

ATTR_PRINTF(1, 2);

KAT_DEV int vprintf(const char* format, va_list arg) ATTR_VPRINTF(1);
///@}

/**
 * An implementation of the C standard's printf/vprintf, backed by CUDA's `printf()`, with a
 * user-provided sized scratch buffer.
 *
 * @note These functions will not allocate anything on the heap.
 *
 * @param scratch an array for staging the formatted output before passing it to CUDA's
 *     `printf()` function. The buffer must have at least @p count available bytes. If `nullptr`
 *      is passed for `scratch`, nothing is written, but the number of characters _to_ be
 *      written is returned.
 * @param count size of the @p scratch buffer
 * @param format A string specifying the format of the output, with %-marked specifiers of how to interpret
 *     additional arguments.
 * @param arg additional arguments to the function, one for each %-specifier in @p format string
 * @return The number of characters that COULD have been written into @p s, not counting the
 *     terminating null character. A value equal or larger than @p count indicates truncation.
 *     Only when the returned value is non-negative and less than @p count, the null-terminated
 *     string has been fully and successfully printed. If `nullptr` was passed as `s`, the number
 *     of _intended_ characters will be returned without any characters being written anywhere.
 */
KAT_DEV int vnprintf_with_scratch(char* scratch, size_t count, const char* format, va_list arg) ATTR_VPRINTF(3);

KAT_DEV int nprintf_with_scratch(char* scratch, size_t count, const char* format, ...)

ATTR_PRINTF(3, 4);


/**
 * An implementation of the C standard's sprintf/vsprintf
 *
 * @note For security considerations (the potential for exceeding the buffer bounds), please
 * consider using the size-constrained variant, @ref kat::snprintf / @ref kat::vsnprintf , instead.
 *
 * @param s An array in which to store the formatted string. It must be large enough to fit the
 *     formatted output!
 * @param format A string specifying the format of the output, with %-marked specifiers of how
 *     to interpret additional arguments.
 * @param arg Additional arguments to the function, one for each specifier in @p format
 * @return The number of characters written into @p s, not counting the terminating null character.
 *     If `nullptr` was passed as `s`, the number of _intended_ characters will be returned without
 *     any characters being written anywhere.
 */
///@{
KAT_DEV int sprintf(char* s, const char* format, ...) ATTR_PRINTF(2, 3);

KAT_DEV int vsprintf(char* s, const char* format, va_list arg) ATTR_VPRINTF(2);
///@}


/**
 * An implementation of the C standard's snprintf/vsnprintf
 *
 * @param s An array in which to store the formatted string. It must be large enough to fit either
 *     the entire formatted output, or at least @p count characters. Alternatively, it can be
 *     `NULL`, in which case nothing will be printed, and only the number of characters which
 *     _could_ have been printed is tallied and returned.
 * @param n The maximum number of characters to write to the array, including a terminating null
 *     character.
 * @param format A string specifying the format of the output, with %-marked specifiers of how to
 *     interpret additional arguments.
 * @param arg Additional arguments to the function, one for each specifier in @p format
 * @return The number of characters that COULD have been written into @p s, not counting the
 *     terminating null character. A value equal or larger than @p count indicates truncation.
 *     Only when the returned value is non-negative and less than @p count, the null-terminated
 *     string has been fully and successfully printed. If `nullptr` was passed as `s`, the number
 *     of _intended_ characters will be returned without any characters being written anywhere.
 */
///@{
KAT_DEV int snprintf(char* s, size_t count, const char* format, ...)

ATTR_PRINTF(3, 4);

KAT_DEV int vsnprintf(char* s, size_t count, const char* format, va_list arg) ATTR_VPRINTF(3);
///@}

} // namespace kat

#endif  // KAT_PRINTF_CUH_
