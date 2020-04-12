#ifndef KAT_ON_DEVICE_DETAIL_ITOA_CUH_
#define KAT_ON_DEVICE_DETAIL_ITOA_CUH_

#include <cstdint>

namespace kat {
namespace detail {

template <typename U> struct max_num_digits { };
template <> struct max_num_digits<uint8_t > { static constexpr const unsigned value {  3 }; };
template <> struct max_num_digits<uint16_t> { static constexpr const unsigned value {  5 }; };
template <> struct max_num_digits<uint32_t> { static constexpr const unsigned value { 10 }; };
template <> struct max_num_digits<uint64_t> { static constexpr const unsigned value { 20 }; };

template <typename I>
inline KAT_DEV unsigned integer_to_string_reversed(I value, char* buffer)
{
	bool append_minus {
#pragma push
#pragma diag_suppress = unsigned_compare_with_zero
		std::is_signed<I>::value and (value < 0)
#pragma pop
	};
	value = builtins::absolute_value(value);

    char *reverse_ptr = buffer;
    do {
        *reverse_ptr++ = '0' + (value % 10);
        value /= 10;
    } while (value > 0);

    if (append_minus) { *reverse_ptr++ = '-'; }
    return reverse_ptr - buffer;
}

inline KAT_DEV char* copy_in_reverse(char* dst, const char* src, std::size_t length)
{
    for(auto i = 0; i < length; i++) {
    	dst[i] = src[length - i - 1];
    }
    return dst;
}

// This is not supposed to be optimal, just a straightforward short implementation
template <typename I, bool WriteTermination = true>
inline KAT_DEV unsigned integer_to_string(I value, char* buffer)
{
	using unsigned_type = typename std::make_unsigned<I>::type;
    char reverse_buffer[max_num_digits<unsigned_type>::value];
    auto length = integer_to_string_reversed<I>(value, reverse_buffer);
    copy_in_reverse(buffer, reverse_buffer, length);
    if (WriteTermination) { buffer[length] = '\0'; }
    return length;
}

} // namespace detail
} // namespace kat


#endif // KAT_ON_DEVICE_DETAIL_ITOA_CUH_
