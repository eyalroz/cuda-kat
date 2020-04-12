#pragma once

#include <cstdint>

struct uint128_t final
{
	uint128_t() = default;
	constexpr uint128_t(const uint64_t high_, const uint64_t low_) : high(high_), low(low_) {}
	constexpr uint128_t(const uint64_t v) : high(0), low(v) {}

	explicit constexpr operator uint64_t() const { return low; }

	constexpr operator bool() const { return low || high; }

	uint64_t high;
	uint64_t low;
};

inline constexpr bool operator<(const uint128_t l, const uint128_t r)
{
	return l.high < r.high || (l.high == r.high && l.low < r.low);
}

inline constexpr bool operator<=(const uint128_t l, const uint128_t r)
{
	return l.high < r.high || (l.high == r.high && l.low <= r.low);
}

inline constexpr bool operator>(const uint128_t l, const uint128_t r)
{
	return l.high > r.high || (l.high == r.high && l.low > r.low);
}

inline constexpr bool operator>=(const uint128_t l, const uint128_t r)
{
	return l.high > r.high || (l.high == r.high && l.low >= r.low);
}

inline constexpr bool operator==(const uint128_t l, const uint128_t r)
{
	return l.low == r.low && l.high == r.high;
}

inline constexpr bool operator!=(const uint128_t l, const uint128_t r)
{
	return l.low != r.low || l.high != r.high;
}

inline constexpr uint128_t operator+(const uint128_t l, const uint128_t r)
{
	uint128_t result{l.high + r.high, l.low + r.low};
	if (result.low < l.low)
	{
		++result.high;
	}
	return result;
}

inline constexpr uint128_t operator-(const uint128_t l, const uint128_t r)
{
	uint128_t result{l.high - r.high, l.low - r.low};
	if (result.low > l.low)
	{
		--result.high;
	}
	return result;
}

inline constexpr uint128_t operator*(const uint128_t l, const uint128_t r)
{
	uint128_t result{(l.low >> 32) * (r.low >> 32), (l.low & 0xffffffff) * (r.low & 0xffffffff)};
	{
		const uint64_t m12 = (l.low & 0xffffffff) * (r.low >> 32);
		{
			const uint64_t m12_l = (m12 & 0xffffffff) << 32;
			const uint64_t old_low = result.low;
			result.low += m12_l;
			if (result.low < old_low)
			{
				++result.high;
			}
			result.high += (m12 >> 32);
			
		}
	}
	{
		const uint64_t m21 = (l.low >> 32) * (r.low & 0xffffffff);
		{
			const uint64_t m21_l = (m21 & 0xffffffff) << 32;
			const uint64_t old_low = result.low;
			result.low += m21_l;
			if (result.low < old_low)
			{
				++result.high;
			}
			result.high += (m21 >> 32);
		}
	}
	result.high +=
		(l.low & 0xffffffff) * (r.high & 0xffffffff) +
		(l.high & 0xffffffff) * (r.low & 0xffffffff) +
		(((l.low & 0xffffffff) * (r.high >> 32)) << 32) +
		(((l.high >> 32) * (r.low & 0xffffffff)) << 32) +
		(((l.low >> 32) * (r.high & 0xffffffff)) << 32) +
		(((l.high & 0xffffffff) * (r.low >> 32)) << 32);

	return result;
}

/*
inline constexpr uint128_t operator/(const uint128_t l, const uint128_t r)
{
	//! \todo implement
	return {};
}

inline constexpr uint128_t operator%(const uint128_t l, const uint128_t r)
{
	//! \todo implement
	return {};
}
*/

inline constexpr uint128_t operator~(const uint128_t v)
{
	return uint128_t{~v.high, ~v.low};
}

inline constexpr uint128_t operator&(const uint128_t l, const uint128_t r)
{
	return uint128_t{l.high & r.high, l.low & r.low};
}

inline constexpr uint128_t operator|(const uint128_t l, const uint128_t r)
{
	return uint128_t{l.high | r.high, l.low | r.low};
}

inline constexpr uint128_t operator^(const uint128_t l, const uint128_t r)
{
	return uint128_t{l.high ^ r.high, l.low ^ r.low};
}

inline constexpr uint128_t operator<<(const uint128_t v, const unsigned s)
{
	if (s >= 64)
	{
		return {v.low << (s - 64), 0};
	}
	return {v.high << s | v.low >> (64 - s), v.low << s};
}

inline constexpr uint128_t operator>>(const uint128_t v, const unsigned s)
{
	if (s >= 64)
	{
		return {0, v.high >> (s - 64)};
	}
	return {v.high >> s, v.high << (64 - s) | v.low >> s};
}

inline constexpr uint128_t & operator++(uint128_t & v)
{
	++v.low;
	if (!v.low)
	{
		++v.high;
	}
	return v;
}

inline constexpr uint128_t & operator--(uint128_t & v)
{
	if (!v.low)
	{
		--v.high;
	}
	--v.low;
	return v;
}

inline constexpr uint128_t operator++(uint128_t & v, int)
{
	uint128_t r = v;
	++v;
	return r;
}

inline constexpr uint128_t operator--(uint128_t & v, int)
{
	uint128_t r = v;
	--v;
	return r;
}

inline constexpr uint128_t & operator+=(uint128_t & l, const uint128_t r)
{
	const uint64_t low = l.low;
	l.low += r.low;
	l.high += r.high;
	if (l.low < low)
	{
		++l.high;
	}
	return l;
}

inline constexpr uint128_t & operator-=(uint128_t & l, const uint128_t r)
{
	const uint64_t low = l.low;
	l.low -= r.low;
	l.high -= r.high;
	if (l.low > low)
	{
		--l.high;
	}
	return l;
}

inline constexpr uint128_t & operator*=(uint128_t & l, const uint128_t r)
{
	l = l * r;
	return l;
}

/*
inline constexpr uint128_t & operator/=(uint128_t & l, const uint128_t r)
{
	//! \todo implement
	return l;
}

inline constexpr uint128_t & operator%=(uint128_t & l, const uint128_t r)
{
	//! \todo implement
	return l;
}
*/

inline constexpr uint128_t & operator&=(uint128_t & l, const uint128_t r)
{
	l.high &= r.high;
	l.low &= r.low;
	return l;
}

inline constexpr uint128_t & operator|=(uint128_t & l, const uint128_t r)
{
	l.high |= r.high;
	l.low |= r.low;
	return l;
}

inline constexpr uint128_t & operator^=(uint128_t & l, const uint128_t r)
{
	l.high ^= r.high;
	l.low ^= r.low;
	return l;
}

inline constexpr uint128_t & operator<<=(uint128_t & v, const unsigned s)
{
	if (s >= 64)
	{
		v.high = v.low << (s - 64);
		v.low = 0;
	}
	else
	{
		v.high = v.high << s | v.low >> (64 - s);
		v.low <<= s;
	}
	return v;
}

inline constexpr uint128_t & operator>>=(uint128_t & v, const unsigned s)
{
	if (s >= 64)
	{
		v.low = v.high >> (s - 64);
		v.high = 0;
	}
	else
	{
		v.low = v.high << (64 - s) | v.low >> s;
		v.high >>= s;
	}
	return v;
}
