/**
 * @file ranges.hpp
 *
 * @brief A few simple, convenient (not-full-fledged-C++20) ranges.
 *
 * @note This file has no CUDA-specific code except for `KAT_HD` specifiers; it
 * is a mechanism for use elsewhere.
 *
 */

#pragma once
#ifndef CUDA_KAT_RANGES_HPP_
#define CUDA_KAT_RANGES_HPP_

#include <kat/on_device/common.cuh>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

#include <iterator>
#include <type_traits>

namespace kat {

namespace detail {

namespace iterators {

template <typename I>
struct simple : std::iterator<std::input_iterator_tag, I> {
    constexpr KAT_HD simple(I position) : position_(position) { }
    constexpr KAT_HD I operator*() const { return position_; }
    constexpr KAT_HD I const* operator->() const { return &position_; }
    constexpr KAT_HD bool operator==(const simple& other) const { return position_ == other.position_; }
    constexpr KAT_HD bool operator!=(const simple& other) const { return not (*this == other); }

    constexpr KAT_HD simple& operator++() {
        ++position_;
        return *this;
    }

    constexpr KAT_HD simple operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

protected:
    I position_;
};

#if __cplusplus >= 201703L

template <typename I>
struct strided : simple<I> {
	using parent = simple<I>;
	using parent::position_;

	/**
	 * @note behavior is undefined if stride is non-positive
	 */
	constexpr KAT_HD strided(I position, I stride) : parent(position), stride_(stride) { }

	constexpr KAT_HD bool operator==(const parent& other) const { return parent::operator==(other); }
	constexpr KAT_HD bool operator!=(const parent& other) const { return parent::operator!=(other); }

	constexpr KAT_HD strided& operator++() {
		position_ += stride_;
		return *this;
	}

	constexpr KAT_HD strided operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

    constexpr I stride() const { return stride_; }

protected:
    I stride_; // needs to be positive!
};


template <typename I>
struct strided_sentinel {
	/**
	 * @note behavior is undefined if stride is non-positive
	 */
	constexpr KAT_HD strided_sentinel(I position) : threshold_position_(position) { }

	constexpr KAT_HD bool operator==(const strided_sentinel& other) const { return threshold_position_ == other.threshold_position_; }
	constexpr KAT_HD bool operator!=(const strided_sentinel& other) const { return threshold_position_ != other.threshold_position_; }

    constexpr KAT_HD I threshold_position() const { return threshold_position_; }

protected:
    I threshold_position_;
};

/*
 * The following  two methods are defined as they are because C++' ranged-for loops only use =/!= comparisons
 * between iterators and sentinels, not <= ...
 */
template <typename I> constexpr KAT_HD bool operator< (const strided<I>& lhs, const strided_sentinel<I>& rhs) { return *lhs <  *rhs; }
template <typename I> constexpr KAT_HD bool operator<=(const strided<I>& lhs, const strided_sentinel<I>& rhs) { return *lhs <= *rhs; }
template <typename I> constexpr KAT_HD bool operator> (const strided<I>& lhs, const strided_sentinel<I>& rhs) { return false;        }
template <typename I> constexpr KAT_HD bool operator>=(const strided<I>& lhs, const strided_sentinel<I>& rhs) { return *lhs >= *rhs; }
template <typename I> constexpr KAT_HD bool operator==(const strided<I>& lhs, const strided_sentinel<I>& rhs) { return *lhs >= *rhs; }
template <typename I> constexpr KAT_HD bool operator!=(const strided<I>& lhs, const strided_sentinel<I>& rhs) { return *lhs <  *rhs; }

template <typename I> constexpr KAT_HD bool operator< (const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs <  lhs;   }
template <typename I> constexpr KAT_HD bool operator<=(const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs <= lhs;   }
template <typename I> constexpr KAT_HD bool operator> (const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs >  lhs;   }
template <typename I> constexpr KAT_HD bool operator>=(const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs >= lhs;   }
template <typename I> constexpr KAT_HD bool operator==(const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs == lhs;   }
template <typename I> constexpr KAT_HD bool operator!=(const strided_sentinel<I>& lhs, const strided<I>& rhs) { return rhs != lhs;   }

#else // __cplusplus >= 201703L

/**
 * A kludge two-classes-in-one arrangement, since we must use this class for sentinels as well.
 * We'll have to cross our fingers and hope the compiler is smart enough to propagate the
 * constness of the choice of "implicit subclass"
 */
template <typename I>
struct strided {

protected:
	constexpr KAT_HD strided(I position, I stride, bool is_sentinel)
		: position_(position), stride_(stride), is_sentinel_(is_sentinel) { }

public:
	/**
	 * @note behavior is undefined if stride is non-positive
	 */
	constexpr KAT_HD strided(I position, I stride)
		: position_(position), stride_(stride), is_sentinel_(false) { }

	constexpr KAT_HD strided(const strided& other) = default;
	constexpr KAT_HD strided(strided&& other) = default;
	constexpr KAT_HD strided& operator=(const strided& other) = default;
	constexpr KAT_HD strided& operator=(strided&& other) = default;

    constexpr KAT_HD I operator*() const { return position_; }
    constexpr KAT_HD I const* operator->() const { return &position_; }


	constexpr KAT_HD bool operator==(const strided& other) const
	{
		if (is_sentinel_ == other.is_sentinel_) { return position_ == other.position_; }
		else if (is_sentinel_)                  { return position_ <= other.position_; }
		else                                    { return position_ >= other.position_; }
	}
	constexpr KAT_HD bool operator!=(const strided& other) const { return not operator==(other); }

	constexpr KAT_HD strided& operator++() {
		position_ += stride_;
		return *this;
	}

	constexpr KAT_HD strided operator++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

    constexpr KAT_HD I stride() const { return stride_; }
    constexpr KAT_HD I is_sentinel() const { return is_sentinel_; }

    static constexpr KAT_HD strided sentinel(I threshold_position) { return strided(threshold_position, I{0}, true); }

protected:
    I stride_; // 1. For non-sentinels - needs to be positive; for sentinels - a junk value
    I position_;
    bool is_sentinel_;
};

#endif // __cplusplus >= 201703L

} // namespace iterators

} // namespace detail

namespace ranges {

template <typename I>
struct strided {
	using iterator_type = detail::iterators::strided<I>;

	/**
	 * With language standard versions before C++17, we must use a range's iterator
	 * type for its sentinel as well - for which reason we must store another copy
	 * of the `stride` value. Hopefully the compiler will optimize this away.
	 */
#if __cplusplus >= 201703L
	using sentinel_type = detail::iterators::strided_sentinel<I>;
#else
	using sentinel_type = iterator_type;
#endif

    /**
     * @note Behavior is undefined when:
     *
     *   1. The range is constructed with the stride value with a different sign than begin - end.
     *   2. std::numeric_limit<I>::max() - end < stride , and stride > 0.
     *   3. end - std::numeric_limit<I>::min() < (-stride) , and stride < 0.
     *   4. stride is 0.
     *
     */
	KAT_HD strided(I begin, I end, I stride)
#if __cplusplus >= 201703L
		: begin_(begin, stride), end_(end) { }
#else
		: begin_(begin, stride), end_(sentinel_type::sentinel(end)) { }
#endif
    KAT_HD iterator_type begin() const { return begin_; }
    KAT_HD sentinel_type end()   const { return end_; }

    KAT_HD I stride() const { return begin_.stride(); }

protected:
    iterator_type begin_;
    sentinel_type end_;
};

/**
 * A simple range of contiguous integers.
 *
 * @tparam I
 */
template <typename I>
struct simple {
    using iterator_type = detail::iterators::simple<I>;
    using sentinel_type = iterator_type;

    /**
     * @note Behavior is undefined when this range is constructed
	 * with end > begin.
     */
    KAT_HD simple(I begin, I end) : begin_(begin), end_(end) { }

    KAT_HD iterator_type begin() const { return begin_; }
    KAT_HD sentinel_type end()   const { return end_; }

    KAT_HD strided<I> at_stride(I stride) const { return strided<I>{*begin_, *end_, stride}; }

protected:
    iterator_type begin_;
    sentinel_type end_;
};

// TODO: Should we also have an "inclusive range", in which the second constructor argument is part of the range?

} // namespace ranges

template <typename I>
KAT_HD ranges::simple<I> irange(I begin, I end)
{
	return ranges::simple<I>(begin, end);
}

template <typename I>
KAT_HD ranges::simple<I> irange(I length)
{
	return ranges::simple<I>(I{0}, length);
}

template <typename I>
KAT_HD ranges::strided<I> strided_irange(I begin, I end, I stride)
{
	return ranges::strided<I>(begin, end, stride);
}

template <typename I>
KAT_HD ranges::strided<I> strided_irange(I length, I stride)
{
	return ranges::strided<I>(I{0}, length, stride);
}

} // namespace kat


#endif // CUDA_KAT_RANGES_HPP_
