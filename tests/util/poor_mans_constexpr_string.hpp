#ifndef CUDA_KAT_TEST_POOR_MANS_CONSTEXPR_STRING_HPP_
#define CUDA_KAT_TEST_POOR_MANS_CONSTEXPR_STRING_HPP_

#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <ostream>

///@cond
#include <kat/detail/execution_space_specifiers.hpp>
///@endcond

#  if __cplusplus < 201103
#    error "C++11 or later required"
#  elif __cplusplus < 201402
#    define CONSTEXPR14_TN
#  else
#    define CONSTEXPR14_TN constexpr
#  endif

namespace util {

class constexpr_string
{
    const char* const p_;
    const std::size_t sz_;

public:
    typedef const char* const_iterator;

    template <std::size_t N>
    constexpr KAT_FHD constexpr_string(const char(&a)[N]) noexcept
        : p_(a)
        , sz_(N-1)
        {}

    constexpr KAT_FHD constexpr_string(const char* p, std::size_t N) noexcept
        : p_(p)
        , sz_(N)
        {}

    constexpr KAT_FHD const char* data() const noexcept {return p_;}
    constexpr KAT_FHD std::size_t size() const noexcept {return sz_;}

    constexpr KAT_FHD const_iterator begin() const noexcept {return p_;}
    constexpr KAT_FHD const_iterator end()   const noexcept {return p_ + sz_;}

    constexpr KAT_FHD char operator[](std::size_t n) const
    {
    	return n < sz_ ? p_[n] :
#ifdef __CUDA_ARCH__
    		0;
#else
            throw std::out_of_range("constexpr_string");
#endif
    }
};

KAT_FHD
std::ostream&
operator<<(std::ostream& os, constexpr_string const& s)
{
    return os.write(s.data(), s.size());
}

} // namespace util
#include <kat/detail/execution_space_specifiers.hpp>

#endif // CUDA_KAT_TEST_POOR_MANS_CONSTEXPR_STRING_HPP_
