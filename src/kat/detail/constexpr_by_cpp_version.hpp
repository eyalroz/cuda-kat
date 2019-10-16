
#ifndef CONSTEXPR_BY_CPP_VERSION_HPP_
#define CONSTEXPR_BY_CPP_VERSION_HPP_

///@cond

#if __cplusplus < 201103L
#error "C++11 or newer is required to use this header"
#endif

#ifndef CONSTEXPR_SINCE_CPP_14
#if __cplusplus >= 201402L
#define CONSTEXPR_SINCE_CPP_14 constexpr
#else
#define CONSTEXPR_SINCE_CPP_14
#endif
#endif

#ifndef CONSTEXPR_SINCE_CPP_17
#if __cplusplus >= 201701L
#define CONSTEXPR_SINCE_CPP_17 constexpr
#else
#define CONSTEXPR_SINCE_CPP_17
#endif
#endif

///@nocond

#endif // CONSTEXPR_BY_CPP_VERSION_HPP_
