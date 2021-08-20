#ifndef CUDA_KAT_TESTS_COMMON_CUH
#define CUDA_KAT_TESTS_COMMON_CUH

#include "util/prettyprint.hpp"
#include "util/type_name.hpp"
#include "util/random.hpp"
#include "util/miscellany.cuh"
#include "util/macro.h"
#include "util/printing.hpp"


#include <doctest.h>
#include <cuda/runtime_api.hpp>
#include <climits>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <iomanip>

#if __cplusplus < 201701L
#include <experimental/optional>
template <typename T>
using optional = std::experimental::optional<T>;
#else
template <typename T>
#include <optional>
using optional = std::optional<T>;
#endif


#endif // CUDA_KAT_TESTS_COMMON_CUH
