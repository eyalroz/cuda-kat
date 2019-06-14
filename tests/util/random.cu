
#include "random.hpp"

namespace util {
namespace random {
std::random_device device;  // Note this is a callable object.
std::default_random_engine engine(device());
} // namespace random
} // namespace util


