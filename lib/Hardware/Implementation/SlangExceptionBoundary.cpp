// Boost exception boundary of the slang library. slang is compiled without
// C++ exceptions, so Boost requires the program to define
// boost::throw_exception; Loom links slang without exceptions as well, so the
// only admissible behavior is process termination.
#include <cstdlib>
#include <exception>

namespace boost {

[[noreturn]] void throw_exception(const std::exception &) { std::abort(); }

} // namespace boost
