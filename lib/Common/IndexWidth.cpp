#include "Common/IndexWidth.h"

#include <cstdlib>

namespace loom {

unsigned getIndexWidth() {
  static unsigned cached = []() -> unsigned {
    if (const char *env = std::getenv("LOOM_INDEX_WIDTH")) {
      char *end = nullptr;
      unsigned long v = std::strtoul(env, &end, 10);
      if (end != env && *end == '\0' && v > 0)
        return static_cast<unsigned>(v);
    }
    return 32;
  }();
  return cached;
}

} // namespace loom
