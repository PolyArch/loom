#include "Common/LoomConstants.h"

#include <cstdlib>

namespace loom {

static unsigned readEnvOrDefault(const char *name, unsigned defaultValue) {
  if (const char *env = std::getenv(name)) {
    char *end = nullptr;
    unsigned long v = std::strtoul(env, &end, 10);
    if (end != env && *end == '\0' && v > 0)
      return static_cast<unsigned>(v);
  }
  return defaultValue;
}

unsigned getDefaultLoomAddrBits() {
  static unsigned cached = readEnvOrDefault("LOOM_ADDR_BITS", 48);
  return cached;
}

unsigned getDefaultLoomMemBusWidth() {
  static unsigned cached = readEnvOrDefault("LOOM_MEM_BUS_WIDTH", 32768);
  return cached;
}

} // namespace loom
