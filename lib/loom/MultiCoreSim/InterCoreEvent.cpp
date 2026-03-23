#include "loom/MultiCoreSim/InterCoreEvent.h"

// InterCoreEvent types are primarily data structures defined in the header.
// This translation unit ensures the types have a single definition home and
// provides any non-inline helper functions.

namespace loom {
namespace mcsim {

// Currently all InterCoreEvent types are header-only POD structs.
// This file exists as a compilation anchor and for future helper functions.

} // namespace mcsim
} // namespace loom
