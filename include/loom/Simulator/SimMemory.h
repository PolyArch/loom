#ifndef LOOM_SIMULATOR_SIMMEMORY_H
#define LOOM_SIMULATOR_SIMMEMORY_H

#include "loom/Simulator/SimModule.h"

#include <memory>

namespace loom {
namespace sim {

std::unique_ptr<SimModule> createMemoryModule(const StaticModuleDesc &module,
                                              const StaticMappedModel &model);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMMEMORY_H
