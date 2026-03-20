#ifndef FCC_SIMULATOR_SIMMEMORY_H
#define FCC_SIMULATOR_SIMMEMORY_H

#include "fcc/Simulator/SimModule.h"

#include <memory>

namespace fcc {
namespace sim {

std::unique_ptr<SimModule> createMemoryModule(const StaticModuleDesc &module,
                                              const StaticMappedModel &model);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMMEMORY_H
