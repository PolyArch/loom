#ifndef LOOM_SIMULATOR_SIMFUNCTIONUNIT_H
#define LOOM_SIMULATOR_SIMFUNCTIONUNIT_H

#include "loom/Simulator/SimModule.h"

#include <cstdint>
#include <memory>

namespace loom {
namespace sim {

std::unique_ptr<SimModule> createFunctionUnitModule(
    const StaticModuleDesc &module, const StaticMappedModel &model,
    bool allowTemporalPE = false);

bool functionUnitModuleSupportedByCycleKernel(const StaticModuleDesc &module,
                                              bool allowTemporalPE = false);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMFUNCTIONUNIT_H
