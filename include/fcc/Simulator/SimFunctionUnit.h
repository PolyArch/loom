#ifndef FCC_SIMULATOR_SIMFUNCTIONUNIT_H
#define FCC_SIMULATOR_SIMFUNCTIONUNIT_H

#include "fcc/Simulator/SimModule.h"

#include <cstdint>
#include <memory>

namespace fcc {
namespace sim {

std::unique_ptr<SimModule> createFunctionUnitModule(
    const StaticModuleDesc &module, const StaticMappedModel &model);

bool functionUnitModuleSupportedByCycleKernel(const StaticModuleDesc &module);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMFUNCTIONUNIT_H
