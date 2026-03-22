#ifndef LOOM_SIMULATOR_SIMTEMPORALPE_H
#define LOOM_SIMULATOR_SIMTEMPORALPE_H

#include "loom/Simulator/SimModule.h"

#include <memory>
#include <vector>

namespace loom {
namespace sim {

std::vector<IdIndex> temporalPEInputRepresentativePorts(
    const StaticPEDesc &pe, const StaticMappedModel &model);

std::vector<IdIndex> temporalPEOutputCandidatePorts(const StaticPEDesc &pe,
                                                    const StaticMappedModel &model);

std::unique_ptr<SimModule> createTemporalPEModule(const StaticPEDesc &pe,
                                                  const StaticMappedModel &model);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMTEMPORALPE_H
