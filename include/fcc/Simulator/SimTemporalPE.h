#ifndef FCC_SIMULATOR_SIMTEMPORALPE_H
#define FCC_SIMULATOR_SIMTEMPORALPE_H

#include "fcc/Simulator/SimModule.h"

#include <memory>
#include <vector>

namespace fcc {
namespace sim {

std::vector<IdIndex> temporalPEInputRepresentativePorts(
    const StaticPEDesc &pe, const StaticMappedModel &model);

std::vector<IdIndex> temporalPEOutputCandidatePorts(const StaticPEDesc &pe,
                                                    const StaticMappedModel &model);

std::unique_ptr<SimModule> createTemporalPEModule(const StaticPEDesc &pe,
                                                  const StaticMappedModel &model);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMTEMPORALPE_H
