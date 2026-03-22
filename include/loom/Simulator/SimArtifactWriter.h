#ifndef LOOM_SIMULATOR_SIMARTIFACTWRITER_H
#define LOOM_SIMULATOR_SIMARTIFACTWRITER_H

#include "loom/Simulator/SimTypes.h"

#include <string>

namespace loom {
namespace sim {

class SimArtifactWriter {
public:
  bool writeTrace(const SimResult &result, const std::string &path) const;
  bool writeStat(const SimResult &result, const std::string &path) const;
};

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_SIMARTIFACTWRITER_H
