#ifndef FCC_SIMULATOR_SIMARTIFACTWRITER_H
#define FCC_SIMULATOR_SIMARTIFACTWRITER_H

#include "fcc/Simulator/SimTypes.h"

#include <string>

namespace fcc {
namespace sim {

class SimArtifactWriter {
public:
  bool writeTrace(const SimResult &result, const std::string &path) const;
  bool writeStat(const SimResult &result, const std::string &path) const;
};

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_SIMARTIFACTWRITER_H
