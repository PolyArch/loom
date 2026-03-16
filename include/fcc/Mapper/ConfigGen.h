#ifndef FCC_MAPPER_CONFIGGEN_H
#define FCC_MAPPER_CONFIGGEN_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include <string>
#include <vector>

namespace fcc {

class ConfigGen {
public:
  /// Generate .map.json and .map.txt from a completed mapping.
  bool generate(const MappingState &state, const Graph &dfg, const Graph &adg,
                const ADGFlattener &flattener, const std::string &basePath,
                int seed);

private:
  bool writeMapJson(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const ADGFlattener &flattener,
                    const std::string &path, int seed);
  bool writeMapText(const MappingState &state, const Graph &dfg,
                    const Graph &adg, const ADGFlattener &flattener,
                    const std::string &path);
};

} // namespace fcc

#endif // FCC_MAPPER_CONFIGGEN_H
