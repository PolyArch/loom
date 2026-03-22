#ifndef LOOM_TOOLS_LOOM_RUNTIME_H
#define LOOM_TOOLS_LOOM_RUNTIME_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include <string>

namespace mlir {
class MLIRContext;
}

namespace loom {

struct LoomArgs;

bool writeRuntimeManifest(const std::string &path,
                          const std::string &caseName,
                          const std::string &dfgMlirPath,
                          const std::string &adgMlirPath,
                          const std::string &configBinPath,
                          const std::string &simImageJsonPath,
                          const std::string &simImageBinPath,
                          const Graph &dfg, const Graph &adg,
                          const MappingState &mapping);

int runRuntimeReplay(const LoomArgs &args, mlir::MLIRContext &context);

} // namespace loom

#endif // LOOM_TOOLS_LOOM_RUNTIME_H
