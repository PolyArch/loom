#ifndef FCC_TOOLS_FCC_RUNTIME_H
#define FCC_TOOLS_FCC_RUNTIME_H

#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"

#include <string>

namespace mlir {
class MLIRContext;
}

namespace fcc {

struct FccArgs;

bool writeRuntimeManifest(const std::string &path,
                          const std::string &caseName,
                          const std::string &dfgMlirPath,
                          const std::string &adgMlirPath,
                          const std::string &configBinPath,
                          const std::string &simImageJsonPath,
                          const std::string &simImageBinPath,
                          const Graph &dfg, const Graph &adg,
                          const MappingState &mapping);

int runRuntimeReplay(const FccArgs &args, mlir::MLIRContext &context);

} // namespace fcc

#endif // FCC_TOOLS_FCC_RUNTIME_H
