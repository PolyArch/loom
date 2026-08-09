#ifndef LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSISCONFIG_H
#define LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSISCONFIG_H

#include "ExternalTool/ExternalFile.h"

#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace loom::evaluation::models {

struct CadenceVoltusStaticRailProviderBinding final {
  std::string stableProviderBuildIdentity;
  std::vector<external_tool::ExternalFileTreeMember> powerGridLibraryMembers;
  std::vector<std::string> powerGridLibraryEntrypoints;

  friend bool operator==(const CadenceVoltusStaticRailProviderBinding &lhs,
                         const CadenceVoltusStaticRailProviderBinding &rhs) {
    return lhs.stableProviderBuildIdentity == rhs.stableProviderBuildIdentity &&
           lhs.powerGridLibraryMembers == rhs.powerGridLibraryMembers &&
           lhs.powerGridLibraryEntrypoints == rhs.powerGridLibraryEntrypoints;
  }
  friend bool operator!=(const CadenceVoltusStaticRailProviderBinding &lhs,
                         const CadenceVoltusStaticRailProviderBinding &rhs) {
    return !(lhs == rhs);
  }
};

llvm::Error validateCadenceVoltusStaticRailProviderBinding(
    const CadenceVoltusStaticRailProviderBinding &binding);

} // namespace loom::evaluation::models

#endif // LOOM_EVALUATION_MODELS_PHYSICALRAILANALYSISCONFIG_H
