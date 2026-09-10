#ifndef LOOM_RUNTIME_GEM5BUILDREADINESS_H
#define LOOM_RUNTIME_GEM5BUILDREADINESS_H

#include "Runtime/Gem5SimulationBinding.h"

namespace loom::runtime {

/// Local build declaration. Loading it does not prove executable identity;
/// the execution provider verifies the binding, version, and actual bytes.
struct Gem5BuildReadiness final {
  Gem5BuildIdentity identity;
  std::string path;
  std::string binary;
  std::string bridgeAbiIdentity;
  std::string versionProbe;
};

llvm::Expected<Gem5BuildReadiness> readGem5BuildReadiness(llvm::StringRef path);

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5BUILDREADINESS_H
