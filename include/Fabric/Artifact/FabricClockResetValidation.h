#ifndef LOOM_FABRIC_ARTIFACT_FABRICCLOCKRESETVALIDATION_H
#define LOOM_FABRIC_ARTIFACT_FABRICCLOCKRESETVALIDATION_H

#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/Support/Error.h"

#include <utility>

namespace loom::fabric {

/// Proof that the complete System-owned domain and connection relations obey
/// the clock/reset contract. This view owns no copied domain catalog.
class ValidatedClockResetView final {
public:
  const FabricSystemRootView &system() const { return system_; }

private:
  explicit ValidatedClockResetView(FabricSystemRootView system)
      : system_(std::move(system)) {}

  FabricSystemRootView system_;

  friend llvm::Expected<ValidatedClockResetView>
  validateClockReset(FabricSystemRootView system);
};

llvm::Expected<ValidatedClockResetView>
validateClockReset(FabricSystemRootView system);

/// Validates that every projected ordinary Module-local physical connection
/// remains within one symbolic Clock and Reset slot pair.
llvm::Error validateModuleClockReset(const FabricModuleRootView &module);

} // namespace loom::fabric

#endif // LOOM_FABRIC_ARTIFACT_FABRICCLOCKRESETVALIDATION_H
