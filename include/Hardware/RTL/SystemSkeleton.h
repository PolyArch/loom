#ifndef LOOM_HARDWARE_RTL_SYSTEMSKELETON_H
#define LOOM_HARDWARE_RTL_SYSTEMSKELETON_H

#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/Specialization.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <vector>

namespace loom::hardware::rtl {

/// One transient, fully specialized CIRCT hierarchy rooted at the exact
/// Fabric System. Only SpatialCore occurrences are concretely instantiated;
/// their System-facing ports preserve the occurrence-qualified interfaces.
struct SystemRootCirctSkeleton final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<ImplementationInterface> interfaces;
  std::size_t spatialDefinitionCount = 0;
  std::size_t spatialInstanceCount = 0;
};

/// Builds the self-contained portable System hierarchy. Identical
/// definition-rebased SpatialCore specializations share one RTL definition,
/// while every physical occurrence retains its own System instance and ports.
llvm::Expected<SystemRootCirctSkeleton> buildPortableSystemRootCirctSkeleton(
    mlir::MLIRContext &context,
    const FinalizedConfigurationABI &configurationAbi,
    const FabricOperationProviderRegistry &providers,
    const ExternalImplementationContractCatalog &externalContracts);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_SYSTEMSKELETON_H
