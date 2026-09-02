#ifndef LOOM_HARDWARE_RTL_COMMONSKELETON_H
#define LOOM_HARDWARE_RTL_COMMONSKELETON_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace loom::hardware::rtl {

struct RtlModuleGraphProjection;

inline constexpr llvm::StringLiteral fabricOperationGeneratorSchemaSymbol =
    "loom_fabric_operation";
inline constexpr llvm::StringLiteral fabricOperationGeneratorDescriptor =
    "loom.fabric.operation";

/// Transient association between one abstract CIRCT leaf and the exact
/// Fabric operation occurrence that owns its capability semantics.
struct FabricOperationLeafAssociation final {
  circt::hw::HWModuleGeneratedOp module;
  fabric::FabricPhysicalOccurrenceOwnerRef occurrence;
};

/// One standalone CIRCT container built from an exact Fabric Module root.
/// Operation-leaf associations remain transient handles into `module`.
struct ModuleRootCirctSkeleton final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<FabricOperationLeafAssociation> operationLeaves;
};

/// Optional graph capture for the generated portable SpatialCore path. The
/// output is populated only after exact post-lowering framed emission and a
/// cold comparison of the post-export HW graph.
struct RtlModuleGraphCapture final {
  llvm::StringRef exactTopModule;
  RtlModuleGraphProjection *output = nullptr;
};

/// A valid finalized Fabric whose structural requirements are outside the
/// target-independent lowerer's supported domain. This is distinct from an
/// invalid Fabric, ABI mismatch, or malformed CIRCT module.
class FabricStructuralLoweringUnsupportedError final
    : public llvm::ErrorInfo<FabricStructuralLoweringUnsupportedError> {
public:
  static char ID;

  explicit FabricStructuralLoweringUnsupportedError(std::string reason)
      : reason_(std::move(reason)) {}

  llvm::StringRef reason() const { return reason_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string reason_;
};

/// Builds one target-independent CIRCT module from a finalized Fabric Module
/// root. Validation and construction happen off to the side; failure publishes
/// no partial skeleton.
llvm::Expected<ModuleRootCirctSkeleton>
buildModuleRootCirctSkeleton(mlir::MLIRContext &context,
                             fabric::SpatialCoreOccurrenceRef spatialCore,
                             const FinalizedConfigurationABI &configurationAbi,
                             llvm::StringRef materializationKey = {});

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const ConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves);

/// Verifies a specialized module and rejects any remaining Loom abstract leaf.
llvm::Error verifySpecializedCirctModule(mlir::ModuleOp module);

/// Verifies, lowers Seq to SV, verifies again, and exports SystemVerilog.
/// The input module is consumed by the lowering pipeline and must contain no
/// unresolved Loom Fabric operation leaf.
llvm::Expected<std::string> lowerAndExportSpecializedSystemVerilog(
    mlir::ModuleOp module, llvm::StringRef materializationKey = {},
    std::optional<RtlModuleGraphCapture> moduleGraph = std::nullopt);

llvm::Error lowerAndExportSpecializedSystemVerilog(
    mlir::ModuleOp module, llvm::raw_ostream &output,
    llvm::StringRef materializationKey = {},
    std::optional<RtlModuleGraphCapture> moduleGraph = std::nullopt);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_COMMONSKELETON_H
