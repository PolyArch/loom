#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H

#include "Fabric/IR/FuCapabilityDomain.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace mlir {
class Operation;
}

namespace fabric {
class FuOp;
}

namespace loom::fabric::detail {

enum class FabricFuCapabilityOrdinalSpace : std::uint8_t {
  AuthoringPhysical,
  CanonicalDefinition,
};

/// Returns the owning FU's finite correlated capability domain expressed in
/// `canonicalNodeOrder`. `sourceOrdinalSpace` identifies whether an existing
/// carrier still uses authoring physical ordinals or already uses canonical
/// definition ordinals. A missing authoring attribute is accepted only when
/// the physical graph has one unambiguous template.
llvm::Expected<::fabric::FuCapabilityDomainRecord>
canonicalizeFabricFuCapabilityDomain(
    ::fabric::FuOp fu, llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder,
    FabricFuCapabilityOrdinalSpace sourceOrdinalSpace);

/// Canonicalizes one exact source row into the supplied definition-node order.
/// This is the row-level form used by transient authoring correspondence; the
/// returned value owns no persistent identity beyond the enclosing FU domain.
llvm::Expected<::fabric::FuCapabilityTemplateSelection>
canonicalizeFabricFuCapabilityTemplate(
    ::fabric::FuOp fu, const ::fabric::FuCapabilityTemplateSelection &selection,
    llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder,
    FabricFuCapabilityOrdinalSpace sourceOrdinalSpace);

llvm::Expected<FabricFuCapabilityTemplateRecord>
deriveFabricFuCapabilityTemplate(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder,
    const ::fabric::FuCapabilityTemplateSelection &selection);

llvm::Expected<std::vector<FabricFuCapabilityTemplateRecord>>
deriveFabricFuCapabilityTemplates(
    ::fabric::FuOp fu, FabricFuTemplateRef owner,
    llvm::ArrayRef<mlir::Operation *> canonicalNodeOrder);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICFUCAPABILITYDERIVATION_H
