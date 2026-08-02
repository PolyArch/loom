#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICMEMORYENGINETEMPLATE_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICMEMORYENGINETEMPLATE_H

#include "Fabric/Identity/FabricRefImport.h"

#include "Fabric/IR/FabricOps.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace loom::fabric::detail {

struct DerivedFabricMemoryEngineTemplate {
  FabricMemoryEngineTemplateRecord record;
  std::vector<std::uint8_t> canonicalBytes;
};

llvm::Expected<::mlir::FunctionType>
resolveFabricMemoryFunctionType(::fabric::MemOp memory);

llvm::Expected<std::optional<DerivedFabricMemoryEngineTemplate>>
deriveFabricMemoryEngineTemplate(::fabric::MemOp memory);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICMEMORYENGINETEMPLATE_H
