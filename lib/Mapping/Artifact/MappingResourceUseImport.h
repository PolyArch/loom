#ifndef LOOM_LIB_MAPPING_ARTIFACT_MAPPINGRESOURCEUSEIMPORT_H
#define LOOM_LIB_MAPPING_ARTIFACT_MAPPINGRESOURCEUSEIMPORT_H

#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/IR/MappingOps.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace loom::mapping::detail {

struct ImportedPatternValues final {
  std::vector<::fabric::UsePatternValue> parameters;
  std::vector<::fabric::UsePatternValue> sharingAssignments;
};

llvm::Expected<ImportedPatternValues> importResourceUsePatternValues(
    ::mapping::ResourceUseOp record,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricUsePatternRef &pattern);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_MAPPINGRESOURCEUSEIMPORT_H
