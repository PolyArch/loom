#include "Dataflow/IR/DataflowArtifactLocalReference.h"

#include <array>
#include <cstddef>

namespace dataflow {
namespace {

constexpr std::array<DataflowArtifactLocalReferenceKindDescriptor,
                     dataflowArtifactLocalReferenceKindCount()>
    kindCatalog = {{
#define LOOM_DATAFLOW_LOCAL_REFERENCE_KIND(Ordinal, Type)                      \
  {DataflowArtifactLocalReferenceKind::Type, llvm::StringLiteral(#Type)},
#include "Dataflow/IR/DataflowRefs.def"
    }};

constexpr bool kindCatalogIsDense() {
  for (std::size_t index = 0; index < kindCatalog.size(); ++index)
    if (dataflowArtifactLocalReferenceKindOrdinal(kindCatalog[index].kind) !=
        index)
      return false;
  return true;
}

static_assert(kindCatalogIsDense(),
              "Canonical Dataflow local kinds must be dense and ordered");

} // namespace

llvm::ArrayRef<DataflowArtifactLocalReferenceKindDescriptor>
dataflowArtifactLocalReferenceKindCatalog() {
  return kindCatalog;
}

} // namespace dataflow
