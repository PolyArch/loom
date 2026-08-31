#ifndef LOOM_TEST_DSE_JOINTDESIGNEXPLORATIONFIXTURE_H
#define LOOM_TEST_DSE_JOINTDESIGNEXPLORATIONFIXTURE_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Fabric/Artifact/FabricArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <vector>

namespace mlir {
class MLIRContext;
}

namespace loom {
class ArtifactStore;
class BlobStore;
struct ResolvedConfig;
} // namespace loom

namespace loom::dse::joint_test {

class TemporaryDirectory final {
public:
  TemporaryDirectory();
  ~TemporaryDirectory();

  llvm::StringRef path() const;

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext();

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  std::int32_t constant);

ArtifactRootReference
publishApplicationWorkload(const dataflow::CanonicalDataflowArtifact &artifact,
                           const ArtifactStore &store);

ArtifactRootReference
publishApplicationRuntimeInput(const ArtifactRootReference &workload,
                               std::int32_t value, const ArtifactStore &store);

evaluation::models::FpaFeatureView
projectFpaFeatures(const ArtifactRootReference &dataflow,
                   const ArtifactRootReference &system,
                   const ResolvedConfig &config, const ArtifactStore &artifacts,
                   const BlobStore &blobs);

std::vector<fabric::FabricModuleEntityCorrespondence>
identityModuleEntityCorrespondence(const fabric::FabricArtifactView &module);

bool everyCoreIsUsed(const ArtifactRootReference &systemReference,
                     llvm::ArrayRef<ArtifactRootReference> mappings,
                     const ArtifactStore &store);

} // namespace loom::dse::joint_test

#endif // LOOM_TEST_DSE_JOINTDESIGNEXPLORATIONFIXTURE_H
