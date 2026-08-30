#ifndef LOOM_FABRIC_ARTIFACT_FABRICARTIFACTCODEC_H
#define LOOM_FABRIC_ARTIFACT_FABRICARTIFACTCODEC_H

#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
namespace fabric {

inline constexpr ArtifactSchemaDescriptor fabricArtifactSchema{
    "loom.fabric", SchemaVersion{7, 1}};

enum class FabricDependencyRole : std::uint32_t {
  ImportedModule = 0,
  RefinedSystem = 1,
  ImplementationInput = 2,
};

struct FabricDirectDependency {
  FabricDependencyRole role = FabricDependencyRole::ImportedModule;
  ArtifactRootReference root;

  friend bool operator==(const FabricDirectDependency &lhs,
                         const FabricDirectDependency &rhs) {
    return lhs.role == rhs.role && lhs.root == rhs.root;
  }
  friend bool operator!=(const FabricDirectDependency &lhs,
                         const FabricDirectDependency &rhs) {
    return !(lhs == rhs);
  }
};

struct DecodedFabricArtifact {
  FabricRootKind rootKind = FabricRootKind::Module;
  std::vector<FabricDirectDependency> dependencies;
  std::vector<std::uint8_t> canonicalMlirBytecode;
};

llvm::Expected<CanonicalSemanticBytes> encodeFabricArtifactEnvelope(
    FabricRootKind rootKind,
    llvm::ArrayRef<FabricDirectDependency> dependencies,
    llvm::ArrayRef<std::uint8_t> canonicalMlirBytecode);

llvm::Expected<DecodedFabricArtifact>
decodeFabricArtifactEnvelope(llvm::ArrayRef<std::uint8_t> bytes);

/// The canonical sorted positions of the given dependencies: result[i] is the
/// rank of row i among the canonically sorted encoded rows. The payload
/// embeds dependency-table ordinals of that sorted order, so a migration that
/// rewrites rows must preserve this permutation exactly.
llvm::Expected<std::vector<std::uint32_t>> canonicalFabricDependencyOrder(
    llvm::ArrayRef<FabricDirectDependency> dependencies);

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_ARTIFACT_FABRICARTIFACTCODEC_H
