#ifndef LOOM_IMPLEMENTATIONPLATFORM_IMPLEMENTATIONPLATFORM_H
#define LOOM_IMPLEMENTATIONPLATFORM_IMPLEMENTATIONPLATFORM_H

#include "ImplementationPlatform/TechnologyCorner.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
}

namespace loom::platform {

enum class FpgaVendor {
  AmdXilinx,
  IntelAltera,
};

struct AsicTarget final {
  std::string technologyIdentity;
  std::string releaseIdentity;

  friend bool operator==(const AsicTarget &lhs, const AsicTarget &rhs) {
    return lhs.technologyIdentity == rhs.technologyIdentity &&
           lhs.releaseIdentity == rhs.releaseIdentity;
  }
};

struct FpgaTarget final {
  FpgaVendor vendor;
  std::string deviceOrderingCode;

  friend bool operator==(const FpgaTarget &lhs, const FpgaTarget &rhs) {
    return lhs.vendor == rhs.vendor &&
           lhs.deviceOrderingCode == rhs.deviceOrderingCode;
  }
};

using ImplementationTarget = std::variant<AsicTarget, FpgaTarget>;

struct ImplementationPlatformDraft final {
  ImplementationTarget target;
  std::vector<std::string> technologyCornerKeys;
};

struct TechnologyCorner final {
  TechnologyCornerId id;
  std::string key;

  friend bool operator==(const TechnologyCorner &lhs,
                         const TechnologyCorner &rhs) {
    return lhs.id == rhs.id && lhs.key == rhs.key;
  }
};

class ImplementationPlatform final {
public:
  const ImplementationTarget &target() const { return target_; }
  llvm::ArrayRef<TechnologyCorner> technologyCorners() const {
    return technologyCorners_;
  }
  const TechnologyCorner *findTechnologyCorner(TechnologyCornerId id) const;

private:
  ImplementationPlatform(ImplementationTarget target,
                         std::vector<TechnologyCorner> technologyCorners)
      : target_(std::move(target)),
        technologyCorners_(std::move(technologyCorners)) {}

  ImplementationTarget target_;
  std::vector<TechnologyCorner> technologyCorners_;

  friend llvm::Expected<class FinalizedImplementationPlatform>
  finalizeImplementationPlatform(ImplementationPlatformDraft,
                                 const ArtifactStore &);
  friend llvm::Expected<class FinalizedImplementationPlatform>
  importImplementationPlatform(const ArtifactRootReference &,
                               const ArtifactStore &);
};

class FinalizedImplementationPlatform final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const ImplementationPlatform &platform() const { return platform_; }

private:
  FinalizedImplementationPlatform(ArtifactRootReference reference,
                                  CanonicalSemanticBytes canonicalBytes,
                                  ImplementationPlatform platform)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        platform_(std::move(platform)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  ImplementationPlatform platform_;

  friend llvm::Expected<FinalizedImplementationPlatform>
  finalizeImplementationPlatform(ImplementationPlatformDraft,
                                 const ArtifactStore &);
  friend llvm::Expected<FinalizedImplementationPlatform>
  importImplementationPlatform(const ArtifactRootReference &,
                               const ArtifactStore &);
};

llvm::Expected<FinalizedImplementationPlatform>
finalizeImplementationPlatform(ImplementationPlatformDraft draft,
                               const ArtifactStore &store);

llvm::Expected<FinalizedImplementationPlatform>
importImplementationPlatform(const ArtifactRootReference &reference,
                             const ArtifactStore &store);

llvm::Expected<TechnologyCorner>
resolveTechnologyCorner(const TechnologyCornerRef &reference,
                        const ArtifactStore &store);

} // namespace loom::platform

#endif // LOOM_IMPLEMENTATIONPLATFORM_IMPLEMENTATIONPLATFORM_H
