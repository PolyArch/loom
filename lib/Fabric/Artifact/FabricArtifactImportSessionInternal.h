#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTIMPORTSESSIONINTERNAL_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTIMPORTSESSIONINTERNAL_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricHandshake.h"

#include "llvm/Support/Error.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace loom::fabric::detail {

struct FabricStrictImportResult;

inline constexpr std::uint64_t fabricArtifactImportAlgorithmVersion = 1;

struct FabricArtifactImportSessionKey final {
  ArtifactRootReference reference;
  std::uint64_t algorithmVersion = fabricArtifactImportAlgorithmVersion;
};

struct FabricArtifactImportSessionEntry final {
  std::shared_ptr<const FabricStrictImportResult> imported;
  std::uint64_t retainedPayloadBytes = 0;
  std::uint64_t constructionNanoseconds = 0;
};

class FabricArtifactImportSessionState final {
public:
  struct Lookup final {
    std::shared_ptr<const FabricArtifactImportSessionEntry> entry;
    bool reservedConstruction = false;
  };

  explicit FabricArtifactImportSessionState(std::size_t entryLimit);

  llvm::Expected<Lookup>
  lookupOrReserve(const ArtifactRootReference &reference);
  void recordRevalidation(std::uint64_t byteCount);
  std::shared_ptr<const FabricArtifactImportSessionEntry>
  complete(const ArtifactRootReference &reference,
           std::shared_ptr<const FabricStrictImportResult> imported,
           std::uint64_t retainedPayloadBytes,
           std::uint64_t constructionNanoseconds);
  void abandon(const ArtifactRootReference &reference,
               std::uint64_t constructionNanoseconds);
  std::shared_ptr<const FabricHandshakeContext>
  lookupHandshakeContext(const ArtifactIdentity &fabric);
  void retainHandshakeContext(
      const ArtifactIdentity &fabric,
      std::shared_ptr<const FabricHandshakeContext> context);
  FabricArtifactImportSessionStatistics statistics() const;

private:
  static void add(std::uint64_t &destination, std::uint64_t value);

  struct KeyLess final {
    bool operator()(const FabricArtifactImportSessionKey &lhs,
                    const FabricArtifactImportSessionKey &rhs) const;
  };

  const std::size_t entryLimit_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::map<FabricArtifactImportSessionKey,
           std::shared_ptr<const FabricArtifactImportSessionEntry>, KeyLess>
      entries_;
  std::map<FabricArtifactImportSessionKey, std::thread::id, KeyLess>
      constructing_;
  struct IdentityLess final {
    bool operator()(const ArtifactIdentity &lhs,
                    const ArtifactIdentity &rhs) const {
      return lhs.bytes() < rhs.bytes();
    }
  };
  std::map<ArtifactIdentity, std::shared_ptr<const FabricHandshakeContext>,
           IdentityLess>
      handshakeContexts_;
  FabricArtifactImportSessionStatistics statistics_;
};

std::shared_ptr<FabricArtifactImportSessionState>
currentFabricArtifactImportSession();

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTIMPORTSESSIONINTERNAL_H
