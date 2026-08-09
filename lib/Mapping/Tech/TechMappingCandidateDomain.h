#ifndef LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATEDOMAIN_H
#define LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATEDOMAIN_H

#include "TechMappingCandidate.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace loom::mapping::detail {

enum class TechMatchSeedRejectionReason : std::uint8_t {
  CapabilityInadmissible,
  CorrespondenceInadmissible,
  RealizationInadmissible,
  Count,
};

class TechMatchRowCollector final {
public:
  TechMatchRowCollector(llvm::ArrayRef<::dataflow::ActorRef> actors,
                        std::uint64_t limit,
                        TechMappingGenerationAccounting &accounting)
      : actors_(actors), limit_(limit), accounting_(accounting) {}

  bool atLimit() const { return accounting_.matchRowAttempts >= limit_; }
  bool truncated() const { return truncated_; }
  llvm::Expected<bool> beginSeed(std::vector<std::uint8_t> key);
  llvm::Error reject(TechMatchSeedRejectionReason reason);
  llvm::Error rejectCanonicalSeedRange(std::vector<std::uint8_t> firstKey,
                                       std::vector<std::uint8_t> lastKey,
                                       std::uint64_t count, bool countOverflow,
                                       TechMatchSeedRejectionReason reason);
  llvm::Error admit(TechMatchRealization realization,
                    llvm::ArrayRef<::dataflow::ActorRef> coveredActors);
  std::uint64_t rejectionCount(TechMatchSeedRejectionReason reason) const {
    return rejectionCounts_[static_cast<std::size_t>(reason)];
  }
  llvm::Expected<std::vector<TechMatchRow>> takeRows();

private:
  llvm::Expected<std::size_t> actorSlot(::dataflow::ActorRef actor) const;

  llvm::ArrayRef<::dataflow::ActorRef> actors_;
  std::uint64_t limit_;
  TechMappingGenerationAccounting &accounting_;
  std::vector<TechMatchRow> rows_;
  std::optional<std::vector<std::uint8_t>> previousSeedKey_;
  std::optional<std::vector<std::uint8_t>> activeSeedKey_;
  std::array<std::uint64_t,
             static_cast<std::size_t>(TechMatchSeedRejectionReason::Count)>
      rejectionCounts_{};
  bool truncated_ = false;
};

void appendU32(std::vector<std::uint8_t> &key, std::uint32_t value);
void appendU64(std::vector<std::uint8_t> &key, std::uint64_t value);
void appendBytes(std::vector<std::uint8_t> &key,
                 llvm::ArrayRef<std::uint8_t> bytes);

template <typename Ref>
llvm::Error appendDataflowRef(std::vector<std::uint8_t> &key,
                              const ArtifactIdentity &owner,
                              const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  appendBytes(key, *bytes);
  return llvm::Error::success();
}

template <typename Ref>
void appendFabricRef(std::vector<std::uint8_t> &key, const Ref &reference) {
  appendBytes(key, ::loom::fabric::canonicalFabricBytes(reference));
}

llvm::Error
deriveComputeRows(const TechMappingGenerationInputs &inputs,
                  llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
                  TechMatchRowCollector &collector);
llvm::Error
deriveMemoryRows(const TechMappingGenerationInputs &inputs,
                 llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
                 TechMatchRowCollector &collector);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_TECH_TECHMAPPINGCANDIDATEDOMAIN_H
