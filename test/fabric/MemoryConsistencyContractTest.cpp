#include "Fabric/IR/MemoryConsistencyContract.h"

#include "Common/Artifact.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace fabric;
using namespace loom;
using namespace loom::fabric;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid memory-consistency contract");
  llvm::consumeError(value.takeError());
}

void requireFabricError(llvm::StringRef test, llvm::Error error,
                        FabricRefErrorKind expected) {
  if (!error)
    fail(test, "accepted an invalid Fabric reference");
  const FabricRefErrorKind actual = takeFabricRefErrorKind(std::move(error));
  require(test, actual == expected,
          "unexpected Fabric reference failure classification");
}

ArtifactIdentity artifact(llvm::StringRef test, std::uint8_t seed) {
  return take(test, ArtifactIdentity::fromBytes(std::vector<std::uint8_t>(
                        ArtifactIdentity::byteSize, seed)));
}

constexpr FabricEntityId kMemory = 31;
constexpr FabricEntityId kSystemService = 61;
constexpr FabricEntityId kConsistencyDomain = 71;
constexpr FabricEntityId kClockDomain = 72;

class AnchorFabric final : public FabricArtifactView {
public:
  explicit AnchorFabric(ArtifactIdentity identity)
      : identity_(std::move(identity)) {}

  const ArtifactIdentity &identity() const override { return identity_; }
  FabricRootKind rootKind() const override { return FabricRootKind::System; }

  std::optional<FabricEntityKind> entityKind(FabricEntityId id) const override {
    switch (id) {
    case kMemory:
      return FabricEntityKind::FabricMemoryOccurrence;
    case kSystemService:
      return FabricEntityKind::SystemMemoryService;
    case kConsistencyDomain:
    case kClockDomain:
      return FabricEntityKind::HardwareDomain;
    default:
      return std::nullopt;
    }
  }

  std::uint64_t transportEndpointCount(
      const FabricTransportEndpointOwnerRef &) const override {
    return 0;
  }
  std::uint64_t memoryEndpointCount(
      const FabricMemoryEndpointOwnerRef &owner) const override {
    return owner.kind() == FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence
               ? 1
               : 0;
  }
  std::uint64_t inventorySize(const FabricInventoryOwnerRef &,
                              FabricInventoryKind) const override {
    return 0;
  }
  std::optional<FabricFuNodeKind> fuNodeKind(const FabricInventoryOwnerRef &,
                                             FabricOrdinal) const override {
    return std::nullopt;
  }
  bool
  declaresLocalMemoryService(FabricMemoryOccurrenceRef memory) const override {
    return memory.id() == kMemory;
  }
  std::optional<FabricMemoryEndpointRole>
  memoryEndpointRole(const FabricMemoryEndpointRef &endpoint) const override {
    if (endpoint.owner.kind() ==
            FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence &&
        std::get<FabricMemoryOccurrenceRef>(endpoint.owner.payload).id() ==
            kMemory)
      return FabricMemoryEndpointRole::Subordinate;
    return std::nullopt;
  }
  std::optional<FabricHardwareDomainKind>
  hardwareDomainKind(HardwareDomainRef domain) const override {
    if (domain.id() == kClockDomain)
      return FabricHardwareDomainKind::Clock;
    if (domain.id() == kConsistencyDomain)
      return FabricHardwareDomainKind::MemoryConsistency;
    return std::nullopt;
  }
  std::optional<FabricFuTemplateRef>
  fuTemplateOf(FabricFuOccurrenceRef) const override {
    return std::nullopt;
  }
  bool hasPointConnection(const FabricTransportEndpointRef &,
                          const FabricTransportEndpointRef &) const override {
    return false;
  }
  bool admitsTraversal(const FabricPhysicalTraversalRef &) const override {
    return false;
  }

private:
  ArtifactIdentity identity_;
};

ResourceContract resourceContract(llvm::StringRef test) {
  ResourceContractDeclaration declaration;
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 1;
  declaration.usePatterns = {UsePatternDeclaration{UsePatternKey(0),
                                                   RequesterKey(0),
                                                   EligibilityKey(0),
                                                   EventKey(0),
                                                   EventKey(0),
                                                   std::nullopt,
                                                   TimingContractKey(0),
                                                   {},
                                                   {}}};
  return take(test, ResourceContract::create(declaration));
}

MemoryConsistencyParticipant localService() {
  return MemoryConsistencyParticipant::service(
      FabricMemoryServiceRef::local(FabricMemoryOccurrenceRef(kMemory)));
}

MemoryConsistencyParticipant systemService() {
  return MemoryConsistencyParticipant::service(
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(kSystemService)));
}

MemoryConsistencyParticipant provider() {
  return MemoryConsistencyParticipant::provider(
      SubordinateEndpointRef(FabricMemoryEndpointRef{
          FabricMemoryEndpointOwnerRef::of(FabricMemoryOccurrenceRef(kMemory)),
          0}));
}

ClockDomainRef clock() {
  return ClockDomainRef(HardwareDomainRef(kClockDomain));
}

MemoryConsistencyContractDeclaration
declaration(llvm::StringRef test,
            std::vector<MemoryConsistencyParticipant> participants,
            ReleaseVisibilityPoint visibility,
            MemoryConsistencyProgress progress) {
  return MemoryConsistencyContractDeclaration{std::move(participants),
                                              visibility, std::move(progress),
                                              resourceContract(test)};
}

class RecordCursor {
public:
  explicit RecordCursor(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  std::size_t position() const { return offset_; }

  std::uint32_t u32(llvm::StringRef test) {
    require(test, offset_ <= bytes_.size() && bytes_.size() - offset_ >= 4,
            "truncated u32");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  std::uint64_t u64(llvm::StringRef test) {
    require(test, offset_ <= bytes_.size() && bytes_.size() - offset_ >= 8,
            "truncated u64");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  void skip(llvm::StringRef test, std::uint64_t count) {
    require(test, count <= bytes_.size() - offset_, "truncated byte string");
    offset_ += static_cast<std::size_t>(count);
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

std::pair<std::uint32_t, std::uint32_t>
wireChoiceTags(llvm::StringRef test, llvm::ArrayRef<std::uint8_t> bytes) {
  RecordCursor cursor(bytes);
  const std::uint64_t participantCount = cursor.u64(test);
  for (std::uint64_t index = 0; index < participantCount; ++index) {
    cursor.u32(test);
    cursor.skip(test, cursor.u64(test));
  }
  return {cursor.u32(test), cursor.u32(test)};
}

std::vector<std::uint8_t>
swapFirstParticipantRows(llvm::StringRef test,
                         llvm::ArrayRef<std::uint8_t> bytes) {
  RecordCursor cursor(bytes);
  require(test, cursor.u64(test) >= 2,
          "canonical fixture needs at least two participants");
  const std::size_t firstStart = cursor.position();
  cursor.u32(test);
  cursor.skip(test, cursor.u64(test));
  const std::size_t secondStart = cursor.position();
  cursor.u32(test);
  cursor.skip(test, cursor.u64(test));
  const std::size_t suffixStart = cursor.position();

  std::vector<std::uint8_t> result;
  result.insert(result.end(), bytes.begin(), bytes.begin() + firstStart);
  result.insert(result.end(), bytes.begin() + secondStart,
                bytes.begin() + suffixStart);
  result.insert(result.end(), bytes.begin() + firstStart,
                bytes.begin() + secondStart);
  result.insert(result.end(), bytes.begin() + suffixStart, bytes.end());
  return result;
}

void checkCanonicalRoundTripAndChoices() {
  const llvm::StringRef test = "canonical-roundtrip-and-choices";
  MemoryConsistencyContract bounded =
      take(test, MemoryConsistencyContract::create(declaration(
                     test, {provider(), systemService(), localService()},
                     ReleaseVisibilityPoint::AtLinearization,
                     BoundedCompletion{clock(), 19})));

  require(test, bounded.participants().size() == 3,
          "participant domain changed size");
  require(test,
          bounded.participants()[0] == localService() &&
              bounded.participants()[1] == systemService() &&
              bounded.participants()[2] == provider(),
          "participants were not normalized into canonical order");
  require(test, std::holds_alternative<BoundedCompletion>(bounded.progress()),
          "bounded progress changed variant");
  require(test,
          std::get<BoundedCompletion>(bounded.progress()) ==
              BoundedCompletion{clock(), 19},
          "bounded progress fields changed");
  require(test, bounded.resourceContract().usePatternCount() == 1,
          "embedded ResourceContract changed");

  std::vector<std::uint8_t> boundedBytes =
      take(test, encodeMemoryConsistencyContractRecord(bounded));
  require(test,
          wireChoiceTags(test, boundedBytes) ==
              std::pair<std::uint32_t, std::uint32_t>{0, 0},
          "AtLinearization or BoundedCompletion wire tag changed");
  MemoryConsistencyContract boundedDecoded =
      take(test, decodeMemoryConsistencyContractRecord(boundedBytes));
  require(test,
          take(test, encodeMemoryConsistencyContractRecord(boundedDecoded)) ==
              boundedBytes,
          "strict roundtrip changed canonical bytes");
  expectRejected<MemoryConsistencyContract>(
      test, decodeMemoryConsistencyContractRecord(
                swapFirstParticipantRows(test, boundedBytes)));

  MemoryConsistencyContract eventual =
      take(test, MemoryConsistencyContract::create(declaration(
                     test, {provider()}, ReleaseVisibilityPoint::ByRetirement,
                     FairEventual{})));
  std::vector<std::uint8_t> eventualBytes =
      take(test, encodeMemoryConsistencyContractRecord(eventual));
  require(test,
          wireChoiceTags(test, eventualBytes) ==
              std::pair<std::uint32_t, std::uint32_t>{1, 1},
          "ByRetirement or FairEventual wire tag changed");
  require(test, std::holds_alternative<FairEventual>(eventual.progress()),
          "eventual progress changed variant");
}

void checkDeclarationFailures() {
  const llvm::StringRef test = "declaration-failures";
  expectRejected<MemoryConsistencyContract>(
      test,
      MemoryConsistencyContract::create(declaration(
          test, {}, ReleaseVisibilityPoint::AtLinearization, FairEventual{})));
  expectRejected<MemoryConsistencyContract>(
      test, MemoryConsistencyContract::create(declaration(
                test, {localService(), localService()},
                ReleaseVisibilityPoint::AtLinearization, FairEventual{})));
  expectRejected<MemoryConsistencyContract>(
      test, MemoryConsistencyContract::create(declaration(
                test, {localService()}, ReleaseVisibilityPoint::AtLinearization,
                BoundedCompletion{clock(), 0})));
}

void checkReferenceValidation() {
  const llvm::StringRef test = "reference-validation";
  const ArtifactIdentity expected = artifact(test, 0x11);
  const AnchorFabric view(expected);
  const FabricImportBinding binding{expected, FabricRootKind::System};

  MemoryConsistencyContract valid =
      take(test, MemoryConsistencyContract::create(
                     declaration(test, {provider(), localService()},
                                 ReleaseVisibilityPoint::AtLinearization,
                                 BoundedCompletion{clock(), 7})));
  if (llvm::Error error =
          validateMemoryConsistencyContractReferences(valid, view, binding))
    fail(test, llvm::toString(std::move(error)));

  MemoryConsistencyContract wrongClock =
      take(test,
           MemoryConsistencyContract::create(declaration(
               test, {localService()}, ReleaseVisibilityPoint::AtLinearization,
               BoundedCompletion{
                   ClockDomainRef(HardwareDomainRef(kConsistencyDomain)), 7})));
  requireFabricError(
      test,
      validateMemoryConsistencyContractReferences(wrongClock, view, binding),
      FabricRefErrorKind::WrongEntityKind);

  const FabricImportBinding foreign{artifact(test, 0x22),
                                    FabricRootKind::System};
  requireFabricError(
      test, validateMemoryConsistencyContractReferences(valid, view, foreign),
      FabricRefErrorKind::ForeignArtifact);
}

void checkFixedSemanticsHaveNoWireOverrides() {
  const llvm::StringRef test = "fixed-semantics-have-no-wire-overrides";
  MemoryConsistencyContract contract =
      take(test, MemoryConsistencyContract::create(declaration(
                     test, {localService()},
                     ReleaseVisibilityPoint::AtLinearization, FairEventual{})));
  std::vector<std::uint8_t> bytes =
      take(test, encodeMemoryConsistencyContractRecord(contract));
  bytes.insert(bytes.end(), 8, 0);
  expectRejected<MemoryConsistencyContract>(
      test, decodeMemoryConsistencyContractRecord(bytes));
}

} // namespace

int main() {
  checkCanonicalRoundTripAndChoices();
  checkDeclarationFailures();
  checkReferenceValidation();
  checkFixedSemanticsHaveNoWireOverrides();
  return EXIT_SUCCESS;
}
