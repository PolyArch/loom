#include "Fabric/IR/MemoryConsistencyContract.h"

#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

using loom::fabric::ClockDomainRef;
using loom::fabric::FabricArtifactView;
using loom::fabric::FabricImportBinding;
using loom::fabric::FabricMemoryServiceRef;
using loom::fabric::SubordinateEndpointRef;

namespace fabric {
namespace {

enum class ParticipantWireTag : std::uint32_t {
  Service = 0,
  Provider = 1,
};

enum class ReleaseVisibilityWireTag : std::uint32_t {
  AtLinearization = 0,
  ByRetirement = 1,
};

enum class ProgressWireTag : std::uint32_t {
  BoundedCompletion = 0,
  FairEventual = 1,
};

llvm::Error invalidRecord(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "invalid MemoryConsistencyContractRecord: " +
                                     message);
}

llvm::Error rejected(MemoryConsistencyContractViolation violation,
                     const llvm::Twine &message) {
  return llvm::make_error<MemoryConsistencyContractError>(
      violation,
      (getMemoryConsistencyContractViolationName(violation) + ": " + message)
          .str());
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendByteString(std::vector<std::uint8_t> &bytes,
                      llvm::ArrayRef<std::uint8_t> value) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                "persistent lengths require a uint64-compatible size_t");
  appendU64(bytes, static_cast<std::uint64_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t participantWireTag(MemoryConsistencyParticipantKind kind) {
  switch (kind) {
  case MemoryConsistencyParticipantKind::Service:
    return static_cast<std::uint32_t>(ParticipantWireTag::Service);
  case MemoryConsistencyParticipantKind::Provider:
    return static_cast<std::uint32_t>(ParticipantWireTag::Provider);
  }
  llvm_unreachable("unknown participant kind");
}

std::uint32_t releaseVisibilityWireTag(ReleaseVisibilityPoint point) {
  switch (point) {
  case ReleaseVisibilityPoint::AtLinearization:
    return static_cast<std::uint32_t>(
        ReleaseVisibilityWireTag::AtLinearization);
  case ReleaseVisibilityPoint::ByRetirement:
    return static_cast<std::uint32_t>(ReleaseVisibilityWireTag::ByRetirement);
  }
  llvm_unreachable("unknown release visibility point");
}

std::uint32_t progressWireTag(const MemoryConsistencyProgress &progress) {
  if (std::holds_alternative<BoundedCompletion>(progress))
    return static_cast<std::uint32_t>(ProgressWireTag::BoundedCompletion);
  return static_cast<std::uint32_t>(ProgressWireTag::FairEventual);
}

std::vector<std::uint8_t>
participantReferenceBytes(const MemoryConsistencyParticipant &participant) {
  switch (participant.kind()) {
  case MemoryConsistencyParticipantKind::Service:
    return loom::fabric::canonicalFabricBytes(
        std::get<FabricMemoryServiceRef>(participant.payload));
  case MemoryConsistencyParticipantKind::Provider:
    return loom::fabric::canonicalFabricBytes(
        std::get<SubordinateEndpointRef>(participant.payload));
  }
  llvm_unreachable("unknown participant kind");
}

std::vector<std::uint8_t>
participantOrderingKey(const MemoryConsistencyParticipant &participant) {
  std::vector<std::uint8_t> key;
  appendU32(key, participantWireTag(participant.kind()));
  std::vector<std::uint8_t> reference = participantReferenceBytes(participant);
  key.insert(key.end(), reference.begin(), reference.end());
  return key;
}

class RecordReader {
public:
  explicit RecordReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32(llvm::StringRef field) {
    if (remaining() < 4)
      return invalidRecord("truncated " + field);
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64(llvm::StringRef field) {
    if (remaining() < 8)
      return invalidRecord("truncated " + field);
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  readByteString(llvm::StringRef field) {
    const std::string lengthField = (field + " length").str();
    llvm::Expected<std::uint64_t> length = readU64(lengthField);
    if (!length)
      return length.takeError();
    if (*length > remaining())
      return invalidRecord("truncated " + field);
    llvm::ArrayRef<std::uint8_t> result =
        bytes_.slice(offset_, static_cast<std::size_t>(*length));
    offset_ += static_cast<std::size_t>(*length);
    return result;
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

  llvm::Error finish() const {
    if (remaining() != 0)
      return invalidRecord("trailing bytes");
    return llvm::Error::success();
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<MemoryConsistencyParticipant>
decodeParticipant(std::uint32_t tag, llvm::ArrayRef<std::uint8_t> bytes) {
  switch (tag) {
  case static_cast<std::uint32_t>(ParticipantWireTag::Service): {
    llvm::Expected<FabricMemoryServiceRef> service =
        loom::fabric::decodeFabricRef<FabricMemoryServiceRef>(bytes);
    if (!service)
      return service.takeError();
    return MemoryConsistencyParticipant::service(std::move(*service));
  }
  case static_cast<std::uint32_t>(ParticipantWireTag::Provider): {
    llvm::Expected<SubordinateEndpointRef> provider =
        loom::fabric::decodeFabricRef<SubordinateEndpointRef>(bytes);
    if (!provider)
      return provider.takeError();
    return MemoryConsistencyParticipant::provider(std::move(*provider));
  }
  default:
    return invalidRecord("unknown participant tag");
  }
}

llvm::Expected<ReleaseVisibilityPoint>
decodeReleaseVisibility(std::uint32_t tag) {
  switch (tag) {
  case static_cast<std::uint32_t>(ReleaseVisibilityWireTag::AtLinearization):
    return ReleaseVisibilityPoint::AtLinearization;
  case static_cast<std::uint32_t>(ReleaseVisibilityWireTag::ByRetirement):
    return ReleaseVisibilityPoint::ByRetirement;
  default:
    return invalidRecord("unknown release visibility tag");
  }
}

llvm::Expected<MemoryConsistencyProgress> decodeProgress(RecordReader &reader,
                                                         std::uint32_t tag) {
  switch (tag) {
  case static_cast<std::uint32_t>(ProgressWireTag::BoundedCompletion): {
    llvm::Expected<llvm::ArrayRef<std::uint8_t>> clockBytes =
        reader.readByteString("progress clock");
    if (!clockBytes)
      return clockBytes.takeError();
    llvm::Expected<ClockDomainRef> clock =
        loom::fabric::decodeFabricRef<ClockDomainRef>(*clockBytes);
    if (!clock)
      return clock.takeError();
    llvm::Expected<std::uint64_t> ticks =
        reader.readU64("max issue-to-retire ticks");
    if (!ticks)
      return ticks.takeError();
    return MemoryConsistencyProgress(
        std::in_place_type<BoundedCompletion>,
        BoundedCompletion{std::move(*clock), *ticks});
  }
  case static_cast<std::uint32_t>(ProgressWireTag::FairEventual):
    return MemoryConsistencyProgress(std::in_place_type<FairEventual>);
  default:
    return invalidRecord("unknown progress tag");
  }
}

} // namespace

char MemoryConsistencyContractError::ID = 0;

llvm::StringRef getMemoryConsistencyContractViolationName(
    MemoryConsistencyContractViolation violation) {
  switch (violation) {
  case MemoryConsistencyContractViolation::EmptyParticipantDomain:
    return "empty_participant_domain";
  case MemoryConsistencyContractViolation::DuplicateParticipant:
    return "duplicate_participant";
  case MemoryConsistencyContractViolation::InvalidReleaseVisibilityPoint:
    return "invalid_release_visibility_point";
  case MemoryConsistencyContractViolation::NonPositiveCompletionBound:
    return "non_positive_completion_bound";
  }
  llvm_unreachable("unknown MemoryConsistencyContract violation");
}

void MemoryConsistencyContractError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code MemoryConsistencyContractError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<MemoryConsistencyContract> MemoryConsistencyContract::create(
    MemoryConsistencyContractDeclaration declaration) {
  if (declaration.participants.empty())
    return rejected(MemoryConsistencyContractViolation::EmptyParticipantDomain,
                    "a MemoryConsistency domain requires a participant");

  if (declaration.releaseVisibilityPoint !=
          ReleaseVisibilityPoint::AtLinearization &&
      declaration.releaseVisibilityPoint !=
          ReleaseVisibilityPoint::ByRetirement)
    return rejected(
        MemoryConsistencyContractViolation::InvalidReleaseVisibilityPoint,
        "release visibility is outside the closed schema");

  if (const auto *bounded =
          std::get_if<BoundedCompletion>(&declaration.progress))
    if (bounded->maxIssueToRetireTicks == 0)
      return rejected(
          MemoryConsistencyContractViolation::NonPositiveCompletionBound,
          "BoundedCompletion requires a positive issue-to-retire bound");

  struct ParticipantRow {
    MemoryConsistencyParticipant participant;
    std::vector<std::uint8_t> orderingKey;
  };
  std::vector<ParticipantRow> rows;
  rows.reserve(declaration.participants.size());
  for (MemoryConsistencyParticipant &participant : declaration.participants) {
    std::vector<std::uint8_t> orderingKey = participantOrderingKey(participant);
    rows.push_back(
        ParticipantRow{std::move(participant), std::move(orderingKey)});
  }
  std::sort(rows.begin(), rows.end(),
            [](const ParticipantRow &lhs, const ParticipantRow &rhs) {
              return lhs.orderingKey < rhs.orderingKey;
            });

  std::vector<MemoryConsistencyParticipant> participants;
  participants.reserve(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (index != 0 && rows[index - 1].orderingKey == rows[index].orderingKey)
      return rejected(
          MemoryConsistencyContractViolation::DuplicateParticipant,
          "the complete typed participant reference appears more than once");
    participants.push_back(std::move(rows[index].participant));
  }

  return MemoryConsistencyContract(
      std::move(participants), declaration.releaseVisibilityPoint,
      std::move(declaration.progress), std::move(declaration.resourceContract));
}

llvm::Expected<std::vector<std::uint8_t>> encodeMemoryConsistencyContractRecord(
    const MemoryConsistencyContract &contract) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, contract.participants().size());
  for (const MemoryConsistencyParticipant &participant :
       contract.participants()) {
    appendU32(bytes, participantWireTag(participant.kind()));
    appendByteString(bytes, participantReferenceBytes(participant));
  }

  appendU32(bytes, releaseVisibilityWireTag(contract.releaseVisibilityPoint()));
  appendU32(bytes, progressWireTag(contract.progress()));

  if (const auto *bounded =
          std::get_if<BoundedCompletion>(&contract.progress())) {
    appendByteString(
        bytes, loom::fabric::canonicalFabricBytes(bounded->progressClock));
    appendU64(bytes, bounded->maxIssueToRetireTicks);
  }

  llvm::Expected<std::vector<std::uint8_t>> resource =
      encodeResourceContractRecord(contract.resourceContract());
  if (!resource)
    return resource.takeError();
  appendByteString(bytes, *resource);
  return bytes;
}

llvm::Expected<MemoryConsistencyContract>
decodeMemoryConsistencyContractRecord(llvm::ArrayRef<std::uint8_t> bytes) {
  RecordReader reader(bytes);
  llvm::Expected<std::uint64_t> participantCount =
      reader.readU64("participant count");
  if (!participantCount)
    return participantCount.takeError();
  constexpr std::size_t minimumParticipantBytes = 4 + 8 + 12;
  if (*participantCount > reader.remaining() / minimumParticipantBytes)
    return invalidRecord("participant count exceeds remaining framing");

  std::vector<MemoryConsistencyParticipant> participants;
  participants.reserve(static_cast<std::size_t>(*participantCount));
  for (std::uint64_t index = 0; index < *participantCount; ++index) {
    llvm::Expected<std::uint32_t> tag = reader.readU32("participant tag");
    if (!tag)
      return tag.takeError();
    llvm::Expected<llvm::ArrayRef<std::uint8_t>> reference =
        reader.readByteString("participant reference");
    if (!reference)
      return reference.takeError();
    llvm::Expected<MemoryConsistencyParticipant> participant =
        decodeParticipant(*tag, *reference);
    if (!participant)
      return participant.takeError();
    participants.push_back(std::move(*participant));
  }

  llvm::Expected<std::uint32_t> visibilityTag =
      reader.readU32("release visibility tag");
  if (!visibilityTag)
    return visibilityTag.takeError();
  llvm::Expected<ReleaseVisibilityPoint> visibility =
      decodeReleaseVisibility(*visibilityTag);
  if (!visibility)
    return visibility.takeError();

  llvm::Expected<std::uint32_t> progressTag = reader.readU32("progress tag");
  if (!progressTag)
    return progressTag.takeError();
  llvm::Expected<MemoryConsistencyProgress> progress =
      decodeProgress(reader, *progressTag);
  if (!progress)
    return progress.takeError();

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> resourceBytes =
      reader.readByteString("ResourceContract record");
  if (!resourceBytes)
    return resourceBytes.takeError();
  llvm::Expected<ResourceContract> resource =
      decodeResourceContractRecord(*resourceBytes);
  if (!resource)
    return resource.takeError();
  if (llvm::Error error = reader.finish())
    return std::move(error);

  llvm::Expected<MemoryConsistencyContract> contract =
      MemoryConsistencyContract::create(MemoryConsistencyContractDeclaration{
          std::move(participants), *visibility, std::move(*progress),
          std::move(*resource)});
  if (!contract)
    return contract.takeError();
  llvm::Expected<std::vector<std::uint8_t>> canonical =
      encodeMemoryConsistencyContractRecord(*contract);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalidRecord("noncanonical encoding");
  return std::move(*contract);
}

llvm::Error validateMemoryConsistencyContractReferences(
    const MemoryConsistencyContract &contract, const FabricArtifactView &view,
    const FabricImportBinding &binding) {
  if (llvm::Error error = loom::fabric::checkFabricBinding(view, binding))
    return error;
  for (const MemoryConsistencyParticipant &participant :
       contract.participants()) {
    switch (participant.kind()) {
    case MemoryConsistencyParticipantKind::Service:
      if (llvm::Error error = loom::fabric::validateFabricRef(
              view, std::get<FabricMemoryServiceRef>(participant.payload)))
        return error;
      break;
    case MemoryConsistencyParticipantKind::Provider:
      if (llvm::Error error = loom::fabric::validateFabricRef(
              view, std::get<SubordinateEndpointRef>(participant.payload)))
        return error;
      break;
    }
  }
  if (const auto *bounded =
          std::get_if<BoundedCompletion>(&contract.progress()))
    return loom::fabric::validateFabricRef(view, bounded->progressClock);
  return llvm::Error::success();
}

} // namespace fabric
