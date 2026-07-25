#include "Fabric/IR/ResourceContractRecord.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace fabric {
namespace {

enum class CommitRecordTag : std::uint32_t {
  Absent = 0,
  Present = 1,
};

enum class GrantPolicyRecordTag : std::uint32_t {
  Absent = 0,
  FixedPriority = 1,
  RoundRobin = 2,
};

llvm::Error invalidRecord(const llvm::Twine &message) {
  return llvm::createStringError("invalid ResourceContractRecord: %s",
                                 message.str().c_str());
}

// Every scalar in the embedded record is one canonical big-endian uint32.
void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Error appendCount(std::vector<std::uint8_t> &bytes, std::size_t count,
                        llvm::StringRef field) {
  if (count > std::numeric_limits<std::uint32_t>::max())
    return invalidRecord(field + " exceeds uint32");
  appendU32(bytes, static_cast<std::uint32_t>(count));
  return llvm::Error::success();
}

class RecordReader {
public:
  explicit RecordReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes(bytes) {}

  llvm::Error readU32(std::uint32_t &value, llvm::StringRef field) {
    if (bytes.size() - offset < 4)
      return invalidRecord("truncated " + field);
    value = (static_cast<std::uint32_t>(bytes[offset]) << 24) |
            (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
            (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
            static_cast<std::uint32_t>(bytes[offset + 3]);
    offset += 4;
    return llvm::Error::success();
  }

  llvm::Error readCount(std::uint32_t &value, llvm::StringRef field,
                        std::uint32_t minimumWordsPerEntry) {
    if (llvm::Error error = readU32(value, field))
      return error;
    if (minimumWordsPerEntry != 0 &&
        value > remainingWords() / minimumWordsPerEntry)
      return invalidRecord(field + " exceeds remaining record framing");
    return llvm::Error::success();
  }

  llvm::Error finish() const {
    if (offset != bytes.size())
      return invalidRecord("trailing bytes");
    return llvm::Error::success();
  }

private:
  std::size_t remainingWords() const { return (bytes.size() - offset) / 4; }

  llvm::ArrayRef<std::uint8_t> bytes;
  std::size_t offset = 0;
};

llvm::Error appendUsePattern(std::vector<std::uint8_t> &bytes,
                             const ResourceContract &contract,
                             UsePatternKey key) {
  const UsePattern pattern = contract.usePattern(key);
  appendU32(bytes, pattern.requester.ordinal());
  appendU32(bytes, pattern.eligibility.ordinal());
  appendU32(bytes, pattern.acquire.ordinal());
  appendU32(bytes, pattern.release.ordinal());
  if (pattern.commit) {
    appendU32(bytes, static_cast<std::uint32_t>(CommitRecordTag::Present));
    appendU32(bytes, pattern.commit->event.ordinal());
    appendU32(bytes, pattern.commit->transition.ordinal());
  } else {
    appendU32(bytes, static_cast<std::uint32_t>(CommitRecordTag::Absent));
  }
  appendU32(bytes, pattern.timingAndProgress.ordinal());

  if (llvm::Error error = appendCount(bytes, pattern.claims.size(), "claims"))
    return error;
  for (const Claim &claim : pattern.claims) {
    appendU32(bytes, claim.state.ordinal());
    appendU32(bytes, claim.dimension.ordinal());
    appendU32(bytes, claim.amount.value());
  }

  appendU32(bytes, pattern.internalTransactionCount);
  for (std::uint32_t transaction = 0;
       transaction < pattern.internalTransactionCount; ++transaction) {
    const llvm::ArrayRef<ClaimKey> selection =
        contract.internalTransaction(key, transaction);
    if (llvm::Error error =
            appendCount(bytes, selection.size(), "transaction claims"))
      return error;
    for (ClaimKey claim : selection)
      appendU32(bytes, claim.ordinal());
  }
  return llvm::Error::success();
}

llvm::Expected<UsePatternDeclaration>
readUsePattern(RecordReader &reader, std::uint32_t patternOrdinal) {
  std::uint32_t requester = 0;
  std::uint32_t eligibility = 0;
  std::uint32_t acquire = 0;
  std::uint32_t release = 0;
  std::uint32_t commitTag = 0;
  std::uint32_t timing = 0;
  if (llvm::Error error = reader.readU32(requester, "pattern requester"))
    return error;
  if (llvm::Error error = reader.readU32(eligibility, "pattern eligibility"))
    return error;
  if (llvm::Error error = reader.readU32(acquire, "pattern acquire event"))
    return error;
  if (llvm::Error error = reader.readU32(release, "pattern release event"))
    return error;
  if (llvm::Error error = reader.readU32(commitTag, "commit variant"))
    return error;

  std::optional<CommitDeclaration> commit;
  switch (commitTag) {
  case static_cast<std::uint32_t>(CommitRecordTag::Absent):
    break;
  case static_cast<std::uint32_t>(CommitRecordTag::Present): {
    std::uint32_t event = 0;
    std::uint32_t transition = 0;
    if (llvm::Error error = reader.readU32(event, "commit event"))
      return error;
    if (llvm::Error error = reader.readU32(transition, "commit transition"))
      return error;
    commit =
        CommitDeclaration{EventKey(event), ResourceTransitionKey(transition)};
    break;
  }
  default:
    return invalidRecord("unknown commit variant");
  }

  if (llvm::Error error = reader.readU32(timing, "timing contract reference"))
    return error;

  std::uint32_t claimCount = 0;
  if (llvm::Error error = reader.readCount(claimCount, "claim count", 3))
    return error;
  std::vector<ClaimDeclaration> claims;
  claims.reserve(claimCount);
  for (std::uint32_t claim = 0; claim < claimCount; ++claim) {
    std::uint32_t state = 0;
    std::uint32_t dimension = 0;
    std::uint32_t amount = 0;
    if (llvm::Error error = reader.readU32(state, "claim state"))
      return error;
    if (llvm::Error error =
            reader.readU32(dimension, "claim capacity dimension"))
      return error;
    if (llvm::Error error = reader.readU32(amount, "claim amount"))
      return error;
    claims.push_back(ClaimDeclaration{
        ClaimKey(claim), StateKey(state), CapacityDimensionKey(dimension),
        CapacityUnits(amount)});
  }

  std::uint32_t transactionCount = 0;
  if (llvm::Error error =
          reader.readCount(transactionCount, "internal transaction count", 1))
    return error;
  std::vector<InternalTransactionDeclaration> transactions;
  transactions.reserve(transactionCount);
  for (std::uint32_t transaction = 0; transaction < transactionCount;
       ++transaction) {
    std::uint32_t transactionClaimCount = 0;
    if (llvm::Error error = reader.readCount(transactionClaimCount,
                                             "transaction claim count", 1))
      return error;
    std::vector<ClaimKey> selection;
    selection.reserve(transactionClaimCount);
    for (std::uint32_t entry = 0; entry < transactionClaimCount; ++entry) {
      std::uint32_t claim = 0;
      if (llvm::Error error =
              reader.readU32(claim, "transaction claim reference"))
        return error;
      selection.push_back(ClaimKey(claim));
    }
    transactions.push_back(
        InternalTransactionDeclaration{std::move(selection)});
  }

  return UsePatternDeclaration{UsePatternKey(patternOrdinal),
                               RequesterKey(requester),
                               EligibilityKey(eligibility),
                               EventKey(acquire),
                               EventKey(release),
                               commit,
                               TimingContractKey(timing),
                               std::move(claims),
                               std::move(transactions)};
}

llvm::Error readGrantPolicy(RecordReader &reader,
                            ResourceContractDeclaration &declaration) {
  std::uint32_t tag = 0;
  if (llvm::Error error = reader.readU32(tag, "grant policy variant"))
    return error;

  switch (tag) {
  case static_cast<std::uint32_t>(GrantPolicyRecordTag::Absent):
    return llvm::Error::success();
  case static_cast<std::uint32_t>(GrantPolicyRecordTag::FixedPriority):
  case static_cast<std::uint32_t>(GrantPolicyRecordTag::RoundRobin):
    break;
  default:
    return invalidRecord("unknown grant policy variant");
  }

  std::uint32_t requesterCount = 0;
  if (llvm::Error error =
          reader.readCount(requesterCount, "policy requester count", 1))
    return error;
  std::vector<RequesterKey> order;
  order.reserve(requesterCount);
  for (std::uint32_t position = 0; position < requesterCount; ++position) {
    std::uint32_t requester = 0;
    if (llvm::Error error =
            reader.readU32(requester, "policy requester reference"))
      return error;
    order.push_back(RequesterKey(requester));
  }

  if (tag == static_cast<std::uint32_t>(GrantPolicyRecordTag::FixedPriority)) {
    declaration.grantPolicy =
        GrantPolicyDeclaration(FixedPriorityDeclaration{std::move(order)});
    return llvm::Error::success();
  }

  std::uint32_t resetCursor = 0;
  if (llvm::Error error =
          reader.readU32(resetCursor, "round-robin reset cursor"))
    return error;
  declaration.grantPolicy = GrantPolicyDeclaration(
      RoundRobinDeclaration{std::move(order), RequesterKey(resetCursor)});
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>>
encodeResourceContractRecord(const ResourceContract &contract) {
  std::vector<std::uint8_t> bytes;

  appendU32(bytes, contract.stateCount());
  for (std::uint32_t state = 0; state < contract.stateCount(); ++state) {
    const llvm::ArrayRef<CapacityDimension> dimensions =
        contract.capacityDimensions(StateKey(state));
    if (llvm::Error error =
            appendCount(bytes, dimensions.size(), "capacity dimensions"))
      return std::move(error);
    for (const CapacityDimension &dimension : dimensions) {
      appendU32(bytes, dimension.capacity.value());
      appendU32(bytes, dimension.initialOccupancy.value());
    }
  }

  appendU32(bytes, contract.resourceTransitionCount());
  appendU32(bytes, contract.timingContractCount());
  for (std::uint32_t timing = 0; timing < contract.timingContractCount();
       ++timing) {
    const llvm::ArrayRef<std::uint32_t> eventRanks =
        contract.eventOrder(TimingContractKey(timing));
    if (llvm::Error error =
            appendCount(bytes, eventRanks.size(), "event ranks"))
      return std::move(error);
    for (std::uint32_t rank : eventRanks)
      appendU32(bytes, rank);
  }

  appendU32(bytes, contract.usePatternCount());
  for (std::uint32_t pattern = 0; pattern < contract.usePatternCount();
       ++pattern)
    if (llvm::Error error =
            appendUsePattern(bytes, contract, UsePatternKey(pattern)))
      return std::move(error);

  appendU32(bytes, contract.requesterCount());
  appendU32(bytes, contract.eligibilityCount());
  appendU32(bytes, contract.eventCount());

  const std::optional<GrantPolicyView> policy = contract.grantPolicy();
  if (!policy) {
    appendU32(bytes, static_cast<std::uint32_t>(GrantPolicyRecordTag::Absent));
    return bytes;
  }

  if (const auto *fixed = std::get_if<FixedPriorityView>(&*policy)) {
    appendU32(bytes,
              static_cast<std::uint32_t>(GrantPolicyRecordTag::FixedPriority));
    if (llvm::Error error = appendCount(bytes, fixed->requesterOrder().size(),
                                        "fixed-priority requester order"))
      return std::move(error);
    for (RequesterKey requester : fixed->requesterOrder())
      appendU32(bytes, requester.ordinal());
    return bytes;
  }

  const RoundRobinView &roundRobin = std::get<RoundRobinView>(*policy);
  appendU32(bytes,
            static_cast<std::uint32_t>(GrantPolicyRecordTag::RoundRobin));
  if (llvm::Error error = appendCount(bytes, roundRobin.requesterCycle().size(),
                                      "round-robin requester cycle"))
    return std::move(error);
  for (RequesterKey requester : roundRobin.requesterCycle())
    appendU32(bytes, requester.ordinal());
  appendU32(bytes, roundRobin.resetCursor().ordinal());
  return bytes;
}

llvm::Expected<ResourceContract>
decodeResourceContractRecord(llvm::ArrayRef<std::uint8_t> bytes) {
  RecordReader reader(bytes);
  ResourceContractDeclaration declaration;

  std::uint32_t stateCount = 0;
  if (llvm::Error error = reader.readCount(stateCount, "state count", 1))
    return std::move(error);
  declaration.states.reserve(stateCount);
  for (std::uint32_t state = 0; state < stateCount; ++state) {
    std::uint32_t dimensionCount = 0;
    if (llvm::Error error =
            reader.readCount(dimensionCount, "capacity dimension count", 2))
      return std::move(error);
    std::vector<CapacityDimensionDeclaration> dimensions;
    dimensions.reserve(dimensionCount);
    for (std::uint32_t dimension = 0; dimension < dimensionCount; ++dimension) {
      std::uint32_t capacity = 0;
      std::uint32_t initialOccupancy = 0;
      if (llvm::Error error = reader.readU32(capacity, "capacity"))
        return std::move(error);
      if (llvm::Error error =
              reader.readU32(initialOccupancy, "initial occupancy"))
        return std::move(error);
      dimensions.push_back(CapacityDimensionDeclaration{
          CapacityDimensionKey(dimension), CapacityUnits(capacity),
          CapacityUnits(initialOccupancy)});
    }
    declaration.states.push_back(
        ResourceStateDeclaration{StateKey(state), std::move(dimensions)});
  }

  std::uint32_t transitionCount = 0;
  if (llvm::Error error =
          reader.readU32(transitionCount, "resource transition count"))
    return std::move(error);
  declaration.resourceTransitions.reserve(transitionCount);
  for (std::uint32_t transition = 0; transition < transitionCount; ++transition)
    declaration.resourceTransitions.push_back(
        ResourceTransitionKey(transition));

  std::uint32_t timingCount = 0;
  if (llvm::Error error =
          reader.readCount(timingCount, "timing contract count", 1))
    return std::move(error);
  declaration.timingContracts.reserve(timingCount);
  for (std::uint32_t timing = 0; timing < timingCount; ++timing) {
    std::uint32_t rankCount = 0;
    if (llvm::Error error = reader.readCount(rankCount, "event rank count", 1))
      return std::move(error);
    std::vector<std::uint32_t> ranks;
    ranks.reserve(rankCount);
    for (std::uint32_t event = 0; event < rankCount; ++event) {
      std::uint32_t rank = 0;
      if (llvm::Error error = reader.readU32(rank, "event rank"))
        return std::move(error);
      ranks.push_back(rank);
    }
    declaration.timingContracts.push_back(
        TimingContractDeclaration{TimingContractKey(timing), std::move(ranks)});
  }

  std::uint32_t patternCount = 0;
  if (llvm::Error error =
          reader.readCount(patternCount, "use pattern count", 8))
    return std::move(error);
  declaration.usePatterns.reserve(patternCount);
  for (std::uint32_t pattern = 0; pattern < patternCount; ++pattern) {
    llvm::Expected<UsePatternDeclaration> decoded =
        readUsePattern(reader, pattern);
    if (!decoded)
      return decoded.takeError();
    declaration.usePatterns.push_back(std::move(*decoded));
  }

  std::uint32_t requesterCount = 0;
  if (llvm::Error error = reader.readU32(requesterCount, "requester count"))
    return std::move(error);
  declaration.requesters.reserve(requesterCount);
  for (std::uint32_t requester = 0; requester < requesterCount; ++requester)
    declaration.requesters.push_back(RequesterKey(requester));
  if (llvm::Error error =
          reader.readU32(declaration.eligibilityCount, "eligibility count"))
    return std::move(error);
  if (llvm::Error error = reader.readU32(declaration.eventCount, "event count"))
    return std::move(error);
  if (llvm::Error error = readGrantPolicy(reader, declaration))
    return std::move(error);
  if (llvm::Error error = reader.finish())
    return std::move(error);

  llvm::Expected<ResourceContract> contract =
      ResourceContract::create(declaration);
  if (!contract)
    return contract.takeError();
  llvm::Expected<std::vector<std::uint8_t>> canonical =
      encodeResourceContractRecord(*contract);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalidRecord("noncanonical encoding");
  return std::move(*contract);
}

} // namespace fabric
