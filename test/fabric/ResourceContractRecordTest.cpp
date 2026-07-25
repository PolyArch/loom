#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

using namespace fabric;

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
    fail(test, "accepted a malformed persistent record");
  llvm::consumeError(value.takeError());
}

ResourceContractDeclaration declaration() {
  ResourceContractDeclaration result;
  result.states = {ResourceStateDeclaration{
      StateKey(0),
      {CapacityDimensionDeclaration{CapacityDimensionKey(0), CapacityUnits(4),
                                    CapacityUnits(1)},
       CapacityDimensionDeclaration{CapacityDimensionKey(1), CapacityUnits(2),
                                    CapacityUnits(0)}}}};
  result.resourceTransitions = {ResourceTransitionKey(0)};
  result.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0, 1, 2}}};
  result.requesters = {RequesterKey(0), RequesterKey(1)};
  result.eligibilityCount = 2;
  result.eventCount = 3;
  result.usePatterns = {
      UsePatternDeclaration{
          UsePatternKey(0),
          RequesterKey(0),
          EligibilityKey(0),
          EventKey(0),
          EventKey(2),
          CommitDeclaration{EventKey(1), ResourceTransitionKey(0)},
          TimingContractKey(0),
          {ClaimDeclaration{ClaimKey(0), StateKey(0), CapacityDimensionKey(0),
                            CapacityUnits(2)},
           ClaimDeclaration{ClaimKey(1), StateKey(0), CapacityDimensionKey(1),
                            CapacityUnits(1)}},
          {InternalTransactionDeclaration{{ClaimKey(0)}},
           InternalTransactionDeclaration{{ClaimKey(0), ClaimKey(1)}}}},
      UsePatternDeclaration{
          UsePatternKey(1),
          RequesterKey(1),
          EligibilityKey(1),
          EventKey(0),
          EventKey(2),
          std::nullopt,
          TimingContractKey(0),
          {ClaimDeclaration{ClaimKey(0), StateKey(0), CapacityDimensionKey(0),
                            CapacityUnits(1)}},
          {InternalTransactionDeclaration{{ClaimKey(0)}}}},
  };
  result.grantPolicy = GrantPolicyDeclaration(RoundRobinDeclaration{
      {RequesterKey(1), RequesterKey(0)}, RequesterKey(1)});
  return result;
}

class RecordCursor {
public:
  explicit RecordCursor(llvm::ArrayRef<std::uint8_t> bytes) : bytes(bytes) {}

  std::size_t position() const { return offset; }

  std::uint32_t u32(llvm::StringRef field) {
    require("record walk", offset <= bytes.size() && bytes.size() - offset >= 4,
            "truncated " + field);
    const std::uint32_t value =
        (static_cast<std::uint32_t>(bytes[offset]) << 24) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
        static_cast<std::uint32_t>(bytes[offset + 3]);
    offset += 4;
    return value;
  }

  void skipWords(std::uint64_t count, llvm::StringRef field) {
    require("record walk",
            offset <= bytes.size() && count <= (bytes.size() - offset) / 4,
            "truncated " + field);
    offset += static_cast<std::size_t>(count) * 4;
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes;
  std::size_t offset = 0;
};

struct RecordLocations {
  std::vector<std::size_t> patternRequesters;
  std::vector<std::size_t> policyRequesters;
};

RecordLocations locateReferences(llvm::ArrayRef<std::uint8_t> bytes) {
  RecordCursor cursor(bytes);
  RecordLocations locations;

  const std::uint32_t stateCount = cursor.u32("state count");
  for (std::uint32_t state = 0; state < stateCount; ++state) {
    const std::uint32_t dimensionCount = cursor.u32("capacity dimension count");
    cursor.skipWords(static_cast<std::uint64_t>(dimensionCount) * 2,
                     "capacity dimensions");
  }

  cursor.u32("resource transition count");
  const std::uint32_t timingCount = cursor.u32("timing contract count");
  for (std::uint32_t timing = 0; timing < timingCount; ++timing)
    cursor.skipWords(cursor.u32("event rank count"), "event ranks");

  const std::uint32_t patternCount = cursor.u32("use pattern count");
  for (std::uint32_t pattern = 0; pattern < patternCount; ++pattern) {
    locations.patternRequesters.push_back(cursor.position());
    cursor.u32("pattern requester");
    cursor.u32("pattern eligibility");
    cursor.u32("pattern acquire event");
    cursor.u32("pattern release event");
    const std::uint32_t commit = cursor.u32("commit variant");
    require("record walk",
            commit == static_cast<std::uint32_t>(CommitRecordTag::Absent) ||
                commit == static_cast<std::uint32_t>(CommitRecordTag::Present),
            "unknown commit variant");
    if (commit == static_cast<std::uint32_t>(CommitRecordTag::Present))
      cursor.skipWords(2, "commit record");
    cursor.u32("timing contract reference");

    const std::uint32_t claimCount = cursor.u32("claim count");
    cursor.skipWords(static_cast<std::uint64_t>(claimCount) * 3, "claims");
    const std::uint32_t transactionCount =
        cursor.u32("internal transaction count");
    for (std::uint32_t transaction = 0; transaction < transactionCount;
         ++transaction)
      cursor.skipWords(cursor.u32("transaction claim count"),
                       "transaction claims");
  }

  cursor.u32("requester count");
  cursor.u32("eligibility count");
  cursor.u32("event count");
  const std::uint32_t policy = cursor.u32("grant policy variant");
  require("record walk",
          policy == static_cast<std::uint32_t>(GrantPolicyRecordTag::Absent) ||
              policy == static_cast<std::uint32_t>(
                            GrantPolicyRecordTag::FixedPriority) ||
              policy ==
                  static_cast<std::uint32_t>(GrantPolicyRecordTag::RoundRobin),
          "unknown grant policy variant");
  if (policy != static_cast<std::uint32_t>(GrantPolicyRecordTag::Absent)) {
    const std::uint32_t requesterCount = cursor.u32("policy requester count");
    for (std::uint32_t requester = 0; requester < requesterCount; ++requester) {
      locations.policyRequesters.push_back(cursor.position());
      cursor.u32("policy requester");
    }
    if (policy == static_cast<std::uint32_t>(GrantPolicyRecordTag::RoundRobin))
      cursor.u32("round-robin reset cursor");
  }

  require("record walk", cursor.position() == bytes.size(),
          "record walk did not consume the complete framing");
  return locations;
}

void writeU32(std::vector<std::uint8_t> &bytes, std::size_t offset,
              std::uint32_t value) {
  require("record mutation",
          offset <= bytes.size() && bytes.size() - offset >= 4,
          "mutation is outside the record");
  bytes[offset] = static_cast<std::uint8_t>(value >> 24);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 16);
  bytes[offset + 2] = static_cast<std::uint8_t>(value >> 8);
  bytes[offset + 3] = static_cast<std::uint8_t>(value);
}

ResourceContract roundTrip(const ResourceContractDeclaration &declared) {
  ResourceContract original =
      take("resource declaration", ResourceContract::create(declared));
  std::vector<std::uint8_t> encoded =
      take("resource encoding", encodeResourceContractRecord(original));
  ResourceContract decoded =
      take("resource decoding", decodeResourceContractRecord(encoded));
  require("resource roundtrip",
          take("resource reencoding", encodeResourceContractRecord(decoded)) ==
              encoded,
          "canonical bytes changed after strict import");
  return decoded;
}

void checkCompleteRoundTrip() {
  ResourceContract decoded = roundTrip(declaration());
  llvm::ArrayRef<CapacityDimension> dimensions =
      decoded.capacityDimensions(StateKey(0));
  require("resource state",
          decoded.stateCount() == 1 && dimensions.size() == 2 &&
              dimensions[0].capacity == CapacityUnits(4) &&
              dimensions[0].initialOccupancy == CapacityUnits(1) &&
              dimensions[1].capacity == CapacityUnits(2) &&
              dimensions[1].initialOccupancy == CapacityUnits(0),
          "state capacity or initial occupancy changed");
  require("resource inventories",
          decoded.resourceTransitionCount() == 1 &&
              decoded.timingContractCount() == 1 &&
              decoded.usePatternCount() == 2 && decoded.requesterCount() == 2 &&
              decoded.eligibilityCount() == 2 && decoded.eventCount() == 3,
          "closed owner inventory changed");
  require("resource timing",
          decoded.eventOrder(TimingContractKey(0)) ==
              llvm::ArrayRef<std::uint32_t>({0, 1, 2}),
          "event ranks changed");

  UsePattern pattern = decoded.usePattern(UsePatternKey(0));
  require(
      "resource pattern",
      pattern.requester == RequesterKey(0) &&
          pattern.eligibility == EligibilityKey(0) &&
          pattern.acquire == EventKey(0) && pattern.release == EventKey(2) &&
          pattern.commit && pattern.commit->event == EventKey(1) &&
          pattern.commit->transition == ResourceTransitionKey(0) &&
          pattern.timingAndProgress == TimingContractKey(0) &&
          pattern.claims.size() == 2 && pattern.internalTransactionCount == 2,
      "atomic use pattern changed");
  require("resource claims",
          pattern.claims[0].state == StateKey(0) &&
              pattern.claims[0].dimension == CapacityDimensionKey(0) &&
              pattern.claims[0].amount == CapacityUnits(2) &&
              pattern.claims[1].dimension == CapacityDimensionKey(1),
          "claim envelope changed");
  require("resource internal transactions",
          decoded.internalTransaction(UsePatternKey(0), 0) ==
                  llvm::ArrayRef<ClaimKey>({ClaimKey(0)}) &&
              decoded.internalTransaction(UsePatternKey(0), 1) ==
                  llvm::ArrayRef<ClaimKey>({ClaimKey(0), ClaimKey(1)}),
          "internal transaction selections changed");

  std::optional<GrantPolicyView> policy = decoded.grantPolicy();
  require("round robin policy",
          policy && std::holds_alternative<RoundRobinView>(*policy),
          "round-robin policy changed variant");
  const RoundRobinView &roundRobin = std::get<RoundRobinView>(*policy);
  require("round robin policy",
          roundRobin.requesterCycle() ==
                  llvm::ArrayRef<RequesterKey>(
                      {RequesterKey(1), RequesterKey(0)}) &&
              roundRobin.resetCursor() == RequesterKey(1),
          "round-robin requester order changed");

  ResourceContractDeclaration fixed = declaration();
  fixed.grantPolicy = GrantPolicyDeclaration(
      FixedPriorityDeclaration{{RequesterKey(1), RequesterKey(0)}});
  ResourceContract fixedDecoded = roundTrip(fixed);
  std::optional<GrantPolicyView> fixedPolicy = fixedDecoded.grantPolicy();
  require(
      "fixed policy",
      fixedPolicy && std::holds_alternative<FixedPriorityView>(*fixedPolicy) &&
          std::get<FixedPriorityView>(*fixedPolicy).requesterOrder() ==
              llvm::ArrayRef<RequesterKey>({RequesterKey(1), RequesterKey(0)}),
      "fixed-priority requester order changed");
}

void checkMalformedRecords() {
  ResourceContract contract =
      take("resource declaration", ResourceContract::create(declaration()));
  std::vector<std::uint8_t> encoded =
      take("resource encoding", encodeResourceContractRecord(contract));
  RecordLocations locations = locateReferences(encoded);
  require("record locations",
          locations.patternRequesters.size() == 2 &&
              locations.policyRequesters.size() == 2,
          "fixture does not expose the expected semantic references");

  std::vector<std::uint8_t> malformedRequester = encoded;
  writeU32(malformedRequester, locations.patternRequesters[1],
           contract.requesterCount());
  expectRejected<ResourceContract>(
      "malformed requester", decodeResourceContractRecord(malformedRequester));

  std::vector<std::uint8_t> malformedPolicy = encoded;
  writeU32(malformedPolicy, locations.policyRequesters[1], 1);
  expectRejected<ResourceContract>(
      "malformed policy", decodeResourceContractRecord(malformedPolicy));

  std::vector<std::uint8_t> truncated = encoded;
  truncated.pop_back();
  expectRejected<ResourceContract>("truncated framing",
                                   decodeResourceContractRecord(truncated));
  std::vector<std::uint8_t> trailing = encoded;
  trailing.push_back(0);
  expectRejected<ResourceContract>("trailing framing",
                                   decodeResourceContractRecord(trailing));
}

} // namespace

int main() {
  checkCompleteRoundTrip();
  checkMalformedRecords();
  return EXIT_SUCCESS;
}
