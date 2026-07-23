#include "Simulator/MemoryAtomicOrder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>

using namespace loom::sim;

namespace {

using Decision = AtomicCompareExchangeDecision;
using Kind = MemoryAtomicOrderError::Kind;

// One atomic object plus the three keys that differ from it in exactly one
// field. A repeated active per-lane address rebuilds the same key value
// because the lane ordinal has no representation.
constexpr AtomicObjectKey kBase = {7, 64, 4};
constexpr AtomicObjectKey kOtherRoot = {8, 64, 4};
constexpr AtomicObjectKey kOtherOffset = {7, 68, 4};
constexpr AtomicObjectKey kOtherSize = {7, 64, 8};
constexpr AtomicObjectKey kNeverInitialized = {9, 0, 4};
constexpr AtomicObjectKey kZeroSize = {7, 64, 0};

// No engine allocates this many ids, so it names no version.
constexpr std::uint64_t kAbsentId = std::numeric_limits<std::uint64_t>::max();

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "MemoryAtomicOrderTest: " << message << "\n";
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
T takeOptional(std::optional<T> value, llvm::StringRef message) {
  if (!value)
    fail(message);
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::Expected<T> value, Kind kind,
                    llvm::StringRef message) {
  require(!value, message);
  std::optional<Kind> rejected;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const MemoryAtomicOrderError &error) { rejected = error.kind(); },
      [&](const llvm::ErrorInfoBase &) {});
  require(rejected && *rejected == kind, message);
}

void expectOrder(const MemoryAtomicOrder &order, const AtomicObjectKey &key,
                 llvm::ArrayRef<AtomicVersionId> expected,
                 llvm::StringRef message) {
  llvm::ArrayRef<AtomicVersionId> actual =
      takeOptional(order.modificationOrder(key), message);
  require(actual.size() == expected.size(), message);
  for (auto [recorded, wanted] : llvm::zip_equal(actual, expected))
    require(recorded == wanted, message);
}

AtomicVersionRecord versionOf(const MemoryAtomicOrder &order,
                              AtomicVersionId id, llvm::StringRef message) {
  return takeOptional(order.versionRecord(id), message);
}

AtomicReadRecord readOf(const MemoryAtomicOrder &order, AtomicReadId id,
                        llvm::StringRef message) {
  return takeOptional(order.readRecord(id), message);
}

void explicitInitialVersionHasNoPredecessor() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));

  expectOrder(order, kBase, {initial},
              "initialization did not create exactly one version");
  AtomicVersionRecord record =
      versionOf(order, initial, "the initial version has no record");
  require(record.id() == initial && record.key() == kBase,
          "the initial version record lost its identity");
  require(!record.predecessor(), "the initial version claims a predecessor");
}

void storeAppendsOneVersion() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId stored = takeExpected(order.atomicStore(kBase));

  expectOrder(order, kBase, {initial, stored},
              "store did not append exactly one version");
  require(versionOf(order, stored, "the stored version has no record")
                  .predecessor() == initial,
          "store did not append behind the previous tail");
}

void loadRecordsSelectionWithoutAppending() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId stored = takeExpected(order.atomicStore(kBase));

  // The engine never chooses reads-from, so a superseded version stays
  // selectable by the provider.
  AtomicReadId read = takeExpected(order.atomicLoad(kBase, initial));
  AtomicReadRecord record = readOf(order, read, "load recorded no read");
  require(record.id() == read && record.key() == kBase &&
              record.version() == initial,
          "load recorded the wrong reads-from relation");
  expectOrder(order, kBase, {initial, stored}, "load appended a version");
}

void rmwReadsTailAndAppendsTogether() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicRmwResult first = takeExpected(order.atomicRmw(kBase, initial));
  AtomicRmwResult second = takeExpected(order.atomicRmw(kBase, first.version));

  expectOrder(order, kBase, {initial, first.version, second.version},
              "two rmw actions did not extend one modification order");
  require(
      readOf(order, first.read, "the first rmw recorded no read").version() ==
              initial &&
          readOf(order, second.read, "the second rmw recorded no read")
                  .version() == first.version,
      "an rmw did not read its immediate predecessor");
  require(versionOf(order, second.version, "the second rmw appended no record")
                  .predecessor() == first.version,
          "an rmw appended behind a version it did not read");
}

void rmwRejectsStalePredecessor() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId stored = takeExpected(order.atomicStore(kBase));

  expectRejected(order.atomicRmw(kBase, initial), Kind::StaleVersion,
                 "rmw accepted a predecessor that is no longer the tail");
  expectOrder(order, kBase, {initial, stored},
              "a rejected rmw changed the modification order");

  AtomicRmwResult retried = takeExpected(order.atomicRmw(kBase, stored));
  expectOrder(order, kBase, {initial, stored, retried.version},
              "rmw did not accept the current tail after a stale attempt");
}

void compareExchangeSuccessAppendsBehindTail() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicCompareExchangeResult result =
      takeExpected(order.compareExchange(kBase, initial, Decision::Success));

  require(result.version.has_value(),
          "a successful compare-exchange appended no version");
  expectOrder(order, kBase, {initial, *result.version},
              "a successful compare-exchange did not append exactly one "
              "version");
  require(readOf(order, result.read, "compare-exchange recorded no read")
                  .version() == initial,
          "a successful compare-exchange did not read the tail it replaced");
  require(
      versionOf(order, *result.version, "the appended version has no record")
              .predecessor() == initial,
      "a successful compare-exchange appended behind the wrong version");
}

void compareExchangeSuccessRejectsStaleSelection() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId stored = takeExpected(order.atomicStore(kBase));

  expectRejected(order.compareExchange(kBase, initial, Decision::Success),
                 Kind::StaleVersion,
                 "a successful compare-exchange accepted a stale selection");
  expectOrder(order, kBase, {initial, stored},
              "a rejected compare-exchange changed the modification order");
}

void compareExchangeFailuresRecordOnlyOneRead() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId stored = takeExpected(order.atomicStore(kBase));

  // A failed compare-exchange is a load, so a superseded version stays
  // selectable.
  AtomicCompareExchangeResult compared = takeExpected(
      order.compareExchange(kBase, initial, Decision::ComparisonFailure));
  require(!compared.version, "a comparison failure appended a version");
  require(readOf(order, compared.read, "a comparison failure recorded no read")
                  .version() == initial,
          "a comparison failure recorded the wrong reads-from relation");

  // Spurious failure is the explicit weak-provider input: the same tail
  // selection that would succeed fails only because the caller said so.
  AtomicCompareExchangeResult spurious = takeExpected(
      order.compareExchange(kBase, stored, Decision::SpuriousFailure));
  require(!spurious.version, "a spurious failure appended a version");
  require(readOf(order, spurious.read, "a spurious failure recorded no read")
                  .version() == stored,
          "a spurious failure recorded the wrong reads-from relation");
  require(spurious.read != compared.read,
          "two failed compare-exchange actions share one read record");
  expectOrder(order, kBase, {initial, stored},
              "a failed compare-exchange changed the modification order");
}

void repeatedKeySharesOneObject() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));

  // A second active lane at the same address rebuilds the same key value.
  const AtomicObjectKey repeated = {
      kBase.logicalRootId, kBase.canonicalByteOffset, kBase.accessByteSize};
  require(repeated == kBase, "an equal key triple is not the same key");
  expectRejected(order.initializeObject(repeated),
                 Kind::DuplicateInitialization,
                 "a repeated key initialized a second object");

  AtomicRmwResult lane = takeExpected(order.atomicRmw(repeated, initial));
  expectOrder(order, kBase, {initial, lane.version},
              "a repeated key did not share one modification order");
  require(readOf(order, lane.read, "the repeated key recorded no read").key() ==
              kBase,
          "a repeated key recorded a read against another object");
}

void keyFieldsDistinguishObjects() {
  MemoryAtomicOrder order;
  AtomicVersionId base = takeExpected(order.initializeObject(kBase));
  AtomicVersionId root = takeExpected(order.initializeObject(kOtherRoot));
  AtomicVersionId offset = takeExpected(order.initializeObject(kOtherOffset));
  AtomicVersionId size = takeExpected(order.initializeObject(kOtherSize));
  AtomicVersionId appended = takeExpected(order.atomicStore(kOtherOffset));

  expectOrder(order, kBase, {base},
              "another root, offset, or size extended the base object");
  expectOrder(order, kOtherRoot, {root},
              "a differing logical root did not identify its own object");
  expectOrder(order, kOtherOffset, {offset, appended},
              "a differing byte offset did not identify its own object");
  expectOrder(order, kOtherSize, {size},
              "a differing access byte size did not identify its own object");
}

void invalidAndUnknownSelectionsAreRejected() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId foreign = takeExpected(order.initializeObject(kOtherRoot));

  expectRejected(order.initializeObject(kZeroSize), Kind::InvalidKey,
                 "a zero access byte size initialized an object");
  expectRejected(order.atomicLoad(kZeroSize, initial), Kind::InvalidKey,
                 "a zero access byte size passed key validation");
  expectRejected(order.atomicStore(kNeverInitialized), Kind::UnknownObject,
                 "store accepted an object with no explicit initial version");
  expectRejected(order.atomicLoad(kBase, AtomicVersionId(kAbsentId)),
                 Kind::UnknownVersion, "load accepted an unknown version");
  expectRejected(order.atomicLoad(kBase, foreign), Kind::ForeignVersion,
                 "load accepted a version of another object");
}

struct Fixture {
  AtomicVersionId baseInitial;
  AtomicVersionId baseTail;
  AtomicVersionId foreignInitial;
};

Fixture buildFixture(MemoryAtomicOrder &order) {
  AtomicVersionId baseInitial = takeExpected(order.initializeObject(kBase));
  AtomicVersionId baseTail = takeExpected(order.atomicStore(kBase));
  AtomicVersionId foreignInitial =
      takeExpected(order.initializeObject(kOtherRoot));
  takeExpected(order.atomicLoad(kBase, baseInitial));
  return {baseInitial, baseTail, foreignInitial};
}

// Two engines run the identical legal sequence and only one of them also sees
// every rejection. Equal ids afterwards prove a rejected action consumes no
// id, and equal orders prove it mutates no relation.
void rejectedActionsConsumeNoIdOrState() {
  MemoryAtomicOrder reference;
  Fixture referenceIds = buildFixture(reference);
  MemoryAtomicOrder probed;
  Fixture probedIds = buildFixture(probed);

  expectRejected(probed.initializeObject(kBase), Kind::DuplicateInitialization,
                 "a duplicate initialization was accepted");
  expectRejected(probed.initializeObject(kZeroSize), Kind::InvalidKey,
                 "an invalid key was accepted");
  expectRejected(probed.atomicStore(kNeverInitialized), Kind::UnknownObject,
                 "an uninitialized object was accepted");
  expectRejected(probed.atomicLoad(kBase, AtomicVersionId(kAbsentId)),
                 Kind::UnknownVersion, "an unknown version was accepted");
  expectRejected(probed.atomicLoad(kBase, probedIds.foreignInitial),
                 Kind::ForeignVersion, "a foreign version was accepted");
  expectRejected(probed.atomicRmw(kBase, probedIds.baseInitial),
                 Kind::StaleVersion, "a stale rmw predecessor was accepted");
  expectRejected(
      probed.compareExchange(kBase, probedIds.baseInitial, Decision::Success),
      Kind::StaleVersion, "a stale compare-exchange selection was accepted");

  AtomicRmwResult referenceRmw =
      takeExpected(reference.atomicRmw(kBase, referenceIds.baseTail));
  AtomicRmwResult probedRmw =
      takeExpected(probed.atomicRmw(kBase, probedIds.baseTail));
  require(referenceRmw.read == probedRmw.read,
          "a rejected action consumed a read id");
  require(referenceRmw.version == probedRmw.version,
          "a rejected action consumed a version id");
  expectOrder(probed, kBase,
              {probedIds.baseInitial, probedIds.baseTail, probedRmw.version},
              "a rejected action changed a modification order");
  expectOrder(probed, kOtherRoot, {probedIds.foreignInitial},
              "a rejected action changed another object");
  require(!probed.modificationOrder(kNeverInitialized) &&
              !probed.modificationOrder(kZeroSize),
          "a rejected action created an object");
}

} // namespace

int main() {
  explicitInitialVersionHasNoPredecessor();
  storeAppendsOneVersion();
  loadRecordsSelectionWithoutAppending();
  rmwReadsTailAndAppendsTogether();
  rmwRejectsStalePredecessor();
  compareExchangeSuccessAppendsBehindTail();
  compareExchangeSuccessRejectsStaleSelection();
  compareExchangeFailuresRecordOnlyOneRead();
  repeatedKeySharesOneObject();
  keyFieldsDistinguishObjects();
  invalidAndUnknownSelectionsAreRejected();
  rejectedActionsConsumeNoIdOrState();
  return 0;
}
