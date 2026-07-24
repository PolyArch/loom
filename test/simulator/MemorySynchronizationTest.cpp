#include "Simulator/MemorySynchronization.h"
#include "Simulator/MemoryAtomicOrder.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <utility>

using namespace loom::sim;

namespace {

using Kind = MemorySynchronizationError::Kind;

constexpr AtomicObjectKey kFlag = {1, 0, 4};
constexpr AtomicObjectKey kGate = {2, 0, 4};

// Two resolved synchronization domains. Their values carry no meaning beyond
// identity: the engine compares them and never interprets a scope.
constexpr SyncDomainId kHome(0);
constexpr SyncDomainId kAway(1);

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "MemorySynchronizationTest: " << message << "\n";
  std::exit(1);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void accept(llvm::Error error, const llvm::Twine &message) {
  if (error)
    fail(message + ": " + llvm::toString(std::move(error)));
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectRejected(llvm::Error error, Kind kind, const llvm::Twine &message) {
  if (!error)
    fail(message);
  std::optional<Kind> rejected;
  llvm::handleAllErrors(
      std::move(error),
      [&](const MemorySynchronizationError &failure) {
        rejected = failure.kind();
      },
      [&](const llvm::ErrorInfoBase &) {});
  require(rejected && *rejected == kind, message);
}

template <typename T>
void expectRejected(llvm::Expected<T> value, Kind kind,
                    const llvm::Twine &message) {
  if (!value)
    return expectRejected(value.takeError(), kind, message);
  fail(message);
}

void expectEffects(llvm::ArrayRef<SyncEffectId> actual,
                   std::initializer_list<SyncEffectId> expected,
                   const llvm::Twine &message) {
  require(actual.size() == expected.size(), message);
  for (auto [recorded, wanted] : llvm::zip_equal(actual, expected))
    require(recorded == wanted, message);
}

// A release store publishes through its own write, and an acquire load imports
// that publication only when its recorded reads-from selects that exact
// version. The imported summary reaches effects sequenced after the acquire
// through happens-before, which composes sequenced-before with
// synchronizes-with.
void releaseAndAcquireImportOnlyThroughTheSelectedVersion() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicReadId selects = takeExpected(order.atomicLoad(kFlag, published));
  AtomicReadId misses = takeExpected(order.atomicLoad(kFlag, initial));

  auto build = [&](AtomicReadId read) {
    MemorySynchronization sync(order);
    SyncEffectId data = sync.declareEffect();
    SyncEffectId release = sync.declareEffect();
    SyncEffectId acquire = sync.declareEffect();
    SyncEffectId use = sync.declareEffect();
    accept(sync.sequencedBefore(data, release), "publisher strand");
    accept(sync.sequencedBefore(acquire, use), "consumer strand");
    accept(sync.registerWrite(release, kHome, published), "release write");
    accept(sync.declareOperationRole(release, SyncRoleKind::Release),
           "release role");
    accept(sync.registerRead(acquire, kHome, read), "acquire read");
    accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
           "acquire role");
    return std::tuple{std::move(sync), data, release, acquire, use};
  };

  auto [synced, data, release, acquire, use] = build(selects);
  require(synced.synchronizesWith(release, acquire),
          "an acquire reading the release version did not synchronize");
  expectEffects(takeExpected(synced.publishedOrigins(published)), {release},
                "the release version published the wrong origins");
  expectEffects(takeExpected(synced.importedVisibility(acquire)),
                {data, release},
                "the acquire imported the wrong visibility summary");
  require(synced.happensBefore(data, use),
          "happens-before did not compose sequenced-before with "
          "synchronizes-with");

  auto [stale, staleData, staleRelease, staleAcquire, staleUse] = build(misses);
  require(!stale.synchronizesWith(staleRelease, staleAcquire),
          "an acquire reading another version synchronized");
  expectEffects(takeExpected(stale.importedVisibility(staleAcquire)), {},
                "an acquire reading another version imported a summary");
  require(!stale.happensBefore(staleData, staleUse),
          "an unsynchronized acquire ordered a later effect");
}

// A release operation publishes through its own write only. A later unrelated
// atomic write in the same strand is not a fence carrier for it.
void releaseOperationDoesNotHookLaterWrites() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId unrelated = takeExpected(order.atomicStore(kFlag));
  AtomicReadId read = takeExpected(order.atomicLoad(kFlag, unrelated));

  MemorySynchronization sync(order);
  SyncEffectId release = sync.declareEffect();
  SyncEffectId later = sync.declareEffect();
  SyncEffectId acquire = sync.declareEffect();
  accept(sync.sequencedBefore(release, later), "publisher strand");
  accept(sync.registerWrite(release, kHome, published), "release write");
  accept(sync.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(sync.registerWrite(later, kHome, unrelated), "relaxed write");
  accept(sync.registerRead(acquire, kHome, read), "acquire read");
  accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");

  expectEffects(takeExpected(sync.publishedOrigins(unrelated)), {},
                "a release operation published through a later write");
  require(!sync.synchronizesWith(release, acquire),
          "a release operation acted as a release fence");
}

// An acquire operation imports through its own read only. An earlier unrelated
// relaxed read in the same strand is not a fence carrier for it.
void acquireOperationDoesNotHookEarlierReads() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  takeExpected(order.initializeObject(kGate));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId other = takeExpected(order.atomicStore(kGate));
  AtomicReadId earlier = takeExpected(order.atomicLoad(kFlag, published));
  AtomicReadId own = takeExpected(order.atomicLoad(kGate, other));

  MemorySynchronization sync(order);
  SyncEffectId release = sync.declareEffect();
  SyncEffectId relaxed = sync.declareEffect();
  SyncEffectId acquire = sync.declareEffect();
  accept(sync.registerWrite(release, kHome, published), "release write");
  accept(sync.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(sync.registerRead(relaxed, kHome, earlier), "relaxed read");
  accept(sync.sequencedBefore(relaxed, acquire), "consumer strand");
  accept(sync.registerRead(acquire, kHome, own), "acquire read");
  accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");

  require(!sync.synchronizesWith(release, acquire),
          "an acquire operation acted as an acquire fence");
  expectEffects(takeExpected(sync.importedVisibility(acquire)), {},
                "an acquire operation imported through an earlier read");
}

// Synchronization requires one domain identity across the release origin, the
// publishing carrier, the reading carrier, and the acquire. A mismatch and an
// absent reads-from are legal and simply create no synchronization.
void domainIdentityAndReadsFromAreRequired() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  takeExpected(order.initializeObject(kGate));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId elsewhere = takeExpected(order.atomicStore(kGate));
  AtomicReadId reads = takeExpected(order.atomicLoad(kFlag, published));
  AtomicReadId absent = takeExpected(order.atomicLoad(kGate, elsewhere));

  auto build = [&](SyncDomainId acquireDomain, AtomicReadId read) {
    MemorySynchronization sync(order);
    SyncEffectId release = sync.declareEffect();
    SyncEffectId acquire = sync.declareEffect();
    accept(sync.registerWrite(release, kHome, published), "release write");
    accept(sync.declareOperationRole(release, SyncRoleKind::Release),
           "release role");
    accept(sync.registerRead(acquire, acquireDomain, read), "acquire read");
    accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
           "acquire role");
    return std::tuple{std::move(sync), release, acquire};
  };

  auto [matched, matchedRelease, matchedAcquire] = build(kHome, reads);
  require(matched.synchronizesWith(matchedRelease, matchedAcquire),
          "one domain identity did not synchronize");

  auto [mismatched, otherRelease, otherAcquire] = build(kAway, reads);
  require(!mismatched.synchronizesWith(otherRelease, otherAcquire),
          "an acquire in another domain synchronized");

  auto [unread, unreadRelease, unreadAcquire] = build(kHome, absent);
  require(!unread.synchronizesWith(unreadRelease, unreadAcquire),
          "an acquire with no reads-from of the published version "
          "synchronized");
}

// A release fence publishes through a sequenced-after atomic write and an
// acquire fence imports through a sequenced-before atomic read. Both hooks and
// the reads-from relation between the carriers are required.
void fenceChainSynchronizesOnlyThroughItsCarriers() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicReadId read = takeExpected(order.atomicLoad(kFlag, published));

  auto build = [&](bool hookPublisher, SyncDomainId carrierDomain,
                   SyncDomainId releaseDomain, SyncDomainId acquireDomain) {
    MemorySynchronization sync(order);
    SyncEffectId data = sync.declareEffect();
    SyncEffectId releaseFence = sync.declareEffect();
    SyncEffectId write = sync.declareEffect();
    SyncEffectId load = sync.declareEffect();
    SyncEffectId acquireFence = sync.declareEffect();
    SyncEffectId use = sync.declareEffect();
    accept(sync.sequencedBefore(data, releaseFence), "publisher strand");
    if (hookPublisher)
      accept(sync.sequencedBefore(releaseFence, write), "release fence hook");
    accept(sync.declareFenceRole(releaseFence, SyncRoleKind::Release,
                                 releaseDomain),
           "release fence role");
    accept(sync.registerWrite(write, carrierDomain, published),
           "relaxed write");
    accept(sync.registerRead(load, carrierDomain, read), "relaxed read");
    accept(sync.sequencedBefore(load, acquireFence), "acquire fence hook");
    accept(sync.sequencedBefore(acquireFence, use), "consumer strand");
    accept(sync.declareFenceRole(acquireFence, SyncRoleKind::Acquire,
                                 acquireDomain),
           "acquire fence role");
    return std::tuple{std::move(sync), data, releaseFence, acquireFence, use};
  };

  auto [chained, data, releaseFence, acquireFence, use] =
      build(true, kHome, kHome, kHome);
  require(chained.synchronizesWith(releaseFence, acquireFence),
          "a complete fence chain did not synchronize");
  expectEffects(takeExpected(chained.importedVisibility(acquireFence)),
                {data, releaseFence},
                "an acquire fence imported the wrong summary");
  require(chained.happensBefore(data, use),
          "a fence chain did not order the surrounding effects");

  auto [unhooked, loneData, loneRelease, loneAcquire, loneUse] =
      build(false, kHome, kHome, kHome);
  require(!unhooked.synchronizesWith(loneRelease, loneAcquire),
          "a release fence with no sequenced-after write synchronized");
  require(!unhooked.happensBefore(loneData, loneUse),
          "an unhooked fence acted as a global barrier");
  expectEffects(takeExpected(unhooked.publishedOrigins(published)), {},
                "an unhooked release fence published through a write");

  // The acquire fence and the carriers agree, so only the release fence's own
  // domain can break this chain.
  auto [split, splitData, splitRelease, splitAcquire, splitUse] =
      build(true, kHome, kAway, kHome);
  require(!split.synchronizesWith(splitRelease, splitAcquire),
          "a release fence published into another domain");
  expectEffects(takeExpected(split.publishedOrigins(published)), {},
                "a release fence in another domain became a published origin");

  // The release side and the carriers agree, so only the acquire fence's own
  // domain can break this chain.
  auto [deaf, deafData, deafRelease, deafAcquire, deafUse] =
      build(true, kHome, kHome, kAway);
  require(!deaf.synchronizesWith(deafRelease, deafAcquire),
          "an acquire fence imported from another domain");
  expectEffects(takeExpected(deaf.publishedOrigins(published)), {deafRelease},
                "the release side stopped publishing in its own domain");

  // The carriers disagree with both fences, so neither hook can form.
  auto [apart, apartData, apartRelease, apartAcquire, apartUse] =
      build(true, kAway, kHome, kHome);
  require(!apart.synchronizesWith(apartRelease, apartAcquire),
          "a fence synchronized through a carrier in another domain");
}

// A read-modify-write carries the release sequence forward only when its read
// source is the exact predecessor of its appended version and every carrier hop
// resolves to one domain. A broken hop stays broken when a later hop returns to
// the origin domain.
void releaseSequenceCarriesOnlyThroughOneDomain() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kFlag));
  AtomicRmwResult first = takeExpected(order.atomicRmw(kFlag, initial));
  AtomicRmwResult second = takeExpected(order.atomicRmw(kFlag, first.version));
  AtomicReadId reads = takeExpected(order.atomicLoad(kFlag, second.version));

  auto build = [&](SyncDomainId middle, SyncDomainId last) {
    MemorySynchronization sync(order);
    SyncEffectId release = sync.declareEffect();
    SyncEffectId middleRmw = sync.declareEffect();
    SyncEffectId acquire = sync.declareEffect();
    accept(sync.registerWrite(release, kHome, first.version, first.read),
           "release rmw write");
    accept(sync.declareOperationRole(release, SyncRoleKind::Release),
           "release role");
    accept(sync.registerWrite(middleRmw, middle, second.version, second.read),
           "middle rmw write");
    accept(sync.registerRead(acquire, last, reads), "acquire read");
    accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
           "acquire role");
    return std::tuple{std::move(sync), release, middleRmw, acquire};
  };

  auto [carried, release, middleRmw, acquire] = build(kHome, kHome);
  require(carried.synchronizesWith(release, acquire),
          "a same-domain release sequence did not carry");
  expectEffects(takeExpected(carried.publishedOrigins(second.version)),
                {release},
                "a carried release sequence published the wrong origins");

  auto [broken, brokenRelease, brokenMiddle, brokenAcquire] =
      build(kAway, kHome);
  require(!broken.synchronizesWith(brokenRelease, brokenAcquire),
          "a release sequence carried across a domain break");
  expectEffects(takeExpected(broken.publishedOrigins(second.version)), {},
                "a broken release sequence still published an origin");

  // An acq_rel read-modify-write imports through its own read and republishes
  // through its own appended version.
  MemorySynchronization sync(order);
  SyncEffectId releaseStore = sync.declareEffect();
  SyncEffectId updater = sync.declareEffect();
  SyncEffectId acquireLoad = sync.declareEffect();
  accept(sync.registerWrite(releaseStore, kHome, first.version, first.read),
         "release rmw write");
  accept(sync.declareOperationRole(releaseStore, SyncRoleKind::Release),
         "release role");
  accept(sync.registerWrite(updater, kHome, second.version, second.read),
         "acq_rel rmw write");
  accept(sync.declareOperationRole(updater, SyncRoleKind::AcqRel),
         "acq_rel role");
  accept(sync.registerRead(acquireLoad, kHome, reads), "acquire read");
  accept(sync.declareOperationRole(acquireLoad, SyncRoleKind::Acquire),
         "acquire role");
  require(sync.synchronizesWith(releaseStore, updater),
          "an acq_rel update did not import from the release it read");
  require(sync.synchronizesWith(updater, acquireLoad),
          "an acq_rel update did not publish through its own version");
}

// Visibility imported by an acquire is part of the summary a later release in
// the same strand publishes, so a second acquire observes the first strand's
// effect.
void importedVisibilityReachesALaterReleaseSummary() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  takeExpected(order.initializeObject(kGate));
  AtomicVersionId flagged = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId gated = takeExpected(order.atomicStore(kGate));
  AtomicReadId readsFlag = takeExpected(order.atomicLoad(kFlag, flagged));
  AtomicReadId readsGate = takeExpected(order.atomicLoad(kGate, gated));

  MemorySynchronization sync(order);
  SyncEffectId data = sync.declareEffect();
  SyncEffectId firstRelease = sync.declareEffect();
  SyncEffectId firstAcquire = sync.declareEffect();
  SyncEffectId secondRelease = sync.declareEffect();
  SyncEffectId secondAcquire = sync.declareEffect();
  accept(sync.sequencedBefore(data, firstRelease), "producer strand");
  accept(sync.sequencedBefore(firstAcquire, secondRelease), "relay strand");
  accept(sync.registerWrite(firstRelease, kHome, flagged), "first release");
  accept(sync.declareOperationRole(firstRelease, SyncRoleKind::Release),
         "first release role");
  accept(sync.registerRead(firstAcquire, kHome, readsFlag), "relay read");
  accept(sync.declareOperationRole(firstAcquire, SyncRoleKind::Acquire),
         "relay acquire role");
  accept(sync.registerWrite(secondRelease, kHome, gated), "second release");
  accept(sync.declareOperationRole(secondRelease, SyncRoleKind::Release),
         "second release role");
  accept(sync.registerRead(secondAcquire, kHome, readsGate), "consumer read");
  accept(sync.declareOperationRole(secondAcquire, SyncRoleKind::Acquire),
         "consumer acquire role");

  expectEffects(takeExpected(sync.visibilitySummary(secondRelease)),
                {data, firstRelease, firstAcquire},
                "a relayed release published the wrong summary");
  expectEffects(takeExpected(sync.importedVisibility(secondAcquire)),
                {data, firstRelease, firstAcquire, secondRelease},
                "transitive imported visibility did not reach the consumer");
  require(sync.happensBefore(data, secondAcquire),
          "transitive visibility did not compose into happens-before");
}

// Effects with no sequenced-before path and no synchronizing relation stay
// unordered in both directions.
void unrelatedStrandsStayUnordered() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicReadId read = takeExpected(order.atomicLoad(kFlag, published));

  MemorySynchronization sync(order);
  SyncEffectId release = sync.declareEffect();
  SyncEffectId acquire = sync.declareEffect();
  SyncEffectId stranger = sync.declareEffect();
  accept(sync.registerWrite(release, kHome, published), "release write");
  accept(sync.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(sync.registerRead(acquire, kHome, read), "acquire read");
  accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");

  require(!sync.happensBefore(stranger, release) &&
              !sync.happensBefore(release, stranger),
          "an unrelated effect became ordered");
  require(!sync.happensBefore(acquire, release),
          "synchronizes-with was applied in both directions");
  require(!sync.synchronizesWith(acquire, release),
          "an acquire synchronized with the release it read");
}

struct Probe {
  SyncEffectId data;
  SyncEffectId release;
  SyncEffectId acquire;
  SyncEffectId use;
};

// One accepted fact set, built by an ordinary caller.
Probe buildProbe(MemorySynchronization &sync, AtomicVersionId published,
                 AtomicReadId read) {
  SyncEffectId data = sync.declareEffect();
  SyncEffectId release = sync.declareEffect();
  SyncEffectId acquire = sync.declareEffect();
  SyncEffectId use = sync.declareEffect();
  accept(sync.sequencedBefore(data, release), "publisher strand");
  accept(sync.registerWrite(release, kHome, published), "release write");
  accept(sync.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(sync.registerRead(acquire, kHome, read), "acquire read");
  accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");
  accept(sync.sequencedBefore(acquire, use), "consumer strand");
  return {data, release, acquire, use};
}

// Every rejection names its exact kind, and a rejected update consumes no
// effect id and changes no accepted fact or derived view.
void rejectionsArePreciseAndAtomic() {
  MemoryAtomicOrder order;
  AtomicVersionId initial = takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId foreignInitial = takeExpected(order.initializeObject(kGate));
  AtomicRmwResult update = takeExpected(order.atomicRmw(kFlag, published));
  AtomicReadId read = takeExpected(order.atomicLoad(kFlag, published));
  AtomicReadId early = takeExpected(order.atomicLoad(kFlag, initial));
  AtomicReadId foreignRead =
      takeExpected(order.atomicLoad(kGate, foreignInitial));
  constexpr auto kAbsent = std::numeric_limits<std::uint64_t>::max();

  MemorySynchronization reference(order);
  Probe referenceProbe = buildProbe(reference, published, read);
  MemorySynchronization probed(order);
  Probe probe = buildProbe(probed, published, read);
  const SyncEffectId unknown(kAbsent);

  expectRejected(probed.sequencedBefore(unknown, probe.use),
                 Kind::UnknownEffect, "an unknown effect was ordered");
  expectRejected(probed.registerWrite(unknown, kHome, published),
                 Kind::UnknownEffect, "an unknown effect took a carrier");
  expectRejected(
      probed.registerWrite(probe.data, kHome, AtomicVersionId(kAbsent)),
      Kind::ForeignRelation, "an unknown version was published");
  expectRejected(probed.registerRead(probe.data, kHome, AtomicReadId(kAbsent)),
                 Kind::ForeignRelation, "an unknown read was imported");
  expectRejected(probed.registerWrite(probe.data, kHome, initial),
                 Kind::InitialVersionPublication,
                 "an initial version was published");
  expectRejected(
      probed.registerWrite(probe.data, kHome, update.version, foreignRead),
      Kind::MismatchedCarry, "a carry from another object was accepted");
  expectRejected(probed.registerWrite(probe.data, kHome, update.version, early),
                 Kind::MismatchedCarry,
                 "a carry that is not the version predecessor was accepted");
  expectRejected(probed.registerWrite(probe.release, kHome, update.version),
                 Kind::DuplicateAssociation, "one effect took two carriers");
  expectRejected(probed.registerWrite(probe.data, kHome, published),
                 Kind::DuplicateAssociation,
                 "one version took two publishing effects");
  expectRejected(probed.registerRead(probe.data, kHome, read),
                 Kind::DuplicateAssociation, "one read took two effects");
  expectRejected(
      probed.declareOperationRole(probe.release, SyncRoleKind::Release),
      Kind::DuplicateRole, "one effect took two roles");
  expectRejected(probed.declareOperationRole(probe.data, SyncRoleKind::Release),
                 Kind::RoleShapeConflict,
                 "a release operation without a write carrier was accepted");
  expectRejected(probed.declareOperationRole(probe.data, SyncRoleKind::AcqRel),
                 Kind::RoleShapeConflict,
                 "an acq_rel operation without a carried write was accepted");
  expectRejected(
      probed.declareFenceRole(probe.release, SyncRoleKind::Release, kHome),
      Kind::DuplicateRole,
      "an effect with an operation role took a fence role");
  expectRejected(probed.sequencedBefore(probe.data, probe.data),
                 Kind::DuplicateEdge, "an effect was ordered before itself");
  expectRejected(probed.sequencedBefore(probe.data, probe.release),
                 Kind::DuplicateEdge, "one edge was recorded twice");
  expectRejected(probed.sequencedBefore(probe.use, probe.data),
                 Kind::CyclicOrder, "a sequenced-before cycle was accepted");
  expectRejected(probed.sequencedBefore(probe.acquire, probe.release),
                 Kind::CyclicOrder,
                 "a cycle through synchronizes-with was accepted");
  expectRejected(probed.visibilitySummary(probe.data), Kind::UnknownRole,
                 "an effect with no release role published a summary");
  expectRejected(probed.importedVisibility(probe.release), Kind::UnknownRole,
                 "an effect with no acquire role imported a summary");
  expectRejected(probed.visibilitySummary(unknown), Kind::UnknownEffect,
                 "an unknown effect published a summary");

  // A fence role on an effect with no role and no carrier is legal, so the
  // rejections above name the existing role or carrier rather than the fence
  // spelling.
  SyncEffectId referenceFence = reference.declareEffect();
  SyncEffectId probedFence = probed.declareEffect();
  require(referenceFence == probedFence,
          "a rejected update consumed an effect id");
  accept(
      reference.declareFenceRole(referenceFence, SyncRoleKind::AcqRel, kHome),
      "reference fence role");
  accept(probed.declareFenceRole(probedFence, SyncRoleKind::AcqRel, kHome),
         "probed fence role");

  require(probed.synchronizesWith(probe.release, probe.acquire) ==
              reference.synchronizesWith(referenceProbe.release,
                                         referenceProbe.acquire),
          "a rejected update changed synchronizes-with");
  require(probed.happensBefore(probe.data, probe.use) ==
              reference.happensBefore(referenceProbe.data, referenceProbe.use),
          "a rejected update changed happens-before");
  expectEffects(takeExpected(probed.importedVisibility(probe.acquire)),
                {probe.data, probe.release},
                "a rejected update changed an imported summary");
  expectEffects(takeExpected(probed.publishedOrigins(published)),
                {probe.release}, "a rejected update changed a publication");
  expectEffects(takeExpected(probed.publishedOrigins(update.version)), {},
                "a rejected update created a publication");
}

// A fence has no addressed access in either declaration order: taking a carrier
// on an effect that already has a fence role is the same conflict as taking a
// fence role on an effect that already has a carrier. A refused association
// stays available to a proper operation effect and moves no derived relation.
void fenceShapeHoldsInEitherDeclarationOrder() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  takeExpected(order.initializeObject(kGate));
  AtomicVersionId flagged = takeExpected(order.atomicStore(kFlag));
  AtomicVersionId gated = takeExpected(order.atomicStore(kGate));
  AtomicReadId readsFlag = takeExpected(order.atomicLoad(kFlag, flagged));
  AtomicReadId readsGate = takeExpected(order.atomicLoad(kGate, gated));

  MemorySynchronization sync(order);
  SyncEffectId data = sync.declareEffect();
  SyncEffectId release = sync.declareEffect();
  SyncEffectId acquire = sync.declareEffect();
  SyncEffectId releaseFence = sync.declareEffect();
  SyncEffectId acquireFence = sync.declareEffect();
  SyncEffectId publisher = sync.declareEffect();
  SyncEffectId consumer = sync.declareEffect();
  accept(sync.sequencedBefore(data, release), "publisher strand");
  accept(sync.registerWrite(release, kHome, flagged), "release write");
  accept(sync.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(sync.registerRead(acquire, kHome, readsFlag), "acquire read");
  accept(sync.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");
  accept(sync.declareFenceRole(releaseFence, SyncRoleKind::Release, kHome),
         "release fence role");
  accept(sync.declareFenceRole(acquireFence, SyncRoleKind::Acquire, kHome),
         "acquire fence role");

  expectRejected(sync.registerWrite(releaseFence, kHome, gated),
                 Kind::RoleShapeConflict,
                 "a release fence took a write carrier");
  expectRejected(sync.registerRead(acquireFence, kHome, readsGate),
                 Kind::RoleShapeConflict,
                 "an acquire fence took a read carrier");

  // The refused version and read stayed unowned, so proper operation effects
  // still take them. Each fresh carrier then refuses a fence role, which is the
  // carrier-first half of the same shape rule, and still accepts the operation
  // role its carrier does support.
  accept(sync.registerWrite(publisher, kHome, gated), "second release write");
  expectRejected(sync.declareFenceRole(publisher, SyncRoleKind::Release, kHome),
                 Kind::RoleShapeConflict, "a write carrier took a fence role");
  accept(sync.declareOperationRole(publisher, SyncRoleKind::Release),
         "second release role");
  accept(sync.registerRead(consumer, kHome, readsGate), "second acquire read");
  expectRejected(sync.declareFenceRole(consumer, SyncRoleKind::Acquire, kHome),
                 Kind::RoleShapeConflict, "a read carrier took a fence role");
  accept(sync.declareOperationRole(consumer, SyncRoleKind::Acquire),
         "second acquire role");
  require(sync.synchronizesWith(publisher, consumer),
          "a refused fence carrier consumed the version or read association");

  // The relations that existed before the refusals did not move, and neither
  // fence acquired one.
  require(sync.synchronizesWith(release, acquire) &&
              !sync.synchronizesWith(releaseFence, acquire) &&
              !sync.synchronizesWith(release, acquireFence),
          "a refused fence carrier changed synchronizes-with");
  require(sync.happensBefore(data, acquire) &&
              !sync.happensBefore(data, acquireFence),
          "a refused fence carrier changed happens-before");
  expectEffects(takeExpected(sync.publishedOrigins(flagged)), {release},
                "a refused fence carrier changed a publication");
  expectEffects(takeExpected(sync.importedVisibility(acquire)), {data, release},
                "a refused fence carrier changed an import");
}

// The derived views are pure functions of the accepted fact set, so a caller
// that interleaves the same facts differently observes the same relations.
void acceptedFactsAreInsertionOrderInvariant() {
  MemoryAtomicOrder order;
  takeExpected(order.initializeObject(kFlag));
  AtomicVersionId published = takeExpected(order.atomicStore(kFlag));
  AtomicReadId read = takeExpected(order.atomicLoad(kFlag, published));

  MemorySynchronization forward(order);
  Probe forwardProbe = buildProbe(forward, published, read);

  // The same facts with every sequenced-before edge recorded last.
  MemorySynchronization reversed(order);
  SyncEffectId data = reversed.declareEffect();
  SyncEffectId release = reversed.declareEffect();
  SyncEffectId acquire = reversed.declareEffect();
  SyncEffectId use = reversed.declareEffect();
  accept(reversed.registerRead(acquire, kHome, read), "acquire read");
  accept(reversed.declareOperationRole(acquire, SyncRoleKind::Acquire),
         "acquire role");
  accept(reversed.registerWrite(release, kHome, published), "release write");
  accept(reversed.declareOperationRole(release, SyncRoleKind::Release),
         "release role");
  accept(reversed.sequencedBefore(acquire, use), "consumer strand");
  accept(reversed.sequencedBefore(data, release), "publisher strand");

  require(
      forward.synchronizesWith(forwardProbe.release, forwardProbe.acquire) ==
          reversed.synchronizesWith(release, acquire),
      "insertion order changed synchronizes-with");
  require(forward.happensBefore(forwardProbe.data, forwardProbe.use) ==
              reversed.happensBefore(data, use),
          "insertion order changed happens-before");
  expectEffects(takeExpected(reversed.importedVisibility(acquire)),
                {data, release}, "insertion order changed an imported summary");
  expectEffects(takeExpected(reversed.visibilitySummary(release)), {data},
                "insertion order changed a published summary");
  expectEffects(takeExpected(reversed.publishedOrigins(published)), {release},
                "insertion order changed a publication");
}

// One issue can inherit a wide token frontier. The authority accepts all of
// those sequenced-before facts as one transaction, and a rejected collection
// commits none of its otherwise-valid prefix.
void wideFrontierUsesOneTransactionalInsertion() {
  MemoryAtomicOrder order;
  MemorySynchronization sync(order);
  constexpr unsigned kFanIn = 1024;
  llvm::SmallVector<SyncEffectId> sources;
  sources.reserve(kFanIn);
  for (unsigned index = 0; index < kFanIn; ++index)
    sources.push_back(sync.declareEffect());
  SyncEffectId target = takeExpected(sync.declareEffectSequencedAfter(sources));
  require(sync.sequencedEdgeCount() == kFanIn,
          "wide incoming declaration dropped an independent predecessor");

  llvm::SmallVector<std::pair<SyncEffectId, SyncEffectId>> facts;
  facts.reserve(kFanIn - 1);
  for (unsigned index = 0; index + 1 < kFanIn; ++index)
    facts.emplace_back(sources[index], sources[index + 1]);
  accept(sync.sequencedBefore(facts), "wide source chain");
  require(sync.sequencedEdgeCount() == kFanIn,
          "wide incoming frontier was not transitively reduced");
  require(sync.areCoveredByHappensBefore(sources, {sources.back()}),
          "the chain tail did not cover its sequenced predecessors");
  require(sync.areCoveredByHappensBefore(sources, {target}),
          "the target did not cover its wide incoming frontier");

  SyncEffectId unrelated = sync.declareEffect();
  sources.push_back(unrelated);
  require(!sync.areCoveredByHappensBefore(sources, {target}),
          "an unrelated effect was covered by the target");
  llvm::SmallVector<SyncEffectId, 2> frontier{target, unrelated};
  require(sync.areCoveredByHappensBefore(sources, frontier),
          "a multi-effect frontier did not cover all requested effects");

  SyncEffectId refusedSource = sync.declareEffect();
  SyncEffectId refusedTarget = sync.declareEffect();
  const SyncEffectId unknown(std::numeric_limits<std::uint64_t>::max());
  llvm::SmallVector<std::pair<SyncEffectId, SyncEffectId>, 2> refused{
      {refusedSource, refusedTarget}, {refusedTarget, unknown}};
  expectRejected(sync.sequencedBefore(refused), Kind::UnknownEffect,
                 "a batch with an unknown effect was accepted");
  require(!sync.happensBefore(refusedSource, refusedTarget),
          "a rejected batch committed its valid prefix");
}

void rejectedIncomingFrontierDoesNotAllocateEffect() {
  MemoryAtomicOrder order;
  MemorySynchronization sync(order);
  llvm::SmallVector<SyncEffectId, 1> unknown{SyncEffectId(0)};

  expectRejected(sync.declareEffectSequencedAfter(unknown), Kind::UnknownEffect,
                 "an unknown predecessor was accepted");
  SyncEffectId first = takeExpected(sync.declareEffectSequencedAfter({}));
  require(first == SyncEffectId(0),
          "a rejected incoming frontier consumed an effect id");
}

void loopCarriedFrontierAndRelationStayReduced() {
  MemoryAtomicOrder order;
  MemorySynchronization sync(order);
  constexpr unsigned kEffects = 2048;
  llvm::SmallVector<SyncEffectId> history;
  llvm::SmallVector<SyncEffectId, 2> frontier;
  history.reserve(kEffects);

  for (unsigned index = 0; index < kEffects; ++index) {
    SyncEffectId effect =
        takeExpected(sync.declareEffectSequencedAfter(frontier));
    history.push_back(effect);
    frontier.push_back(effect);
    frontier = takeExpected(sync.maximalHappensBeforeFrontier(frontier));
    require(frontier.size() == 1 && frontier.front() == effect,
            "a loop-carried frontier retained a transitive predecessor");
  }
  require(sync.sequencedEdgeCount() == kEffects - 1,
          "a linear effect chain stored a transitive edge");

  SyncEffectId joined = takeExpected(sync.declareEffectSequencedAfter(history));
  require(sync.sequencedEdgeCount() == kEffects,
          "a historical incoming frontier stored redundant direct edges");
  require(sync.happensBefore(history.front(), joined),
          "frontier reduction changed happens-before");
}

} // namespace

int main() {
  releaseAndAcquireImportOnlyThroughTheSelectedVersion();
  releaseOperationDoesNotHookLaterWrites();
  acquireOperationDoesNotHookEarlierReads();
  domainIdentityAndReadsFromAreRequired();
  fenceChainSynchronizesOnlyThroughItsCarriers();
  releaseSequenceCarriesOnlyThroughOneDomain();
  importedVisibilityReachesALaterReleaseSummary();
  unrelatedStrandsStayUnordered();
  rejectionsArePreciseAndAtomic();
  fenceShapeHoldsInEitherDeclarationOrder();
  acceptedFactsAreInsertionOrderInvariant();
  wideFrontierUsesOneTransactionalInsertion();
  rejectedIncomingFrontierDoesNotAllocateEffect();
  loopCarriedFrontierAndRelationStayReduced();
  return 0;
}
