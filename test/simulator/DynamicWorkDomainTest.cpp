#include "Simulator/DynamicWorkDomain.h"
#include "Simulator/DynamicWorkScheduler.h"
#include "Simulator/ThreadDispatchIdentity.h"

#include "DynamicWorkOrdinal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

using namespace loom::sim;

static_assert(std::is_copy_constructible<WorkItemId>::value,
              "WorkItemId must remain copy constructible");
static_assert(std::is_copy_assignable<WorkItemId>::value,
              "WorkItemId must remain copy assignable");

static_assert(!std::is_copy_constructible<WorkResponsibility>::value,
              "WorkResponsibility must not be copy constructible");
static_assert(!std::is_copy_assignable<WorkResponsibility>::value,
              "WorkResponsibility must not be copy assignable");
static_assert(!std::is_default_constructible<WorkResponsibility>::value,
              "WorkResponsibility must not be default constructible");
static_assert(!std::is_constructible<WorkResponsibility, WorkItemId>::value,
              "WorkItemId must not forge a WorkResponsibility");
static_assert(std::is_move_constructible<WorkResponsibility>::value,
              "WorkResponsibility must be move constructible");
static_assert(!std::is_move_assignable<WorkResponsibility>::value,
              "WorkResponsibility assignment must not discard responsibility");
static_assert(
    std::is_same<decltype(std::declval<const WorkResponsibility &>().id()),
                 WorkItemId>::value,
    "WorkResponsibility must return its observation by value");

static_assert(!std::is_copy_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy constructible");
static_assert(!std::is_copy_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy assignable");
static_assert(!std::is_move_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move constructible");
static_assert(!std::is_move_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move assignable");

static_assert(!std::is_copy_constructible<DynamicWorkAssignment>::value,
              "a worker assignment must not be copied");
static_assert(std::is_move_constructible<DynamicWorkAssignment>::value,
              "a worker assignment must support ownership transfer");
static_assert(!std::is_move_assignable<DynamicWorkAssignment>::value,
              "assignment overwrite must not abandon live work");
static_assert(
    !std::is_constructible<DynamicWorkCancellationResult,
                           DynamicWorkCancellationKind,
                           std::optional<RetirementEffect>>::value,
    "cancellation callers must not forge contradictory typed outcomes");

static_assert(
    !std::is_invocable<decltype(&DynamicWorkDomain::spawnChild),
                       DynamicWorkDomain &, const WorkItemId &>::value,
    "WorkItemId must not authorize child spawn");
static_assert(!std::is_invocable<decltype(&DynamicWorkDomain::retire),
                                 DynamicWorkDomain &, WorkItemId &&>::value,
              "WorkItemId must not authorize retirement");

namespace {

using Kind = DynamicWorkDomainError::Kind;
using SchedulerKind = DynamicWorkSchedulerError::Kind;

constexpr ThreadDispatchOccurrenceId kDispatch(7);

[[noreturn]] void fail(llvm::StringRef message);

dataflow::RootThreadLaunchRef rootLaunch(std::uint8_t seed,
                                         std::uint64_t entity) {
  loom::ArtifactIdentity::Storage bytes{};
  bytes.fill(seed);
  auto identity = loom::ArtifactIdentity::fromBytes(bytes);
  if (!identity)
    fail(llvm::toString(identity.takeError()));
  return {*identity, dataflow::RootThreadLaunchId(entity)};
}

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DynamicWorkDomainTest: " << message << "\n";
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
void expectRejected(llvm::Expected<T> value, Kind kind,
                    llvm::StringRef message) {
  require(!value, message);
  std::optional<Kind> rejected;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const DynamicWorkDomainError &error) { rejected = error.kind(); },
      [&](const llvm::ErrorInfoBase &) {});
  require(rejected && *rejected == kind, message);
}

void expectEffect(llvm::Expected<RetirementEffect> value, RetirementEffect want,
                  llvm::StringRef message) {
  require(takeExpected(std::move(value)) == want, message);
}

template <typename T>
void expectSchedulerRejected(llvm::Expected<T> value, SchedulerKind kind,
                             llvm::StringRef message) {
  require(!value, message);
  std::optional<SchedulerKind> rejected;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const DynamicWorkSchedulerError &error) { rejected = error.kind(); },
      [&](const llvm::ErrorInfoBase &) {});
  require(rejected && *rejected == kind, message);
}

void rootIdentityAndImmediateSourceClosure() {
  DynamicWorkDomain domain(kDispatch);
  require(domain.activeCount() == 0 && !domain.completed(),
          "a fresh domain is neither active nor completed");

  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();
  require(rootId == WorkItemId::root(kDispatch),
          "the admitted root is not the canonical root identity");
  require(rootId.domainInstance() == kDispatch && rootId.isRoot() &&
              rootId.ordinal() == 0 && !rootId.parent(),
          "the root identity is not (dispatch occurrence, Root, 0)");
  require(domain.activeCount() == 1,
          "root admission did not acquire responsibility before publication");
  require(!domain.completed(), "a domain with an active root is completed");

  expectRejected(domain.admitRoot(), Kind::RootAlreadyAdmitted,
                 "the root source admitted a second root");
  require(domain.activeCount() == 1,
          "a rejected second admission changed the active set");

  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainCompleted,
               "retiring a childless root did not complete the domain");
  require(domain.completed() && domain.activeCount() == 0,
          "a completed domain is not empty and stable");
}

void deterministicChildOrdinals() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();

  WorkResponsibility first = takeExpected(domain.spawnChild(root));
  WorkResponsibility second = takeExpected(domain.spawnChild(root));
  WorkResponsibility third = takeExpected(domain.spawnChild(root));
  WorkItemId firstId = first.id();
  WorkItemId secondId = second.id();
  WorkItemId thirdId = third.id();

  require(firstId == WorkItemId::child(rootId, 0) &&
              secondId == WorkItemId::child(rootId, 1) &&
              thirdId == WorkItemId::child(rootId, 2),
          "child identities are not a zero-based program-order sequence");
  require(firstId.ordinal() == 0 && secondId.ordinal() == 1 &&
              thirdId.ordinal() == 2,
          "a child ordinal does not match its spawn order");
  require(firstId.parent() == rootId && secondId.parent() == rootId &&
              thirdId.parent() == rootId,
          "a child identity did not recursively include its parent");
  require(domain.activeCount() == 4,
          "spawning did not acquire one responsibility per child");
}

void descendantsPreventEarlyCompletion() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkResponsibility child = takeExpected(domain.spawnChild(root));
  require(domain.activeCount() == 2,
          "the child responsibility was not acquired before publication");

  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainStillActive,
               "retiring the parent completed a domain with an active child");
  require(!domain.completed() && domain.activeCount() == 1,
          "an active descendant did not keep the domain alive");

  expectEffect(domain.retire(std::move(child)),
               RetirementEffect::DomainCompleted,
               "retiring the last descendant did not complete the domain");
  require(domain.completed(), "the drained domain is not completed");
}

void completionTransitionIsExactlyOnce() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkResponsibility a = takeExpected(domain.spawnChild(root));
  WorkResponsibility b = takeExpected(domain.spawnChild(root));

  expectEffect(domain.retire(std::move(a)), RetirementEffect::DomainStillActive,
               "retiring one of several active items completed the domain");
  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainStillActive,
               "retiring the root with an active child completed the domain");
  expectEffect(domain.retire(std::move(b)), RetirementEffect::DomainCompleted,
               "the final retirement did not complete the domain");

  require(domain.completed() && domain.activeCount() == 0,
          "the completed domain is not stable");
  expectRejected(domain.retire(std::move(b)), Kind::InvalidResponsibility,
                 "a consumed responsibility retired twice");
  require(domain.completed() && domain.activeCount() == 0,
          "a rejected repeated retirement disturbed the completed domain");
}

void movedFromCapabilityRejectsAtomically() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility original = takeExpected(domain.admitRoot());
  WorkResponsibility owner(std::move(original));

  expectRejected(domain.spawnChild(original), Kind::InvalidResponsibility,
                 "a moved-from capability authorized child spawn");
  expectRejected(domain.retire(std::move(original)),
                 Kind::InvalidResponsibility,
                 "a moved-from capability authorized retirement");
  require(domain.activeCount() == 1 && !domain.completed(),
          "a moved-from capability changed domain state");

  WorkResponsibility child = takeExpected(domain.spawnChild(owner));
  expectEffect(domain.retire(std::move(child)),
               RetirementEffect::DomainStillActive,
               "the moved-to parent lost its responsibility");
  expectEffect(domain.retire(std::move(owner)),
               RetirementEffect::DomainCompleted,
               "the moved-to capability did not retire the root");
}

void observationMutationCannotRewriteCapability() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();

  const WorkItemId &observation = root.id();
  WorkItemId replacement = WorkItemId::child(rootId, 99);
  const_cast<WorkItemId &>(observation) = replacement;
  require(observation == replacement,
          "the adversarial observation mutation did not execute");
  require(root.id() == rootId,
          "mutating an observation rewrote the responsibility identity");

  WorkResponsibility child = takeExpected(domain.spawnChild(root));
  require(child.id() == WorkItemId::child(rootId, 0),
          "an observation mutation redirected child spawn");
  expectEffect(domain.retire(std::move(child)),
               RetirementEffect::DomainStillActive,
               "observation mutation changed child responsibility");
  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainCompleted,
               "observation mutation changed root responsibility");
}

void logicalPointsRetainRootAndDispatchIdentity() {
  dataflow::RootThreadLaunchRef firstRoot = rootLaunch(0x11, 3);
  dataflow::RootThreadLaunchRef secondRoot = rootLaunch(0x11, 4);

  DenseLogicalThreadPoint dense{firstRoot, {2, 5}};
  require(dense.rootThreadLaunch == firstRoot &&
              dense.coordinates.size() == 2 && dense.coordinates[0] == 2 &&
              dense.coordinates[1] == 5,
          "a dense logical point lost its rooted coordinate tuple");
  require(!(dense == DenseLogicalThreadPoint{secondRoot, {2, 5}}),
          "dense points from distinct root launches were conflated");

  WorkItemId item = WorkItemId::child(WorkItemId::root(kDispatch), 9);
  DynamicLogicalThreadPoint dynamic{firstRoot, item};
  require(dynamic.rootThreadLaunch == firstRoot && dynamic.workItem == item &&
              dynamic.workItem.domainInstance() == kDispatch,
          "a dynamic logical point lost its root or dispatch occurrence");
  require(!(dynamic == DynamicLogicalThreadPoint{secondRoot, item}),
          "dynamic points from distinct root launches were conflated");

  LogicalThreadPoint densePoint = dense;
  LogicalThreadPoint dynamicPoint = dynamic;
  require(std::holds_alternative<DenseLogicalThreadPoint>(densePoint) &&
              std::holds_alternative<DynamicLogicalThreadPoint>(dynamicPoint),
          "logical thread point alternatives are not closed and typed");
}

void observationsFromDistinctDispatchesCannotAuthorize() {
  constexpr ThreadDispatchOccurrenceId leftDispatch(7);
  constexpr ThreadDispatchOccurrenceId rightDispatch(8);
  DynamicWorkDomain left(leftDispatch);
  DynamicWorkDomain right(rightDispatch);
  WorkResponsibility leftRoot = takeExpected(left.admitRoot());
  WorkResponsibility rightRoot = takeExpected(right.admitRoot());

  WorkItemId leftPredictedRoot = WorkItemId::root(leftDispatch);
  WorkItemId rightPredictedRoot = WorkItemId::root(rightDispatch);
  require(leftRoot.id() == leftPredictedRoot &&
              rightRoot.id() == rightPredictedRoot &&
              leftRoot.id() != rightRoot.id(),
          "distinct dispatch occurrences produced the same root observation");

  expectRejected(left.spawnChild(rightRoot), Kind::ForeignDomain,
                 "a foreign dispatch capability authorized child spawn");
  expectRejected(left.retire(std::move(rightRoot)), Kind::ForeignDomain,
                 "a foreign dispatch capability authorized retirement");
  require(left.activeCount() == 1 && right.activeCount() == 1 &&
              !left.completed() && !right.completed(),
          "cross-domain rejection changed either active set");

  WorkResponsibility leftChild = takeExpected(left.spawnChild(leftRoot));
  WorkResponsibility rightChild = takeExpected(right.spawnChild(rightRoot));
  require(leftChild.id() == WorkItemId::child(leftPredictedRoot, 0) &&
              rightChild.id() == WorkItemId::child(rightPredictedRoot, 0),
          "rejected cross-domain operations consumed an ordinal");

  expectEffect(left.retire(std::move(leftChild)),
               RetirementEffect::DomainStillActive,
               "left child retirement completed an active root");
  expectEffect(left.retire(std::move(leftRoot)),
               RetirementEffect::DomainCompleted,
               "left root did not complete its own domain");
  expectEffect(right.retire(std::move(rightChild)),
               RetirementEffect::DomainStillActive,
               "right child retirement completed an active root");
  expectEffect(right.retire(std::move(rightRoot)),
               RetirementEffect::DomainCompleted,
               "foreign rejection consumed the right root capability");
}

void independentOrdinalSequencesForInterleavedParents() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();

  WorkResponsibility a = takeExpected(domain.spawnChild(root));
  WorkResponsibility ga = takeExpected(domain.spawnChild(a));
  WorkResponsibility b = takeExpected(domain.spawnChild(root));
  WorkResponsibility gb = takeExpected(domain.spawnChild(a));
  WorkResponsibility c = takeExpected(domain.spawnChild(root));
  WorkItemId aId = a.id();
  WorkItemId gaId = ga.id();
  WorkItemId bId = b.id();
  WorkItemId gbId = gb.id();
  WorkItemId cId = c.id();

  require(aId.ordinal() == 0 && bId.ordinal() == 1 && cId.ordinal() == 2,
          "the root's child ordinals are not an independent sequence");
  require(gaId.ordinal() == 0 && gbId.ordinal() == 1,
          "a nested parent's child ordinals are not an independent sequence");
  require(aId.parent() == rootId && bId.parent() == rootId &&
              cId.parent() == rootId,
          "a root child did not name the root as parent");
  require(gaId.parent() == aId && gbId.parent() == aId,
          "a nested child did not name its own parent");
}

void retiredParentRejectsSpawnAtomically() {
  DynamicWorkDomain domain(kDispatch);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkResponsibility parent = takeExpected(domain.spawnChild(root));

  expectEffect(domain.retire(std::move(parent)),
               RetirementEffect::DomainStillActive,
               "retiring the child parent completed an active domain");
  expectRejected(domain.spawnChild(parent), Kind::InvalidResponsibility,
                 "spawn accepted a moved-from parent responsibility");
  require(domain.activeCount() == 1 && !domain.completed(),
          "a rejected spawn from a retired parent changed domain state");

  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainCompleted,
               "the remaining root did not complete the domain");
}

void maximumChildOrdinalPrecedesExhaustion() {
  detail::ChildOrdinalCursor cursor(std::numeric_limits<std::uint64_t>::max() -
                                    1);

  std::optional<std::uint64_t> ordinal = cursor.take();
  require(ordinal && *ordinal == std::numeric_limits<std::uint64_t>::max() - 1,
          "the cursor did not issue the penultimate ordinal");

  ordinal = cursor.take();
  require(ordinal && *ordinal == std::numeric_limits<std::uint64_t>::max(),
          "the cursor did not issue UINT64_MAX");

  ordinal = cursor.take();
  require(!ordinal, "the cursor did not enter explicit exhaustion");
  ordinal = cursor.take();
  require(!ordinal, "an exhausted cursor resumed issuing ordinals");
}

void boundedDequeStealingPreservesLogicalIdentity() {
  auto scheduler =
      takeExpected(DynamicWorkScheduler::create(kDispatch, 2, 2, {0x41}));
  auto rootResult = takeExpected(scheduler->acquire(1));
  require(rootResult.has_value(), "an idle worker did not steal the root");
  DynamicWorkAssignment root = std::move(*rootResult);
  const WorkItemId rootId = root.id();
  require(rootId == WorkItemId::root(kDispatch) && root.workerOrdinal() == 1 &&
              root.payload().size() == 1 && root.payload().front() == 0x41,
          "root stealing changed logical identity or payload");

  auto first = takeExpected(scheduler->publishChild(root, {0x51}));
  auto second = takeExpected(scheduler->publishChild(root, {0x52}));
  require(first.kind == DynamicWorkPublishKind::Published && first.child &&
              *first.child == WorkItemId::child(rootId, 0) &&
              second.kind == DynamicWorkPublishKind::Published &&
              second.child && *second.child == WorkItemId::child(rootId, 1),
          "child publication did not use canonical program-order identities");

  auto blocked = takeExpected(scheduler->publishChild(root, {0x53}));
  require(blocked.kind == DynamicWorkPublishKind::WouldBlock && !blocked.child,
          "a full local deque did not provide typed backpressure");
  expectSchedulerRejected(scheduler->acquire(1), SchedulerKind::WorkerBusy,
                          "a worker acquired two active assignments");

  auto stolenResult = takeExpected(scheduler->acquire(0));
  require(stolenResult.has_value(),
          "the published child was not available to a thief");
  DynamicWorkAssignment stolen = std::move(*stolenResult);
  require(stolen.id() == *first.child && stolen.workerOrdinal() == 0 &&
              stolen.payload().front() == 0x51,
          "stealing did not take the victim deque front");

  auto afterCapacity = takeExpected(scheduler->publishChild(root, {0x53}));
  require(afterCapacity.kind == DynamicWorkPublishKind::Published &&
              afterCapacity.child &&
              *afterCapacity.child == WorkItemId::child(rootId, 2),
          "backpressure consumed a child identity or failed to recover");

  expectEffect(scheduler->complete(std::move(root)),
               RetirementEffect::DomainStillActive,
               "root completion retired its published descendants");
  auto localResult = takeExpected(scheduler->acquire(1));
  require(localResult.has_value(),
          "the local worker did not reacquire published work");
  DynamicWorkAssignment local = std::move(*localResult);
  require(local.id() == *afterCapacity.child && local.workerOrdinal() == 1 &&
              local.payload().front() == 0x53,
          "local acquisition did not take the owner deque back");
  expectEffect(scheduler->complete(std::move(stolen)),
               RetirementEffect::DomainStillActive,
               "stolen child completion retired unrelated work");
  expectEffect(scheduler->complete(std::move(local)),
               RetirementEffect::DomainStillActive,
               "local child completion retired unrelated work");
  auto lastResult = takeExpected(scheduler->acquire(1));
  require(lastResult && (*lastResult).id() == *second.child,
          "the remaining child was not replayably queued");
  expectEffect(scheduler->complete(std::move(*lastResult)),
               RetirementEffect::DomainCompleted,
               "last child completion did not close the drained domain");
  require(scheduler->completed() && scheduler->activeCount() == 0 &&
              scheduler->queuedCount() == 0,
          "the drained scheduler retained work or completion authority");

  const auto replay = scheduler->replay();
  const DynamicWorkScheduleActionKind expectedKinds[] = {
      DynamicWorkScheduleActionKind::AdmitRoot,
      DynamicWorkScheduleActionKind::Steal,
      DynamicWorkScheduleActionKind::PublishChild,
      DynamicWorkScheduleActionKind::PublishChild,
      DynamicWorkScheduleActionKind::Steal,
      DynamicWorkScheduleActionKind::PublishChild,
      DynamicWorkScheduleActionKind::Complete,
      DynamicWorkScheduleActionKind::AcquireLocal,
      DynamicWorkScheduleActionKind::Complete,
      DynamicWorkScheduleActionKind::Complete,
      DynamicWorkScheduleActionKind::AcquireLocal,
      DynamicWorkScheduleActionKind::Complete,
  };
  require(replay.size() == sizeof(expectedKinds) / sizeof(expectedKinds[0]),
          "replay retained a rejected or omitted a committed transition");
  for (std::size_t index = 0; index < replay.size(); ++index)
    require(replay[index].kind == expectedKinds[index],
            "scheduler replay order is not deterministic");
  require(replay[1].item == rootId && replay[1].sourceWorker == 0 &&
              replay[1].targetWorker == 1 && replay[4].item == *first.child &&
              replay[4].sourceWorker == 1 && replay[4].targetWorker == 0,
          "replay lost a deterministic ownership transfer");
}

void cancellationRetiresExactlyOneResponsibility() {
  auto scheduler =
      takeExpected(DynamicWorkScheduler::create(kDispatch, 2, 2, {0x61}));
  auto rootResult = takeExpected(scheduler->acquire(0));
  require(rootResult.has_value(), "root acquisition failed");
  DynamicWorkAssignment root = std::move(*rootResult);
  auto child = takeExpected(scheduler->publishChild(root, {0x62}));
  require(child.child.has_value(), "cancellation fixture child was not queued");

  auto queuedCancellation =
      takeExpected(scheduler->requestCancellation(*child.child));
  require(queuedCancellation.kind() ==
                  DynamicWorkCancellationKind::CancelledQueued &&
              queuedCancellation.retirementEffect() ==
                  RetirementEffect::DomainStillActive,
          "queued cancellation did not retire its responsibility");
  require(scheduler->activeCount() == 1 && scheduler->queuedCount() == 0,
          "queued cancellation changed the root responsibility");
  auto descendant = takeExpected(scheduler->publishChild(root, {0x64}));
  require(descendant.kind == DynamicWorkPublishKind::Published &&
              descendant.child.has_value(),
          "active cancellation fixture did not publish its descendant");
  auto activeCancellation =
      takeExpected(scheduler->requestCancellation(root.id()));
  require(activeCancellation.kind() ==
                  DynamicWorkCancellationKind::RequestedActive &&
              !activeCancellation.retirementEffect(),
          "active cancellation was not delivered as a request");
  auto repeatedCancellation =
      takeExpected(scheduler->requestCancellation(root.id()));
  require(repeatedCancellation.kind() ==
                  DynamicWorkCancellationKind::AlreadyRequested &&
              !repeatedCancellation.retirementEffect(),
          "repeated active cancellation changed its typed outcome");
  expectSchedulerRejected(scheduler->complete(std::move(root)),
                          SchedulerKind::CancellationPending,
                          "completion bypassed a pending cancellation");
  require(takeExpected(scheduler->cancellationRequested(root)),
          "the worker could not observe its cancellation request");
  auto rejectedPublish = takeExpected(scheduler->publishChild(root, {0x63}));
  require(rejectedPublish.kind ==
                  DynamicWorkPublishKind::CancellationRequested &&
              !rejectedPublish.child,
          "a cancelled active item published new responsibility");
  expectEffect(scheduler->cancel(std::move(root)),
               RetirementEffect::DomainStillActive,
               "active cancellation recursively retired a descendant");
  auto descendantResult = takeExpected(scheduler->acquire(0));
  require(descendantResult && descendantResult->id() == *descendant.child,
          "a surviving descendant was not available after parent cancellation");
  expectEffect(scheduler->complete(std::move(*descendantResult)),
               RetirementEffect::DomainCompleted,
               "the surviving descendant lost the completion transition");

  const auto replay = scheduler->replay();
  require(replay.size() == 9 &&
              replay[3].kind == DynamicWorkScheduleActionKind::CancelQueued &&
              replay[4].kind == DynamicWorkScheduleActionKind::PublishChild &&
              replay[5].kind ==
                  DynamicWorkScheduleActionKind::RequestCancellation &&
              replay[6].kind == DynamicWorkScheduleActionKind::CancelActive &&
              replay[7].kind == DynamicWorkScheduleActionKind::AcquireLocal &&
              replay[8].kind == DynamicWorkScheduleActionKind::Complete,
          "cancellation replay retained a typed no-op or lost retirement");
}

void queuedRootCancellationCarriesCompletionTransition() {
  auto scheduler =
      takeExpected(DynamicWorkScheduler::create(kDispatch, 1, 1, {0x68}));
  const WorkItemId root = WorkItemId::root(kDispatch);
  auto cancellation = takeExpected(scheduler->requestCancellation(root));
  require(cancellation.kind() == DynamicWorkCancellationKind::CancelledQueued &&
              cancellation.retirementEffect() ==
                  RetirementEffect::DomainCompleted,
          "queued root cancellation lost the completion transition");
  require(scheduler->completed() && scheduler->activeCount() == 0 &&
              scheduler->queuedCount() == 0,
          "queued root cancellation left live scheduler state");
  const auto replay = scheduler->replay();
  require(replay.size() == 2 && replay[1].item == root &&
              replay[1].kind == DynamicWorkScheduleActionKind::CancelQueued &&
              replay[1].sourceWorker == 0 && !replay[1].targetWorker,
          "queued root cancellation replay is not canonical");
  expectSchedulerRejected(scheduler->requestCancellation(root),
                          SchedulerKind::UnknownItem,
                          "a retired root accepted repeated cancellation");
}

void assignmentAuthorityCannotCrossSchedulers() {
  auto left =
      takeExpected(DynamicWorkScheduler::create(kDispatch, 1, 1, {0x71}));
  auto right =
      takeExpected(DynamicWorkScheduler::create(kDispatch, 1, 1, {0x72}));
  auto leftResult = takeExpected(left->acquire(0));
  auto rightResult = takeExpected(right->acquire(0));
  require(leftResult && rightResult,
          "foreign assignment fixture is incomplete");
  DynamicWorkAssignment leftAssignment = std::move(*leftResult);
  DynamicWorkAssignment rightAssignment = std::move(*rightResult);
  require(leftAssignment.id() == rightAssignment.id(),
          "foreign assignment fixture did not collide in logical identity");

  expectSchedulerRejected(right->complete(std::move(leftAssignment)),
                          SchedulerKind::InvalidAssignment,
                          "a worker assignment crossed scheduler owners");
  require(left->activeCount() == 1 && right->activeCount() == 1,
          "foreign assignment rejection changed responsibility state");
  expectEffect(left->complete(std::move(leftAssignment)),
               RetirementEffect::DomainCompleted,
               "the rejected assignment lost its original authority");
  expectSchedulerRejected(left->complete(std::move(leftAssignment)),
                          SchedulerKind::InvalidAssignment,
                          "a consumed assignment retired twice");
  expectSchedulerRejected(right->cancel(std::move(rightAssignment)),
                          SchedulerKind::CancellationNotRequested,
                          "an assignment cancelled without a request");
  expectEffect(right->complete(std::move(rightAssignment)),
               RetirementEffect::DomainCompleted,
               "foreign rejection disturbed the peer assignment");
}

} // namespace

int main() {
  rootIdentityAndImmediateSourceClosure();
  deterministicChildOrdinals();
  descendantsPreventEarlyCompletion();
  completionTransitionIsExactlyOnce();
  movedFromCapabilityRejectsAtomically();
  observationMutationCannotRewriteCapability();
  logicalPointsRetainRootAndDispatchIdentity();
  observationsFromDistinctDispatchesCannotAuthorize();
  independentOrdinalSequencesForInterleavedParents();
  retiredParentRejectsSpawnAtomically();
  maximumChildOrdinalPrecedesExhaustion();
  boundedDequeStealingPreservesLogicalIdentity();
  cancellationRetiresExactlyOneResponsibility();
  queuedRootCancellationCarriesCompletionTransition();
  assignmentAuthorityCannotCrossSchedulers();
  return 0;
}
