#include "Simulator/DynamicWorkDomain.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <type_traits>
#include <utility>

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

static_assert(!std::is_copy_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy constructible");
static_assert(!std::is_copy_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy assignable");
static_assert(!std::is_move_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move constructible");
static_assert(!std::is_move_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move assignable");

static_assert(
    !std::is_invocable<decltype(&DynamicWorkDomain::spawnChild),
                       DynamicWorkDomain &, const WorkItemId &>::value,
    "WorkItemId must not authorize child spawn");
static_assert(!std::is_invocable<decltype(&DynamicWorkDomain::retire),
                                 DynamicWorkDomain &, WorkItemId &&>::value,
              "WorkItemId must not authorize retirement");

namespace {

using Kind = DynamicWorkDomainError::Kind;

constexpr DomainInstanceId kInstance(7);

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

void rootIdentityAndImmediateSourceClosure() {
  DynamicWorkDomain domain(kInstance);
  require(domain.activeCount() == 0 && !domain.completed(),
          "a fresh domain is neither active nor completed");

  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();
  require(rootId == WorkItemId::root(kInstance),
          "the admitted root is not the canonical root identity");
  require(rootId.instance() == kInstance && rootId.isRoot() &&
              rootId.ordinal() == 0 && !rootId.parent(),
          "the root identity is not (instance, Root, 0)");
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
  DynamicWorkDomain domain(kInstance);
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
  DynamicWorkDomain domain(kInstance);
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
  DynamicWorkDomain domain(kInstance);
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
  DynamicWorkDomain domain(kInstance);
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

void observationsAndSameIdDomainsCannotAuthorize() {
  DynamicWorkDomain left(kInstance);
  DynamicWorkDomain right(kInstance);
  WorkResponsibility leftRoot = takeExpected(left.admitRoot());
  WorkResponsibility rightRoot = takeExpected(right.admitRoot());

  WorkItemId predictedRoot = WorkItemId::root(kInstance);
  WorkItemId predictedChild = WorkItemId::child(predictedRoot, 0);
  require(leftRoot.id() == predictedRoot && rightRoot.id() == predictedRoot,
          "same-id domains do not expose the same root observation");

  expectRejected(left.spawnChild(rightRoot), Kind::ForeignDomain,
                 "a same-id foreign capability authorized child spawn");
  expectRejected(left.retire(std::move(rightRoot)), Kind::ForeignDomain,
                 "a same-id foreign capability authorized retirement");
  require(left.activeCount() == 1 && right.activeCount() == 1 &&
              !left.completed() && !right.completed(),
          "cross-domain rejection changed either active set");

  WorkResponsibility leftChild = takeExpected(left.spawnChild(leftRoot));
  WorkResponsibility rightChild = takeExpected(right.spawnChild(rightRoot));
  require(leftChild.id() == predictedChild && rightChild.id() == predictedChild,
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
  DynamicWorkDomain domain(kInstance);
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
  DynamicWorkDomain domain(kInstance);
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

void childOrdinalExhaustionIsAtomic() {
  DynamicWorkDomain domain(kInstance);
  WorkResponsibility root = takeExpected(domain.admitRoot());
  WorkItemId rootId = root.id();
  WorkResponsibility first = takeExpected(domain.spawnChild(root));
  WorkResponsibility second = takeExpected(domain.spawnChild(root));
  WorkResponsibility third = takeExpected(domain.spawnChild(root));

  require(first.id() == WorkItemId::child(rootId, 0) &&
              second.id() == WorkItemId::child(rootId, 1) &&
              third.id() == WorkItemId::child(rootId, 2),
          "the public domain did not issue each unique test ordinal");
  require(domain.activeCount() == 4 && !domain.completed(),
          "the exhaustion setup has the wrong responsibility state");

  expectRejected(domain.spawnChild(root), Kind::ChildOrdinalExhausted,
                 "an exhausted parent published another child");
  require(domain.activeCount() == 4 && !domain.completed(),
          "ordinal exhaustion changed active state or completion");
  expectRejected(domain.spawnChild(root), Kind::ChildOrdinalExhausted,
                 "repeated exhaustion published another child");
  require(domain.activeCount() == 4 && !domain.completed(),
          "repeated exhaustion changed active state or completion");

  expectEffect(domain.retire(std::move(first)),
               RetirementEffect::DomainStillActive,
               "first child retirement completed the domain");
  expectEffect(domain.retire(std::move(second)),
               RetirementEffect::DomainStillActive,
               "second child retirement completed the domain");
  expectEffect(domain.retire(std::move(third)),
               RetirementEffect::DomainStillActive,
               "third child retirement completed the domain");
  expectEffect(domain.retire(std::move(root)),
               RetirementEffect::DomainCompleted,
               "exhaustion created an untracked responsibility");
}

} // namespace

int main() {
  rootIdentityAndImmediateSourceClosure();
  deterministicChildOrdinals();
  descendantsPreventEarlyCompletion();
  completionTransitionIsExactlyOnce();
  movedFromCapabilityRejectsAtomically();
  observationsAndSameIdDomainsCannotAuthorize();
  independentOrdinalSequencesForInterleavedParents();
  retiredParentRejectsSpawnAtomically();
  childOrdinalExhaustionIsAtomic();
  return 0;
}
