#include "Simulator/DynamicWorkDomain.h"

#include "DynamicWorkOrdinal.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

using namespace loom::sim;

// One coordinator owns exactly one domain instance, so the kernel is an
// exclusive owner: it is neither copy nor move constructible or assignable.
static_assert(!std::is_copy_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy constructible");
static_assert(!std::is_copy_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be copy assignable");
static_assert(!std::is_move_constructible<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move constructible");
static_assert(!std::is_move_assignable<DynamicWorkDomain>::value,
              "DynamicWorkDomain must not be move assignable");

namespace {

using Kind = DynamicWorkDomainError::Kind;

constexpr DomainInstanceId kInstance(7);
constexpr DomainInstanceId kForeignInstance(99);

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

// Root identity is exactly (instance, Root, 0); admission acquires the root
// responsibility before publication and closes the root source in the same
// step, so a childless root completes the domain the moment it retires.
void rootIdentityAndImmediateSourceClosure() {
  DynamicWorkDomain domain(kInstance);
  require(domain.activeCount() == 0 && !domain.completed(),
          "a fresh domain is neither active nor completed");

  WorkItemId root = takeExpected(domain.admitRoot());
  require(root == WorkItemId::root(kInstance),
          "the admitted root is not the canonical root identity");
  require(root.instance() == kInstance && root.isRoot() &&
              root.ordinal() == 0 && !root.parent(),
          "the root identity is not (instance, Root, 0)");
  require(domain.activeCount() == 1,
          "root admission did not acquire the responsibility before "
          "publication");
  require(!domain.completed(), "a domain with an active root is completed");

  expectRejected(domain.admitRoot(), Kind::RootAlreadyAdmitted,
                 "the root source admitted a second root");
  require(domain.activeCount() == 1,
          "a rejected second admission changed the active set");

  expectEffect(domain.retire(root), RetirementEffect::DomainCompleted,
               "retiring a childless root did not complete the domain");
  require(domain.completed() && domain.activeCount() == 0,
          "a completed domain is not empty and stable");
}

// A child's ordinal is the zero-based program-order occurrence within its
// parent, and its identity recursively names that parent.
void deterministicChildOrdinals() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());

  WorkItemId first = takeExpected(domain.spawnChild(root));
  WorkItemId second = takeExpected(domain.spawnChild(root));
  WorkItemId third = takeExpected(domain.spawnChild(root));

  require(first == WorkItemId::child(root, 0) &&
              second == WorkItemId::child(root, 1) &&
              third == WorkItemId::child(root, 2),
          "child identities are not a zero-based program-order sequence");
  require(first.ordinal() == 0 && second.ordinal() == 1 && third.ordinal() == 2,
          "a child ordinal does not match its spawn order");
  require(first.parent() == root && second.parent() == root &&
              third.parent() == root,
          "a child identity did not recursively include its parent");
  require(domain.activeCount() == 4,
          "spawning did not acquire one responsibility per child before "
          "publication");
}

// A child responsibility acquired at spawn keeps the domain alive when its
// parent retires first, and the final retirement is the single completion.
void descendantsPreventEarlyCompletion() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());
  WorkItemId child = takeExpected(domain.spawnChild(root));
  require(domain.activeCount() == 2,
          "the child responsibility was not acquired before publication");

  expectEffect(
      domain.retire(root), RetirementEffect::DomainStillActive,
      "retiring the parent completed the domain while a child was active");
  require(!domain.completed() && domain.activeCount() == 1,
          "an active descendant did not keep the domain alive");

  expectEffect(domain.retire(child), RetirementEffect::DomainCompleted,
               "retiring the last descendant did not complete the domain");
  require(domain.completed(), "the drained domain is not completed");
}

// Exactly one retirement produces the completion transition; retiring again is
// rejected and never yields a second completion.
void completionTransitionIsExactlyOnce() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());
  WorkItemId a = takeExpected(domain.spawnChild(root));
  WorkItemId b = takeExpected(domain.spawnChild(root));

  expectEffect(domain.retire(a), RetirementEffect::DomainStillActive,
               "retiring one of several active items completed the domain");
  expectEffect(domain.retire(root), RetirementEffect::DomainStillActive,
               "retiring the root with an active child completed the domain");
  expectEffect(domain.retire(b), RetirementEffect::DomainCompleted,
               "the final retirement did not complete the domain");

  require(domain.completed() && domain.activeCount() == 0,
          "the completed domain is not stable");
  expectRejected(domain.retire(b), Kind::AlreadyRetired,
                 "an item retired twice");
  require(domain.completed() && domain.activeCount() == 0,
          "a rejected retirement disturbed the completed domain");
}

// Foreign, unknown, and already-retired identities are rejected with no
// mutation: no acquired responsibility, no consumed ordinal, no completion.
void rejectsForeignUnknownAndDoubleRetire() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());
  WorkItemId first = takeExpected(domain.spawnChild(root));

  // The root of another domain instance is foreign to this owner, and a child
  // ordinal the domain never handed out was never acquired.
  WorkItemId foreign = WorkItemId::root(kForeignInstance);
  WorkItemId unknown = WorkItemId::child(root, 99);

  expectRejected(domain.spawnChild(foreign), Kind::ForeignDomain,
                 "spawn accepted a foreign parent");
  expectRejected(domain.spawnChild(unknown), Kind::UnknownItem,
                 "spawn accepted an unacquired parent");
  expectRejected(domain.retire(foreign), Kind::ForeignDomain,
                 "retire accepted a foreign identity");
  expectRejected(domain.retire(unknown), Kind::UnknownItem,
                 "retire accepted an unknown identity");
  require(domain.activeCount() == 2 && !domain.completed(),
          "a rejected action changed the active set or completion");

  // The rejected spawns consumed no ordinal: the next real child is ordinal 1.
  WorkItemId second = takeExpected(domain.spawnChild(root));
  require(second == WorkItemId::child(root, 1),
          "a rejected spawn consumed a program-order ordinal");

  // Retirement is exactly once per identity.
  expectEffect(domain.retire(first), RetirementEffect::DomainStillActive,
               "retiring an active item failed");
  expectRejected(domain.retire(first), Kind::AlreadyRetired,
                 "an item was retired twice");
  require(domain.activeCount() == 2,
          "a rejected double retirement changed the active set");
}

// Each parent owns an independent zero-based ordinal sequence, unaffected by
// spawns interleaved from other parents.
void independentOrdinalSequencesForInterleavedParents() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());

  WorkItemId a = takeExpected(domain.spawnChild(root));
  WorkItemId ga = takeExpected(domain.spawnChild(a));
  WorkItemId b = takeExpected(domain.spawnChild(root));
  WorkItemId gb = takeExpected(domain.spawnChild(a));
  WorkItemId c = takeExpected(domain.spawnChild(root));

  require(a.ordinal() == 0 && b.ordinal() == 1 && c.ordinal() == 2,
          "the root's child ordinals are not an independent sequence");
  require(ga.ordinal() == 0 && gb.ordinal() == 1,
          "a nested parent's child ordinals are not an independent sequence");
  require(a.parent() == root && b.parent() == root && c.parent() == root,
          "a root child did not name the root as parent");
  require(ga.parent() == a && gb.parent() == a,
          "a nested child did not name its own parent");
  require(ga == WorkItemId::child(a, 0) && gb == WorkItemId::child(a, 1),
          "nested child identities are not recursively derived");
}

// A retired parent cannot spawn, and rejection neither acquires a child
// responsibility nor consumes the child's first ordinal.
void retiredParentRejectsSpawnAtomically() {
  DynamicWorkDomain domain(kInstance);
  WorkItemId root = takeExpected(domain.admitRoot());
  WorkItemId parent = takeExpected(domain.spawnChild(root));

  expectEffect(domain.retire(parent), RetirementEffect::DomainStillActive,
               "retiring the child parent completed an active domain");
  expectRejected(domain.spawnChild(parent), Kind::AlreadyRetired,
                 "spawn accepted an already-retired parent");
  require(domain.activeCount() == 1 && !domain.completed(),
          "a rejected spawn from a retired parent changed domain state");

  expectRejected(domain.retire(WorkItemId::child(parent, 0)), Kind::UnknownItem,
                 "a rejected spawn consumed an ordinal or acquired a child");
  expectEffect(domain.retire(root), RetirementEffect::DomainCompleted,
               "the remaining root did not complete the domain");
}

// The checked cursor hands out each representable identity at most once, then
// rejects exhaustion without wrapping or mutating its saturated state.
void childOrdinalExhaustionIsAtomic() {
  std::uint64_t next = std::numeric_limits<std::uint64_t>::max() - 1;
  std::optional<std::uint64_t> ordinal = detail::takeChildOrdinal(next);
  require(ordinal && *ordinal == std::numeric_limits<std::uint64_t>::max() - 1,
          "the cursor did not return its final unique child ordinal");
  require(next == std::numeric_limits<std::uint64_t>::max(),
          "the cursor did not advance to its exhausted state");

  ordinal = detail::takeChildOrdinal(next);
  require(!ordinal, "an exhausted cursor wrapped to a duplicate ordinal");
  require(next == std::numeric_limits<std::uint64_t>::max(),
          "an exhausted transition mutated the cursor");

  ordinal = detail::takeChildOrdinal(next);
  require(!ordinal && next == std::numeric_limits<std::uint64_t>::max(),
          "repeated exhaustion changed the saturated cursor");
}

} // namespace

int main() {
  rootIdentityAndImmediateSourceClosure();
  deterministicChildOrdinals();
  descendantsPreventEarlyCompletion();
  completionTransitionIsExactlyOnce();
  rejectsForeignUnknownAndDoubleRetire();
  independentOrdinalSequencesForInterleavedParents();
  retiredParentRejectsSpawnAtomically();
  childOrdinalExhaustionIsAtomic();
  return 0;
}
