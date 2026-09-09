#include "Simulator/MemorySynchronization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace sim {
namespace {

using Kind = MemorySynchronizationError::Kind;
using Graph = std::vector<llvm::SmallVector<SyncEffectId, 2>>;

llvm::Error reject(Kind kind, const llvm::Twine &message) {
  return llvm::make_error<MemorySynchronizationError>(kind, message.str());
}

/// Canonical shape of every returned effect collection: ascending identity,
/// no repeats, so a view never exposes discovery order.
void canonicalize(llvm::SmallVectorImpl<SyncEffectId> &effects) {
  llvm::sort(effects);
  effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
}

void canonicalize(Graph &graph) {
  for (auto &edges : graph)
    canonicalize(edges);
}

bool publishes(SyncRoleKind kind) { return kind != SyncRoleKind::Acquire; }

bool imports(SyncRoleKind kind) { return kind != SyncRoleKind::Release; }

Graph reverseGraph(const Graph &graph) {
  Graph reversed(graph.size());
  for (std::uint64_t from = 0; from < graph.size(); ++from)
    for (SyncEffectId to : graph[from])
      reversed[to.value()].push_back(SyncEffectId(from));
  canonicalize(reversed);
  return reversed;
}

/// The only traversal primitive: everything reachable from one effect, in
/// canonical order. Sequenced-before reachability, happens-before, and every
/// visibility summary are the same walk over a different graph.
llvm::SmallVector<SyncEffectId> reachable(const Graph &graph,
                                          SyncEffectId start) {
  llvm::SmallVector<SyncEffectId> found;
  std::vector<bool> seen(graph.size(), false);
  llvm::SmallVector<SyncEffectId> worklist{start};
  seen[start.value()] = true;
  while (!worklist.empty()) {
    const SyncEffectId node = worklist.pop_back_val();
    for (const SyncEffectId next : graph[node.value()]) {
      if (seen[next.value()])
        continue;
      seen[next.value()] = true;
      found.push_back(next);
      worklist.push_back(next);
    }
  }
  canonicalize(found);
  return found;
}

/// Iterative depth-first search over the proposed relation graph. A back edge
/// to an effect still on the stack is a cycle.
bool hasCycle(const Graph &graph) {
  enum class Mark : std::uint8_t { Unseen, Active, Done };
  std::vector<Mark> marks(graph.size(), Mark::Unseen);
  llvm::SmallVector<std::pair<std::uint64_t, std::size_t>> stack;

  for (std::uint64_t root = 0; root < graph.size(); ++root) {
    if (marks[root] != Mark::Unseen)
      continue;
    marks[root] = Mark::Active;
    stack.push_back({root, 0});
    while (!stack.empty()) {
      const std::uint64_t node = stack.back().first;
      const std::size_t index = stack.back().second;
      if (index >= graph[node].size()) {
        marks[node] = Mark::Done;
        stack.pop_back();
        continue;
      }
      stack.back().second = index + 1;
      const SyncEffectId next = graph[node][index];
      if (marks[next.value()] == Mark::Active)
        return true;
      if (marks[next.value()] == Mark::Unseen) {
        marks[next.value()] = Mark::Active;
        stack.push_back({next.value(), 0});
      }
    }
  }
  return false;
}

} // namespace

llvm::SmallVector<SyncEffectId>
MemorySynchronization::maximalCandidates(
    llvm::ArrayRef<SyncEffectId> candidates, const Graph &predecessors,
    bool declarationOrderIsTopological) const {
  llvm::SmallVector<SyncEffectId> accepted(candidates);
  canonicalize(accepted);
  if (accepted.size() < 2)
    return accepted;

  llvm::SmallVector<bool> dominated(accepted.size(), false);
  std::size_t remaining = accepted.size();
  beginTraversal(predecessors.size());
  for (SyncEffectId effect : accepted) {
    markVisited(effect);
    traversalWorklist_.push_back(effect);
  }
  while (!traversalWorklist_.empty() && remaining > 1) {
    const SyncEffectId effect = traversalWorklist_.pop_back_val();
    for (SyncEffectId predecessor : predecessors[effect.value()]) {
      // When declaration order is a verified topological order, the prefix
      // before the oldest candidate cannot contain another candidate ancestor.
      if (declarationOrderIsTopological && predecessor < accepted.front())
        continue;
      auto candidate = llvm::lower_bound(accepted, predecessor);
      if (candidate != accepted.end() && *candidate == predecessor) {
        const std::size_t ordinal = candidate - accepted.begin();
        if (!dominated[ordinal]) {
          dominated[ordinal] = true;
          --remaining;
        }
      }
      if (markVisited(predecessor))
        traversalWorklist_.push_back(predecessor);
    }
  }
  traversalWorklist_.clear();
  llvm::SmallVector<SyncEffectId> maximal;
  for (auto [ordinal, effect] : llvm::enumerate(accepted))
    if (!dominated[ordinal])
      maximal.push_back(effect);
  return maximal;
}

MemorySynchronization::Graph MemorySynchronization::transitivelyReduced(
    const Graph &graph, bool declarationOrderIsTopological) const {
  const Graph predecessors = reverseGraph(graph);
  Graph reduced(graph.size());
  for (std::uint64_t target = 0; target < predecessors.size(); ++target)
    for (SyncEffectId predecessor : maximalCandidates(
             predecessors[target], predecessors, declarationOrderIsTopological))
      reduced[predecessor.value()].push_back(SyncEffectId(target));
  return reduced;
}

char MemorySynchronizationError::ID = 0;

MemorySynchronizationError::MemorySynchronizationError(Kind kind,
                                                       std::string message)
    : kind_(kind), message_(std::move(message)) {}

void MemorySynchronizationError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code MemorySynchronizationError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

const MemorySynchronization::Carrier *
MemorySynchronization::carrierOf(const Facts &facts,
                                 SyncEffectId effect) const {
  auto carrier = facts.carriers.find(effect.value());
  return carrier == facts.carriers.end() ? nullptr : &carrier->second;
}

const MemorySynchronization::Role *
MemorySynchronization::roleOf(const Facts &facts, SyncEffectId effect) const {
  auto role = facts.roles.find(effect.value());
  return role == facts.roles.end() ? nullptr : &role->second;
}

std::optional<SyncDomainId>
MemorySynchronization::domainOf(const Facts &facts, SyncEffectId effect) const {
  if (const Role *role = roleOf(facts, effect))
    if (role->fenceDomain)
      return *role->fenceDomain;
  if (const Carrier *carrier = carrierOf(facts, effect))
    return carrier->domain;
  return std::nullopt;
}

bool MemorySynchronization::sequencedReaches(const Facts &facts,
                                             SyncEffectId from,
                                             SyncEffectId to) const {
  return reaches(facts.sequenced, from, to);
}

void MemorySynchronization::beginTraversal(std::size_t effectCount) const {
  if (traversalMarks_.size() < effectCount)
    traversalMarks_.resize(effectCount, 0);
  ++traversalGeneration_;
  if (traversalGeneration_ == 0) {
    std::fill(traversalMarks_.begin(), traversalMarks_.end(), 0);
    traversalGeneration_ = 1;
  }
  traversalWorklist_.clear();
}

bool MemorySynchronization::markVisited(SyncEffectId effect) const {
  std::uint32_t &mark = traversalMarks_[effect.value()];
  if (mark == traversalGeneration_)
    return false;
  mark = traversalGeneration_;
  return true;
}

bool MemorySynchronization::reaches(const Graph &graph, SyncEffectId start,
                                    SyncEffectId target) const {
  if (start == target)
    return false;
  if (llvm::is_contained(graph[start.value()], target))
    return true;

  beginTraversal(graph.size());
  markVisited(start);
  traversalWorklist_.push_back(start);
  while (!traversalWorklist_.empty()) {
    const SyncEffectId node = traversalWorklist_.pop_back_val();
    for (const SyncEffectId next : graph[node.value()]) {
      if (next == target) {
        traversalWorklist_.clear();
        return true;
      }
      if (!markVisited(next))
        continue;
      traversalWorklist_.push_back(next);
    }
  }
  return false;
}

llvm::SmallVector<SyncEffectId>
MemorySynchronization::collectOrigins(const Facts &facts,
                                      AtomicVersionId version) const {
  llvm::SmallVector<SyncEffectId> origins;
  auto owner = facts.versionOwner.find(version.value());
  if (owner == facts.versionOwner.end())
    return origins;

  // Every hop of a release sequence must resolve to the domain of the version
  // an acquire actually read. A hop in another domain ends the walk, so a later
  // hop that returns to this domain cannot erase the break.
  const SyncDomainId domain = facts.carriers.at(owner->second.value()).domain;
  std::optional<AtomicVersionId> current = version;
  while (current) {
    auto hop = facts.versionOwner.find(current->value());
    if (hop == facts.versionOwner.end())
      break;
    const SyncEffectId writer = hop->second;
    const Carrier &carrier = facts.carriers.at(writer.value());
    if (carrier.domain != domain)
      break;

    // A release operation publishes through its own write only.
    if (const Role *role = roleOf(facts, writer))
      if (!role->fenceDomain && publishes(role->kind))
        origins.push_back(writer);
    // A release fence publishes through a write sequenced after it.
    for (const auto &[value, fence] : facts.roles) {
      if (!fence.fenceDomain || *fence.fenceDomain != domain)
        continue;
      if (!publishes(fence.kind))
        continue;
      const SyncEffectId candidate(value);
      if (sequencedReaches(facts, candidate, writer))
        origins.push_back(candidate);
    }

    // A carried write continues the release sequence through the version it
    // read, which validation pinned to its exact predecessor. The walk
    // therefore strictly descends the modification order and terminates.
    if (!carrier.read)
      break;
    std::optional<AtomicReadRecord> record = order_->readRecord(*carrier.read);
    if (!record)
      break;
    current = record->version();
  }

  canonicalize(origins);
  return origins;
}

void MemorySynchronization::forEachSynchronization(
    const Facts &facts,
    llvm::function_ref<void(SyncEffectId, SyncEffectId)> action) const {
  for (const auto &[value, carrier] : facts.carriers) {
    if (!carrier.read)
      continue;
    std::optional<AtomicReadRecord> record = order_->readRecord(*carrier.read);
    if (!record)
      continue;
    const llvm::SmallVector<SyncEffectId> origins =
        collectOrigins(facts, record->version());
    if (origins.empty())
      continue;

    const SyncEffectId reader(value);
    llvm::SmallVector<SyncEffectId> targets;
    // An acquire operation imports through its own read only.
    if (const Role *role = roleOf(facts, reader))
      if (!role->fenceDomain && imports(role->kind))
        targets.push_back(reader);
    // An acquire fence imports through a read sequenced before it.
    for (const auto &[fenceValue, fence] : facts.roles) {
      if (!fence.fenceDomain || *fence.fenceDomain != carrier.domain)
        continue;
      if (!imports(fence.kind))
        continue;
      const SyncEffectId candidate(fenceValue);
      if (sequencedReaches(facts, reader, candidate))
        targets.push_back(candidate);
    }
    if (targets.empty())
      continue;

    for (const SyncEffectId origin : origins) {
      if (domainOf(facts, origin) != carrier.domain)
        continue;
      for (const SyncEffectId target : targets)
        action(origin, target);
    }
  }
}

MemorySynchronization::Graph
MemorySynchronization::buildGraph(const Facts &facts) const {
  Graph graph(facts.effects);
  auto link = [&](SyncEffectId from, SyncEffectId to) {
    graph[from.value()].push_back(to);
  };
  for (std::uint64_t value = 0; value < facts.sequenced.size(); ++value)
    for (const SyncEffectId later : facts.sequenced[value])
      link(SyncEffectId(value), later);
  forEachSynchronization(facts, link);
  canonicalize(graph);
  return graph;
}

llvm::Error MemorySynchronization::requireKnown(SyncEffectId effect) const {
  if (effect.value() >= facts_.effects)
    return reject(Kind::UnknownEffect, "effect " + llvm::Twine(effect.value()) +
                                           " was never declared");
  return llvm::Error::success();
}

llvm::Error
MemorySynchronization::requireNoFenceRole(SyncEffectId effect) const {
  const Role *role = roleOf(facts_, effect);
  if (role && role->fenceDomain)
    return reject(Kind::RoleShapeConflict,
                  "effect " + llvm::Twine(effect.value()) +
                      " has a fence role and cannot carry an addressed access");
  return llvm::Error::success();
}

llvm::Error MemorySynchronization::commit(Facts candidate,
                                          bool reduceSequenced) {
  Graph relation = buildGraph(candidate);
  if (hasCycle(relation))
    return reject(Kind::CyclicOrder,
                  "the update closes a happens-before cycle");
  bool declarationOrderIsTopological = true;
  for (auto [ordinal, successors] : llvm::enumerate(relation))
    for (SyncEffectId successor : successors)
      declarationOrderIsTopological &= ordinal < successor.value();
  if (reduceSequenced) {
    candidate.sequenced =
        transitivelyReduced(candidate.sequenced, declarationOrderIsTopological);
    relation = buildGraph(candidate);
  }
  Graph sequencedPredecessors = reverseGraph(candidate.sequenced);
  relation = transitivelyReduced(relation, declarationOrderIsTopological);
  Graph predecessors = reverseGraph(relation);
  coverageRoot_.reset();
  coverageAncestors_.clear();
  facts_ = std::move(candidate);
  sequencedPredecessors_ = std::move(sequencedPredecessors);
  declarationOrderIsTopological_ = declarationOrderIsTopological;
  relation_ = std::move(relation);
  predecessors_ = std::move(predecessors);
  return llvm::Error::success();
}

llvm::Expected<SyncEffectId> MemorySynchronization::declareEffectSequencedAfter(
    llvm::ArrayRef<SyncEffectId> predecessors) {
  llvm::SmallVector<SyncEffectId, 2> accepted(predecessors);
  for (SyncEffectId predecessor : accepted)
    if (llvm::Error error = requireKnown(predecessor))
      return std::move(error);
  llvm::sort(accepted);
  auto duplicate = std::adjacent_find(accepted.begin(), accepted.end());
  if (duplicate != accepted.end())
    return reject(Kind::DuplicateEdge,
                  "effect " + llvm::Twine(duplicate->value()) +
                      " occurs more than once in an incoming frontier");

  accepted = maximalCandidates(accepted, sequencedPredecessors_,
                                declarationOrderIsTopological_);
  const auto relationPredecessors = maximalCandidates(
      accepted, predecessors_, declarationOrderIsTopological_);

  const SyncEffectId effect(facts_.effects);
  ++facts_.effects;
  facts_.sequenced.emplace_back();
  sequencedPredecessors_.emplace_back();
  relation_.emplace_back();
  predecessors_.emplace_back();
  for (SyncEffectId predecessor : accepted) {
    facts_.sequenced[predecessor.value()].push_back(effect);
    sequencedPredecessors_[effect.value()].push_back(predecessor);
  }
  for (SyncEffectId predecessor : relationPredecessors) {
    relation_[predecessor.value()].push_back(effect);
    predecessors_[effect.value()].push_back(predecessor);
  }
  return effect;
}

SyncEffectId MemorySynchronization::declareEffect() {
  return llvm::cantFail(declareEffectSequencedAfter({}));
}

llvm::Error MemorySynchronization::sequencedBefore(SyncEffectId earlier,
                                                   SyncEffectId later) {
  if (llvm::Error error = requireKnown(earlier))
    return error;
  if (llvm::Error error = requireKnown(later))
    return error;
  if (earlier == later)
    return reject(Kind::DuplicateEdge, "effect " +
                                           llvm::Twine(earlier.value()) +
                                           " cannot precede itself");

  Facts candidate = facts_;
  auto &successors = candidate.sequenced[earlier.value()];
  if (llvm::is_contained(successors, later))
    return reject(Kind::DuplicateEdge,
                  "effect " + llvm::Twine(earlier.value()) +
                      " already precedes " + llvm::Twine(later.value()));
  successors.push_back(later);
  return commit(std::move(candidate), /*reduceSequenced=*/true);
}

llvm::Error
MemorySynchronization::registerWrite(SyncEffectId effect, SyncDomainId domain,
                                     AtomicVersionId version,
                                     std::optional<AtomicReadId> readsFrom) {
  if (llvm::Error error = requireKnown(effect))
    return error;
  std::optional<AtomicVersionRecord> appended = order_->versionRecord(version);
  if (!appended)
    return reject(Kind::ForeignRelation,
                  "version " + llvm::Twine(version.value()) +
                      " is not a version of the bound atomic order");
  std::optional<AtomicReadRecord> carried;
  if (readsFrom) {
    carried = order_->readRecord(*readsFrom);
    if (!carried)
      return reject(Kind::ForeignRelation,
                    "read " + llvm::Twine(readsFrom->value()) +
                        " is not a read of the bound atomic order");
  }
  if (!appended->predecessor())
    return reject(Kind::InitialVersionPublication,
                  "version " + llvm::Twine(version.value()) +
                      " is an initial version and has no publishing effect");
  if (carried && (carried->key() != appended->key() ||
                  carried->version() != *appended->predecessor()))
    return reject(Kind::MismatchedCarry,
                  "read " + llvm::Twine(readsFrom->value()) +
                      " does not select the predecessor of version " +
                      llvm::Twine(version.value()));
  if (facts_.carriers.count(effect.value()))
    return reject(Kind::DuplicateAssociation, "effect " +
                                                  llvm::Twine(effect.value()) +
                                                  " already has a carrier");
  if (facts_.versionOwner.count(version.value()))
    return reject(Kind::DuplicateAssociation,
                  "version " + llvm::Twine(version.value()) +
                      " already has a publishing effect");
  if (readsFrom && facts_.readOwner.count(readsFrom->value()))
    return reject(Kind::DuplicateAssociation,
                  "read " + llvm::Twine(readsFrom->value()) +
                      " already has a reading effect");
  if (llvm::Error error = requireNoFenceRole(effect))
    return error;

  Facts candidate = facts_;
  candidate.carriers.insert(
      {effect.value(), Carrier{domain, version, readsFrom}});
  candidate.versionOwner.insert({version.value(), effect});
  if (readsFrom)
    candidate.readOwner.insert({readsFrom->value(), effect});
  return commit(std::move(candidate));
}

llvm::Error MemorySynchronization::registerRead(SyncEffectId effect,
                                                SyncDomainId domain,
                                                AtomicReadId read) {
  if (llvm::Error error = requireKnown(effect))
    return error;
  if (!order_->readRecord(read))
    return reject(Kind::ForeignRelation,
                  "read " + llvm::Twine(read.value()) +
                      " is not a read of the bound atomic order");
  if (facts_.carriers.count(effect.value()))
    return reject(Kind::DuplicateAssociation, "effect " +
                                                  llvm::Twine(effect.value()) +
                                                  " already has a carrier");
  if (facts_.readOwner.count(read.value()))
    return reject(Kind::DuplicateAssociation,
                  "read " + llvm::Twine(read.value()) +
                      " already has a reading effect");
  if (llvm::Error error = requireNoFenceRole(effect))
    return error;

  Facts candidate = facts_;
  candidate.carriers.insert(
      {effect.value(), Carrier{domain, std::nullopt, read}});
  candidate.readOwner.insert({read.value(), effect});
  return commit(std::move(candidate));
}

llvm::Error MemorySynchronization::declareOperationRole(SyncEffectId effect,
                                                        SyncRoleKind kind) {
  if (llvm::Error error = requireKnown(effect))
    return error;
  if (facts_.roles.count(effect.value()))
    return reject(Kind::DuplicateRole, "effect " + llvm::Twine(effect.value()) +
                                           " already has a role");
  const Carrier *carrier = carrierOf(facts_, effect);
  const bool publishable = carrier && carrier->version;
  const bool importable = carrier && carrier->read;
  const bool compatible =
      kind == SyncRoleKind::Release
          ? publishable
          : (kind == SyncRoleKind::Acquire ? importable
                                           : publishable && importable);
  if (!compatible)
    return reject(Kind::RoleShapeConflict,
                  "effect " + llvm::Twine(effect.value()) +
                      " has no carrier matching this operation role");

  Facts candidate = facts_;
  candidate.roles.insert({effect.value(), Role{kind, std::nullopt}});
  return commit(std::move(candidate));
}

llvm::Error MemorySynchronization::declareFenceRole(SyncEffectId effect,
                                                    SyncRoleKind kind,
                                                    SyncDomainId domain) {
  if (llvm::Error error = requireKnown(effect))
    return error;
  if (facts_.roles.count(effect.value()))
    return reject(Kind::DuplicateRole, "effect " + llvm::Twine(effect.value()) +
                                           " already has a role");
  if (facts_.carriers.count(effect.value()))
    return reject(Kind::RoleShapeConflict,
                  "effect " + llvm::Twine(effect.value()) +
                      " has an addressed carrier and cannot be a fence");

  Facts candidate = facts_;
  candidate.roles.insert({effect.value(), Role{kind, domain}});
  return commit(std::move(candidate));
}

llvm::Error
MemorySynchronization::appendSequentiallyConsistent(SyncEffectId effect,
                                                    SyncDomainId domain) {
  if (llvm::Error error = requireKnown(effect))
    return error;
  if (facts_.sequentiallyConsistent.count(effect.value()))
    return reject(Kind::DuplicateSequentiallyConsistentAssociation,
                  "effect " + llvm::Twine(effect.value()) +
                      " already belongs to a sequentially-consistent order");

  Facts candidate = facts_;
  std::optional<SyncEffectId> predecessor;
  auto tail = candidate.sequentiallyConsistentTails.find(domain.value());
  if (tail != candidate.sequentiallyConsistentTails.end())
    predecessor = tail->second;
  candidate.sequentiallyConsistent.insert(
      {effect.value(), SequentiallyConsistentAssociation{domain, predecessor}});
  candidate.sequentiallyConsistentTails.insert_or_assign(domain.value(),
                                                         effect);
  return commit(std::move(candidate));
}

llvm::Expected<std::optional<SyncEffectId>>
MemorySynchronization::sequentiallyConsistentPredecessor(
    SyncEffectId effect) const {
  if (llvm::Error error = requireKnown(effect))
    return std::move(error);
  auto association = facts_.sequentiallyConsistent.find(effect.value());
  if (association == facts_.sequentiallyConsistent.end())
    return reject(Kind::UnknownSequentiallyConsistentAssociation,
                  "effect " + llvm::Twine(effect.value()) +
                      " does not belong to a sequentially-consistent order");
  return association->second.predecessor;
}

bool MemorySynchronization::synchronizesWith(SyncEffectId origin,
                                             SyncEffectId target) const {
  bool found = false;
  forEachSynchronization(facts_, [&](SyncEffectId from, SyncEffectId to) {
    found |= from == origin && to == target;
  });
  return found;
}

bool MemorySynchronization::happensBefore(SyncEffectId earlier,
                                          SyncEffectId later) const {
  if (earlier.value() >= facts_.effects || later.value() >= facts_.effects)
    return false;
  return reaches(relation_, earlier, later);
}

bool MemorySynchronization::areCoveredByHappensBefore(
    llvm::ArrayRef<SyncEffectId> effects,
    llvm::ArrayRef<SyncEffectId> frontier) const {
  if (effects.empty())
    return true;
  if (frontier.empty())
    return false;
  if (llvm::any_of(effects,
                   [&](SyncEffectId effect) {
                     return effect.value() >= facts_.effects;
                   }) ||
      llvm::any_of(frontier, [&](SyncEffectId effect) {
        return effect.value() >= facts_.effects;
      }))
    return false;
  if (effects.size() == 1 && frontier.size() == 1)
    return effects.front() == frontier.front() ||
           happensBefore(effects.front(), frontier.front());

  llvm::SmallVector<SyncEffectId, 4> seeds(frontier);
  canonicalize(seeds);
  auto isSeed = [&](SyncEffectId effect) {
    return std::binary_search(seeds.begin(), seeds.end(), effect);
  };

  // Settle from each requested effect's own outgoing edges first. Plain-memory
  // admission asks this question once per ready access, about the hazards of
  // the bytes that access covers under the ctrl frontier it has just
  // inherited, so a hazard is normally a frontier member or one reduced edge
  // behind one. Deciding those here keeps the query proportional to itself
  // rather than to every effect the run has declared.
  llvm::SmallVector<SyncEffectId, 4> undecided;
  for (SyncEffectId effect : effects) {
    if (isSeed(effect) || llvm::any_of(relation_[effect.value()], isSeed))
      continue;
    undecided.push_back(effect);
  }
  if (undecided.empty())
    return true;

  // A distant predecessor needs a reverse traversal, but a successful query
  // is complete as soon as every requested effect has been reached.
  canonicalize(undecided);
  std::size_t remaining = undecided.size();
  beginTraversal(facts_.effects);
  llvm::SmallVector<SyncEffectId, 16> visitedEffects;
  llvm::SmallVector<bool, 4> covered(undecided.size(), false);
  bool reachedCachedRoot = false;
  // Only a long traversal amortizes retaining its proof. These bits name
  // ancestors actually reached from this one frontier, never inferred order.
  constexpr std::size_t kCoverageMemoMinimumVisits = 4096;
  llvm::scope_exit rememberCoverage([&] {
    if (seeds.size() != 1 || visitedEffects.size() < kCoverageMemoMinimumVisits)
      return;
    if (!reachedCachedRoot)
      coverageAncestors_.reset();
    coverageAncestors_.resize(facts_.effects);
    for (SyncEffectId effect : visitedEffects)
      coverageAncestors_.set(effect.value());
    coverageRoot_ = seeds.front();
  });
  auto cover = [&](std::size_t ordinal) {
    if (!covered[ordinal]) {
      covered[ordinal] = true;
      --remaining;
    }
  };
  for (SyncEffectId effect : seeds) {
    if (markVisited(effect)) {
      traversalWorklist_.push_back(effect);
      visitedEffects.push_back(effect);
    }
  }
  while (!traversalWorklist_.empty()) {
    SyncEffectId effect = traversalWorklist_.pop_back_val();
    if (coverageRoot_ && effect == *coverageRoot_) {
      reachedCachedRoot = true;
      for (auto [ordinal, requested] : llvm::enumerate(undecided))
        if (requested.value() < coverageAncestors_.size() &&
            coverageAncestors_.test(requested.value()))
          cover(ordinal);
      if (remaining == 0) {
        traversalWorklist_.clear();
        return true;
      }
    }
    for (SyncEffectId predecessor : predecessors_[effect.value()]) {
      if (declarationOrderIsTopological_ && predecessor < undecided.front())
        continue;
      if (!markVisited(predecessor))
        continue;
      visitedEffects.push_back(predecessor);
      auto requested = llvm::lower_bound(undecided, predecessor);
      if (requested != undecided.end() && *requested == predecessor) {
        cover(requested - undecided.begin());
        if (remaining == 0) {
          traversalWorklist_.clear();
          return true;
        }
      }
      traversalWorklist_.push_back(predecessor);
    }
  }
  return false;
}

llvm::Expected<llvm::SmallVector<SyncEffectId>>
MemorySynchronization::maximalHappensBeforeFrontier(
    llvm::ArrayRef<SyncEffectId> effects) const {
  for (SyncEffectId effect : effects)
    if (llvm::Error error = requireKnown(effect))
      return std::move(error);
  // A preceding coverage query often proved the same old hazards before
  // this newly declared write. Reuse that proof only after reaching its root
  // from the proposed maximal effect; an unrelated frontier proves nothing.
  if (coverageRoot_ && effects.size() > 1) {
    const SyncEffectId newest = *llvm::max_element(effects);
    if (llvm::all_of(effects, [&](SyncEffectId effect) {
          return effect == newest ||
                 (effect.value() < coverageAncestors_.size() &&
                  coverageAncestors_.test(effect.value()));
        }) && areCoveredByHappensBefore(effects, {newest}))
      return llvm::SmallVector<SyncEffectId>{newest};
  }
  return maximalCandidates(effects, predecessors_,
                            declarationOrderIsTopological_);
}

llvm::Expected<llvm::SmallVector<SyncEffectId>>
MemorySynchronization::publishedOrigins(AtomicVersionId version) const {
  if (!order_->versionRecord(version))
    return reject(Kind::ForeignRelation,
                  "version " + llvm::Twine(version.value()) +
                      " is not a version of the bound atomic order");
  return collectOrigins(facts_, version);
}

llvm::Expected<llvm::SmallVector<SyncEffectId>>
MemorySynchronization::visibilitySummary(SyncEffectId origin) const {
  if (llvm::Error error = requireKnown(origin))
    return std::move(error);
  const Role *role = roleOf(facts_, origin);
  if (!role || !publishes(role->kind))
    return reject(Kind::UnknownRole, "effect " + llvm::Twine(origin.value()) +
                                         " has no release role and publishes "
                                         "no summary");
  return reachable(predecessors_, origin);
}

llvm::Expected<llvm::SmallVector<SyncEffectId>>
MemorySynchronization::importedVisibility(SyncEffectId target) const {
  if (llvm::Error error = requireKnown(target))
    return std::move(error);
  const Role *role = roleOf(facts_, target);
  if (!role || !imports(role->kind))
    return reject(Kind::UnknownRole, "effect " + llvm::Twine(target.value()) +
                                         " has no acquire role and imports no "
                                         "summary");

  llvm::SmallVector<SyncEffectId> imported;
  forEachSynchronization(facts_, [&](SyncEffectId origin, SyncEffectId into) {
    if (into != target)
      return;
    imported.push_back(origin);
    llvm::append_range(imported, reachable(predecessors_, origin));
  });
  canonicalize(imported);
  return imported;
}

} // namespace sim
} // namespace loom
