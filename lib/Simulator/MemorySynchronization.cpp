#include "Simulator/MemorySynchronization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace sim {
namespace {

using Kind = MemorySynchronizationError::Kind;
using Graph = std::map<std::uint64_t, llvm::SmallVector<SyncEffectId, 2>>;

llvm::Error reject(Kind kind, const llvm::Twine &message) {
  return llvm::make_error<MemorySynchronizationError>(kind, message.str());
}

/// Canonical shape of every returned effect collection: ascending identity,
/// no repeats, so a view never exposes discovery order.
void canonicalize(llvm::SmallVectorImpl<SyncEffectId> &effects) {
  llvm::sort(effects);
  effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
}

bool publishes(SyncRoleKind kind) { return kind != SyncRoleKind::Acquire; }

bool imports(SyncRoleKind kind) { return kind != SyncRoleKind::Release; }

/// The only traversal primitive: everything reachable from one effect, in
/// canonical order. Sequenced-before reachability, happens-before, and every
/// visibility summary are the same walk over a different graph.
llvm::SmallVector<SyncEffectId>
reachable(const Graph &graph, SyncEffectId start, std::uint64_t effects) {
  llvm::SmallVector<SyncEffectId> found;
  std::vector<bool> seen(effects, false);
  llvm::SmallVector<SyncEffectId> worklist{start};
  seen[start.value()] = true;
  while (!worklist.empty()) {
    const SyncEffectId node = worklist.pop_back_val();
    auto edges = graph.find(node.value());
    if (edges == graph.end())
      continue;
    for (const SyncEffectId next : edges->second) {
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
bool hasCycle(const Graph &graph, std::uint64_t effects) {
  enum class Mark : std::uint8_t { Unseen, Active, Done };
  std::vector<Mark> marks(effects, Mark::Unseen);
  llvm::SmallVector<std::pair<std::uint64_t, std::size_t>> stack;

  for (std::uint64_t root = 0; root < effects; ++root) {
    if (marks[root] != Mark::Unseen)
      continue;
    marks[root] = Mark::Active;
    stack.push_back({root, 0});
    while (!stack.empty()) {
      const std::uint64_t node = stack.back().first;
      const std::size_t index = stack.back().second;
      auto edges = graph.find(node);
      if (edges == graph.end() || index >= edges->second.size()) {
        marks[node] = Mark::Done;
        stack.pop_back();
        continue;
      }
      stack.back().second = index + 1;
      const SyncEffectId next = edges->second[index];
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
  return llvm::is_contained(reachable(facts.sequenced, from, facts.effects),
                            to);
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
MemorySynchronization::buildGraph(const Facts &facts, bool reversed) const {
  Graph graph;
  auto link = [&](SyncEffectId from, SyncEffectId to) {
    if (reversed)
      std::swap(from, to);
    graph[from.value()].push_back(to);
  };
  for (const auto &[value, laters] : facts.sequenced)
    for (const SyncEffectId later : laters)
      link(SyncEffectId(value), later);
  forEachSynchronization(facts, link);
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

llvm::Error MemorySynchronization::commit(Facts candidate) {
  if (hasCycle(buildGraph(candidate, /*reversed=*/false), candidate.effects))
    return reject(Kind::CyclicOrder,
                  "the update closes a happens-before cycle");
  facts_ = std::move(candidate);
  return llvm::Error::success();
}

SyncEffectId MemorySynchronization::declareEffect() {
  return SyncEffectId(facts_.effects++);
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
  auto existing = facts_.sequenced.find(earlier.value());
  if (existing != facts_.sequenced.end() &&
      llvm::is_contained(existing->second, later))
    return reject(Kind::DuplicateEdge,
                  "effect " + llvm::Twine(earlier.value()) +
                      " already precedes " + llvm::Twine(later.value()));

  Facts candidate = facts_;
  candidate.sequenced[earlier.value()].push_back(later);
  return commit(std::move(candidate));
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
  return llvm::is_contained(reachable(buildGraph(facts_, /*reversed=*/false),
                                      earlier, facts_.effects),
                            later);
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
  return reachable(buildGraph(facts_, /*reversed=*/true), origin,
                   facts_.effects);
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

  const Graph predecessors = buildGraph(facts_, /*reversed=*/true);
  llvm::SmallVector<SyncEffectId> imported;
  forEachSynchronization(facts_, [&](SyncEffectId origin, SyncEffectId into) {
    if (into != target)
      return;
    imported.push_back(origin);
    llvm::append_range(imported,
                       reachable(predecessors, origin, facts_.effects));
  });
  canonicalize(imported);
  return imported;
}

} // namespace sim
} // namespace loom
