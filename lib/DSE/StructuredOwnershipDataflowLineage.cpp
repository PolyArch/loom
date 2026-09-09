#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

using detail::ArtifactReferenceSet;
using detail::CanonicalDerivationIndex;
using detail::CanonicalDerivationKey;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_ownership_invocation_invalid: " +
                                     message);
}

struct StaticGraphLaunchLineage final {
  dataflow::StaticGraphLaunchRef parent;
  dataflow::StaticGraphLaunchRef child;

  bool operator==(const StaticGraphLaunchLineage &other) const {
    return parent == other.parent && child == other.child;
  }
};

struct DataflowRewriteLineageEdge final {
  dataflow::DataflowRewriteDerivation derivation;
  std::vector<StaticGraphLaunchLineage> staticGraphLaunches;

  bool operator==(const DataflowRewriteLineageEdge &other) const {
    return derivation == other.derivation &&
           staticGraphLaunches == other.staticGraphLaunches;
  }
};

} // namespace

class detail::StructuredOwnershipDataflowLineageIndex::Impl final {
public:
  std::map<ArtifactRootReference, ArtifactRootReference,
           decltype(&artifactRootReferenceLess)>
      roots{&artifactRootReferenceLess};
  CanonicalDerivationIndex<DataflowRewriteLineageEdge> lineage;
  ArtifactReferenceSet nodes;
};

detail::StructuredOwnershipDataflowLineageIndex::
    StructuredOwnershipDataflowLineageIndex()
    : impl_(std::make_unique<Impl>()) {}

detail::StructuredOwnershipDataflowLineageIndex::
    ~StructuredOwnershipDataflowLineageIndex() = default;

bool detail::StructuredOwnershipDataflowLineageIndex::empty() const {
  return impl_->roots.empty();
}

llvm::Error detail::StructuredOwnershipDataflowLineageIndex::recordRoot(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &dataflowRoot) {
  if (structuredParent.schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      structuredParent.schemaVersion !=
          frontend::structuredProgramArtifactSchema.version ||
      dataflowRoot.schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      dataflowRoot.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow root has a foreign Artifact schema");
  auto [root, inserted] =
      impl_->roots.try_emplace(structuredParent, dataflowRoot);
  if (!inserted && root->second != dataflowRoot)
    return invalid("Structured parent changed its Canonical Dataflow root");
  impl_->nodes.insert(dataflowRoot);
  return llvm::Error::success();
}

llvm::Expected<ArtifactRootReference>
detail::StructuredOwnershipDataflowLineageIndex::root(
    const ArtifactRootReference &structuredParent) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  return root->second;
}

llvm::Error detail::StructuredOwnershipDataflowLineageIndex::recordDecision(
    const ArtifactRootReference &parent, const ArtifactRootReference &child,
    const dataflow::DataflowRewriteDecision &decision,
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> parentLaunches,
    llvm::ArrayRef<dataflow::StaticGraphLaunchRef> childLaunches) {
  if (parent.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      parent.schemaVersion != dataflow::canonicalDataflowSchema.version ||
      child.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      child.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return invalid("Dataflow rewrite lineage contains a foreign reference");
  if (impl_->nodes.find(parent) == impl_->nodes.end())
    return invalid("Dataflow rewrite parent is outside the prepared lineage");
  if (parent == child)
    return invalid("Dataflow rewrite cannot derive a candidate from itself");
  if (parentLaunches.size() != childLaunches.size())
    return invalid("Dataflow rewrite changed tracked launch cardinality");
  std::vector<StaticGraphLaunchLineage> launches;
  launches.reserve(parentLaunches.size());
  for (auto [parentLaunch, childLaunch] :
       llvm::zip_equal(parentLaunches, childLaunches)) {
    if (parentLaunch.artifact != parent.artifact ||
        childLaunch.artifact != child.artifact)
      return invalid("Dataflow rewrite launch lineage has a foreign owner");
    launches.push_back({parentLaunch, childLaunch});
  }
  auto payload = dataflow::encodeDataflowRewriteDecision(decision);
  if (!payload)
    return payload.takeError();
  dataflow::DataflowRewriteDerivation derivation{parent, child, decision};
  DataflowRewriteLineageEdge edge{derivation, std::move(launches)};
  auto [lineage, inserted] = impl_->lineage[child].try_emplace(
      CanonicalDerivationKey{parent, std::move(*payload)}, edge);
  if (!inserted && !(lineage->second == edge))
    return invalid("Dataflow lineage key has conflicting decisions");
  impl_->nodes.insert(child);
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::vector<dataflow::DataflowRewriteDerivation>>>
detail::StructuredOwnershipDataflowLineageIndex::tryResolve(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &candidate) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  if (impl_->nodes.find(candidate) == impl_->nodes.end())
    return std::optional<std::vector<dataflow::DataflowRewriteDerivation>>{};

  std::vector<dataflow::DataflowRewriteDerivation> result;
  enum class VisitState { Visiting, DeadEnd, ReachesRoot };
  std::map<ArtifactRootReference, VisitState,
           decltype(&artifactRootReferenceLess)>
      states(&artifactRootReferenceLess);
  std::function<llvm::Expected<bool>(const ArtifactRootReference &)> visit =
      [&](const ArtifactRootReference &reference) -> llvm::Expected<bool> {
    if (reference == root->second)
      return true;
    auto state = states.find(reference);
    if (state != states.end()) {
      if (state->second == VisitState::Visiting)
        return invalid("Dataflow candidate lineage contains a cycle");
      return state->second == VisitState::ReachesRoot;
    }
    auto edges = impl_->lineage.find(reference);
    if (edges == impl_->lineage.end()) {
      states.try_emplace(reference, VisitState::DeadEnd);
      return false;
    }
    states.try_emplace(reference, VisitState::Visiting);
    bool reachesRoot = false;
    for (const auto &entry : edges->second) {
      const dataflow::DataflowRewriteDerivation &edge = entry.second.derivation;
      auto reaches = visit(edge.parent);
      if (!reaches)
        return reaches.takeError();
      if (!*reaches)
        continue;
      reachesRoot = true;
      result.push_back(edge);
    }
    states.find(reference)->second =
        reachesRoot ? VisitState::ReachesRoot : VisitState::DeadEnd;
    return reachesRoot;
  };
  auto reaches = visit(candidate);
  if (!reaches)
    return reaches.takeError();
  if (!*reaches)
    return std::optional<std::vector<dataflow::DataflowRewriteDerivation>>{};
  llvm::sort(result, [](const dataflow::DataflowRewriteDerivation &lhs,
                        const dataflow::DataflowRewriteDerivation &rhs) {
    if (artifactRootReferenceLess(lhs.parent, rhs.parent))
      return true;
    if (artifactRootReferenceLess(rhs.parent, lhs.parent))
      return false;
    if (artifactRootReferenceLess(lhs.child, rhs.child))
      return true;
    if (artifactRootReferenceLess(rhs.child, lhs.child))
      return false;
    return dataflow::dataflowRewriteDecisionLess(lhs.decision, rhs.decision);
  });
  return std::optional<std::vector<dataflow::DataflowRewriteDerivation>>(
      std::move(result));
}

llvm::Expected<dataflow::StaticGraphLaunchRef>
detail::StructuredOwnershipDataflowLineageIndex::projectStaticGraphLaunch(
    const ArtifactRootReference &structuredParent,
    const ArtifactRootReference &candidate,
    dataflow::StaticGraphLaunchRef rootLaunch) const {
  auto root = impl_->roots.find(structuredParent);
  if (root == impl_->roots.end())
    return invalid("Structured parent has no Canonical Dataflow root");
  if (rootLaunch.artifact != root->second.artifact)
    return invalid("Dataflow root launch has a foreign artifact owner");

  ArtifactReferenceSet visiting;
  std::function<llvm::Expected<std::optional<dataflow::StaticGraphLaunchRef>>(
      const ArtifactRootReference &)>
      visit = [&](const ArtifactRootReference &reference)
      -> llvm::Expected<std::optional<dataflow::StaticGraphLaunchRef>> {
    if (reference == root->second)
      return std::optional<dataflow::StaticGraphLaunchRef>(rootLaunch);
    if (!visiting.insert(reference).second)
      return invalid("Dataflow launch lineage contains a cycle");
    auto edges = impl_->lineage.find(reference);
    if (edges == impl_->lineage.end()) {
      visiting.erase(reference);
      return std::optional<dataflow::StaticGraphLaunchRef>{};
    }

    std::optional<dataflow::StaticGraphLaunchRef> projected;
    for (const auto &entry : edges->second) {
      const DataflowRewriteLineageEdge &edge = entry.second;
      auto parentLaunch = visit(edge.derivation.parent);
      if (!parentLaunch)
        return parentLaunch.takeError();
      if (!*parentLaunch)
        continue;
      auto mapped = llvm::find_if(edge.staticGraphLaunches,
                                  [&](const StaticGraphLaunchLineage &lineage) {
                                    return lineage.parent == **parentLaunch;
                                  });
      if (mapped == edge.staticGraphLaunches.end())
        return invalid("Dataflow rewrite omitted a tracked graph launch");
      if (projected && *projected != mapped->child)
        return invalid("Dataflow rewrite paths disagree on graph launch");
      projected = mapped->child;
    }
    visiting.erase(reference);
    return projected;
  };

  auto projected = visit(candidate);
  if (!projected)
    return projected.takeError();
  if (!*projected)
    return invalid("Dataflow candidate does not descend from its exact D0");
  if ((*projected)->artifact != candidate.artifact)
    return invalid("Dataflow projected launch has the wrong child owner");
  return **projected;
}

} // namespace loom::dse
