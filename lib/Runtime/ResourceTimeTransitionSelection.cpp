#include "Runtime/ResourceTimeTransitionSelection.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::runtime {
namespace {

llvm::Error reject(ResourceTimeSelectionErrorReason reason,
                   const llvm::Twine &message) {
  return llvm::make_error<ResourceTimeSelectionError>(reason, message.str());
}

bool isSupportedSelectionProfile(const pnr::ResourceTimeTransition &transition,
                                 const ArtifactRootReference &dataflow) {
  return transition.status == pnr::ResourceTimeTransitionStatus::Verified &&
         transition.safePoint &&
         transition.safePoint->kind ==
             pnr::ResourceTimeSafePointKind::Completion &&
         transition.safePoint->artifact == dataflow &&
         transition.beforeActive.size() == 1 &&
         transition.beforeLiveWork.empty() &&
         transition.afterLiveWork.empty() &&
         !transition.tokenLiveStateCorrespondence &&
         transition.reprogrammingTimePicoseconds ==
             std::optional<std::uint64_t>(0) &&
         transition.migrationTimePicoseconds == std::optional<std::uint64_t>(0);
}

bool sameRootSet(llvm::ArrayRef<dataflow::RootThreadLaunchRef> lhs,
                 llvm::ArrayRef<dataflow::RootThreadLaunchRef> rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(lhs, [&](auto root) {
           return llvm::is_contained(rhs, root);
         });
}

bool sameDecision(const ResourceTimeCompletionDecision &lhs,
                  const ResourceTimeCompletionDecision &rhs) {
  return lhs.completedRoot == rhs.completedRoot && lhs.parent == rhs.parent &&
         lhs.child == rhs.child;
}

bool sameRecord(const ResourceTimeSelectionReplayRecord &lhs,
                const ResourceTimeSelectionReplayRecord &rhs) {
  if (lhs.action != rhs.action ||
      lhs.completion.has_value() != rhs.completion.has_value())
    return false;
  return !lhs.completion || sameDecision(*lhs.completion, *rhs.completion);
}

} // namespace

char ResourceTimeSelectionError::ID = 0;

void ResourceTimeSelectionError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code ResourceTimeSelectionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<ResourceTimeTransitionSelectionSession>
ResourceTimeTransitionSelectionSession::create(
    pnr::ResourceTimeTransitionGraph graph,
    const deployment::FinalizedDeployment &entryDeployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error =
          pnr::verifyResourceTimeTransitionGraph(graph, artifacts, blobs))
    return std::move(error);

  auto importedEntry = deployment::importDeployment(entryDeployment.reference(),
                                                    artifacts, blobs);
  if (!importedEntry)
    return importedEntry.takeError();
  if (!graph.entry.deployment ||
      *graph.entry.deployment != entryDeployment.reference() ||
      importedEntry->reference() != entryDeployment.reference() ||
      importedEntry->deployment().systemMapping() != graph.entry.mapping ||
      entryDeployment.deployment().systemMapping() != graph.entry.mapping)
    return reject(ResourceTimeSelectionErrorReason::EntryDeploymentMismatch,
                  "resource-time selector entry Deployment does not match "
                  "the graph entry");

  auto mapping = mapping::importSystemMapping(graph.entry.mapping, artifacts);
  if (!mapping)
    return mapping.takeError();
  ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping->view().dataflowIdentity()};
  for (const pnr::ResourceTimeTransition &transition : graph.transitions)
    if (!isSupportedSelectionProfile(transition, dataflowReference))
      return reject(
          ResourceTimeSelectionErrorReason::UnsupportedTransitionProfile,
          "resource-time selector supports only verified completion edges "
          "with no live state and exact zero transition cost");

  const auto roots = mapping->view().executionBindings().rootThreadLaunches();
  return ResourceTimeTransitionSelectionSession(
      std::move(graph), std::move(dataflowReference),
      std::vector<dataflow::RootThreadLaunchRef>(roots.begin(), roots.end()));
}

llvm::Expected<std::optional<pnr::ResourceTimeTransition>>
ResourceTimeTransitionSelectionSession::completeRoot(
    dataflow::RootThreadLaunchRef completedRoot,
    std::optional<pnr::ResourceTimeTransitionEndpointReference> child) {
  if (state_ != State::Running)
    return reject(ResourceTimeSelectionErrorReason::InvalidLifecycle,
                  "resource-time selector is not running");
  const auto root = llvm::find(mappedRoots_, completedRoot);
  if (root == mappedRoots_.end())
    return reject(ResourceTimeSelectionErrorReason::UnknownMappedRoot,
                  "completion does not name a root imported from the entry "
                  "SystemMapping");
  const std::size_t rootIndex =
      static_cast<std::size_t>(std::distance(mappedRoots_.begin(), root));
  if (completedMarks_[rootIndex])
    return reject(ResourceTimeSelectionErrorReason::DuplicateCompletion,
                  "mapped root completion was already accepted");

  const pnr::ResourceTimeTransitionEndpointReference parent = current_;
  std::optional<pnr::ResourceTimeTransition> selected;
  if (child) {
    const dataflow::EventFamilyKey trigger =
        dataflow::rootThreadCompletionEventFamily(completedRoot);
    for (const pnr::ResourceTimeTransition &transition : graph_.transitions) {
      if (transition.parent != parent || transition.child != *child ||
          transition.trigger != trigger || !transition.safePoint ||
          transition.safePoint->kind !=
              pnr::ResourceTimeSafePointKind::Completion ||
          transition.safePoint->artifact != dataflow_ ||
          !sameRootSet(transition.completedBefore, completedRoots_))
        continue;
      if (selected)
        return reject(ResourceTimeSelectionErrorReason::TransitionUnavailable,
                      "resource-time selector found more than one exact edge");
      selected = transition;
    }
    if (!selected)
      return reject(ResourceTimeSelectionErrorReason::TransitionUnavailable,
                    "selected child has no exact verified edge at the current "
                    "completion frontier");
  }

  completedMarks_[rootIndex] = true;
  completedRoots_.clear();
  completedRoots_.reserve(mappedRoots_.size());
  for (auto [index, mappedRoot] : llvm::enumerate(mappedRoots_))
    if (completedMarks_[index])
      completedRoots_.push_back(mappedRoot);
  if (selected)
    current_ = selected->child;
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::CompleteRoot,
      ResourceTimeCompletionDecision{completedRoot, parent, std::move(child)}});
  return selected;
}

llvm::Error ResourceTimeTransitionSelectionSession::joinMappedRoots() {
  if (state_ == State::Joined)
    return llvm::Error::success();
  if (state_ == State::Cancelled)
    return reject(ResourceTimeSelectionErrorReason::InvalidLifecycle,
                  "cancelled resource-time selector cannot join mapped roots");
  if (completedRoots_.size() != mappedRoots_.size())
    return reject(ResourceTimeSelectionErrorReason::IncompleteMappedRootJoin,
                  "resource-time selector cannot join before every mapped "
                  "root completes");
  state_ = State::Joined;
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::JoinMappedRoots, std::nullopt});
  return llvm::Error::success();
}

llvm::Error ResourceTimeTransitionSelectionSession::cancel() {
  if (state_ == State::Cancelled)
    return llvm::Error::success();
  if (state_ == State::Joined)
    return reject(ResourceTimeSelectionErrorReason::InvalidLifecycle,
                  "joined resource-time selector cannot be cancelled");
  state_ = State::Cancelled;
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::Cancel, std::nullopt});
  return llvm::Error::success();
}

llvm::Expected<ResourceTimeTransitionSelectionSession>
ResourceTimeTransitionSelectionSession::replay(
    pnr::ResourceTimeTransitionGraph graph,
    const deployment::FinalizedDeployment &entryDeployment,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    llvm::ArrayRef<ResourceTimeSelectionReplayRecord> records) {
  auto replayed = create(std::move(graph), entryDeployment, artifacts, blobs);
  if (!replayed)
    return replayed.takeError();
  for (const ResourceTimeSelectionReplayRecord &record : records) {
    const std::size_t priorSize = replayed->replay_.size();
    switch (record.action) {
    case ResourceTimeSelectionAction::CompleteRoot:
      if (!record.completion || record.completion->parent != replayed->current_)
        return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                      "completion replay does not match the current endpoint");
      if (auto selected = replayed->completeRoot(
              record.completion->completedRoot, record.completion->child);
          !selected)
        return selected.takeError();
      break;
    case ResourceTimeSelectionAction::JoinMappedRoots:
      if (record.completion)
        return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                      "mapped-root join replay carries a completion decision");
      if (llvm::Error error = replayed->joinMappedRoots())
        return std::move(error);
      break;
    case ResourceTimeSelectionAction::Cancel:
      if (record.completion)
        return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                      "cancellation replay carries a completion decision");
      if (llvm::Error error = replayed->cancel())
        return std::move(error);
      break;
    }
    if (replayed->replay_.size() != priorSize + 1 ||
        !sameRecord(replayed->replay_.back(), record))
      return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                    "resource-time selector replay diverged");
  }
  return replayed;
}

} // namespace loom::runtime
