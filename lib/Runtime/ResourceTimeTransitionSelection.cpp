#include "Runtime/ResourceTimeTransitionSelection.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Runtime/DeploymentLoader.h"

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
      lhs.startedRoot.has_value() != rhs.startedRoot.has_value() ||
      lhs.completion.has_value() != rhs.completion.has_value())
    return false;
  return (!lhs.startedRoot || lhs.startedRoot == rhs.startedRoot) &&
         (!lhs.completion || sameDecision(*lhs.completion, *rhs.completion));
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
          "with no live state and zero reprogramming and migration cost");

  const auto roots = mapping->view().executionBindings().rootThreadLaunches();
  return ResourceTimeTransitionSelectionSession(
      std::move(graph), std::move(dataflowReference),
      std::vector<dataflow::RootThreadLaunchRef>(roots.begin(), roots.end()));
}

llvm::Expected<ResourceTimeTransitionSelectionSession>
ResourceTimeTransitionSelectionSession::createPrepared(
    pnr::ResourceTimeTransitionGraph graph, LoadedDeployment &loaded,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto session = create(graph, loaded.deployment(), artifacts, blobs);
  if (!session)
    return session.takeError();
  if (graph.transitions.empty())
    return session;
  auto token = loaded.prepareResourceTimeActivations(graph, artifacts, blobs);
  if (!token)
    return token.takeError();
  session->preparedActivationToken_ = std::move(*token);
  return session;
}

std::vector<dataflow::RootThreadLaunchRef>
ResourceTimeTransitionSelectionSession::activeRoots() const {
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (auto [index, root] : llvm::enumerate(mappedRoots_))
    if (rootStates_[index] == RootState::Active)
      roots.push_back(root);
  return roots;
}

std::vector<dataflow::RootThreadLaunchRef>
ResourceTimeTransitionSelectionSession::completedRoots() const {
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (auto [index, root] : llvm::enumerate(mappedRoots_))
    if (rootStates_[index] == RootState::Completed)
      roots.push_back(root);
  return roots;
}

llvm::Error ResourceTimeTransitionSelectionSession::startRoot(
    dataflow::RootThreadLaunchRef startedRoot) {
  if (state_ != State::Running)
    return reject(ResourceTimeSelectionErrorReason::InvalidLifecycle,
                  "resource-time selector is not running");
  const auto root = llvm::find(mappedRoots_, startedRoot);
  if (root == mappedRoots_.end())
    return reject(ResourceTimeSelectionErrorReason::UnknownMappedRoot,
                  "start does not name a root imported from the entry "
                  "SystemMapping");
  const std::size_t rootIndex =
      static_cast<std::size_t>(std::distance(mappedRoots_.begin(), root));
  if (rootStates_[rootIndex] != RootState::NotStarted)
    return reject(ResourceTimeSelectionErrorReason::DuplicateStart,
                  "mapped root was already started");
  if (!requiredStarts_.empty() &&
      !llvm::is_contained(requiredStarts_, startedRoot))
    return reject(ResourceTimeSelectionErrorReason::ActiveSetMismatch,
                  "root is absent from the selected child active set");
  rootStates_[rootIndex] = RootState::Active;
  if (!requiredStarts_.empty())
    requiredStarts_.erase(llvm::find(requiredStarts_, startedRoot));
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::StartRoot, startedRoot, std::nullopt});
  return llvm::Error::success();
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
  if (rootStates_[rootIndex] == RootState::Completed)
    return reject(ResourceTimeSelectionErrorReason::DuplicateCompletion,
                  "mapped root completion was already accepted");
  if (rootStates_[rootIndex] != RootState::Active)
    return reject(ResourceTimeSelectionErrorReason::CompletionBeforeStart,
                  "mapped root completion preceded its start");
  if (!requiredStarts_.empty())
    return reject(ResourceTimeSelectionErrorReason::ActiveSetMismatch,
                  "selected child active roots have not all started");

  const pnr::ResourceTimeTransitionEndpointReference parent = current_;
  const std::vector<dataflow::RootThreadLaunchRef> completed = completedRoots();
  const std::vector<dataflow::RootThreadLaunchRef> active = activeRoots();
  std::optional<pnr::ResourceTimeTransition> selected;
  if (child) {
    const dataflow::EventFamilyKey trigger =
        dataflow::rootThreadCompletionEventFamily(completedRoot);
    bool matchedCompletionFrontier = false;
    for (const pnr::ResourceTimeTransition &transition : graph_.transitions) {
      if (transition.parent != parent || transition.child != *child ||
          transition.trigger != trigger || !transition.safePoint ||
          transition.safePoint->kind !=
              pnr::ResourceTimeSafePointKind::Completion ||
          transition.safePoint->artifact != dataflow_ ||
          !sameRootSet(transition.completedBefore, completed))
        continue;
      matchedCompletionFrontier = true;
      std::vector<dataflow::RootThreadLaunchRef> expectedActive;
      expectedActive.reserve(transition.beforeActive.size());
      for (const pnr::ResourceTimeRegionAllocation &allocation :
           transition.beforeActive)
        expectedActive.push_back(allocation.region);
      if (!sameRootSet(expectedActive, active))
        continue;
      if (selected)
        return reject(ResourceTimeSelectionErrorReason::TransitionUnavailable,
                      "resource-time selector found more than one exact edge");
      selected = transition;
    }
    if (!selected)
      return reject(
          matchedCompletionFrontier
              ? ResourceTimeSelectionErrorReason::ActiveSetMismatch
              : ResourceTimeSelectionErrorReason::TransitionUnavailable,
          matchedCompletionFrontier
              ? "actual active roots differ from the preverified edge"
              : "selected child has no exact verified edge at the current "
                "completion frontier");
  }

  rootStates_[rootIndex] = RootState::Completed;
  if (selected) {
    current_ = selected->child;
    requiredStarts_.clear();
    requiredStarts_.reserve(selected->afterActive.size());
    for (const pnr::ResourceTimeRegionAllocation &allocation :
         selected->afterActive)
      requiredStarts_.push_back(allocation.region);
  }
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::CompleteRoot, std::nullopt,
      ResourceTimeCompletionDecision{completedRoot, parent, std::move(child)}});
  return selected;
}

llvm::Expected<std::optional<pnr::ResourceTimeTransition>>
ResourceTimeTransitionSelectionSession::completeRootAndActivate(
    dataflow::RootThreadLaunchRef completedRoot,
    std::optional<pnr::ResourceTimeTransitionEndpointReference> child,
    LoadedDeployment &loaded) {
  if (!current_.deployment ||
      loaded.deployment().reference() != *current_.deployment)
    return reject(ResourceTimeSelectionErrorReason::ActiveDeploymentMismatch,
                  "loaded Deployment does not match the current resource-time "
                  "endpoint");

  const pnr::ResourceTimeTransitionEndpointReference priorCurrent = current_;
  const std::vector<RootState> priorRootStates = rootStates_;
  const std::vector<dataflow::RootThreadLaunchRef> priorRequiredStarts =
      requiredStarts_;
  const std::size_t priorReplaySize = replay_.size();
  auto selected = completeRoot(completedRoot, std::move(child));
  if (!selected)
    return selected.takeError();
  if (!*selected)
    return selected;
  if (llvm::Error error = loaded.activatePreparedTransition(
          **selected, preparedActivationToken_)) {
    current_ = priorCurrent;
    rootStates_ = priorRootStates;
    requiredStarts_ = priorRequiredStarts;
    replay_.resize(priorReplaySize);
    return std::move(error);
  }
  return selected;
}

llvm::Error ResourceTimeTransitionSelectionSession::joinMappedRoots() {
  if (state_ == State::Joined)
    return llvm::Error::success();
  if (state_ == State::Cancelled)
    return reject(ResourceTimeSelectionErrorReason::InvalidLifecycle,
                  "cancelled resource-time selector cannot join mapped roots");
  if (!requiredStarts_.empty() ||
      completedRoots().size() != mappedRoots_.size())
    return reject(ResourceTimeSelectionErrorReason::IncompleteMappedRootJoin,
                  "resource-time selector cannot join before every mapped "
                  "root completes");
  state_ = State::Joined;
  replay_.push_back(ResourceTimeSelectionReplayRecord{
      ResourceTimeSelectionAction::JoinMappedRoots, std::nullopt,
      std::nullopt});
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
      ResourceTimeSelectionAction::Cancel, std::nullopt, std::nullopt});
  return llvm::Error::success();
}

llvm::Error ResourceTimeTransitionSelectionSession::applyReplayRecord(
    const ResourceTimeSelectionReplayRecord &record) {
  const std::size_t priorSize = replay_.size();
  switch (record.action) {
  case ResourceTimeSelectionAction::StartRoot:
    if (!record.startedRoot || record.completion)
      return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                    "root-start replay has an invalid payload");
    if (llvm::Error error = startRoot(*record.startedRoot))
      return error;
    break;
  case ResourceTimeSelectionAction::CompleteRoot: {
    if (record.startedRoot || !record.completion ||
        record.completion->parent != current_)
      return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                    "completion replay does not match the current endpoint");
    auto selected = completeRoot(record.completion->completedRoot,
                                 record.completion->child);
    if (!selected)
      return selected.takeError();
    break;
  }
  case ResourceTimeSelectionAction::JoinMappedRoots:
    if (record.startedRoot || record.completion)
      return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                    "mapped-root join replay carries a completion decision");
    if (llvm::Error error = joinMappedRoots())
      return error;
    break;
  case ResourceTimeSelectionAction::Cancel:
    if (record.startedRoot || record.completion)
      return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                    "cancellation replay carries a completion decision");
    if (llvm::Error error = cancel())
      return error;
    break;
  }
  if (replay_.size() != priorSize + 1 || !sameRecord(replay_.back(), record))
    return reject(ResourceTimeSelectionErrorReason::ReplayMismatch,
                  "resource-time selector replay diverged");
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
  for (const ResourceTimeSelectionReplayRecord &record : records)
    if (llvm::Error error = replayed->applyReplayRecord(record))
      return std::move(error);
  return replayed;
}

} // namespace loom::runtime
