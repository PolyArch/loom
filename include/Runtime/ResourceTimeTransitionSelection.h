#ifndef LOOM_RUNTIME_RESOURCETIMETRANSITIONSELECTION_H
#define LOOM_RUNTIME_RESOURCETIMETRANSITIONSELECTION_H

#include "Deployment/Deployment.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::runtime {

enum class ResourceTimeSelectionErrorReason : std::uint8_t {
  EntryDeploymentMismatch,
  UnsupportedTransitionProfile,
  UnknownMappedRoot,
  DuplicateCompletion,
  TransitionUnavailable,
  IncompleteMappedRootJoin,
  InvalidLifecycle,
  ReplayMismatch,
};

class ResourceTimeSelectionError final
    : public llvm::ErrorInfo<ResourceTimeSelectionError> {
public:
  static char ID;

  ResourceTimeSelectionError(ResourceTimeSelectionErrorReason reason,
                             std::string message)
      : reason_(reason), message_(std::move(message)) {}

  ResourceTimeSelectionErrorReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ResourceTimeSelectionErrorReason reason_;
  std::string message_;
};

/// One accepted completion decision. `parent` is the exact endpoint current
/// before the decision. An absent child is an explicit stay decision.
struct ResourceTimeCompletionDecision final {
  ::dataflow::RootThreadLaunchRef completedRoot;
  ::loom::pnr::ResourceTimeTransitionEndpointReference parent;
  std::optional<::loom::pnr::ResourceTimeTransitionEndpointReference> child;
};

enum class ResourceTimeSelectionAction : std::uint8_t {
  CompleteRoot,
  JoinMappedRoots,
  Cancel,
};

struct ResourceTimeSelectionReplayRecord final {
  ResourceTimeSelectionAction action =
      ResourceTimeSelectionAction::CompleteRoot;
  std::optional<ResourceTimeCompletionDecision> completion;
};

/// Invocation-local selector for one immutable compiler-preverified graph.
/// It records safe-point choices but does not program or activate a child
/// Deployment and does not snapshot or move live runtime state.
class ResourceTimeTransitionSelectionSession final {
public:
  static llvm::Expected<ResourceTimeTransitionSelectionSession>
  create(::loom::pnr::ResourceTimeTransitionGraph graph,
         const ::loom::deployment::FinalizedDeployment &entryDeployment,
         const ArtifactStore &artifacts, const BlobStore &blobs);

  static llvm::Expected<ResourceTimeTransitionSelectionSession>
  replay(::loom::pnr::ResourceTimeTransitionGraph graph,
         const ::loom::deployment::FinalizedDeployment &entryDeployment,
         const ArtifactStore &artifacts, const BlobStore &blobs,
         llvm::ArrayRef<ResourceTimeSelectionReplayRecord> records);

  ResourceTimeTransitionSelectionSession(
      const ResourceTimeTransitionSelectionSession &) = delete;
  ResourceTimeTransitionSelectionSession &
  operator=(const ResourceTimeTransitionSelectionSession &) = delete;
  ResourceTimeTransitionSelectionSession(
      ResourceTimeTransitionSelectionSession &&) noexcept = default;
  ResourceTimeTransitionSelectionSession &
  operator=(ResourceTimeTransitionSelectionSession &&) noexcept = default;

  const ::loom::pnr::ResourceTimeTransitionEndpointReference &
  currentEndpoint() const {
    return current_;
  }
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> completedRoots() const {
    return completedRoots_;
  }
  bool mappedRootsJoined() const { return state_ == State::Joined; }
  bool cancelled() const { return state_ == State::Cancelled; }

  /// Records one collective root completion in caller commit order. A child
  /// is selected only through an exact verified edge; graph order is never a
  /// priority rule. Rejection leaves endpoint, completion, and replay state
  /// unchanged.
  llvm::Expected<std::optional<::loom::pnr::ResourceTimeTransition>>
  completeRoot(
      ::dataflow::RootThreadLaunchRef completedRoot,
      std::optional<::loom::pnr::ResourceTimeTransitionEndpointReference>
          child);

  /// Joins only the root inventory imported from the entry SystemMapping.
  /// Host residual execution and process termination remain separate owners.
  llvm::Error joinMappedRoots();

  /// Stops future selection. This is idempotent but does not cancel provider,
  /// channel, or DynamicWork responsibilities.
  llvm::Error cancel();

  llvm::ArrayRef<ResourceTimeSelectionReplayRecord> replay() const {
    return replay_;
  }

private:
  enum class State : std::uint8_t { Running, Joined, Cancelled };

  ResourceTimeTransitionSelectionSession(
      ::loom::pnr::ResourceTimeTransitionGraph graph,
      ArtifactRootReference dataflow,
      std::vector<::dataflow::RootThreadLaunchRef> mappedRoots)
      : graph_(std::move(graph)), dataflow_(std::move(dataflow)),
        mappedRoots_(std::move(mappedRoots)),
        completedMarks_(mappedRoots_.size(), false), current_(graph_.entry) {}

  ::loom::pnr::ResourceTimeTransitionGraph graph_;
  ArtifactRootReference dataflow_;
  std::vector<::dataflow::RootThreadLaunchRef> mappedRoots_;
  std::vector<bool> completedMarks_;
  std::vector<::dataflow::RootThreadLaunchRef> completedRoots_;
  ::loom::pnr::ResourceTimeTransitionEndpointReference current_;
  State state_ = State::Running;
  std::vector<ResourceTimeSelectionReplayRecord> replay_;
};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_RESOURCETIMETRANSITIONSELECTION_H
