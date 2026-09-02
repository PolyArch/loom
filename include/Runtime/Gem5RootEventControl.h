#ifndef LOOM_RUNTIME_GEM5ROOTEVENTCONTROL_H
#define LOOM_RUNTIME_GEM5ROOTEVENTCONTROL_H

#include "PnR/System/SystemMappingMigration.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Runtime/Gem5SystemExecution.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace loom {
class ArtifactStore;
} // namespace loom

namespace loom::runtime {

enum class Gem5RootEventControlErrorReason : std::uint8_t {
  EndpointWithoutDeployment,
  NonTerminalEdge,
  EndpointBoundExceeded,
  SocketUnavailable,
  ProtocolFailure,
  ControllerRejected,
};

llvm::StringRef gem5RootEventControlErrorReasonSpelling(
    Gem5RootEventControlErrorReason reason);

class Gem5RootEventControlError final
    : public llvm::ErrorInfo<Gem5RootEventControlError> {
public:
  static char ID;

  Gem5RootEventControlError(Gem5RootEventControlErrorReason reason,
                            std::string message)
      : reason_(reason), message_(std::move(message)) {}

  Gem5RootEventControlErrorReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Gem5RootEventControlErrorReason reason_;
  std::string message_;
};

/// Derives the controller endpoint table of one independently verified
/// transition graph. The gem5 Thread Dispatch device cannot dispatch under a
/// nonzero endpoint, so every edge must be terminal: the completing root plus
/// `completed_before` cover the entry Mapping root inventory and no region
/// stays active under the child. A non-terminal edge is a typed refusal of the
/// synchronous drive, never a silently uncontrolled invocation.
llvm::Expected<Gem5RootEventEndpointTable>
deriveGem5RootEventEndpointTable(const pnr::ResourceTimeTransitionGraph &graph,
                                 const ArtifactStore &artifacts);

/// One controller decision for one acknowledged root event. `endpoint` names
/// the active endpoint after the decision in `Gem5RootEventEndpointTable`
/// order.
struct Gem5RootEventDecision final {
  Gem5RootEventControlDecision decision =
      Gem5RootEventControlDecision::Reject;
  std::uint64_t endpoint = 0;
};

/// One acknowledged request in controller order: the device generation, the
/// canonical root lifecycle observation, and the decision returned to the
/// device before it continued.
struct Gem5RootEventAcknowledgement final {
  std::uint64_t generation = 0;
  sim::SystemRootLifecycleObservation observation;
  Gem5RootEventDecision decision;
};

using Gem5RootEventDecisionCallback =
    std::function<llvm::Expected<Gem5RootEventDecision>(
        const sim::SystemRootLifecycleObservation &)>;

/// Host side of the gem5 root event control socket for one bundle execution.
/// It listens at `gem5RootEventControlSocketPath` under the bundle root before
/// the invocation starts, decodes each device request into the canonical root
/// lifecycle observation, asks the callback for the decision at that exact
/// safe point, and acknowledges it before the device continues. A callback
/// error is answered with `Reject`, which the device reports as a protocol
/// failure, and surfaces from `finish`.
class Gem5RootEventController final {
public:
  static llvm::Expected<Gem5RootEventController>
  listen(llvm::StringRef bundleRoot, ArtifactIdentity dataflow,
         Gem5RootEventDecisionCallback callback);

  Gem5RootEventController(Gem5RootEventController &&) noexcept;
  Gem5RootEventController &operator=(Gem5RootEventController &&) noexcept;
  Gem5RootEventController(const Gem5RootEventController &) = delete;
  Gem5RootEventController &operator=(const Gem5RootEventController &) = delete;
  ~Gem5RootEventController();

  /// Stops serving, removes the socket, and returns every acknowledged event
  /// in device order. The first callback or protocol failure is returned
  /// instead of the records.
  llvm::Expected<std::vector<Gem5RootEventAcknowledgement>> finish();

private:
  class Impl;
  explicit Gem5RootEventController(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace loom::runtime

#endif // LOOM_RUNTIME_GEM5ROOTEVENTCONTROL_H
