#ifndef LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
#define LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H

#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace loom::pnr {

using SystemServiceTargetDomain =
    std::variant<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>,
                 std::vector<::loom::fabric::MemoryConsistencyDomainRef>>;

struct SystemServiceRouteNodeSelection final {
  PnrIndex endpoint = 0;
  PnrIndex parentNode = getInvalidPnrIndex();
  PnrIndex incomingTraversal = getInvalidPnrIndex();
};

struct SystemServiceRouteSinkSelection final {
  PnrIndex terminal = 0;
  PnrIndex node = 0;
};

struct SystemServiceRouteSelection final {
  PnrIndex leg = 0;
  PnrIndex rootEndpoint = getInvalidPnrIndex();
  PnrIndex nodeOffset = 0;
  PnrIndex nodeCount = 0;
  PnrIndex sinkOffset = 0;
  PnrIndex sinkCount = 0;
};

struct SystemCandidateInitialization final {
  llvm::ArrayRef<PnrIndex> threadChoices;
  llvm::ArrayRef<PnrIndex> graphChoices;
  llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes;
  llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes;
  llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks;
};

class SystemCandidateState;

class SystemCandidateState final {
public:
  static llvm::Expected<SystemCandidateStateHandle>
  create(FrozenSystemPnrProblemHandle problem,
         SystemCandidateInitialization initialization);

  SystemCandidateState(const SystemCandidateState &) = delete;
  SystemCandidateState(SystemCandidateState &&) = delete;
  SystemCandidateState &operator=(const SystemCandidateState &) = delete;
  SystemCandidateState &operator=(SystemCandidateState &&) = delete;
  ~SystemCandidateState() = default;

  const FrozenSystemPnrProblem &problem() const { return *problem_; }
  llvm::ArrayRef<PnrIndex> threadChoices() const { return threadChoices_; }
  llvm::ArrayRef<PnrIndex> graphChoices() const { return graphChoices_; }
  llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes() const {
    return serviceRoutes_;
  }
  llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes() const {
    return serviceRouteNodes_;
  }
  llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks() const {
    return serviceRouteSinks_;
  }
  PnrIndex threadChoice(PnrIndex decision) const;
  PnrIndex graphChoice(PnrIndex decision) const;
  ::loom::fabric::AccCoreOccurrenceRef selectedAccCore(PnrIndex decision) const;
  const ArtifactRootReference &selectedSpatialMapping(PnrIndex decision) const;
  llvm::Expected<SystemServiceTargetDomain>
  serviceTargetDomain(PnrIndex context) const;

  llvm::Error verify() const;

private:
  SystemCandidateState(
      FrozenSystemPnrProblemHandle problem, std::vector<PnrIndex> threadChoices,
      std::vector<PnrIndex> graphChoices,
      std::vector<SystemServiceRouteSelection> serviceRoutes,
      std::vector<SystemServiceRouteNodeSelection> serviceRouteNodes,
      std::vector<SystemServiceRouteSinkSelection> serviceRouteSinks)
      : problem_(std::move(problem)), threadChoices_(std::move(threadChoices)),
        graphChoices_(std::move(graphChoices)),
        serviceRoutes_(std::move(serviceRoutes)),
        serviceRouteNodes_(std::move(serviceRouteNodes)),
        serviceRouteSinks_(std::move(serviceRouteSinks)) {}

  FrozenSystemPnrProblemHandle problem_;
  std::vector<PnrIndex> threadChoices_;
  std::vector<PnrIndex> graphChoices_;
  std::vector<SystemServiceRouteSelection> serviceRoutes_;
  std::vector<SystemServiceRouteNodeSelection> serviceRouteNodes_;
  std::vector<SystemServiceRouteSinkSelection> serviceRouteSinks_;
};

struct InitializedSystemCandidate final {
  SystemCandidateStateHandle state;
  std::uint64_t assignmentAttempts = 0;
};

llvm::Expected<InitializedSystemCandidate>
initializeCanonicalSystemCandidate(FrozenSystemPnrProblemHandle problem);

llvm::Expected<SystemCandidateStateHandle>
initializeSystemCandidate(FrozenSystemPnrProblemHandle problem,
                          llvm::ArrayRef<PnrIndex> threadChoices,
                          llvm::ArrayRef<PnrIndex> graphChoices);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
