#ifndef LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
#define LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H

#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

namespace detail {
class SystemCandidateProjectionCache;
}

using SystemMemoryServiceTargetPlan =
    ::loom::fabric::FabricMemoryServiceTargetPlan;

struct SystemMemoryServiceTargetDomain final {
  std::vector<std::vector<SystemMemoryServiceTargetPlan>> plansBySubject;
};

struct SystemConsistencyServiceTargetDomain final {
  std::vector<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>
      domainsBySubject;
};

using SystemServiceTargetDomain =
    std::variant<SystemMemoryServiceTargetDomain,
                 SystemConsistencyServiceTargetDomain>;

struct SystemMemoryServiceTargetSelection final {
  std::vector<SystemMemoryServiceTargetPlan> plansBySubject;

  friend bool operator==(const SystemMemoryServiceTargetSelection &lhs,
                         const SystemMemoryServiceTargetSelection &rhs) {
    return lhs.plansBySubject == rhs.plansBySubject;
  }
};

struct SystemConsistencyServiceTargetSelection final {
  std::vector<::loom::fabric::MemoryConsistencyDomainRef> domainsBySubject;

  friend bool operator==(const SystemConsistencyServiceTargetSelection &lhs,
                         const SystemConsistencyServiceTargetSelection &rhs) {
    return lhs.domainsBySubject == rhs.domainsBySubject;
  }
};

using SystemServiceTargetSelection =
    std::variant<std::monostate, SystemMemoryServiceTargetSelection,
                 SystemConsistencyServiceTargetSelection>;

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

struct SystemRouteCapacityOveruseWitness final {
  PnrIndex capacityCell = getInvalidPnrIndex();
  std::uint64_t usage = 0;
  std::uint64_t capacity = 0;
  std::uint64_t overuse = 0;

  friend bool operator==(const SystemRouteCapacityOveruseWitness &lhs,
                         const SystemRouteCapacityOveruseWitness &rhs) {
    return lhs.capacityCell == rhs.capacityCell && lhs.usage == rhs.usage &&
           lhs.capacity == rhs.capacity && lhs.overuse == rhs.overuse;
  }
};

struct SystemInstructionResourceUseSelection final {
  ::dataflow::RootThreadLaunchRef root;
  ::loom::fabric::InstructionCoreContextRef context;
  ::loom::fabric::FabricUsePatternRef pattern;
};

struct SystemServiceResourceUseSelection final {
  PnrIndex context = getInvalidPnrIndex();
  PnrIndex subject = getInvalidPnrIndex();
  PnrIndex branch = 0;
  ::loom::fabric::FabricUsePatternRef pattern;
};

struct SystemCandidateInitialization final {
  llvm::ArrayRef<PnrIndex> threadChoices;
  llvm::ArrayRef<PnrIndex> graphChoices;
  llvm::ArrayRef<SystemServiceRouteSelection> serviceRoutes;
  llvm::ArrayRef<SystemServiceRouteNodeSelection> serviceRouteNodes;
  llvm::ArrayRef<SystemServiceRouteSinkSelection> serviceRouteSinks;
  llvm::ArrayRef<SystemServiceTargetSelection> serviceTargets;
  llvm::ArrayRef<SystemInstructionResourceUseSelection> instructionResourceUses;
  llvm::ArrayRef<SystemServiceResourceUseSelection> serviceResourceUses;
};

enum class SystemCandidateMutationDomain : std::uint8_t {
  TransportRoutes,
  ResourceSelection,
};

class SystemCandidateState;

class SystemCandidateState final {
public:
  static llvm::Expected<SystemCandidateStateHandle>
  create(FrozenSystemPnrProblemHandle problem,
         SystemCandidateInitialization initialization);

  /// Commits one immutable invocation-local mutation. The source remains
  /// unchanged on failure, so dropping the returned Error is the rollback.
  /// Accepted candidates are checked by the independent full oracle before
  /// they replace the working state.
  static llvm::Expected<SystemCandidateStateHandle>
  createMutation(const SystemCandidateState &source,
                 SystemCandidateInitialization initialization,
                 SystemCandidateMutationDomain domain);

  SystemCandidateState(const SystemCandidateState &) = delete;
  SystemCandidateState(SystemCandidateState &&) = delete;
  SystemCandidateState &operator=(const SystemCandidateState &) = delete;
  SystemCandidateState &operator=(SystemCandidateState &&) = delete;
  ~SystemCandidateState();

  const FrozenSystemPnrProblem &problem() const { return *problem_; }
  FrozenSystemPnrProblemHandle problemHandle() const { return problem_; }
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
  llvm::ArrayRef<SystemServiceTargetSelection> serviceTargets() const {
    return serviceTargets_;
  }
  llvm::ArrayRef<SystemInstructionResourceUseSelection>
  instructionResourceUses() const {
    return instructionResourceUses_;
  }
  llvm::ArrayRef<SystemServiceResourceUseSelection>
  serviceResourceUses() const {
    return serviceResourceUses_;
  }
  std::uint64_t routeCapacityOveruse() const { return routeCapacityOveruse_; }
  std::uint64_t capacityOveruse() const { return capacityOveruse_; }
  const SpatialRecurrenceTimingProjection &recurrenceTiming() const {
    return recurrenceTiming_;
  }
  std::uint64_t resourceMinimumInitiationIntervalCycles() const {
    return resourceMinimumInitiationIntervalCycles_;
  }
  std::uint64_t transportBitCycleDemand() const {
    return transportBitCycleDemand_;
  }
  const ::loom::mapping::MappingProgressClosure &progressClosure() const {
    return progressClosure_;
  }
  llvm::ArrayRef<SystemRouteCapacityOveruseWitness>
  routeCapacityOveruseWitnesses() const {
    return routeCapacityOveruseWitnesses_;
  }
  const SystemServiceTargetSelection &serviceTarget(PnrIndex context) const;
  PnrIndex threadChoice(PnrIndex decision) const;
  PnrIndex graphChoice(PnrIndex decision) const;
  ::loom::fabric::AccCoreOccurrenceRef selectedAccCore(PnrIndex decision) const;
  const ArtifactRootReference &selectedSpatialMapping(PnrIndex decision) const;
  llvm::Expected<SystemServiceTargetDomain>
  serviceTargetDomain(PnrIndex context) const;

  /// Invocation-local removable cache used only to derive mutation deltas.
  /// Candidate legality never trusts this cache; verify() rebuilds the full
  /// projection from the immutable selections.
  const detail::SystemCandidateProjectionCache &projectionCache() const {
    return *projectionCache_;
  }

  llvm::Error verify() const;

private:
  static llvm::Expected<SystemCandidateStateHandle>
  createImpl(FrozenSystemPnrProblemHandle problem,
             SystemCandidateInitialization initialization,
             const SystemCandidateState *source,
             std::optional<SystemCandidateMutationDomain> domain,
             bool runFullOracle);

  SystemCandidateState(
      FrozenSystemPnrProblemHandle problem, std::vector<PnrIndex> threadChoices,
      std::vector<PnrIndex> graphChoices,
      std::vector<SystemServiceRouteSelection> serviceRoutes,
      std::vector<SystemServiceRouteNodeSelection> serviceRouteNodes,
      std::vector<SystemServiceRouteSinkSelection> serviceRouteSinks,
      std::vector<SystemServiceTargetSelection> serviceTargets,
      std::vector<SystemInstructionResourceUseSelection>
          instructionResourceUses,
      std::vector<SystemServiceResourceUseSelection> serviceResourceUses,
      std::shared_ptr<const detail::SystemCandidateProjectionCache>
          projectionCache,
      std::uint64_t capacityOveruse,
      ::loom::mapping::MappingProgressClosure progressClosure,
      SpatialRecurrenceTimingProjection recurrenceTiming,
      std::uint64_t resourceMinimumInitiationIntervalCycles,
      std::uint64_t transportBitCycleDemand, std::uint64_t routeCapacityOveruse,
      std::vector<SystemRouteCapacityOveruseWitness>
          routeCapacityOveruseWitnesses)
      : problem_(std::move(problem)), threadChoices_(std::move(threadChoices)),
        graphChoices_(std::move(graphChoices)),
        serviceRoutes_(std::move(serviceRoutes)),
        serviceRouteNodes_(std::move(serviceRouteNodes)),
        serviceRouteSinks_(std::move(serviceRouteSinks)),
        serviceTargets_(std::move(serviceTargets)),
        instructionResourceUses_(std::move(instructionResourceUses)),
        serviceResourceUses_(std::move(serviceResourceUses)),
        projectionCache_(std::move(projectionCache)),
        capacityOveruse_(capacityOveruse), progressClosure_(progressClosure),
        recurrenceTiming_(std::move(recurrenceTiming)),
        resourceMinimumInitiationIntervalCycles_(
            resourceMinimumInitiationIntervalCycles),
        transportBitCycleDemand_(transportBitCycleDemand),
        routeCapacityOveruse_(routeCapacityOveruse),
        routeCapacityOveruseWitnesses_(
            std::move(routeCapacityOveruseWitnesses)) {}

  FrozenSystemPnrProblemHandle problem_;
  std::vector<PnrIndex> threadChoices_;
  std::vector<PnrIndex> graphChoices_;
  std::vector<SystemServiceRouteSelection> serviceRoutes_;
  std::vector<SystemServiceRouteNodeSelection> serviceRouteNodes_;
  std::vector<SystemServiceRouteSinkSelection> serviceRouteSinks_;
  std::vector<SystemServiceTargetSelection> serviceTargets_;
  std::vector<SystemInstructionResourceUseSelection> instructionResourceUses_;
  std::vector<SystemServiceResourceUseSelection> serviceResourceUses_;
  std::shared_ptr<const detail::SystemCandidateProjectionCache>
      projectionCache_;
  std::uint64_t capacityOveruse_ = 0;
  ::loom::mapping::MappingProgressClosure progressClosure_;
  SpatialRecurrenceTimingProjection recurrenceTiming_;
  std::uint64_t resourceMinimumInitiationIntervalCycles_ = 1;
  std::uint64_t transportBitCycleDemand_ = 0;
  std::uint64_t routeCapacityOveruse_ = 0;
  std::vector<SystemRouteCapacityOveruseWitness> routeCapacityOveruseWitnesses_;
};

struct InitializedSystemCandidate final {
  SystemCandidateStateHandle state;
  std::uint64_t assignmentAttempts = 0;
  std::uint64_t endpointExpansions = 0;
  std::uint64_t negotiationIterations = 0;
};

enum class SystemCandidateInitializationFailureKind : std::uint8_t {
  ProvenInfeasible,
  SemanticLimitReached,
  Internal,
};

class SystemCandidateInitializationFailure final
    : public llvm::ErrorInfo<SystemCandidateInitializationFailure> {
public:
  static char ID;

  SystemCandidateInitializationFailure(
      SystemCandidateInitializationFailureKind kind,
      std::uint64_t assignmentAttempts, std::uint64_t endpointExpansions,
      std::uint64_t negotiationIterations, std::string message)
      : kind_(kind), assignmentAttempts_(assignmentAttempts),
        endpointExpansions_(endpointExpansions),
        negotiationIterations_(negotiationIterations),
        message_(std::move(message)) {}

  SystemCandidateInitializationFailureKind kind() const { return kind_; }
  std::uint64_t assignmentAttempts() const { return assignmentAttempts_; }
  std::uint64_t endpointExpansions() const { return endpointExpansions_; }
  std::uint64_t negotiationIterations() const { return negotiationIterations_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SystemCandidateInitializationFailureKind kind_;
  std::uint64_t assignmentAttempts_ = 0;
  std::uint64_t endpointExpansions_ = 0;
  std::uint64_t negotiationIterations_ = 0;
  std::string message_;
};

llvm::Expected<InitializedSystemCandidate>
initializeCanonicalSystemCandidate(FrozenSystemPnrProblemHandle problem);

llvm::Expected<InitializedSystemCandidate>
initializeSystemCandidateAttempt(FrozenSystemPnrProblemHandle problem,
                                 std::uint32_t attemptOrdinal);

llvm::Expected<InitializedSystemCandidate>
initializeSystemCandidateWithFixedChoices(
    FrozenSystemPnrProblemHandle problem,
    llvm::ArrayRef<PnrIndex> fixedChoices);

llvm::Expected<SystemCandidateStateHandle>
initializeSystemCandidate(FrozenSystemPnrProblemHandle problem,
                          llvm::ArrayRef<PnrIndex> threadChoices,
                          llvm::ArrayRef<PnrIndex> graphChoices,
                          std::uint64_t *endpointExpansions = nullptr,
                          std::uint64_t *negotiationIterations = nullptr);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
