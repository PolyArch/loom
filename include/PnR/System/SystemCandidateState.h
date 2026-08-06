#ifndef LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
#define LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H

#include "PnR/System/SystemPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

struct SystemCandidateInitialization final {
  llvm::ArrayRef<PnrIndex> threadChoices;
  llvm::ArrayRef<PnrIndex> graphChoices;
};

class SystemCandidateState;
using SystemCandidateStateHandle = std::shared_ptr<SystemCandidateState>;

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
  PnrIndex threadChoice(PnrIndex decision) const;
  PnrIndex graphChoice(PnrIndex decision) const;
  ::loom::fabric::AccCoreOccurrenceRef selectedAccCore(PnrIndex decision) const;
  const ArtifactRootReference &selectedSpatialMapping(PnrIndex decision) const;

  llvm::Error verify() const;

private:
  SystemCandidateState(FrozenSystemPnrProblemHandle problem,
                       std::vector<PnrIndex> threadChoices,
                       std::vector<PnrIndex> graphChoices)
      : problem_(std::move(problem)), threadChoices_(std::move(threadChoices)),
        graphChoices_(std::move(graphChoices)) {}

  FrozenSystemPnrProblemHandle problem_;
  std::vector<PnrIndex> threadChoices_;
  std::vector<PnrIndex> graphChoices_;
};

struct InitializedSystemCandidate final {
  SystemCandidateStateHandle state;
  std::uint64_t assignmentAttempts = 0;
};

llvm::Expected<InitializedSystemCandidate>
initializeCanonicalSystemCandidate(FrozenSystemPnrProblemHandle problem);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMCANDIDATESTATE_H
