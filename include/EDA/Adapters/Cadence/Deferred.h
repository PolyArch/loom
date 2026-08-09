#ifndef LOOM_EDA_ADAPTERS_CADENCE_DEFERRED_H
#define LOOM_EDA_ADAPTERS_CADENCE_DEFERRED_H

#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace loom::eda::cadence {

enum class CadenceDeferredStage : std::uint8_t {
  XceliumFunctionalEvaluation,
  InnovusAsicPhysical,
  JoulesPowerEvaluation,
  TempusTimingEvaluation,
  VoltusRailEvaluation,
};

enum class CadenceDeferredBoundary : std::uint8_t {
  Prepare,
  Parse,
  StrictImport,
};

class CadenceStageUnsupportedError final
    : public llvm::ErrorInfo<CadenceStageUnsupportedError> {
public:
  static char ID;

  CadenceStageUnsupportedError(CadenceDeferredStage stage,
                               CadenceDeferredBoundary boundary,
                               std::string missingOwner)
      : stage_(stage), boundary_(boundary),
        missingOwner_(std::move(missingOwner)) {}

  CadenceDeferredStage stage() const { return stage_; }
  CadenceDeferredBoundary boundary() const { return boundary_; }
  llvm::StringRef missingOwner() const { return missingOwner_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  CadenceDeferredStage stage_;
  CadenceDeferredBoundary boundary_;
  std::string missingOwner_;
};

/// These boundaries deliberately publish nothing. They make the unsupported
/// ownership boundary executable without inventing a physical, Evaluation,
/// or Simulation result carrier.
llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareDeferredCadenceStage(CadenceDeferredStage stage);

llvm::Error parseDeferredCadenceStage(CadenceDeferredStage stage,
                                      llvm::StringRef outputBytes);

llvm::Error importDeferredCadenceStage(
    CadenceDeferredStage stage,
    const external_tool::PreparedExternalToolInvocation &prepared);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_DEFERRED_H
