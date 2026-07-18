#ifndef LOOM_PNR_PNRPROBLEMINPUTS_H
#define LOOM_PNR_PNRPROBLEMINPUTS_H

#include "Mapping/Artifact.h"

#include "llvm/Support/Error.h"

#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

namespace loom::mapping {
class ValidatedTechMapping;
} // namespace loom::mapping

namespace loom::pnr {

struct ResolvedPnrConfigView {};

struct MappingConstraintSetInput {
  mapping::ArtifactIdentity identity;
  mapping::ArtifactIdentity dataflowIdentity;
  mapping::ArtifactIdentity techMappingIdentity;
  mapping::ArtifactIdentity fabricIdentity;
};

struct PnrProblemInputs {
  template <
      typename Dataflow, typename TechMapping, typename Fabric, typename Config,
      std::enable_if_t<
          std::is_lvalue_reference_v<Dataflow &&> &&
              std::is_same_v<
                  std::remove_cv_t<std::remove_reference_t<Dataflow>>,
                  mapping::DataflowProgramView> &&
              std::is_lvalue_reference_v<TechMapping &&> &&
              std::is_same_v<
                  std::remove_cv_t<std::remove_reference_t<TechMapping>>,
                  mapping::ValidatedTechMapping> &&
              std::is_lvalue_reference_v<Fabric &&> &&
              std::is_same_v<std::remove_cv_t<std::remove_reference_t<Fabric>>,
                             mapping::FabricHardwareView> &&
              std::is_lvalue_reference_v<Config &&> &&
              std::is_same_v<std::remove_cv_t<std::remove_reference_t<Config>>,
                             ResolvedPnrConfigView>,
          int> = 0>
  PnrProblemInputs(Dataflow &&dataflow, TechMapping &&techMapping,
                   Fabric &&fabric, Config &&config,
                   mapping::ArtifactIdentity resolvedConfigIdentity,
                   MappingConstraintSetInput constraints)
      : dataflow(dataflow), techMapping(techMapping), fabric(fabric),
        config(config),
        resolvedConfigIdentity(std::move(resolvedConfigIdentity)),
        constraints(std::move(constraints)) {}

  const mapping::DataflowProgramView &dataflow;
  const mapping::ValidatedTechMapping &techMapping;
  const mapping::FabricHardwareView &fabric;
  const ResolvedPnrConfigView &config;
  mapping::ArtifactIdentity resolvedConfigIdentity;
  MappingConstraintSetInput constraints;
};

enum class PnrProblemInputErrorCode {
  TechMappingDataflowIdentityMismatch,
  TechMappingFabricIdentityMismatch,
  ConstraintSetDataflowIdentityMismatch,
  ConstraintSetTechMappingIdentityMismatch,
  ConstraintSetFabricIdentityMismatch,
};

class PnrProblemInputError final
    : public llvm::ErrorInfo<PnrProblemInputError> {
public:
  static char ID;

  PnrProblemInputError(PnrProblemInputErrorCode code,
                       mapping::ArtifactIdentity expectedIdentity,
                       mapping::ArtifactIdentity actualIdentity,
                       std::string message)
      : code_(code), expectedIdentity_(std::move(expectedIdentity)),
        actualIdentity_(std::move(actualIdentity)),
        message_(std::move(message)) {}

  PnrProblemInputErrorCode code() const { return code_; }
  const mapping::ArtifactIdentity &expectedIdentity() const {
    return expectedIdentity_;
  }
  const mapping::ArtifactIdentity &actualIdentity() const {
    return actualIdentity_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  PnrProblemInputErrorCode code_;
  mapping::ArtifactIdentity expectedIdentity_;
  mapping::ArtifactIdentity actualIdentity_;
  std::string message_;
};

llvm::Error validatePnrProblemInputs(const PnrProblemInputs &inputs);

} // namespace loom::pnr

#endif // LOOM_PNR_PNRPROBLEMINPUTS_H
