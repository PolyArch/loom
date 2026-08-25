#ifndef LOOM_DSE_MODELPARAMETERCALIBRATIONACQUISITION_H
#define LOOM_DSE_MODELPARAMETERCALIBRATIONACQUISITION_H

#include "DSE/EvidenceObligationSetConfig.h"
#include "DSE/PromotionAcquisition.h"

#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

enum class ModelParameterCalibrationTarget : std::uint8_t {
  Fpa = 0,
  SystemRuntime = 1,
};

const PromotionAcquisitionDescriptor &
modelParameterCalibrationPromotionAcquisitionDescriptor(
    ModelParameterCalibrationTarget target, CalibrationPartitionRole partition);

llvm::Error registerModelParameterCalibrationPromotionAcquisitions();

llvm::Expected<EvidenceObligationTemplate>
prepareModelParameterCalibrationEvidenceObligationTemplate(
    ModelParameterCalibrationTarget target, CalibrationPartitionRole partition,
    const ResolvedConfig &resolvedConfig);

} // namespace loom::dse

#endif // LOOM_DSE_MODELPARAMETERCALIBRATIONACQUISITION_H
