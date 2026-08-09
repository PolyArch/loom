#ifndef LOOM_EDA_ADAPTERS_CADENCE_JOULES_H
#define LOOM_EDA_ADAPTERS_CADENCE_JOULES_H

#include "EDA/Adapters/Cadence/Common.h"

namespace loom::eda::cadence {

struct JoulesPowerObservation final {
  evaluation::DecimalValue dynamicPowerWatts;
  evaluation::DecimalValue leakagePowerWatts;
};

const CadenceInvocationDescriptor &joulesPowerDescriptor();

llvm::Expected<std::string>
renderJoulesPowerDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                        llvm::StringRef generationConstraint,
                        llvm::StringRef activity, llvm::StringRef activityScope,
                        llvm::StringRef timingLiberty);

llvm::Expected<JoulesPowerObservation>
parseJoulesPowerObservation(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeJoulesPowerBundleSpec(const CadenceBundleInputs &inputs,
                          llvm::StringRef top, llvm::StringRef gateNetlist,
                          llvm::StringRef generationConstraint,
                          llvm::StringRef activity,
                          llvm::StringRef activityScope);

llvm::Expected<JoulesPowerObservation> importJoulesPowerObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_JOULES_H
