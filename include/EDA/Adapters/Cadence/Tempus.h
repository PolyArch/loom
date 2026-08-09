#ifndef LOOM_EDA_ADAPTERS_CADENCE_TEMPUS_H
#define LOOM_EDA_ADAPTERS_CADENCE_TEMPUS_H

#include "EDA/Adapters/Cadence/Common.h"

namespace loom::eda::cadence {

struct TempusTimingObservation final {
  evaluation::DecimalValue clockPeriodSeconds;
  evaluation::DecimalValue limitingClockFrequencyHz;
};

const CadenceInvocationDescriptor &tempusTimingDescriptor();

llvm::Expected<std::string>
renderTempusTimingDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                         llvm::StringRef generationConstraint,
                         llvm::StringRef physicalDatabase,
                         llvm::StringRef timingLiberty);

llvm::Expected<TempusTimingObservation>
parseTempusTimingObservation(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeTempusTimingBundleSpec(const CadenceBundleInputs &inputs,
                           llvm::StringRef top, llvm::StringRef gateNetlist,
                           llvm::StringRef generationConstraint,
                           llvm::StringRef physicalDatabase);

llvm::Expected<TempusTimingObservation> importTempusTimingObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_TEMPUS_H
