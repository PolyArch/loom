#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMEPOWER_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMEPOWER_H

#include "EDA/Adapters/Synopsys/Common.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::synopsys {

struct PrimePowerObservation final {
  evaluation::DecimalValue dynamicPowerWatts;
  evaluation::DecimalValue leakagePowerWatts;
};

const SynopsysInvocationDescriptor &primePowerDescriptor();

llvm::Expected<std::string> renderPrimePowerDriver(
    llvm::StringRef top, llvm::StringRef gateNetlist,
    llvm::StringRef generationConstraint, llvm::StringRef activity,
    llvm::StringRef activityStripPath, llvm::StringRef powerLibrary);

llvm::Expected<PrimePowerObservation>
parsePrimePowerObservation(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makePrimePowerBundleSpec(const SynopsysBundleInputs &inputs,
                         llvm::StringRef top, llvm::StringRef gateNetlist,
                         llvm::StringRef generationConstraint,
                         llvm::StringRef activity,
                         llvm::StringRef activityStripPath);

llvm::Expected<PrimePowerObservation> importPrimePowerObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMEPOWER_H
