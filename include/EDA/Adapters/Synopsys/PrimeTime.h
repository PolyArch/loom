#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMETIME_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMETIME_H

#include "EDA/Adapters/Synopsys/Common.h"
#include "Evaluation/NumericValue.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::synopsys {

struct PrimeTimeObservation final {
  evaluation::DecimalValue clockPeriodSeconds;
  evaluation::DecimalValue limitingClockFrequencyHz;
};

const SynopsysInvocationDescriptor &primeTimeDescriptor();

llvm::Expected<std::string>
renderPrimeTimeDriver(llvm::StringRef top, llvm::StringRef gateNetlist,
                      llvm::StringRef generationConstraint,
                      llvm::StringRef timingLibrary);

llvm::Expected<PrimeTimeObservation>
parsePrimeTimeObservation(llvm::StringRef contents);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makePrimeTimeBundleSpec(const SynopsysBundleInputs &inputs, llvm::StringRef top,
                        llvm::StringRef gateNetlist,
                        llvm::StringRef generationConstraint);

llvm::Expected<PrimeTimeObservation> importPrimeTimeObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_PRIMETIME_H
