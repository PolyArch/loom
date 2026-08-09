#ifndef LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H
#define LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H

#include "EDA/Adapters/Cadence/Common.h"

namespace loom::eda::cadence {

struct VoltusRailObservation final {
  evaluation::DecimalValue maximumVoltageDropVolts;
};

const CadenceInvocationDescriptor &voltusRailDescriptor();

llvm::Expected<VoltusRailObservation>
parseVoltusRailObservation(llvm::StringRef contents);

/// Voltus rail preparation requires a complete directory-valued PGV input.
/// It remains closed until the exact Evaluation model projects its analysis
/// method, activity basis, network coverage, and supply conditions into a
/// provider configuration.
llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVoltusRailBundleSpec(const CadenceBundleInputs &inputs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H
