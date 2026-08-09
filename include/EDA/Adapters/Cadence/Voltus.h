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

/// Voltus rail preparation remains closed until ExternalTool can freeze a
/// complete directory-valued PGV input. Treating one member file as the
/// directory identity would allow undeclared commercial inputs to affect the
/// result.
llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVoltusRailBundleSpec(const CadenceBundleInputs &inputs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H
