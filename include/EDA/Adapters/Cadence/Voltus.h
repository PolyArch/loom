#ifndef LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H
#define LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H

#include "EDA/Adapters/Cadence/Common.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "Hardware/Implementation/DefPhysical.h"

#include <string>
#include <vector>

namespace loom::eda::cadence {

struct VoltusRailObservation final {
  evaluation::DecimalValue maximumVoltageDropVolts;
};

const CadenceInvocationDescriptor &voltusRailDescriptor();

llvm::Expected<VoltusRailObservation>
parseVoltusRailObservation(llvm::StringRef contents);

struct VoltusRailInvocationConfiguration final {
  std::string top;
  std::vector<std::string> netlists;
  std::vector<std::string> generationConstraints;
  std::string physicalDatabase;
  hardware::DefSingleSupplyNetwork supplyNetwork;
  evaluation::models::CompleteRailAnalysisConfiguration analysis;
};

llvm::Expected<std::string>
renderVoltusRailDriver(const VoltusRailInvocationConfiguration &configuration,
                       llvm::ArrayRef<std::string> powerGridLibraryEntrypoints);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVoltusRailBundleSpec(
    const CadenceBundleInputs &inputs,
    const VoltusRailInvocationConfiguration &configuration);

llvm::Expected<VoltusRailObservation> importVoltusRailObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs);

/// Registers the exact Evaluation model provider that projects one complete
/// single-domain static rail invocation from its immutable Request and routed
/// HardwareImplementation closure.
llvm::Error registerVoltusRailEvaluationProvider();

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_VOLTUS_H
