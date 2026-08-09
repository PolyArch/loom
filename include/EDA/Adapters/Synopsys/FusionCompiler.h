#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H

#include "EDA/Adapters/Synopsys/Common.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::synopsys {

struct FusionCompilerPhysicalSnapshot final {
  hardware::RepresentationPhysicalStage stage;
  std::string netlistVerilog;
  std::string designExchangeFormat;
  std::string generationConstraints;
};

const SynopsysInvocationDescriptor &fusionCompilerDescriptor();

llvm::Expected<std::string> renderFusionCompilerDriver(
    llvm::StringRef top, llvm::StringRef gateNetlist,
    llvm::StringRef generationConstraint, llvm::StringRef floorplan,
    llvm::StringRef referenceLibrary, llvm::StringRef earlyParasiticTech,
    llvm::StringRef lateParasiticTech, llvm::StringRef parasiticLayerMap);

llvm::Expected<FusionCompilerPhysicalSnapshot>
parseFusionCompilerPhysicalSnapshot(
    llvm::StringRef netlist, llvm::StringRef designExchangeFormat,
    llvm::StringRef generationConstraints, llvm::StringRef top,
    hardware::RepresentationPhysicalStage stage);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeFusionCompilerBundleSpec(const SynopsysBundleInputs &inputs,
                             llvm::StringRef top, llvm::StringRef gateNetlist,
                             llvm::StringRef generationConstraint,
                             llvm::StringRef floorplan);

llvm::Expected<FusionCompilerPhysicalSnapshot>
importFusionCompilerPhysicalSnapshot(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top);

llvm::Error fusionCompilerPublicationUnavailable();

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H
