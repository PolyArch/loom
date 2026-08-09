#ifndef LOOM_EDA_ADAPTERS_CADENCE_INNOVUS_H
#define LOOM_EDA_ADAPTERS_CADENCE_INNOVUS_H

#include "EDA/Adapters/Cadence/Common.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::eda::cadence {

struct InnovusPhysicalSnapshot final {
  hardware::RepresentationPhysicalStage stage;
  std::string netlistVerilog;
  std::string designExchangeFormat;
  std::string generationConstraints;
};

const CadenceInvocationDescriptor &innovusDescriptor();

llvm::Expected<std::string> renderInnovusDriver(llvm::StringRef top,
                                                llvm::StringRef gateNetlist,
                                                llvm::StringRef floorplan,
                                                llvm::StringRef technologyLef,
                                                llvm::StringRef cellLef);

llvm::Expected<std::string>
renderInnovusMmmcDriver(llvm::StringRef generationConstraint,
                        llvm::StringRef timingLiberty,
                        llvm::StringRef qrcTechnologyFile);

llvm::Expected<InnovusPhysicalSnapshot> parseInnovusPhysicalSnapshot(
    llvm::StringRef netlist, llvm::StringRef designExchangeFormat,
    llvm::StringRef generationConstraints, llvm::StringRef top,
    hardware::RepresentationPhysicalStage stage);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeInnovusBundleSpec(const CadenceBundleInputs &inputs, llvm::StringRef top,
                      llvm::StringRef gateNetlist,
                      llvm::StringRef generationConstraint,
                      llvm::StringRef floorplan);

llvm::Expected<InnovusPhysicalSnapshot> importInnovusPhysicalSnapshot(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs, llvm::StringRef top);

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishInnovusPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const InnovusPhysicalSnapshot &snapshot, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_INNOVUS_H
