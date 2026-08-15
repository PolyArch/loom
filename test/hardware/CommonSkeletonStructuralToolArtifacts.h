#ifndef LOOM_TEST_HARDWARE_COMMONSKELETONSTRUCTURALTOOLARTIFACTS_H
#define LOOM_TEST_HARDWARE_COMMONSKELETONSTRUCTURALTOOLARTIFACTS_H

#include "ConfigurationTransportTestSupport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace loom::hardware::test {

using PortableConfigurationImage =
    std::pair<PortableConfigurationTarget, std::vector<std::uint8_t>>;

llvm::Error writeBoundaryStructuralToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog);

llvm::Error writeSpatialHierarchyToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog,
    llvm::ArrayRef<PortableConfigurationImage> inactiveConfigurations);

llvm::Error writeRepeatedSpatialCoreToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog,
    const PortableConfigurationTarget &target,
    llvm::ArrayRef<std::uint8_t> activeImage);

} // namespace loom::hardware::test

#endif // LOOM_TEST_HARDWARE_COMMONSKELETONSTRUCTURALTOOLARTIFACTS_H
