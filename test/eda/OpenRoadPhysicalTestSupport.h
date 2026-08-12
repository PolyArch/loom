#ifndef LOOM_TEST_EDA_OPENROADPHYSICALTESTSUPPORT_H
#define LOOM_TEST_EDA_OPENROADPHYSICALTESTSUPPORT_H

#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/CandidateGenerator.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace loom::eda::open_source::test {

struct OpenRoadTechnologyFixture final {
  std::string technologyLef;
  std::string cellLef;
  std::string liberty;
  std::string netlist;
  std::string constraints;
  std::string blackBoxContract;
  std::vector<std::string> unresolvedCellModules;
  OpenRoadPlacementParameters placement;
};

OpenRoadTechnologyFixture syntheticOpenRoadTechnologyFixture();

llvm::Expected<OpenRoadTechnologyFixture>
loadSaed32OpenRoadTechnologyFixture(const std::filesystem::path &technologyLef,
                                    const std::filesystem::path &cellLef,
                                    const std::filesystem::path &liberty);

llvm::Expected<OpenRoadTechnologyFixture>
loadGpdk045OpenRoadTechnologyFixture(const std::filesystem::path &technologyLef,
                                     const std::filesystem::path &cellLef,
                                     const std::filesystem::path &liberty);

struct OpenRoadGateFixture final {
  hardware::ExternalImplementationContractCatalog contracts;
  hardware::FinalizedHardwareImplementation gate;
  platform::FinalizedImplementationPlatform platform;
  OpenRoadPlacedConfig config;
  std::filesystem::path technologyLefPath;
  std::filesystem::path cellLefPath;
  std::filesystem::path libertyPath;
};

llvm::Expected<OpenRoadGateFixture> makeOpenRoadGateFixture(
    const std::filesystem::path &root, const ArtifactStore &artifacts,
    const BlobStore &blobs, llvm::StringRef providerBuild,
    const OpenRoadTechnologyFixture &technology,
    llvm::StringRef designIdentity = "openroad-routed-fixture",
    std::uint32_t designPortBitWidth = 0);

struct OpenRoadRouteHarness final {
  std::vector<dse::CandidateGeneratorInputBinding> inputs;
  dse::ResolvedCandidateGeneratorBinding binding;
  external_tool::ExternalToolPreparationContext context;
};

llvm::Expected<OpenRoadRouteHarness>
makeOpenRoadRouteHarness(const std::filesystem::path &bundleRoot,
                         const OpenRoadGateFixture &fixture,
                         const external_tool::LocalToolConfig &localConfig);

external_tool::LocalToolConfig
makeOpenRoadLocalToolConfig(const OpenRoadGateFixture &fixture,
                            const std::filesystem::path &toolExecutable);

OpenRoadResolvedExecution
makeOpenRoadResolvedExecution(llvm::StringRef executable,
                              llvm::StringRef version, bool moduleBound);

enum class AuthoredOpenRoadRouteBehavior {
  Complete,
  ToolFailure,
  MissingOutput,
};

llvm::Expected<std::filesystem::path>
writeAuthoredOpenRoadRouteTool(const std::filesystem::path &root,
                               AuthoredOpenRoadRouteBehavior behavior =
                                   AuthoredOpenRoadRouteBehavior::Complete);

enum class AuthoredOpenRoadStaticFpaBehavior {
  Complete,
  ToolFailure,
  MalformedResult,
};

llvm::Expected<std::filesystem::path> writeAuthoredOpenRoadStaticFpaTool(
    const std::filesystem::path &root,
    AuthoredOpenRoadStaticFpaBehavior behavior =
        AuthoredOpenRoadStaticFpaBehavior::Complete);

llvm::Expected<hardware::FinalizedHardwareImplementation>
runOpenRoadRouteFixture(const OpenRoadGateFixture &fixture,
                        OpenRoadRouteHarness &harness,
                        const OpenRoadResolvedExecution &execution,
                        const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<std::string> readText(const std::filesystem::path &path);
llvm::Error writeText(const std::filesystem::path &path,
                      llvm::StringRef contents, bool executable = false);
ExternalFileFingerprint contentFingerprint(llvm::StringRef contents);

} // namespace loom::eda::open_source::test

#endif // LOOM_TEST_EDA_OPENROADPHYSICALTESTSUPPORT_H
