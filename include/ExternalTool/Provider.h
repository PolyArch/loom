#ifndef LOOM_EXTERNALTOOL_PROVIDER_H
#define LOOM_EXTERNALTOOL_PROVIDER_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::external_tool {

struct ExternalToolProviderDescriptor {
  ToolProviderDescriptor binding;
  ToolVersionProbe versionProbe;
  ToolRuntimeCompatibility runtimeCompatibility;
};

struct BackendToolReleaseProfile final {
  std::string conformanceFeature;
  std::optional<std::string> moduleAlias;
  ToolVersionProbe exactVersionProbe;
};

struct BackendToolCatalogEntry final {
  std::string officialProductName;
  ExternalToolProviderDescriptor provider;
  std::vector<BackendToolReleaseProfile> validatedReleases;
};

llvm::ArrayRef<BackendToolCatalogEntry> backendToolCatalog();
const BackendToolCatalogEntry *findBackendTool(llvm::StringRef logicalToolKey);
llvm::Error validateBackendToolCatalog();

/// The catalog-owned qualification relation: the validated release whose
/// exact version probe accepts one resolved version line, or null when the
/// tool key is unknown or no validated release accepts that version. Adapters
/// qualify a resolved binding through this relation, never through a private
/// version literal.
const BackendToolReleaseProfile *
findValidatedRelease(llvm::StringRef logicalToolKey,
                     llvm::StringRef resolvedVersion);

const ExternalToolProviderDescriptor &polyArchContainerProvider();
const ExternalToolProviderDescriptor &verilatorProvider();
const ExternalToolProviderDescriptor &yosysProvider();
const ExternalToolProviderDescriptor &openRoadProvider();
const ExternalToolProviderDescriptor &gem5Provider();
const ExternalToolProviderDescriptor &vcsProvider();
const ExternalToolProviderDescriptor &designCompilerProvider();
const ExternalToolProviderDescriptor &fusionCompilerProvider();
const ExternalToolProviderDescriptor &primeTimeProvider();
const ExternalToolProviderDescriptor &xceliumProvider();
const ExternalToolProviderDescriptor &genusProvider();
const ExternalToolProviderDescriptor &innovusProvider();
const ExternalToolProviderDescriptor &joulesProvider();
const ExternalToolProviderDescriptor &tempusProvider();
const ExternalToolProviderDescriptor &voltusProvider();
const ExternalToolProviderDescriptor &vivadoProvider();
const ExternalToolProviderDescriptor &quartusPrimeProvider();

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_PROVIDER_H
