#ifndef LOOM_EXTERNALTOOL_PROVIDER_H
#define LOOM_EXTERNALTOOL_PROVIDER_H

#include "ExternalTool/Binding.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"

namespace loom::external_tool {

struct ExternalToolProviderDescriptor {
  ToolProviderDescriptor binding;
  ToolVersionProbe versionProbe;
  ToolRuntimeCompatibility runtimeCompatibility;
};

const ExternalToolProviderDescriptor &polyArchContainerProvider();
const ExternalToolProviderDescriptor &verilatorProvider();
const ExternalToolProviderDescriptor &yosysProvider();
const ExternalToolProviderDescriptor &openRoadProvider();
const ExternalToolProviderDescriptor &vcsProvider();
const ExternalToolProviderDescriptor &designCompilerProvider();
const ExternalToolProviderDescriptor &fusionCompilerProvider();
const ExternalToolProviderDescriptor &xceliumProvider();
const ExternalToolProviderDescriptor &genusProvider();
const ExternalToolProviderDescriptor &innovusProvider();
const ExternalToolProviderDescriptor &vivadoProvider();
const ExternalToolProviderDescriptor &quartusPrimeProvider();

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_PROVIDER_H
