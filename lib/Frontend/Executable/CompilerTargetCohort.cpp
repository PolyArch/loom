#include "Frontend/Executable/CompilerTargetBinding.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace {

llvm::Error cohortError(llvm::StringRef marker, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 marker + ": " + message);
}

} // namespace

llvm::Expected<SystemCompilerTargetBindings>
resolveSystemCompilerTargetBindings(const fabric::FinalizedFabricRoot &system,
                                    const CompilerTargetPolicy &policy,
                                    const ArtifactStore &store) {
  auto systemView = fabric::requireSystemRoot(system.view());
  if (!systemView)
    return systemView.takeError();

  std::optional<fabric::HostCoreOccurrenceRef> host;
  std::vector<fabric::InstructionCoreContextRef> instructionCores;
  for (std::uint64_t id = 0;; ++id) {
    const std::optional<fabric::FabricEntityKind> kind =
        system.view().entityKind(id);
    if (!kind)
      break;
    if (*kind == fabric::FabricEntityKind::HostCoreOccurrence) {
      if (host)
        return cohortError("system_compiler_target_invalid",
                           "System contains more than one HostCore");
      host = fabric::HostCoreOccurrenceRef(id);
      continue;
    }
    if (*kind == fabric::FabricEntityKind::AccCoreOccurrence)
      instructionCores.push_back({fabric::AccCoreOccurrenceRef(id)});
  }
  if (!host)
    return cohortError("system_compiler_target_invalid",
                       "System contains no HostCore");
  if (instructionCores.empty())
    return cohortError("system_compiler_target_invalid",
                       "System contains no AccCore InstructionCore");

  const ArtifactIdentity &fabricIdentity = system.reference().artifact;
  auto hostBinding = resolveCompilerTargetBinding(
      CompilerProcessorArchitectureRef::host({fabricIdentity, *host}), policy,
      store);
  if (!hostBinding)
    return hostBinding.takeError();

  struct GroupDraft final {
    std::vector<CompilerProcessorArchitectureRef::Instruction> processors;
  };
  std::vector<GroupDraft> groups;
  std::map<std::vector<std::uint8_t>, std::size_t> groupByArchitecture;
  for (const fabric::InstructionCoreContextRef &core : instructionCores) {
    const auto *architecture = systemView->instructionCoreArchitecture(core);
    if (!architecture)
      return cohortError("system_compiler_target_invalid",
                         "InstructionCore has no Architectural Contract");
    auto architectureBytes =
        fabric::encodeInstructionCoreArchitecturalContract(*architecture);
    if (!architectureBytes)
      return architectureBytes.takeError();
    const CompilerProcessorArchitectureRef::Instruction processor{
        fabricIdentity, core};
    const auto [position, inserted] = groupByArchitecture.try_emplace(
        std::move(*architectureBytes), groups.size());
    if (!inserted) {
      groups[position->second].processors.push_back(processor);
      continue;
    }
    groups.push_back({{processor}});
  }

  std::vector<InstructionCompilerTargetGroup> resolvedGroups;
  resolvedGroups.reserve(groups.size());
  for (GroupDraft &group : groups) {
    auto binding = resolveCompilerTargetBinding(
        CompilerProcessorArchitectureRef::instruction(group.processors.front()),
        policy, store);
    if (!binding)
      return binding.takeError();
    resolvedGroups.push_back(InstructionCompilerTargetGroup(
        std::move(*binding), std::move(group.processors)));
  }
  return SystemCompilerTargetBindings(std::move(*hostBinding),
                                      std::move(resolvedGroups));
}

llvm::Error validateModuleCompilerTarget(const llvm::Module &module,
                                         const CompilerTargetBinding &binding) {
  const std::string moduleTriple = module.getTargetTriple().str();
  if (moduleTriple.empty())
    return cohortError("module_target_triple_missing",
                       "LLVM module has no target triple");
  if (moduleTriple != binding.targetTriple())
    return cohortError("module_target_triple_mismatch",
                       "LLVM module target triple '" + moduleTriple +
                           "' does not equal binding triple '" +
                           binding.targetTriple() + "'");

  if (module.getDataLayoutStr().empty())
    return cohortError("module_data_layout_missing",
                       "LLVM module has no DataLayout");
  auto moduleLayout = llvm::DataLayout::parse(module.getDataLayoutStr());
  if (!moduleLayout)
    return cohortError("module_data_layout_invalid",
                       llvm::toString(moduleLayout.takeError()));
  auto bindingLayout = llvm::DataLayout::parse(binding.dataLayout());
  if (!bindingLayout)
    return cohortError("binding_data_layout_invalid",
                       llvm::toString(bindingLayout.takeError()));
  if (*moduleLayout != *bindingLayout)
    return cohortError("module_data_layout_mismatch",
                       "LLVM module DataLayout is not structurally compatible "
                       "with the exact CompilerTargetBinding");
  return llvm::Error::success();
}

} // namespace loom
