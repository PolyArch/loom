#include "CompilerTargetBindingInternal.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::detail {
namespace {

llvm::Error targetError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

void initializeTargets() {
  static std::once_flag once;
  std::call_once(once, [] {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
  });
}

llvm::StringRef abiSpelling(fabric::RiscVAbi abi) {
  switch (abi) {
  case fabric::RiscVAbi::Ilp32:
    return "ilp32";
  case fabric::RiscVAbi::Ilp32e:
    return "ilp32e";
  case fabric::RiscVAbi::Ilp32f:
    return "ilp32f";
  case fabric::RiscVAbi::Ilp32d:
    return "ilp32d";
  case fabric::RiscVAbi::Lp64:
    return "lp64";
  case fabric::RiscVAbi::Lp64f:
    return "lp64f";
  case fabric::RiscVAbi::Lp64d:
    return "lp64d";
  }
  llvm_unreachable("unknown RISC-V ABI");
}

llvm::StringRef extensionSpelling(fabric::RiscVExtension extension) {
  switch (extension) {
  case fabric::RiscVExtension::M:
    return "+m";
  case fabric::RiscVExtension::A:
    return "+a";
  case fabric::RiscVExtension::F:
    return "+f";
  case fabric::RiscVExtension::D:
    return "+d";
  case fabric::RiscVExtension::C:
    return "+c";
  case fabric::RiscVExtension::V:
    return "+v";
  case fabric::RiscVExtension::Zicsr:
    return "+zicsr";
  case fabric::RiscVExtension::Zifencei:
    return "+zifencei";
  case fabric::RiscVExtension::Zba:
    return "+zba";
  case fabric::RiscVExtension::Zbb:
    return "+zbb";
  case fabric::RiscVExtension::Zbs:
    return "+zbs";
  case fabric::RiscVExtension::Ztso:
    return "+ztso";
  }
  llvm_unreachable("unknown RISC-V extension");
}

std::string
targetTriple(const fabric::InstructionCoreArchitecturalContract &architecture) {
  std::string arch;
  if (architecture.xlen() == fabric::RiscVXLen::X64)
    arch = architecture.endianness() == fabric::InstructionEndianness::Little
               ? "riscv64"
               : "riscv64be";
  else
    arch = architecture.endianness() == fabric::InstructionEndianness::Little
               ? "riscv32"
               : "riscv32be";
  return llvm::Triple::normalize(arch + "-unknown-elf");
}

std::vector<std::string> targetFeatures(
    const fabric::InstructionCoreArchitecturalContract &architecture) {
  std::vector<std::string> features;
  features.reserve(architecture.extensions().size() + 1);
  if (architecture.base() == fabric::RiscVBase::E)
    features.emplace_back("+e");
  for (fabric::RiscVExtension extension : architecture.extensions())
    features.push_back(extensionSpelling(extension).str());
  return features;
}

std::vector<TargetScopeBinding>
targetScopes(const fabric::InstructionCoreArchitecturalContract &architecture) {
  std::vector<TargetScopeBinding> result;
  result.reserve(architecture.syncScopes().size());
  for (fabric::InstructionSyncScope scope : architecture.syncScopes()) {
    switch (scope) {
    case fabric::InstructionSyncScope::SingleThread:
    case fabric::InstructionSyncScope::Hart:
      result.push_back({scope, "singlethread"});
      break;
    case fabric::InstructionSyncScope::System:
      result.push_back({scope, "system"});
      break;
    }
  }
  return result;
}

std::optional<llvm::Reloc::Model>
relocationModel(fabric::RelocationModel model) {
  switch (model) {
  case fabric::RelocationModel::Static:
    return llvm::Reloc::Static;
  case fabric::RelocationModel::PositionIndependent:
    return llvm::Reloc::PIC_;
  }
  llvm_unreachable("unknown relocation model");
}

std::optional<llvm::CodeModel::Model> codeModel(fabric::RiscVCodeModel model) {
  switch (model) {
  case fabric::RiscVCodeModel::MediumLow:
    return llvm::CodeModel::Small;
  case fabric::RiscVCodeModel::MediumAny:
    return llvm::CodeModel::Medium;
  }
  llvm_unreachable("unknown RISC-V code model");
}

template <typename Ref>
const fabric::InstructionCoreArchitecturalContract *
architectureFor(const fabric::FabricSystemRootView &system, const Ref &ref) {
  return system.instructionCoreArchitecture(ref.entity);
}

} // namespace

llvm::Expected<fabric::InstructionCoreArchitecturalContract>
resolveProcessorArchitecture(const CompilerProcessorArchitectureRef &processor,
                             const ArtifactStore &store) {
  ArtifactRootReference root{fabric::fabricArtifactSchema.identity.str(),
                             fabric::fabricArtifactSchema.version,
                             processor.fabricArtifact()};
  llvm::Expected<fabric::FinalizedFabricRoot> imported =
      fabric::importEntireFabricRoot(root, store);
  if (!imported)
    return imported.takeError();
  llvm::Expected<fabric::FabricSystemRootView> system =
      fabric::requireSystemRoot(imported->view());
  if (!system)
    return system.takeError();

  const fabric::InstructionCoreArchitecturalContract *architecture = std::visit(
      [&](const auto &reference) {
        return architectureFor(*system, reference);
      },
      processor.value());
  if (!architecture)
    return targetError("processor_architecture_ref_invalid",
                       "the reference does not resolve in its exact Fabric");
  return *architecture;
}

llvm::Expected<ReconstructedCompilerTarget> reconstructCompilerTarget(
    const fabric::InstructionCoreArchitecturalContract &architecture,
    fabric::RiscVAbi backendAbi, fabric::RiscVCodeModel selectedCodeModel,
    fabric::RelocationModel selectedRelocationModel,
    llvm::StringRef backendCpu) {
  if (!llvm::is_contained(architecture.abiCapabilities(), backendAbi))
    return targetError("backend_abi_not_admitted",
                       "the Fabric architecture does not admit the selected "
                       "backend ABI");
  if (!llvm::is_contained(architecture.codeModels(), selectedCodeModel))
    return targetError("code_model_not_admitted",
                       "the Fabric architecture does not admit the selected "
                       "code model");
  if (!llvm::is_contained(architecture.relocationModels(),
                          selectedRelocationModel))
    return targetError("relocation_model_not_admitted",
                       "the Fabric architecture does not admit the selected "
                       "relocation model");
  ReconstructedCompilerTarget result;
  result.provider = buildSelectedLlvmProvider();
  result.targetTriple = targetTriple(architecture);
  result.objectFormat = CompilerObjectFormat::Elf;
  result.backendFeatures = targetFeatures(architecture);
  result.targetScopeBindings = targetScopes(architecture);

  auto machine = createCompilerTargetMachine(
      result.targetTriple, backendAbi, selectedCodeModel,
      selectedRelocationModel, backendCpu, result.backendFeatures);
  if (!machine)
    return machine.takeError();
  result.dataLayout = (*machine)->createDataLayout().getStringRepresentation();
  if (result.dataLayout.empty())
    return targetError("compiler_target_reconstruction_mismatch",
                       "the selected target machine produced no DataLayout");
  return result;
}

llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
createCompilerTargetMachine(llvm::StringRef targetTriple,
                            fabric::RiscVAbi backendAbi,
                            fabric::RiscVCodeModel selectedCodeModel,
                            fabric::RelocationModel selectedRelocationModel,
                            llvm::StringRef backendCpu,
                            llvm::ArrayRef<std::string> backendFeatures) {
  if (backendCpu.empty())
    return targetError("backend_cpu_invalid", "backend CPU must be nonempty");

  initializeTargets();
  std::string lookupError;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(targetTriple), lookupError);
  if (!target)
    return targetError("compiler_target_provider_unavailable",
                       "the pinned LLVM provider has no target for '" +
                           targetTriple + "': " + lookupError);
  const std::string featureString = llvm::join(backendFeatures, ",");
  std::unique_ptr<llvm::MCSubtargetInfo> subtarget(
      target->createMCSubtargetInfo(llvm::Triple(targetTriple), backendCpu,
                                    featureString));
  if (!subtarget || !subtarget->isCPUStringValid(backendCpu) ||
      !subtarget->checkFeatures(featureString))
    return targetError("compiler_target_provider_unavailable",
                       "the pinned LLVM provider rejects CPU/features '" +
                           backendCpu + "'/'" + featureString + "'");

  llvm::TargetOptions options;
  options.MCOptions.ABIName = abiSpelling(backendAbi).str();
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      llvm::Triple(targetTriple), backendCpu, featureString, options,
      relocationModel(selectedRelocationModel), codeModel(selectedCodeModel)));
  if (!machine)
    return targetError("compiler_target_provider_unavailable",
                       "the pinned LLVM provider could not construct the "
                       "selected target machine");
  if (machine->getTargetTriple().normalize() != targetTriple ||
      machine->getTargetCPU() != backendCpu ||
      machine->getTargetFeatureString() != featureString ||
      machine->getRelocationModel() !=
          *relocationModel(selectedRelocationModel) ||
      machine->getCodeModel() != *codeModel(selectedCodeModel))
    return targetError("compiler_target_reconstruction_mismatch",
                       "the pinned LLVM provider changed an exact target "
                       "selection field");
  return machine;
}

} // namespace loom::detail
