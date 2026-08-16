#ifndef LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDING_H
#define LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDING_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class Module;
}

namespace loom {

class ArtifactStore;
namespace fabric {
class FinalizedFabricRoot;
}

inline constexpr ArtifactSchemaDescriptor compilerTargetBindingSchema{
    "loom.compiler_target_binding", SchemaVersion{1, 0}};

class ArchitectureFingerprint final {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  static llvm::Expected<ArchitectureFingerprint>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const ArchitectureFingerprint &lhs,
                         const ArchitectureFingerprint &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ArchitectureFingerprint &lhs,
                         const ArchitectureFingerprint &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit ArchitectureFingerprint(Storage bytes) : bytes_(bytes) {}

  friend ArchitectureFingerprint computeArchitectureFingerprint(
      const fabric::InstructionCoreArchitecturalContract &contract);

  Storage bytes_;
};

ArchitectureFingerprint computeArchitectureFingerprint(
    const fabric::InstructionCoreArchitecturalContract &contract);
std::string
formatArchitectureFingerprintHex(const ArchitectureFingerprint &fingerprint);
llvm::Expected<ArchitectureFingerprint>
parseArchitectureFingerprintHex(llvm::StringRef spelling);

class CompilerProcessorArchitectureRef final {
public:
  using Host = ArtifactReference<fabric::HostCoreOccurrenceRef>;
  using Instruction = ArtifactReference<fabric::InstructionCoreContextRef>;

  static CompilerProcessorArchitectureRef host(Host reference);
  static CompilerProcessorArchitectureRef instruction(Instruction reference);

  bool isHost() const { return std::holds_alternative<Host>(value_); }
  const std::variant<Host, Instruction> &value() const { return value_; }
  const ArtifactIdentity &fabricArtifact() const;

  friend bool operator==(const CompilerProcessorArchitectureRef &lhs,
                         const CompilerProcessorArchitectureRef &rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend bool operator!=(const CompilerProcessorArchitectureRef &lhs,
                         const CompilerProcessorArchitectureRef &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit CompilerProcessorArchitectureRef(std::variant<Host, Instruction> v)
      : value_(std::move(v)) {}

  std::variant<Host, Instruction> value_;
};

enum class CompilerObjectFormat : std::uint32_t { Elf };
enum class CompilerSupportRole : std::uint32_t {
  StartupObject,
  RuntimeLibrary,
  BuiltinLibrary,
};
enum class CompilerSupportLinkMode : std::uint32_t { Static, Dynamic };

struct TargetScopeBinding final {
  fabric::InstructionSyncScope architectureScope;
  std::string llvmSyncScopeId;

  friend bool operator==(const TargetScopeBinding &lhs,
                         const TargetScopeBinding &rhs) {
    return lhs.architectureScope == rhs.architectureScope &&
           lhs.llvmSyncScopeId == rhs.llvmSyncScopeId;
  }
};

struct CompilerSupportComponent final {
  CompilerSupportRole role;
  std::string interfaceAbiIdentity;
  BlobDigest contentBlob;
  CompilerSupportLinkMode linkMode;

  friend bool operator==(const CompilerSupportComponent &lhs,
                         const CompilerSupportComponent &rhs) {
    return lhs.role == rhs.role &&
           lhs.interfaceAbiIdentity == rhs.interfaceAbiIdentity &&
           lhs.contentBlob == rhs.contentBlob && lhs.linkMode == rhs.linkMode;
  }
};

struct CompilerTargetPolicy final {
  fabric::RiscVAbi backendAbi;
  fabric::RiscVCodeModel codeModel;
  fabric::RelocationModel relocationModel;
  std::string backendCpu;
  std::vector<CompilerSupportComponent> supportComponents;
};

/// The first portable stored-program policy. It is kept beside target
/// reconstruction so product drivers and focused tools cannot independently
/// spell the ABI, code model, relocation model, or backend CPU.
CompilerTargetPolicy portableRiscV64CompilerTargetPolicy();

/// Invocation-only projection of one exact binding into the public Clang/LLD
/// command-line vocabulary. These strings are boundary spellings, not a
/// second target capability authority.
struct CompilerTargetCommandLineProjection final {
  std::string targetTriple;
  std::string architecture;
  std::string abi;
  std::string codeModel;
  std::string backendCpu;
  std::string ltoFeatures;
  bool positionIndependent = false;
};

class CompilerTargetBinding final {
public:
  const CompilerProcessorArchitectureRef &processorArchitecture() const {
    return processorArchitecture_;
  }
  const ArchitectureFingerprint &architectureFingerprint() const {
    return architectureFingerprint_;
  }
  const LlvmProviderIdentity &compilerProvider() const { return provider_; }
  llvm::StringRef targetTriple() const { return targetTriple_; }
  llvm::StringRef dataLayout() const { return dataLayout_; }
  fabric::RiscVAbi backendAbi() const { return backendAbi_; }
  CompilerObjectFormat objectFormat() const { return objectFormat_; }
  fabric::RiscVCodeModel codeModel() const { return codeModel_; }
  fabric::RelocationModel relocationModel() const { return relocationModel_; }
  llvm::StringRef backendCpu() const { return backendCpu_; }
  llvm::ArrayRef<std::string> backendFeatures() const {
    return backendFeatures_;
  }
  llvm::ArrayRef<TargetScopeBinding> targetScopeBindings() const {
    return targetScopeBindings_;
  }
  llvm::ArrayRef<CompilerSupportComponent> supportComponents() const {
    return supportComponents_;
  }

private:
  CompilerTargetBinding(CompilerProcessorArchitectureRef processorArchitecture,
                        ArchitectureFingerprint architectureFingerprint,
                        LlvmProviderIdentity provider, std::string targetTriple,
                        std::string dataLayout, fabric::RiscVAbi backendAbi,
                        CompilerObjectFormat objectFormat,
                        fabric::RiscVCodeModel codeModel,
                        fabric::RelocationModel relocationModel,
                        std::string backendCpu,
                        std::vector<std::string> backendFeatures,
                        std::vector<TargetScopeBinding> targetScopeBindings,
                        std::vector<CompilerSupportComponent> supportComponents)
      : processorArchitecture_(std::move(processorArchitecture)),
        architectureFingerprint_(architectureFingerprint),
        provider_(std::move(provider)), targetTriple_(std::move(targetTriple)),
        dataLayout_(std::move(dataLayout)), backendAbi_(backendAbi),
        objectFormat_(objectFormat), codeModel_(codeModel),
        relocationModel_(relocationModel), backendCpu_(std::move(backendCpu)),
        backendFeatures_(std::move(backendFeatures)),
        targetScopeBindings_(std::move(targetScopeBindings)),
        supportComponents_(std::move(supportComponents)) {}

  CompilerProcessorArchitectureRef processorArchitecture_;
  ArchitectureFingerprint architectureFingerprint_;
  LlvmProviderIdentity provider_;
  std::string targetTriple_;
  std::string dataLayout_;
  fabric::RiscVAbi backendAbi_;
  CompilerObjectFormat objectFormat_;
  fabric::RiscVCodeModel codeModel_;
  fabric::RelocationModel relocationModel_;
  std::string backendCpu_;
  std::vector<std::string> backendFeatures_;
  std::vector<TargetScopeBinding> targetScopeBindings_;
  std::vector<CompilerSupportComponent> supportComponents_;

  friend llvm::Expected<CompilerTargetBinding>
  decodeCompilerTargetBinding(llvm::StringRef, const ArtifactStore &);
  friend llvm::Expected<class FinalizedCompilerTargetBinding>
  resolveCompilerTargetBinding(const CompilerProcessorArchitectureRef &,
                               const CompilerTargetPolicy &,
                               const ArtifactStore &);
};

class FinalizedCompilerTargetBinding final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const CompilerTargetBinding &binding() const { return binding_; }

private:
  FinalizedCompilerTargetBinding(ArtifactRootReference reference,
                                 CanonicalSemanticBytes canonicalBytes,
                                 CompilerTargetBinding binding)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)),
        binding_(std::move(binding)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  CompilerTargetBinding binding_;

  friend llvm::Expected<FinalizedCompilerTargetBinding>
  resolveCompilerTargetBinding(const CompilerProcessorArchitectureRef &,
                               const CompilerTargetPolicy &,
                               const ArtifactStore &);
  friend llvm::Expected<FinalizedCompilerTargetBinding>
  importCompilerTargetBinding(const ArtifactRootReference &,
                              const ArtifactStore &);
};

class InstructionCompilerTargetGroup final {
public:
  const FinalizedCompilerTargetBinding &binding() const { return binding_; }
  llvm::ArrayRef<CompilerProcessorArchitectureRef::Instruction>
  processors() const {
    return processors_;
  }

private:
  InstructionCompilerTargetGroup(
      FinalizedCompilerTargetBinding binding,
      std::vector<CompilerProcessorArchitectureRef::Instruction> processors)
      : binding_(std::move(binding)), processors_(std::move(processors)) {}

  FinalizedCompilerTargetBinding binding_;
  std::vector<CompilerProcessorArchitectureRef::Instruction> processors_;

  friend llvm::Expected<class SystemCompilerTargetBindings>
  resolveSystemCompilerTargetBindings(const fabric::FinalizedFabricRoot &,
                                      const CompilerTargetPolicy &,
                                      const ArtifactStore &);
};

/// Invocation-local projection of the exact CompilerTargetBindings selected
/// for every stored-program engine of one finalized System Fabric. Equal
/// same-kind InstructionCore contracts share one binding group. Each binding
/// remains an independently exact Artifact; this aggregate is not persistent.
class SystemCompilerTargetBindings final {
public:
  const FinalizedCompilerTargetBinding &host() const { return host_; }
  llvm::ArrayRef<InstructionCompilerTargetGroup> instructionGroups() const {
    return instructionGroups_;
  }

private:
  SystemCompilerTargetBindings(
      FinalizedCompilerTargetBinding host,
      std::vector<InstructionCompilerTargetGroup> instructionGroups)
      : host_(std::move(host)),
        instructionGroups_(std::move(instructionGroups)) {}

  FinalizedCompilerTargetBinding host_;
  std::vector<InstructionCompilerTargetGroup> instructionGroups_;

  friend llvm::Expected<SystemCompilerTargetBindings>
  resolveSystemCompilerTargetBindings(const fabric::FinalizedFabricRoot &,
                                      const CompilerTargetPolicy &,
                                      const ArtifactStore &);
};

llvm::Expected<FinalizedCompilerTargetBinding>
resolveCompilerTargetBinding(const CompilerProcessorArchitectureRef &processor,
                             const CompilerTargetPolicy &policy,
                             const ArtifactStore &store);

llvm::Expected<FinalizedCompilerTargetBinding>
importCompilerTargetBinding(const ArtifactRootReference &reference,
                            const ArtifactStore &store);

llvm::Error requireCompilerTargetCompatibility(
    const CompilerTargetBinding &binding,
    const CompilerProcessorArchitectureRef &processor,
    const ArtifactStore &store);

llvm::Expected<SystemCompilerTargetBindings>
resolveSystemCompilerTargetBindings(const fabric::FinalizedFabricRoot &system,
                                    const CompilerTargetPolicy &policy,
                                    const ArtifactStore &store);

llvm::Expected<CompilerTargetCommandLineProjection>
projectCompilerTargetCommandLine(const CompilerTargetBinding &binding);

llvm::Expected<CompilerTargetCommandLineProjection>
projectCompilerTargetCommandLine(
    const fabric::InstructionCoreArchitecturalContract &architecture,
    const CompilerTargetPolicy &policy);

/// Validates one LLVM module against an exact binding without rewriting
/// either owner. The target triple must already use the binding's canonical
/// spelling; DataLayout compatibility is structural under the pinned LLVM
/// provider.
llvm::Error validateModuleCompilerTarget(const llvm::Module &module,
                                         const CompilerTargetBinding &binding);

/// Emits one relocatable object using only the exact target selection stored
/// by the binding. The module is consumed and is never retargeted or repaired.
llvm::Expected<std::vector<std::uint8_t>>
emitCompilerTargetObject(std::unique_ptr<llvm::Module> module,
                         const CompilerTargetBinding &binding);

} // namespace loom

#endif // LOOM_FRONTEND_EXECUTABLE_COMPILERTARGETBINDING_H
