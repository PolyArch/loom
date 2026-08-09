#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_COMMON_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_COMMON_H

#include "Evaluation/NumericValue.h"
#include "ExternalTool/InvocationBundle.h"
#include "Hardware/Implementation/ImplementationRepresentationRoot.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::eda::synopsys {

enum class SynopsysOperation : std::uint8_t {
  FunctionalEvaluation,
  LogicSynthesis,
  PhysicalImplementation,
  TimingEvaluation,
  PowerEvaluation,
};

struct SynopsysImplementationState final {
  hardware::RepresentationRootVariant variant;
  std::optional<hardware::RepresentationPhysicalStage> stage;

  friend bool operator==(SynopsysImplementationState lhs,
                         SynopsysImplementationState rhs) {
    return lhs.variant == rhs.variant && lhs.stage == rhs.stage;
  }
};

/// Adapter-local invocation facts. Persistent generator and evaluation
/// descriptor references remain owned by their central registries.
struct SynopsysInvocationDescriptor final {
  llvm::StringLiteral toolKey;
  llvm::StringLiteral implementationSemanticIdentity;
  SynopsysOperation operation;
  llvm::ArrayRef<SynopsysImplementationState> acceptedStates;
  bool requiresAsicPlatform;
  bool requiresTechnologyCorner;
  bool requiresGenerationConstraint;
  llvm::ArrayRef<llvm::StringLiteral> requiredProviderInputs;
  llvm::ArrayRef<llvm::StringLiteral> declaredOutputs;
};

enum class SynopsysAdapterFailureKind : std::uint8_t {
  DescriptorMismatch,
  MissingSemanticInput,
  UnsupportedImplementation,
  MissingTarget,
  MissingCorner,
  MissingProviderInput,
  ExecutableUnavailable,
  ActivationUnavailable,
  IncompatibleVersion,
  LicenseUnavailable,
  CancelledOrTimeout,
  ToolExecutionFailed,
  MissingDeclaredOutput,
  ParserFailure,
  IntegrityFailure,
  PublicationUnavailable,
};

class SynopsysAdapterError final
    : public llvm::ErrorInfo<SynopsysAdapterError> {
public:
  static char ID;

  SynopsysAdapterError(SynopsysAdapterFailureKind kind, std::string adapter,
                       std::string detail)
      : kind_(kind), adapter_(std::move(adapter)), detail_(std::move(detail)) {}

  SynopsysAdapterFailureKind kind() const { return kind_; }
  llvm::StringRef adapter() const { return adapter_; }
  llvm::StringRef detail() const { return detail_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SynopsysAdapterFailureKind kind_;
  std::string adapter_;
  std::string detail_;
};

llvm::Error makeSynopsysAdapterError(SynopsysAdapterFailureKind kind,
                                     llvm::StringRef adapter,
                                     const llvm::Twine &detail);

bool acceptsSynopsysImplementationState(
    const SynopsysInvocationDescriptor &descriptor,
    hardware::RepresentationRootVariant variant,
    std::optional<hardware::RepresentationPhysicalStage> stage);

llvm::Error validateSynopsysRepresentation(
    const SynopsysInvocationDescriptor &descriptor,
    const hardware::ImplementationRepresentationRoot &representation);

llvm::Error validateSynopsysProviderInputs(
    const SynopsysInvocationDescriptor &descriptor,
    llvm::ArrayRef<external_tool::ResolvedExternalFile> files,
    llvm::ArrayRef<external_tool::ResolvedExternalFileTree> fileTrees);

/// A binding already frozen by the shared resolver. No adapter API accepts a
/// LocalToolConfig, probes PATH, activates modules, or chooses a runtime.
struct SynopsysFrozenInvocation final {
  external_tool::ResolvedToolBinding tool;
  external_tool::ToolVersionProbe toolVersionProbe;
  external_tool::InvocationRuntimeBinding runtime;
  external_tool::ToolVersionProbe containerVersionProbe;
  std::vector<std::string> inheritEnvironment;
  std::vector<external_tool::ResolvedExternalFile> externalFiles;
  std::vector<external_tool::ResolvedExternalFileTree> externalFileTrees;
};

struct SynopsysBundleInputs final {
  external_tool::ExternalToolSemanticContract semanticContract;
  const hardware::ImplementationRepresentationRoot *implementation = nullptr;
  std::optional<ArtifactRootReference> implementationPlatform;
  const platform::FinalizedImplementationPlatform *platform = nullptr;
  std::optional<EncodedArtifactLocalReference> technologyCorner;
  SynopsysFrozenInvocation frozen;
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
};

llvm::Error
validateSynopsysSemanticInputs(const SynopsysInvocationDescriptor &descriptor,
                               const SynopsysBundleInputs &inputs,
                               llvm::ArrayRef<std::string> requiredPaths);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeSynopsysInvocationBundleSpec(
    const SynopsysInvocationDescriptor &descriptor,
    const SynopsysBundleInputs &inputs,
    std::vector<std::vector<std::string>> commands,
    std::vector<external_tool::MaterializedBundleFile> drivers);

llvm::Expected<external_tool::ImportedExternalToolInvocationBundle>
importSynopsysInvocation(
    const SynopsysInvocationDescriptor &descriptor,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs);

llvm::Expected<std::string> readSynopsysDeclaredOutput(
    const SynopsysInvocationDescriptor &descriptor,
    const external_tool::ImportedExternalToolInvocationBundle &bundle,
    llvm::StringRef relativePath);

bool isPortableHdlIdentifier(llvm::StringRef value);
llvm::Error validateBundleInputPath(llvm::StringRef adapter,
                                    llvm::StringRef path);
llvm::Expected<std::string> renderTclWord(llvm::StringRef adapter,
                                          llvm::StringRef value);
std::string renderSynopsysTclBatch(llvm::StringRef commands,
                                   llvm::StringRef publication);
llvm::Expected<evaluation::DecimalValue>
parseSynopsysDecimal(llvm::StringRef adapter, llvm::StringRef field,
                     llvm::StringRef value, bool permitZero);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_COMMON_H
