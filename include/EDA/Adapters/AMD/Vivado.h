#ifndef LOOM_EDA_ADAPTERS_AMD_VIVADO_H
#define LOOM_EDA_ADAPTERS_AMD_VIVADO_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::amd {

inline constexpr dse::CandidateGeneratorKind
    vivadoStaticFullDeviceCandidateGeneratorKind(0x56495641);

class VivadoStaticFullDeviceUnavailableError final
    : public llvm::ErrorInfo<VivadoStaticFullDeviceUnavailableError> {
public:
  static char ID;

  explicit VivadoStaticFullDeviceUnavailableError(std::string detail)
      : detail_(std::move(detail)) {}

  llvm::StringRef detail() const { return detail_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string detail_;
};

class VivadoStaticFullDeviceUnsupportedError final
    : public llvm::ErrorInfo<VivadoStaticFullDeviceUnsupportedError> {
public:
  static char ID;

  explicit VivadoStaticFullDeviceUnsupportedError(std::string detail)
      : detail_(std::move(detail)) {}

  llvm::StringRef detail() const { return detail_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string detail_;
};

class ResolvedVivadoStaticFullDeviceConfigView final {
public:
  llvm::StringRef stableProviderBuildIdentity() const {
    return stableProviderBuildIdentity_;
  }
  llvm::StringRef deviceResourceKey() const { return deviceResourceKey_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedVivadoStaticFullDeviceConfigView(
      std::string stableProviderBuildIdentity, std::string deviceResourceKey,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : stableProviderBuildIdentity_(std::move(stableProviderBuildIdentity)),
        deviceResourceKey_(std::move(deviceResourceKey)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::string stableProviderBuildIdentity_;
  std::string deviceResourceKey_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
      projectResolvedVivadoStaticFullDeviceConfigView(llvm::StringRef,
                                                      llvm::StringRef);
  friend llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
  adoptResolvedVivadoStaticFullDeviceConfigView(llvm::ArrayRef<std::uint8_t>,
                                                llvm::ArrayRef<std::uint8_t>,
                                                const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedVivadoStaticFullDeviceConfigSchemaBytes();

llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
projectResolvedVivadoStaticFullDeviceConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    llvm::StringRef deviceResourceKey);

llvm::Expected<ResolvedVivadoStaticFullDeviceConfigView>
adoptResolvedVivadoStaticFullDeviceConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

std::string vivadoToolBundledResourceProviderIdentity(
    llvm::StringRef stableProviderBuildIdentity);

const dse::CandidateGeneratorDescriptor &
vivadoStaticFullDeviceCandidateGeneratorDescriptor();
llvm::Error registerVivadoStaticFullDeviceCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindVivadoStaticFullDeviceCandidateGeneratorInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveVivadoStaticFullDeviceCandidateGeneratorBinding(
    const ResolvedVivadoStaticFullDeviceConfigView &config);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareVivadoStaticFullDeviceInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<dse::CandidateGeneratorProviderResult>
importVivadoStaticFullDeviceInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<std::string>
renderVivadoSynthesisDriver(llvm::StringRef topModule,
                            llvm::StringRef deviceOrderingCode,
                            llvm::ArrayRef<std::string> rtlSources,
                            llvm::ArrayRef<std::string> generationConstraints);

llvm::Expected<std::string>
renderVivadoImplementationDriver(llvm::StringRef topModule,
                                 llvm::StringRef deviceOrderingCode);

llvm::Expected<std::string>
renderVivadoImageDriver(llvm::StringRef topModule,
                        llvm::StringRef deviceOrderingCode);

} // namespace loom::eda::amd

#endif // LOOM_EDA_ADAPTERS_AMD_VIVADO_H
