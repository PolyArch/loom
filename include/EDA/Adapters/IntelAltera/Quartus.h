#ifndef LOOM_EDA_ADAPTERS_INTELALTERA_QUARTUS_H
#define LOOM_EDA_ADAPTERS_INTELALTERA_QUARTUS_H

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::intel_altera {

inline constexpr dse::CandidateGeneratorKind
    quartusPrimeStaticFullDeviceCandidateGeneratorKind(0x51525453);

enum class QuartusPrimeUnsupportedReason : std::uint8_t {
  InputRepresentation,
  TargetVendor,
  PlatformBinding,
  DeviceResourceBinding,
  ProviderResourceBinding,
  ExplicitFileDependency,
  MemoryMacroBinding,
  TopModule,
  PayloadRole,
};

class QuartusPrimeUnsupportedError final
    : public llvm::ErrorInfo<QuartusPrimeUnsupportedError> {
public:
  static char ID;

  QuartusPrimeUnsupportedError(QuartusPrimeUnsupportedReason reason,
                               std::string detail)
      : reason_(reason), detail_(std::move(detail)) {}

  QuartusPrimeUnsupportedReason reason() const { return reason_; }
  llvm::StringRef detail() const { return detail_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  QuartusPrimeUnsupportedReason reason_;
  std::string detail_;
};

enum class QuartusPrimeUnavailableReason : std::uint8_t {
  ToolResolution,
  ProviderBuild,
  RuntimeResolution,
};

class QuartusPrimeUnavailableError final
    : public llvm::ErrorInfo<QuartusPrimeUnavailableError> {
public:
  static char ID;

  QuartusPrimeUnavailableError(QuartusPrimeUnavailableReason reason,
                               std::string detail)
      : reason_(reason), detail_(std::move(detail)) {}

  QuartusPrimeUnavailableReason reason() const { return reason_; }
  llvm::StringRef detail() const { return detail_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  QuartusPrimeUnavailableReason reason_;
  std::string detail_;
};

class ResolvedQuartusPrimeStaticFullDeviceConfigView final {
public:
  llvm::StringRef stableProviderBuildIdentity() const {
    return stableProviderBuildIdentity_;
  }
  llvm::StringRef verifiedToolVersion() const { return verifiedToolVersion_; }
  llvm::StringRef deviceResourceKey() const { return deviceResourceKey_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedQuartusPrimeStaticFullDeviceConfigView(
      std::string stableProviderBuildIdentity, std::string verifiedToolVersion,
      std::string deviceResourceKey, std::vector<std::uint8_t> canonicalBytes,
      ComponentViewDigest digest)
      : stableProviderBuildIdentity_(std::move(stableProviderBuildIdentity)),
        verifiedToolVersion_(std::move(verifiedToolVersion)),
        deviceResourceKey_(std::move(deviceResourceKey)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::string stableProviderBuildIdentity_;
  std::string verifiedToolVersion_;
  std::string deviceResourceKey_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
      projectResolvedQuartusPrimeStaticFullDeviceConfigView(llvm::StringRef,
                                                            llvm::StringRef,
                                                            llvm::StringRef);
  friend llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
  adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedQuartusPrimeStaticFullDeviceConfigSchemaBytes();

llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
projectResolvedQuartusPrimeStaticFullDeviceConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    llvm::StringRef verifiedToolVersion, llvm::StringRef deviceResourceKey);

llvm::Expected<ResolvedQuartusPrimeStaticFullDeviceConfigView>
adoptResolvedQuartusPrimeStaticFullDeviceConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const dse::CandidateGeneratorDescriptor &
quartusPrimeStaticFullDeviceCandidateGeneratorDescriptor();
llvm::Error registerQuartusPrimeStaticFullDeviceCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindQuartusPrimeStaticFullDeviceCandidateGeneratorInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveQuartusPrimeStaticFullDeviceCandidateGeneratorBinding(
    const ResolvedQuartusPrimeStaticFullDeviceConfigView &config);

} // namespace loom::eda::intel_altera

#endif // LOOM_EDA_ADAPTERS_INTELALTERA_QUARTUS_H
