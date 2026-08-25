#ifndef LOOM_APPLICATION_SOURCEADMISSION_H
#define LOOM_APPLICATION_SOURCEADMISSION_H

#include "Application/Manifest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::application {

enum class SourceUnavailableReason : std::uint8_t {
  GitExecutable,
  GitlinkCheckout,
  CacheRoot,
  CachedInput,
};

llvm::StringRef toString(SourceUnavailableReason reason);

struct AdmittedCachedInput final {
  std::string logicalName;
  std::string path;
};

struct AdmittedApplicationInput final {
  std::string inputName;
  /// Canonical absolute oracle path validated by admission.
  std::string oraclePath;
  /// Canonical absolute cache paths in selected manifest order.
  std::vector<AdmittedCachedInput> cachedInputs;
};

struct AdmittedApplicationSource final {
  std::string applicationIdentity;
  /// Canonical absolute repository root validated by admission.
  std::string repositoryRoot;
  std::string sourceRoot;
  /// Canonical absolute source paths in manifest build order.
  std::vector<std::string> sourcePaths;
  std::vector<AdmittedApplicationInput> inputs;
};

struct UnavailableApplicationSource final {
  SourceUnavailableReason reason;
  std::string applicationIdentity;
  std::string path;
};

using ApplicationSourceAdmissionOutcome =
    std::variant<AdmittedApplicationSource, UnavailableApplicationSource>;

/// Resolves one canonical explicit subset against the manifest. Missing
/// Gitlink checkouts and cache content remain per-application typed outcomes;
/// malformed repository state or digest disagreement is invalid.
llvm::Expected<std::vector<ApplicationSourceAdmissionOutcome>>
admitApplicationSources(
    const ApplicationManifest &manifest,
    llvm::ArrayRef<std::string> applicationIdentities,
    llvm::StringRef repositoryRoot,
    std::optional<llvm::StringRef> cacheRoot = std::nullopt);

/// Resolves one application and one named input. Only that input's oracle and
/// cached-input references participate in admission.
llvm::Expected<ApplicationSourceAdmissionOutcome> admitApplicationSource(
    const ApplicationManifest &manifest, llvm::StringRef applicationIdentity,
    llvm::StringRef inputName, llvm::StringRef repositoryRoot,
    std::optional<llvm::StringRef> cacheRoot = std::nullopt);

} // namespace loom::application

#endif // LOOM_APPLICATION_SOURCEADMISSION_H
