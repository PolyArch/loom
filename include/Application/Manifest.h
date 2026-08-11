#ifndef LOOM_APPLICATION_MANIFEST_H
#define LOOM_APPLICATION_MANIFEST_H

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::application {

enum class SourceKind : std::uint8_t { Gitlink, Repository };
enum class LanguageMode : std::uint8_t { C, Cxx };
enum class OracleKind : std::uint8_t { Exact, TypedInvariant };
enum class ExecutionSelection : std::uint8_t {
  Smoke,
  Validation,
  ScaleEda,
};

llvm::StringRef toString(SourceKind kind);
llvm::StringRef toString(LanguageMode mode);
llvm::StringRef toString(OracleKind kind);
llvm::StringRef toString(ExecutionSelection selection);

struct SourceSelection final {
  SourceKind kind;
  std::string root;
};

struct BuildSelection final {
  std::string entry;
  LanguageMode language;
  std::vector<std::string> sources;
  std::vector<std::string> compilerOptions;
  std::vector<std::string> linkOptions;
};

struct CachedInput final {
  std::string logicalName;
  std::string path;
  BlobDigest digest;
};

struct OracleSelection final {
  OracleKind kind;
  std::string entry;
};

struct WorkloadInputSelection final {
  std::string name;
  std::string workload;
  std::string runtimeInput;
  std::vector<std::string> cachedInputs;
  OracleSelection oracle;
};

struct ApplicationDefinition final {
  std::string identity;
  SourceSelection source;
  BuildSelection build;
  std::vector<CachedInput> cachedInputs;
  std::vector<WorkloadInputSelection> inputs;
  std::vector<ExecutionSelection> selections;
};

/// Thin repository conformance input. This is not an Artifact and does not own
/// source, workload, runtime-input, oracle, or external revision semantics.
class ApplicationManifest final {
public:
  static constexpr llvm::StringLiteral schemaIdentity =
      "loom.application_portfolio";
  static constexpr llvm::StringLiteral schemaVersion = "1.0";

  llvm::ArrayRef<ApplicationDefinition> applications() const {
    return applications_;
  }

private:
  explicit ApplicationManifest(std::vector<ApplicationDefinition> applications)
      : applications_(std::move(applications)) {}

  std::vector<ApplicationDefinition> applications_;

  friend llvm::Expected<ApplicationManifest>
      parseApplicationManifest(llvm::StringRef);
};

llvm::Expected<ApplicationManifest>
parseApplicationManifest(llvm::StringRef jsonText);

llvm::Expected<ApplicationManifest>
loadApplicationManifest(llvm::StringRef path);

/// Returns the canonical manifest identities carrying one execution selection.
std::vector<std::string>
selectApplicationIdentities(const ApplicationManifest &manifest,
                            ExecutionSelection selection);

} // namespace loom::application

#endif // LOOM_APPLICATION_MANIFEST_H
