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

namespace llvm {
class raw_ostream;
namespace json {
class Object;
}
} // namespace llvm

namespace loom::application {

enum class SourceKind : std::uint8_t { Gitlink, Repository };
enum class LanguageMode : std::uint8_t { C, Cxx };
enum class OracleKind : std::uint8_t { Exact, TypedInvariant };
enum class OracleCoverage : std::uint8_t { AllMeasuredSamples };
enum class ExecutionSelection : std::uint8_t {
  Smoke,
  Validation,
  ScaleEda,
};

llvm::StringRef toString(SourceKind kind);
llvm::StringRef toString(LanguageMode mode);
llvm::StringRef toString(OracleKind kind);
llvm::StringRef toString(OracleCoverage coverage);
llvm::StringRef toString(ExecutionSelection selection);

llvm::Expected<ExecutionSelection>
parseExecutionSelection(llvm::StringRef spelling);

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
  std::vector<std::string> operatorProtocolSymbols;
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

struct WorkloadExecutionProfile final {
  std::uint64_t warmupSamples;
  std::uint64_t measuredSamples;
  OracleCoverage oracleCoverage;
  std::uint64_t deadlineMilliseconds;

  std::uint64_t totalSamples() const { return warmupSamples + measuredSamples; }
};

struct WorkloadInputSelection final {
  std::string name;
  std::string workload;
  std::string runtimeInput;
  std::vector<std::string> cachedInputs;
  std::vector<std::string> compilerOptions;
  OracleSelection oracle;
  WorkloadExecutionProfile profile;
};

struct ExecutionSelectionInputs final {
  ExecutionSelection selection;
  std::vector<std::string> inputNames;
};

struct ApplicationDefinition final {
  std::string identity;
  SourceSelection source;
  BuildSelection build;
  std::vector<CachedInput> cachedInputs;
  std::vector<WorkloadInputSelection> inputs;
  std::vector<ExecutionSelectionInputs> selectionInputs;
};

/// Transient copy of one exact application/input selection. Cached inputs are
/// narrowed to the logical references owned by the selected input.
struct SelectedApplicationInput final {
  std::string applicationIdentity;
  SourceSelection source;
  BuildSelection build;
  std::vector<CachedInput> cachedInputs;
  WorkloadInputSelection input;
};

/// Thin repository conformance input. This is not an Artifact and does not own
/// source, workload, runtime-input, oracle, or external revision semantics.
class ApplicationManifest final {
public:
  static constexpr llvm::StringLiteral schemaIdentity =
      "loom.application_portfolio";
  static constexpr llvm::StringLiteral schemaVersion = "3.0";

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

/// Deterministic JSON projection shared by host and inventory reports.
llvm::json::Object
projectSelectedApplicationInputJson(const SelectedApplicationInput &selection);

/// Emits the exact tier/input inventory after canonical manifest parsing.
void writeApplicationManifestInventoryJson(llvm::raw_ostream &output,
                                           const ApplicationManifest &manifest);

/// Resolves the exact application/input rows selected by one execution tier.
std::vector<SelectedApplicationInput>
selectApplicationInputs(const ApplicationManifest &manifest,
                        ExecutionSelection selection);

/// Resolves one application/input name pair into an independent derived copy.
llvm::Expected<SelectedApplicationInput>
selectApplicationInput(const ApplicationManifest &manifest,
                       llvm::StringRef applicationIdentity,
                       llvm::StringRef inputName);

} // namespace loom::application

#endif // LOOM_APPLICATION_MANIFEST_H
