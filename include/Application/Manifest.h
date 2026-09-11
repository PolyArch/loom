#ifndef LOOM_APPLICATION_MANIFEST_H
#define LOOM_APPLICATION_MANIFEST_H

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
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
enum class OracleEncoding : std::uint8_t { Utf8, HexSampleLines };
enum class OracleCoverage : std::uint8_t { AllMeasuredSamples };
enum class ExecutionSelection : std::uint8_t {
  Smoke,
  Validation,
  ScaleEda,
};

/// The evaluation target one input row declares. It is the single owner of the
/// question "does this row carry the System QoR saturation gate?"; no consumer
/// re-derives it from an input or execution-selection name.
///
/// `Functional` requires a published Mapping, an execution output matching the
/// product oracle, and complete diagnostics. Its measurements are published
/// exactly as measured and no saturation target applies, because an input
/// sized below the System's fixed launch cost cannot reach one by
/// construction. `Qualified` additionally requires the full System QoR target.
enum class EvaluationTier : std::uint8_t { Functional, Qualified };

llvm::StringRef toString(SourceKind kind);
llvm::StringRef toString(LanguageMode mode);
llvm::StringRef toString(OracleKind kind);
llvm::StringRef toString(OracleEncoding encoding);
llvm::StringRef toString(OracleCoverage coverage);
llvm::StringRef toString(ExecutionSelection selection);
llvm::StringRef toString(EvaluationTier tier);

llvm::Expected<ExecutionSelection>
parseExecutionSelection(llvm::StringRef spelling);

llvm::Expected<EvaluationTier> parseEvaluationTier(llvm::StringRef spelling);

struct SourceSelection final {
  SourceKind kind;
  std::string root;
};

/// The product callable selected by the Application manifest. Its full ABI is
/// derived from the selected cached-input order and execution profile; the
/// manifest owns only the application-specific symbol and per-sample output
/// extent.
struct ProductExecutionSelection final {
  std::string entrySymbol;
  std::uint64_t measuredOutputBytesPerSample = 0;
};

struct BuildSelection final {
  std::string entry;
  LanguageMode language;
  std::vector<std::string> sources;
  std::vector<std::string> compilerOptions;
  std::vector<std::string> linkOptions;
  std::vector<std::string> operatorProtocolSymbols;
  std::optional<ProductExecutionSelection> productExecution;
};

struct CachedInput final {
  std::string logicalName;
  std::string path;
  BlobDigest digest;
};

struct OracleSelection final {
  OracleKind kind;
  std::string entry;
  BlobDigest digest;
  OracleEncoding encoding;
};

struct WorkloadExecutionProfile final {
  std::uint64_t warmupSamples;
  std::uint64_t measuredSamples;
  OracleCoverage oracleCoverage;
  std::uint64_t deadlineMilliseconds;
  /// Optional upper bound for the guest simulation tick horizon. This is an
  /// input contract, distinct from the host wall-time deadline.
  std::optional<std::uint64_t> maximumSimulatedTicks;

  std::uint64_t totalSamples() const { return warmupSamples + measuredSamples; }
};

struct WorkloadInputSelection final {
  std::string name;
  EvaluationTier evaluationTier;
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
  static constexpr llvm::StringLiteral schemaVersion = "5.0";

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

/// The derived tier/input inventory projection published by
/// `writeApplicationManifestInventoryJson`. It is a mechanical view of the
/// manifest above, never an independent contract.
inline constexpr llvm::StringLiteral applicationPortfolioInventorySchema =
    "loom.application_portfolio_inventory";
inline constexpr llvm::StringLiteral applicationPortfolioInventoryVersion =
    "3.0";

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
