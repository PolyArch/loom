#ifndef LOOM_APPLICATION_HOSTRUNNER_H
#define LOOM_APPLICATION_HOSTRUNNER_H

#include "Application/Manifest.h"
#include "Application/SourceAdmission.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace loom::application {

enum class ApplicationHostRunOutcome : std::uint8_t {
  Succeeded,
  SourceUnavailable,
  CompileFailure,
  ExecutionFailure,
  Timeout,
  OracleMismatch,
  UnsupportedOracle,
  UnsupportedProfile,
};

enum class ApplicationHostOracleStatus : std::uint8_t {
  NotChecked,
  Matched,
  Mismatched,
  Unsupported,
};

llvm::StringRef toString(ApplicationHostRunOutcome outcome);
llvm::StringRef toString(ApplicationHostOracleStatus status);

struct ApplicationHostRunRequest final {
  std::string applicationIdentity;
  std::string inputName;
  std::string repositoryRoot;
  std::optional<std::string> cacheRoot;
  std::optional<std::string> compilerExecutable;
};

/// Operational host conformance result. This report is neither an Artifact nor
/// EvaluationEvidence and establishes no canonical Simulation binding.
struct ApplicationHostRunReport final {
  static constexpr llvm::StringLiteral schemaIdentity =
      "loom.application_host_run";
  static constexpr llvm::StringLiteral schemaVersion = "1.0";

  SelectedApplicationInput selection;
  ApplicationHostRunOutcome outcome;
  ApplicationHostOracleStatus oracleStatus;
  std::optional<UnavailableApplicationSource> unavailableSource;
  std::optional<std::string> compilerExecutable;
  std::optional<int> compileExitStatus;
  std::optional<int> executionExitStatus;
  std::optional<std::uint64_t> hostWallTimeNanoseconds;

  /// Human-facing compiler or execution diagnostic. It is intentionally not
  /// serialized into the deterministic JSON projection.
  std::string diagnostic;
};

struct ApplicationHostSelectionRunRequest final {
  ExecutionSelection selection;
  std::string repositoryRoot;
  std::optional<std::string> cacheRoot;
  std::optional<std::string> compilerExecutable;
};

/// One explicitly requested execution tier and all of its exact input rows.
/// The tier is scheduling provenance; individual host reports remain keyed by
/// their canonical application/input selection.
struct ApplicationHostSelectionRunReport final {
  static constexpr llvm::StringLiteral schemaIdentity =
      "loom.application_host_selection_run";
  static constexpr llvm::StringLiteral schemaVersion = "1.0";

  ExecutionSelection selection;
  std::vector<ApplicationHostRunReport> reports;
};

/// Compiles and executes one exact selected input on a Linux host. When the
/// selection carries cached inputs, the executable receives their admitted
/// absolute paths in selection order, followed by decimal warm-up and measured
/// sample counts. A selection without cached inputs receives none of those
/// derived arguments.
llvm::Expected<ApplicationHostRunReport>
runApplicationInputOnHost(const ApplicationManifest &manifest,
                          const ApplicationHostRunRequest &request);

llvm::Expected<ApplicationHostSelectionRunReport> runApplicationSelectionOnHost(
    const ApplicationManifest &manifest,
    const ApplicationHostSelectionRunRequest &request);

void writeApplicationHostRunReportJson(llvm::raw_ostream &output,
                                       const ApplicationHostRunReport &report);

void writeApplicationHostSelectionRunReportJson(
    llvm::raw_ostream &output, const ApplicationHostSelectionRunReport &report);

bool applicationHostRunSucceeded(const ApplicationHostRunReport &report);
bool applicationHostSelectionRunSucceeded(
    const ApplicationHostSelectionRunReport &report);

} // namespace loom::application

#endif // LOOM_APPLICATION_HOSTRUNNER_H
