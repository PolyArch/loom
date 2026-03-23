#ifndef LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
#define LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H

#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/MultiCoreSim/MultiCoreSimSession.h"
#include "loom/SVGen/MultiCoreSVGen.h"
#include "loom/SVGen/MultiCoreConfigGen.h"

#include "mlir/IR/MLIRContext.h"

#include <optional>
#include <string>
#include <vector>

namespace loom {

//===----------------------------------------------------------------------===//
// Pipeline Stage Enum
//===----------------------------------------------------------------------===//

/// Individual pipeline stages that can be enabled selectively.
enum class PipelineStage {
  COMPILE,
  SIMULATE,
  RTLGEN
};

//===----------------------------------------------------------------------===//
// Pipeline Configuration
//===----------------------------------------------------------------------===//

/// Configuration for the Tapestry multi-core pipeline.
struct TapestryPipelineConfig {
  /// Path to TDG input (MLIR text).
  std::string tdgPath;

  /// Path to system architecture JSON.
  std::string systemArchPath;

  /// Output directory for all artifacts.
  std::string outputDir;

  /// Stages to run (empty = run all stages).
  std::vector<PipelineStage> stages;

  /// Benders driver options.
  BendersDriverOptions bendersOpts;

  /// Base mapper options for L2 compilation.
  MapperOptions baseMapperOpts;

  /// Multi-core simulation configuration.
  mcsim::MultiCoreSimConfig simConfig;

  /// Multi-core SVGen options.
  svgen::MultiCoreSVGenOptions svgenOpts;

  /// Path to RTL source directory (for SVGen).
  std::string rtlSourceDir;

  /// Enable verbose logging.
  bool verbose = false;

  /// Helper: check if a specific stage should run.
  bool shouldRunStage(PipelineStage stage) const;
};

//===----------------------------------------------------------------------===//
// Pipeline Result
//===----------------------------------------------------------------------===//

/// Complete result of the Tapestry pipeline execution.
struct TapestryPipelineResult {
  bool success = false;

  /// Diagnostics and status messages.
  std::string diagnostics;

  /// Compilation result (always populated if COMPILE stage runs).
  std::optional<TapestryCompilationResult> compilationResult;

  /// Simulation result (populated if SIMULATE stage runs).
  std::optional<mcsim::MultiCoreSimResult> simResult;

  /// RTL generation result (populated if RTLGEN stage runs).
  std::optional<svgen::MultiCoreSVGenResult> rtlResult;

  /// Multi-core configuration image (populated if RTLGEN stage runs).
  std::optional<svgen::MultiCoreConfigImage> configImage;

  /// Path to the JSON report file (populated after report generation).
  std::string reportPath;
};

//===----------------------------------------------------------------------===//
// TapestryPipeline
//===----------------------------------------------------------------------===//

/// Orchestrates the full Tapestry multi-core pipeline: compile, simulate,
/// and generate RTL. Each stage can be run independently or as a full
/// end-to-end flow.
class TapestryPipeline {
public:
  /// Run the configured pipeline stages.
  ///
  /// \param config   Pipeline configuration (paths, options, stages).
  /// \param ctx      MLIR context for module parsing and lowering.
  /// \returns        Complete pipeline result.
  TapestryPipelineResult run(const TapestryPipelineConfig &config,
                             mlir::MLIRContext &ctx);

  /// Run only the compilation stage.
  TapestryCompilationResult
  runCompile(const TapestryPipelineConfig &config, mlir::MLIRContext &ctx);

  /// Run only the simulation stage, given an existing compilation result.
  mcsim::MultiCoreSimResult
  runSimulate(const TapestryCompilationResult &compilation,
              const TapestryPipelineConfig &config);

  /// Run only the RTL generation stage, given an existing compilation result.
  svgen::MultiCoreSVGenResult
  runRtlGen(const TapestryCompilationResult &compilation,
            const TapestryPipelineConfig &config, mlir::MLIRContext &ctx);

  /// Generate a JSON report from the pipeline result.
  static std::string generateReport(const TapestryPipelineResult &result,
                                    const std::string &outputDir);

private:
  /// Load system architecture from JSON file.
  static SystemArchitecture loadSystemArch(const std::string &jsonPath);

  /// Build multi-core compilation descriptor from compilation result.
  static svgen::MultiCoreCompilationDesc
  buildCompilationDesc(const TapestryCompilationResult &compilation);

  /// Write per-core config binaries to the output directory.
  static void writeConfigBinaries(const TapestryCompilationResult &compilation,
                                  const std::string &outputDir);

  /// Write system-level JSON outputs (assignment, NoC schedule, etc.).
  static void writeSystemOutputs(const TapestryCompilationResult &compilation,
                                 const std::string &outputDir);

  /// Write simulation results to JSON.
  static void writeSimResults(const mcsim::MultiCoreSimResult &simResult,
                              const std::string &outputDir);

  /// Serialize SystemMetrics to JSON string.
  static std::string metricsToJson(const SystemMetrics &metrics);

  /// Serialize iteration history to JSON string.
  static std::string iterationHistoryToJson(
      const std::vector<IterationRecord> &history);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
