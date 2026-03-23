#ifndef LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
#define LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H

#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/TDG/ContractLegalityChecker.h"
#include "loom/MultiCoreSim/MultiCoreSimSession.h"

#include "mlir/IR/MLIRContext.h"

#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace syscomp {

// End-to-end pipeline for the Tapestry system compiler.
//
// The pipeline orchestrates three stages:
//   1. Benders decomposition: partition tasks across cores.
//   2. Contract legality: validate that data-movement contracts are legal.
//   3. Multi-core simulation: estimate end-to-end latency.
//
// This class wires the three subsystems together.
class TapestryPipeline {
public:
  explicit TapestryPipeline(const BendersDriverOptions &options);

  // Add a task to the pipeline.
  void addTask(const BendersTask &task);

  // Add an edge (data dependency) to the pipeline.
  void addEdge(const BendersEdge &edge);

  // Run the full pipeline: partition, check legality, simulate.
  // Returns an error string on failure, empty string on success.
  std::string run();

  // Accessors for results after a successful run().
  const BendersResult &getBendersResult() const { return bendersResult_; }
  const mcsim::MultiCoreSimResult &getSimResult() const { return simResult_; }
  bool legalityPassed() const { return legalityPassed_; }

private:
  BendersDriverOptions options_;
  std::vector<BendersTask> tasks_;
  std::vector<BendersEdge> edges_;

  BendersResult bendersResult_;
  mcsim::MultiCoreSimResult simResult_;
  bool legalityPassed_ = false;
};

} // namespace syscomp

// -----------------------------------------------------------------------
// Config-driven full pipeline API (used by tapestry CLI tools)
// -----------------------------------------------------------------------

/// Pipeline stages that can be selectively enabled.
enum class PipelineStage {
  COMPILE,
  SIMULATE,
  RTLGEN,
};

/// Benders decomposition options for the config-driven pipeline.
struct PipelineBendersOptions {
  unsigned maxIterations = 10;
  double costTighteningThreshold = 0.01;
  bool perfectNoC = false;
  bool verbose = false;
};

/// Simulation sub-config for the config-driven pipeline.
struct PipelineSimConfig {
  uint64_t maxGlobalCycles = 1000000;
  bool enableNoCContention = true;
  bool enableTracing = false;
};

/// SVGen sub-options for RTL generation.
struct PipelineSVGenOptions {
  std::string fpIpProfile;
  unsigned meshRows = 0;
  unsigned meshCols = 0;
};

/// Top-level configuration for the config-driven pipeline.
struct TapestryPipelineConfig {
  std::string tdgPath;
  std::string systemArchPath;
  std::string outputDir = "tapestry-output";
  bool verbose = false;

  std::vector<PipelineStage> stages;

  PipelineBendersOptions bendersOpts;
  PipelineSimConfig simConfig;
  PipelineSVGenOptions svgenOpts;
  std::string rtlSourceDir;
};

/// Compilation metrics from a successful compile stage.
struct PipelineCompilationMetrics {
  unsigned numBendersIterations = 0;
  double compilationTimeSec = 0.0;
};

/// Per-core result from the compilation stage.
struct PipelineCoreResult {
  std::string coreName;
  bool success = false;
};

/// Result of the compilation stage.
struct PipelineCompilationResult {
  std::vector<PipelineCoreResult> coreResults;
  PipelineCompilationMetrics metrics;
};

/// NoC statistics from the simulation stage.
struct PipelineNoCStats {
  uint64_t totalFlitsTransferred = 0;
};

/// Per-core result from the simulation stage.
struct PipelineCoreSimResult {
  unsigned coreId = 0;
  uint64_t cycles = 0;
  bool completed = false;
};

/// Result of the simulation stage.
struct PipelineSimResult {
  uint64_t totalGlobalCycles = 0;
  PipelineNoCStats nocStats;
  std::vector<PipelineCoreSimResult> coreResults;
};

/// Result of the RTL generation stage.
struct PipelineRTLResult {
  std::string systemTopFile;
  std::vector<std::string> allGeneratedFiles;
};

/// Top-level result from the full pipeline.
struct TapestryPipelineResult {
  bool success = false;
  std::string diagnostics;
  std::string reportPath;

  std::optional<PipelineCompilationResult> compilationResult;
  std::optional<PipelineSimResult> simResult;
  std::optional<PipelineRTLResult> rtlResult;
};

/// Config-driven full pipeline: orchestrates compile, simulate, and RTL
/// generation stages based on the provided configuration.
class TapestryPipeline {
public:
  TapestryPipeline() = default;

  /// Run the pipeline with the given configuration and MLIR context.
  TapestryPipelineResult run(const TapestryPipelineConfig &config,
                             mlir::MLIRContext &context);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
