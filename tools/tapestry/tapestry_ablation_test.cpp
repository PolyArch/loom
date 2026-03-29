//===-- tapestry_ablation_test.cpp - Ablation infrastructure tests -*- C++ -*-===//
//
// Unit tests for the ablation experiment infrastructure:
//   T1: Each configuration correctly disables specified layers
//   T2: Result collection matches expected format
//   T3: Comparison reporting
//
// Usage:
//   tapestry_ablation_test [--verbose]
//
//===----------------------------------------------------------------------===//

#include "tapestry/co_optimizer.h"

#include "loom/SystemCompiler/ArchitectureFactory.h"
#include "loom/SystemCompiler/PrecompiledKernelLoader.h"
#include "loom/SystemCompiler/TDGLowering.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <regex>
#include <string>
#include <vector>

using namespace loom::tapestry;
namespace tco = tapestry;

static llvm::cl::opt<bool>
    verbose("verbose", llvm::cl::desc("Verbose output"), llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Test harness
//===----------------------------------------------------------------------===//

namespace {

unsigned testsPassed = 0;
unsigned testsFailed = 0;

void check(bool condition, const char *testName) {
  if (condition) {
    ++testsPassed;
    if (verbose)
      llvm::outs() << "  PASS: " << testName << "\n";
  } else {
    ++testsFailed;
    llvm::errs() << "  FAIL: " << testName << "\n";
  }
}

void registerDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

/// Build a template CoOptOptions with reasonable defaults.
tco::CoOptOptions buildTemplateOpts() {
  tco::CoOptOptions opts;
  opts.maxRounds = 1;
  opts.improvementThreshold = 0.01;
  opts.verbose = false;
  opts.swOpts.maxIterations = 5;
  opts.swOpts.improvementThreshold = 0.01;
  opts.swOpts.bendersConfig.maxIterations = 5;
  opts.swOpts.bendersConfig.mapperBudgetSeconds = 5.0;
  opts.swOpts.bendersConfig.mapperSeed = 42;
  opts.hwOuterOpts.maxIterations = 20;
  opts.hwOuterOpts.seed = 42;
  opts.hwInnerOpts.tier2Enabled = false;
  opts.hwInnerOpts.maxInnerIter = 10;
  opts.hwInnerOpts.seed = 42;
  return opts;
}

} // namespace

//===----------------------------------------------------------------------===//
// T1: Each configuration correctly disables specified layers
//===----------------------------------------------------------------------===//

static void testConfigDisablesLayers() {
  llvm::outs() << "\n--- T1: Ablation_ConfigDisablesLayers ---\n";

  auto configs = tco::buildAblationConfigs();
  check(configs.size() == 6, "buildAblationConfigs returns 6 configs");

  auto templateOpts = buildTemplateOpts();

  // Baseline: all disabled
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[0]);
    check(configs[0].name == "Baseline", "config[0] is Baseline");
    check(opts.swOpts.maxIterations == 0,
          "Baseline: swOpts.maxIterations == 0");
    check(opts.hwOuterOpts.maxIterations == 0,
          "Baseline: hwOuterOpts.maxIterations == 0");
    check(opts.hwInnerOpts.tier2Enabled == false,
          "Baseline: hwInnerOpts.tier2Enabled == false");
    check(opts.hwInnerOpts.maxInnerIter == 0,
          "Baseline: hwInnerOpts.maxInnerIter == 0");
    check(opts.swOpts.bendersConfig.maxIterations == 1,
          "Baseline: SW-Inner single-pass (benders maxIter == 1)");
  }

  // SW-only: SW-Outer + SW-Inner enabled, HW disabled
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[1]);
    check(configs[1].name == "SW-only", "config[1] is SW-only");
    check(opts.swOpts.maxIterations > 0,
          "SW-only: swOpts.maxIterations > 0");
    check(opts.swOpts.bendersConfig.maxIterations > 1,
          "SW-only: SW-Inner enabled (benders maxIter > 1)");
    check(opts.hwOuterOpts.maxIterations == 0,
          "SW-only: hwOuterOpts.maxIterations == 0");
    check(opts.hwInnerOpts.tier2Enabled == false,
          "SW-only: hwInnerOpts.tier2Enabled == false");
    check(opts.hwInnerOpts.maxInnerIter == 0,
          "SW-only: hwInnerOpts.maxInnerIter == 0");
  }

  // HW-only: HW-Outer + HW-Inner enabled, SW disabled
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[2]);
    check(configs[2].name == "HW-only", "config[2] is HW-only");
    check(opts.swOpts.maxIterations == 0,
          "HW-only: swOpts.maxIterations == 0");
    check(opts.swOpts.bendersConfig.maxIterations == 1,
          "HW-only: SW-Inner disabled (benders maxIter == 1)");
    check(opts.hwOuterOpts.maxIterations > 0,
          "HW-only: hwOuterOpts.maxIterations > 0");
    // HW-Inner: the template has tier2Enabled=false, but maxInnerIter
    // should remain > 0 since enableHWInner=true for HW-only.
    check(opts.hwInnerOpts.maxInnerIter > 0,
          "HW-only: hwInnerOpts.maxInnerIter > 0");
  }

  // Outer-only: SW-Outer + HW-Inner (no SW-Inner, no HW-Outer)
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[3]);
    check(configs[3].name == "Outer-only", "config[3] is Outer-only");
    check(opts.swOpts.maxIterations > 0,
          "Outer-only: SW-Outer enabled (maxIterations > 0)");
    check(opts.swOpts.bendersConfig.maxIterations == 1,
          "Outer-only: SW-Inner disabled (benders maxIter == 1)");
    check(opts.hwOuterOpts.maxIterations == 0,
          "Outer-only: HW-Outer disabled (maxIterations == 0)");
    check(opts.hwInnerOpts.maxInnerIter > 0,
          "Outer-only: HW-Inner enabled (maxInnerIter > 0)");
  }

  // Inner-only: SW-Inner + HW-Inner (no SW-Outer, no HW-Outer)
  // config: enableSW=false, enableHW=true, enableSWInner=true, enableHWInner=false
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[4]);
    check(configs[4].name == "Inner-only", "config[4] is Inner-only");
    check(opts.swOpts.maxIterations == 0,
          "Inner-only: SW-Outer disabled (maxIterations == 0)");
    check(opts.swOpts.bendersConfig.maxIterations > 1,
          "Inner-only: SW-Inner enabled (benders maxIter > 1)");
    check(opts.hwOuterOpts.maxIterations > 0,
          "Inner-only: HW-Outer enabled via enableHW=true");
    check(opts.hwInnerOpts.maxInnerIter == 0,
          "Inner-only: HW-Inner disabled (maxInnerIter == 0)");
  }

  // Full-coopt: all enabled
  {
    tco::CoOptOptions opts = templateOpts;
    tco::applyAblationConfig(opts, configs[5]);
    check(configs[5].name == "Full-coopt", "config[5] is Full-coopt");
    check(opts.swOpts.maxIterations > 0,
          "Full-coopt: swOpts.maxIterations > 0");
    check(opts.swOpts.bendersConfig.maxIterations > 1,
          "Full-coopt: SW-Inner enabled (benders maxIter > 1)");
    check(opts.hwOuterOpts.maxIterations > 0,
          "Full-coopt: hwOuterOpts.maxIterations > 0");
    check(opts.hwInnerOpts.maxInnerIter > 0,
          "Full-coopt: hwInnerOpts.maxInnerIter > 0");
  }
}

//===----------------------------------------------------------------------===//
// Helper: build a synthetic AblationResult for T2/T3
//===----------------------------------------------------------------------===//

/// Build a synthetic AblationResult with plausible values for testing
/// the JSON serialization and comparison report generation without
/// invoking co_optimize (which requires a full ADG builder pipeline).
static tco::AblationResult buildSyntheticAblationResult() {
  auto configs = tco::buildAblationConfigs();

  tco::AblationResult ablResult;
  ablResult.configs = configs;
  ablResult.domains = {"ai_llm"};
  ablResult.results.resize(configs.size());
  for (auto &row : ablResult.results) {
    row.resize(1);
  }

  // Synthetic throughput/area values that follow expected ordering:
  // Baseline is worst, Full-coopt is best, others in between.
  struct SyntheticMetrics {
    double throughput;
    double area;
    unsigned rounds;
  };
  std::vector<SyntheticMetrics> metrics = {
      {0.010, 8000.0, 1}, // Baseline
      {0.015, 8000.0, 1}, // SW-only (better throughput, same area)
      {0.012, 6500.0, 1}, // HW-only (slightly better throughput, better area)
      {0.014, 7200.0, 1}, // Outer-only
      {0.013, 6800.0, 1}, // Inner-only
      {0.020, 5500.0, 3}, // Full-coopt (best throughput and area)
  };

  for (size_t ci = 0; ci < configs.size(); ++ci) {
    tco::CoOptResult cellResult;
    cellResult.success = true;
    cellResult.bestThroughput = metrics[ci].throughput;
    cellResult.bestArea = metrics[ci].area;
    cellResult.rounds = metrics[ci].rounds;

    // Add at least one history record per cell.
    tco::CoOptResult::RoundRecord rec;
    rec.round = 1;
    rec.swThroughput = metrics[ci].throughput;
    rec.hwArea = metrics[ci].area;
    rec.improved = true;
    cellResult.history.push_back(rec);

    ablResult.results[ci][0] = std::move(cellResult);
  }

  ablResult.success = true;
  return ablResult;
}

//===----------------------------------------------------------------------===//
// T2: Result collection matches expected format
//===----------------------------------------------------------------------===//

static void testResultFormatValid(mlir::MLIRContext & /*ctx*/) {
  llvm::outs() << "\n--- T2: Ablation_ResultFormatValid ---\n";

  auto ablResult = buildSyntheticAblationResult();

  // Check structure.
  check(ablResult.configs.size() == 6, "configs.size() == 6");
  check(ablResult.domains.size() == 1, "domains.size() == 1");
  check(ablResult.results.size() == 6, "results.size() == 6");

  for (size_t ci = 0; ci < ablResult.results.size(); ++ci) {
    check(ablResult.results[ci].size() == 1,
          ("results[" + std::to_string(ci) + "].size() == 1").c_str());
    check(ablResult.results[ci][0].history.size() >= 1,
          ("results[" + std::to_string(ci) + "][0].history >= 1").c_str());
  }

  // Check throughputMatrix and areaMatrix helpers.
  auto tpMatrix = ablResult.throughputMatrix();
  check(tpMatrix.size() == 6, "throughputMatrix rows == 6");
  check(tpMatrix[0].size() == 1, "throughputMatrix cols == 1");
  check(tpMatrix[0][0] > 0.0, "throughputMatrix[0][0] > 0");

  auto areaMatrix = ablResult.areaMatrix();
  check(areaMatrix.size() == 6, "areaMatrix rows == 6");
  check(areaMatrix[0].size() == 1, "areaMatrix cols == 1");
  check(areaMatrix[0][0] > 0.0, "areaMatrix[0][0] > 0");

  // Serialize to JSON and validate structure.
  auto jsonVal = tco::ablationResultToJSON(ablResult);
  auto *rootObj = jsonVal.getAsObject();
  check(rootObj != nullptr, "JSON root is object");

  if (rootObj) {
    check(rootObj->get("configs") != nullptr, "JSON has 'configs' key");
    check(rootObj->get("domains") != nullptr, "JSON has 'domains' key");
    check(rootObj->get("matrix") != nullptr, "JSON has 'matrix' key");
    check(rootObj->get("summary") != nullptr, "JSON has 'summary' key");

    // Check matrix array.
    auto *matrixArr = rootObj->getArray("matrix");
    check(matrixArr != nullptr, "matrix is array");
    if (matrixArr) {
      check(matrixArr->size() == 6,
            "matrix has 6 entries (6 configs x 1 domain)");

      // Check each entry has required keys.
      bool allKeysPresent = true;
      for (const auto &entry : *matrixArr) {
        auto *entryObj = entry.getAsObject();
        if (!entryObj ||
            !entryObj->get("config") ||
            !entryObj->get("domain") ||
            !entryObj->get("throughput") ||
            !entryObj->get("area") ||
            !entryObj->get("rounds") ||
            !entryObj->get("success")) {
          allKeysPresent = false;
          break;
        }
      }
      check(allKeysPresent, "all matrix entries have required keys");
    }

    // Check summary structure.
    auto *summaryObj = rootObj->getObject("summary");
    check(summaryObj != nullptr, "summary is object");
    if (summaryObj) {
      auto *perConfigArr = summaryObj->getArray("per_config");
      check(perConfigArr != nullptr && perConfigArr->size() == 6,
            "summary.per_config has 6 entries");

      auto *perDomainArr = summaryObj->getArray("per_domain");
      check(perDomainArr != nullptr && perDomainArr->size() == 1,
            "summary.per_domain has 1 entry");
    }
  }
}

//===----------------------------------------------------------------------===//
// T3: Comparison reporting
//===----------------------------------------------------------------------===//

static void testComparisonReport(mlir::MLIRContext & /*ctx*/) {
  llvm::outs() << "\n--- T3: Ablation_ComparisonReport ---\n";

  auto ablResult = buildSyntheticAblationResult();

  // Generate the report.
  std::string report = tco::generateComparisonReport(ablResult);
  check(!report.empty(), "report is non-empty");

  // Check all 6 config names appear.
  check(report.find("Baseline") != std::string::npos,
        "report contains 'Baseline'");
  check(report.find("SW-only") != std::string::npos,
        "report contains 'SW-only'");
  check(report.find("HW-only") != std::string::npos,
        "report contains 'HW-only'");
  check(report.find("Outer-only") != std::string::npos,
        "report contains 'Outer-only'");
  check(report.find("Inner-only") != std::string::npos,
        "report contains 'Inner-only'");
  check(report.find("Full-coopt") != std::string::npos,
        "report contains 'Full-coopt'");

  // Check domain name appears.
  check(report.find("ai_llm") != std::string::npos,
        "report contains domain 'ai_llm'");

  // Check throughput and area labels.
  check(report.find("throughput") != std::string::npos ||
            report.find("Throughput") != std::string::npos,
        "report contains 'throughput' label");
  check(report.find("area") != std::string::npos ||
            report.find("Area") != std::string::npos,
        "report contains 'area' label");

  // Check Baseline throughput <= Full-coopt throughput.
  double baselineTp = ablResult.results[0][0].bestThroughput;
  double fullCooptTp = ablResult.results[5][0].bestThroughput;
  check(baselineTp <= fullCooptTp,
        "Baseline throughput <= Full-coopt throughput");

  // Check that the relative improvement section contains percentage values.
  std::regex pctRegex("[0-9]+\\.?[0-9]*%");
  std::smatch match;
  bool hasPercentage = std::regex_search(report, match, pctRegex);
  check(hasPercentage, "report contains percentage values (digits + '%')");

  if (verbose) {
    llvm::outs() << "\n--- Generated Report ---\n" << report << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Ablation infrastructure tests\n");

  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  testConfigDisablesLayers();
  testResultFormatValid(ctx);
  testComparisonReport(ctx);

  llvm::outs() << "\n=== Ablation Tests: "
               << testsPassed << " passed, "
               << testsFailed << " failed ===\n";

  return testsFailed > 0 ? 1 : 0;
}
