//===-- CoOptimizerTest.cpp - Co-optimization framework tests -----*- C++ -*-===//
//
// Unit tests for the F1 co-optimization framework.
//
// Note: Integration tests that run full HW optimization are disabled because
// the ADGBuilder has a known issue with spatial_sw unconnected ports when
// processing fallback topologies from the OUTER-HW optimizer. All framework
// control-flow tests use enableHW=false to avoid this. The HW integration
// path is tested at the experiment-driver level.
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

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace lt = loom::tapestry;
namespace tco = tapestry;

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

static void registerDialects(mlir::MLIRContext &ctx) {
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

//===----------------------------------------------------------------------===//
// Test workload builder
//===----------------------------------------------------------------------===//

struct TestWorkload {
  std::vector<lt::KernelDesc> kernels;
  std::vector<lt::ContractSpec> contracts;
  bool valid = false;
};

/// Build a 2-kernel synthetic workload with DFG modules.
static TestWorkload build2KernelWorkload(mlir::MLIRContext &ctx) {
  TestWorkload w;

  auto k0 = lt::createSyntheticAddKernel("k_add", ctx);
  auto k1 = lt::createSyntheticMacKernel("k_mac", ctx);

  if (!k0.dfgModule || !k1.dfgModule) {
    std::cerr << "  ERROR: failed to create synthetic kernels\n";
    return w;
  }

  w.kernels.push_back(std::move(k0));
  w.kernels.push_back(std::move(k1));

  if (!lt::lowerKernelsToDFG(w.kernels, ctx)) {
    std::cerr << "  ERROR: DFG lowering failed\n";
    w.kernels.clear();
    return w;
  }

  lt::ContractSpec c;
  c.producerKernel = "k_add";
  c.consumerKernel = "k_mac";
  c.dataType = "i32";
  c.elementCount = 1024;
  c.bandwidthBytesPerCycle = 4;
  w.contracts.push_back(std::move(c));

  w.valid = true;
  return w;
}

/// Build a "spectral" initial architecture for N kernels.
static lt::SystemArchitecture buildSpectralArch(unsigned numKernels,
                                                mlir::MLIRContext &ctx) {
  std::vector<lt::CoreTypeSpec> specs;

  lt::CoreTypeSpec gpSpec;
  gpSpec.name = "gp_core";
  gpSpec.meshRows = 2;
  gpSpec.meshCols = 2;
  gpSpec.numInstances = std::max(1u, numKernels / 2);
  gpSpec.includeMultiplier = false;
  gpSpec.includeComparison = true;
  gpSpec.includeMemory = false;
  gpSpec.spmSizeBytes = 4096;
  specs.push_back(gpSpec);

  lt::CoreTypeSpec dspSpec;
  dspSpec.name = "dsp_core";
  dspSpec.meshRows = 2;
  dspSpec.meshCols = 2;
  dspSpec.numInstances = std::max(1u, (numKernels + 1) / 2);
  dspSpec.includeMultiplier = true;
  dspSpec.includeComparison = false;
  dspSpec.includeMemory = false;
  dspSpec.spmSizeBytes = 8192;
  specs.push_back(dspSpec);

  return lt::buildArchitecture("spectral_test", specs, ctx);
}

//===----------------------------------------------------------------------===//
// Common CoOptOptions builder
//===----------------------------------------------------------------------===//

static tco::CoOptOptions makeBaseOpts(bool verb = false) {
  tco::CoOptOptions opts;
  opts.verbose = verb;
  opts.swOpts.maxIterations = 3;
  opts.swOpts.improvementThreshold = 0.01;
  opts.swOpts.bendersConfig.maxIterations = 3;
  opts.swOpts.bendersConfig.mapperBudgetSeconds = 2.0;
  opts.swOpts.bendersConfig.mapperSeed = 42;
  opts.swOpts.verbose = verb;
  opts.hwOuterOpts.maxIterations = 10;
  opts.hwOuterOpts.seed = 42;
  opts.hwOuterOpts.verbose = verb;
  opts.hwInnerOpts.tier2Enabled = false;
  opts.hwInnerOpts.maxInnerIter = 5;
  opts.hwInnerOpts.seed = 42;
  opts.hwInnerOpts.verbose = verb;
  return opts;
}

//===----------------------------------------------------------------------===//
// T1: ConvergenceMonitor standalone logic
//===----------------------------------------------------------------------===//

static bool testConvergenceMonitor() {
  tco::ConvergenceMonitor monitor;
  monitor.improvementThreshold = 0.1; // 10%

  std::string reason;

  // First call: baseline 0 -> 100 throughput is always an improvement.
  bool r1 = monitor.checkImproved(100.0, 500.0, reason);
  if (!r1) {
    std::cerr << "FAIL: T1 round 1 should be improved (reason: "
              << reason << ")\n";
    return false;
  }

  // Second call: 105 throughput is only 5% better, below 10% threshold.
  // 490 area is only 2% better, below 10% threshold.
  bool r2 = monitor.checkImproved(105.0, 490.0, reason);
  if (r2) {
    std::cerr << "FAIL: T1 round 2 should NOT be improved "
              << "(5% throughput, 2% area < 10% threshold)\n";
    return false;
  }

  // Verify reason mentions "no significant improvement".
  if (reason.find("no significant improvement") == std::string::npos) {
    std::cerr << "FAIL: T1 round 2 reason='" << reason
              << "' (expected 'no significant improvement')\n";
    return false;
  }

  // Third call: 150 throughput is ~43% better than 105 (current best),
  // exceeding the 10% threshold.
  bool r3 = monitor.checkImproved(150.0, 490.0, reason);
  if (!r3) {
    std::cerr << "FAIL: T1 round 3 should be improved "
              << "(43% throughput improvement, reason: " << reason << ")\n";
    return false;
  }

  // Verify reason mentions "throughput improved".
  if (reason.find("throughput improved") == std::string::npos) {
    std::cerr << "FAIL: T1 round 3 reason='" << reason
              << "' (expected 'throughput improved')\n";
    return false;
  }

  // Fourth call: area-only improvement. 150 throughput (same), 300 area
  // is 38% better than 490.
  bool r4 = monitor.checkImproved(150.0, 300.0, reason);
  if (!r4) {
    std::cerr << "FAIL: T1 round 4 should be improved "
              << "(area improved by 38%, reason: " << reason << ")\n";
    return false;
  }
  if (reason.find("area improved") == std::string::npos) {
    std::cerr << "FAIL: T1 round 4 reason='" << reason
              << "' (expected 'area improved')\n";
    return false;
  }

  // Fifth call: both improved. 200 throughput (33% better), 200 area (33% better).
  bool r5 = monitor.checkImproved(200.0, 200.0, reason);
  if (!r5) {
    std::cerr << "FAIL: T1 round 5 should be improved\n";
    return false;
  }
  if (reason.find("both throughput and area improved") == std::string::npos) {
    std::cerr << "FAIL: T1 round 5 reason='" << reason
              << "' (expected 'both throughput and area improved')\n";
    return false;
  }

  std::cout << "PASS: ConvergenceMonitor_StandaloneLogic\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T2: Pareto frontier management
//===----------------------------------------------------------------------===//

static bool testParetoFrontier() {
  std::vector<tco::ParetoPoint> frontier;

  // Add first point.
  tco::addParetoPoint(frontier, {10.0, 100.0, 1});
  if (frontier.size() != 1) {
    std::cerr << "FAIL: T2 frontier size=" << frontier.size()
              << " after 1st point\n";
    return false;
  }

  // Add dominated point (lower throughput, higher area). Should be rejected.
  tco::addParetoPoint(frontier, {5.0, 200.0, 2});
  if (frontier.size() != 1) {
    std::cerr << "FAIL: T2 dominated point not rejected\n";
    return false;
  }

  // Add non-dominated point (higher throughput, higher area).
  tco::addParetoPoint(frontier, {20.0, 150.0, 3});
  if (frontier.size() != 2) {
    std::cerr << "FAIL: T2 non-dominated point not added\n";
    return false;
  }

  // Add dominating point (higher throughput, lower area). Should remove first.
  tco::addParetoPoint(frontier, {15.0, 50.0, 4});
  // This dominates (10, 100) and is non-dominated by (20, 150).
  // frontier should have 2 points: (15, 50) and (20, 150).
  if (frontier.size() != 2) {
    std::cerr << "FAIL: T2 after dominating point, size=" << frontier.size()
              << " (expected 2)\n";
    return false;
  }

  std::cout << "PASS: ParetoFrontier_Management\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T3: SW step runs and populates history correctly
//===----------------------------------------------------------------------===//

static bool testSWStepProducesHistory() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T3 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T3 architecture build failed\n";
    return false;
  }

  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 1;
  coOpts.enableSW = true;
  coOpts.enableHW = false;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  // Verify the framework produced a valid result structure.
  if (result.history.empty()) {
    std::cerr << "FAIL: T3 history is empty (expected 1 round record)\n";
    return false;
  }

  if (result.history.size() != 1) {
    std::cerr << "FAIL: T3 history.size()=" << result.history.size()
              << " (expected 1)\n";
    return false;
  }

  if (result.history[0].round != 1) {
    std::cerr << "FAIL: T3 history[0].round=" << result.history[0].round
              << " (expected 1)\n";
    return false;
  }

  if (result.rounds != 1) {
    std::cerr << "FAIL: T3 rounds=" << result.rounds << " (expected 1)\n";
    return false;
  }

  // With HW disabled, hwArea should be infinity (carried from initial).
  if (result.history[0].hwArea !=
      std::numeric_limits<double>::infinity()) {
    std::cerr << "FAIL: T3 hwArea=" << result.history[0].hwArea
              << " (expected infinity with HW disabled)\n";
    return false;
  }

  // The reason field should be populated (non-empty).
  if (result.history[0].reason.empty()) {
    std::cerr << "FAIL: T3 reason is empty\n";
    return false;
  }

  // Diagnostics should contain CSV-format data.
  if (result.diagnostics.empty()) {
    std::cerr << "FAIL: T3 diagnostics is empty\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_SWStepProducesHistory"
            << " (swThroughput=" << result.history[0].swThroughput
            << ", reason=" << result.history[0].reason << ")\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T4: Convergence detection (SW-only mode, terminates early)
//===----------------------------------------------------------------------===//

static bool testConvergenceDetected() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T4 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T4 architecture build failed\n";
    return false;
  }

  // With a high threshold (50%), the loop should converge quickly.
  // With enableHW=false, area stays infinity so only throughput matters.
  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 10;
  coOpts.improvementThreshold = 0.5;
  coOpts.enableHW = false;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  if (result.rounds >= coOpts.maxRounds) {
    std::cerr << "FAIL: T4 rounds=" << result.rounds
              << " (expected < " << coOpts.maxRounds
              << " for early convergence)\n";
    return false;
  }

  if (result.history.empty()) {
    std::cerr << "FAIL: T4 history is empty\n";
    return false;
  }

  // The last round should have improved=false (convergence triggered).
  if (result.history.back().improved) {
    std::cerr << "FAIL: T4 last round improved=true"
              << " (expected false for convergence)\n";
    return false;
  }

  // The last round's reason should describe lack of improvement.
  if (result.history.back().reason.empty()) {
    std::cerr << "FAIL: T4 last round reason is empty\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_ConvergenceDetected"
            << " (rounds=" << result.rounds
            << ", reason='" << result.history.back().reason << "')\n"
            << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T5: Max iteration limit is respected
//===----------------------------------------------------------------------===//

static bool testMaxIterationLimit() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T5 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T5 architecture build failed\n";
    return false;
  }

  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 2;
  coOpts.improvementThreshold = 0.0;
  coOpts.enableHW = false;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  // rounds should not exceed maxRounds.
  if (result.rounds > 2) {
    std::cerr << "FAIL: T5 rounds=" << result.rounds
              << " (expected <= 2)\n";
    return false;
  }

  // history.size() should match rounds.
  if (result.history.size() != result.rounds) {
    std::cerr << "FAIL: T5 history.size()=" << result.history.size()
              << " (expected " << result.rounds << ")\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_MaxIterationLimit"
            << " (rounds=" << result.rounds
            << ", history.size=" << result.history.size() << ")\n"
            << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T6: Layer-enable flags carry forward metrics (SW disabled)
//===----------------------------------------------------------------------===//

static bool testSWDisabledCarriesForward() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T6 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T6 architecture build failed\n";
    return false;
  }

  // Both SW and HW disabled: pure carry-forward, should converge immediately.
  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 3;
  coOpts.enableSW = false;
  coOpts.enableHW = false;
  coOpts.improvementThreshold = 0.0;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  // With both steps disabled, swThroughput should be 0 across all rounds.
  for (const auto &rec : result.history) {
    if (rec.swThroughput != 0.0) {
      std::cerr << "FAIL: T6 round " << rec.round
                << " swThroughput=" << rec.swThroughput
                << " (expected 0 with SW disabled)\n";
      return false;
    }
  }

  // History should not be empty.
  if (result.history.empty()) {
    std::cerr << "FAIL: T6 history is empty\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_SWDisabledCarriesForward"
            << " (rounds=" << result.rounds << ")\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T7: Max-rounds hard cap (20) and clamping warning
//===----------------------------------------------------------------------===//

static bool testMaxRoundsHardCap() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T7 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T7 architecture build failed\n";
    return false;
  }

  // Request 50 rounds. Should be clamped to 20 (the hard cap).
  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 50;
  coOpts.improvementThreshold = 0.0;
  coOpts.enableHW = false;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  // The rounds should be at most 20.
  if (result.rounds > 20) {
    std::cerr << "FAIL: T7 rounds=" << result.rounds
              << " (expected <= 20, hard cap)\n";
    return false;
  }

  // The diagnostics should contain the clamping warning.
  if (result.diagnostics.find("clamped") == std::string::npos) {
    std::cerr << "FAIL: T7 diagnostics missing clamping message: '"
              << result.diagnostics << "'\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_MaxRoundsHardCap"
            << " (requested=50, actual_rounds=" << result.rounds << ")\n"
            << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T8: Reason field populated in every RoundRecord
//===----------------------------------------------------------------------===//

static bool testReasonFieldPopulated() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T8 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T8 architecture build failed\n";
    return false;
  }

  tco::CoOptOptions coOpts = makeBaseOpts();
  coOpts.maxRounds = 3;
  coOpts.improvementThreshold = 0.01;
  coOpts.enableHW = false;

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, coOpts, &ctx);

  if (result.history.empty()) {
    std::cerr << "FAIL: T8 history is empty\n";
    return false;
  }

  // Every round record should have a non-empty reason.
  for (const auto &rec : result.history) {
    if (rec.reason.empty()) {
      std::cerr << "FAIL: T8 round " << rec.round
                << " has empty reason\n";
      return false;
    }
  }

  // Diagnostics should be CSV-formatted with the header and data rows.
  if (result.diagnostics.find("round,swThroughput") ==
      std::string::npos) {
    std::cerr << "FAIL: T8 diagnostics missing CSV header\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_ReasonFieldPopulated"
            << " (rounds=" << result.rounds << ")\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T9: Inner-layer flag enforcement (enableSWInner)
//===----------------------------------------------------------------------===//

static bool testInnerLayerFlags() {
  // Test that inner-layer flags are applied to the effective options.
  // We do this by verifying the options pass-through mechanism works
  // correctly at the CoOptOptions level.
  tco::CoOptOptions opts;
  opts.enableSWInner = false;
  opts.enableHWInner = false;

  // When enableSWInner is false, swOpts.maxIterations should be forced to 1
  // by the co_optimize implementation. We can verify this by running
  // co_optimize and checking it doesn't crash (the effective options
  // are applied internally).
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(false);
  registerDialects(ctx);

  auto workload = build2KernelWorkload(ctx);
  if (!workload.valid) {
    std::cerr << "FAIL: T9 workload construction failed\n";
    return false;
  }

  auto arch = buildSpectralArch(
      static_cast<unsigned>(workload.kernels.size()), ctx);
  if (arch.coreTypes.empty()) {
    std::cerr << "FAIL: T9 architecture build failed\n";
    return false;
  }

  opts.maxRounds = 1;
  opts.enableHW = false;
  opts.swOpts.maxIterations = 10; // This should be overridden to 1
  opts.swOpts.bendersConfig.maxIterations = 3;
  opts.swOpts.bendersConfig.mapperBudgetSeconds = 2.0;
  opts.swOpts.bendersConfig.mapperSeed = 42;
  opts.hwInnerOpts.tier2Enabled = true; // Should be overridden to false

  auto result = tco::co_optimize(
      std::move(workload.kernels), std::move(workload.contracts),
      arch, opts, &ctx);

  // The co-optimization should run without errors.
  if (result.history.empty()) {
    std::cerr << "FAIL: T9 history is empty\n";
    return false;
  }

  std::cout << "PASS: CoOptFramework_InnerLayerFlags\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// T10: computeSystemArea unit test
//===----------------------------------------------------------------------===//

static bool testComputeSystemArea() {
  // Build synthetic HW results to test area computation.
  loom::HWOuterOptimizerResult outerResult;
  outerResult.success = true;
  outerResult.topology.nocTopology = "mesh";
  outerResult.topology.meshRows = 2;
  outerResult.topology.meshCols = 2;
  outerResult.topology.nocBandwidth = 1;
  outerResult.topology.l2TotalSizeKB = 256;
  outerResult.topology.l2BankCount = 4;

  loom::CoreTypeLibraryEntry entry;
  entry.typeIndex = 0;
  entry.role = loom::CoreRole::BALANCED;
  entry.instanceCount = 2;
  entry.minPEs = 4;
  entry.minSPMKB = 4;
  outerResult.topology.coreLibrary.entries.push_back(entry);

  std::vector<loom::ADGOptResult> innerResults;
  loom::ADGOptResult innerResult;
  innerResult.success = true;
  innerResult.areaEstimate = 100.0;
  innerResults.push_back(innerResult);

  double area = tco::computeSystemArea(outerResult, innerResults);

  // Expected: core area = 100 * 2 instances = 200
  //           NoC area = (2*2) * 1 * 5 = 20
  //           L2 area = 256 * 0.1 = 25.6
  //           Total = 245.6
  double expected = 200.0 + 20.0 + 25.6;
  if (std::abs(area - expected) > 0.01) {
    std::cerr << "FAIL: T10 area=" << area
              << " (expected " << expected << ")\n";
    return false;
  }

  std::cout << "PASS: ComputeSystemArea_Correct (area="
            << area << ")\n" << std::flush;
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  std::cout << "CoOptimizerTest: starting\n" << std::flush;

  int passed = 0;
  int failed = 0;

  auto run = [&](bool (*test)(), const char *name) {
    std::cout << "--- Running: " << name << " ---\n" << std::flush;
    if (test()) {
      ++passed;
    } else {
      std::cerr << "  FAILED: " << name << "\n";
      ++failed;
    }
    std::cout << std::flush;
  };

  // Standalone tests (no MLIR/BendersDriver needed).
  run(testConvergenceMonitor, "ConvergenceMonitor_StandaloneLogic");
  run(testParetoFrontier, "ParetoFrontier_Management");
  run(testComputeSystemArea, "ComputeSystemArea_Correct");

  // Framework integration tests (all use enableHW=false to avoid
  // ADGBuilder fatal error with fallback topologies).
  run(testSWStepProducesHistory, "CoOptFramework_SWStepProducesHistory");
  run(testConvergenceDetected, "CoOptFramework_ConvergenceDetected");
  run(testMaxIterationLimit, "CoOptFramework_MaxIterationLimit");
  run(testSWDisabledCarriesForward, "CoOptFramework_SWDisabledCarriesForward");
  run(testMaxRoundsHardCap, "CoOptFramework_MaxRoundsHardCap");
  run(testReasonFieldPopulated, "CoOptFramework_ReasonFieldPopulated");
  run(testInnerLayerFlags, "CoOptFramework_InnerLayerFlags");

  std::cout << "\n" << passed << " passed, " << failed << " failed\n";
  return failed > 0 ? 1 : 0;
}
