/// HierarchicalCompiler unit tests for bilevel loop convergence (D1).
///
/// Tests:
///   T1: L1 produces valid AssignmentResult
///   T2: L1 type compatibility enforcement
///   T3: InfeasibilityCut prevents re-assignment (cut feedback)
///   T4: L1 infeasibility when all types banned
///   T5: ConvergenceTracker stall detection
///   T6: ConvergenceTracker convergence detection
///   T7: ConvergenceTracker best solution tracking
///   T8: CompilerConfig sub-options propagation

#include "loom/SystemCompiler/ConvergenceTracker.h"
#include "loom/SystemCompiler/InfeasibilityCut.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

using namespace loom;

/// Build a minimal 2x2 system architecture with 2 core types,
/// each having 2 instances.
///   Type-A (instances 0,1): supports "arith.addi" (4 FUs), "arith.muli" (2 FUs)
///   Type-B (instances 2,3): supports "arith.addi" (2 FUs), "arith.divsi" (2 FUs)
static SystemArchitecture buildMinimal2x2Arch() {
  SystemArchitecture arch;
  arch.nocSpec.meshRows = 2;
  arch.nocSpec.meshCols = 2;

  CoreTypeSpec typeA;
  typeA.typeName = "PE_A";
  typeA.instanceCount = 2;
  typeA.spmBytes = 8192;
  typeA.numPEs = 4;
  typeA.numFUs = 6;
  typeA.fuTypeCounts["arith.addi"] = 4;
  typeA.fuTypeCounts["arith.muli"] = 2;
  arch.coreTypes.push_back(typeA);

  CoreTypeSpec typeB;
  typeB.typeName = "PE_B";
  typeB.instanceCount = 2;
  typeB.spmBytes = 8192;
  typeB.numPEs = 4;
  typeB.numFUs = 4;
  typeB.fuTypeCounts["arith.addi"] = 2;
  typeB.fuTypeCounts["arith.divsi"] = 2;
  arch.coreTypes.push_back(typeB);

  return arch;
}

/// Build 4 kernel profiles with diverse FU requirements.
///  K0: needs addi(2), muli(1) -> compatible with PE_A
///  K1: needs addi(1), muli(1) -> compatible with PE_A
///  K2: needs addi(1), divsi(1) -> compatible with PE_B
///  K3: needs addi(1), divsi(1) -> compatible with PE_B
static std::vector<KernelProfile> build4KernelProfiles() {
  std::vector<KernelProfile> kernels(4);

  kernels[0].name = "K0";
  kernels[0].requiredOps["arith.addi"] = 2;
  kernels[0].requiredOps["arith.muli"] = 1;
  kernels[0].estimatedSPMBytes = 1024;
  kernels[0].estimatedMinII = 1;
  kernels[0].estimatedComputeCycles = 100.0;

  kernels[1].name = "K1";
  kernels[1].requiredOps["arith.addi"] = 1;
  kernels[1].requiredOps["arith.muli"] = 1;
  kernels[1].estimatedSPMBytes = 1024;
  kernels[1].estimatedMinII = 1;
  kernels[1].estimatedComputeCycles = 80.0;

  kernels[2].name = "K2";
  kernels[2].requiredOps["arith.addi"] = 1;
  kernels[2].requiredOps["arith.divsi"] = 1;
  kernels[2].estimatedSPMBytes = 1024;
  kernels[2].estimatedMinII = 1;
  kernels[2].estimatedComputeCycles = 120.0;

  kernels[3].name = "K3";
  kernels[3].requiredOps["arith.addi"] = 1;
  kernels[3].requiredOps["arith.divsi"] = 1;
  kernels[3].estimatedSPMBytes = 1024;
  kernels[3].estimatedMinII = 1;
  kernels[3].estimatedComputeCycles = 90.0;

  return kernels;
}

/// Build diamond dependency contracts: K0->K1, K0->K2, K1->K3, K2->K3.
static std::vector<ContractSpec> buildDiamondContracts() {
  std::vector<ContractSpec> contracts;

  ContractSpec c0;
  c0.producerKernel = "K0";
  c0.consumerKernel = "K1";
  c0.dataTypeName = "f32";
  c0.productionRate = 64;
  contracts.push_back(c0);

  ContractSpec c1;
  c1.producerKernel = "K0";
  c1.consumerKernel = "K2";
  c1.dataTypeName = "f32";
  c1.productionRate = 64;
  contracts.push_back(c1);

  ContractSpec c2;
  c2.producerKernel = "K1";
  c2.consumerKernel = "K3";
  c2.dataTypeName = "f32";
  c2.productionRate = 32;
  contracts.push_back(c2);

  ContractSpec c3;
  c3.producerKernel = "K2";
  c3.consumerKernel = "K3";
  c3.dataTypeName = "f32";
  c3.productionRate = 32;
  contracts.push_back(c3);

  return contracts;
}

// =========================================================================
// T1: L1 produces valid AssignmentResult
// =========================================================================
static bool testL1ProducesValidAssignment() {
  SystemArchitecture arch = buildMinimal2x2Arch();
  std::vector<KernelProfile> kernels = build4KernelProfiles();
  std::vector<ContractSpec> contracts = buildDiamondContracts();
  std::vector<InfeasibilityCut> cuts; // empty

  L1CoreAssigner assigner;
  L1AssignerOptions opts;
  opts.verbose = false;

  AssignmentResult result = assigner.solve(kernels, contracts, arch, cuts, opts);

  if (!result.feasible) {
    std::cerr << "FAIL: testL1ProducesValidAssignment - not feasible\n";
    return false;
  }

  // Check that all 4 kernels are assigned.
  if (result.kernelToCore.size() != 4) {
    std::cerr << "FAIL: testL1ProducesValidAssignment - expected 4 assignments, got "
              << result.kernelToCore.size() << "\n";
    return false;
  }

  // Check that every assigned core index is in range [0, 4).
  unsigned totalCores = arch.totalCoreInstances();
  for (const auto &entry : result.kernelToCore) {
    if (entry.second >= totalCores) {
      std::cerr << "FAIL: testL1ProducesValidAssignment - kernel '"
                << entry.first << "' assigned to out-of-range core "
                << entry.second << "\n";
      return false;
    }
  }

  // Check type compatibility: K0 and K1 need muli -> must be on PE_A (cores 0,1).
  // K2 and K3 need divsi -> must be on PE_B (cores 2,3).
  unsigned k0Core = result.kernelToCore.at("K0");
  unsigned k1Core = result.kernelToCore.at("K1");
  unsigned k2Core = result.kernelToCore.at("K2");
  unsigned k3Core = result.kernelToCore.at("K3");

  if (arch.typeNameForInstance(k0Core) != "PE_A") {
    std::cerr << "FAIL: testL1ProducesValidAssignment - K0 on wrong type: "
              << arch.typeNameForInstance(k0Core) << "\n";
    return false;
  }
  if (arch.typeNameForInstance(k1Core) != "PE_A") {
    std::cerr << "FAIL: testL1ProducesValidAssignment - K1 on wrong type: "
              << arch.typeNameForInstance(k1Core) << "\n";
    return false;
  }
  if (arch.typeNameForInstance(k2Core) != "PE_B") {
    std::cerr << "FAIL: testL1ProducesValidAssignment - K2 on wrong type: "
              << arch.typeNameForInstance(k2Core) << "\n";
    return false;
  }
  if (arch.typeNameForInstance(k3Core) != "PE_B") {
    std::cerr << "FAIL: testL1ProducesValidAssignment - K3 on wrong type: "
              << arch.typeNameForInstance(k3Core) << "\n";
    return false;
  }

  std::cout << "PASS: testL1ProducesValidAssignment\n";
  return true;
}

// =========================================================================
// T2: L1 type compatibility enforcement (no compatible core -> infeasible)
// =========================================================================
static bool testL1TypeCompatibility() {
  // Single core type that supports only "arith.addi".
  // Kernel requires "arith.divsi" which is not available.
  SystemArchitecture arch;
  arch.nocSpec.meshRows = 1;
  arch.nocSpec.meshCols = 1;

  CoreTypeSpec onlyType;
  onlyType.typeName = "PE_BASIC";
  onlyType.instanceCount = 1;
  onlyType.spmBytes = 8192;
  onlyType.numPEs = 4;
  onlyType.numFUs = 4;
  onlyType.fuTypeCounts["arith.addi"] = 4;
  arch.coreTypes.push_back(onlyType);

  std::vector<KernelProfile> kernels(1);
  kernels[0].name = "K_div";
  kernels[0].requiredOps["arith.divsi"] = 1;
  kernels[0].estimatedSPMBytes = 512;

  std::vector<ContractSpec> contracts;
  std::vector<InfeasibilityCut> cuts;

  L1CoreAssigner assigner;
  L1AssignerOptions opts;
  opts.verbose = false;

  AssignmentResult result = assigner.solve(kernels, contracts, arch, cuts, opts);

  if (result.feasible) {
    std::cerr << "FAIL: testL1TypeCompatibility - should be infeasible "
              << "(kernel requires divsi, core only has addi)\n";
    return false;
  }

  std::cout << "PASS: testL1TypeCompatibility\n";
  return true;
}

// =========================================================================
// T3: Cut feedback prevents re-assignment to banned core type
// =========================================================================
static bool testCutFeedbackPreventsReassignment() {
  SystemArchitecture arch = buildMinimal2x2Arch();

  // Single kernel that needs both addi and muli -> compatible with PE_A only.
  std::vector<KernelProfile> kernels(1);
  kernels[0].name = "K_mul";
  kernels[0].requiredOps["arith.addi"] = 1;
  kernels[0].requiredOps["arith.muli"] = 1;
  kernels[0].estimatedSPMBytes = 512;

  std::vector<ContractSpec> contracts;

  // Create a cut banning K_mul from PE_A.
  InfeasibilityCut cut;
  cut.kernelName = "K_mul";
  cut.coreType = "PE_A";
  cut.reason = CutReason::ROUTING_CONGESTION;
  cut.evidence = CongestionInfo{1.0};
  std::vector<InfeasibilityCut> cuts = {cut};

  L1CoreAssigner assigner;
  L1AssignerOptions opts;
  opts.verbose = false;

  AssignmentResult result = assigner.solve(kernels, contracts, arch, cuts, opts);

  // K_mul needs muli which PE_B doesn't have. With PE_A banned by cut,
  // there's no valid assignment.
  if (result.feasible) {
    std::cerr << "FAIL: testCutFeedbackPreventsReassignment - "
              << "should be infeasible (PE_A banned, PE_B incompatible)\n";
    return false;
  }

  std::cout << "PASS: testCutFeedbackPreventsReassignment\n";
  return true;
}

// =========================================================================
// T4: L1 infeasibility when all alternatives are banned
// =========================================================================
static bool testL1InfeasibilityAllBanned() {
  SystemArchitecture arch = buildMinimal2x2Arch();

  // Kernel compatible with both PE_A and PE_B (only needs addi).
  std::vector<KernelProfile> kernels(1);
  kernels[0].name = "K_add";
  kernels[0].requiredOps["arith.addi"] = 1;
  kernels[0].estimatedSPMBytes = 256;

  std::vector<ContractSpec> contracts;

  // Ban K_add from both PE_A and PE_B.
  InfeasibilityCut cutA;
  cutA.kernelName = "K_add";
  cutA.coreType = "PE_A";
  cutA.reason = CutReason::SPM_OVERFLOW;
  cutA.evidence = SPMInfo{16384, 8192};

  InfeasibilityCut cutB;
  cutB.kernelName = "K_add";
  cutB.coreType = "PE_B";
  cutB.reason = CutReason::SPM_OVERFLOW;
  cutB.evidence = SPMInfo{16384, 8192};

  std::vector<InfeasibilityCut> cuts = {cutA, cutB};

  L1CoreAssigner assigner;
  L1AssignerOptions opts;
  opts.verbose = false;

  AssignmentResult result = assigner.solve(kernels, contracts, arch, cuts, opts);

  if (result.feasible) {
    std::cerr << "FAIL: testL1InfeasibilityAllBanned - "
              << "should be infeasible (all core types banned)\n";
    return false;
  }

  std::cout << "PASS: testL1InfeasibilityAllBanned\n";
  return true;
}

// =========================================================================
// T5: ConvergenceTracker stall detection
// =========================================================================
static bool testConvergenceTrackerStallDetection() {
  ConvergenceTracker tracker(10, 3); // stallWindow = 3

  // Iterations 1-2: improving objective, some new cuts.
  tracker.recordIteration(1, 2, 100.0, false);
  tracker.recordSuccess(1, 100.0, 1);

  tracker.recordIteration(2, 1, 80.0, true);
  tracker.recordSuccess(2, 80.0, 2);

  // Iteration 3: no improvement, no new cuts.
  tracker.recordIteration(3, 0, 80.0, true);

  // After 3 iterations: only 1 non-improving iteration in window -> not stalled.
  if (tracker.isStalled()) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "stalled too early (after iter 3)\n";
    return false;
  }

  // Iteration 4: still no improvement.
  tracker.recordIteration(4, 0, 80.0, true);

  if (tracker.isStalled()) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "stalled too early (after iter 4, only 2 in window)\n";
    return false;
  }

  // Iteration 5: still no improvement, no new cuts -> now 3 consecutive.
  tracker.recordIteration(5, 0, 80.0, true);

  if (!tracker.isStalled()) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "should be stalled after 3 non-improving iterations\n";
    return false;
  }

  // Verify best solution tracking.
  if (!tracker.hasSolution()) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "should have a solution\n";
    return false;
  }
  if (std::fabs(tracker.getBestObjective() - 80.0) > 0.001) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "best objective should be 80.0, got "
              << tracker.getBestObjective() << "\n";
    return false;
  }
  if (tracker.getBestResultTag() != 2) {
    std::cerr << "FAIL: testConvergenceTrackerStallDetection - "
              << "best result tag should be 2, got "
              << tracker.getBestResultTag() << "\n";
    return false;
  }

  std::cout << "PASS: testConvergenceTrackerStallDetection\n";
  return true;
}

// =========================================================================
// T6: ConvergenceTracker convergence detection
// =========================================================================
static bool testConvergenceTrackerConvergenceDetection() {
  ConvergenceTracker tracker(10, 3);

  // Single iteration: all mapped, zero new cuts -> converged.
  tracker.recordIteration(1, 0, 50.0, true);
  tracker.recordSuccess(1, 50.0, 1);

  if (!tracker.isConverged()) {
    std::cerr << "FAIL: testConvergenceTrackerConvergenceDetection - "
              << "should be converged (all mapped, no new cuts)\n";
    return false;
  }

  if (tracker.isStalled()) {
    std::cerr << "FAIL: testConvergenceTrackerConvergenceDetection - "
              << "should not be stalled\n";
    return false;
  }

  if (!tracker.hasSolution()) {
    std::cerr << "FAIL: testConvergenceTrackerConvergenceDetection - "
              << "should have a solution\n";
    return false;
  }

  std::cout << "PASS: testConvergenceTrackerConvergenceDetection\n";
  return true;
}

// =========================================================================
// T7: ConvergenceTracker best solution tracking
// =========================================================================
static bool testConvergenceTrackerBestSolution() {
  ConvergenceTracker tracker(10, 5);

  // Record 3 successful iterations with different objectives.
  tracker.recordIteration(1, 2, 200.0, true);
  tracker.recordSuccess(1, 200.0, 1);

  tracker.recordIteration(2, 1, 150.0, true);
  tracker.recordSuccess(2, 150.0, 2);

  tracker.recordIteration(3, 0, 180.0, true); // worse than iter 2
  tracker.recordSuccess(3, 180.0, 3);

  // Best should be iteration 2 (objective = 150.0).
  if (std::fabs(tracker.getBestObjective() - 150.0) > 0.001) {
    std::cerr << "FAIL: testConvergenceTrackerBestSolution - "
              << "best objective should be 150.0, got "
              << tracker.getBestObjective() << "\n";
    return false;
  }
  if (tracker.getBestResultTag() != 2) {
    std::cerr << "FAIL: testConvergenceTrackerBestSolution - "
              << "best tag should be 2, got " << tracker.getBestResultTag() << "\n";
    return false;
  }

  // History should have 3 entries.
  if (tracker.history().size() != 3) {
    std::cerr << "FAIL: testConvergenceTrackerBestSolution - "
              << "expected 3 history entries, got "
              << tracker.history().size() << "\n";
    return false;
  }

  // Verify history content.
  if (tracker.history()[0].iterationIndex != 1 ||
      tracker.history()[0].numNewCuts != 2) {
    std::cerr << "FAIL: testConvergenceTrackerBestSolution - "
              << "history[0] incorrect\n";
    return false;
  }
  if (tracker.history()[1].iterationIndex != 2 ||
      tracker.history()[1].numNewCuts != 1) {
    std::cerr << "FAIL: testConvergenceTrackerBestSolution - "
              << "history[1] incorrect\n";
    return false;
  }

  std::cout << "PASS: testConvergenceTrackerBestSolution\n";
  return true;
}

// =========================================================================
// T8: CompilerConfig sub-options propagation
// =========================================================================
static bool testCompilerConfigSubOptions() {
  loom::tapestry::CompilerConfig config;

  // Verify defaults.
  if (config.maxIterations != 10) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - maxIterations default\n";
    return false;
  }
  if (config.convergenceStallWindow != 3) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - stallWindow default\n";
    return false;
  }
  if (std::fabs(config.mapperBudgetSeconds - 15.0) > 0.001) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - mapperBudgetSeconds default\n";
    return false;
  }

  // L1 options are accessible and have sane defaults.
  if (std::fabs(config.l1Options.latencyWeight - 1.0) > 0.001) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - l1 latencyWeight default\n";
    return false;
  }
  if (config.l1Options.verbose != false) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - l1 verbose default\n";
    return false;
  }

  // NoC options accessible.
  if (config.nocOptions.routing != NoCSchedulerOptions::XY_DOR) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - noc routing default\n";
    return false;
  }

  // Buffer options accessible.
  if (std::fabs(config.bufferOptions.spmReserveFraction - 0.2) > 0.001) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - buffer reserve default\n";
    return false;
  }

  // DMA options accessible.
  if (config.dmaOptions.estimatedComputeCycles != 1000) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - dma compute default\n";
    return false;
  }

  // Execution model default.
  if (config.executionModel.mode != ExecutionMode::BATCH_SEQUENTIAL) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - exec mode default\n";
    return false;
  }

  // CompilationResult has allCuts field.
  loom::tapestry::CompilationResult compResult;
  if (!compResult.allCuts.empty()) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - allCuts not empty\n";
    return false;
  }

  // CompilationResult has finalAssignment field.
  if (compResult.finalAssignment.has_value()) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - finalAssignment set\n";
    return false;
  }

  // CompilationResult has nocSchedule field.
  if (compResult.nocSchedule.has_value()) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - nocSchedule set\n";
    return false;
  }

  // Set and verify sub-options can be customized.
  config.l1Options.latencyWeight = 2.0;
  config.l1Options.maxSolverTimeSec = 120;
  config.nocOptions.routing = NoCSchedulerOptions::YX_DOR;
  config.convergenceStallWindow = 5;

  if (std::fabs(config.l1Options.latencyWeight - 2.0) > 0.001) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - modified l1 weight\n";
    return false;
  }
  if (config.l1Options.maxSolverTimeSec != 120) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - modified solver time\n";
    return false;
  }
  if (config.nocOptions.routing != NoCSchedulerOptions::YX_DOR) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - modified noc routing\n";
    return false;
  }
  if (config.convergenceStallWindow != 5) {
    std::cerr << "FAIL: testCompilerConfigSubOptions - modified stall window\n";
    return false;
  }

  std::cout << "PASS: testCompilerConfigSubOptions\n";
  return true;
}

// =========================================================================
// main
// =========================================================================
int main() {
  int failures = 0;

  if (!testL1ProducesValidAssignment()) ++failures;
  if (!testL1TypeCompatibility()) ++failures;
  if (!testCutFeedbackPreventsReassignment()) ++failures;
  if (!testL1InfeasibilityAllBanned()) ++failures;
  if (!testConvergenceTrackerStallDetection()) ++failures;
  if (!testConvergenceTrackerConvergenceDetection()) ++failures;
  if (!testConvergenceTrackerBestSolution()) ++failures;
  if (!testCompilerConfigSubOptions()) ++failures;

  std::cout << "\n" << (8 - failures) << "/8 tests passed";
  if (failures > 0)
    std::cout << " (" << failures << " FAILED)";
  std::cout << "\n";

  return failures > 0 ? 1 : 0;
}
