/// TemporalScheduler and ADGPartitioner unit tests.
///
/// Tests:
/// T1:  BATCH_SEQUENTIAL produces valid schedule
/// T2:  PIPELINE_PARALLEL produces valid schedule with overlap
/// T3:  PIPELINE_PARALLEL same-core serialization
/// T4:  PIPELINE_PARALLEL respects multi-hop dependencies
/// T5:  SPATIAL_PARALLEL produces valid schedule
/// T6:  SPATIAL_PARALLEL with intra-core serialization
/// T7:  SPATIAL_SHARING partitions ADG correctly
/// T8:  SPATIAL_SHARING partition validation catches overlap
/// T9:  SPATIAL_SHARING produces valid schedule (no reconfig)
/// T10: Configuration merge for SPATIAL_SHARING
/// T11: ExecutionMode string round-trip

#include "loom/SystemCompiler/ADGPartitioner.h"
#include "loom/SystemCompiler/TemporalScheduler.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace loom;

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

/// Build a minimal AssignmentResult with kernel-to-core mappings.
/// Takes a vector of (kernelName, coreInstanceIdx) pairs and the total
/// number of cores. All cores share the coreTypeName "PE_A".
static AssignmentResult
buildAssignment(const std::vector<std::pair<std::string, unsigned>> &mappings,
                unsigned numCores) {
  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.coreAssignments.resize(numCores);
  for (unsigned c = 0; c < numCores; ++c) {
    assignment.coreAssignments[c].coreInstanceIdx = c;
    assignment.coreAssignments[c].coreTypeName = "PE_A";
  }
  for (const auto &m : mappings) {
    assignment.kernelToCore[m.first] = m.second;
    assignment.coreAssignments[m.second].assignedKernels.push_back(m.first);
  }
  return assignment;
}

/// Build a CoreCostSummary for one core with the given per-kernel achieved IIs.
/// The kernelNames and achievedIIs vectors must be parallel.
static CoreCostSummary
buildCostSummary(const std::string &coreInstanceName,
                 const std::vector<std::string> &kernelNames,
                 const std::vector<unsigned> &achievedIIs) {
  CoreCostSummary cs;
  cs.coreInstanceName = coreInstanceName;
  cs.coreType = "PE_A";
  cs.success = true;
  for (size_t idx = 0; idx < kernelNames.size(); ++idx) {
    KernelMetrics km;
    km.kernelName = kernelNames[idx];
    km.achievedII = achievedIIs[idx];
    cs.kernelMetrics.push_back(km);
  }
  return cs;
}

/// Build a dependency contract from producer to consumer.
static ContractSpec buildContract(const std::string &producer,
                                  const std::string &consumer) {
  ContractSpec c;
  c.producerKernel = producer;
  c.consumerKernel = consumer;
  c.dataTypeName = "f32";
  c.productionRate = 256;
  return c;
}

//===----------------------------------------------------------------------===//
// T1: BATCH_SEQUENTIAL produces valid schedule
//===----------------------------------------------------------------------===//

static bool testBatchSequential() {
  // 2 cores, 3 kernels: K0, K1 on core 0, K2 on core 1. K0 -> K1 dependency.
  auto assignment = buildAssignment(
      {{"K0", 0}, {"K1", 0}, {"K2", 1}}, 2);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0", "K1"}, {2, 2}),
      buildCostSummary("PE_A_1", {"K2"}, {2}),
  };

  std::vector<ContractSpec> contracts = {buildContract("K0", "K1")};

  ExecutionModelConfig config;
  config.mode = ExecutionMode::BATCH_SEQUENTIAL;
  config.reconfigCycles = 100;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testBatchSequential - error: " << err << "\n";
    return false;
  }
  if (schedule.mode != ExecutionMode::BATCH_SEQUENTIAL) {
    std::cerr << "FAIL: testBatchSequential - wrong mode\n";
    return false;
  }
  // lookupTripCount returns 1000 by default.
  // Core 0: K0 exec = 1000*2 = 2000, reconfig = 100, K1 exec = 2000 -> 4100.
  // Core 1: K2 exec = 2000.
  bool foundCore0 = false;
  bool foundCore1 = false;
  for (const auto &cs : schedule.coreSchedules) {
    if (cs.kernelOrder.size() == 2) {
      foundCore0 = true;
      if (cs.totalCycles != 4100) {
        std::cerr << "FAIL: testBatchSequential - core 0 totalCycles="
                  << cs.totalCycles << " expected 4100\n";
        return false;
      }
      if (cs.reconfigCount != 1) {
        std::cerr << "FAIL: testBatchSequential - core 0 reconfigCount\n";
        return false;
      }
      // K0 must come before K1 (dependency ordering).
      if (cs.kernelOrder[0] != "K0" || cs.kernelOrder[1] != "K1") {
        std::cerr << "FAIL: testBatchSequential - core 0 kernel order\n";
        return false;
      }
    } else if (cs.kernelOrder.size() == 1) {
      foundCore1 = true;
      if (cs.totalCycles != 2000) {
        std::cerr << "FAIL: testBatchSequential - core 1 totalCycles="
                  << cs.totalCycles << " expected 2000\n";
        return false;
      }
    }
  }
  if (!foundCore0 || !foundCore1) {
    std::cerr << "FAIL: testBatchSequential - missing core schedules\n";
    return false;
  }
  if (schedule.maxCoreCycles != 4100) {
    std::cerr << "FAIL: testBatchSequential - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 4100\n";
    return false;
  }

  std::cerr << "PASS: testBatchSequential\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T2: PIPELINE_PARALLEL produces valid schedule with overlap
//===----------------------------------------------------------------------===//

static bool testPipelineParallelOverlap() {
  // K0 on core 0, K1 on core 1. Dependency K0 -> K1.
  // K0 duration = 1000 * 2 = 2000 cycles. K1 duration = 2000.
  // Pipeline initiation delay = 2000 / defaultTileCount(4) = 500.
  auto assignment = buildAssignment({{"K0", 0}, {"K1", 1}}, 2);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0"}, {2}),
      buildCostSummary("PE_A_1", {"K1"}, {2}),
  };

  std::vector<ContractSpec> contracts = {buildContract("K0", "K1")};

  ExecutionModelConfig config;
  config.mode = ExecutionMode::PIPELINE_PARALLEL;
  config.reconfigCycles = 100;
  config.defaultTileCount = 4;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testPipelineParallelOverlap - error: " << err << "\n";
    return false;
  }
  if (schedule.mode != ExecutionMode::PIPELINE_PARALLEL) {
    std::cerr << "FAIL: testPipelineParallelOverlap - wrong mode\n";
    return false;
  }

  // Find K0 and K1 start times from the schedule.
  uint64_t k0Start = 0, k1Start = 0;
  bool foundK0 = false, foundK1 = false;
  for (const auto &cs : schedule.coreSchedules) {
    for (const auto &kt : cs.kernelTimings) {
      if (kt.kernelName == "K0") {
        k0Start = kt.startTime;
        foundK0 = true;
      }
      if (kt.kernelName == "K1") {
        k1Start = kt.startTime;
        foundK1 = true;
      }
    }
  }
  if (!foundK0 || !foundK1) {
    std::cerr << "FAIL: testPipelineParallelOverlap - missing kernel timings\n";
    return false;
  }

  // K0 starts at 0, K1 starts at 500 (pipeline delay).
  if (k0Start != 0) {
    std::cerr << "FAIL: testPipelineParallelOverlap - K0 startTime="
              << k0Start << " expected 0\n";
    return false;
  }
  if (k1Start != 500) {
    std::cerr << "FAIL: testPipelineParallelOverlap - K1 startTime="
              << k1Start << " expected 500\n";
    return false;
  }

  // System latency = max(0+2000, 500+2000) = 2500 + nocOverhead.
  if (schedule.maxCoreCycles != 2500) {
    std::cerr << "FAIL: testPipelineParallelOverlap - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 2500\n";
    return false;
  }

  // Verify pipeline provides benefit over batch.
  // Batch would be: core 0 = 2000, core 1 = 2000, system = 2000 + noc.
  // But sequential would require K1 to wait for K0 to finish: 2000 + 2000 = 4000.
  // Pipeline: 2500 < 4000.
  if (schedule.maxCoreCycles >= 4000) {
    std::cerr << "FAIL: testPipelineParallelOverlap - no pipeline benefit\n";
    return false;
  }

  std::cerr << "PASS: testPipelineParallelOverlap\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T3: PIPELINE_PARALLEL same-core serialization
//===----------------------------------------------------------------------===//

static bool testPipelineSameCoreSerialization() {
  // 1 core, K0 and K1 both on core 0. K0 -> K1.
  // K0 duration = 1000 * 1 = 1000. K1 duration = 1000 * 1 = 1000.
  // Reconfig = 100.
  auto assignment = buildAssignment({{"K0", 0}, {"K1", 0}}, 1);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0", "K1"}, {1, 1}),
  };

  std::vector<ContractSpec> contracts = {buildContract("K0", "K1")};

  ExecutionModelConfig config;
  config.mode = ExecutionMode::PIPELINE_PARALLEL;
  config.reconfigCycles = 100;
  config.defaultTileCount = 4;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testPipelineSameCoreSerialization - error: " << err
              << "\n";
    return false;
  }

  // Find kernel start times.
  uint64_t k0Start = 0, k1Start = 0;
  for (const auto &cs : schedule.coreSchedules) {
    for (const auto &kt : cs.kernelTimings) {
      if (kt.kernelName == "K0")
        k0Start = kt.startTime;
      if (kt.kernelName == "K1")
        k1Start = kt.startTime;
    }
  }

  // Same core: K1 must wait for K0 to finish + reconfig.
  // K0 starts at 0, finishes at 1000. K1 starts at 1000 + 100 = 1100.
  if (k0Start != 0) {
    std::cerr << "FAIL: testPipelineSameCoreSerialization - K0 start="
              << k0Start << " expected 0\n";
    return false;
  }
  if (k1Start < 1100) {
    std::cerr << "FAIL: testPipelineSameCoreSerialization - K1 start="
              << k1Start << " expected >= 1100\n";
    return false;
  }

  // System latency = 1100 + 1000 = 2100 + noc.
  if (schedule.maxCoreCycles < 2100) {
    std::cerr << "FAIL: testPipelineSameCoreSerialization - maxCoreCycles="
              << schedule.maxCoreCycles << " expected >= 2100\n";
    return false;
  }

  std::cerr << "PASS: testPipelineSameCoreSerialization\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T4: PIPELINE_PARALLEL multi-hop dependencies
//===----------------------------------------------------------------------===//

static bool testPipelineMultiHop() {
  // 3 cores. K0 on core 0, K1 on core 1, K2 on core 2.
  // K0 -> K1, K1 -> K2. All durations = 1000*1 = 1000. Pipeline delay = 250.
  auto assignment = buildAssignment(
      {{"K0", 0}, {"K1", 1}, {"K2", 2}}, 3);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0"}, {1}),
      buildCostSummary("PE_A_1", {"K1"}, {1}),
      buildCostSummary("PE_A_2", {"K2"}, {1}),
  };

  std::vector<ContractSpec> contracts = {
      buildContract("K0", "K1"),
      buildContract("K1", "K2"),
  };

  ExecutionModelConfig config;
  config.mode = ExecutionMode::PIPELINE_PARALLEL;
  config.reconfigCycles = 100;
  config.defaultTileCount = 4; // delay = 1000 / 4 = 250

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testPipelineMultiHop - error: " << err << "\n";
    return false;
  }

  uint64_t k0Start = 0, k1Start = 0, k2Start = 0;
  for (const auto &cs : schedule.coreSchedules) {
    for (const auto &kt : cs.kernelTimings) {
      if (kt.kernelName == "K0")
        k0Start = kt.startTime;
      if (kt.kernelName == "K1")
        k1Start = kt.startTime;
      if (kt.kernelName == "K2")
        k2Start = kt.startTime;
    }
  }

  // K0 at 0. K1 at 250 (pipeline delay from K0). K2 at 500 (250 from K1).
  if (k0Start != 0) {
    std::cerr << "FAIL: testPipelineMultiHop - K0 start=" << k0Start << "\n";
    return false;
  }
  if (k1Start != 250) {
    std::cerr << "FAIL: testPipelineMultiHop - K1 start=" << k1Start
              << " expected 250\n";
    return false;
  }
  if (k2Start != 500) {
    std::cerr << "FAIL: testPipelineMultiHop - K2 start=" << k2Start
              << " expected 500\n";
    return false;
  }

  // System latency = 500 + 1000 = 1500 + noc.
  if (schedule.maxCoreCycles != 1500) {
    std::cerr << "FAIL: testPipelineMultiHop - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 1500\n";
    return false;
  }

  // Verify benefit over batch (3000 + noc).
  if (schedule.maxCoreCycles >= 3000) {
    std::cerr << "FAIL: testPipelineMultiHop - no pipeline benefit\n";
    return false;
  }

  std::cerr << "PASS: testPipelineMultiHop\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T5: SPATIAL_PARALLEL produces valid schedule
//===----------------------------------------------------------------------===//

static bool testSpatialParallel() {
  // 4 cores, 4 independent kernels, one per core.
  auto assignment = buildAssignment(
      {{"K0", 0}, {"K1", 1}, {"K2", 2}, {"K3", 3}}, 4);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0"}, {1}),
      buildCostSummary("PE_A_1", {"K1"}, {1}),
      buildCostSummary("PE_A_2", {"K2"}, {1}),
      buildCostSummary("PE_A_3", {"K3"}, {1}),
  };

  std::vector<ContractSpec> contracts; // No dependencies.

  ExecutionModelConfig config;
  config.mode = ExecutionMode::SPATIAL_PARALLEL;
  config.reconfigCycles = 100;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testSpatialParallel - error: " << err << "\n";
    return false;
  }
  if (schedule.mode != ExecutionMode::SPATIAL_PARALLEL) {
    std::cerr << "FAIL: testSpatialParallel - wrong mode\n";
    return false;
  }

  // All kernels start at time 0 (one per core), duration = 1000.
  for (const auto &cs : schedule.coreSchedules) {
    if (cs.totalCycles != 1000) {
      std::cerr << "FAIL: testSpatialParallel - core totalCycles="
                << cs.totalCycles << " expected 1000\n";
      return false;
    }
    for (const auto &kt : cs.kernelTimings) {
      if (kt.startTime != 0) {
        std::cerr << "FAIL: testSpatialParallel - " << kt.kernelName
                  << " startTime=" << kt.startTime << " expected 0\n";
        return false;
      }
    }
  }

  if (schedule.maxCoreCycles != 1000) {
    std::cerr << "FAIL: testSpatialParallel - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 1000\n";
    return false;
  }

  std::cerr << "PASS: testSpatialParallel\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T6: SPATIAL_PARALLEL with intra-core serialization
//===----------------------------------------------------------------------===//

static bool testSpatialParallelIntraCoreSerial() {
  // 2 cores. K0, K1 on core 0 (sequential). K2 on core 1.
  // No inter-kernel dependencies. II=1 for all.
  // K0 duration = 1000, K1 = 500 (via achievedII), K2 = 800.
  // NOTE: lookupTripCount always returns 1000, so we control duration via II.
  // K0: II=1, dur=1000. K1: II=1, dur=1000. K2: II=1, dur=1000.
  // With achievedII=1 and tripCount=1000 for all, dur=1000 each.
  // Core 0: 1000 + 100 + 1000 = 2100. Core 1: 1000.
  auto assignment = buildAssignment({{"K0", 0}, {"K1", 0}, {"K2", 1}}, 2);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0", "K1"}, {1, 1}),
      buildCostSummary("PE_A_1", {"K2"}, {1}),
  };

  std::vector<ContractSpec> contracts; // No dependencies.

  ExecutionModelConfig config;
  config.mode = ExecutionMode::SPATIAL_PARALLEL;
  config.reconfigCycles = 100;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testSpatialParallelIntraCoreSerial - error: " << err
              << "\n";
    return false;
  }

  bool foundCore0 = false, foundCore1 = false;
  for (const auto &cs : schedule.coreSchedules) {
    if (cs.kernelOrder.size() == 2) {
      foundCore0 = true;
      // Core 0: 1000 + 100 + 1000 = 2100
      if (cs.totalCycles != 2100) {
        std::cerr << "FAIL: testSpatialParallelIntraCoreSerial - core0 cycles="
                  << cs.totalCycles << " expected 2100\n";
        return false;
      }
    } else if (cs.kernelOrder.size() == 1) {
      foundCore1 = true;
      if (cs.totalCycles != 1000) {
        std::cerr << "FAIL: testSpatialParallelIntraCoreSerial - core1 cycles="
                  << cs.totalCycles << " expected 1000\n";
        return false;
      }
    }
  }
  if (!foundCore0 || !foundCore1) {
    std::cerr << "FAIL: testSpatialParallelIntraCoreSerial - missing cores\n";
    return false;
  }

  // Max = 2100 (core 0 is bottleneck). All cores start at 0.
  if (schedule.maxCoreCycles != 2100) {
    std::cerr << "FAIL: testSpatialParallelIntraCoreSerial - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 2100\n";
    return false;
  }

  std::cerr << "PASS: testSpatialParallelIntraCoreSerial\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T7: SPATIAL_SHARING partitions ADG correctly (2-way)
//===----------------------------------------------------------------------===//

static bool testSpatialSharingPartition2Way() {
  // 4x4 PE grid (16 PEs), request 2-way partition.
  CoreTypeSpec coreType;
  coreType.typeName = "PE_A";
  coreType.numPEs = 16;
  coreType.numFUs = 16;
  coreType.spmBytes = 4096;

  PartitionPlan plan =
      ADGPartitioner::generatePartitions(coreType, 2, 4, 4);

  if (plan.partitions.size() != 2) {
    std::cerr << "FAIL: testSpatialSharingPartition2Way - expected 2 "
              << "partitions, got " << plan.partitions.size() << "\n";
    return false;
  }

  // Row-wise split: p0 rows [0,2), p1 rows [2,4).
  const auto &p0 = plan.partitions[0];
  const auto &p1 = plan.partitions[1];

  if (p0.rowStart != 0 || p0.rowEnd != 2 || p0.colStart != 0 ||
      p0.colEnd != 4) {
    std::cerr << "FAIL: testSpatialSharingPartition2Way - p0 bounds\n";
    return false;
  }
  if (p1.rowStart != 2 || p1.rowEnd != 4 || p1.colStart != 0 ||
      p1.colEnd != 4) {
    std::cerr << "FAIL: testSpatialSharingPartition2Way - p1 bounds\n";
    return false;
  }
  if (p0.numPEs != 8 || p1.numPEs != 8) {
    std::cerr << "FAIL: testSpatialSharingPartition2Way - PE counts\n";
    return false;
  }

  // Validate: no overlap, full coverage.
  auto validation = ADGPartitioner::validatePartition(plan);
  if (!validation.valid) {
    std::cerr << "FAIL: testSpatialSharingPartition2Way - validation: "
              << validation.errorMessage << "\n";
    return false;
  }

  std::cerr << "PASS: testSpatialSharingPartition2Way\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T8: SPATIAL_SHARING partition validation catches overlap
//===----------------------------------------------------------------------===//

static bool testPartitionValidationOverlap() {
  // Manually construct an overlapping partition plan.
  PartitionPlan plan;
  plan.coreTypeName = "PE_A";
  plan.totalRows = 4;
  plan.totalCols = 4;
  plan.totalPEs = 16;

  PartitionSpec p0;
  p0.rowStart = 0;
  p0.rowEnd = 3; // overlaps with p1
  p0.colStart = 0;
  p0.colEnd = 4;
  p0.numPEs = 12;
  plan.partitions.push_back(p0);

  PartitionSpec p1;
  p1.rowStart = 2; // overlaps with p0 at row 2
  p1.rowEnd = 4;
  p1.colStart = 0;
  p1.colEnd = 4;
  p1.numPEs = 8;
  plan.partitions.push_back(p1);

  auto validation = ADGPartitioner::validatePartition(plan);
  if (validation.valid) {
    std::cerr << "FAIL: testPartitionValidationOverlap - should detect "
              << "overlap\n";
    return false;
  }

  std::cerr << "PASS: testPartitionValidationOverlap\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T9: SPATIAL_SHARING produces valid schedule (no reconfig)
//===----------------------------------------------------------------------===//

static bool testSpatialSharingSchedule() {
  // 2 cores. K0 and K1 share core 0. K2 on core 1.
  // K0 achievedII=2 -> dur=2000. K1 achievedII=1 -> dur=1000. K2 II=1->1000.
  auto assignment = buildAssignment(
      {{"K0", 0}, {"K1", 0}, {"K2", 1}}, 2);

  std::vector<CoreCostSummary> costSummaries = {
      buildCostSummary("PE_A_0", {"K0", "K1"}, {2, 1}),
      buildCostSummary("PE_A_1", {"K2"}, {1}),
  };

  std::vector<ContractSpec> contracts; // No dependencies.

  ExecutionModelConfig config;
  config.mode = ExecutionMode::SPATIAL_SHARING;
  config.reconfigCycles = 100;

  TemporalSchedule schedule;
  TemporalScheduler scheduler;
  std::string err =
      scheduler.schedule(assignment, costSummaries, contracts, config, schedule);

  if (!err.empty()) {
    std::cerr << "FAIL: testSpatialSharingSchedule - error: " << err << "\n";
    return false;
  }
  if (schedule.mode != ExecutionMode::SPATIAL_SHARING) {
    std::cerr << "FAIL: testSpatialSharingSchedule - wrong mode\n";
    return false;
  }

  // Core 0: concurrent. Latency = max(2000, 1000) = 2000. No reconfig.
  bool foundCore0 = false;
  for (const auto &cs : schedule.coreSchedules) {
    if (cs.kernelOrder.size() == 2) {
      foundCore0 = true;
      if (cs.totalCycles != 2000) {
        std::cerr << "FAIL: testSpatialSharingSchedule - core 0 totalCycles="
                  << cs.totalCycles << " expected 2000\n";
        return false;
      }
      if (cs.reconfigCount != 0) {
        std::cerr << "FAIL: testSpatialSharingSchedule - core 0 "
                  << "reconfigCount=" << cs.reconfigCount << " expected 0\n";
        return false;
      }
      // Both kernels start at time 0.
      for (const auto &kt : cs.kernelTimings) {
        if (kt.startTime != 0) {
          std::cerr << "FAIL: testSpatialSharingSchedule - "
                    << kt.kernelName << " startTime=" << kt.startTime
                    << " expected 0\n";
          return false;
        }
      }
    }
  }
  if (!foundCore0) {
    std::cerr << "FAIL: testSpatialSharingSchedule - missing core 0\n";
    return false;
  }

  // maxCoreCycles = max(2000, 1000) = 2000.
  if (schedule.maxCoreCycles != 2000) {
    std::cerr << "FAIL: testSpatialSharingSchedule - maxCoreCycles="
              << schedule.maxCoreCycles << " expected 2000\n";
    return false;
  }

  std::cerr << "PASS: testSpatialSharingSchedule\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T10: Configuration merge for SPATIAL_SHARING
//===----------------------------------------------------------------------===//

static bool testConfigMerge() {
  // 4x4 core, 2 partitions: rows [0,2) and [2,4).
  // Full config size = 16 PEs * 4 bytes/PE = 64 bytes.
  PartitionPlan plan;
  plan.coreTypeName = "PE_A";
  plan.totalRows = 4;
  plan.totalCols = 4;
  plan.totalPEs = 16;

  PartitionSpec p0;
  p0.rowStart = 0;
  p0.rowEnd = 2;
  p0.colStart = 0;
  p0.colEnd = 4;
  p0.numPEs = 8;
  plan.partitions.push_back(p0);

  PartitionSpec p1;
  p1.rowStart = 2;
  p1.rowEnd = 4;
  p1.colStart = 0;
  p1.colEnd = 4;
  p1.numPEs = 8;
  plan.partitions.push_back(p1);

  size_t fullConfigSize = 64; // 16 PEs * 4 bytes

  // Build partition configs: p0 sets 0xAA for PEs 0-7, p1 sets 0xBB for PEs 8-15.
  std::vector<uint8_t> p0Config(32, 0xAA); // 8 PEs * 4 bytes
  std::vector<uint8_t> p1Config(32, 0xBB); // 8 PEs * 4 bytes

  std::vector<std::vector<uint8_t>> partConfigs = {p0Config, p1Config};

  auto merged = ADGPartitioner::mergeConfigurations(partConfigs, plan,
                                                    fullConfigSize);

  if (merged.size() != fullConfigSize) {
    std::cerr << "FAIL: testConfigMerge - merged size=" << merged.size()
              << " expected " << fullConfigSize << "\n";
    return false;
  }

  // Check that first 32 bytes are 0xAA (PEs 0-7) and last 32 are 0xBB (8-15).
  bool firstHalfCorrect = true;
  bool secondHalfCorrect = true;
  for (size_t idx = 0; idx < 32; ++idx) {
    if (merged[idx] != 0xAA)
      firstHalfCorrect = false;
  }
  for (size_t idx = 32; idx < 64; ++idx) {
    if (merged[idx] != 0xBB)
      secondHalfCorrect = false;
  }

  if (!firstHalfCorrect) {
    std::cerr << "FAIL: testConfigMerge - first half not 0xAA\n";
    return false;
  }
  if (!secondHalfCorrect) {
    std::cerr << "FAIL: testConfigMerge - second half not 0xBB\n";
    return false;
  }

  std::cerr << "PASS: testConfigMerge\n";
  return true;
}

//===----------------------------------------------------------------------===//
// T11: ExecutionMode string round-trip
//===----------------------------------------------------------------------===//

static bool testExecutionModeRoundTrip() {
  ExecutionMode modes[] = {
      ExecutionMode::BATCH_SEQUENTIAL,
      ExecutionMode::PIPELINE_PARALLEL,
      ExecutionMode::SPATIAL_PARALLEL,
      ExecutionMode::SPATIAL_SHARING,
  };

  for (auto mode : modes) {
    const char *str = executionModeToString(mode);
    ExecutionMode parsed = executionModeFromString(str);
    if (parsed != mode) {
      std::cerr << "FAIL: testExecutionModeRoundTrip - " << str
                << " did not round-trip\n";
      return false;
    }
  }

  std::cerr << "PASS: testExecutionModeRoundTrip\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  unsigned passed = 0;
  unsigned failed = 0;

  auto run = [&](bool (*test)(), const char *name) {
    if (test()) {
      passed++;
    } else {
      failed++;
      std::cerr << "  -> FAILED: " << name << "\n";
    }
  };

  run(testBatchSequential, "T1: BATCH_SEQUENTIAL");
  run(testPipelineParallelOverlap, "T2: PIPELINE_PARALLEL overlap");
  run(testPipelineSameCoreSerialization, "T3: PIPELINE_PARALLEL same-core");
  run(testPipelineMultiHop, "T4: PIPELINE_PARALLEL multi-hop");
  run(testSpatialParallel, "T5: SPATIAL_PARALLEL");
  run(testSpatialParallelIntraCoreSerial, "T6: SPATIAL_PARALLEL intra-core");
  run(testSpatialSharingPartition2Way, "T7: SPATIAL_SHARING partition 2-way");
  run(testPartitionValidationOverlap, "T8: partition validation overlap");
  run(testSpatialSharingSchedule, "T9: SPATIAL_SHARING schedule");
  run(testConfigMerge, "T10: config merge");
  run(testExecutionModeRoundTrip, "T11: ExecutionMode round-trip");

  std::cerr << "\n=== Results: " << passed << " passed, " << failed
            << " failed ===\n";

  return failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
