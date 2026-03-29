/// NoC/Buffer/DMA wiring integration tests (D3).
///
/// Tests:
///   T1: NoC schedule produces valid routes
///   T2: NoC schedule handles co-located kernels (zero-hop)
///   T3: Buffer allocation fits SPM
///   T4: Buffer allocation falls back to L2 when SPM overflows
///   T5: Buffer allocation with double-buffering
///   T6: DMA schedule respects producer-consumer ordering
///   T7: DMA schedule with double-buffer overlap
///   T8: End-to-end iteration includes NoC + buffer + DMA
///   T9: NoC contention detection

#include "loom/SystemCompiler/BendersHelpers.h"
#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/DMAScheduler.h"
#include "loom/SystemCompiler/L1CoreAssignment.h"
#include "loom/SystemCompiler/NoCScheduler.h"
#include "loom/SystemCompiler/SystemTypes.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace loom;

/// Build a 2x2 mesh architecture with a single core type (4 instances).
/// Each core has configurable SPM size.
static SystemArchitecture build2x2Arch(uint64_t spmPerCore = 16384) {
  SystemArchitecture arch;
  arch.nocSpec.meshRows = 2;
  arch.nocSpec.meshCols = 2;
  arch.nocSpec.flitWidth = 32;
  arch.nocSpec.routerPipelineStages = 2;
  arch.nocSpec.linkBandwidth = 1;

  arch.sharedMemSpec.l2SizeBytes = 262144;
  arch.sharedMemSpec.numBanks = 4;

  CoreTypeSpec coreType;
  coreType.typeName = "PE";
  coreType.instanceCount = 4;
  coreType.spmBytes = spmPerCore;
  coreType.numPEs = 4;
  coreType.numFUs = 4;
  coreType.fuTypeCounts["arith.addi"] = 4;
  arch.coreTypes.push_back(coreType);

  return arch;
}

/// Build a simple AssignmentResult placing kernel A on core srcCore
/// and kernel B on core dstCore.
static AssignmentResult buildAssignment2Kernels(
    unsigned srcCore, unsigned dstCore,
    const SystemArchitecture &arch) {
  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["A"] = srcCore;
  assignment.kernelToCore["B"] = dstCore;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[srcCore].assignedKernels.push_back("A");
  assignment.coreAssignments[dstCore].assignedKernels.push_back("B");

  return assignment;
}

/// Build a single contract edge A->B with specified data volume.
static std::vector<ContractSpec> buildSingleEdge(
    int64_t productionRate = 1024, const std::string &dataType = "i8") {
  ContractSpec c;
  c.producerKernel = "A";
  c.consumerKernel = "B";
  c.dataTypeName = dataType;
  c.productionRate = productionRate;
  c.minBufferElements = productionRate;
  return {c};
}

// =========================================================================
// T1: NoC schedule produces valid routes
// =========================================================================
static bool testNoCScheduleValidRoutes() {
  SystemArchitecture arch = build2x2Arch();
  // Place A on core 0 (0,0), B on core 3 (1,1).
  AssignmentResult assignment = buildAssignment2Kernels(0, 3, arch);
  std::vector<ContractSpec> contracts = buildSingleEdge(1024, "i8");

  NoCScheduler scheduler;
  NoCSchedulerOptions opts;
  opts.routing = NoCSchedulerOptions::XY_DOR;
  opts.verbose = false;

  NoCSchedule schedule =
      scheduler.schedule(assignment, contracts, arch, opts);

  // Should have exactly one route for the A->B edge.
  if (schedule.routes.size() != 1) {
    std::cerr << "FAIL: testNoCScheduleValidRoutes - expected 1 route, got "
              << schedule.routes.size() << "\n";
    return false;
  }

  const NoCRoute &route = schedule.routes[0];

  // XY routing from (0,0) to (1,1): go X first (col 0->1), then Y (row 0->1).
  // Hops: (0,0) -> (0,1) -> (1,1) = 2 links.
  if (route.numHops != 2) {
    std::cerr << "FAIL: testNoCScheduleValidRoutes - expected 2 hops, got "
              << route.numHops << "\n";
    return false;
  }

  // Check that route contains 3 positions (src + 2 intermediate/dst).
  if (route.hops.size() != 3) {
    std::cerr << "FAIL: testNoCScheduleValidRoutes - expected 3 hop positions, got "
              << route.hops.size() << "\n";
    return false;
  }

  // Verify start and end positions.
  if (route.hops.front() != std::make_pair(0, 0)) {
    std::cerr << "FAIL: testNoCScheduleValidRoutes - route does not start at (0,0)\n";
    return false;
  }
  if (route.hops.back() != std::make_pair(1, 1)) {
    std::cerr << "FAIL: testNoCScheduleValidRoutes - route does not end at (1,1)\n";
    return false;
  }

  // All intermediate links should be valid (adjacent mesh nodes).
  for (size_t i = 0; i + 1 < route.hops.size(); ++i) {
    int dr = std::abs(route.hops[i + 1].first - route.hops[i].first);
    int dc = std::abs(route.hops[i + 1].second - route.hops[i].second);
    if (dr + dc != 1) {
      std::cerr << "FAIL: testNoCScheduleValidRoutes - non-adjacent hop at index "
                << i << "\n";
      return false;
    }
  }

  // Per-link utilization should be non-negative and within bounds.
  for (const auto &lu : schedule.linkUtilizations) {
    if (lu.utilization < 0.0) {
      std::cerr << "FAIL: testNoCScheduleValidRoutes - negative utilization\n";
      return false;
    }
  }

  std::cout << "PASS: testNoCScheduleValidRoutes\n";
  return true;
}

// =========================================================================
// T2: NoC schedule handles co-located kernels (zero-hop)
// =========================================================================
static bool testNoCScheduleColocated() {
  SystemArchitecture arch = build2x2Arch();
  // Place both A and B on core 0 (same core).
  AssignmentResult assignment = buildAssignment2Kernels(0, 0, arch);
  std::vector<ContractSpec> contracts = buildSingleEdge(1024, "i8");

  NoCScheduler scheduler;
  NoCSchedulerOptions opts;
  opts.verbose = false;

  NoCSchedule schedule =
      scheduler.schedule(assignment, contracts, arch, opts);

  // Co-located kernels should produce zero routes (intra-core).
  if (!schedule.routes.empty()) {
    std::cerr << "FAIL: testNoCScheduleColocated - expected 0 routes, got "
              << schedule.routes.size() << "\n";
    return false;
  }

  // No mesh links should be consumed.
  if (!schedule.linkUtilizations.empty()) {
    std::cerr << "FAIL: testNoCScheduleColocated - expected 0 link utilizations, got "
              << schedule.linkUtilizations.size() << "\n";
    return false;
  }

  // Total transfer cycles should be zero.
  if (schedule.totalTransferCycles != 0) {
    std::cerr << "FAIL: testNoCScheduleColocated - expected 0 transfer cycles, got "
              << schedule.totalTransferCycles << "\n";
    return false;
  }

  std::cout << "PASS: testNoCScheduleColocated\n";
  return true;
}

// =========================================================================
// T3: Buffer allocation fits SPM
// =========================================================================
static bool testBufferAllocationFitsSPM() {
  // Core with 16 KB SPM. Two edges arriving, each requiring 4 KB buffer.
  // With spmReserveFraction = 0.0, full 16 KB available.
  SystemArchitecture arch = build2x2Arch(16384);

  // Two edges: X->C (4 KB) and Y->C (4 KB), both arrive at core 1.
  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["X"] = 0;
  assignment.kernelToCore["Y"] = 2;
  assignment.kernelToCore["C"] = 1;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[0].assignedKernels.push_back("X");
  assignment.coreAssignments[2].assignedKernels.push_back("Y");
  assignment.coreAssignments[1].assignedKernels.push_back("C");

  ContractSpec c1;
  c1.producerKernel = "X";
  c1.consumerKernel = "C";
  c1.dataTypeName = "i8";
  c1.productionRate = 4096;
  c1.minBufferElements = 4096;
  c1.visibility = Visibility::LOCAL_SPM;

  ContractSpec c2;
  c2.producerKernel = "Y";
  c2.consumerKernel = "C";
  c2.dataTypeName = "i8";
  c2.productionRate = 4096;
  c2.minBufferElements = 4096;
  c2.visibility = Visibility::LOCAL_SPM;

  std::vector<ContractSpec> contracts = {c1, c2};

  // Schedule NoC first.
  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  // Allocate buffers.
  BufferAllocator allocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0; // Use full SPM.
  bufOpts.preferDoubleBuffering = false;

  BufferAllocationPlan plan =
      allocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  if (!plan.feasible) {
    std::cerr << "FAIL: testBufferAllocationFitsSPM - plan not feasible\n";
    return false;
  }

  // Both buffers should be in SPM (consumer side).
  if (plan.allocations.size() != 2) {
    std::cerr << "FAIL: testBufferAllocationFitsSPM - expected 2 allocations, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  unsigned spmCount = 0;
  uint64_t totalSPMUsed = 0;
  for (const auto &alloc : plan.allocations) {
    if (alloc.location == BufferAllocation::SPM_CONSUMER ||
        alloc.location == BufferAllocation::SPM_PRODUCER) {
      spmCount++;
      totalSPMUsed += alloc.sizeBytes;
    }
  }

  if (spmCount != 2) {
    std::cerr << "FAIL: testBufferAllocationFitsSPM - expected 2 SPM allocations, got "
              << spmCount << "\n";
    return false;
  }

  if (totalSPMUsed != 8192) {
    std::cerr << "FAIL: testBufferAllocationFitsSPM - expected 8192 bytes SPM used, got "
              << totalSPMUsed << "\n";
    return false;
  }

  // No L2 or DRAM should be used.
  if (plan.l2UsedBytes != 0) {
    std::cerr << "FAIL: testBufferAllocationFitsSPM - expected 0 L2 used, got "
              << plan.l2UsedBytes << "\n";
    return false;
  }

  std::cout << "PASS: testBufferAllocationFitsSPM\n";
  return true;
}

// =========================================================================
// T4: Buffer allocation falls back to L2 when SPM overflows
// =========================================================================
static bool testBufferAllocationL2Fallback() {
  // Core with 4 KB SPM. Two producers send 4 KB each to the same consumer.
  // Consumer SPM holds only one 4 KB buffer. Producer SPM also holds only
  // one 4 KB buffer. The second edge's buffer must spill to L2.
  //
  // We use 2 producers and 1 consumer, all with 4 KB SPM.
  // Edge X->C: 4 KB, fits consumer SPM.
  // Edge Y->C: 4 KB, consumer full; producer Y core (core 2) has 4 KB but
  //            has no room if we fill it too. Instead, we make the edge
  //            require 5 KB which exceeds any single core's SPM.
  //
  // Simpler approach: use 4 KB SPM cores, one edge is 4 KB (fits consumer),
  // another edge is 5 KB (does not fit any core's SPM -> goes to L2).
  SystemArchitecture arch = build2x2Arch(4096);

  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["X"] = 0;
  assignment.kernelToCore["Y"] = 2;
  assignment.kernelToCore["C"] = 1;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[0].assignedKernels.push_back("X");
  assignment.coreAssignments[2].assignedKernels.push_back("Y");
  assignment.coreAssignments[1].assignedKernels.push_back("C");

  ContractSpec c1;
  c1.producerKernel = "X";
  c1.consumerKernel = "C";
  c1.dataTypeName = "i8";
  c1.productionRate = 2048;
  c1.minBufferElements = 2048;
  c1.visibility = Visibility::SHARED_L2;

  ContractSpec c2;
  c2.producerKernel = "Y";
  c2.consumerKernel = "C";
  c2.dataTypeName = "i8";
  c2.productionRate = 5000;
  c2.minBufferElements = 5000;
  c2.visibility = Visibility::SHARED_L2;

  std::vector<ContractSpec> contracts = {c1, c2};

  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  BufferAllocator allocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0;
  bufOpts.preferDoubleBuffering = false;

  BufferAllocationPlan plan =
      allocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  if (!plan.feasible) {
    std::cerr << "FAIL: testBufferAllocationL2Fallback - plan not feasible\n";
    return false;
  }

  if (plan.allocations.size() != 2) {
    std::cerr << "FAIL: testBufferAllocationL2Fallback - expected 2 allocations, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  // The 2 KB edge should fit in SPM; the 5 KB edge exceeds any core's 4 KB
  // SPM and must go to L2.
  unsigned spmCount = 0;
  unsigned l2Count = 0;
  for (const auto &alloc : plan.allocations) {
    if (alloc.location == BufferAllocation::SPM_CONSUMER ||
        alloc.location == BufferAllocation::SPM_PRODUCER)
      spmCount++;
    else if (alloc.location == BufferAllocation::SHARED_L2)
      l2Count++;
  }

  if (spmCount != 1) {
    std::cerr << "FAIL: testBufferAllocationL2Fallback - expected 1 SPM, got "
              << spmCount << "\n";
    return false;
  }
  if (l2Count != 1) {
    std::cerr << "FAIL: testBufferAllocationL2Fallback - expected 1 L2, got "
              << l2Count << "\n";
    return false;
  }

  // L2 should have 5000 bytes used.
  if (plan.l2UsedBytes != 5000) {
    std::cerr << "FAIL: testBufferAllocationL2Fallback - expected 5000 L2 used, got "
              << plan.l2UsedBytes << "\n";
    return false;
  }

  // Verify SPM usage on consumer core does not exceed capacity.
  for (const auto &usage : plan.coreSPMUsage) {
    if (usage.usedBytes > arch.typeForInstance(usage.coreInstanceIdx).spmBytes) {
      std::cerr << "FAIL: testBufferAllocationL2Fallback - SPM overflow on core "
                << usage.coreName << "\n";
      return false;
    }
  }

  std::cout << "PASS: testBufferAllocationL2Fallback\n";
  return true;
}

// =========================================================================
// T5: Buffer allocation with double-buffering
// =========================================================================
static bool testBufferAllocationDoubleBuffering() {
  // Core with 16 KB SPM. One edge requiring 4 KB buffer, double-buffering on.
  SystemArchitecture arch = build2x2Arch(16384);
  AssignmentResult assignment = buildAssignment2Kernels(0, 1, arch);

  ContractSpec c;
  c.producerKernel = "A";
  c.consumerKernel = "B";
  c.dataTypeName = "i8";
  c.productionRate = 4096;
  c.minBufferElements = 4096;
  c.doubleBuffering = true;
  c.visibility = Visibility::LOCAL_SPM;
  std::vector<ContractSpec> contracts = {c};

  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  BufferAllocator allocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0;
  bufOpts.preferDoubleBuffering = true;

  BufferAllocationPlan plan =
      allocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  if (!plan.feasible) {
    std::cerr << "FAIL: testBufferAllocationDoubleBuffering - not feasible\n";
    return false;
  }

  if (plan.allocations.size() != 1) {
    std::cerr << "FAIL: testBufferAllocationDoubleBuffering - expected 1 allocation, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  const auto &alloc = plan.allocations[0];
  if (!alloc.doubleBuffered) {
    std::cerr << "FAIL: testBufferAllocationDoubleBuffering - not double-buffered\n";
    return false;
  }

  // Double-buffering should reserve 2x the buffer size = 8192 bytes.
  if (alloc.sizeBytes != 8192) {
    std::cerr << "FAIL: testBufferAllocationDoubleBuffering - expected 8192 bytes, got "
              << alloc.sizeBytes << "\n";
    return false;
  }

  // Should fit in SPM (16 KB capacity).
  if (alloc.location != BufferAllocation::SPM_CONSUMER &&
      alloc.location != BufferAllocation::SPM_PRODUCER) {
    std::cerr << "FAIL: testBufferAllocationDoubleBuffering - expected SPM location\n";
    return false;
  }

  std::cout << "PASS: testBufferAllocationDoubleBuffering\n";
  return true;
}

// =========================================================================
// T6: DMA schedule respects producer-consumer ordering
// =========================================================================
static bool testDMAScheduleOrdering() {
  // Dependency chain: A -> B -> C, each on a different core.
  SystemArchitecture arch = build2x2Arch(16384);

  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["A"] = 0;
  assignment.kernelToCore["B"] = 1;
  assignment.kernelToCore["C"] = 2;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[0].assignedKernels.push_back("A");
  assignment.coreAssignments[1].assignedKernels.push_back("B");
  assignment.coreAssignments[2].assignedKernels.push_back("C");

  ContractSpec c1;
  c1.producerKernel = "A";
  c1.consumerKernel = "B";
  c1.dataTypeName = "i8";
  c1.productionRate = 256;
  c1.minBufferElements = 256;
  c1.visibility = Visibility::LOCAL_SPM;

  ContractSpec c2;
  c2.producerKernel = "B";
  c2.consumerKernel = "C";
  c2.dataTypeName = "i8";
  c2.productionRate = 256;
  c2.minBufferElements = 256;
  c2.visibility = Visibility::LOCAL_SPM;

  std::vector<ContractSpec> contracts = {c1, c2};

  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  BufferAllocator bufAllocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0;
  bufOpts.preferDoubleBuffering = false;
  BufferAllocationPlan bufferPlan =
      bufAllocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  DMAScheduler dmaScheduler;
  DMASchedulerOptions dmaOpts;
  dmaOpts.estimatedComputeCycles = 500;
  dmaOpts.kernelComputeCycles["A"] = 500;
  dmaOpts.kernelComputeCycles["B"] = 500;
  dmaOpts.kernelComputeCycles["C"] = 500;

  DMASchedule dmaSchedule = dmaScheduler.schedule(
      bufferPlan, nocSchedule, contracts, assignment, arch, dmaOpts);

  // Should have exactly 2 transfers: A->B and B->C.
  if (dmaSchedule.transfers.size() != 2) {
    std::cerr << "FAIL: testDMAScheduleOrdering - expected 2 transfers, got "
              << dmaSchedule.transfers.size() << "\n";
    return false;
  }

  // Find the two transfers.
  const DMATransfer *xferAB = nullptr;
  const DMATransfer *xferBC = nullptr;
  for (const auto &xfer : dmaSchedule.transfers) {
    if (xfer.contractEdgeName == "A -> B")
      xferAB = &xfer;
    else if (xfer.contractEdgeName == "B -> C")
      xferBC = &xfer;
  }

  if (!xferAB || !xferBC) {
    std::cerr << "FAIL: testDMAScheduleOrdering - missing transfer(s)\n";
    return false;
  }

  // T1 (A->B) should start after A's computation completes.
  if (xferAB->startCycle < 500) {
    std::cerr << "FAIL: testDMAScheduleOrdering - A->B starts too early: "
              << xferAB->startCycle << " (A completes at 500)\n";
    return false;
  }

  // T2 (B->C) should start after B's computation completes.
  // B cannot start until A->B data arrives, so T2 start >= arrival of A->B + B compute.
  if (xferBC->startCycle < xferAB->endCycle + 500) {
    std::cerr << "FAIL: testDMAScheduleOrdering - B->C starts before B completes: "
              << xferBC->startCycle << " (B data ready at "
              << xferAB->endCycle << " + 500 compute)\n";
    return false;
  }

  // T2 cannot begin before T1's data has been delivered.
  if (xferBC->startCycle < xferAB->endCycle) {
    std::cerr << "FAIL: testDMAScheduleOrdering - B->C starts before A->B ends\n";
    return false;
  }

  std::cout << "PASS: testDMAScheduleOrdering\n";
  return true;
}

// =========================================================================
// T7: DMA schedule with double-buffer overlap
// =========================================================================
static bool testDMAScheduleDoubleBufferOverlap() {
  // Pipeline: A -> B with double-buffering.
  SystemArchitecture arch = build2x2Arch(16384);
  AssignmentResult assignment = buildAssignment2Kernels(0, 1, arch);

  ContractSpec c;
  c.producerKernel = "A";
  c.consumerKernel = "B";
  c.dataTypeName = "i8";
  c.productionRate = 256;
  c.minBufferElements = 256;
  c.doubleBuffering = true;
  c.visibility = Visibility::LOCAL_SPM;
  std::vector<ContractSpec> contracts = {c};

  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  BufferAllocator bufAllocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0;
  bufOpts.preferDoubleBuffering = true;
  BufferAllocationPlan bufferPlan =
      bufAllocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  // Verify double-buffering was applied.
  if (bufferPlan.allocations.empty() || !bufferPlan.allocations[0].doubleBuffered) {
    std::cerr << "FAIL: testDMAScheduleDoubleBufferOverlap - buffer not double-buffered\n";
    return false;
  }

  DMAScheduler dmaScheduler;
  DMASchedulerOptions dmaOpts;
  dmaOpts.estimatedComputeCycles = 1000;
  dmaOpts.kernelComputeCycles["A"] = 1000;
  dmaOpts.kernelComputeCycles["B"] = 1000;

  DMASchedule dmaSchedule = dmaScheduler.schedule(
      bufferPlan, nocSchedule, contracts, assignment, arch, dmaOpts);

  if (dmaSchedule.transfers.empty()) {
    std::cerr << "FAIL: testDMAScheduleDoubleBufferOverlap - no transfers\n";
    return false;
  }

  // With double-buffering, the DMA transfer for the second buffer fill
  // should be able to overlap with computation. Since we only have one
  // edge, check that the overlap mechanism is correctly set up.
  const auto &xfer = dmaSchedule.transfers[0];
  if (xfer.durationCycles == 0) {
    std::cerr << "FAIL: testDMAScheduleDoubleBufferOverlap - zero duration\n";
    return false;
  }

  // The transfer uses double-buffer slots.
  // Buffer slot should be 0 or 1.
  if (xfer.bufferSlot > 1) {
    std::cerr << "FAIL: testDMAScheduleDoubleBufferOverlap - invalid buffer slot: "
              << xfer.bufferSlot << "\n";
    return false;
  }

  std::cout << "PASS: testDMAScheduleDoubleBufferOverlap\n";
  return true;
}

// =========================================================================
// T8: End-to-end iteration includes NoC + buffer + DMA
// =========================================================================
static bool testEndToEndNoCBufferDMA() {
  // 2x2 system with diamond dependency: A->B, A->C, B->D, C->D.
  SystemArchitecture arch = build2x2Arch(16384);

  // Assign each kernel to a distinct core.
  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["A"] = 0;
  assignment.kernelToCore["B"] = 1;
  assignment.kernelToCore["C"] = 2;
  assignment.kernelToCore["D"] = 3;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[0].assignedKernels.push_back("A");
  assignment.coreAssignments[1].assignedKernels.push_back("B");
  assignment.coreAssignments[2].assignedKernels.push_back("C");
  assignment.coreAssignments[3].assignedKernels.push_back("D");

  auto makeEdge = [](const std::string &src, const std::string &dst) {
    ContractSpec c;
    c.producerKernel = src;
    c.consumerKernel = dst;
    c.dataTypeName = "f32";
    c.productionRate = 64;
    c.minBufferElements = 64;
    c.visibility = Visibility::LOCAL_SPM;
    return c;
  };

  std::vector<ContractSpec> contracts = {
      makeEdge("A", "B"), makeEdge("A", "C"),
      makeEdge("B", "D"), makeEdge("C", "D")};

  // Run NoC scheduling.
  NoCScheduler nocScheduler;
  NoCSchedulerOptions nocOpts;
  NoCSchedule nocSchedule =
      nocScheduler.schedule(assignment, contracts, arch, nocOpts);

  // All 4 edges are cross-core -> 4 routes.
  if (nocSchedule.routes.size() != 4) {
    std::cerr << "FAIL: testEndToEndNoCBufferDMA - expected 4 NoC routes, got "
              << nocSchedule.routes.size() << "\n";
    return false;
  }

  // Run buffer allocation.
  BufferAllocator bufAllocator;
  BufferAllocatorOptions bufOpts;
  bufOpts.spmReserveFraction = 0.0;
  bufOpts.preferDoubleBuffering = false;
  BufferAllocationPlan bufferPlan =
      bufAllocator.allocate(assignment, contracts, nocSchedule, arch, bufOpts);

  if (!bufferPlan.feasible) {
    std::cerr << "FAIL: testEndToEndNoCBufferDMA - buffer plan not feasible\n";
    return false;
  }

  // 4 cross-core edges -> 4 allocations.
  if (bufferPlan.allocations.size() != 4) {
    std::cerr << "FAIL: testEndToEndNoCBufferDMA - expected 4 allocations, got "
              << bufferPlan.allocations.size() << "\n";
    return false;
  }

  // Run DMA scheduling.
  DMAScheduler dmaScheduler;
  DMASchedulerOptions dmaOpts;
  dmaOpts.estimatedComputeCycles = 500;
  DMASchedule dmaSchedule = dmaScheduler.schedule(
      bufferPlan, nocSchedule, contracts, assignment, arch, dmaOpts);

  // 4 cross-core edges -> 4 DMA transfers.
  if (dmaSchedule.transfers.size() != 4) {
    std::cerr << "FAIL: testEndToEndNoCBufferDMA - expected 4 DMA transfers, got "
              << dmaSchedule.transfers.size() << "\n";
    return false;
  }

  // Verify the objective with buffer costs differs from without.
  std::vector<CoreCostSummary> dummySummaries;
  double objWithoutBuf =
      computeObjective(assignment, nocSchedule, dummySummaries);
  double objWithBuf =
      computeObjective(assignment, nocSchedule, bufferPlan, dummySummaries);

  // If there are any L2/DRAM allocations, the buffer-inclusive objective
  // should be higher. Even if all are SPM, they should be equal (no penalty).
  if (objWithBuf < objWithoutBuf) {
    std::cerr << "FAIL: testEndToEndNoCBufferDMA - buffer objective < base objective\n";
    return false;
  }

  std::cout << "PASS: testEndToEndNoCBufferDMA\n";
  return true;
}

// =========================================================================
// T9: NoC contention detection
// =========================================================================
static bool testNoCContentionDetection() {
  // 2x2 mesh. Create 4 edges that all route through the same link
  // under XY routing. Specifically, route many edges from column 0 to
  // column 1, all passing through link (0,0)->(0,1).
  SystemArchitecture arch = build2x2Arch();
  // Set link bandwidth very low to cause contention.
  arch.nocSpec.linkBandwidth = 1;
  arch.nocSpec.flitWidth = 1; // 1 byte per flit

  // 4 kernels: P0, P1 on core 0 (0,0); C0 on core 1 (0,1); C1 on core 3 (1,1).
  // But we can't put 4 kernels on the same core easily. Instead:
  // Use 4 separate contract edges, all producer on core 0, consumer on core 1.
  // This forces all traffic through link (0,0)->(0,1).
  AssignmentResult assignment;
  assignment.feasible = true;
  assignment.kernelToCore["P0"] = 0;
  assignment.kernelToCore["P1"] = 0;
  assignment.kernelToCore["P2"] = 0;
  assignment.kernelToCore["P3"] = 0;
  assignment.kernelToCore["C0"] = 1;
  assignment.kernelToCore["C1"] = 1;
  assignment.kernelToCore["C2"] = 1;
  assignment.kernelToCore["C3"] = 1;

  unsigned totalCores = arch.totalCoreInstances();
  assignment.coreAssignments.resize(totalCores);
  for (unsigned ci = 0; ci < totalCores; ++ci) {
    assignment.coreAssignments[ci].coreInstanceIdx = ci;
    assignment.coreAssignments[ci].coreTypeName =
        arch.typeNameForInstance(ci);
  }
  assignment.coreAssignments[0].assignedKernels = {"P0", "P1", "P2", "P3"};
  assignment.coreAssignments[1].assignedKernels = {"C0", "C1", "C2", "C3"};

  // 4 edges, each carrying high data volume.
  auto makeHeavyEdge = [](const std::string &prod, const std::string &cons) {
    ContractSpec c;
    c.producerKernel = prod;
    c.consumerKernel = cons;
    c.dataTypeName = "i8";
    c.productionRate = 1024; // 1024 bytes each
    c.minBufferElements = 1024;
    return c;
  };

  std::vector<ContractSpec> contracts = {
      makeHeavyEdge("P0", "C0"), makeHeavyEdge("P1", "C1"),
      makeHeavyEdge("P2", "C2"), makeHeavyEdge("P3", "C3")};

  NoCScheduler scheduler;
  NoCSchedulerOptions opts;
  opts.routing = NoCSchedulerOptions::XY_DOR;
  opts.verbose = false;

  NoCSchedule schedule =
      scheduler.schedule(assignment, contracts, arch, opts);

  // All 4 edges pass through (0,0)->(0,1). With linkBandwidth=1 and
  // 1 byte flits, the demand is 4 * 1024 = 4096 flits on a link with
  // capacity 1 flit/cycle. Utilization should be > 1.0.
  if (!schedule.hasContention) {
    std::cerr << "FAIL: testNoCContentionDetection - expected contention\n";
    return false;
  }

  if (schedule.maxLinkUtilization <= 1.0) {
    std::cerr << "FAIL: testNoCContentionDetection - max utilization "
              << schedule.maxLinkUtilization << " should exceed 1.0\n";
    return false;
  }

  // Find the congested link and verify it's (0,0)->(0,1).
  bool foundCongestedLink = false;
  for (const auto &lu : schedule.linkUtilizations) {
    if (lu.utilization > 1.0) {
      foundCongestedLink = true;
      if (lu.srcNode != std::make_pair(0, 0) ||
          lu.dstNode != std::make_pair(0, 1)) {
        std::cerr << "FAIL: testNoCContentionDetection - unexpected congested link ("
                  << lu.srcNode.first << "," << lu.srcNode.second << ")->("
                  << lu.dstNode.first << "," << lu.dstNode.second << ")\n";
        return false;
      }
    }
  }

  if (!foundCongestedLink) {
    std::cerr << "FAIL: testNoCContentionDetection - no congested link found\n";
    return false;
  }

  std::cout << "PASS: testNoCContentionDetection\n";
  return true;
}

// =========================================================================
// main
// =========================================================================
int main() {
  int failures = 0;

  if (!testNoCScheduleValidRoutes()) ++failures;
  if (!testNoCScheduleColocated()) ++failures;
  if (!testBufferAllocationFitsSPM()) ++failures;
  if (!testBufferAllocationL2Fallback()) ++failures;
  if (!testBufferAllocationDoubleBuffering()) ++failures;
  if (!testDMAScheduleOrdering()) ++failures;
  if (!testDMAScheduleDoubleBufferOverlap()) ++failures;
  if (!testEndToEndNoCBufferDMA()) ++failures;
  if (!testNoCContentionDetection()) ++failures;

  std::cout << "\n" << (9 - failures) << "/9 tests passed";
  if (failures > 0)
    std::cout << " (" << failures << " FAILED)";
  std::cout << "\n";

  return failures > 0 ? 1 : 0;
}
