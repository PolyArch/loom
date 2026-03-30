/// BufferAllocator constraint-driven placement tests.
/// Verifies that MemoryConstraints from a ConstraintSet override the
/// default SPM-first allocation policy in BufferAllocator::allocate().
///
/// Tests:
///  1. LOCAL_SPM constraint forces SPM placement
///  2. SHARED_L2 constraint skips SPM, places in L2
///  3. EXTERNAL constraint goes directly to DRAM
///  4. No constraint falls back to default SPM-first policy
///  5. Mixed constraints: per-edge independent placement

#include "loom/SystemCompiler/BufferAllocator.h"
#include "loom/SystemCompiler/ContractConstraintTranslator.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace loom;

/// Build a minimal 2-core architecture with SPM and shared L2.
static SystemArchitecture makeTestArch(uint64_t spmPerCore = 4096,
                                       uint64_t l2Size = 65536) {
  SystemArchitecture arch;

  CoreTypeSpec ct;
  ct.typeName = "test_core";
  ct.instanceCount = 2;
  ct.spmBytes = spmPerCore;
  ct.numPEs = 16;
  ct.numFUs = 4;
  arch.coreTypes.push_back(ct);

  arch.sharedMemSpec.l2SizeBytes = l2Size;
  arch.sharedMemSpec.numBanks = 4;

  arch.nocSpec.meshRows = 1;
  arch.nocSpec.meshCols = 2;

  return arch;
}

/// Build a cross-core assignment with two kernels on different cores.
static AssignmentResult makeTestAssignment() {
  AssignmentResult assignment;
  assignment.kernelToCore["producer_k"] = 0;
  assignment.kernelToCore["consumer_k"] = 1;
  return assignment;
}

/// Build a single cross-core contract.
static std::vector<ContractSpec> makeTestContracts() {
  ContractSpec c;
  c.producerKernel = "producer_k";
  c.consumerKernel = "consumer_k";
  c.dataTypeName = "f32";
  c.minBufferElements = 64;
  c.visibility = Visibility::LOCAL_SPM;
  c.doubleBuffering = false;
  return {c};
}

/// Build a minimal NoCSchedule.
static NoCSchedule makeTestNoCSchedule() {
  NoCSchedule sched;
  sched.hasContention = false;
  return sched;
}

/// T1: LOCAL_SPM constraint forces SPM placement.
static bool testConstraintLocalSPM() {
  auto arch = makeTestArch();
  auto assignment = makeTestAssignment();
  auto contracts = makeTestContracts();
  auto nocSched = makeTestNoCSchedule();
  BufferAllocatorOptions opts;
  opts.spmReserveFraction = 0.0; // use all SPM

  ConstraintSet cs;
  MemoryConstraint mc;
  mc.edgeProducer = "producer_k";
  mc.edgeConsumer = "consumer_k";
  mc.level = MemoryLevel::LOCAL_SPM;
  cs.memory.push_back(mc);

  BufferAllocator allocator;
  auto plan = allocator.allocate(assignment, contracts, nocSched, arch, opts, cs);

  if (plan.allocations.size() != 1) {
    std::cerr << "FAIL: testConstraintLocalSPM - expected 1 allocation, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  auto loc = plan.allocations[0].location;
  if (loc != BufferAllocation::SPM_CONSUMER &&
      loc != BufferAllocation::SPM_PRODUCER) {
    std::cerr << "FAIL: testConstraintLocalSPM - expected SPM placement, got "
              << static_cast<int>(loc) << "\n";
    return false;
  }

  std::cerr << "PASS: testConstraintLocalSPM\n";
  return true;
}

/// T2: SHARED_L2 constraint skips SPM, places in L2.
static bool testConstraintSharedL2() {
  auto arch = makeTestArch();
  auto assignment = makeTestAssignment();
  auto contracts = makeTestContracts();
  auto nocSched = makeTestNoCSchedule();
  BufferAllocatorOptions opts;
  opts.spmReserveFraction = 0.0;

  ConstraintSet cs;
  MemoryConstraint mc;
  mc.edgeProducer = "producer_k";
  mc.edgeConsumer = "consumer_k";
  mc.level = MemoryLevel::SHARED_L2;
  cs.memory.push_back(mc);

  BufferAllocator allocator;
  auto plan = allocator.allocate(assignment, contracts, nocSched, arch, opts, cs);

  if (plan.allocations.size() != 1) {
    std::cerr << "FAIL: testConstraintSharedL2 - expected 1 allocation, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  if (plan.allocations[0].location != BufferAllocation::SHARED_L2) {
    std::cerr << "FAIL: testConstraintSharedL2 - expected SHARED_L2, got "
              << static_cast<int>(plan.allocations[0].location) << "\n";
    return false;
  }

  std::cerr << "PASS: testConstraintSharedL2\n";
  return true;
}

/// T3: EXTERNAL constraint goes directly to DRAM.
static bool testConstraintExternal() {
  auto arch = makeTestArch();
  auto assignment = makeTestAssignment();
  auto contracts = makeTestContracts();
  auto nocSched = makeTestNoCSchedule();
  BufferAllocatorOptions opts;
  opts.spmReserveFraction = 0.0;

  ConstraintSet cs;
  MemoryConstraint mc;
  mc.edgeProducer = "producer_k";
  mc.edgeConsumer = "consumer_k";
  mc.level = MemoryLevel::EXTERNAL;
  cs.memory.push_back(mc);

  BufferAllocator allocator;
  auto plan = allocator.allocate(assignment, contracts, nocSched, arch, opts, cs);

  if (plan.allocations.size() != 1) {
    std::cerr << "FAIL: testConstraintExternal - expected 1 allocation, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  if (plan.allocations[0].location != BufferAllocation::EXTERNAL_DRAM) {
    std::cerr << "FAIL: testConstraintExternal - expected EXTERNAL_DRAM, got "
              << static_cast<int>(plan.allocations[0].location) << "\n";
    return false;
  }

  std::cerr << "PASS: testConstraintExternal\n";
  return true;
}

/// T4: No constraint falls back to default SPM-first policy.
static bool testNoConstraintDefaultPolicy() {
  auto arch = makeTestArch();
  auto assignment = makeTestAssignment();
  auto contracts = makeTestContracts();
  auto nocSched = makeTestNoCSchedule();
  BufferAllocatorOptions opts;
  opts.spmReserveFraction = 0.0;

  BufferAllocator allocator;
  auto plan = allocator.allocate(assignment, contracts, nocSched, arch, opts);

  if (plan.allocations.size() != 1) {
    std::cerr << "FAIL: testNoConstraintDefaultPolicy - expected 1 allocation\n";
    return false;
  }

  // Default policy is SPM-first (consumer side).
  auto loc = plan.allocations[0].location;
  if (loc != BufferAllocation::SPM_CONSUMER) {
    std::cerr << "FAIL: testNoConstraintDefaultPolicy - expected SPM_CONSUMER, "
              << "got " << static_cast<int>(loc) << "\n";
    return false;
  }

  std::cerr << "PASS: testNoConstraintDefaultPolicy\n";
  return true;
}

/// T5: Mixed constraints: two edges with independent placement requirements.
static bool testMixedConstraints() {
  SystemArchitecture arch;
  CoreTypeSpec ct;
  ct.typeName = "test_core";
  ct.instanceCount = 3;
  ct.spmBytes = 4096;
  ct.numPEs = 16;
  ct.numFUs = 4;
  arch.coreTypes.push_back(ct);
  arch.sharedMemSpec.l2SizeBytes = 65536;
  arch.sharedMemSpec.numBanks = 4;
  arch.nocSpec.meshRows = 1;
  arch.nocSpec.meshCols = 3;

  AssignmentResult assignment;
  assignment.kernelToCore["k_a"] = 0;
  assignment.kernelToCore["k_b"] = 1;
  assignment.kernelToCore["k_c"] = 2;

  ContractSpec c1;
  c1.producerKernel = "k_a";
  c1.consumerKernel = "k_b";
  c1.dataTypeName = "f32";
  c1.minBufferElements = 16;
  c1.visibility = Visibility::LOCAL_SPM;
  c1.doubleBuffering = false;

  ContractSpec c2;
  c2.producerKernel = "k_b";
  c2.consumerKernel = "k_c";
  c2.dataTypeName = "f32";
  c2.minBufferElements = 16;
  c2.visibility = Visibility::LOCAL_SPM;
  c2.doubleBuffering = false;

  ConstraintSet cs;
  // First edge forced to L2, second forced to DRAM.
  MemoryConstraint mc1;
  mc1.edgeProducer = "k_a";
  mc1.edgeConsumer = "k_b";
  mc1.level = MemoryLevel::SHARED_L2;
  cs.memory.push_back(mc1);

  MemoryConstraint mc2;
  mc2.edgeProducer = "k_b";
  mc2.edgeConsumer = "k_c";
  mc2.level = MemoryLevel::EXTERNAL;
  cs.memory.push_back(mc2);

  BufferAllocatorOptions opts;
  opts.spmReserveFraction = 0.0;
  auto nocSched = makeTestNoCSchedule();

  BufferAllocator allocator;
  auto plan = allocator.allocate(assignment, {c1, c2}, nocSched, arch, opts, cs);

  if (plan.allocations.size() != 2) {
    std::cerr << "FAIL: testMixedConstraints - expected 2 allocations, got "
              << plan.allocations.size() << "\n";
    return false;
  }

  // Find each edge's allocation.
  const BufferAllocation *allocA = nullptr;
  const BufferAllocation *allocB = nullptr;
  for (const auto &a : plan.allocations) {
    if (a.contractEdgeName.find("k_a") != std::string::npos)
      allocA = &a;
    else if (a.contractEdgeName.find("k_b") != std::string::npos)
      allocB = &a;
  }

  if (!allocA || !allocB) {
    std::cerr << "FAIL: testMixedConstraints - could not find both allocations\n";
    return false;
  }

  if (allocA->location != BufferAllocation::SHARED_L2) {
    std::cerr << "FAIL: testMixedConstraints - edge A should be SHARED_L2\n";
    return false;
  }

  if (allocB->location != BufferAllocation::EXTERNAL_DRAM) {
    std::cerr << "FAIL: testMixedConstraints - edge B should be EXTERNAL_DRAM\n";
    return false;
  }

  std::cerr << "PASS: testMixedConstraints\n";
  return true;
}

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  run(testConstraintLocalSPM);
  run(testConstraintSharedL2);
  run(testConstraintExternal);
  run(testNoConstraintDefaultPolicy);
  run(testMixedConstraints);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
