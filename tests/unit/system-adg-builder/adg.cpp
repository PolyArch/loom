//===-- adg.cpp - System ADG builder test -------------------------*- C++ -*-//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests the SystemADGBuilder with a 2x2 mesh of heterogeneous cores.
// Also produces a simple per-core ADG at the output path for the unit test
// framework (so the mapper has something to work with).
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilder.h"
#include "loom/ADG/SystemADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace loom::adg;

/// Build a core type A: a compute core with add/mul FUs and NoC ports.
static std::string buildCoreTypeA() {
  ADGBuilder builder("CoreType_A");
  constexpr unsigned dataWidth = 32;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto fuMul = builder.defineBinaryFU("fu_mul", "arith.muli", "i32", "i32");

  auto pe =
      builder.defineSpatialPE("compute_pe", 2, 1, dataWidth, {fuAdd, fuMul});
  auto sw = builder.defineFullCrossbarSpatialSW("sw", 4, 4, dataWidth);
  builder.buildMesh(2, 2, pe, sw);

  // Add NoC ports for each direction
  builder.addNoCIngressPort("noc_in_N", dataWidth);
  builder.addNoCIngressPort("noc_in_S", dataWidth);
  builder.addNoCIngressPort("noc_in_E", dataWidth);
  builder.addNoCIngressPort("noc_in_W", dataWidth);

  builder.addNoCEgressPort("noc_out_N", dataWidth);
  builder.addNoCEgressPort("noc_out_S", dataWidth);
  builder.addNoCEgressPort("noc_out_E", dataWidth);
  builder.addNoCEgressPort("noc_out_W", dataWidth);

  builder.setSPMCapacity(16384); // 16KB

  return builder.exportCoreType("CoreType_A");
}

/// Build a core type B: a memory-focused core with NoC ports.
static std::string buildCoreTypeB() {
  ADGBuilder builder("CoreType_B");
  constexpr unsigned dataWidth = 32;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe = builder.defineSingleFUSpatialPE("mem_pe", 2, 1, dataWidth, fuAdd);
  auto sw = builder.defineFullCrossbarSpatialSW("sw", 3, 3, dataWidth);
  builder.buildMesh(1, 2, pe, sw);

  builder.addNoCIngressPort("noc_in_N", dataWidth);
  builder.addNoCIngressPort("noc_in_S", dataWidth);
  builder.addNoCIngressPort("noc_in_E", dataWidth);
  builder.addNoCIngressPort("noc_in_W", dataWidth);

  builder.addNoCEgressPort("noc_out_N", dataWidth);
  builder.addNoCEgressPort("noc_out_S", dataWidth);
  builder.addNoCEgressPort("noc_out_E", dataWidth);
  builder.addNoCEgressPort("noc_out_W", dataWidth);

  builder.setSPMCapacity(32768); // 32KB

  return builder.exportCoreType("CoreType_B");
}

/// Run all SystemADGBuilder tests, writing output to outputDir.
static void runSystemTests(const std::string &outputDir) {
  std::string coreAMLIR = buildCoreTypeA();
  std::string coreBMLIR = buildCoreTypeB();

  // Verify core type export produced non-empty MLIR
  assert(!coreAMLIR.empty() && "CoreType_A export should not be empty");
  assert(!coreBMLIR.empty() && "CoreType_B export should not be empty");

  // Verify NoC port metadata is present in exported core type
  assert(coreAMLIR.find("noc_port") != std::string::npos &&
         "CoreType_A should contain noc_port metadata");
  assert(coreAMLIR.find("spm_capacity_bytes") != std::string::npos &&
         "CoreType_A should contain spm_capacity_bytes metadata");

  // Test 1: Build a 2x2 mesh system with 2 core types
  {
    SystemADGBuilder sysBuilder("test_system_mesh");

    auto typeA = sysBuilder.registerCoreType("CoreType_A", coreAMLIR);
    auto typeB = sysBuilder.registerCoreType("CoreType_B", coreBMLIR);

    sysBuilder.instantiateCore(typeA, "core_0_0", 0, 0);
    sysBuilder.instantiateCore(typeB, "core_0_1", 0, 1);
    sysBuilder.instantiateCore(typeA, "core_1_0", 1, 0);
    sysBuilder.instantiateCore(typeB, "core_1_1", 1, 1);

    NoCSpec noc;
    noc.topology = NoCSpec::MESH;
    noc.flitWidth = 32;
    noc.virtualChannels = 2;
    noc.linkBandwidth = 1;
    noc.routerPipelineStages = 2;
    sysBuilder.setNoCSpec(noc);

    SharedMemorySpec mem;
    mem.l2SizeBytes = 262144;
    mem.numBanks = 4;
    mem.bankWidthBytes = 32;
    sysBuilder.setSharedMemorySpec(mem);

    sysBuilder.build();

    // Verify instance count
    const auto &instances = sysBuilder.getCoreInstances();
    assert(instances.size() == 4 && "expected 4 core instances");

    // Verify specs are stored correctly
    assert(sysBuilder.getNoCSpec().topology == NoCSpec::MESH);
    assert(sysBuilder.getNoCSpec().flitWidth == 32);
    assert(sysBuilder.getSharedMemorySpec().numBanks == 4);
    assert(sysBuilder.getSharedMemorySpec().l2SizeBytes == 262144);

    // Verify generated MLIR contains typed ops (not comments)
    std::string mlir = sysBuilder.getSystemMLIR();

    // System module name should be present
    assert(mlir.find("test_system_mesh") != std::string::npos);

    // Core instance names should appear as real ops, not comments
    assert(mlir.find("core_0_0") != std::string::npos);
    assert(mlir.find("core_0_1") != std::string::npos);
    assert(mlir.find("core_1_0") != std::string::npos);
    assert(mlir.find("core_1_1") != std::string::npos);

    // Core type references should be present
    assert(mlir.find("CoreType_A") != std::string::npos);
    assert(mlir.find("CoreType_B") != std::string::npos);

    // Typed ops should be present (not comment-based)
    assert(mlir.find("fabric.router") != std::string::npos &&
           "should contain fabric.router ops");
    assert(mlir.find("fabric.shared_mem") != std::string::npos &&
           "should contain fabric.shared_mem ops");
    assert(mlir.find("fabric.noc_link") != std::string::npos &&
           "should contain fabric.noc_link ops");
    assert(mlir.find("fabric.instance") != std::string::npos &&
           "should contain fabric.instance ops");

    // Verify router attributes are present
    assert(mlir.find("num_ports") != std::string::npos);
    assert(mlir.find("virtual_channels") != std::string::npos);
    assert(mlir.find("flit_width_bits") != std::string::npos);

    // Verify shared memory attributes
    assert(mlir.find("l2_bank_") != std::string::npos);
    assert(mlir.find("size_bytes") != std::string::npos);
    assert(mlir.find("l2_cache") != std::string::npos);
    assert(mlir.find("external_dram") != std::string::npos);

    // Verify NoC link attributes
    assert(mlir.find("source") != std::string::npos);
    assert(mlir.find("dest") != std::string::npos);
    assert(mlir.find("width_bits") != std::string::npos);

    // Verify no comment-based instances remain
    assert(mlir.find("// fabric.instance") == std::string::npos &&
           "should not have comment-based instances");

    sysBuilder.exportSystemMLIR(outputDir + "/system_mesh.mlir");
    llvm::outs() << "PASS: 2x2 mesh system with 2 core types\n";
  }

  // Test 2: Ring topology
  {
    SystemADGBuilder ringBuilder("test_system_ring");
    auto rtA = ringBuilder.registerCoreType("CoreType_A", coreAMLIR);
    ringBuilder.instantiateCore(rtA, "ring_0", 0, 0);
    ringBuilder.instantiateCore(rtA, "ring_1", 0, 1);
    ringBuilder.instantiateCore(rtA, "ring_2", 0, 2);

    NoCSpec ringNoc;
    ringNoc.topology = NoCSpec::RING;
    ringBuilder.setNoCSpec(ringNoc);

    SharedMemorySpec mem;
    mem.numBanks = 2;
    ringBuilder.setSharedMemorySpec(mem);

    ringBuilder.build();

    std::string mlir = ringBuilder.getSystemMLIR();
    // Verify typed ops for ring topology
    assert(mlir.find("fabric.router") != std::string::npos);
    assert(mlir.find("fabric.noc_link") != std::string::npos);
    assert(mlir.find("fabric.shared_mem") != std::string::npos);

    ringBuilder.exportSystemMLIR(outputDir + "/system_ring.mlir");
    llvm::outs() << "PASS: ring topology\n";
  }

  // Test 3: Hierarchical topology
  {
    SystemADGBuilder hierBuilder("test_system_hier");
    auto htA = hierBuilder.registerCoreType("CoreType_A", coreAMLIR);
    for (int i = 0; i < 8; ++i) {
      hierBuilder.instantiateCore(htA, "hier_" + std::to_string(i), i / 4,
                                  i % 4);
    }

    NoCSpec hierNoc;
    hierNoc.topology = NoCSpec::HIERARCHICAL;
    hierBuilder.setNoCSpec(hierNoc);

    SharedMemorySpec mem;
    mem.numBanks = 4;
    hierBuilder.setSharedMemorySpec(mem);

    hierBuilder.build();

    std::string mlir = hierBuilder.getSystemMLIR();
    // Verify typed ops for hierarchical topology
    assert(mlir.find("fabric.router") != std::string::npos);
    assert(mlir.find("fabric.noc_link") != std::string::npos);
    assert(mlir.find("fabric.shared_mem") != std::string::npos);

    hierBuilder.exportSystemMLIR(outputDir + "/system_hier.mlir");
    llvm::outs() << "PASS: hierarchical topology\n";
  }

  llvm::outs() << "All system ADG builder tests passed.\n";
}

/// Build a simple per-core ADG for the mapper test framework.
/// Uses a single PE with a switch bank domain for a clean fabric.
static void buildSimpleADGForMapper(const std::string &outputPath) {
  ADGBuilder builder("system_adg_builder_test_adg");
  constexpr unsigned dataWidth = 32;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");
  auto pe =
      builder.defineSingleFUSpatialPE("add_pe", 2, 1, dataWidth, fuAdd);

  // Build a switch bank domain with one PE and scalar I/O.
  // SW needs: 1 PE output + 2 scalar inputs = 3 inputs
  //           2 PE inputs + 1 scalar output = 3 outputs
  SwitchBankDomainSpec spec;
  spec.sw = builder.defineFullCrossbarSpatialSW("sw", 3, 3, dataWidth);
  spec.pe = pe;
  spec.numPEs = 1;
  spec.peInputCount = 2;
  spec.peOutputCount = 1;
  spec.scalarInputTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  spec.scalarOutputTypes = {"!fabric.bits<32>"};

  builder.buildSwitchBankDomain(spec);
  builder.exportMLIR(outputPath);
}

static llvm::cl::opt<std::string> outputFile(
    llvm::cl::Positional, llvm::cl::desc("<output .fabric.mlir file>"),
    llvm::cl::init("system-adg-builder.fabric.mlir"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "System ADG builder test\n");

  auto parentPath = llvm::sys::path::parent_path(outputFile);
  if (!parentPath.empty()) {
    if (auto ec = llvm::sys::fs::create_directories(parentPath)) {
      llvm::errs() << "error: cannot create output directory: " << parentPath
                   << "\n";
      return 1;
    }
  }

  std::string outputDir = parentPath.str();
  if (outputDir.empty())
    outputDir = ".";

  // Run system-level tests (these use assertions)
  runSystemTests(outputDir);

  // Build simple ADG for the mapper framework
  buildSimpleADGForMapper(outputFile);

  return 0;
}
