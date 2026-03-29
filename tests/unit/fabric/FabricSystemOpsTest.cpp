/// Fabric system-level op tests: attribute creation, verifier rejection,
/// and round-trip serialization for RouterOp, NoCLinkOp, SharedMemOp.

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace mlir;
namespace fabric = loom::fabric;

static MLIRContext &getContext() {
  static MLIRContext ctx;
  static bool initialized = false;
  if (!initialized) {
    ctx.getOrLoadDialect<fabric::FabricDialect>();
    initialized = true;
  }
  return ctx;
}

/// Parse an MLIR string and return the module (null on failure).
static OwningOpRef<ModuleOp> parseMLIR(const std::string &input) {
  auto &ctx = getContext();
  return parseSourceString<ModuleOp>(input, &ctx);
}

/// Print an MLIR module to a string.
static std::string printMLIR(ModuleOp module) {
  std::string output;
  llvm::raw_string_ostream os(output);
  module.print(os);
  return output;
}

//===----------------------------------------------------------------------===//
// Test 1: RouterOp creation with new attributes
//===----------------------------------------------------------------------===//

static bool testRouterOpCreation() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  auto routerOp = fabric::RouterOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("r0"),
      /*num_ports=*/static_cast<uint64_t>(5),
      /*virtual_channels=*/static_cast<uint64_t>(2),
      /*buffer_depth=*/static_cast<uint64_t>(4),
      /*pipeline_stages=*/static_cast<uint64_t>(2),
      /*flit_width_bits=*/static_cast<uint64_t>(32),
      /*routing_strategy=*/llvm::StringRef("xy_dor"),
      /*topology_role=*/llvm::StringRef("mesh"));

  if (!routerOp) {
    std::cerr << "FAIL: testRouterOpCreation - create returned null\n";
    return false;
  }

  if (routerOp.getRoutingStrategy() != "xy_dor") {
    std::cerr << "FAIL: testRouterOpCreation - routing_strategy mismatch\n";
    return false;
  }

  if (routerOp.getTopologyRole() != "mesh") {
    std::cerr << "FAIL: testRouterOpCreation - topology_role mismatch\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testRouterOpCreation - verify failed\n";
    return false;
  }

  std::cerr << "PASS: testRouterOpCreation\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 2: RouterOp invalid routing_strategy
//===----------------------------------------------------------------------===//

static bool testRouterOpInvalidRoutingStrategy() {
  std::string input = R"mlir(
    module {
      fabric.router @r0 {
        num_ports = 5 : i64,
        virtual_channels = 2 : i64,
        buffer_depth = 4 : i64,
        pipeline_stages = 2 : i64,
        flit_width_bits = 32 : i64,
        routing_strategy = "unknown_strategy",
        topology_role = "mesh"
      }
    }
  )mlir";

  auto &ctx = getContext();
  std::string diagMsg;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagMsg += diag.str();
    return success();
  });

  auto module = parseMLIR(input);
  if (!module) {
    if (diagMsg.find("routing_strategy") != std::string::npos) {
      std::cerr << "PASS: testRouterOpInvalidRoutingStrategy (rejected at parse)\n";
      return true;
    }
    std::cerr << "FAIL: testRouterOpInvalidRoutingStrategy - parse failed for wrong reason\n";
    return false;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testRouterOpInvalidRoutingStrategy - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testRouterOpInvalidRoutingStrategy\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 3: RouterOp invalid topology_role
//===----------------------------------------------------------------------===//

static bool testRouterOpInvalidTopologyRole() {
  std::string input = R"mlir(
    module {
      fabric.router @r0 {
        num_ports = 5 : i64,
        virtual_channels = 2 : i64,
        buffer_depth = 4 : i64,
        pipeline_stages = 2 : i64,
        flit_width_bits = 32 : i64,
        routing_strategy = "xy_dor",
        topology_role = "torus"
      }
    }
  )mlir";

  auto &ctx = getContext();
  std::string diagMsg;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagMsg += diag.str();
    return success();
  });

  auto module = parseMLIR(input);
  if (!module) {
    if (diagMsg.find("topology_role") != std::string::npos) {
      std::cerr << "PASS: testRouterOpInvalidTopologyRole (rejected at parse)\n";
      return true;
    }
    std::cerr << "FAIL: testRouterOpInvalidTopologyRole - parse failed for wrong reason\n";
    return false;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testRouterOpInvalidTopologyRole - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testRouterOpInvalidTopologyRole\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 4: NoCLinkOp bandwidth attribute
//===----------------------------------------------------------------------===//

static bool testNoCLinkOpBandwidthAttribute() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  auto linkOp = fabric::NoCLinkOp::create(
      builder, loc,
      /*source=*/llvm::StringRef("r0"),
      /*source_port=*/static_cast<uint64_t>(0),
      /*dest=*/llvm::StringRef("r1"),
      /*dest_port=*/static_cast<uint64_t>(1),
      /*width_bits=*/static_cast<uint64_t>(32),
      /*latency_cycles=*/static_cast<uint64_t>(2),
      /*bandwidth=*/static_cast<uint64_t>(4));

  if (!linkOp) {
    std::cerr << "FAIL: testNoCLinkOpBandwidthAttribute - create returned null\n";
    return false;
  }

  if (linkOp.getBandwidth() != 4) {
    std::cerr << "FAIL: testNoCLinkOpBandwidthAttribute - bandwidth mismatch, got "
              << linkOp.getBandwidth() << "\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testNoCLinkOpBandwidthAttribute - verify failed\n";
    return false;
  }

  std::cerr << "PASS: testNoCLinkOpBandwidthAttribute\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 5: NoCLinkOp invalid bandwidth
//===----------------------------------------------------------------------===//

static bool testNoCLinkOpInvalidBandwidth() {
  std::string input = R"mlir(
    module {
      fabric.noc_link {
        source = @r0,
        source_port = 0 : i64,
        dest = @r1,
        dest_port = 1 : i64,
        width_bits = 32 : i64,
        latency_cycles = 2 : i64,
        bandwidth = 0 : i64
      }
    }
  )mlir";

  auto &ctx = getContext();
  std::string diagMsg;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagMsg += diag.str();
    return success();
  });

  auto module = parseMLIR(input);
  if (!module) {
    if (diagMsg.find("bandwidth") != std::string::npos) {
      std::cerr << "PASS: testNoCLinkOpInvalidBandwidth (rejected at parse)\n";
      return true;
    }
    std::cerr << "FAIL: testNoCLinkOpInvalidBandwidth - parse failed for wrong reason\n";
    return false;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testNoCLinkOpInvalidBandwidth - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testNoCLinkOpInvalidBandwidth\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 6: SharedMemOp port_count attribute
//===----------------------------------------------------------------------===//

static bool testSharedMemOpPortCount() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  auto memOp = fabric::SharedMemOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("bank0"),
      /*size_bytes=*/static_cast<uint64_t>(65536),
      /*width_bytes=*/static_cast<uint64_t>(32),
      /*num_banks=*/static_cast<uint64_t>(1),
      /*mem_type=*/llvm::StringRef("l2_cache"),
      /*port_count=*/static_cast<uint64_t>(2));

  if (!memOp) {
    std::cerr << "FAIL: testSharedMemOpPortCount - create returned null\n";
    return false;
  }

  if (memOp.getPortCount() != 2) {
    std::cerr << "FAIL: testSharedMemOpPortCount - port_count mismatch, got "
              << memOp.getPortCount() << "\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testSharedMemOpPortCount - verify failed\n";
    return false;
  }

  std::cerr << "PASS: testSharedMemOpPortCount\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 7: SharedMemOp invalid port_count
//===----------------------------------------------------------------------===//

static bool testSharedMemOpInvalidPortCount() {
  std::string input = R"mlir(
    module {
      fabric.shared_mem @bank0 {
        size_bytes = 65536 : i64,
        width_bytes = 32 : i64,
        num_banks = 1 : i64,
        mem_type = "l2_cache",
        port_count = 0 : i64
      }
    }
  )mlir";

  auto &ctx = getContext();
  std::string diagMsg;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    diagMsg += diag.str();
    return success();
  });

  auto module = parseMLIR(input);
  if (!module) {
    if (diagMsg.find("port_count") != std::string::npos) {
      std::cerr << "PASS: testSharedMemOpInvalidPortCount (rejected at parse)\n";
      return true;
    }
    std::cerr << "FAIL: testSharedMemOpInvalidPortCount - parse failed for wrong reason\n";
    return false;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testSharedMemOpInvalidPortCount - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testSharedMemOpInvalidPortCount\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 8: RouterOp round-trip
//===----------------------------------------------------------------------===//

static bool testRouterRoundTrip() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  fabric::RouterOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("r_rt"),
      /*num_ports=*/static_cast<uint64_t>(5),
      /*virtual_channels=*/static_cast<uint64_t>(2),
      /*buffer_depth=*/static_cast<uint64_t>(4),
      /*pipeline_stages=*/static_cast<uint64_t>(2),
      /*flit_width_bits=*/static_cast<uint64_t>(32),
      /*routing_strategy=*/llvm::StringRef("adaptive"),
      /*topology_role=*/llvm::StringRef("hierarchical"));

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testRouterRoundTrip - initial verify failed\n";
    return false;
  }

  // Print to string
  std::string printed = printMLIR(*module);

  // Re-parse
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testRouterRoundTrip - re-parse failed\n";
    std::cerr << "  Printed IR:\n" << printed << "\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testRouterRoundTrip - round-trip verify failed\n";
    return false;
  }

  // Check attributes on re-parsed op
  fabric::RouterOp reparsedRouter = nullptr;
  module2->walk([&](fabric::RouterOp op) {
    reparsedRouter = op;
  });

  if (!reparsedRouter) {
    std::cerr << "FAIL: testRouterRoundTrip - router not found after re-parse\n";
    return false;
  }

  if (reparsedRouter.getRoutingStrategy() != "adaptive") {
    std::cerr << "FAIL: testRouterRoundTrip - routing_strategy mismatch after round-trip\n";
    return false;
  }

  if (reparsedRouter.getTopologyRole() != "hierarchical") {
    std::cerr << "FAIL: testRouterRoundTrip - topology_role mismatch after round-trip\n";
    return false;
  }

  std::cerr << "PASS: testRouterRoundTrip\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 9: NoCLinkOp round-trip
//===----------------------------------------------------------------------===//

static bool testNoCLinkRoundTrip() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  fabric::NoCLinkOp::create(
      builder, loc,
      /*source=*/llvm::StringRef("r0"),
      /*source_port=*/static_cast<uint64_t>(0),
      /*dest=*/llvm::StringRef("r1"),
      /*dest_port=*/static_cast<uint64_t>(1),
      /*width_bits=*/static_cast<uint64_t>(32),
      /*latency_cycles=*/static_cast<uint64_t>(2),
      /*bandwidth=*/static_cast<uint64_t>(8));

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testNoCLinkRoundTrip - initial verify failed\n";
    return false;
  }

  std::string printed = printMLIR(*module);
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testNoCLinkRoundTrip - re-parse failed\n";
    std::cerr << "  Printed IR:\n" << printed << "\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testNoCLinkRoundTrip - round-trip verify failed\n";
    return false;
  }

  fabric::NoCLinkOp reparsedLink = nullptr;
  module2->walk([&](fabric::NoCLinkOp op) {
    reparsedLink = op;
  });

  if (!reparsedLink) {
    std::cerr << "FAIL: testNoCLinkRoundTrip - link not found after re-parse\n";
    return false;
  }

  if (reparsedLink.getBandwidth() != 8) {
    std::cerr << "FAIL: testNoCLinkRoundTrip - bandwidth mismatch, got "
              << reparsedLink.getBandwidth() << "\n";
    return false;
  }

  std::cerr << "PASS: testNoCLinkRoundTrip\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 10: SharedMemOp round-trip
//===----------------------------------------------------------------------===//

static bool testSharedMemRoundTrip() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  fabric::SharedMemOp::create(
      builder, loc,
      /*sym_name=*/llvm::StringRef("mem_rt"),
      /*size_bytes=*/static_cast<uint64_t>(131072),
      /*width_bytes=*/static_cast<uint64_t>(64),
      /*num_banks=*/static_cast<uint64_t>(2),
      /*mem_type=*/llvm::StringRef("l2_cache"),
      /*port_count=*/static_cast<uint64_t>(4));

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testSharedMemRoundTrip - initial verify failed\n";
    return false;
  }

  std::string printed = printMLIR(*module);
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testSharedMemRoundTrip - re-parse failed\n";
    std::cerr << "  Printed IR:\n" << printed << "\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testSharedMemRoundTrip - round-trip verify failed\n";
    return false;
  }

  fabric::SharedMemOp reparsedMem = nullptr;
  module2->walk([&](fabric::SharedMemOp op) {
    reparsedMem = op;
  });

  if (!reparsedMem) {
    std::cerr << "FAIL: testSharedMemRoundTrip - shared_mem not found after re-parse\n";
    return false;
  }

  if (reparsedMem.getPortCount() != 4) {
    std::cerr << "FAIL: testSharedMemRoundTrip - port_count mismatch, got "
              << reparsedMem.getPortCount() << "\n";
    return false;
  }

  std::cerr << "PASS: testSharedMemRoundTrip\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Test 11: All three ops in a fabric.module system
//===----------------------------------------------------------------------===//

static bool testAllThreeOpsInSystemModule() {
  // Build inline MLIR text with all three ops inside a fabric.module
  std::string input = R"mlir(
    module {
      fabric.router @r0 {
        num_ports = 5 : i64,
        virtual_channels = 2 : i64,
        buffer_depth = 4 : i64,
        pipeline_stages = 2 : i64,
        flit_width_bits = 32 : i64,
        routing_strategy = "xy_dor",
        topology_role = "mesh"
      }
      fabric.router @r1 {
        num_ports = 5 : i64,
        virtual_channels = 2 : i64,
        buffer_depth = 4 : i64,
        pipeline_stages = 2 : i64,
        flit_width_bits = 32 : i64,
        routing_strategy = "xy_dor",
        topology_role = "mesh"
      }
      fabric.noc_link {
        source = @r0,
        source_port = 0 : i64,
        dest = @r1,
        dest_port = 1 : i64,
        width_bits = 32 : i64,
        latency_cycles = 2 : i64,
        bandwidth = 4 : i64
      }
      fabric.shared_mem @bank0 {
        size_bytes = 65536 : i64,
        width_bytes = 32 : i64,
        num_banks = 1 : i64,
        mem_type = "l2_cache",
        port_count = 2 : i64
      }
    }
  )mlir";

  auto module = parseMLIR(input);
  if (!module) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - parse failed\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - verify failed\n";
    return false;
  }

  // Round-trip
  std::string printed = printMLIR(*module);
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - round-trip parse failed\n";
    std::cerr << "  Printed IR:\n" << printed << "\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - round-trip verify failed\n";
    return false;
  }

  // Verify attributes survived round-trip
  fabric::RouterOp routerOp = nullptr;
  fabric::NoCLinkOp linkOp = nullptr;
  fabric::SharedMemOp memOp = nullptr;

  module2->walk([&](fabric::RouterOp op) {
    if (op.getSymName() == "r0")
      routerOp = op;
  });
  module2->walk([&](fabric::NoCLinkOp op) {
    linkOp = op;
  });
  module2->walk([&](fabric::SharedMemOp op) {
    memOp = op;
  });

  if (!routerOp || !linkOp || !memOp) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - ops not found after round-trip\n";
    return false;
  }

  if (routerOp.getRoutingStrategy() != "xy_dor") {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - router routing_strategy mismatch\n";
    return false;
  }

  if (routerOp.getTopologyRole() != "mesh") {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - router topology_role mismatch\n";
    return false;
  }

  if (linkOp.getBandwidth() != 4) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - link bandwidth mismatch\n";
    return false;
  }

  if (memOp.getPortCount() != 2) {
    std::cerr << "FAIL: testAllThreeOpsInSystemModule - mem port_count mismatch\n";
    return false;
  }

  std::cerr << "PASS: testAllThreeOpsInSystemModule\n";
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

  run(testRouterOpCreation);
  run(testRouterOpInvalidRoutingStrategy);
  run(testRouterOpInvalidTopologyRole);
  run(testNoCLinkOpBandwidthAttribute);
  run(testNoCLinkOpInvalidBandwidth);
  run(testSharedMemOpPortCount);
  run(testSharedMemOpInvalidPortCount);
  run(testRouterRoundTrip);
  run(testNoCLinkRoundTrip);
  run(testSharedMemRoundTrip);
  run(testAllThreeOpsInSystemModule);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
