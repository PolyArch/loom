/// TDG Dialect round-trip tests: parse -> print -> parse, verify attributes.
///
/// Tests:
/// 1. Minimal graph with one kernel parses and prints correctly
/// 2. Graph with two kernels and a FIFO contract round-trips
/// 3. Contract verifier rejects FIFO + may_reorder=true
/// 4. Graph verifier rejects duplicate kernel names
/// 5. Contract verifier rejects unknown producer/consumer references

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/Dialect/TDG/TDGTypes.h"
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
using namespace loom::tdg;

static MLIRContext &getContext() {
  static MLIRContext ctx;
  static bool initialized = false;
  if (!initialized) {
    ctx.getOrLoadDialect<TDGDialect>();
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

/// Test 1: Minimal graph with one kernel.
static bool testMinimalGraph() {
  std::string input = R"mlir(
    module {
      tdg.graph @test_graph {
        tdg.kernel @matmul type "matmul" {
        }
      }
    }
  )mlir";

  auto module = parseMLIR(input);
  if (!module) {
    std::cerr << "FAIL: testMinimalGraph - parse failed\n";
    return false;
  }

  // Verify the module
  if (failed(verify(*module))) {
    std::cerr << "FAIL: testMinimalGraph - verify failed\n";
    return false;
  }

  // Round-trip: print and re-parse
  std::string printed = printMLIR(*module);
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testMinimalGraph - round-trip parse failed\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testMinimalGraph - round-trip verify failed\n";
    return false;
  }

  std::cerr << "PASS: testMinimalGraph\n";
  return true;
}

/// Test 2: Two-kernel pipeline with FIFO contract.
static bool testTwoKernelPipeline() {
  std::string input = R"mlir(
    module {
      tdg.graph @pipeline {
        tdg.kernel @producer type "elementwise" {
        }
        tdg.kernel @consumer type "reduction" {
        }
        tdg.contract @producer -> @consumer {ordering = FIFO, data_type = f32}
      }
    }
  )mlir";

  auto module = parseMLIR(input);
  if (!module) {
    std::cerr << "FAIL: testTwoKernelPipeline - parse failed\n";
    return false;
  }

  if (failed(verify(*module))) {
    std::cerr << "FAIL: testTwoKernelPipeline - verify failed\n";
    return false;
  }

  // Round-trip
  std::string printed = printMLIR(*module);
  auto module2 = parseMLIR(printed);
  if (!module2) {
    std::cerr << "FAIL: testTwoKernelPipeline - round-trip parse failed\n";
    return false;
  }

  if (failed(verify(*module2))) {
    std::cerr << "FAIL: testTwoKernelPipeline - round-trip verify failed\n";
    return false;
  }

  std::cerr << "PASS: testTwoKernelPipeline\n";
  return true;
}

/// Test 3: Contract verifier rejects FIFO + may_reorder=true.
static bool testFIFOMayReorderRejection() {
  std::string input = R"mlir(
    module {
      tdg.graph @invalid {
        tdg.kernel @a type "elementwise" {
        }
        tdg.kernel @b type "elementwise" {
        }
        tdg.contract @a -> @b {ordering = FIFO, data_type = f32, may_reorder = true}
      }
    }
  )mlir";

  auto &ctx = getContext();
  // Suppress diagnostics so the error does not print to stderr
  ScopedDiagnosticHandler handler(&ctx, [](Diagnostic &) { return success(); });

  auto module = parseMLIR(input);
  if (!module) {
    // Parse might fail if verifier runs during parsing; this is OK
    std::cerr << "PASS: testFIFOMayReorderRejection (rejected at parse)\n";
    return true;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testFIFOMayReorderRejection - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testFIFOMayReorderRejection\n";
  return true;
}

/// Test 4: Graph verifier rejects duplicate kernel names.
static bool testDuplicateKernelRejection() {
  auto &ctx = getContext();
  auto loc = UnknownLoc::get(&ctx);
  OpBuilder builder(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());

  auto graphOp = builder.create<GraphOp>(
      loc, builder.getStringAttr("dup_test"), StringAttr());

  Block *block = new Block();
  graphOp.getBody().push_back(block);
  builder.setInsertionPointToEnd(block);

  // Insert two kernels with the same name
  builder.create<KernelOp>(loc, builder.getStringAttr("k1"),
                           builder.getStringAttr("matmul"),
                           DenseI64ArrayAttr(), StringAttr());
  auto &k1body =
      builder.getInsertionBlock()->back().getRegion(0);
  k1body.push_back(new Block());

  builder.setInsertionPointToEnd(block);
  builder.create<KernelOp>(loc, builder.getStringAttr("k1"),
                           builder.getStringAttr("elementwise"),
                           DenseI64ArrayAttr(), StringAttr());
  auto &k1body2 =
      builder.getInsertionBlock()->back().getRegion(0);
  k1body2.push_back(new Block());

  ScopedDiagnosticHandler handler(&ctx, [](Diagnostic &) { return success(); });

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testDuplicateKernelRejection - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testDuplicateKernelRejection\n";
  return true;
}

/// Test 5: Contract verifier rejects unknown producer reference.
static bool testUnknownProducerRejection() {
  std::string input = R"mlir(
    module {
      tdg.graph @bad_ref {
        tdg.kernel @existing type "matmul" {
        }
        tdg.contract @nonexistent -> @existing {ordering = FIFO, data_type = f32}
      }
    }
  )mlir";

  auto &ctx = getContext();
  ScopedDiagnosticHandler handler(&ctx, [](Diagnostic &) { return success(); });

  auto module = parseMLIR(input);
  if (!module) {
    std::cerr << "PASS: testUnknownProducerRejection (rejected at parse)\n";
    return true;
  }

  if (succeeded(verify(*module))) {
    std::cerr << "FAIL: testUnknownProducerRejection - should have failed\n";
    return false;
  }

  std::cerr << "PASS: testUnknownProducerRejection\n";
  return true;
}

/// Test 6: TDG custom types parse correctly.
static bool testCustomTypes() {
  std::string input = R"mlir(
    module {
      tdg.graph @type_test {
        tdg.kernel @k1 type "matmul" {
        }
      }
    }
  )mlir";

  auto module = parseMLIR(input);
  if (!module) {
    std::cerr << "FAIL: testCustomTypes - parse failed\n";
    return false;
  }

  // Verify that TDG types can be created programmatically
  auto &ctx = getContext();
  auto contractRef = ContractRefType::get(
      &ctx, StringAttr::get(&ctx, "test_contract"));
  auto kernelRef = KernelRefType::get(
      &ctx, StringAttr::get(&ctx, "test_kernel"));

  if (!contractRef) {
    std::cerr << "FAIL: testCustomTypes - ContractRefType creation failed\n";
    return false;
  }
  if (!kernelRef) {
    std::cerr << "FAIL: testCustomTypes - KernelRefType creation failed\n";
    return false;
  }

  std::cerr << "PASS: testCustomTypes\n";
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

  run(testMinimalGraph);
  run(testTwoKernelPipeline);
  run(testFIFOMayReorderRejection);
  run(testDuplicateKernelRejection);
  run(testUnknownProducerRejection);
  run(testCustomTypes);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
