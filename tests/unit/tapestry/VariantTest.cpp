/// Kernel variant system unit tests.
///
/// Tests:
/// T1. addVariant API -- basic registration
/// T2. addVariant API -- duplicate name rejection
/// T3. addVariant API -- invalid kernel handle
/// T4. Variant enumeration -- multiple kernels with edges
/// T5. TDG MLIR round-trip -- variant attributes emitted
/// T6. Variant registration via TaskGraph + emitTDG round-trip
/// T7. VariantOptions equality operators
/// T8. Variant info preserved in dump
/// T9. Multiple kernels with different variant counts -- end-to-end MLIR

#include "tapestry/task_graph.h"
#include "tapestry/tdg_emitter.h"

#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace tapestry;

// Dummy kernel functions (never executed, only referenced by pointer).
static void matmul(float *, float *, float *, int) {}
static void fir(float *, float *, int) {}
static void stencil(float *, float *, int) {}

// -------------------------------------------------------------------------
// T1: addVariant API -- basic registration
// -------------------------------------------------------------------------
static bool testAddVariantBasic() {
  TaskGraph tdg("variant_basic");

  auto k = tdg.kernel("matmul", matmul);

  // Should start with 1 default variant.
  if (tdg.variants(k).size() != 1) {
    std::cerr << "FAIL: testAddVariantBasic - initial variants().size()="
              << tdg.variants(k).size() << " (expected 1)\n";
    return false;
  }

  // Add a second variant.
  tdg.addVariant(k, "matmul_u2",
                 VariantOptions{/*unrollFactor=*/2, /*domainRank=*/0});

  if (tdg.variants(k).size() != 2) {
    std::cerr << "FAIL: testAddVariantBasic - variants().size()="
              << tdg.variants(k).size() << " (expected 2)\n";
    return false;
  }

  // Check variants returns both.
  const auto &variants = tdg.variants(k);
  if (variants.size() != 2) {
    std::cerr << "FAIL: testAddVariantBasic - variants.size()="
              << variants.size() << " (expected 2)\n";
    return false;
  }

  // Default variant should be first.
  if (variants[0].variantName != "matmul_default") {
    std::cerr << "FAIL: testAddVariantBasic - default name='"
              << variants[0].variantName << "'\n";
    return false;
  }
  if (variants[0].options.unrollFactor != 1 ||
      variants[0].options.domainRank != 0) {
    std::cerr << "FAIL: testAddVariantBasic - default options\n";
    return false;
  }

  // Second variant.
  if (variants[1].variantName != "matmul_u2") {
    std::cerr << "FAIL: testAddVariantBasic - variant name='"
              << variants[1].variantName << "'\n";
    return false;
  }
  if (variants[1].options.unrollFactor != 2 ||
      variants[1].options.domainRank != 0) {
    std::cerr << "FAIL: testAddVariantBasic - variant options\n";
    return false;
  }

  std::cout << "PASS: testAddVariantBasic\n";
  return true;
}

// -------------------------------------------------------------------------
// T2: addVariant API -- duplicate name rejection
// -------------------------------------------------------------------------
static bool testAddVariantDuplicate() {
  TaskGraph tdg("variant_duplicate");

  auto k = tdg.kernel("fir", fir);

  // Add first variant.
  tdg.addVariant(k, "fir_v1", VariantOptions{2, 0});

  if (tdg.variants(k).size() != 2) {
    std::cerr << "FAIL: testAddVariantDuplicate - first addVariant failed\n";
    return false;
  }

  // Add duplicate name -> should be rejected (count stays at 2).
  tdg.addVariant(k, "fir_v1", VariantOptions{4, 0});

  // Variant count should still be 2 (default + fir_v1).
  if (tdg.variants(k).size() != 2) {
    std::cerr << "FAIL: testAddVariantDuplicate - variants().size()="
              << tdg.variants(k).size() << " (expected 2)\n";
    return false;
  }

  std::cout << "PASS: testAddVariantDuplicate\n";
  return true;
}

// -------------------------------------------------------------------------
// T3: addVariant API -- invalid kernel handle
// -------------------------------------------------------------------------
static bool testAddVariantInvalidHandle() {
  TaskGraph tdg("variant_invalid");

  // Default-constructed handle is invalid (graph_ == nullptr).
  KernelHandle invalid;
  tdg.addVariant(invalid, "v1", VariantOptions{1, 0});

  // variants() on invalid handle returns empty.
  if (tdg.variants(invalid).size() != 0) {
    std::cerr << "FAIL: testAddVariantInvalidHandle - variants().size() != 0\n";
    return false;
  }

  const auto &variants = tdg.variants(invalid);
  if (!variants.empty()) {
    std::cerr << "FAIL: testAddVariantInvalidHandle - variants not empty\n";
    return false;
  }

  std::cout << "PASS: testAddVariantInvalidHandle\n";
  return true;
}

// -------------------------------------------------------------------------
// T4: Variant enumeration -- multiple kernels with edges
// -------------------------------------------------------------------------
static bool testVariantMultipleKernels() {
  TaskGraph tdg("variant_multi");

  auto kA = tdg.kernel("A", matmul);
  auto kB = tdg.kernel("B", fir);

  // Add 1 extra variant to A (total: default + unroll2 = 2).
  tdg.addVariant(kA, "A_u2", VariantOptions{2, 0});

  // Add 2 extra variants to B (total: default + unroll2 + domain1 = 3).
  tdg.addVariant(kB, "B_u2", VariantOptions{2, 0});
  tdg.addVariant(kB, "B_d1", VariantOptions{1, 1});

  // Connect A -> B.
  tdg.connect(kA, kB).data_type<float>();

  if (tdg.variants(kA).size() != 2) {
    std::cerr << "FAIL: testVariantMultipleKernels - variants(A).size()="
              << tdg.variants(kA).size() << " (expected 2)\n";
    return false;
  }
  if (tdg.variants(kB).size() != 3) {
    std::cerr << "FAIL: testVariantMultipleKernels - variants(B).size()="
              << tdg.variants(kB).size() << " (expected 3)\n";
    return false;
  }

  // Edge should still exist.
  if (tdg.numEdges() != 1) {
    std::cerr << "FAIL: testVariantMultipleKernels - numEdges="
              << tdg.numEdges() << " (expected 1)\n";
    return false;
  }

  // Edge data should be intact.
  auto e = tdg.edge("A", "B");
  if (e.producerName() != "A" || e.consumerName() != "B") {
    std::cerr << "FAIL: testVariantMultipleKernels - edge endpoints\n";
    return false;
  }

  std::cout << "PASS: testVariantMultipleKernels\n";
  return true;
}

// -------------------------------------------------------------------------
// T5: TDG MLIR round-trip -- variant attributes emitted
// -------------------------------------------------------------------------
static bool testVariantMLIREmission() {
  TaskGraph tdg("variant_emit");

  auto k = tdg.kernel("stencil", stencil);
  k.target(ExecutionTarget::CGRA);
  tdg.addVariant(k, "stencil_u2", VariantOptions{2, 0});

  mlir::MLIRContext ctx;
  auto module = emitTDG(tdg, ctx);
  if (!module) {
    std::cerr << "FAIL: testVariantMLIREmission - null module\n";
    return false;
  }

  // Find the kernel op and check its variants attribute.
  bool foundKernel = false;
  unsigned variantAttrCount = 0;

  module->walk([&](loom::tdg::KernelOp kernelOp) {
    if (kernelOp.getSymName() != "stencil")
      return;
    foundKernel = true;

    auto variantsAttr = kernelOp->getAttrOfType<mlir::ArrayAttr>("variants");
    if (!variantsAttr) {
      std::cerr << "FAIL: testVariantMLIREmission - no variants attr\n";
      return;
    }

    variantAttrCount = variantsAttr.size();
    if (variantAttrCount != 2) {
      std::cerr << "FAIL: testVariantMLIREmission - variant count="
                << variantAttrCount << " (expected 2)\n";
      return;
    }

    // Check first variant (default).
    auto dict0 = mlir::dyn_cast<mlir::DictionaryAttr>(variantsAttr[0]);
    if (!dict0) {
      std::cerr << "FAIL: testVariantMLIREmission - dict0 not a dict\n";
      return;
    }
    auto name0 = dict0.getAs<mlir::StringAttr>("name");
    auto uf0 = dict0.getAs<mlir::IntegerAttr>("unroll_factor");
    auto dr0 = dict0.getAs<mlir::IntegerAttr>("domain_rank");
    if (!name0 || name0.getValue() != "stencil_default") {
      std::cerr << "FAIL: testVariantMLIREmission - default variant name\n";
      return;
    }
    if (!uf0 || uf0.getInt() != 1) {
      std::cerr << "FAIL: testVariantMLIREmission - default unroll_factor\n";
      return;
    }
    if (!dr0 || dr0.getInt() != 0) {
      std::cerr << "FAIL: testVariantMLIREmission - default domain_rank\n";
      return;
    }

    // Check second variant.
    auto dict1 = mlir::dyn_cast<mlir::DictionaryAttr>(variantsAttr[1]);
    if (!dict1) {
      std::cerr << "FAIL: testVariantMLIREmission - dict1 not a dict\n";
      return;
    }
    auto name1 = dict1.getAs<mlir::StringAttr>("name");
    auto uf1 = dict1.getAs<mlir::IntegerAttr>("unroll_factor");
    auto dr1 = dict1.getAs<mlir::IntegerAttr>("domain_rank");
    if (!name1 || name1.getValue() != "stencil_u2") {
      std::cerr << "FAIL: testVariantMLIREmission - u2 variant name\n";
      return;
    }
    if (!uf1 || uf1.getInt() != 2) {
      std::cerr << "FAIL: testVariantMLIREmission - u2 unroll_factor\n";
      return;
    }
    if (!dr1 || dr1.getInt() != 0) {
      std::cerr << "FAIL: testVariantMLIREmission - u2 domain_rank\n";
      return;
    }
  });

  if (!foundKernel) {
    std::cerr << "FAIL: testVariantMLIREmission - kernel op not found\n";
    return false;
  }
  if (variantAttrCount != 2)
    return false;

  std::cout << "PASS: testVariantMLIREmission\n";
  return true;
}

// -------------------------------------------------------------------------
// T6: Variant registration via TaskGraph + emitTDG round-trip
// -------------------------------------------------------------------------
static bool testTDGBuilderVariants() {
  TaskGraph tdg("builder_variant_test");

  auto k1 = tdg.kernel("conv", matmul);
  k1.target(ExecutionTarget::CGRA);

  // Default: should have 1 variant.
  if (tdg.variants(k1).size() != 1) {
    std::cerr << "FAIL: testTDGBuilderVariants - initial count="
              << tdg.variants(k1).size() << "\n";
    return false;
  }

  // Register a variant.
  tdg.addVariant(k1, "conv_u4", VariantOptions{4, 0});

  if (tdg.variants(k1).size() != 2) {
    std::cerr << "FAIL: testTDGBuilderVariants - count after register="
              << tdg.variants(k1).size() << "\n";
    return false;
  }

  // Duplicate should be rejected (count stays 2).
  tdg.addVariant(k1, "conv_u4", VariantOptions{8, 0});
  if (tdg.variants(k1).size() != 2) {
    std::cerr << "FAIL: testTDGBuilderVariants - duplicate accepted\n";
    return false;
  }

  // Build MLIR and check variant attributes.
  mlir::MLIRContext ctx;
  auto module = emitTDG(tdg, ctx);
  if (!module) {
    std::cerr << "FAIL: testTDGBuilderVariants - null module\n";
    return false;
  }

  bool foundVariants = false;
  module->walk([&](loom::tdg::KernelOp kernelOp) {
    if (kernelOp.getSymName() != "conv")
      return;
    auto variantsAttr = kernelOp->getAttrOfType<mlir::ArrayAttr>("variants");
    if (variantsAttr && variantsAttr.size() == 2)
      foundVariants = true;
  });

  if (!foundVariants) {
    std::cerr << "FAIL: testTDGBuilderVariants - variant attrs not found in "
                 "MLIR\n";
    return false;
  }

  std::cout << "PASS: testTDGBuilderVariants\n";
  return true;
}

// -------------------------------------------------------------------------
// T7: VariantOptions equality operators
// -------------------------------------------------------------------------
static bool testVariantOptionsEquality() {
  VariantOptions a{1, 0};
  VariantOptions b{1, 0};
  VariantOptions c{2, 0};
  VariantOptions d{1, 1};

  if (!(a == b)) {
    std::cerr << "FAIL: testVariantOptionsEquality - a == b\n";
    return false;
  }
  if (a != b) {
    std::cerr << "FAIL: testVariantOptionsEquality - a != b\n";
    return false;
  }
  if (a == c) {
    std::cerr << "FAIL: testVariantOptionsEquality - a == c (diff unroll)\n";
    return false;
  }
  if (a == d) {
    std::cerr << "FAIL: testVariantOptionsEquality - a == d (diff domain)\n";
    return false;
  }

  std::cout << "PASS: testVariantOptionsEquality\n";
  return true;
}

// -------------------------------------------------------------------------
// T8: Variant info preserved in dump
// -------------------------------------------------------------------------
static bool testVariantDump() {
  TaskGraph tdg("variant_dump_test");

  auto k = tdg.kernel("fft", fir);
  tdg.addVariant(k, "fft_u2", VariantOptions{2, 0});
  tdg.addVariant(k, "fft_d1", VariantOptions{1, 1});

  // Just make sure dump() does not crash with variants present.
  tdg.dump();

  // Verify variant info through variants().
  const auto &variants = tdg.variants(k);
  if (variants.size() != 3) {
    std::cerr << "FAIL: testVariantDump - variant count=" << variants.size()
              << " (expected 3)\n";
    return false;
  }

  std::cout << "PASS: testVariantDump\n";
  return true;
}

// -------------------------------------------------------------------------
// T9: Multiple kernels with different variant counts -- end-to-end MLIR
// -------------------------------------------------------------------------
static bool testMultiKernelVariantMLIR() {
  TaskGraph tdg("multi_variant_mlir");

  auto kA = tdg.kernel("A", matmul);
  auto kB = tdg.kernel("B", fir);
  auto kC = tdg.kernel("C", stencil);

  // A: 1 extra variant (total 2).
  tdg.addVariant(kA, "A_u2", VariantOptions{2, 0});

  // B: 2 extra variants (total 3).
  tdg.addVariant(kB, "B_u2", VariantOptions{2, 0});
  tdg.addVariant(kB, "B_d1", VariantOptions{1, 1});

  // C: no extra variants (total 1, just default).

  tdg.connect(kA, kB);
  tdg.connect(kB, kC);

  mlir::MLIRContext ctx;
  auto module = emitTDG(tdg, ctx);
  if (!module) {
    std::cerr << "FAIL: testMultiKernelVariantMLIR - null module\n";
    return false;
  }

  // Walk kernel ops and check variant counts for base kernels only.
  bool allCorrect = true;
  unsigned baseKernelCount = 0;

  module->walk([&](loom::tdg::KernelOp kernelOp) {
    llvm::StringRef name = kernelOp.getSymName();

    // Only check base kernels (A, B, C), skip variant kernel nodes.
    unsigned expected = 0;
    if (name == "A")
      expected = 2;
    else if (name == "B")
      expected = 3;
    else if (name == "C")
      expected = 1;
    else
      return; // Skip variant kernel nodes.

    baseKernelCount++;
    auto variantsAttr = kernelOp->getAttrOfType<mlir::ArrayAttr>("variants");
    if (!variantsAttr) {
      std::cerr << "FAIL: testMultiKernelVariantMLIR - kernel '"
                << name.str() << "' has no variants attr\n";
      allCorrect = false;
      return;
    }

    if (variantsAttr.size() != expected) {
      std::cerr << "FAIL: testMultiKernelVariantMLIR - kernel '"
                << name.str() << "' has " << variantsAttr.size()
                << " variants (expected " << expected << ")\n";
      allCorrect = false;
    }
  });

  if (baseKernelCount != 3) {
    std::cerr << "FAIL: testMultiKernelVariantMLIR - base kernel count="
              << baseKernelCount << " (expected 3)\n";
    return false;
  }
  if (!allCorrect)
    return false;

  std::cout << "PASS: testMultiKernelVariantMLIR\n";
  return true;
}

// =========================================================================
// Main
// =========================================================================
int main() {
  int passed = 0, failed = 0;

  auto run = [&](bool (*test)(), const char *name) {
    if (test()) {
      ++passed;
    } else {
      std::cerr << "  FAILED: " << name << "\n";
      ++failed;
    }
  };

  run(testAddVariantBasic, "testAddVariantBasic");
  run(testAddVariantDuplicate, "testAddVariantDuplicate");
  run(testAddVariantInvalidHandle, "testAddVariantInvalidHandle");
  run(testVariantMultipleKernels, "testVariantMultipleKernels");
  run(testVariantMLIREmission, "testVariantMLIREmission");
  run(testTDGBuilderVariants, "testTDGBuilderVariants");
  run(testVariantOptionsEquality, "testVariantOptionsEquality");
  run(testVariantDump, "testVariantDump");
  run(testMultiKernelVariantMLIR, "testMultiKernelVariantMLIR");

  std::cout << "\n" << passed << " passed, " << failed << " failed\n";
  return failed > 0 ? 1 : 0;
}
