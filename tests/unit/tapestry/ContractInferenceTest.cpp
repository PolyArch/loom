/// Contract inference engine tests: rate analysis, tile shape inference,
/// buffer size inference, and visibility inference.
///
/// Tests:
/// 1. TileShapeInference: basic shape derivation
/// 2. TileShapeInference: 1D large dimension
/// 3. TileShapeInference: multi-dimensional with budget constraint
/// 4. TileShapeInference: power-of-2 rounding
/// 5. TileShapeInference: edge cases (empty, zero)
/// 6. BufferSizeInference: FIFO ordering
/// 7. BufferSizeInference: UNORDERED ordering
/// 8. BufferSizeInference: min <= max invariant (FIFO with tiny SPM)
/// 9. BufferSizeInference: double buffering heuristic
/// 10. VisibilityInference: small volume -> LOCAL_SPM
/// 11. VisibilityInference: medium volume -> SHARED_L2
/// 12. VisibilityInference: large volume -> EXTERNAL_DRAM
/// 13. VisibilityInference: mayFuse override

#include "loom/ContractInference/BufferSizeInference.h"
#include "loom/ContractInference/TileShapeInference.h"
#include "loom/SystemCompiler/Contract.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace loom;

// Forward declare the visibility function from VisibilityInference.cpp.
namespace loom {
Visibility inferVisibility(int64_t productionRate, uint64_t tileElements,
                           unsigned elementSizeBytes,
                           uint64_t spmBudgetBytes, double spmThresholdFraction,
                           uint64_t l2BudgetBytes, double l2ThresholdFraction,
                           bool mayFuse);
} // namespace loom

//===----------------------------------------------------------------------===//
// TileShapeInference tests
//===----------------------------------------------------------------------===//

/// Test 1: Basic tile shape that already fits SPM.
static bool testTileShapeFitsDirectly() {
  TileShapeInference tsi;
  // 4x4 of f32 = 64 bytes; SPM = 256 bytes -> fits directly.
  auto result = tsi.infer({4, 4}, 4, 256);
  if (result.tileShape.size() != 2 || result.tileShape[0] != 4 ||
      result.tileShape[1] != 4) {
    std::cerr << "FAIL: testTileShapeFitsDirectly - shape\n";
    return false;
  }
  if (result.elementsPerTile != 16) {
    std::cerr << "FAIL: testTileShapeFitsDirectly - elements\n";
    return false;
  }
  if (result.bytesPerTile != 64) {
    std::cerr << "FAIL: testTileShapeFitsDirectly - bytes\n";
    return false;
  }
  if (!result.fitsSPM) {
    std::cerr << "FAIL: testTileShapeFitsDirectly - fitsSPM\n";
    return false;
  }
  std::cerr << "PASS: testTileShapeFitsDirectly\n";
  return true;
}

/// Test 2: 1D large dimension that must be halved.
static bool testTileShape1DLarge() {
  TileShapeInference tsi;
  // 1024 of f32 = 4096 bytes; SPM = 256 bytes.
  // Must halve repeatedly: 1024->512->256->128->64 (64*4=256 fits).
  auto result = tsi.infer({1024}, 4, 256);
  if (result.tileShape.size() != 1) {
    std::cerr << "FAIL: testTileShape1DLarge - dims\n";
    return false;
  }
  if (result.bytesPerTile > 256) {
    std::cerr << "FAIL: testTileShape1DLarge - doesn't fit (bytes="
              << result.bytesPerTile << ")\n";
    return false;
  }
  if (!result.fitsSPM) {
    std::cerr << "FAIL: testTileShape1DLarge - fitsSPM\n";
    return false;
  }
  // Should be power of 2.
  int64_t dim = result.tileShape[0];
  if ((dim & (dim - 1)) != 0 || dim <= 0) {
    std::cerr << "FAIL: testTileShape1DLarge - not power of 2 (dim="
              << dim << ")\n";
    return false;
  }
  std::cerr << "PASS: testTileShape1DLarge\n";
  return true;
}

/// Test 3: Multi-dimensional with tight budget.
static bool testTileShapeMultiDim() {
  TileShapeInference tsi;
  // 64x64x64 of f32 = 1,048,576 bytes; SPM = 4096 bytes.
  auto result = tsi.infer({64, 64, 64}, 4, 4096);
  if (result.bytesPerTile > 4096) {
    std::cerr << "FAIL: testTileShapeMultiDim - doesn't fit (bytes="
              << result.bytesPerTile << ")\n";
    return false;
  }
  if (!result.fitsSPM) {
    std::cerr << "FAIL: testTileShapeMultiDim - fitsSPM\n";
    return false;
  }
  // Each dimension should be power of 2 and >= 1.
  for (size_t i = 0; i < result.tileShape.size(); i++) {
    int64_t dim = result.tileShape[i];
    if (dim < 1 || (dim & (dim - 1)) != 0) {
      std::cerr << "FAIL: testTileShapeMultiDim - dim " << i
                << " not valid pow2 (=" << dim << ")\n";
      return false;
    }
  }
  std::cerr << "PASS: testTileShapeMultiDim\n";
  return true;
}

/// Test 4: roundDownToPow2 correctness.
static bool testRoundDownToPow2() {
  if (TileShapeInference::roundDownToPow2(1) != 1) {
    std::cerr << "FAIL: testRoundDownToPow2 - 1\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(3) != 2) {
    std::cerr << "FAIL: testRoundDownToPow2 - 3\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(4) != 4) {
    std::cerr << "FAIL: testRoundDownToPow2 - 4\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(5) != 4) {
    std::cerr << "FAIL: testRoundDownToPow2 - 5\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(7) != 4) {
    std::cerr << "FAIL: testRoundDownToPow2 - 7\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(8) != 8) {
    std::cerr << "FAIL: testRoundDownToPow2 - 8\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(0) != 1) {
    std::cerr << "FAIL: testRoundDownToPow2 - 0\n";
    return false;
  }
  if (TileShapeInference::roundDownToPow2(1024) != 1024) {
    std::cerr << "FAIL: testRoundDownToPow2 - 1024\n";
    return false;
  }
  std::cerr << "PASS: testRoundDownToPow2\n";
  return true;
}

/// Test 5: Edge cases for tile shape inference.
static bool testTileShapeEdgeCases() {
  TileShapeInference tsi;

  // Empty problem shape.
  auto result1 = tsi.infer({}, 4, 256);
  if (result1.fitsSPM) {
    std::cerr << "FAIL: testTileShapeEdgeCases - empty should not fit\n";
    return false;
  }

  // Zero element size.
  auto result2 = tsi.infer({16, 16}, 0, 256);
  if (result2.fitsSPM) {
    std::cerr << "FAIL: testTileShapeEdgeCases - zero elem size\n";
    return false;
  }

  // Zero SPM budget.
  auto result3 = tsi.infer({16, 16}, 4, 0);
  if (result3.fitsSPM) {
    std::cerr << "FAIL: testTileShapeEdgeCases - zero spm budget\n";
    return false;
  }

  std::cerr << "PASS: testTileShapeEdgeCases\n";
  return true;
}

//===----------------------------------------------------------------------===//
// BufferSizeInference tests
//===----------------------------------------------------------------------===//

/// Test 6: FIFO ordering buffer sizing.
static bool testBufferFIFO() {
  BufferSizeInference bsi;
  ContractSpec spec;
  spec.ordering = Ordering::FIFO;
  spec.consumptionRate = 2;
  spec.productionRate = 4;

  // SPM budget = 1024 bytes, element size = 4, latency = 10.
  auto result = bsi.infer(spec, 1024, 4, 10);

  // min = max(1, 10 / 2) = 5
  if (result.minElements != 5) {
    std::cerr << "FAIL: testBufferFIFO - minElements=" << result.minElements
              << " expected 5\n";
    return false;
  }
  // max = 1024 / 4 = 256
  if (result.maxElements != 256) {
    std::cerr << "FAIL: testBufferFIFO - maxElements=" << result.maxElements
              << " expected 256\n";
    return false;
  }
  // production (4) >= 2 * consumption (2) => double buffering.
  if (!result.requiresDoubleBuffering) {
    std::cerr << "FAIL: testBufferFIFO - should require double buffering\n";
    return false;
  }
  std::cerr << "PASS: testBufferFIFO\n";
  return true;
}

/// Test 7: UNORDERED ordering buffer sizing.
static bool testBufferUnordered() {
  BufferSizeInference bsi;
  ContractSpec spec;
  spec.ordering = Ordering::UNORDERED;
  spec.productionRate = 1;
  spec.consumptionRate = 1;

  auto result = bsi.infer(spec, 512, 4, 5);
  if (result.minElements != 1) {
    std::cerr << "FAIL: testBufferUnordered - minElements="
              << result.minElements << " expected 1\n";
    return false;
  }
  if (result.maxElements != 128) {
    std::cerr << "FAIL: testBufferUnordered - maxElements="
              << result.maxElements << " expected 128\n";
    return false;
  }
  std::cerr << "PASS: testBufferUnordered\n";
  return true;
}

/// Test 8: min <= max invariant with FIFO ordering and tiny SPM.
static bool testBufferMinMaxInvariant() {
  BufferSizeInference bsi;
  ContractSpec spec;
  spec.ordering = Ordering::FIFO;
  spec.productionRate = 1;
  spec.consumptionRate = 1;

  // Very small SPM: 16 bytes / 4 = 4 elements max.
  // High latency so min would want to be large.
  auto result = bsi.infer(spec, 16, 4, 100);
  if (result.minElements > result.maxElements) {
    std::cerr << "FAIL: testBufferMinMaxInvariant - min("
              << result.minElements << ") > max(" << result.maxElements
              << ")\n";
    return false;
  }
  std::cerr << "PASS: testBufferMinMaxInvariant\n";
  return true;
}

/// Test 9: Double buffering heuristic.
static bool testDoubleBufferingHeuristic() {
  BufferSizeInference bsi;

  // Case 1: prod >= 2*cons -> double buffering.
  ContractSpec spec1;
  spec1.ordering = Ordering::FIFO;
  spec1.productionRate = 100;
  spec1.consumptionRate = 10;
  auto r1 = bsi.infer(spec1, 4096, 4, 1);
  if (!r1.requiresDoubleBuffering) {
    std::cerr << "FAIL: testDoubleBufferingHeuristic - should be true\n";
    return false;
  }

  // Case 2: prod < 2*cons -> no double buffering.
  ContractSpec spec2;
  spec2.ordering = Ordering::FIFO;
  spec2.productionRate = 10;
  spec2.consumptionRate = 10;
  auto r2 = bsi.infer(spec2, 4096, 4, 1);
  if (r2.requiresDoubleBuffering) {
    std::cerr << "FAIL: testDoubleBufferingHeuristic - should be false\n";
    return false;
  }

  std::cerr << "PASS: testDoubleBufferingHeuristic\n";
  return true;
}

//===----------------------------------------------------------------------===//
// VisibilityInference tests
//===----------------------------------------------------------------------===//

/// Test 10: Small volume -> LOCAL_SPM.
static bool testVisibilityLocalSPM() {
  // volume = 4 * 16 * 4 = 256 bytes.
  // spm threshold = 4096 * 0.5 = 2048 bytes.
  // 256 <= 2048 => LOCAL_SPM.
  auto vis = inferVisibility(4, 16, 4, 4096, 0.5, 262144, 0.8, false);
  if (vis != Visibility::LOCAL_SPM) {
    std::cerr << "FAIL: testVisibilityLocalSPM\n";
    return false;
  }
  std::cerr << "PASS: testVisibilityLocalSPM\n";
  return true;
}

/// Test 11: Medium volume -> SHARED_L2.
static bool testVisibilitySharedL2() {
  // volume = 100 * 100 * 4 = 40,000 bytes.
  // spm threshold = 4096 * 0.5 = 2048. 40000 > 2048.
  // l2 threshold = 262144 * 0.8 = 209,715. 40000 <= 209715.
  // => SHARED_L2.
  auto vis = inferVisibility(100, 100, 4, 4096, 0.5, 262144, 0.8, false);
  if (vis != Visibility::SHARED_L2) {
    std::cerr << "FAIL: testVisibilitySharedL2\n";
    return false;
  }
  std::cerr << "PASS: testVisibilitySharedL2\n";
  return true;
}

/// Test 12: Large volume -> EXTERNAL_DRAM.
static bool testVisibilityExternalDRAM() {
  // volume = 1000 * 1000 * 4 = 4,000,000 bytes.
  // spm threshold = 4096 * 0.5 = 2048.
  // l2 threshold = 262144 * 0.8 = 209,715.
  // 4,000,000 > 209,715 => EXTERNAL_DRAM.
  auto vis = inferVisibility(1000, 1000, 4, 4096, 0.5, 262144, 0.8, false);
  if (vis != Placement::EXTERNAL) {
    std::cerr << "FAIL: testVisibilityExternalDRAM\n";
    return false;
  }
  std::cerr << "PASS: testVisibilityExternalDRAM\n";
  return true;
}

/// Test 13: mayFuse override -> always LOCAL_SPM regardless of volume.
static bool testVisibilityMayFuseOverride() {
  // Large volume but mayFuse=true should force LOCAL_SPM.
  auto vis = inferVisibility(1000, 1000, 4, 4096, 0.5, 262144, 0.8, true);
  if (vis != Visibility::LOCAL_SPM) {
    std::cerr << "FAIL: testVisibilityMayFuseOverride\n";
    return false;
  }
  std::cerr << "PASS: testVisibilityMayFuseOverride\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main() {
  int passed = 0;
  int total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test())
      passed++;
  };

  // TileShapeInference tests
  run(testTileShapeFitsDirectly);
  run(testTileShape1DLarge);
  run(testTileShapeMultiDim);
  run(testRoundDownToPow2);
  run(testTileShapeEdgeCases);

  // BufferSizeInference tests
  run(testBufferFIFO);
  run(testBufferUnordered);
  run(testBufferMinMaxInvariant);
  run(testDoubleBufferingHeuristic);

  // VisibilityInference tests
  run(testVisibilityLocalSPM);
  run(testVisibilitySharedL2);
  run(testVisibilityExternalDRAM);
  run(testVisibilityMayFuseOverride);

  std::cerr << "\nResults: " << passed << "/" << total << " tests passed\n";
  return (passed == total) ? 0 : 1;
}
