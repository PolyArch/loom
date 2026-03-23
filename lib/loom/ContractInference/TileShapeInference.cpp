#include "loom/ContractInference/TileShapeInference.h"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace loom;

int64_t TileShapeInference::roundDownToPow2(int64_t val) {
  if (val <= 1)
    return 1;

  // Find the highest set bit position and return 2^pos.
  int64_t result = 1;
  while (result * 2 <= val)
    result *= 2;
  return result;
}

TileShapeResult
TileShapeInference::infer(const std::vector<int64_t> &problemShape,
                          unsigned elementSizeBytes, uint64_t spmBudgetBytes) {
  TileShapeResult result;

  if (problemShape.empty() || elementSizeBytes == 0 || spmBudgetBytes == 0) {
    result.fitsSPM = false;
    return result;
  }

  // Start with full problem shape.
  result.tileShape = problemShape;

  // Ensure all dimensions are at least 1.
  for (auto &dim : result.tileShape) {
    if (dim <= 0)
      dim = 1;
  }

  // Iteratively halve the largest dimension until the tile fits.
  auto computeBytes = [&]() -> uint64_t {
    uint64_t elements = 1;
    for (int64_t dim : result.tileShape)
      elements *= static_cast<uint64_t>(dim);
    return elements * elementSizeBytes;
  };

  // Safety limit to prevent infinite loops on degenerate inputs.
  int maxIterations = 64;
  while (computeBytes() > spmBudgetBytes && maxIterations-- > 0) {
    // Find the largest dimension.
    auto maxIt =
        std::max_element(result.tileShape.begin(), result.tileShape.end());
    if (*maxIt <= 1)
      break; // Cannot halve further.
    *maxIt = (*maxIt + 1) / 2;
  }

  // Round each dimension down to the nearest power of 2.
  for (auto &dim : result.tileShape) {
    dim = roundDownToPow2(dim);
  }

  // Compute final metrics.
  result.elementsPerTile = 1;
  for (int64_t dim : result.tileShape)
    result.elementsPerTile *= static_cast<uint64_t>(dim);

  result.bytesPerTile = result.elementsPerTile * elementSizeBytes;
  result.fitsSPM = (result.bytesPerTile <= spmBudgetBytes);

  return result;
}
