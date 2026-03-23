#ifndef LOOM_CONTRACTINFERENCE_TILESHAPEINFERENCE_H
#define LOOM_CONTRACTINFERENCE_TILESHAPEINFERENCE_H

#include <cstdint>
#include <vector>

namespace loom {

/// Result of tile shape inference for a given problem shape and SPM budget.
struct TileShapeResult {
  /// Inferred tile dimensions (one per problem dimension).
  std::vector<int64_t> tileShape;

  /// Total number of elements per tile.
  uint64_t elementsPerTile = 0;

  /// Total bytes per tile (elementsPerTile * elementSizeBytes).
  uint64_t bytesPerTile = 0;

  /// Whether the tile fits in the SPM budget.
  bool fitsSPM = false;
};

/// Derives tile shapes from problem dimensions and scratchpad memory budget.
/// Iteratively halves the largest dimension until the tile fits, then rounds
/// to powers of 2 for efficient DMA transfer.
class TileShapeInference {
public:
  /// Infer tile shape from problem dimensions and SPM budget.
  ///
  /// Algorithm:
  ///   1. Start with full problem shape
  ///   2. While total bytes > spmBudgetBytes: halve the largest dimension
  ///   3. Round each dimension down to the nearest power of 2
  ///   4. Ensure all dimensions are at least 1
  TileShapeResult infer(const std::vector<int64_t> &problemShape,
                        unsigned elementSizeBytes,
                        uint64_t spmBudgetBytes);

  /// Round a value down to the nearest power of 2. Returns 1 for values <= 1.
  static int64_t roundDownToPow2(int64_t val);
};

} // namespace loom

#endif
