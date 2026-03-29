//===-- ADGPartitioner.h - Spatial partitioning of CGRA fabrics -----*- C++ -*-===//
//
// Partitions a core's PE grid into disjoint spatial regions for concurrent
// kernel execution in SPATIAL_SHARING mode. Supports 2-way and 4-way splits
// via row-wise or column-wise strategies.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_SYSTEMCOMPILER_ADGPARTITIONER_H
#define LOOM_SYSTEMCOMPILER_ADGPARTITIONER_H

#include "loom/SystemCompiler/L1CoreAssignment.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

/// Describes a rectangular sub-region of the PE grid.
struct PartitionSpec {
  unsigned rowStart = 0;
  unsigned rowEnd = 0;   // exclusive
  unsigned colStart = 0;
  unsigned colEnd = 0;   // exclusive

  /// Estimated resources within this partition.
  unsigned numPEs = 0;
  unsigned numFUs = 0;
  uint64_t spmBytes = 0;

  /// Number of rows and columns in this partition.
  unsigned numRows() const { return rowEnd - rowStart; }
  unsigned numCols() const { return colEnd - colStart; }
};

/// A partition plan for one physical core, describing how the PE grid
/// is split among concurrent kernels.
struct PartitionPlan {
  std::string coreTypeName;
  unsigned totalRows = 0;
  unsigned totalCols = 0;
  unsigned totalPEs = 0;

  std::vector<PartitionSpec> partitions;
};

/// Validation result for a partition plan.
struct PartitionValidation {
  bool valid = false;
  std::string errorMessage;
};

/// Partitions the PE grid of a CGRA core for spatial sharing.
class ADGPartitioner {
public:
  /// Generate a partition plan that splits the core's PE grid into
  /// \p numPartitions disjoint regions.
  ///
  /// Strategy:
  ///   - Prefer row-wise splits (e.g., 8x8 -> two 4x8 partitions).
  ///   - If row-wise yields partitions with fewer than 2 rows, use col-wise.
  ///   - Supports 2-way and 4-way splits. Returns an error for other values.
  ///
  /// \param coreType       Core type specification with PE grid dimensions.
  /// \param numPartitions  Number of partitions to create (2 or 4).
  /// \param gridRows       Number of rows in the PE grid.
  /// \param gridCols       Number of columns in the PE grid.
  /// \returns A PartitionPlan describing the spatial split.
  static PartitionPlan
  generatePartitions(const CoreTypeSpec &coreType, unsigned numPartitions,
                     unsigned gridRows, unsigned gridCols);

  /// Validate that a partition plan is non-overlapping and covers the
  /// entire physical grid with no gaps.
  static PartitionValidation validatePartition(const PartitionPlan &plan);

  /// Merge per-partition configuration blobs into a single unified config
  /// for the physical core. Since partitions are disjoint, the merge is a
  /// union of per-partition configuration bits at their original addresses.
  ///
  /// \param partitionConfigs  Per-partition config blobs (parallel to plan.partitions).
  /// \param plan              The partition plan describing region layout.
  /// \param fullConfigSize    Total config size in bytes for the physical core.
  /// \returns Merged configuration blob.
  static std::vector<uint8_t>
  mergeConfigurations(const std::vector<std::vector<uint8_t>> &partitionConfigs,
                      const PartitionPlan &plan,
                      size_t fullConfigSize);
};

} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_ADGPARTITIONER_H
