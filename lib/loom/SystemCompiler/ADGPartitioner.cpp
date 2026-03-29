//===-- ADGPartitioner.cpp - Spatial partitioning implementation -----------===//
//
// Implements spatial partitioning of CGRA PE grids for SPATIAL_SHARING mode.
// Supports 2-way and 4-way splits via row-wise or column-wise strategies.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/ADGPartitioner.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

namespace loom {

//===----------------------------------------------------------------------===//
// Partition Generation
//===----------------------------------------------------------------------===//

PartitionPlan
ADGPartitioner::generatePartitions(const CoreTypeSpec &coreType,
                                   unsigned numPartitions,
                                   unsigned gridRows,
                                   unsigned gridCols) {
  PartitionPlan plan;
  plan.coreTypeName = coreType.typeName;
  plan.totalRows = gridRows;
  plan.totalCols = gridCols;
  plan.totalPEs = coreType.numPEs;

  // Only support 2-way and 4-way splits.
  if (numPartitions != 2 && numPartitions != 4) {
    // Return a single partition covering the entire grid as fallback.
    PartitionSpec full;
    full.rowStart = 0;
    full.rowEnd = gridRows;
    full.colStart = 0;
    full.colEnd = gridCols;
    full.numPEs = coreType.numPEs;
    full.numFUs = coreType.numFUs;
    full.spmBytes = coreType.spmBytes;
    plan.partitions.push_back(full);
    return plan;
  }

  if (numPartitions == 2) {
    // Prefer row-wise split if each partition gets >= 2 rows.
    if (gridRows >= 4) {
      unsigned midRow = gridRows / 2;
      // Partition 0: rows [0, midRow)
      PartitionSpec p0;
      p0.rowStart = 0;
      p0.rowEnd = midRow;
      p0.colStart = 0;
      p0.colEnd = gridCols;
      p0.numPEs = midRow * gridCols;
      p0.numFUs =
          static_cast<unsigned>(static_cast<double>(coreType.numFUs) *
                                midRow / gridRows);
      p0.spmBytes =
          static_cast<uint64_t>(static_cast<double>(coreType.spmBytes) *
                                midRow / gridRows);
      plan.partitions.push_back(p0);

      // Partition 1: rows [midRow, gridRows)
      PartitionSpec p1;
      p1.rowStart = midRow;
      p1.rowEnd = gridRows;
      p1.colStart = 0;
      p1.colEnd = gridCols;
      p1.numPEs = (gridRows - midRow) * gridCols;
      p1.numFUs = coreType.numFUs - p0.numFUs;
      p1.spmBytes = coreType.spmBytes - p0.spmBytes;
      plan.partitions.push_back(p1);
    } else if (gridCols >= 4) {
      // Column-wise split.
      unsigned midCol = gridCols / 2;

      PartitionSpec p0;
      p0.rowStart = 0;
      p0.rowEnd = gridRows;
      p0.colStart = 0;
      p0.colEnd = midCol;
      p0.numPEs = gridRows * midCol;
      p0.numFUs =
          static_cast<unsigned>(static_cast<double>(coreType.numFUs) *
                                midCol / gridCols);
      p0.spmBytes =
          static_cast<uint64_t>(static_cast<double>(coreType.spmBytes) *
                                midCol / gridCols);
      plan.partitions.push_back(p0);

      PartitionSpec p1;
      p1.rowStart = 0;
      p1.rowEnd = gridRows;
      p1.colStart = midCol;
      p1.colEnd = gridCols;
      p1.numPEs = gridRows * (gridCols - midCol);
      p1.numFUs = coreType.numFUs - p0.numFUs;
      p1.spmBytes = coreType.spmBytes - p0.spmBytes;
      plan.partitions.push_back(p1);
    } else {
      // Grid too small for 2-way split; return single partition.
      PartitionSpec full;
      full.rowStart = 0;
      full.rowEnd = gridRows;
      full.colStart = 0;
      full.colEnd = gridCols;
      full.numPEs = coreType.numPEs;
      full.numFUs = coreType.numFUs;
      full.spmBytes = coreType.spmBytes;
      plan.partitions.push_back(full);
    }
  } else {
    // 4-way split: try row-wise first, then 2x2 grid.
    if (gridRows >= 8) {
      // 4-way row-wise split.
      unsigned rowsPerPart = gridRows / 4;
      for (unsigned idx = 0; idx < 4; ++idx) {
        PartitionSpec p;
        p.rowStart = idx * rowsPerPart;
        p.rowEnd = (idx == 3) ? gridRows : (idx + 1) * rowsPerPart;
        p.colStart = 0;
        p.colEnd = gridCols;
        unsigned partRows = p.rowEnd - p.rowStart;
        p.numPEs = partRows * gridCols;
        p.numFUs =
            static_cast<unsigned>(static_cast<double>(coreType.numFUs) *
                                  partRows / gridRows);
        p.spmBytes =
            static_cast<uint64_t>(static_cast<double>(coreType.spmBytes) *
                                  partRows / gridRows);
        plan.partitions.push_back(p);
      }
    } else if (gridRows >= 4 && gridCols >= 4) {
      // 2x2 grid split.
      unsigned midRow = gridRows / 2;
      unsigned midCol = gridCols / 2;

      unsigned rowBounds[] = {0, midRow, midRow, gridRows};
      unsigned colBounds[] = {0, midCol, midCol, gridCols};

      for (unsigned rIdx = 0; rIdx < 2; ++rIdx) {
        for (unsigned cIdx = 0; cIdx < 2; ++cIdx) {
          PartitionSpec p;
          p.rowStart = rowBounds[rIdx * 2];
          p.rowEnd = rowBounds[rIdx * 2 + 1];
          p.colStart = colBounds[cIdx * 2];
          p.colEnd = colBounds[cIdx * 2 + 1];
          unsigned partRows = p.rowEnd - p.rowStart;
          unsigned partCols = p.colEnd - p.colStart;
          p.numPEs = partRows * partCols;
          double fraction = static_cast<double>(p.numPEs) /
                            (gridRows * gridCols);
          p.numFUs =
              static_cast<unsigned>(coreType.numFUs * fraction);
          p.spmBytes =
              static_cast<uint64_t>(coreType.spmBytes * fraction);
          plan.partitions.push_back(p);
        }
      }
    } else {
      // Grid too small for 4-way; return single partition.
      PartitionSpec full;
      full.rowStart = 0;
      full.rowEnd = gridRows;
      full.colStart = 0;
      full.colEnd = gridCols;
      full.numPEs = coreType.numPEs;
      full.numFUs = coreType.numFUs;
      full.spmBytes = coreType.spmBytes;
      plan.partitions.push_back(full);
    }
  }

  return plan;
}

//===----------------------------------------------------------------------===//
// Partition Validation
//===----------------------------------------------------------------------===//

PartitionValidation ADGPartitioner::validatePartition(
    const PartitionPlan &plan) {
  PartitionValidation result;

  if (plan.partitions.empty()) {
    result.valid = false;
    result.errorMessage = "Empty partition plan";
    return result;
  }

  unsigned totalRows = plan.totalRows;
  unsigned totalCols = plan.totalCols;

  if (totalRows == 0 || totalCols == 0) {
    result.valid = false;
    result.errorMessage = "Zero grid dimensions";
    return result;
  }

  // Build a coverage bitmap to check non-overlapping and full coverage.
  std::vector<unsigned> coverage(totalRows * totalCols, 0);

  for (size_t pIdx = 0; pIdx < plan.partitions.size(); ++pIdx) {
    const auto &p = plan.partitions[pIdx];

    // Bounds check.
    if (p.rowStart >= p.rowEnd || p.colStart >= p.colEnd) {
      result.valid = false;
      result.errorMessage = "Partition " + std::to_string(pIdx) +
                            " has invalid bounds";
      return result;
    }
    if (p.rowEnd > totalRows || p.colEnd > totalCols) {
      result.valid = false;
      result.errorMessage = "Partition " + std::to_string(pIdx) +
                            " exceeds grid dimensions";
      return result;
    }

    for (unsigned r = p.rowStart; r < p.rowEnd; ++r) {
      for (unsigned c = p.colStart; c < p.colEnd; ++c) {
        unsigned idx = r * totalCols + c;
        coverage[idx]++;
        if (coverage[idx] > 1) {
          result.valid = false;
          result.errorMessage =
              "Overlap at (" + std::to_string(r) + "," +
              std::to_string(c) + ")";
          return result;
        }
      }
    }
  }

  // Check full coverage: every cell must be covered exactly once.
  for (unsigned r = 0; r < totalRows; ++r) {
    for (unsigned c = 0; c < totalCols; ++c) {
      unsigned idx = r * totalCols + c;
      if (coverage[idx] == 0) {
        result.valid = false;
        result.errorMessage =
            "Gap at (" + std::to_string(r) + "," + std::to_string(c) + ")";
        return result;
      }
    }
  }

  result.valid = true;
  return result;
}

//===----------------------------------------------------------------------===//
// Configuration Merge
//===----------------------------------------------------------------------===//

std::vector<uint8_t>
ADGPartitioner::mergeConfigurations(
    const std::vector<std::vector<uint8_t>> &partitionConfigs,
    const PartitionPlan &plan,
    size_t fullConfigSize) {

  std::vector<uint8_t> merged(fullConfigSize, 0);

  if (partitionConfigs.size() != plan.partitions.size())
    return merged;

  // Since partitions are disjoint by construction, merging is a simple
  // bitwise OR of each partition's config into the full config blob.
  // Each partition's config addresses are relative to the partition's
  // starting PE, but the actual config bits are at their physical addresses.
  for (size_t pIdx = 0; pIdx < partitionConfigs.size(); ++pIdx) {
    const auto &pConfig = partitionConfigs[pIdx];
    const auto &pSpec = plan.partitions[pIdx];

    // Compute the byte offset for this partition's PE range.
    // Each PE has (fullConfigSize / totalPEs) bytes of config space.
    unsigned totalPEs = plan.totalRows * plan.totalCols;
    if (totalPEs == 0)
      continue;

    size_t bytesPerPE = fullConfigSize / totalPEs;
    if (bytesPerPE == 0)
      continue;

    // Map partition PEs to physical addresses and copy config bytes.
    unsigned partitionPEIdx = 0;
    for (unsigned r = pSpec.rowStart; r < pSpec.rowEnd; ++r) {
      for (unsigned c = pSpec.colStart; c < pSpec.colEnd; ++c) {
        unsigned physPEIdx = r * plan.totalCols + c;
        size_t physOffset = physPEIdx * bytesPerPE;
        size_t partOffset = partitionPEIdx * bytesPerPE;

        // Copy config bytes for this PE, OR-ing into merged.
        for (size_t byteIdx = 0; byteIdx < bytesPerPE; ++byteIdx) {
          if (partOffset + byteIdx < pConfig.size() &&
              physOffset + byteIdx < merged.size()) {
            merged[physOffset + byteIdx] |= pConfig[partOffset + byteIdx];
          }
        }
        partitionPEIdx++;
      }
    }
  }

  return merged;
}

} // namespace loom
