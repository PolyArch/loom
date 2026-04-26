#ifndef FABRIC_TECH_PARTITIONER_COSTMODEL_H
#define FABRIC_TECH_PARTITIONER_COSTMODEL_H

#include "Common/Config.h"
#include "Fabric/Tech/Partitioner/Partitioner.h"
#include "Fabric/Tech/TemplateLibrary.h"

namespace fabric {

// Cost of a partition under the tech-mapping objective.
//
// Formula:
//
//   cost = alpha * |blocks_with_template|
//        + beta  * cross_edges
//        - gamma * avg_density
//
// Where:
//
//   * |blocks_with_template| counts blocks whose `tpl != nullptr`. Blocks
//     left at the graph level (`tpl == nullptr`) are not counted toward
//     this term.
//   * cross_edges is the number of (def, use) pairs (op_d, op_u) where
//     op_d and op_u live in the body of the same dataflow.graph but in
//     different Blocks of the partition. Edges from / to ops that are not
//     in any Block of the partition are ignored.
//   * density(block) is defined for blocks with `tpl != nullptr` as
//
//       density(block) = block.ops.size()
//                      / max(1, max_template_size_for_root_op_in_block)
//
//     where max_template_size_for_root_op_in_block is the largest
//     `bodyOpCount` across all templates whose `rootOpName` matches the
//     bound template's root. avg_density is the arithmetic mean of
//     density(block) over blocks with `tpl != nullptr`. Graph-level
//     (tpl == nullptr) blocks do NOT contribute to avg_density.
//   * If no block has a template, avg_density is treated as 0.
//
// All weights are taken from `cfg`; coefficients and the formula are kept
// in this header so cost-model changes have a single point of authority.
double computeCost(const PartitionResult &partition,
                   const TemplateLibrary &lib,
                   const ::loom::TechMapConfig &cfg);

} // namespace fabric

#endif // FABRIC_TECH_PARTITIONER_COSTMODEL_H
