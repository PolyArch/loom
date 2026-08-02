#ifndef LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H

#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/Operation.h"

#include <cstdint>
#include <vector>

namespace loom::mapping::detail {

std::vector<std::uint8_t> canonicalTechChildKey(mlir::Operation &operation);
std::vector<std::uint8_t>
canonicalTechRealizationPayloadKey(::mapping::ComputeRealizationOp realization);
std::vector<std::uint8_t>
canonicalTechRealizationPayloadKey(::mapping::MemoryRealizationOp realization);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_TECHMAPPINGCANONICALKEYINTERNAL_H
