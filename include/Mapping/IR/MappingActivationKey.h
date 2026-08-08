#ifndef LOOM_MAPPING_IR_MAPPINGACTIVATIONKEY_H
#define LOOM_MAPPING_IR_MAPPINGACTIVATIONKEY_H

#include "Mapping/IR/MappingAttrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::mapping {

std::vector<std::uint8_t>
canonicalSpatialActivityEventKey(std::uint32_t eventKind,
                                 llvm::ArrayRef<std::uint8_t> eventRecord,
                                 std::optional<std::uint32_t> transition);

std::vector<std::uint8_t> canonicalSpatialEventPointKey(
    std::uint32_t eventKind, llvm::ArrayRef<std::uint8_t> eventRecord,
    std::optional<std::uint32_t> transition,
    std::optional<llvm::ArrayRef<std::uint8_t>> guaranteedOffset);

std::vector<std::uint8_t>
canonicalEventPointKey(::mapping::SpatialEventPointAttr point);
std::vector<std::uint8_t>
canonicalEventPointKey(::mapping::SystemEventPointAttr point);

} // namespace loom::mapping

#endif // LOOM_MAPPING_IR_MAPPINGACTIVATIONKEY_H
