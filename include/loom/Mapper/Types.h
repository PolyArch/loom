//===-- Types.h - Mapper central type definitions -----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Central type definitions shared across all mapper components.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_TYPES_H
#define LOOM_MAPPER_TYPES_H

#include <cstdint>

namespace loom {

/// Single index/ID type used throughout the mapper. The position of an entity
/// in its owning vector IS its ID (ID-as-Index principle).
using IdIndex = uint32_t;

/// Invalid sentinel value (all-ones: 0xFFFFFFFF).
constexpr IdIndex INVALID_ID = static_cast<IdIndex>(-1);

/// Discriminator for entity kinds in isValid() queries.
enum class EntityKind { Node, Port, Edge };

} // namespace loom

#endif // LOOM_MAPPER_TYPES_H
