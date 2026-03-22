#ifndef LOOM_MAPPER_TYPES_H
#define LOOM_MAPPER_TYPES_H

#include <cstdint>

namespace loom {

using IdIndex = uint32_t;
constexpr IdIndex INVALID_ID = static_cast<IdIndex>(-1);

enum class EntityKind { Node, Port, Edge };

} // namespace loom

#endif // LOOM_MAPPER_TYPES_H
