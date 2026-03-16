#ifndef FCC_MAPPER_TYPES_H
#define FCC_MAPPER_TYPES_H

#include <cstdint>

namespace fcc {

using IdIndex = uint32_t;
constexpr IdIndex INVALID_ID = static_cast<IdIndex>(-1);

enum class EntityKind { Node, Port, Edge };

} // namespace fcc

#endif // FCC_MAPPER_TYPES_H
