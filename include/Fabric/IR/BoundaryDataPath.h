#ifndef FABRIC_IR_BOUNDARYDATAPATH_H
#define FABRIC_IR_BOUNDARYDATAPATH_H

#include <cstdint>

namespace fabric {

enum class BoundaryDirection : std::uint32_t;

enum class DataPathKind { Bits, BitsTag };

struct DataPathType {
  DataPathKind kind;
  std::uint32_t payloadWidthBits;
  std::uint32_t tagWidthBits;

  constexpr bool isWellFormed() const {
    switch (kind) {
    case DataPathKind::Bits:
      return tagWidthBits == 0;
    case DataPathKind::BitsTag:
      return tagWidthBits > 0;
    }
    return false;
  }
};

enum class BoundaryDataPathError {
  None,
  InvalidDirection,
  InvalidSource,
  InvalidTarget,
  PayloadWidthMismatch,
};

BoundaryDataPathError checkBoundaryDataPath(BoundaryDirection direction,
                                            DataPathType source,
                                            DataPathType target);

} // namespace fabric

#endif // FABRIC_IR_BOUNDARYDATAPATH_H
