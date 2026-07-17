#include "Fabric/IR/BoundaryDataPath.h"

#include "Fabric/IR/FabricOps.h"

using namespace fabric;

namespace {

BoundaryDataPathError checkKinds(DataPathKind sourceKind,
                                 DataPathKind targetKind, DataPathType source,
                                 DataPathType target) {
  if (!source.isWellFormed() || source.kind != sourceKind)
    return BoundaryDataPathError::InvalidSource;
  if (!target.isWellFormed() || target.kind != targetKind)
    return BoundaryDataPathError::InvalidTarget;
  if (source.payloadWidthBits != target.payloadWidthBits)
    return BoundaryDataPathError::PayloadWidthMismatch;
  return BoundaryDataPathError::None;
}

} // namespace

BoundaryDataPathError fabric::checkBoundaryDataPath(BoundaryDirection direction,
                                                    DataPathType source,
                                                    DataPathType target) {
  switch (direction) {
  case BoundaryDirection::S2t:
    return checkKinds(DataPathKind::Bits, DataPathKind::BitsTag, source,
                      target);
  case BoundaryDirection::T2t:
    return checkKinds(DataPathKind::BitsTag, DataPathKind::BitsTag, source,
                      target);
  case BoundaryDirection::T2s:
    return checkKinds(DataPathKind::BitsTag, DataPathKind::Bits, source,
                      target);
  }
  return BoundaryDataPathError::InvalidDirection;
}
