#include "Fabric/IR/BoundaryDataPath.h"
#include "Fabric/IR/FabricOps.h"

#include <cstdlib>
#include <iostream>

namespace {

using fabric::BoundaryDataPathError;
using fabric::BoundaryDirection;
using fabric::DataPathKind;
using fabric::DataPathType;

[[noreturn]] void fail(const char *message) {
  std::cerr << message << '\n';
  std::exit(1);
}

void expect(BoundaryDirection direction, DataPathType source,
            DataPathType target, BoundaryDataPathError expected) {
  if (fabric::checkBoundaryDataPath(direction, source, target) != expected)
    fail("boundary data-path legality result differs");
}

} // namespace

int main() {
  const DataPathType bits32{DataPathKind::Bits, 32, 0};
  const DataPathType tagged32x4{DataPathKind::BitsTag, 32, 4};
  const DataPathType tagged32x8{DataPathKind::BitsTag, 32, 8};

  if (!bits32.isWellFormed() || !tagged32x4.isWellFormed() ||
      DataPathType{DataPathKind::Bits, 32, 1}.isWellFormed() ||
      DataPathType{DataPathKind::BitsTag, 32, 0}.isWellFormed())
    fail("Fabric data-path type well-formedness differs");

  expect(BoundaryDirection::S2t, bits32, tagged32x4,
         BoundaryDataPathError::None);
  expect(BoundaryDirection::T2t, tagged32x4, tagged32x8,
         BoundaryDataPathError::None);
  expect(BoundaryDirection::T2s, tagged32x4, bits32,
         BoundaryDataPathError::None);
  expect(BoundaryDirection::S2t, tagged32x4, tagged32x4,
         BoundaryDataPathError::InvalidSource);
  expect(BoundaryDirection::S2t, bits32, bits32,
         BoundaryDataPathError::InvalidTarget);
  expect(BoundaryDirection::T2t, tagged32x4,
         DataPathType{DataPathKind::BitsTag, 16, 8},
         BoundaryDataPathError::PayloadWidthMismatch);
  expect(static_cast<BoundaryDirection>(99), bits32, tagged32x4,
         BoundaryDataPathError::InvalidDirection);
  return 0;
}
