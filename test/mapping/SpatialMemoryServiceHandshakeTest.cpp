#include "SpatialMemoryMappingArtifactTestSupport.h"

#include "llvm/Support/raw_ostream.h"

int main() {
  loom::test::completeMemorySpatialMappingRoundTrip(false);
  loom::test::completeMemorySpatialMappingRoundTrip(true);
  llvm::outs() << "spatial memory-service handshake tests passed\n";
  return 0;
}
