#include "MappingCoreTestSupport.h"

int main() {
  loom::mapping::test::runComputeFreezeTests();
  loom::mapping::test::runMemoryMappingTests();
  loom::mapping::test::runMappingVerifierTests();
  return 0;
}
