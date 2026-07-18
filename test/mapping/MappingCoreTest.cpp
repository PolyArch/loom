#include "MappingCoreTestSupport.h"

int main() {
  loom::mapping::test::runComputeFreezeTests();
  loom::mapping::test::runMemoryMappingTests();
  loom::mapping::test::runMappingVerifierTests();
  loom::mapping::test::runPnrProblemInputsTests();
  return 0;
}
