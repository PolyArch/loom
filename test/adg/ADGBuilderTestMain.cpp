#include "ADGBuilderTestSupport.h"

#include <cstdlib>

int main() {
  loom::adg::test::runBuilderTests();
  loom::adg::test::runBuiltinTests();
  loom::adg::test::runTopologyTests();
  return EXIT_SUCCESS;
}
