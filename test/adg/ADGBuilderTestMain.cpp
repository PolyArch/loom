#include "ADGBuilderTestSupport.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, char **argv) {
  if (argc != 1) {
    llvm::errs() << "usage: loom-adg-builder-api-test\n";
    return EXIT_FAILURE;
  }
  loom::adg::test::runServiceLegCarrierTests();
  loom::adg::test::runBuilderTests();
  loom::adg::test::runBuiltinTests();
  loom::adg::test::runTopologyTests();
  loom::adg::test::runDomainAuthoringTests();
  llvm::outs() << "{\"anchors\":[\"regular-topology\","
                  "\"irregular-directed-topology\","
                  "\"heterogeneous-multi-acc-core\","
                  "\"temporal-resource-grant\","
                  "\"memory-service-forwarding\"]}\n";
  return EXIT_SUCCESS;
}
