#include "ADGBuilderTestSupport.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace loom::adg::test {

void runConformanceAnchorTests() {
  regularAndIrregularSpatialCoresFinalize();
  heterogeneousSystemFinalizes();
  temporalResourceGrantFinalizes();
  publicMemoryLibraryBuildsHybridLocalMemories();
  builtinPresetsExpandThroughPublicBuilder();
}

} // namespace loom::adg::test

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--conformance-anchors") {
    loom::adg::test::runConformanceAnchorTests();
    llvm::outs() << "{\"anchors\":[\"regular-topology\","
                    "\"irregular-directed-topology\","
                    "\"heterogeneous-multi-acc-core\","
                    "\"temporal-resource-grant\","
                    "\"memory-service-forwarding\"]}\n";
    return EXIT_SUCCESS;
  }
  if (argc != 1) {
    llvm::errs() << "usage: loom-adg-builder-api-test "
                    "[--conformance-anchors]\n";
    return EXIT_FAILURE;
  }
  loom::adg::test::runServiceLegCarrierTests();
  loom::adg::test::runBuilderTests();
  loom::adg::test::runBuiltinTests();
  loom::adg::test::runTopologyTests();
  return EXIT_SUCCESS;
}
