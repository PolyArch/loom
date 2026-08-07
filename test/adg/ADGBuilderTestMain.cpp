#include "ADGBuilderTestSupport.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: loom-adg-builder-api-test <test-group>\n";
    return EXIT_FAILURE;
  }

  using TestFunction = void (*)();
  TestFunction test =
      llvm::StringSwitch<TestFunction>(argv[1])
          .Case("service-leg-carrier",
                loom::adg::test::runServiceLegCarrierTests)
          .Case("builder", loom::adg::test::runBuilderTests)
          .Case("builtin", loom::adg::test::runBuiltinTests)
          .Case("topology", loom::adg::test::runTopologyTests)
          .Case("domain-authoring", loom::adg::test::runDomainAuthoringTests)
          .Default(nullptr);
  if (!test) {
    llvm::errs() << "unknown ADG builder test group: " << argv[1] << '\n';
    return EXIT_FAILURE;
  }
  test();
  return EXIT_SUCCESS;
}
