#include "TechMappingArtifactTestSupport.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: loom-spatial-candidate-workflow-test <case>\n";
    return EXIT_FAILURE;
  }

  const llvm::StringRef testCase(argv[1]);
  loom::test::tech_mapping_artifact::spatialCandidateWorkflow(testCase);
  llvm::outs() << "spatial candidate workflow tests passed\n";
  return EXIT_SUCCESS;
}
