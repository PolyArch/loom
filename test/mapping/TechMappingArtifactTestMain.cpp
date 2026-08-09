#include "TechMappingArtifactTestSupport.h"

#include "llvm/Support/raw_ostream.h"

int main() {
  loom::test::tech_mapping_artifact::artifactRoundTripAndReferenceValidation();
  llvm::outs() << "tech mapping artifact tests passed\n";
  return 0;
}
