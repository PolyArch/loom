#include "Fabric/Artifact/FabricArtifact.h"

#include "FabricArtifactBytecodeInternal.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace loom::fabric {

llvm::Error writeFabricMlir(const FinalizedFabricRoot &root,
                            llvm::raw_ostream &output) {
  llvm::Expected<DecodedFabricArtifact> decoded =
      decodeFabricArtifactEnvelope(root.canonicalBytes().bytes());
  if (!decoded)
    return decoded.takeError();
  if (decoded->rootKind != root.view().rootKind())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_artifact_invalid: finalized root kind disagrees with its "
        "canonical envelope");

  llvm::Expected<detail::ParsedFabricBytecodeModule> parsed =
      detail::parseFabricBytecodeModule(decoded->canonicalMlirBytecode);
  if (!parsed)
    return parsed.takeError();
  parsed->module->print(output);
  output << '\n';
  return llvm::Error::success();
}

} // namespace loom::fabric
