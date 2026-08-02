#include "FabricArtifactBytecodeInternal.h"

#include "Fabric/IR/FabricDialect.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Expected<std::vector<std::uint8_t>>
writeBytecodeOnce(Operation *operation) {
  llvm::SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  BytecodeWriterConfig config("loom.fabric.1.1");
  config.setElideLocations();
  if (failed(writeBytecodeToFile(operation, stream, config)))
    return invalid("MLIR bytecode writer rejected the canonical root");
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

} // namespace

llvm::Expected<ParsedFabricBytecodeModule>
parseFabricBytecodeModule(llvm::ArrayRef<std::uint8_t> bytes) {
  DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();

  llvm::StringRef byteString(reinterpret_cast<const char *>(bytes.data()),
                             bytes.size());
  llvm::MemoryBufferRef buffer(byteString, "<canonical-fabric>");
  ParserConfig parserConfig(context.get());
  Block topLevel;
  if (failed(readBytecodeFile(buffer, &topLevel, parserConfig)))
    return invalid("canonical MLIR bytecode cannot be parsed");
  if (!llvm::hasSingleElement(topLevel))
    return invalid("canonical MLIR bytecode has multiple top-level roots");
  auto module = dyn_cast<ModuleOp>(&topLevel.front());
  if (!module || failed(verify(module)))
    return invalid("canonical MLIR bytecode is not a valid builtin module");
  module->remove();
  return ParsedFabricBytecodeModule{std::move(context),
                                    OwningOpRef<ModuleOp>(module)};
}

llvm::Expected<std::vector<std::uint8_t>>
writeCanonicalFabricBytecode(Operation *operation) {
  auto initial = writeBytecodeOnce(operation);
  if (!initial)
    return initial.takeError();
  auto normalizedModule = parseFabricBytecodeModule(*initial);
  if (!normalizedModule)
    return normalizedModule.takeError();
  auto canonical = writeBytecodeOnce(normalizedModule->module.get());
  if (!canonical)
    return canonical.takeError();

  auto verificationModule = parseFabricBytecodeModule(*canonical);
  if (!verificationModule)
    return verificationModule.takeError();
  auto verified = writeBytecodeOnce(verificationModule->module.get());
  if (!verified)
    return verified.takeError();
  if (*verified != *canonical)
    return invalid("the Fabric schema writer did not reach a byte-stable "
                   "canonical form");
  return canonical;
}

} // namespace loom::fabric::detail
