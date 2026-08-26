#include "FabricArtifactBytecodeInternal.h"

#include "Fabric/IR/FabricDialect.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
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
  BytecodeWriterConfig config("loom.fabric.3.0");
  config.setElideLocations();
  if (failed(writeBytecodeToFile(operation, stream, config)))
    return invalid("MLIR bytecode writer rejected the canonical root");
  return std::vector<std::uint8_t>(storage.begin(), storage.end());
}

std::shared_ptr<MLIRContext> createFabricContext() {
  DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  auto context =
      std::make_shared<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

llvm::Expected<ParsedFabricBytecodeModule>
normalizeFabricAssembly(Operation *operation) {
  // Rebuild intern tables from canonical traversal order so source-context
  // uniquing history cannot affect bytecode numbering. This transient generic
  // form retains inherent properties and never enters artifact identity.
  std::string assembly;
  llvm::raw_string_ostream stream(assembly);
  OpPrintingFlags flags;
  flags.printGenericOpForm().enableDebugInfo(false);
  operation->print(stream, flags);

  auto context = createFabricContext();
  auto module = parseSourceString<ModuleOp>(assembly, context.get());
  if (!module || failed(verify(module.get())))
    return invalid("canonical Fabric assembly cannot be reparsed");
  return ParsedFabricBytecodeModule{std::move(context), std::move(module)};
}

} // namespace

llvm::Expected<ParsedFabricBytecodeModule>
parseFabricBytecodeModule(llvm::ArrayRef<std::uint8_t> bytes) {
  auto context = createFabricContext();

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
  auto normalizedAssembly = normalizeFabricAssembly(operation);
  if (!normalizedAssembly)
    return normalizedAssembly.takeError();
  auto initial = writeBytecodeOnce(normalizedAssembly->module.get());
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

llvm::Error verifyCanonicalFabricBytecodeStability(
    Operation *operation, llvm::ArrayRef<std::uint8_t> canonical) {
  auto rewritten = writeBytecodeOnce(operation);
  if (!rewritten)
    return rewritten.takeError();
  if (!llvm::equal(*rewritten, canonical))
    return invalid("canonical MLIR bytecode is not byte stable");
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
