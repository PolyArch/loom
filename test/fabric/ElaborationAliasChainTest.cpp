#include "Fabric/IR/Elaboration.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static constexpr unsigned chainLength = 20000;

static std::string buildInput(bool reverseOrder) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "module {\n"
        "  fabric.module @identity(%arg : !fabric.bits<8>) -> "
        "(!fabric.bits<8>) {\n"
        "    fabric.yield %arg : !fabric.bits<8>\n"
        "  }\n"
        "  fabric.module @root(%arg : !fabric.bits<8>) -> "
        "(!fabric.bits<8>) {\n";

  auto emitInstance = [&](unsigned index) {
    os << "    %v" << index << " = fabric.instantiate @identity(";
    if (index == 0)
      os << "%arg";
    else
      os << "%v" << index - 1;
    os << " : !fabric.bits<8>) -> (!fabric.bits<8>) "
          "{domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}\n";
  };

  if (reverseOrder) {
    for (unsigned index = chainLength; index > 0; --index)
      emitInstance(index - 1);
  } else {
    for (unsigned index = 0; index < chainLength; ++index)
      emitInstance(index);
  }

  os << "    fabric.yield %v" << chainLength - 1
     << " : !fabric.bits<8>\n"
        "  }\n"
        "}\n";
  return text;
}

static FailureOr<std::string> elaborate(MLIRContext &context,
                                        bool reverseOrder) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(buildInput(reverseOrder), &context);
  if (!module || failed(verify(*module)))
    return failure();
  fabric::ModuleOp root = module->lookupSymbol<fabric::ModuleOp>("root");
  if (failed(fabric::elaborateInstances(root)) || failed(verify(*module)))
    return failure();

  std::string text;
  llvm::raw_string_ostream(text) << root;
  return text;
}

int main() {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect<fabric::FabricDialect>();

  FailureOr<std::string> forward = elaborate(context, false);
  FailureOr<std::string> reverse = elaborate(context, true);
  if (failed(forward) || failed(reverse) || *forward != *reverse)
    return 1;

  llvm::outs() << "fabric alias chain ok\n";
  return 0;
}
