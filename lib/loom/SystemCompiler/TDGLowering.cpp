#include "loom/SystemCompiler/BendersDriver.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace loom {

std::map<std::string, mlir::ModuleOp>
lowerKernelsToDFG(mlir::ModuleOp tdgModule, mlir::MLIRContext *ctx) {
  std::map<std::string, mlir::ModuleOp> kernelDFGs;

  if (!tdgModule || !ctx)
    return kernelDFGs;

  // Stub implementation: walk the TDG module looking for named operations
  // that represent kernel bodies.  For each kernel found, create a
  // standalone MLIR module containing a placeholder for its DFG.
  //
  // The full pipeline will:
  //   1. Clone each tdg.kernel body into a standalone module.
  //   2. Run SCF-to-handshake lowering passes.
  //   3. Return a module containing a handshake.func for each kernel.
  //
  // For now, we create empty placeholder modules so the rest of the
  // Benders driver can operate structurally.

  mlir::OpBuilder builder(ctx);

  for (auto &op : tdgModule.getBody()->getOperations()) {
    // Use SymbolTable to extract named ops as kernel stand-ins.
    if (auto sym = op.getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName())) {
      std::string kernelName = sym.getValue().str();
      auto loc = builder.getUnknownLoc();
      auto kernelModule = mlir::ModuleOp::create(loc, kernelName);
      kernelDFGs[kernelName] = kernelModule;
    }
  }

  return kernelDFGs;
}

} // namespace loom
