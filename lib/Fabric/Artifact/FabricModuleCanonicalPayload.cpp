#include "FabricModuleCanonicalPayload.h"

#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace loom::fabric::detail {
namespace {

constexpr llvm::StringLiteral moduleAuthoringOnlyAttrs[] = {
    "sel",
    "discard",
    "disconnect",
    "bypassed",
    "sw_configs",
    "pe_enable",
    "instruction_mem",
    "per_fu_sw_configs",
    "visual_layout",
    "coordinates_semantic"};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

llvm::Error stripFabricModuleAuthoringState(::fabric::ModuleOp root) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](Operation *operation) {
    if (result)
      return WalkResult::interrupt();
    operation->removeAttr(::fabric::kEntityIdAttrName);
    operation->removeAttr(::fabric::kFuTemplateIdAttrName);
    operation->removeAttr(::fabric::kMemoryEngineTemplateIdAttrName);
    operation->removeAttr("domain_slots");
    operation->removeAttr("domain_assignments");
    if (!isa<::fabric::OpOp>(operation))
      operation->removeAttr(::fabric::kResourceContractRecordAttrName);

    if (auto semantic =
            operation->getAttrOfType<BoolAttr>("coordinates_semantic");
        semantic && semantic.getValue()) {
      result = invalid("authoring coordinates claim semantic authority");
      return WalkResult::interrupt();
    }
    for (llvm::StringLiteral name : moduleAuthoringOnlyAttrs)
      operation->removeAttr(name);
    return WalkResult::advance();
  });
  return result;
}

llvm::Error eraseElaboratedFabricModuleDeclarations(::fabric::ModuleOp root) {
  llvm::SmallVector<Operation *> declarations;
  root->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    if (operation == root.getOperation())
      return;
    auto symbol = dyn_cast<SymbolOpInterface>(operation);
    if (symbol && symbol.getNameAttr())
      declarations.push_back(operation);
  });
  for (Operation *declaration : declarations) {
    for (Value result : declaration->getResults())
      if (!result.use_empty())
        return invalid("an elaborated declaration still has an SSA use");
    declaration->erase();
  }
  bool residualInstance = false;
  root->walk([&](::fabric::InstantiateOp) { residualInstance = true; });
  if (residualInstance)
    return invalid("a fully elaborated Fabric contains fabric.instantiate");
  return llvm::Error::success();
}

llvm::Error validateCanonicalFabricModulePayload(::fabric::ModuleOp root) {
  enum class Violation { None, AuthoringState, NamedDeclaration };
  Violation violation = Violation::None;
  root->walk([&](Operation *operation) {
    for (llvm::StringLiteral name : moduleAuthoringOnlyAttrs)
      if (operation->hasAttr(name)) {
        violation = Violation::AuthoringState;
        return WalkResult::interrupt();
      }
    if (operation != root.getOperation())
      if (auto symbol = dyn_cast<SymbolOpInterface>(operation))
        if (symbol.getNameAttr()) {
          violation = Violation::NamedDeclaration;
          return WalkResult::interrupt();
        }
    return WalkResult::advance();
  });
  if (violation == Violation::AuthoringState)
    return invalid("canonical Module payload retains authoring-only state");
  if (violation == Violation::NamedDeclaration)
    return invalid("canonical Module payload retains a named declaration");
  return llvm::Error::success();
}

} // namespace loom::fabric::detail
