#include "StructuredOwnershipAnalysis.h"

#include "StructuredCallSpecialization.h"

#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Lowering/GraphMemoryAddressing.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"

namespace loom::frontend::detail {
namespace {

bool pointerLineageRequiresMemoryService(mlir::Value pointer,
                                         llvm::DenseSet<mlir::Value> &visited) {
  if (!pointer || !visited.insert(pointer).second)
    return false;
  for (mlir::OpOperand &use : pointer.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(owner)) {
      if (load.getAddr() == pointer)
        return true;
    } else if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(owner)) {
      if (store.getAddr() == pointer)
        return true;
    }
    for (mlir::Value result : owner->getResults())
      if (llvm::isa<mlir::LLVM::LLVMPointerType>(result.getType()) &&
          pointerLineageRequiresMemoryService(result, visited))
        return true;
  }
  return false;
}

mlir::Operation *lastDynamicPointerServiceSource(mlir::Block &block) {
  mlir::Operation *last = nullptr;
  for (mlir::Operation &operation : block.without_terminator()) {
    auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation);
    if (!load ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(load.getResult().getType()))
      continue;
    llvm::DenseSet<mlir::Value> visited;
    if (pointerLineageRequiresMemoryService(load.getResult(), visited))
      last = &operation;
  }
  return last;
}

bool isDefinedInSelection(
    mlir::Value value,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &selected) {
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value))
    return selected.contains(result.getOwner());
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  return argument && selected.contains(argument.getOwner()->getParentOp());
}

llvm::SmallVector<mlir::Value, 8> deriveSelectionLiveIns(
    llvm::ArrayRef<mlir::Operation *> selectedBody,
    const llvm::SmallPtrSetImpl<mlir::Operation *> &selected) {
  llvm::SmallVector<mlir::Value, 8> liveIns;
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  for (mlir::Operation *topLevel : selectedBody)
    topLevel->walk([&](mlir::Operation *operation) {
      for (mlir::Value operand : operation->getOperands())
        if (!isDefinedInSelection(operand, selected) &&
            seen.insert(operand).second)
          liveIns.push_back(operand);
    });
  return liveIns;
}

} // namespace

std::optional<std::string>
explainCallableOwnershipRejection(mlir::LLVM::LLVMFuncOp function) {
  if (function.isExternal())
    return "selected callable has no definition";
  if (function.isVarArg())
    return "variadic callable ownership is not materialized";
  if (!function.getBody().hasOneBlock())
    return "whole-callable ownership requires one structured block";

  mlir::Block &body = function.getBody().front();
  auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(body.getTerminator());
  if (!returnOp)
    return "selected callable has no direct LLVM return";
  const bool returnsVoid = llvm::isa<mlir::LLVM::LLVMVoidType>(
      function.getFunctionType().getReturnType());
  if (returnOp.getNumOperands() != static_cast<unsigned>(!returnsVoid))
    return "selected callable return does not match its LLVM ABI";
  return std::nullopt;
}

std::optional<std::string>
explainGraphStructuralOwnershipRejection(mlir::ModuleOp module,
                                         mlir::Operation *selection) {
  std::optional<ExactDirectCallSiteInliningCandidate> directCall =
      findExactDirectCallSiteInliningCandidate(module, selection);
  return lowering::explainGraphRegionStructuralRejection(
      selection, directCall ? directCall->callSite : nullptr);
}

bool containsGeneralCall(mlir::Operation *selection) {
  bool contains = false;
  selection->walk([&](mlir::Operation *operation) {
    if (!llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::InvokeOp>(operation))
      return mlir::WalkResult::advance();
    contains = true;
    return mlir::WalkResult::interrupt();
  });
  return contains;
}

CallableSpatialSlice
deriveCallableSpatialSlice(mlir::LLVM::LLVMFuncOp function,
                           const CallableOwnershipBoundary &boundary) {
  CallableSpatialSlice slice;
  mlir::Block &block = function.getBody().front();
  mlir::Operation *servicePrefixEnd = lastDynamicPointerServiceSource(block);
  bool afterServicePrefix = servicePrefixEnd == nullptr;
  for (mlir::Operation &operation : block.without_terminator()) {
    if (!afterServicePrefix) {
      afterServicePrefix = &operation == servicePrefixEnd;
      continue;
    }
    if (!llvm::isa<mlir::LLVM::AddressOfOp, mlir::LLVM::UndefOp>(operation))
      slice.body.push_back(&operation);
  }

  if (!servicePrefixEnd) {
    slice.liveIns.assign(boundary.inputs.begin(), boundary.inputs.end());
    slice.liveOuts.assign(boundary.outputs.begin(), boundary.outputs.end());
    return slice;
  }

  llvm::SmallPtrSet<mlir::Operation *, 32> selected;
  for (mlir::Operation *operation : slice.body)
    operation->walk([&](mlir::Operation *nested) { selected.insert(nested); });
  slice.liveIns = deriveSelectionLiveIns(slice.body, selected);
  for (mlir::Value output : boundary.outputs)
    if (isDefinedInSelection(output, selected))
      slice.liveOuts.push_back(output);
  return slice;
}

CallableOwnershipBoundary
deriveCallableOwnershipBoundary(mlir::LLVM::LLVMFuncOp function) {
  CallableOwnershipBoundary boundary;
  function.walk([&](mlir::LLVM::AddressOfOp address) {
    if (!address.getRes().use_empty())
      boundary.addresses.push_back(address);
  });
  function.walk([&](mlir::LLVM::UndefOp undef) {
    if (!undef.getRes().use_empty())
      boundary.undefs.push_back(undef);
  });
  mlir::Block &entry = function.getBody().front();
  for (mlir::BlockArgument argument : entry.getArguments())
    if (!argument.use_empty())
      boundary.inputs.push_back(argument);
  for (mlir::LLVM::AddressOfOp address : boundary.addresses)
    boundary.inputs.push_back(address.getRes());
  for (mlir::LLVM::UndefOp undef : boundary.undefs)
    boundary.inputs.push_back(undef.getRes());
  if (auto returnOp =
          llvm::dyn_cast<mlir::LLVM::ReturnOp>(entry.getTerminator()))
    llvm::append_range(boundary.outputs, returnOp.getOperands());
  return boundary;
}

std::optional<std::string>
completeMemoryServiceBoundary(llvm::ArrayRef<mlir::Operation *> selectedBody,
                              std::vector<mlir::Value> &liveIns) {
  llvm::SmallPtrSet<mlir::Value, 8> boundaryPointers;
  for (mlir::Value value : liveIns)
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
      boundaryPointers.insert(value);

  llvm::SmallVector<mlir::Operation *> unbound;
  for (mlir::Operation *topLevel : selectedBody)
    topLevel->walk([&](mlir::Operation *operation) {
      mlir::Value address;
      if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation))
        address = load.getAddr();
      else if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation))
        address = store.getAddr();
      if (address && !lowering::resolveMemoryServiceBoundaryRoot(
                         address, [&](mlir::Value value) {
                           return boundaryPointers.contains(value);
                         }) &&
          lowering::usesLoadedPointerService(address))
        unbound.push_back(operation);
    });
  if (unbound.empty())
    return std::nullopt;

  auto callable = unbound.front()->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!callable)
    return "unbound pointer service has no exact LLVM invocation owner";
  // The analysis is created after preparation's IR changes and ends here,
  // before materialization clones operations or changes the input boundary.
  analysis::StoredMemoryProvenance provenance(callable);
  mlir::DominanceInfo dominance(callable);
  for (mlir::Operation *operation : unbound) {
    mlir::Value address =
        llvm::isa<mlir::LLVM::LoadOp>(operation)
            ? llvm::cast<mlir::LLVM::LoadOp>(operation).getAddr()
            : llvm::cast<mlir::LLVM::StoreOp>(operation).getAddr();
    auto projected = provenance.projectPointerTarget(address);
    if (auto refusal = std::get_if<analysis::StoredPointerRefusal>(&projected))
      return (llvm::Twine("unbound pointer service: ") +
              analysis::storedPointerRefusalSpelling(*refusal))
          .str();
    mlir::Value root = std::get<analysis::StoredPointerTarget>(projected).root;
    for (mlir::Operation *selected : selectedBody)
      if (!dominance.dominates(root, selected))
        return "pointer service origin is unavailable before the selected "
               "scope";
    if (boundaryPointers.insert(root).second)
      liveIns.push_back(root);
  }
  auto projected = lowering::projectPointerServiceBindings(selectedBody,
                                                           liveIns, provenance);
  if (auto refusal = std::get_if<analysis::StoredPointerRefusal>(&projected))
    return (llvm::Twine("unbound pointer service: ") +
            analysis::storedPointerRefusalSpelling(*refusal))
        .str();
  const auto &bindings = std::get<lowering::PointerServiceBindings>(projected);
  for (mlir::Operation *operation : unbound) {
    mlir::Value address =
        llvm::isa<mlir::LLVM::LoadOp>(operation)
            ? llvm::cast<mlir::LLVM::LoadOp>(operation).getAddr()
            : llvm::cast<mlir::LLVM::StoreOp>(operation).getAddr();
    if (!lowering::resolveMemoryServiceBoundaryRoot(
            address,
            [&](mlir::Value value) { return boundaryPointers.contains(value); },
            bindings))
      return "memory access has no proven pointer service at the selected "
             "Spatial boundary";
  }
  return std::nullopt;
}

} // namespace loom::frontend::detail
