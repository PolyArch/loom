#include "Dataflow/IR/DataflowCanonicalArtifact.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <utility>
#include <vector>

namespace dataflow {
namespace {

llvm::Error invocationPathError(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "dataflow_direct_invocation_invalid: " + message);
}

llvm::Expected<std::vector<CanonicalDirectInvocationPathView>>
enumerateDirectInvocationPaths(mlir::ModuleOp module,
                               llvm::StringRef entrySymbol,
                               mlir::LLVM::LLVMFuncOp target) {
  auto entry = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      mlir::SymbolTable::lookupSymbolIn(module, entrySymbol));
  if (!entry || entry.isExternal() || !target || target.isExternal())
    return invocationPathError("entry or target is not a defined LLVM "
                               "callable");

  std::vector<CanonicalDirectInvocationPathView> result;
  if (entry == target) {
    result.push_back({});
    return result;
  }

  std::vector<mlir::Operation *> path;
  llvm::SmallPtrSet<mlir::Operation *, 16> active;
  std::function<llvm::Error(mlir::LLVM::LLVMFuncOp)> visit =
      [&](mlir::LLVM::LLVMFuncOp function) -> llvm::Error {
    if (!active.insert(function.getOperation()).second)
      return invocationPathError("application invocation closure is "
                                 "recursive");
    llvm::Error error = llvm::Error::success();
    function.walk([&](mlir::LLVM::CallOp call) {
      if (error)
        return mlir::WalkResult::interrupt();
      if (!call.getCalleeAttr()) {
        error = invocationPathError(
            "application invocation closure contains an indirect call");
        return mlir::WalkResult::interrupt();
      }
      auto callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr());
      if (!callee || callee.isExternal())
        return mlir::WalkResult::advance();
      path.push_back(call.getOperation());
      if (callee == target)
        result.push_back({path});
      else if (llvm::Error nested = visit(callee))
        error = std::move(nested);
      path.pop_back();
      return error ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
    });
    active.erase(function.getOperation());
    return error;
  };
  if (llvm::Error error = visit(entry))
    return std::move(error);
  return result;
}

} // namespace

llvm::Expected<std::vector<RootThreadLaunchRef>>
CanonicalDataflowProgramView::projectRootThreadLaunchesReachableFromAbiEntry(
    llvm::StringRef entrySymbol) const {
  if (entrySymbol.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "application ABI entry symbol is empty");
  auto entry = mlir::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      mlir::SymbolTable::lookupSymbolIn(module_, entrySymbol));
  if (!entry || entry.isExternal())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "application ABI entry is not a defined LLVM function");

  std::vector<RootThreadLaunchRef> roots;
  llvm::DenseMap<mlir::Operation *, bool> reachableOwners;
  for (const CanonicalRootThreadLaunchView &root : rootThreadLaunches_) {
    auto owner = root.op->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!owner || owner.isExternal())
      return invocationPathError(
          "root launch is not owned by a defined LLVM callable");
    auto [known, inserted] =
        reachableOwners.try_emplace(owner.getOperation(), false);
    if (inserted) {
      auto paths = enumerateDirectInvocationPaths(module_, entrySymbol, owner);
      if (!paths)
        return paths.takeError();
      known->second = !paths->empty();
    }
    if (known->second)
      roots.push_back(root.ref);
  }
  return roots;
}

llvm::Expected<std::vector<CanonicalDirectInvocationPathView>>
CanonicalDataflowProgramView::projectRootThreadInvocationPathsFromAbiEntry(
    llvm::StringRef entrySymbol, RootThreadLaunchRef root) const {
  auto resolved = resolve(root);
  if (!resolved)
    return resolved.takeError();
  auto owner = resolved->op
                   ? resolved->op->getParentOfType<mlir::LLVM::LLVMFuncOp>()
                   : mlir::LLVM::LLVMFuncOp{};
  if (!owner || owner.isExternal())
    return invocationPathError(
        "root launch is not owned by a defined LLVM callable");
  auto paths = enumerateDirectInvocationPaths(module_, entrySymbol, owner);
  if (!paths)
    return paths.takeError();
  if (paths->empty())
    return invocationPathError(
        "root owner '" + owner.getSymName() +
        "' has no direct call path from application entry '" + entrySymbol +
        "'");
  return paths;
}

} // namespace dataflow
