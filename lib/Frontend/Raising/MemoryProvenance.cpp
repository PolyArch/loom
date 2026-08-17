#include "Frontend/Raising/MemoryProvenance.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

namespace loom::raising {
namespace {

bool isEnclosingBlockArgument(mlir::Value value, mlir::Operation *operation) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return false;
  for (mlir::Operation *current = operation; current;
       current = current->getParentOp())
    if (current->getBlock() == argument.getOwner())
      return true;
  return false;
}

bool areDistinctNoAliasFunctionArguments(mlir::Value lhs, mlir::Value rhs) {
  auto lhsArgument = llvm::dyn_cast<mlir::BlockArgument>(lhs);
  auto rhsArgument = llvm::dyn_cast<mlir::BlockArgument>(rhs);
  if (!lhsArgument || !rhsArgument ||
      lhsArgument.getOwner() != rhsArgument.getOwner() ||
      lhsArgument.getArgNumber() == rhsArgument.getArgNumber())
    return false;
  mlir::Operation *owner = lhsArgument.getOwner()->getParentOp();
  mlir::DictionaryAttr lhsAttrs;
  mlir::DictionaryAttr rhsAttrs;
  if (auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(owner)) {
    lhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, lhsArgument.getArgNumber());
    rhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, rhsArgument.getArgNumber());
  } else if (auto function =
                 llvm::dyn_cast_or_null<mlir::func::FuncOp>(owner)) {
    lhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, lhsArgument.getArgNumber());
    rhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, rhsArgument.getArgNumber());
  } else if (auto function =
                 llvm::dyn_cast_or_null<dataflow::ThreadOp>(owner)) {
    if (lhsArgument.getArgNumber() >=
            function.getFunctionType().getNumInputs() ||
        rhsArgument.getArgNumber() >= function.getFunctionType().getNumInputs())
      return false;
    lhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, lhsArgument.getArgNumber());
    rhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, rhsArgument.getArgNumber());
  } else {
    return false;
  }
  llvm::StringRef noAlias = mlir::LLVM::LLVMDialect::getNoAliasAttrName();
  return lhsAttrs && rhsAttrs && lhsAttrs.contains(noAlias) &&
         rhsAttrs.contains(noAlias);
}

bool haveDirectlyProvenDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs) {
  if (lhs == rhs)
    return false;
  if (areDistinctNoAliasFunctionArguments(lhs, rhs))
    return true;

  auto lhsGlobal = lhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto lhsAlloc = lhs.getDefiningOp<mlir::memref::AllocOp>();
  auto rhsAlloc = rhs.getDefiningOp<mlir::memref::AllocOp>();
  auto lhsStack = lhs.getDefiningOp<mlir::memref::AllocaOp>();
  auto rhsStack = rhs.getDefiningOp<mlir::memref::AllocaOp>();
  auto lhsAlloca = lhs.getDefiningOp<mlir::LLVM::AllocaOp>();
  auto rhsAlloca = rhs.getDefiningOp<mlir::LLVM::AllocaOp>();
  auto lhsAddress = lhs.getDefiningOp<mlir::LLVM::AddressOfOp>();
  auto rhsAddress = rhs.getDefiningOp<mlir::LLVM::AddressOfOp>();
  if (lhsAlloca && rhsAlloca)
    return lhsAlloca != rhsAlloca;
  if (lhsStack && rhsStack)
    return lhsStack != rhsStack;
  if (lhsAlloca)
    return rhsAlloc || rhsStack || rhsGlobal || rhsAddress ||
           isEnclosingBlockArgument(rhs, lhsAlloca);
  if (rhsAlloca)
    return lhsAlloc || lhsStack || lhsGlobal || lhsAddress ||
           isEnclosingBlockArgument(lhs, rhsAlloca);
  if (lhsStack)
    return rhsAlloc || rhsAlloca || rhsGlobal || rhsAddress ||
           isEnclosingBlockArgument(rhs, lhsStack);
  if (rhsStack)
    return lhsAlloc || lhsAlloca || lhsGlobal || lhsAddress ||
           isEnclosingBlockArgument(lhs, rhsStack);
  if (lhsAddress && rhsAddress)
    return lhsAddress.getGlobalName() != rhsAddress.getGlobalName();
  if (lhsGlobal && rhsGlobal)
    return lhsGlobal.getName() != rhsGlobal.getName();
  if (lhsAlloc && rhsAlloc)
    return lhsAlloc != rhsAlloc;
  return (lhsGlobal && (rhsAlloc || rhsStack)) ||
         ((lhsAlloc || lhsStack) && rhsGlobal) || (lhsAlloc && rhsStack) ||
         (lhsStack && rhsAlloc);
}

bool hasClosedDirectCallDomain(mlir::Operation *callable) {
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(callable)) {
    if (function.isExternal() || function.isVarArg())
      return false;
    return function.getLinkage() == mlir::LLVM::Linkage::Internal ||
           function.getLinkage() == mlir::LLVM::Linkage::Private;
  }
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(callable))
    return !function.isExternal() && function.isPrivate();
  if (auto thread = llvm::dyn_cast<dataflow::ThreadOp>(callable)) {
    auto symbol = llvm::cast<mlir::SymbolOpInterface>(thread.getOperation());
    return !thread.isExternal() &&
           symbol.getVisibility() == mlir::SymbolTable::Visibility::Private;
  }
  return false;
}

bool isFormalCallableArgument(mlir::BlockArgument argument,
                              mlir::Operation *callable) {
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(callable))
    return !function.isExternal() &&
           argument.getOwner() == &function.getBody().front() &&
           argument.getArgNumber() <
               function.getFunctionType().getParams().size();
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(callable))
    return !function.isExternal() &&
           argument.getOwner() == &function.getBody().front() &&
           argument.getArgNumber() < function.getNumArguments();
  if (auto thread = llvm::dyn_cast<dataflow::ThreadOp>(callable))
    return !thread.isExternal() &&
           argument.getOwner() == &thread.getBody().front() &&
           argument.getArgNumber() < thread.getFunctionType().getNumInputs();
  return false;
}

class CompleteCallSiteProvenance final {
public:
  explicit CompleteCallSiteProvenance(mlir::ModuleOp module)
      : users_(symbols_, module) {}

  bool proveDistinct(mlir::Value lhs, mlir::Value rhs) {
    lhs = projectMemoryRoot(lhs);
    rhs = projectMemoryRoot(rhs);
    if (haveDirectlyProvenDistinctMemoryRoots(lhs, rhs))
      return true;
    if (lhs == rhs)
      return false;

    auto lhsArgument = llvm::dyn_cast<mlir::BlockArgument>(lhs);
    auto rhsArgument = llvm::dyn_cast<mlir::BlockArgument>(rhs);
    if (!lhsArgument || !rhsArgument ||
        lhsArgument.getOwner() != rhsArgument.getOwner() ||
        lhsArgument.getArgNumber() == rhsArgument.getArgNumber())
      return false;
    mlir::Operation *callable = lhsArgument.getOwner()->getParentOp();
    if (!hasClosedDirectCallDomain(callable) ||
        !isFormalCallableArgument(lhsArgument, callable) ||
        !isFormalCallableArgument(rhsArgument, callable))
      return false;

    const std::pair<mlir::Value, mlir::Value> query{lhs, rhs};
    if (llvm::is_contained(active_, query))
      return false;
    active_.push_back(query);

    llvm::ArrayRef<mlir::Operation *> callSites = users_.getUsers(callable);
    bool proven = !callSites.empty();
    for (mlir::Operation *callSite : callSites) {
      auto actuals = exactActuals(callable, callSite);
      if (!actuals || lhsArgument.getArgNumber() >= actuals->size() ||
          rhsArgument.getArgNumber() >= actuals->size() ||
          !proveDistinct((*actuals)[lhsArgument.getArgNumber()],
                         (*actuals)[rhsArgument.getArgNumber()])) {
        proven = false;
        break;
      }
    }

    active_.pop_back();
    return proven;
  }

private:
  std::optional<mlir::ValueRange> exactActuals(mlir::Operation *callable,
                                               mlir::Operation *callSite) {
    if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(callable)) {
      auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(callSite);
      if (!call || !call.getCalleeAttr() ||
          symbols_.lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr()) != function)
        return std::nullopt;
      return call.getArgOperands();
    }
    if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(callable)) {
      auto call = llvm::dyn_cast<mlir::func::CallOp>(callSite);
      if (!call || symbols_.lookupNearestSymbolFrom<mlir::func::FuncOp>(
                       call, call.getCalleeAttr()) != function)
        return std::nullopt;
      return call.getOperands();
    }
    if (auto thread = llvm::dyn_cast<dataflow::ThreadOp>(callable)) {
      auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(callSite);
      if (!launch || symbols_.lookupNearestSymbolFrom<dataflow::ThreadOp>(
                         launch, launch.getCalleeAttr()) != thread)
        return std::nullopt;
      return launch.getBodyOperands();
    }
    return std::nullopt;
  }

  mlir::SymbolTableCollection symbols_;
  mlir::SymbolUserMap users_;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> active_;
};

} // namespace

mlir::Value projectMemoryRoot(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      value = view.getSource();
      continue;
    }
    if (auto reinterpret =
            value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      value = reinterpret.getSource();
      continue;
    }
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getOperand();
      continue;
    }
    if (auto addressSpace =
            value.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>()) {
      value = addressSpace.getOperand();
      continue;
    }
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      auto spatial = llvm::dyn_cast_or_null<loom::SpatialRegionOp>(
          argument.getOwner()->getParentOp());
      if (spatial && !spatial.getBody().empty() &&
          argument.getOwner() == &spatial.getBody().front() &&
          argument.getArgNumber() < spatial->getNumOperands()) {
        value = spatial->getOperand(argument.getArgNumber());
        continue;
      }
    }
    return value;
  }
}

bool haveProvenDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs) {
  lhs = projectMemoryRoot(lhs);
  rhs = projectMemoryRoot(rhs);
  if (haveDirectlyProvenDistinctMemoryRoots(lhs, rhs))
    return true;
  mlir::Operation *anchor =
      lhs.getParentBlock() ? lhs.getParentBlock()->getParentOp() : nullptr;
  if (!anchor)
    anchor =
        rhs.getParentBlock() ? rhs.getParentBlock()->getParentOp() : nullptr;
  mlir::ModuleOp module =
      anchor ? anchor->getParentOfType<mlir::ModuleOp>() : mlir::ModuleOp{};
  if (!module)
    return false;
  return CompleteCallSiteProvenance(module).proveDistinct(lhs, rhs);
}

} // namespace loom::raising
