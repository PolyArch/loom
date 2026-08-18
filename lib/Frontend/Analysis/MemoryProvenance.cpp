#include "Frontend/Analysis/MemoryProvenance.h"

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

namespace loom::frontend::analysis {
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
  } else if (auto function =
                 llvm::dyn_cast_or_null<dataflow::GraphOp>(owner)) {
    if (lhsArgument.getArgNumber() == 0 ||
        rhsArgument.getArgNumber() == 0 ||
        lhsArgument.getArgNumber() >
            function.getFunctionType().getNumInputs() ||
        rhsArgument.getArgNumber() >
            function.getFunctionType().getNumInputs())
      return false;
    lhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, lhsArgument.getArgNumber() - 1);
    rhsAttrs = mlir::function_interface_impl::getArgAttrDict(
        function, rhsArgument.getArgNumber() - 1);
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
  if (auto graph = llvm::dyn_cast<dataflow::GraphOp>(callable)) {
    auto symbol = llvm::cast<mlir::SymbolOpInterface>(graph.getOperation());
    return !graph.isExternal() &&
           symbol.getVisibility() == mlir::SymbolTable::Visibility::Private;
  }
  return false;
}

std::optional<unsigned>
formalCallableArgumentOrdinal(mlir::BlockArgument argument,
                              mlir::Operation *callable) {
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(callable))
    if (!function.isExternal() &&
        argument.getOwner() == &function.getBody().front() &&
        argument.getArgNumber() <
            function.getFunctionType().getParams().size())
      return argument.getArgNumber();
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(callable))
    if (!function.isExternal() &&
        argument.getOwner() == &function.getBody().front() &&
        argument.getArgNumber() < function.getNumArguments())
      return argument.getArgNumber();
  if (auto thread = llvm::dyn_cast<dataflow::ThreadOp>(callable))
    if (!thread.isExternal() &&
        argument.getOwner() == &thread.getBody().front() &&
        argument.getArgNumber() < thread.getFunctionType().getNumInputs())
      return argument.getArgNumber();
  if (auto graph = llvm::dyn_cast<dataflow::GraphOp>(callable))
    if (!graph.isExternal() &&
        argument.getOwner() == &graph.getBody().front() &&
        argument.getArgNumber() > 0 &&
        argument.getArgNumber() <= graph.getFunctionType().getNumInputs())
      return argument.getArgNumber() - 1;
  return std::nullopt;
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

    std::optional<FormalRoot> lhsFormal = formalRoot(lhs);
    std::optional<FormalRoot> rhsFormal = formalRoot(rhs);
    if (!lhsFormal && !rhsFormal)
      return false;

    const std::pair<mlir::Value, mlir::Value> query{lhs, rhs};
    const std::pair<mlir::Value, mlir::Value> reverseQuery{rhs, lhs};
    if (llvm::is_contained(active_, query) ||
        llvm::is_contained(active_, reverseQuery))
      return false;
    active_.push_back(query);

    bool proven = false;
    if (lhsFormal && rhsFormal &&
        lhsFormal->callable == rhsFormal->callable)
      proven = proveAlignedFormals(*lhsFormal, *rhsFormal);
    else if (lhsFormal)
      proven = proveFormalAgainst(*lhsFormal, rhs);
    else
      proven = proveFormalAgainst(*rhsFormal, lhs);

    active_.pop_back();
    return proven;
  }

private:
  struct FormalRoot final {
    mlir::Operation *callable;
    unsigned ordinal;
  };

  std::optional<FormalRoot> formalRoot(mlir::Value value) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (!argument)
      return std::nullopt;
    mlir::Operation *callable = argument.getOwner()->getParentOp();
    if (!hasClosedDirectCallDomain(callable))
      return std::nullopt;
    std::optional<unsigned> ordinal =
        formalCallableArgumentOrdinal(argument, callable);
    if (!ordinal)
      return std::nullopt;
    return FormalRoot{callable, *ordinal};
  }

  bool proveAlignedFormals(const FormalRoot &lhs, const FormalRoot &rhs) {
    llvm::ArrayRef<mlir::Operation *> callSites = users_.getUsers(lhs.callable);
    if (callSites.empty())
      return false;
    for (mlir::Operation *callSite : callSites) {
      auto actuals = exactActuals(lhs.callable, callSite);
      if (!actuals || lhs.ordinal >= actuals->size() ||
          rhs.ordinal >= actuals->size() ||
          !proveDistinct((*actuals)[lhs.ordinal], (*actuals)[rhs.ordinal]))
        return false;
    }
    return true;
  }

  bool proveFormalAgainst(const FormalRoot &formal, mlir::Value other) {
    llvm::ArrayRef<mlir::Operation *> callSites = users_.getUsers(formal.callable);
    if (callSites.empty())
      return false;
    for (mlir::Operation *callSite : callSites) {
      auto actuals = exactActuals(formal.callable, callSite);
      if (!actuals || formal.ordinal >= actuals->size() ||
          !proveDistinct((*actuals)[formal.ordinal], other))
        return false;
    }
    return true;
  }

  std::optional<llvm::SmallVector<mlir::Value, 8>>
  exactActuals(mlir::Operation *callable, mlir::Operation *callSite) {
    if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(callable)) {
      auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(callSite);
      if (!call || !call.getCalleeAttr() ||
          symbols_.lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr()) != function)
        return std::nullopt;
      return llvm::SmallVector<mlir::Value, 8>(call.getArgOperands());
    }
    if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(callable)) {
      auto call = llvm::dyn_cast<mlir::func::CallOp>(callSite);
      if (!call || symbols_.lookupNearestSymbolFrom<mlir::func::FuncOp>(
                       call, call.getCalleeAttr()) != function)
        return std::nullopt;
      return llvm::SmallVector<mlir::Value, 8>(call.getOperands());
    }
    if (auto thread = llvm::dyn_cast<dataflow::ThreadOp>(callable)) {
      auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(callSite);
      if (!launch || symbols_.lookupNearestSymbolFrom<dataflow::ThreadOp>(
                         launch, launch.getCalleeAttr()) != thread)
        return std::nullopt;
      return llvm::SmallVector<mlir::Value, 8>(launch.getBodyOperands());
    }
    if (auto graph = llvm::dyn_cast<dataflow::GraphOp>(callable)) {
      auto launch = llvm::dyn_cast<dataflow::GraphLaunchOp>(callSite);
      if (!launch || symbols_.lookupNearestSymbolFrom<dataflow::GraphOp>(
                         launch, launch.getCalleeAttr()) != graph)
        return std::nullopt;
      llvm::SmallVector<mlir::Value, 8> actuals;
      actuals.append(launch.getValueInputs().begin(),
                     launch.getValueInputs().end());
      actuals.append(launch.getStreamInputs().begin(),
                     launch.getStreamInputs().end());
      actuals.append(launch.getMemoryInputs().begin(),
                     launch.getMemoryInputs().end());
      return actuals;
    }
    return std::nullopt;
  }

  mlir::SymbolTableCollection symbols_;
  mlir::SymbolUserMap users_;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 4> active_;
};

} // namespace

mlir::Value projectMemoryDerivationRoot(mlir::Value value) {
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
    return value;
  }
}

mlir::Value projectMemoryRoot(mlir::Value value) {
  while (true) {
    value = projectMemoryDerivationRoot(value);
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

} // namespace loom::frontend::analysis
