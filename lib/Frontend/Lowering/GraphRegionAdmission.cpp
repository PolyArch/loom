#include "GraphRegionAdmission.h"
#include "GraphRegionLowering.h"

#include "Frontend/Lowering/CanonicalDataflowLowering.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace loom::lowering {
namespace {

bool isGraphMemoryAddressLeaf(mlir::Operation *operation) {
  return llvm::isa<mlir::memref::CastOp, mlir::memref::GetGlobalOp,
                   mlir::LLVM::AddressOfOp, mlir::LLVM::GEPOp>(operation);
}

bool isGraphFrontier(mlir::Block *block) {
  auto graph = llvm::dyn_cast_or_null<dataflow::GraphOp>(block->getParentOp());
  return graph && block == &graph.getBody().front();
}

bool isProjectedGraphFrontier(mlir::Operation *scope, mlir::Operation *leaf) {
  if (!scope || !leaf)
    return false;
  if (isGraphFrontier(leaf->getBlock()) || scope == leaf)
    return true;
  auto callable = llvm::dyn_cast<mlir::FunctionOpInterface>(scope);
  return callable && !callable.getFunctionBody().empty() &&
         leaf->getBlock() == &callable.getFunctionBody().front();
}

std::optional<std::string>
explainGraphRegionStructuralRejectionImpl(mlir::Operation *scope,
                                          mlir::Operation *deferredLeaf) {
  if (!scope)
    return std::string("missing graph-region scope");
  if (deferredLeaf && !scope->isAncestor(deferredLeaf))
    return std::string("deferred graph-region leaf is outside the scope");

  const bool callableRoot = llvm::isa<mlir::FunctionOpInterface>(scope);
  std::optional<std::string> rejection;
  scope->walk([&](mlir::Operation *operation) {
    if (operation != scope && llvm::isa<mlir::FunctionOpInterface>(operation))
      return mlir::WalkResult::skip();
    if (operation == deferredLeaf || (operation == scope && callableRoot) ||
        (callableRoot && llvm::isa<mlir::LLVM::ReturnOp>(operation)) ||
        (callableRoot && llvm::isa<mlir::LLVM::UndefOp>(operation)) ||
        llvm::isa<mlir::LLVM::FMulAddOp>(operation) ||
        (llvm::isa<mlir::memref::AllocOp>(operation) &&
         isProjectedGraphFrontier(scope, operation)) ||
        detail::isGraphRegionControlOperation(operation) ||
        classifyGraphLoweringLeaf(operation) != GraphLeafLowering::Unsupported)
      return mlir::WalkResult::advance();

    rejection = ("operation '" + operation->getName().getStringRef() +
                 "' has no graph-region lowering")
                    .str();
    return mlir::WalkResult::interrupt();
  });
  return rejection;
}

} // namespace

namespace detail {

bool isGraphRegionControlOperation(mlir::Operation *operation) {
  return llvm::isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp,
                   mlir::scf::IndexSwitchOp, mlir::scf::ParallelOp,
                   mlir::scf::ForallOp, mlir::scf::YieldOp,
                   mlir::scf::ConditionOp, mlir::scf::ReduceOp,
                   mlir::scf::InParallelOp, dataflow::GraphReturnOp>(operation);
}

bool isGraphRegionRepresentationBitcast(mlir::Operation *operation) {
  auto bitcast = llvm::dyn_cast_or_null<mlir::LLVM::BitcastOp>(operation);
  if (!bitcast)
    return false;
  mlir::Type input = bitcast.getArg().getType();
  mlir::Type result = bitcast.getRes().getType();
  return (llvm::isa<mlir::VectorType>(input) &&
          llvm::isa<mlir::IntegerType>(result)) ||
         (llvm::isa<mlir::IntegerType>(input) &&
          llvm::isa<mlir::VectorType>(result));
}

} // namespace detail

GraphLeafLowering classifyGraphLoweringLeaf(mlir::Operation *operation) {
  const bool isEffectFree =
      mlir::isMemoryEffectFree(operation) ||
      dataflow::isCanonicalDataflowActor(
          operation, dataflow::CanonicalDataflowActorKind::Compute);
  if (operation->getNumRegions() == 0 && isEffectFree &&
      (dataflow::isCanonicalDataflowActor(operation) ||
       isGraphMemoryAddressLeaf(operation)))
    return GraphLeafLowering::Movable;
  if (llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                mlir::memref::DeallocOp, dataflow::LoadOp, dataflow::StoreOp,
                dataflow::ChannelSendOp, dataflow::ChannelReceiveOp>(operation))
    return GraphLeafLowering::Implemented;
  if (detail::isGraphRegionRepresentationBitcast(operation))
    return GraphLeafLowering::Implemented;
  if (llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp, mlir::LLVM::MemcpyOp,
                mlir::LLVM::MemmoveOp, mlir::LLVM::MemsetOp>(operation))
    return GraphLeafLowering::Implemented;
  if (llvm::isa<mlir::memref::AllocOp>(operation))
    return isGraphFrontier(operation->getBlock())
               ? GraphLeafLowering::Implemented
               : GraphLeafLowering::Unsupported;
  return GraphLeafLowering::Unsupported;
}

std::optional<std::string>
explainGraphRegionStructuralRejection(mlir::Operation *scope) {
  return explainGraphRegionStructuralRejectionImpl(scope, nullptr);
}

std::optional<std::string>
explainGraphRegionStructuralRejection(mlir::Operation *scope,
                                      mlir::Operation *deferredLeaf) {
  return explainGraphRegionStructuralRejectionImpl(scope, deferredLeaf);
}

} // namespace loom::lowering
