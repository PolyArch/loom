#include "GraphRegionAdmission.h"

#include "Frontend/IR/LoomOps.h"
#include "GraphRegionLowering.h"
#include "GraphStreamBoundaryLowering.h"

#include "RankedMemRefLowering.h"

#include "Common/IndexWidth.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Lowering/GraphParallelLowering.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"
#include "Frontend/Analysis/PointerLoopProjection.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <limits>

namespace loom::lowering {
namespace {

using detail::analyzeStreamBoundary;
using detail::checkStreamBoundaryUses;
using detail::StreamBoundaryInfo;

bool isGraphMemoryAddressLeaf(mlir::Operation *operation) {
  return llvm::isa<mlir::memref::CastOp, mlir::memref::ViewOp,
                   mlir::memref::GetGlobalOp,
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
  if (!rejection) {
    llvm::Expected<unsigned> indexBits = loom::getIndexBitWidth(scope);
    if (!indexBits)
      rejection =
          "loom-lower-graph-memory: " + llvm::toString(indexBits.takeError());
    else
      rejection = detail::explainStreamScheduleRejection(scope, *indexBits);
  }
  return rejection;
}

::mlir::LogicalResult checkOneGraph(::dataflow::GraphOp graph,
                                    const StreamBoundaryInfo &boundary,
                                    unsigned indexBits) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0 ||
      !::llvm::isa<::mlir::NoneType>(entry.getArgument(0).getType()))
    return graph.emitError(
        "loom-lower-graph-memory: graph entry must start with none");
  if (::mlir::failed(checkStreamBoundaryUses(graph, boundary, indexBits)))
    return ::mlir::failure();

  ::mlir::WalkResult result = graph.getBody().walk([&](::mlir::Operation *op)
                                                       -> ::mlir::WalkResult {
    if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op)) {
      if (::mlir::failed(::loom::lowering::detail::checkRankedMemRefAccess(
              load, load.getMemRefType(), load.getIndices(), indexBits)))
        return ::mlir::WalkResult::interrupt();
    } else if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op)) {
      if (::mlir::failed(::loom::lowering::detail::checkRankedMemRefAccess(
              store, store.getMemRefType(), store.getIndices(), indexBits)))
        return ::mlir::WalkResult::interrupt();
    } else if (auto read =
                   ::llvm::dyn_cast<::mlir::vector::TransferReadOp>(op)) {
      if (::mlir::failed(
              ::loom::lowering::detail::checkRankedVectorTransferRead(
                  read, indexBits)))
        return ::mlir::WalkResult::interrupt();
    } else if (auto write =
                   ::llvm::dyn_cast<::mlir::vector::TransferWriteOp>(op)) {
      if (::mlir::failed(
              ::loom::lowering::detail::checkRankedVectorTransferWrite(
                  write, indexBits)))
        return ::mlir::WalkResult::interrupt();
    } else if (auto dealloc = ::llvm::dyn_cast<::mlir::memref::DeallocOp>(op)) {
      auto allocation =
          dealloc.getMemref().getDefiningOp<::mlir::memref::AllocOp>();
      if (!allocation ||
          allocation->getParentOfType<::dataflow::GraphOp>() != graph) {
        dealloc.emitOpError(
            "loom-lower-graph-memory: only graph-local allocations may be "
            "deallocated inside a graph");
        return ::mlir::WalkResult::interrupt();
      }
    }

    auto findMemoryCapability = [&](::mlir::TypeRange types) {
      for (::mlir::Type type : types)
        if (::dataflow::DataflowDialect::isMemoryCapabilityType(type))
          return type;
      return ::mlir::Type{};
    };
    if (::llvm::isa<::dataflow::CarryOp, ::dataflow::MuxOp, ::dataflow::DemuxOp,
                    ::dataflow::GateOp, ::dataflow::InvariantOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(op->getOperandTypes());
      if (!memory)
        memory = findMemoryCapability(op->getResultTypes());
      if (memory) {
        op->emitError() << "cannot lower memory capability " << memory
                        << " through " << op->getName().getStringRef();
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(ifOp.getResultTypes());
      if (memory) {
        ifOp.emitError() << "cannot lower selected memory capability " << memory
                         << " through dataflow.mux/demux";
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto switchOp =
                   ::llvm::dyn_cast<::mlir::scf::IndexSwitchOp>(op)) {
      if (switchOp.getNumCases() == 0) {
        switchOp.emitError(
            "loom-lower-graph-memory: zero-case scf.index_switch requires "
            "upstream normalization before graph-region lowering");
        return ::mlir::WalkResult::interrupt();
      }
      if (indexBits < std::numeric_limits<std::size_t>::digits &&
          switchOp.getNumCases() >= (std::size_t{1} << indexBits)) {
        switchOp.emitError(
            "loom-lower-graph-memory: scf.index_switch lane count exceeds "
            "the configured index width");
        return ::mlir::WalkResult::interrupt();
      }
      ::mlir::Type memory = findMemoryCapability(switchOp.getResultTypes());
      if (memory) {
        switchOp.emitError() << "cannot lower selected memory capability "
                             << memory << " through dataflow.mux/demux";
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
      ::mlir::Type memory =
          findMemoryCapability(forOp.getInitArgs().getTypes());
      if (memory) {
        forOp.emitError() << "cannot lower loop-carried memory capability "
                          << memory << " through dataflow.carry";
        return ::mlir::WalkResult::interrupt();
      }
      if (::mlir::failed(::loom::lowering::inferStreamStepKind(forOp))) {
        forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                        "'loom.stream_step_kind'");
        return ::mlir::WalkResult::interrupt();
      }
      if (::mlir::failed(::loom::lowering::inferStreamPredicate(forOp))) {
        forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                        "'loom.stream_predicate'");
        return ::mlir::WalkResult::interrupt();
      }
    } else if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
      ::mlir::Type memory = findMemoryCapability(whileOp.getInits().getTypes());
      if (memory) {
        whileOp.emitError() << "cannot lower loop-carried memory capability "
                            << memory << " through dataflow.carry";
        return ::mlir::WalkResult::interrupt();
      }
    }
    if (::llvm::isa<::mlir::scf::SCFDialect>(op->getDialect()) &&
        !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                     ::mlir::scf::WhileOp, ::mlir::scf::IndexSwitchOp,
                     ::mlir::scf::ParallelOp, ::mlir::scf::ForallOp,
                     ::mlir::scf::YieldOp, ::mlir::scf::ConditionOp,
                     ::mlir::scf::ReduceOp, ::mlir::scf::InParallelOp>(op)) {
      op->emitError("loom-lower-graph-memory: unsupported residual SCF "
                    "must be normalized before graph-region lowering");
      return ::mlir::WalkResult::interrupt();
    }
    bool modeled =
        ::loom::lowering::detail::isGraphRegionControlOperation(op) ||
        ::loom::lowering::classifyGraphLoweringLeaf(op) !=
            ::loom::lowering::GraphLeafLowering::Unsupported;
    if (::llvm::isa<::dataflow::ChannelSendOp, ::dataflow::ChannelReceiveOp>(
            op))
      modeled = boundary.isTransient();
    // A registered actor that no capability covers is reported for what it is,
    // so an effectful memory actor is not mistaken for an unregistered one.
    if (!modeled && ::dataflow::isCanonicalDataflowActor(op)) {
      op->emitError() << "loom-lower-graph-memory: canonical Dataflow actor '"
                      << op->getName().getStringRef()
                      << "' has no graph-region lowering";
      return ::mlir::WalkResult::interrupt();
    }
    if (!modeled && (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)) {
      op->emitError()
          << "loom-lower-graph-memory: effectful or unmodeled graph "
             "operation '"
          << op->getName().getStringRef() << "' is unsupported";
      return ::mlir::WalkResult::interrupt();
    }
    if (!modeled) {
      op->emitError()
          << "loom-lower-graph-memory: operation '"
          << op->getName().getStringRef()
          << "' is not a registered canonical Dataflow actor or a supported "
             "graph-lowering operation";
      return ::mlir::WalkResult::interrupt();
    }
    return ::mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
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
  if (auto compare = llvm::dyn_cast<mlir::LLVM::ICmpOp>(operation)) {
    auto loop = compare->getParentOfType<mlir::scf::WhileOp>();
    auto projection = frontend::analysis::projectPointerLoopTermination(loop);
    return projection && projection->comparison == compare
               ? GraphLeafLowering::Implemented
               : GraphLeafLowering::Unsupported;
  }
  const bool isEffectFree =
      mlir::isMemoryEffectFree(operation) ||
      dataflow::isCanonicalDataflowActor(
          operation, dataflow::CanonicalDataflowActorKind::Compute);
  if (operation->getNumRegions() == 0 && isEffectFree &&
      (dataflow::isCanonicalDataflowActor(operation) ||
       isGraphMemoryAddressLeaf(operation)))
    return GraphLeafLowering::Movable;
  // A pointer view is a pure reinterpretation of a Spatial pointer input. It
  // moves into the graph frontier unchanged; graph memory lowering resolves it
  // to that input's memory service.
  if (llvm::isa<loom::PointerViewOp>(operation))
    return GraphLeafLowering::Movable;
  if (llvm::isa<mlir::memref::AssumeAlignmentOp,
                mlir::memref::DistinctObjectsOp, mlir::memref::LoadOp,
                mlir::memref::StoreOp, mlir::vector::TransferReadOp,
                mlir::vector::TransferWriteOp, mlir::memref::DeallocOp,
                dataflow::LoadOp, dataflow::StoreOp, dataflow::AtomicRmwOp,
                dataflow::CmpXchgOp, dataflow::FenceOp, dataflow::ChannelSendOp,
                dataflow::ChannelReceiveOp>(operation))
    return GraphLeafLowering::Implemented;
  if (detail::isGraphRegionRepresentationBitcast(operation))
    return GraphLeafLowering::Implemented;
  if (llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                mlir::LLVM::AtomicRMWOp, mlir::LLVM::AtomicCmpXchgOp,
                mlir::LLVM::FenceOp, mlir::LLVM::MemcpyOp,
                mlir::LLVM::MemmoveOp, mlir::LLVM::MemsetOp>(operation))
    return GraphLeafLowering::Implemented;
  // Static LLVM stack objects are normalized by graph-memory lowering before
  // structured regions are flattened. Unsupported dynamic or aggregate forms
  // fail at that owner with a typed diagnostic.
  if (llvm::isa<mlir::LLVM::AllocaOp>(operation))
    return GraphLeafLowering::Implemented;
  if (llvm::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp>(
          operation))
    return GraphLeafLowering::Implemented;
  if (llvm::isa<mlir::memref::AllocOp>(operation))
    return isGraphFrontier(operation->getBlock())
               ? GraphLeafLowering::Implemented
               : GraphLeafLowering::Unsupported;
  return GraphLeafLowering::Unsupported;
}

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module) {
  ::llvm::SmallVector<::mlir::Operation *, 8> parallelOps;
  module.walk([&](::mlir::Operation *op) {
    if (::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op) &&
        op->getParentOfType<::dataflow::GraphOp>())
      parallelOps.push_back(op);
  });
  if (::mlir::failed(checkGraphOwnedParallelPreconditions(parallelOps)))
    return ::mlir::failure();

  ::mlir::WalkResult result =
      module.walk([&](::dataflow::GraphOp graph) -> ::mlir::WalkResult {
        if (graph.isExternal())
          return ::mlir::WalkResult::advance();
        ::llvm::Expected<unsigned> indexBits = ::loom::getIndexBitWidth(graph);
        if (!indexBits) {
          graph.emitError("loom-lower-graph-memory: ")
              << ::llvm::toString(indexBits.takeError());
          return ::mlir::WalkResult::interrupt();
        }
        auto boundary = analyzeStreamBoundary(graph);
        if (::mlir::failed(boundary) ||
            ::mlir::failed(checkOneGraph(graph, *boundary, *indexBits)))
          return ::mlir::WalkResult::interrupt();
        return ::mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
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
