#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "StructuredAddressIndexNarrowing.h"

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "ownership_candidate_invalid: " + message);
}

std::string typeSpelling(mlir::FunctionType type) {
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  stream << type;
  return spelling;
}

std::string uniqueSymbol(mlir::ModuleOp module, llvm::StringRef prefix,
                         llvm::StringRef sourceName) {
  std::string base = (llvm::Twine(prefix) + sourceName).str();
  std::string candidate = base;
  for (std::uint64_t suffix = 0; module.lookupSymbol(candidate); ++suffix)
    candidate = (llvm::Twine(base) + "_" + llvm::Twine(suffix)).str();
  return candidate;
}

llvm::Error verifyEligibleCallable(mlir::LLVM::LLVMFuncOp function) {
  if (function.isExternal())
    return invalid("selected callable has no definition");
  if (function.isVarArg())
    return invalid("variadic callable ownership is not materialized");
  if (!llvm::isa<mlir::LLVM::LLVMVoidType>(
          function.getFunctionType().getReturnType()))
    return invalid("whole-callable ownership currently requires void return");
  if (!function.getBody().hasOneBlock())
    return invalid("whole-callable ownership requires one structured block");

  mlir::Block &body = function.getBody().front();
  auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 0)
    return invalid("selected callable must return void directly");

  mlir::Operation *nestedCall = nullptr;
  function.getBody().walk([&](mlir::Operation *operation) {
    if (llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::InvokeOp>(operation)) {
      nestedCall = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (nestedCall)
    return invalid("selected callable contains an unresolved nested call");
  return llvm::Error::success();
}

/// One new private rank-zero dataflow.thread holding one loom.spatial_region.
/// The captured source values become the thread's function inputs, partitioned
/// into the region's ordinary values and memory capabilities.
struct SpatialThreadBoundary {
  dataflow::ThreadOp thread;
  mlir::Block *spatialEntry;
  llvm::SmallVector<mlir::Value> captureArguments;
};

llvm::Expected<SpatialThreadBoundary> createSpatialThreadBoundary(
    mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp sourceCallable,
    mlir::ValueRange captures, mlir::Location location) {
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  const std::string threadName =
      uniqueSymbol(module, "__loom_thread_", sourceCallable.getSymName());
  const std::string graphName =
      uniqueSymbol(module, "__loom_graph_", sourceCallable.getSymName());

  builder.setInsertionPointToEnd(module.getBody());
  auto thread = dataflow::ThreadOp::create(
      builder, location, threadName,
      builder.getFunctionType(captures.getTypes(), mlir::TypeRange{}),
      dataflow::ThreadDomainAttr::get(context));
  thread.setSymVisibilityAttr(builder.getStringAttr("private"));

  llvm::SmallVector<mlir::DictionaryAttr, 8> argumentAttrs;
  argumentAttrs.reserve(captures.size());
  for (mlir::Value capture : captures) {
    mlir::DictionaryAttr attrs = mlir::DictionaryAttr::get(context);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(capture))
      if (argument.getOwner() == &sourceCallable.getBody().front())
        attrs = mlir::function_interface_impl::getArgAttrDict(
            sourceCallable, argument.getArgNumber());
    argumentAttrs.push_back(attrs);
  }
  mlir::function_interface_impl::setAllArgAttrDicts(thread, argumentAttrs);

  mlir::Block *threadEntry = builder.createBlock(&thread.getBody());
  for (mlir::Type type : captures.getTypes())
    threadEntry->addArgument(type, location);
  threadEntry->addArgument(builder.getNoneType(), location);

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Value, 8> memories;
  llvm::SmallVector<std::size_t, 8> spatialArgument(captures.size());
  for (auto [index, argument] : llvm::enumerate(
           threadEntry->getArguments().take_front(captures.size()))) {
    if (dataflow::DataflowDialect::isMemoryCapabilityType(argument.getType())) {
      spatialArgument[index] = memories.size();
      memories.push_back(argument);
      continue;
    }
    if (dataflow::DataflowDialect::containsMemoryCapability(argument.getType()))
      return invalid(
          "captured input embeds an unmaterialized memory capability");
    spatialArgument[index] = values.size();
    values.push_back(argument);
  }
  for (std::size_t index = 0; index < captures.size(); ++index)
    if (dataflow::DataflowDialect::isMemoryCapabilityType(
            captures[index].getType()))
      spatialArgument[index] += values.size();

  builder.setInsertionPointToStart(threadEntry);
  auto spatial = loom::SpatialRegionOp::create(
      builder, location, values, mlir::ValueRange{}, memories,
      mlir::ValueRange{}, mlir::TypeRange{}, mlir::TypeRange{},
      builder.getArrayAttr({}), builder.getStringAttr(graphName));
  mlir::Block *spatialEntry = builder.createBlock(&spatial.getBody());
  for (mlir::Type type : spatial.getOperandTypes())
    spatialEntry->addArgument(type, location);

  builder.setInsertionPointToEnd(threadEntry);
  dataflow::ThreadYieldOp::create(builder, location, mlir::ValueRange{});

  llvm::SmallVector<mlir::Value, 8> captureArguments;
  captureArguments.reserve(captures.size());
  for (std::size_t index = 0; index < captures.size(); ++index)
    captureArguments.push_back(
        spatialEntry->getArgument(spatialArgument[index]));
  return SpatialThreadBoundary{thread, spatialEntry,
                               std::move(captureArguments)};
}

llvm::Error materializeThread(mlir::ModuleOp module,
                              mlir::LLVM::LLVMFuncOp function) {
  if (llvm::Error error = verifyEligibleCallable(function))
    return error;
  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = function.getLoc();
  mlir::Block &source = function.getBody().front();
  auto boundary = createSpatialThreadBoundary(module, function,
                                              source.getArguments(), location);
  if (!boundary)
    return boundary.takeError();

  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, argument] : llvm::enumerate(source.getArguments()))
    mapping.map(argument, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::Operation &operation : source.without_terminator())
    builder.clone(operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  while (!source.empty())
    source.back().erase();
  builder.setInsertionPointToStart(&source);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, source.getArguments(), mlir::ValueRange{},
      mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  mlir::LLVM::ReturnOp::create(builder, location, mlir::ValueRange{});
  return llvm::Error::success();
}

/// Returns the ordinary LLVM callable that owns the selected operation, after
/// proving the operation is an ownership-free structured entity: inside an
/// LLVM callable, outside every dataflow.thread, dataflow.graph, and
/// loom.spatial_region, with at least one region and no SSA result used
/// outside itself.
llvm::Expected<mlir::LLVM::LLVMFuncOp>
eligibleOwningCallable(mlir::Operation *operation) {
  mlir::LLVM::LLVMFuncOp callable;
  for (mlir::Operation *ancestor = operation->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    if (llvm::isa<dataflow::ThreadOp, dataflow::GraphOp, loom::SpatialRegionOp>(
            ancestor))
      return invalid("selected operation already has an execution owner");
    if (auto candidate = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(ancestor)) {
      callable = candidate;
      break;
    }
  }
  if (!callable)
    return invalid(
        "selected operation is not inside an ordinary LLVM callable");
  if (operation->getNumRegions() == 0)
    return invalid("selected operation owns no region");
  for (mlir::OpResult result : operation->getResults())
    for (mlir::Operation *user : result.getUsers())
      if (!operation->isProperAncestor(user))
        return invalid(
            "selected operation has an SSA result used outside itself");
  return callable;
}

/// Derives every external SSA live-in used by the selected operation itself or
/// recursively inside it, exactly once, in deterministic first-use order.
llvm::SmallVector<mlir::Value> externalLiveIns(mlir::Operation *operation) {
  llvm::SmallVector<mlir::Value> liveIns;
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  operation->walk([&](mlir::Operation *nested) {
    for (mlir::Value operand : nested->getOperands()) {
      if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        if (operation->isAncestor(argument.getOwner()->getParentOp()))
          continue;
      } else if (operation->isAncestor(operand.getDefiningOp())) {
        continue;
      }
      if (seen.insert(operand).second)
        liveIns.push_back(operand);
    }
    return mlir::WalkResult::advance();
  });
  return liveIns;
}

llvm::Error
materializeSelectedOperation(mlir::ModuleOp module, mlir::Operation *operation,
                             std::optional<unsigned> canonicalIndexWidth) {
  auto callable = eligibleOwningCallable(operation);
  if (!callable)
    return callable.takeError();
  if (llvm::Error error = detail::materializeAddressIndexContract(
          module, operation, canonicalIndexWidth))
    return error;
  const llvm::SmallVector<mlir::Value> liveIns = externalLiveIns(operation);
  mlir::Location location = operation->getLoc();
  auto boundary =
      createSpatialThreadBoundary(module, *callable, liveIns, location);
  if (!boundary)
    return boundary.takeError();

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, liveIn] : llvm::enumerate(liveIns))
    mapping.map(liveIn, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  builder.clone(*operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  builder.setInsertionPoint(operation);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  auto launch =
      dataflow::ThreadLaunchOp::create(builder, location, callee, liveIns,
                                       mlir::ValueRange{}, mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  operation->erase();
  return llvm::Error::success();
}

llvm::Error requireExactFabricCapabilities(
    const dataflow::CanonicalDataflowArtifact &program,
    const fabric::FinalizedFabricRoot &fabric) {
  auto view = program.view();
  if (!view)
    return view.takeError();
  if (view->graphs().empty() || view->actors().empty())
    return invalid("materialized candidate has no SpatialCore workload");

  FabricCapabilityIndex capabilities(fabric.view());
  for (const dataflow::CanonicalActorView &actor : view->actors()) {
    auto projection =
        dataflow::projectRegisteredActorSchemaProjection(actor.op);
    if (!projection)
      return projection.takeError();
    if (actor.kind == dataflow::CanonicalDataflowActorKind::Memory) {
      auto resources = capabilities.admittingMemoryResources(actor.op);
      if (!resources)
        return resources.takeError();
      if (resources->empty())
        return invalid("exact Fabric admits no memory resource for actor " +
                       dataflow::operationSchemaSpelling(projection->schema));
      continue;
    }
    auto resources = capabilities.admittingOperationResources(actor.op);
    if (!resources)
      return resources.takeError();
    if (resources->empty())
      return invalid("exact Fabric admits no operation resource for actor " +
                     dataflow::operationSchemaSpelling(projection->schema) +
                     " with type " + typeSpelling(projection->type));
  }
  return llvm::Error::success();
}

/// Resolves the exact parent-local selection, clones the complete parent
/// candidate, and resolves the same reference in the clone.
struct PrivateSelection {
  mlir::OwningOpRef<mlir::ModuleOp> clone;
  mlir::Operation *operation;
};

llvm::Expected<PrivateSelection>
cloneSelectedOperation(const StructuredProgramCandidate &parent,
                       const StructuredEntityRef &selection) {
  auto parentView = parent.view();
  if (!parentView)
    return parentView.takeError();
  auto parentEntity = parentView->resolve(selection);
  if (!parentEntity)
    return parentEntity.takeError();
  if (!parentEntity->operation)
    return invalid("selected StructuredEntityRef is not an operation");

  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone()));
  auto cloneView =
      buildStructuredProgramCandidateView(clone.get(), parent.identity());
  if (!cloneView)
    return cloneView.takeError();
  auto clonedEntity = cloneView->resolve(selection);
  if (!clonedEntity)
    return clonedEntity.takeError();
  if (!clonedEntity->operation)
    return invalid("selected operation changed kind in the private clone");
  return PrivateSelection{std::move(clone), clonedEntity->operation};
}

llvm::Expected<MaterializedOwnershipCandidate> finalizeOwnershipCandidate(
    mlir::ModuleOp module, const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &loweringOptions) {
  if (mlir::failed(mlir::verify(module)))
    return invalid("materialized Structured Program does not verify");
  auto structured = finalizeStructuredProgram(module);
  if (!structured)
    return structured.takeError();
  auto canonical = lowering::lowerStructuredProgramToCanonicalDataflow(
      *structured, loweringOptions);
  if (!canonical)
    return canonical.takeError();
  if (llvm::Error error = requireExactFabricCapabilities(*canonical, fabric))
    return std::move(error);
  return MaterializedOwnershipCandidate{std::move(*structured),
                                        std::move(*canonical)};
}

} // namespace

llvm::Expected<MaterializedOwnershipCandidate>
materializeWholeCallableSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &callable,
    const fabric::FinalizedFabricRoot &fabric,
    const WholeCallableSpatialOwnershipOptions &options) {
  auto selection = cloneSelectedOperation(parent, callable);
  if (!selection)
    return selection.takeError();
  auto function =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(selection->operation);
  if (!function)
    return invalid("selected StructuredEntityRef is not an LLVM callable");

  if (options.fmuladdExecutionShape) {
    mlir::PassManager materialization(
        function.getContext(), mlir::LLVM::LLVMFuncOp::getOperationName());
    materialization.enableVerifier(options.lowering.verifyEach);
    materialization.addPass(
        raising::createMaterializeFMulAddPass(*options.fmuladdExecutionShape));
    if (mlir::failed(materialization.run(function.getOperation())))
      return invalid("selected fmuladd execution shape is not materializable");
  }

  if (llvm::Error error = materializeThread(selection->clone.get(), function))
    return std::move(error);
  return finalizeOwnershipCandidate(selection->clone.get(), fabric,
                                    options.lowering);
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeOperationSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &operation,
    const fabric::FinalizedFabricRoot &fabric,
    const OperationSpatialOwnershipOptions &options) {
  auto selection = cloneSelectedOperation(parent, operation);
  if (!selection)
    return selection.takeError();
  if (llvm::Error error = materializeSelectedOperation(
          selection->clone.get(), selection->operation,
          options.canonicalIndexWidth))
    return std::move(error);
  return finalizeOwnershipCandidate(selection->clone.get(), fabric,
                                    options.lowering);
}

} // namespace loom::frontend
