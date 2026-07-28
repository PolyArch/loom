#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "StructuredAddressIndexNarrowing.h"

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

namespace loom::frontend {
char SpatialOwnershipCandidateRejection::ID = 0;

void SpatialOwnershipCandidateRejection::log(llvm::raw_ostream &stream) const {
  switch (kind_) {
  case SpatialOwnershipCandidateRejectionKind::NonFinalizable:
    stream << "ownership_candidate_non_finalizable: ";
    break;
  case SpatialOwnershipCandidateRejectionKind::ExactFabricInadmissible:
    stream << "ownership_candidate_exact_fabric_inadmissible: ";
    break;
  }
  stream << message_;
}

std::error_code SpatialOwnershipCandidateRejection::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "ownership_candidate_invalid: " + message);
}

llvm::Error reject(SpatialOwnershipCandidateRejectionKind kind,
                   const llvm::Twine &message) {
  return llvm::make_error<SpatialOwnershipCandidateRejection>(kind,
                                                              message.str());
}

llvm::Error reject(SpatialOwnershipCandidateRejectionKind kind,
                   llvm::Error error) {
  return reject(kind, llvm::toString(std::move(error)));
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
  loom::SpatialRegionOp spatial;
  mlir::Block *spatialEntry;
  llvm::SmallVector<mlir::Value> captureArguments;
  llvm::SmallVector<mlir::Value> threadOnlyArguments;
};

llvm::Expected<SpatialThreadBoundary> createSpatialThreadBoundary(
    mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp sourceCallable,
    mlir::ValueRange captures, mlir::ValueRange threadOnlyCaptures,
    mlir::TypeRange valueResultTypes, mlir::TypeRange memoryResultTypes,
    mlir::Location location) {
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  const std::string threadName =
      uniqueSymbol(module, "__loom_thread_", sourceCallable.getSymName());
  const std::string graphName =
      uniqueSymbol(module, "__loom_graph_", sourceCallable.getSymName());

  builder.setInsertionPointToEnd(module.getBody());
  llvm::SmallVector<mlir::Type, 8> threadInputTypes(captures.getTypes());
  llvm::append_range(threadInputTypes, threadOnlyCaptures.getTypes());
  auto thread = dataflow::ThreadOp::create(
      builder, location, threadName,
      builder.getFunctionType(threadInputTypes, mlir::TypeRange{}),
      dataflow::ThreadDomainAttr::get(context));
  thread.setSymVisibilityAttr(builder.getStringAttr("private"));

  llvm::SmallVector<mlir::DictionaryAttr, 8> argumentAttrs;
  argumentAttrs.reserve(captures.size() + threadOnlyCaptures.size());
  for (mlir::Value capture : captures) {
    mlir::DictionaryAttr attrs = mlir::DictionaryAttr::get(context);
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(capture))
      if (argument.getOwner() == &sourceCallable.getBody().front())
        attrs = mlir::function_interface_impl::getArgAttrDict(
            sourceCallable, argument.getArgNumber());
    argumentAttrs.push_back(attrs);
  }
  for ([[maybe_unused]] mlir::Value capture : threadOnlyCaptures)
    argumentAttrs.push_back(mlir::DictionaryAttr::get(context));
  mlir::function_interface_impl::setAllArgAttrDicts(thread, argumentAttrs);

  mlir::Block *threadEntry = builder.createBlock(&thread.getBody());
  for (mlir::Type type : threadInputTypes)
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
      mlir::ValueRange{}, valueResultTypes, memoryResultTypes,
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
  llvm::SmallVector<mlir::Value, 4> threadOnlyArguments;
  threadOnlyArguments.reserve(threadOnlyCaptures.size());
  for (std::size_t index = 0; index < threadOnlyCaptures.size(); ++index)
    threadOnlyArguments.push_back(
        threadEntry->getArgument(captures.size() + index));
  return SpatialThreadBoundary{thread, spatial, spatialEntry,
                               std::move(captureArguments),
                               std::move(threadOnlyArguments)};
}

llvm::Error materializeThread(mlir::ModuleOp module,
                              mlir::LLVM::LLVMFuncOp function) {
  if (llvm::Error error = verifyEligibleCallable(function))
    return error;
  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = function.getLoc();
  mlir::Block &source = function.getBody().front();

  // A global address is resolved by the stored-program launch owner and
  // crosses the thread ABI as a memory capability. It is not a SpatialCore
  // actor. Hoist nested address leaves to the wrapper entry so every retained
  // leaf has one explicit launch binding and can be removed from the cloned
  // graph body.
  llvm::SmallVector<mlir::LLVM::AddressOfOp, 4> addressCaptures;
  llvm::SmallVector<mlir::LLVM::UndefOp, 4> undefCaptures;
  function.walk([&](mlir::LLVM::AddressOfOp address) {
    if (!address.getRes().use_empty())
      addressCaptures.push_back(address);
  });
  function.walk([&](mlir::LLVM::UndefOp undef) {
    if (!undef.getRes().use_empty())
      undefCaptures.push_back(undef);
  });
  for (mlir::LLVM::AddressOfOp address : llvm::reverse(addressCaptures))
    if (&source.front() != address.getOperation())
      address->moveBefore(&source, source.begin());
  for (mlir::LLVM::UndefOp undef : llvm::reverse(undefCaptures))
    if (&source.front() != undef.getOperation())
      undef->moveBefore(&source, source.begin());

  llvm::SmallVector<mlir::Value, 8> captures(source.getArguments().begin(),
                                             source.getArguments().end());
  for (mlir::LLVM::AddressOfOp address : addressCaptures)
    captures.push_back(address.getRes());
  for (mlir::LLVM::UndefOp undef : undefCaptures)
    captures.push_back(undef.getRes());
  auto boundary = createSpatialThreadBoundary(
      module, function, captures, mlir::ValueRange{}, mlir::TypeRange{},
      mlir::TypeRange{}, location);
  if (!boundary)
    return boundary.takeError();

  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, capture] : llvm::enumerate(captures))
    mapping.map(capture, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::Operation &operation : source.without_terminator())
    if (!llvm::isa<mlir::LLVM::AddressOfOp, mlir::LLVM::UndefOp>(operation))
      builder.clone(operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  llvm::SmallPtrSet<mlir::Operation *, 8> retainedBoundarySources;
  for (mlir::LLVM::AddressOfOp address : addressCaptures)
    retainedBoundarySources.insert(address.getOperation());
  for (mlir::LLVM::UndefOp undef : undefCaptures)
    retainedBoundarySources.insert(undef.getOperation());
  llvm::SmallVector<mlir::Operation *, 16> oldBody;
  for (mlir::Operation &operation : source)
    oldBody.push_back(&operation);
  for (mlir::Operation *operation : llvm::reverse(oldBody))
    if (!retainedBoundarySources.contains(operation))
      operation->erase();

  builder.setInsertionPointToEnd(&source);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  auto launch =
      dataflow::ThreadLaunchOp::create(builder, location, callee, captures,
                                       mlir::ValueRange{}, mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  mlir::LLVM::ReturnOp::create(builder, location, mlir::ValueRange{});
  return llvm::Error::success();
}

enum class OwnershipScopeRejection {
  None,
  AlreadyOwned,
  NoOrdinaryCallable,
  NoRegion,
};

struct OwnershipScopeAnalysis {
  mlir::LLVM::LLVMFuncOp callable;
  OwnershipScopeRejection rejection = OwnershipScopeRejection::None;
};

OwnershipScopeAnalysis analyzeOwnershipScope(mlir::Operation *operation) {
  mlir::LLVM::LLVMFuncOp callable;
  for (mlir::Operation *ancestor = operation->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    if (llvm::isa<dataflow::ThreadOp, dataflow::GraphOp, loom::SpatialRegionOp>(
            ancestor))
      return {nullptr, OwnershipScopeRejection::AlreadyOwned};
    if (auto candidate = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(ancestor)) {
      callable = candidate;
      break;
    }
  }
  if (!callable)
    return {nullptr, OwnershipScopeRejection::NoOrdinaryCallable};
  if (operation->getNumRegions() == 0)
    return {nullptr, OwnershipScopeRejection::NoRegion};
  return {callable, OwnershipScopeRejection::None};
}

/// Returns the ordinary LLVM callable that owns the selected operation after
/// proving that the operation is an ownership-free structured entity.
llvm::Expected<mlir::LLVM::LLVMFuncOp>
eligibleOwningCallable(mlir::Operation *operation) {
  OwnershipScopeAnalysis analysis = analyzeOwnershipScope(operation);
  switch (analysis.rejection) {
  case OwnershipScopeRejection::None:
    break;
  case OwnershipScopeRejection::AlreadyOwned:
    return invalid("selected operation already has an execution owner");
  case OwnershipScopeRejection::NoOrdinaryCallable:
    return invalid(
        "selected operation is not inside an ordinary LLVM callable");
  case OwnershipScopeRejection::NoRegion:
    return invalid("selected operation owns no region");
  }
  return analysis.callable;
}

struct OperationClosure {
  llvm::SmallVector<mlir::Value> liveIns;
  llvm::SmallVector<mlir::Value> liveOuts;
  llvm::SmallVector<mlir::arith::ConstantOp> constants;
};

/// Derives the selected operation's external SSA closure exactly once in
/// deterministic first-use order. Scalar literals remain part of the selected
/// program; only genuinely dynamic values cross its launch boundary.
OperationClosure deriveOperationClosure(mlir::Operation *operation) {
  OperationClosure closure;
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  operation->walk([&](mlir::Operation *nested) {
    for (mlir::Value operand : nested->getOperands()) {
      if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        if (operation->isAncestor(argument.getOwner()->getParentOp()))
          continue;
      } else if (operation->isAncestor(operand.getDefiningOp())) {
        continue;
      }
      if (!seen.insert(operand).second)
        continue;
      if (auto constant = operand.getDefiningOp<mlir::arith::ConstantOp>()) {
        closure.constants.push_back(constant);
        continue;
      }
      closure.liveIns.push_back(operand);
    }
    return mlir::WalkResult::advance();
  });
  for (mlir::OpResult result : operation->getResults())
    if (!result.use_empty())
      closure.liveOuts.push_back(result);
  return closure;
}

llvm::Error materializePreparedOperation(mlir::ModuleOp module,
                                         mlir::Operation *operation) {
  auto callable = eligibleOwningCallable(operation);
  if (!callable)
    return callable.takeError();
  OperationClosure closure = deriveOperationClosure(operation);
  mlir::Location location = operation->getLoc();
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);

  llvm::SmallVector<mlir::Value, 4> resultSlots;
  if (!closure.liveOuts.empty()) {
    builder.setInsertionPointToStart(&callable->getBody().front());
    mlir::Value one = mlir::LLVM::ConstantOp::create(
        builder, location, builder.getI64Type(), builder.getI64IntegerAttr(1));
    mlir::Type pointerType = mlir::LLVM::LLVMPointerType::get(context);
    resultSlots.reserve(closure.liveOuts.size());
    for (mlir::Value liveOut : closure.liveOuts) {
      if (dataflow::DataflowDialect::containsMemoryCapability(
              liveOut.getType()))
        return invalid("selected operation has an externally used memory "
                       "capability result that cannot cross a thread as data");
      if (!mlir::LLVM::isLoadableType(liveOut.getType()))
        return invalid("selected operation has an externally used result "
                       "that cannot be materialized in caller-owned storage");
      resultSlots.push_back(mlir::LLVM::AllocaOp::create(
          builder, location, pointerType, liveOut.getType(), one));
    }
  }

  llvm::SmallVector<mlir::Type, 4> valueResultTypes;
  valueResultTypes.reserve(closure.liveOuts.size());
  for (mlir::Value liveOut : closure.liveOuts)
    valueResultTypes.push_back(liveOut.getType());
  auto boundary = createSpatialThreadBoundary(
      module, *callable, closure.liveIns, resultSlots, valueResultTypes,
      mlir::TypeRange{}, location);
  if (!boundary)
    return boundary.takeError();

  mlir::IRMapping mapping;
  for (auto [index, liveIn] : llvm::enumerate(closure.liveIns))
    mapping.map(liveIn, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::arith::ConstantOp constant : closure.constants)
    builder.clone(*constant, mapping);
  builder.clone(*operation, mapping);
  llvm::SmallVector<mlir::Value, 4> yieldedValues;
  yieldedValues.reserve(closure.liveOuts.size());
  for (mlir::Value liveOut : closure.liveOuts)
    yieldedValues.push_back(mapping.lookup(liveOut));
  loom::SpatialYieldOp::create(builder, location, yieldedValues,
                               mlir::ValueRange{});

  builder.setInsertionPoint(boundary->thread.getBody().front().getTerminator());
  for (auto [value, slot] : llvm::zip_equal(boundary->spatial.getValueResults(),
                                            boundary->threadOnlyArguments))
    mlir::LLVM::StoreOp::create(builder, location, value, slot);

  builder.setInsertionPoint(operation);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  llvm::SmallVector<mlir::Value, 8> launchOperands(closure.liveIns.begin(),
                                                   closure.liveIns.end());
  llvm::append_range(launchOperands, resultSlots);
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, launchOperands, mlir::ValueRange{},
      mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  for (auto [liveOut, slot] : llvm::zip_equal(closure.liveOuts, resultSlots)) {
    mlir::Value loaded =
        mlir::LLVM::LoadOp::create(builder, location, liveOut.getType(), slot);
    liveOut.replaceAllUsesWith(loaded);
  }
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
        return reject(
            SpatialOwnershipCandidateRejectionKind::ExactFabricInadmissible,
            "exact Fabric admits no memory resource for actor " +
                dataflow::operationSchemaSpelling(projection->schema));
      continue;
    }
    auto resources = capabilities.admittingOperationResources(actor.op);
    if (!resources)
      return resources.takeError();
    if (resources->empty())
      return reject(
          SpatialOwnershipCandidateRejectionKind::ExactFabricInadmissible,
          "exact Fabric admits no operation resource for actor " +
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

  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  mlir::Operation *clonedOperation =
      mapping.lookupOrNull(parentEntity->operation);
  if (!clonedOperation)
    return invalid("selected operation was not mapped into the private clone");
  return PrivateSelection{std::move(clone), clonedOperation};
}

llvm::Expected<MaterializedOwnershipCandidate> finalizeOwnershipCandidate(
    mlir::ModuleOp module, const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &loweringOptions) {
  if (mlir::failed(mlir::verify(module)))
    return invalid("materialized Structured Program does not verify");
  auto canonical = lowering::lowerStructuredModuleToCanonicalDataflow(
      module, loweringOptions);
  if (!canonical)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  canonical.takeError());
  if (llvm::Error error = requireExactFabricCapabilities(*canonical, fabric))
    return std::move(error);
  auto structured = finalizeStructuredProgram(module);
  if (!structured)
    return structured.takeError();
  return MaterializedOwnershipCandidate{std::move(*structured),
                                        std::move(*canonical)};
}

} // namespace

llvm::Expected<std::vector<StructuredEntityRef>>
enumerateWholeCallableSpatialOwnershipScopes(
    const StructuredProgramCandidate &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  std::vector<StructuredEntityRef> scopes;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    auto callable =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (!callable)
      continue;
    if (llvm::Error rejection = verifyEligibleCallable(callable)) {
      llvm::consumeError(std::move(rejection));
      continue;
    }
    scopes.push_back(entity.reference);
  }
  return scopes;
}

llvm::Expected<std::vector<SpatialOwnershipScope>>
enumerateSpatialOwnershipScopes(const StructuredProgramCandidate &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  std::vector<SpatialOwnershipScope> scopes;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    if (!entity.operation)
      continue;
    if (auto callable =
            llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(entity.operation)) {
      if (llvm::Error rejection = verifyEligibleCallable(callable)) {
        llvm::consumeError(std::move(rejection));
        continue;
      }
      scopes.push_back(
          {SpatialOwnershipScopeKind::WholeCallable, entity.reference});
      continue;
    }
    if (auto callable = eligibleOwningCallable(entity.operation); !callable) {
      llvm::consumeError(callable.takeError());
      continue;
    }
    scopes.push_back({SpatialOwnershipScopeKind::Operation, entity.reference});
  }
  return scopes;
}

llvm::Expected<std::vector<StructuredEntityRef>>
enumerateOperationSpatialOwnershipScopes(
    const StructuredProgramCandidate &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  std::vector<StructuredEntityRef> scopes;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    if (!entity.operation)
      continue;
    if (auto callable = eligibleOwningCallable(entity.operation); !callable) {
      llvm::consumeError(callable.takeError());
      continue;
    }
    scopes.push_back(entity.reference);
  }
  return scopes;
}

llvm::Expected<std::vector<SpatialOwnershipDecisionPoint>>
enumerateSpatialOwnershipDecisionDomain(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &scope) {
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(scope);
  if (!entity)
    return entity.takeError();
  if (!entity->operation)
    return invalid("selected StructuredEntityRef is not an operation");

  mlir::Operation *operation = entity->operation;
  if (auto callable = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(operation)) {
    if (llvm::Error error = verifyEligibleCallable(callable))
      return std::move(error);
  } else if (auto callable = eligibleOwningCallable(operation); !callable) {
    return callable.takeError();
  }

  bool containsFmuladd = false;
  operation->walk([&](mlir::Operation *nested) {
    if (nested != operation && llvm::isa<mlir::FunctionOpInterface>(nested))
      return mlir::WalkResult::skip();
    if (llvm::isa<mlir::LLVM::FMulAddOp>(nested)) {
      containsFmuladd = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  llvm::SmallVector<std::optional<unsigned>, 2> indexWidths;
  if (detail::requiresCanonicalAddressIndexDecision(parent.module(),
                                                    operation)) {
    for (::fabric::ResolvedIndexWidth width :
         ::fabric::resolvedIndexWidthDomain)
      indexWidths.push_back(::fabric::getResolvedIndexBitWidth(width));
  } else {
    indexWidths.push_back(std::nullopt);
  }

  llvm::SmallVector<std::optional<raising::FMulAddExecutionShape>, 2>
      executionShapes;
  if (containsFmuladd) {
    executionShapes.push_back(raising::FMulAddExecutionShape::Fused);
    executionShapes.push_back(raising::FMulAddExecutionShape::Split);
  } else {
    executionShapes.push_back(std::nullopt);
  }

  std::vector<SpatialOwnershipDecisionPoint> result;
  result.reserve(indexWidths.size() * executionShapes.size());
  for (std::optional<unsigned> indexWidth : indexWidths)
    for (std::optional<raising::FMulAddExecutionShape> shape : executionShapes)
      result.push_back(SpatialOwnershipDecisionPoint{shape, indexWidth});
  return result;
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnershipDecision(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision,
    const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &lowering) {
  auto prepared = prepareSpatialOwnershipSelection(parent, scope, decision);
  if (!prepared)
    return prepared.takeError();

  switch (scope.kind) {
  case SpatialOwnershipScopeKind::WholeCallable: {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(prepared->operation);
    if (!function)
      return invalid("selected StructuredEntityRef is not an LLVM callable");
    if (llvm::Error error = materializeThread(prepared->module.get(), function))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
    break;
  }
  case SpatialOwnershipScopeKind::Operation: {
    if (llvm::Error error = materializePreparedOperation(prepared->module.get(),
                                                         prepared->operation))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
    break;
  }
  }
  return finalizeOwnershipCandidate(prepared->module.get(), fabric, lowering);
}

llvm::Expected<PreparedSpatialOwnershipSelection>
prepareSpatialOwnershipSelection(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision) {
  auto domain =
      enumerateSpatialOwnershipDecisionDomain(parent, scope.selection);
  if (!domain)
    return domain.takeError();
  if (llvm::find(*domain, decision) == domain->end()) {
    if (!decision.canonicalIndexWidth &&
        llvm::all_of(*domain, [](const SpatialOwnershipDecisionPoint &point) {
          return point.canonicalIndexWidth.has_value();
        }))
      return invalid(
          "selected scope requires an explicit canonical index width");
    if (!decision.fmuladdExecutionShape &&
        llvm::all_of(*domain, [](const SpatialOwnershipDecisionPoint &point) {
          return point.fmuladdExecutionShape.has_value();
        }))
      return invalid(
          "selected scope requires an explicit fmuladd execution shape");
    return invalid("decision is not in the selected scope's typed domain");
  }

  auto selection = cloneSelectedOperation(parent, scope.selection);
  if (!selection)
    return selection.takeError();
  mlir::Operation *operation = selection->operation;
  switch (scope.kind) {
  case SpatialOwnershipScopeKind::WholeCallable: {
    auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(operation);
    if (!function)
      return invalid("selected StructuredEntityRef is not an LLVM callable");
    if (llvm::Error error = verifyEligibleCallable(function))
      return std::move(error);
    auto normalized = detail::materializeAddressIndexContract(
        selection->clone.get(), function.getOperation(),
        decision.canonicalIndexWidth);
    if (!normalized)
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    normalized.takeError());
    operation = *normalized;
    break;
  }
  case SpatialOwnershipScopeKind::Operation: {
    if (auto callable = eligibleOwningCallable(operation); !callable)
      return callable.takeError();
    auto normalized = detail::materializeAddressIndexContract(
        selection->clone.get(), operation, decision.canonicalIndexWidth);
    if (!normalized)
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    normalized.takeError());
    operation = *normalized;
    break;
  }
  }
  if (decision.fmuladdExecutionShape)
    raising::materializeFMulAddInOperation(*operation,
                                           *decision.fmuladdExecutionShape);
  if (mlir::failed(mlir::verify(selection->clone.get())))
    return invalid("prepared ownership selection does not verify");
  std::vector<mlir::Value> liveIns;
  std::vector<mlir::Value> liveOuts;
  if (scope.kind == SpatialOwnershipScopeKind::Operation) {
    OperationClosure closure = deriveOperationClosure(operation);
    liveIns.assign(closure.liveIns.begin(), closure.liveIns.end());
    liveOuts.assign(closure.liveOuts.begin(), closure.liveOuts.end());
  }
  return PreparedSpatialOwnershipSelection{std::move(selection->clone),
                                           operation, std::move(liveIns),
                                           std::move(liveOuts)};
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeWholeCallableSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &callable,
    const fabric::FinalizedFabricRoot &fabric,
    const WholeCallableSpatialOwnershipOptions &options) {
  return materializeSpatialOwnershipDecision(
      parent, {SpatialOwnershipScopeKind::WholeCallable, callable},
      {options.fmuladdExecutionShape, options.canonicalIndexWidth}, fabric,
      options.lowering);
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeOperationSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &operation,
    const fabric::FinalizedFabricRoot &fabric,
    const OperationSpatialOwnershipOptions &options) {
  return materializeSpatialOwnershipDecision(
      parent, {SpatialOwnershipScopeKind::Operation, operation},
      {options.fmuladdExecutionShape, options.canonicalIndexWidth}, fabric,
      options.lowering);
}

} // namespace loom::frontend
