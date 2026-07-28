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

  // A global address is resolved by the stored-program launch owner and
  // crosses the thread ABI as a memory capability. It is not a SpatialCore
  // actor. Hoist nested address leaves to the wrapper entry so every retained
  // leaf has one explicit launch binding and can be removed from the cloned
  // graph body.
  llvm::SmallVector<mlir::LLVM::AddressOfOp, 4> addressCaptures;
  function.walk([&](mlir::LLVM::AddressOfOp address) {
    if (!address.getRes().use_empty())
      addressCaptures.push_back(address);
  });
  for (mlir::LLVM::AddressOfOp address : llvm::reverse(addressCaptures))
    if (&source.front() != address.getOperation())
      address->moveBefore(&source, source.begin());

  llvm::SmallVector<mlir::Value, 8> captures(source.getArguments().begin(),
                                             source.getArguments().end());
  for (mlir::LLVM::AddressOfOp address : addressCaptures)
    captures.push_back(address.getRes());
  auto boundary =
      createSpatialThreadBoundary(module, function, captures, location);
  if (!boundary)
    return boundary.takeError();

  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, capture] : llvm::enumerate(captures))
    mapping.map(capture, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::Operation &operation : source.without_terminator())
    if (!llvm::isa<mlir::LLVM::AddressOfOp>(operation))
      builder.clone(operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  llvm::SmallPtrSet<mlir::Operation *, 4> retainedAddresses;
  for (mlir::LLVM::AddressOfOp address : addressCaptures)
    retainedAddresses.insert(address.getOperation());
  llvm::SmallVector<mlir::Operation *, 16> oldBody;
  for (mlir::Operation &operation : source)
    oldBody.push_back(&operation);
  for (mlir::Operation *operation : llvm::reverse(oldBody))
    if (!retainedAddresses.contains(operation))
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

/// Returns the ordinary LLVM callable that owns the selected operation, after
/// proving the operation is an ownership-free structured entity with no SSA
/// result used outside itself.
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
  for (mlir::OpResult result : operation->getResults())
    for (mlir::Operation *user : result.getUsers())
      if (!operation->isProperAncestor(user))
        return invalid(
            "selected operation has an SSA result used outside itself");
  return analysis.callable;
}

struct OperationClosure {
  llvm::SmallVector<mlir::Value> liveIns;
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
  return closure;
}

llvm::Error materializeSelectedOperation(
    mlir::ModuleOp module, mlir::Operation *operation,
    std::optional<unsigned> canonicalIndexWidth,
    std::optional<raising::FMulAddExecutionShape> fmuladdExecutionShape) {
  auto callable = eligibleOwningCallable(operation);
  if (!callable)
    return callable.takeError();
  auto normalized = detail::materializeAddressIndexContract(
      module, operation, canonicalIndexWidth);
  if (!normalized)
    return normalized.takeError();
  operation = *normalized;
  if (fmuladdExecutionShape)
    raising::materializeFMulAddInOperation(*operation, *fmuladdExecutionShape);
  const OperationClosure closure = deriveOperationClosure(operation);
  mlir::Location location = operation->getLoc();
  auto boundary =
      createSpatialThreadBoundary(module, *callable, closure.liveIns, location);
  if (!boundary)
    return boundary.takeError();

  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, liveIn] : llvm::enumerate(closure.liveIns))
    mapping.map(liveIn, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::arith::ConstantOp constant : closure.constants)
    builder.clone(*constant, mapping);
  builder.clone(*operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  builder.setInsertionPoint(operation);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, closure.liveIns, mlir::ValueRange{},
      mlir::ValueRange{});
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
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  canonical.takeError());
  if (llvm::Error error = requireExactFabricCapabilities(*canonical, fabric))
    return std::move(error);
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
  auto domain =
      enumerateSpatialOwnershipDecisionDomain(parent, scope.selection);
  if (!domain)
    return domain.takeError();
  if (llvm::find(*domain, decision) == domain->end())
    return invalid("decision is not in the selected scope's typed domain");

  switch (scope.kind) {
  case SpatialOwnershipScopeKind::WholeCallable: {
    WholeCallableSpatialOwnershipOptions options;
    options.lowering = lowering;
    options.fmuladdExecutionShape = decision.fmuladdExecutionShape;
    options.canonicalIndexWidth = decision.canonicalIndexWidth;
    return materializeWholeCallableSpatialOwnership(parent, scope.selection,
                                                    fabric, options);
  }
  case SpatialOwnershipScopeKind::Operation: {
    OperationSpatialOwnershipOptions options;
    options.lowering = lowering;
    options.fmuladdExecutionShape = decision.fmuladdExecutionShape;
    options.canonicalIndexWidth = decision.canonicalIndexWidth;
    return materializeOperationSpatialOwnership(parent, scope.selection, fabric,
                                                options);
  }
  }
  llvm_unreachable("unknown Spatial ownership scope kind");
}

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

  auto normalized = detail::materializeAddressIndexContract(
      selection->clone.get(), function.getOperation(),
      options.canonicalIndexWidth);
  if (!normalized)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  normalized.takeError());

  if (options.fmuladdExecutionShape) {
    mlir::PassManager materialization(
        function.getContext(), mlir::LLVM::LLVMFuncOp::getOperationName());
    materialization.enableVerifier(options.lowering.verifyEach);
    materialization.addPass(
        raising::createMaterializeFMulAddPass(*options.fmuladdExecutionShape));
    if (mlir::failed(materialization.run(function.getOperation())))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    "selected fmuladd execution shape is not materializable");
  }

  if (llvm::Error error = materializeThread(selection->clone.get(), function))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  std::move(error));
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
          options.canonicalIndexWidth, options.fmuladdExecutionShape))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  std::move(error));
  return finalizeOwnershipCandidate(selection->clone.get(), fabric,
                                    options.lowering);
}

} // namespace loom::frontend
