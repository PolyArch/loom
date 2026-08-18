#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "StructuredAddressIndexNarrowing.h"
#include "StructuredCallSpecialization.h"
#include "StructuredOwnershipAnalysis.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Lowering/GraphParallelLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

std::string uniqueSymbol(mlir::ModuleOp module, llvm::StringRef prefix,
                         llvm::StringRef sourceName) {
  std::string base = (llvm::Twine(prefix) + sourceName).str();
  std::string candidate = base;
  for (std::uint64_t suffix = 0; module.lookupSymbol(candidate); ++suffix)
    candidate = (llvm::Twine(base) + "_" + llvm::Twine(suffix)).str();
  return candidate;
}

llvm::Error verifyEligibleCallable(mlir::LLVM::LLVMFuncOp function) {
  if (std::optional<std::string> rejection =
          detail::explainCallableOwnershipRejection(function))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  *rejection);
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<mlir::Value, 4>>
createCallerOwnedResultStorage(mlir::LLVM::LLVMFuncOp callable,
                               mlir::TypeRange resultTypes,
                               mlir::Location location) {
  llvm::SmallVector<mlir::Value, 4> slots;
  if (resultTypes.empty())
    return slots;

  mlir::OpBuilder builder(callable.getContext());
  builder.setInsertionPointToStart(&callable.getBody().front());
  mlir::Value one = mlir::LLVM::ConstantOp::create(
      builder, location, builder.getI64Type(), builder.getI64IntegerAttr(1));
  const mlir::Type pointerType =
      mlir::LLVM::LLVMPointerType::get(callable.getContext());
  slots.reserve(resultTypes.size());
  for (mlir::Type type : resultTypes) {
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(type) ||
        dataflow::DataflowDialect::containsMemoryCapability(type))
      return invalid("selected Spatial result is an address or memory "
                     "capability that cannot cross a thread as data");
    if (!mlir::LLVM::isLoadableType(type))
      return invalid("selected Spatial result cannot use caller-owned "
                     "storage");
    slots.push_back(mlir::LLVM::AllocaOp::create(builder, location, pointerType,
                                                 type, one));
  }
  return slots;
}

/// One new private dataflow.thread holding one loom.spatial_region. The
/// captured source values become the thread's function inputs, partitioned
/// into the region's ordinary values and memory capabilities. Logical
/// coordinates are definition-owned suffix arguments and enter the Spatial
/// region as ordinary values.
struct SpatialThreadBoundary {
  dataflow::ThreadOp thread;
  loom::SpatialRegionOp spatial;
  mlir::Block *spatialEntry;
  llvm::SmallVector<mlir::Value> captureArguments;
  llvm::SmallVector<mlir::Value> threadOnlyArguments;
  llvm::SmallVector<mlir::Value> spatialCoordinates;
};

llvm::Expected<StructuredEntityRef>
sourceBlockReference(const PreparedSpatialOwnershipSelection &prepared,
                     mlir::Block *block) {
  auto found = llvm::find_if(
      prepared.sourceBlocks,
      [&](const PreparedSpatialOwnershipSelection::SourceBlockBinding &entry) {
        return entry.candidateBlock == block;
      });
  if (found == prepared.sourceBlocks.end())
    return invalid("ownership block has no parent activity lineage");
  return found->parentBlock;
}

llvm::Error addSourceBlockBinding(PreparedSpatialOwnershipSelection &prepared,
                                  mlir::Block *candidateBlock,
                                  const StructuredEntityRef &parentBlock) {
  if (!candidateBlock || parentBlock.kind != StructuredEntityKind::Block)
    return invalid("ownership block activity lineage has the wrong kind");
  auto found = llvm::find_if(
      prepared.sourceBlocks,
      [&](const PreparedSpatialOwnershipSelection::SourceBlockBinding &entry) {
        return entry.candidateBlock == candidateBlock;
      });
  if (found != prepared.sourceBlocks.end()) {
    if (found->parentBlock != parentBlock)
      return invalid("ownership block has conflicting parent activity lineage");
    return llvm::Error::success();
  }
  prepared.sourceBlocks.push_back({candidateBlock, parentBlock});
  return llvm::Error::success();
}

llvm::Error
propagateClonedBlockBindings(PreparedSpatialOwnershipSelection &prepared,
                             const mlir::IRMapping &mapping,
                             std::size_t sourceBindingCount) {
  if (sourceBindingCount > prepared.sourceBlocks.size())
    return invalid("ownership block activity lineage changed during cloning");
  llvm::DenseMap<mlir::Block *, StructuredEntityRef> knownBindings;
  knownBindings.reserve(prepared.sourceBlocks.size() + sourceBindingCount);
  for (const auto &binding : prepared.sourceBlocks)
    knownBindings.try_emplace(binding.candidateBlock, binding.parentBlock);
  for (std::size_t index = 0; index < sourceBindingCount; ++index) {
    const auto source = prepared.sourceBlocks[index];
    mlir::Block *cloned = mapping.lookupOrNull(source.candidateBlock);
    if (!cloned)
      continue;
    auto [found, inserted] =
        knownBindings.try_emplace(cloned, source.parentBlock);
    if (!inserted) {
      if (found->second != source.parentBlock)
        return invalid(
            "ownership block has conflicting parent activity lineage");
      continue;
    }
    prepared.sourceBlocks.push_back({cloned, source.parentBlock});
  }
  return llvm::Error::success();
}

llvm::Error propagateInlinedBlockBindings(
    std::vector<PreparedSpatialOwnershipSelection::SourceBlockBinding>
        &sourceBlocks,
    const mlir::IRMapping &mapping, std::size_t sourceBindingCount) {
  if (sourceBindingCount > sourceBlocks.size())
    return invalid("ownership block activity lineage changed during inlining");
  llvm::DenseSet<mlir::Block *> preexisting;
  llvm::DenseMap<mlir::Block *, StructuredEntityRef> added;
  preexisting.reserve(sourceBlocks.size());
  for (const auto &binding : sourceBlocks)
    preexisting.insert(binding.candidateBlock);
  for (std::size_t index = 0; index < sourceBindingCount; ++index) {
    const auto source = sourceBlocks[index];
    mlir::Block *cloned = mapping.lookupOrNull(source.candidateBlock);
    if (!cloned || preexisting.contains(cloned))
      continue;
    auto [found, inserted] = added.try_emplace(cloned, source.parentBlock);
    if (!inserted) {
      if (found->second != source.parentBlock)
        return invalid("inlined block has conflicting parent activity lineage");
      continue;
    }
    sourceBlocks.push_back({cloned, source.parentBlock});
  }
  return llvm::Error::success();
}

llvm::Expected<SpatialThreadBoundary> createSpatialThreadBoundary(
    mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp sourceCallable,
    mlir::ValueRange captures, mlir::ValueRange threadOnlyCaptures,
    mlir::TypeRange valueResultTypes, mlir::TypeRange memoryResultTypes,
    mlir::Location location, unsigned threadRank = 0) {
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
  for (unsigned dimension = 0; dimension < threadRank; ++dimension)
    threadEntry->addArgument(builder.getIndexType(), location);

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
  const std::size_t coordinateValueOffset = values.size();
  llvm::append_range(values, threadEntry->getArguments().drop_front(
                                 threadInputTypes.size() + 1));
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
  llvm::SmallVector<mlir::Value, 4> spatialCoordinates;
  spatialCoordinates.reserve(threadRank);
  for (unsigned dimension = 0; dimension < threadRank; ++dimension)
    spatialCoordinates.push_back(
        spatialEntry->getArgument(coordinateValueOffset + dimension));
  return SpatialThreadBoundary{thread,
                               spatial,
                               spatialEntry,
                               std::move(captureArguments),
                               std::move(threadOnlyArguments),
                               std::move(spatialCoordinates)};
}

llvm::Error materializeThread(PreparedSpatialOwnershipSelection &prepared,
                              mlir::LLVM::LLVMFuncOp function) {
  mlir::ModuleOp module = prepared.module.get();
  if (llvm::Error error = verifyEligibleCallable(function))
    return error;
  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = function.getLoc();
  mlir::Block &source = function.getBody().front();
  auto activationSource = sourceBlockReference(prepared, &source);
  if (!activationSource)
    return activationSource.takeError();
  auto sourceReturn = llvm::cast<mlir::LLVM::ReturnOp>(source.getTerminator());

  // A global address is resolved by the stored-program launch owner and
  // crosses the thread ABI as a memory capability. It is not a SpatialCore
  // actor. Hoist nested address leaves to the wrapper entry so every retained
  // leaf has one explicit launch binding and can be removed from the cloned
  // graph body.
  detail::CallableOwnershipBoundary callableBoundary =
      detail::deriveCallableOwnershipBoundary(function);
  llvm::ArrayRef<mlir::Operation *> selectedBody = prepared.callableSpatialBody;
  if (selectedBody.empty())
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  "dynamic pointer service prelude leaves no Spatial body");
  llvm::ArrayRef<mlir::Value> captures = prepared.liveIns;
  llvm::ArrayRef<mlir::Value> yieldedValues = prepared.liveOuts;
  llvm::SmallVector<std::optional<std::size_t>, 1> outputResultOrdinals;
  for (mlir::Value output : callableBoundary.outputs) {
    auto found = llvm::find(yieldedValues, output);
    outputResultOrdinals.push_back(
        found == yieldedValues.end()
            ? std::optional<std::size_t>{}
            : std::optional<std::size_t>(found - yieldedValues.begin()));
  }
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  for (mlir::Value value : yieldedValues)
    resultTypes.push_back(value.getType());
  auto resultSlots =
      createCallerOwnedResultStorage(function, resultTypes, location);
  if (!resultSlots)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  resultSlots.takeError());

  auto boundary =
      createSpatialThreadBoundary(module, function, captures, *resultSlots,
                                  resultTypes, mlir::TypeRange{}, location);
  if (!boundary)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  boundary.takeError());
  prepared.materializedSpatialRegion = boundary->spatial.getOperation();

  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  const std::size_t sourceBindingCount = prepared.sourceBlocks.size();
  for (auto [index, capture] : llvm::enumerate(captures))
    mapping.map(capture, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::Operation *operation : selectedBody)
    builder.clone(*operation, mapping);
  if (llvm::Error error =
          propagateClonedBlockBindings(prepared, mapping, sourceBindingCount))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, &boundary->thread.getBody().front(), *activationSource))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, boundary->spatialEntry, *activationSource))
    return error;
  llvm::SmallVector<mlir::Value, 1> mappedResults;
  mappedResults.reserve(yieldedValues.size());
  for (mlir::Value value : yieldedValues)
    mappedResults.push_back(mapping.lookup(value));
  loom::SpatialYieldOp::create(builder, location, mappedResults,
                               mlir::ValueRange{});

  builder.setInsertionPoint(boundary->thread.getBody().front().getTerminator());
  for (auto [value, slot] : llvm::zip_equal(boundary->spatial.getValueResults(),
                                            boundary->threadOnlyArguments))
    mlir::LLVM::StoreOp::create(builder, location, value, slot);

  sourceReturn.erase();
  for (mlir::Operation *operation : llvm::reverse(selectedBody))
    operation->erase();

  builder.setInsertionPointToEnd(&source);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, boundary->thread.getSymName());
  llvm::SmallVector<mlir::Value, 8> launchOperands(captures);
  llvm::append_range(launchOperands, *resultSlots);
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, launchOperands, mlir::ValueRange{},
      mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  llvm::SmallVector<mlir::Value, 1> loadedResults;
  for (auto [slot, type] : llvm::zip_equal(*resultSlots, resultTypes))
    loadedResults.push_back(
        mlir::LLVM::LoadOp::create(builder, location, type, slot));
  llvm::SmallVector<mlir::Value, 1> returnedValues;
  for (auto [output, resultOrdinal] :
       llvm::zip_equal(callableBoundary.outputs, outputResultOrdinals))
    returnedValues.push_back(resultOrdinal ? loadedResults[*resultOrdinal]
                                           : output);
  mlir::LLVM::ReturnOp::create(builder, location, returnedValues);
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

bool isExactDirectCallLeafScope(mlir::ModuleOp module,
                                mlir::Operation *operation) {
  if (!llvm::isa_and_nonnull<mlir::LLVM::CallOp>(operation))
    return false;
  std::optional<detail::ExactDirectCallSiteInliningCandidate> candidate =
      detail::findExactDirectCallSiteInliningCandidate(module, operation);
  return candidate && candidate->callSite == operation;
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

bool fitsSignedIndexWidth(__int128 value, unsigned width);

llvm::Error verifyThreadDomainForall(mlir::scf::ForallOp forall) {
  if (forall.getMapping())
    return invalid("thread-domain forall must not carry a physical mapping");
  if (!forall.getOutputs().empty() || forall.getNumResults() != 0)
    return invalid(
        "thread-domain forall must materialize aggregation before ownership");
  auto inParallel = forall.getTerminator();
  if (!inParallel.getRegion().front().empty())
    return invalid("thread-domain forall must have an empty combining action");
  if (mlir::failed(lowering::checkLogicalThreadParallelPreconditions(forall)))
    return invalid("thread-domain forall failed parallel dependence and "
                   "effect legality");
  return llvm::Error::success();
}

llvm::Error verifyDynamicThreadDomainWidth(mlir::scf::ForallOp forall,
                                           unsigned indexWidth) {
  for (auto [lower, upper, step] :
       llvm::zip_equal(forall.getMixedLowerBound(), forall.getMixedUpperBound(),
                       forall.getMixedStep())) {
    if (mlir::getConstantIntValue(lower) && mlir::getConstantIntValue(upper) &&
        mlir::getConstantIntValue(step))
      continue;
    if (indexWidth != 32 && indexWidth != 64)
      return invalid("dynamic thread-domain extent requires a 32-bit or "
                     "64-bit selected index ABI");
    if (!detail::provesThreadDomainExtentFits(lower, upper, step, indexWidth))
      return invalid("dynamic thread-domain bounds have no complete signed "
                     "value-domain proof for the selected index width");
  }
  return llvm::Error::success();
}

void addForallThreadBodyCapture(mlir::scf::ForallOp forall, mlir::Value value,
                                llvm::SmallVectorImpl<mlir::Value> &liveIns,
                                llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!seen.insert(value).second)
    return;
  if (llvm::is_contained(forall.getInductionVars(), value))
    return;
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    if (forall->isAncestor(argument.getOwner()->getParentOp()))
      return;
  } else if (forall->isAncestor(value.getDefiningOp())) {
    return;
  }
  if (value.getDefiningOp<mlir::arith::ConstantOp>())
    return;
  liveIns.push_back(value);
}

struct DerivedForallThreadBoundary final {
  llvm::SmallVector<mlir::Value, 8> liveIns;
  std::vector<PreparedSpatialOwnershipSelection::SourceInductionBinding>
      sourceInductions;
};

std::uint64_t
appendForallThreadInput(llvm::SmallVectorImpl<mlir::Value> &liveIns,
                        mlir::Value value) {
  auto found = llvm::find(liveIns, value);
  if (found != liveIns.end())
    return static_cast<std::uint64_t>(std::distance(liveIns.begin(), found));
  const std::uint64_t ordinal = liveIns.size();
  liveIns.push_back(value);
  return ordinal;
}

llvm::Expected<DerivedForallThreadBoundary>
deriveForallThreadBoundary(mlir::scf::ForallOp forall) {
  if (llvm::Error error = verifyThreadDomainForall(forall))
    return std::move(error);

  DerivedForallThreadBoundary boundary;
  llvm::SmallPtrSet<mlir::Value, 8> seen;
  for (mlir::Operation &operation : forall.getBody()->without_terminator()) {
    operation.walk([&](mlir::Operation *nested) {
      for (mlir::Value operand : nested->getOperands())
        addForallThreadBodyCapture(forall, operand, boundary.liveIns, seen);
    });
  }

  mlir::OpBuilder builder(forall);
  boundary.sourceInductions.reserve(forall.getRank());
  for (auto [lower, step] :
       llvm::zip_equal(forall.getMixedLowerBound(), forall.getMixedStep())) {
    PreparedSpatialOwnershipSelection::SourceInductionBinding binding;
    if (mlir::getConstantIntValue(lower) != 0) {
      mlir::Value value = mlir::getValueOrCreateConstantIndexOp(
          builder, forall.getLoc(), lower);
      binding.lowerInputOrdinal =
          appendForallThreadInput(boundary.liveIns, value);
    }
    if (mlir::getConstantIntValue(step) != 1) {
      mlir::Value value =
          mlir::getValueOrCreateConstantIndexOp(builder, forall.getLoc(), step);
      binding.stepInputOrdinal =
          appendForallThreadInput(boundary.liveIns, value);
    }
    boundary.sourceInductions.push_back(binding);
  }
  return boundary;
}

llvm::Expected<mlir::Value> materializeIndexValue(mlir::OpBuilder &builder,
                                                  mlir::Location location,
                                                  mlir::OpFoldResult value) {
  if (auto dynamic = llvm::dyn_cast<mlir::Value>(value))
    return dynamic;
  return mlir::getValueOrCreateConstantIndexOp(builder, location, value);
}

bool fitsSignedIndexWidth(__int128 value, unsigned width) {
  if (width == 0)
    return false;
  if (width >= 128)
    return true;
  const __int128 limit = static_cast<__int128>(1) << (width - 1);
  return value >= -limit && value < limit;
}

llvm::Expected<mlir::Value>
materializeThreadExtent(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::OpFoldResult lower, mlir::OpFoldResult upper,
                        mlir::OpFoldResult step, unsigned indexWidth) {
  std::optional<int64_t> staticLower = mlir::getConstantIntValue(lower);
  std::optional<int64_t> staticUpper = mlir::getConstantIntValue(upper);
  std::optional<int64_t> staticStep = mlir::getConstantIntValue(step);
  if (staticStep && *staticStep <= 0)
    return invalid("thread-domain forall step must be positive");
  if (staticLower && staticUpper && staticStep) {
    if (!fitsSignedIndexWidth(*staticLower, indexWidth) ||
        !fitsSignedIndexWidth(*staticUpper, indexWidth) ||
        !fitsSignedIndexWidth(*staticStep, indexWidth))
      return invalid(
          "thread-domain bounds exceed the selected signed index width");
    __int128 distance = static_cast<__int128>(*staticUpper) - *staticLower;
    __int128 extent =
        distance <= 0 ? 0 : (distance + *staticStep - 1) / *staticStep;
    if (!fitsSignedIndexWidth(extent, indexWidth))
      return invalid(
          "thread-domain extent exceeds the selected signed index width");
    if (extent != 0) {
      __int128 last =
          static_cast<__int128>(*staticLower) + (extent - 1) * *staticStep;
      if (!fitsSignedIndexWidth(last, indexWidth))
        return invalid("thread-domain source induction exceeds the selected "
                       "signed index width");
    }
    return mlir::arith::ConstantIndexOp::create(builder, location,
                                                static_cast<int64_t>(extent));
  }
  if (indexWidth != 32 && indexWidth != 64)
    return invalid("dynamic thread-domain extent requires a 32-bit or 64-bit "
                   "selected index ABI");
  auto lowerValue = materializeIndexValue(builder, location, lower);
  if (!lowerValue)
    return lowerValue.takeError();
  auto upperValue = materializeIndexValue(builder, location, upper);
  if (!upperValue)
    return upperValue.takeError();
  auto stepValue = materializeIndexValue(builder, location, step);
  if (!stepValue)
    return stepValue.takeError();

  if (!detail::provesThreadDomainExtentFits(lower, upper, step, indexWidth))
    return invalid("dynamic thread-domain bounds have no complete signed "
                   "value-domain proof for the selected index width");

  const unsigned arithmeticWidth = indexWidth * 2;
  mlir::IntegerType arithmeticType = builder.getIntegerType(arithmeticWidth);
  auto widen = [&](mlir::Value value) {
    return mlir::arith::IndexCastOp::create(builder, location, arithmeticType,
                                            value)
        .getResult();
  };
  mlir::Value widenedLower = widen(*lowerValue);
  mlir::Value widenedUpper = widen(*upperValue);
  mlir::Value widenedStep = widen(*stepValue);
  auto integerConstant = [&](const llvm::APInt &value) {
    return mlir::arith::ConstantOp::create(
        builder, location, arithmeticType,
        mlir::IntegerAttr::get(arithmeticType, value));
  };
  mlir::Value zero = integerConstant(llvm::APInt(arithmeticWidth, 0));
  mlir::Value distance = mlir::arith::SubIOp::create(
      builder, location, widenedUpper, widenedLower);
  mlir::Value positiveDistance =
      mlir::arith::MaxSIOp::create(builder, location, distance, zero);
  mlir::Value extent = mlir::arith::CeilDivSIOp::create(
      builder, location, positiveDistance, widenedStep);
  mlir::Value narrowed = mlir::arith::IndexCastOp::create(
      builder, location, builder.getIndexType(), extent);
  return narrowed;
}

llvm::Expected<mlir::Value> materializeSourceInduction(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Value coordinate,
    std::optional<mlir::Value> lower, std::optional<mlir::Value> step,
    unsigned indexWidth) {
  if (!lower && !step)
    return coordinate;
  if (indexWidth == 0 || indexWidth > mlir::IntegerType::kMaxWidth / 2)
    return invalid("thread-domain source-IV arithmetic width is invalid");

  mlir::IntegerType arithmeticType = builder.getIntegerType(indexWidth * 2);
  auto widen = [&](mlir::Value value) -> llvm::Expected<mlir::Value> {
    if (!llvm::isa<mlir::IndexType>(value.getType()))
      return invalid("thread-domain source-IV operand is not index-typed");
    return mlir::arith::IndexCastOp::create(builder, location, arithmeticType,
                                            value)
        .getResult();
  };

  auto widenedCoordinate = widen(coordinate);
  if (!widenedCoordinate)
    return widenedCoordinate.takeError();
  mlir::Value reconstructed = *widenedCoordinate;
  if (step) {
    auto widenedStep = widen(*step);
    if (!widenedStep)
      return widenedStep.takeError();
    reconstructed = mlir::arith::MulIOp::create(builder, location,
                                                reconstructed, *widenedStep);
  }
  if (lower) {
    auto widenedLower = widen(*lower);
    if (!widenedLower)
      return widenedLower.takeError();
    reconstructed = mlir::arith::AddIOp::create(builder, location,
                                                *widenedLower, reconstructed);
  }
  return mlir::arith::IndexCastOp::create(
      builder, location, builder.getIndexType(), reconstructed);
}

llvm::Error materializePreparedForallThreadDomain(
    PreparedSpatialOwnershipSelection &prepared, mlir::scf::ForallOp forall) {
  mlir::ModuleOp module = prepared.module.get();
  if (!prepared.sourceInductions ||
      prepared.sourceInductions->size() !=
          static_cast<std::size_t>(forall.getRank()))
    return invalid("thread-domain source-IV boundary is not total");
  if (!prepared.threadExtents || prepared.threadExtents->size() !=
                                     static_cast<std::size_t>(forall.getRank()))
    return invalid("thread-domain extent projection is not total");
  auto indexWidth = ::loom::getIndexBitWidth(forall);
  if (!indexWidth)
    return indexWidth.takeError();
  auto callable = eligibleOwningCallable(forall);
  if (!callable)
    return callable.takeError();
  auto activationSource = sourceBlockReference(prepared, forall.getBody());
  if (!activationSource)
    return activationSource.takeError();

  mlir::Location location = forall.getLoc();
  auto boundary = createSpatialThreadBoundary(
      module, *callable, prepared.liveIns, mlir::ValueRange{},
      mlir::TypeRange{}, mlir::TypeRange{}, location, forall.getRank());
  if (!boundary)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  boundary.takeError());
  prepared.materializedSpatialRegion = boundary->spatial.getOperation();

  mlir::IRMapping mapping;
  const std::size_t sourceBindingCount = prepared.sourceBlocks.size();
  for (auto [index, liveIn] : llvm::enumerate(prepared.liveIns))
    mapping.map(liveIn, boundary->captureArguments[index]);
  mlir::OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  llvm::SmallPtrSet<mlir::Operation *, 8> clonedConstants;
  forall.getBody()->walk([&](mlir::Operation *nested) {
    for (mlir::Value operand : nested->getOperands()) {
      auto constant = operand.getDefiningOp<mlir::arith::ConstantOp>();
      if (constant && !forall->isAncestor(constant) &&
          !mapping.contains(operand) && clonedConstants.insert(constant).second)
        builder.clone(*constant, mapping);
    }
  });
  for (auto [dimension, induction] :
       llvm::enumerate(forall.getInductionVars())) {
    const auto &binding = (*prepared.sourceInductions)[dimension];
    auto resolveInput = [&](std::optional<std::uint64_t> ordinal)
        -> llvm::Expected<std::optional<mlir::Value>> {
      if (!ordinal)
        return std::optional<mlir::Value>{};
      if (*ordinal >= boundary->captureArguments.size())
        return invalid("thread-domain source-IV input is out of range");
      return std::optional<mlir::Value>(boundary->captureArguments[*ordinal]);
    };
    auto lower = resolveInput(binding.lowerInputOrdinal);
    if (!lower)
      return lower.takeError();
    auto step = resolveInput(binding.stepInputOrdinal);
    if (!step)
      return step.takeError();
    auto sourceInduction = materializeSourceInduction(
        builder, location, boundary->spatialCoordinates[dimension],
        std::move(*lower), std::move(*step), *indexWidth);
    if (!sourceInduction)
      return sourceInduction.takeError();
    mapping.map(induction, *sourceInduction);
  }
  for (mlir::Operation &operation : forall.getBody()->without_terminator())
    builder.clone(operation, mapping);
  if (llvm::Error error =
          propagateClonedBlockBindings(prepared, mapping, sourceBindingCount))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, &boundary->thread.getBody().front(), *activationSource))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, boundary->spatialEntry, *activationSource))
    return error;
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  builder.setInsertionPoint(forall);
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(module.getContext(),
                                   boundary->thread.getSymName()),
      prepared.liveIns, *prepared.threadExtents, mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  forall.erase();
  return llvm::Error::success();
}

llvm::Expected<mlir::Value> projectThreadValueToLaunch(
    mlir::Value value, loom::SpatialRegionOp spatial, dataflow::ThreadOp thread,
    dataflow::ThreadLaunchOp launch, mlir::scf::ForallOp selected,
    mlir::OpBuilder &builder,
    llvm::DenseMap<mlir::Value, mlir::Value> &projected) {
  if (auto found = projected.find(value); found != projected.end())
    return found->second;

  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    if (argument.getOwner() == &spatial.getBody().front()) {
      if (argument.getArgNumber() >= spatial->getNumOperands())
        return invalid("Spatial argument has no owning operand");
      auto result = projectThreadValueToLaunch(
          spatial->getOperand(argument.getArgNumber()), spatial, thread, launch,
          selected, builder, projected);
      if (!result)
        return result.takeError();
      projected.try_emplace(value, *result);
      return *result;
    }
    if (argument.getOwner() == &thread.getBody().front()) {
      const std::size_t inputCount = thread.getFunctionType().getNumInputs();
      if (argument.getArgNumber() >= inputCount)
        return invalid("thread control or coordinate cannot define a launch "
                       "extent");
      if (argument.getArgNumber() >= launch.getBodyOperands().size())
        return invalid("thread launch operand projection is not total");
      mlir::Value result = launch.getBodyOperands()[argument.getArgNumber()];
      projected.try_emplace(value, result);
      return result;
    }
    return invalid("nested block argument cannot define a root thread "
                   "extent");
  }

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !thread->isAncestor(definition) ||
      selected->isAncestor(definition))
    return invalid("thread extent is not launch-projectable");
  if (definition->getNumRegions() != 0 || definition->getNumSuccessors() != 0 ||
      !mlir::isPure(definition))
    return invalid("thread extent depends on a non-speculatable operation");

  mlir::IRMapping mapping;
  for (mlir::Value operand : definition->getOperands()) {
    auto mapped = projectThreadValueToLaunch(operand, spatial, thread, launch,
                                             selected, builder, projected);
    if (!mapped)
      return mapped.takeError();
    mapping.map(operand, *mapped);
  }
  builder.clone(*definition, mapping);
  for (mlir::Value result : definition->getResults()) {
    mlir::Value mapped = mapping.lookupOrNull(result);
    if (!mapped)
      return invalid("cloned thread extent operation lost a result");
    projected.try_emplace(result, mapped);
  }
  auto found = projected.find(value);
  if (found == projected.end())
    return invalid("thread extent projection lost the requested value");
  return found->second;
}

llvm::Expected<mlir::OpFoldResult> projectThreadBoundToLaunch(
    mlir::OpFoldResult bound, loom::SpatialRegionOp spatial,
    dataflow::ThreadOp thread, dataflow::ThreadLaunchOp launch,
    mlir::scf::ForallOp selected, mlir::OpBuilder &builder,
    llvm::DenseMap<mlir::Value, mlir::Value> &projected) {
  if (auto attribute = llvm::dyn_cast<mlir::Attribute>(bound))
    return mlir::OpFoldResult(attribute);
  auto value =
      projectThreadValueToLaunch(llvm::cast<mlir::Value>(bound), spatial,
                                 thread, launch, selected, builder, projected);
  if (!value)
    return value.takeError();
  return mlir::OpFoldResult(*value);
}

llvm::Error
materializeOwnedSpatialForallThreadDomainImpl(mlir::scf::ForallOp forall) {
  if (!forall)
    return invalid("thread-domain promotion has no forall");
  auto spatial = forall->getParentOfType<loom::SpatialRegionOp>();
  auto thread = spatial ? spatial->getParentOfType<dataflow::ThreadOp>()
                        : dataflow::ThreadOp{};
  if (!spatial || !thread || spatial->getBlock() != &thread.getBody().front())
    return invalid("thread-domain forall is not inside one owned Spatial "
                   "carrier");
  const std::size_t inputCount = thread.getFunctionType().getNumInputs();
  mlir::Block &threadEntry = thread.getBody().front();
  if (threadEntry.getNumArguments() != inputCount + 1)
    return invalid("only a rank-zero thread can acquire a scheduled domain");
  if (llvm::Error error = verifyThreadDomainForall(forall))
    return error;
  auto indexWidth = ::loom::getIndexBitWidth(forall);
  if (!indexWidth)
    return indexWidth.takeError();
  if (llvm::Error error = verifyDynamicThreadDomainWidth(forall, *indexWidth))
    return error;

  mlir::ModuleOp module = thread->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTableCollection symbols;
  llvm::SmallVector<dataflow::ThreadLaunchOp, 2> launches;
  module.walk([&](dataflow::ThreadLaunchOp launch) {
    if (symbols.lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr()) == thread)
      launches.push_back(launch);
  });
  if (launches.empty())
    return invalid("owned thread has no exact launch");

  struct LaunchExtents final {
    dataflow::ThreadLaunchOp launch;
    llvm::SmallVector<mlir::Value, 4> values;
  };
  llvm::SmallVector<LaunchExtents, 2> launchExtents;
  launchExtents.reserve(launches.size());
  for (dataflow::ThreadLaunchOp launch : launches) {
    mlir::OpBuilder builder(launch);
    llvm::DenseMap<mlir::Value, mlir::Value> projected;
    LaunchExtents extents{launch, {}};
    extents.values.reserve(forall.getRank());
    for (auto [lower, upper, step] :
         llvm::zip_equal(forall.getMixedLowerBound(),
                         forall.getMixedUpperBound(), forall.getMixedStep())) {
      auto projectedLower = projectThreadBoundToLaunch(
          lower, spatial, thread, launch, forall, builder, projected);
      if (!projectedLower)
        return projectedLower.takeError();
      auto projectedUpper = projectThreadBoundToLaunch(
          upper, spatial, thread, launch, forall, builder, projected);
      if (!projectedUpper)
        return projectedUpper.takeError();
      auto projectedStep = projectThreadBoundToLaunch(
          step, spatial, thread, launch, forall, builder, projected);
      if (!projectedStep)
        return projectedStep.takeError();
      auto extent =
          materializeThreadExtent(builder, forall.getLoc(), *projectedLower,
                                  *projectedUpper, *projectedStep, *indexWidth);
      if (!extent)
        return extent.takeError();
      extents.values.push_back(*extent);
    }
    launchExtents.push_back(std::move(extents));
  }

  mlir::Block &spatialEntry = spatial.getBody().front();
  llvm::SmallVector<mlir::Value, 4> coordinates;
  coordinates.reserve(forall.getRank());
  for (unsigned dimension = 0; dimension < forall.getRank(); ++dimension) {
    mlir::BlockArgument threadCoordinate = threadEntry.addArgument(
        mlir::IndexType::get(module.getContext()), forall.getLoc());
    const std::size_t valueOrdinal = spatial.getValueInputs().size();
    spatial.getValueInputsMutable().append(threadCoordinate);
    coordinates.push_back(spatialEntry.insertArgument(
        valueOrdinal, mlir::IndexType::get(module.getContext()),
        forall.getLoc()));
  }

  mlir::OpBuilder builder(forall);
  mlir::IRMapping mapping;
  for (auto [dimension, induction] :
       llvm::enumerate(forall.getInductionVars())) {
    mlir::OpFoldResult mixedLower = forall.getMixedLowerBound()[dimension];
    mlir::OpFoldResult mixedStep = forall.getMixedStep()[dimension];
    std::optional<mlir::Value> lower;
    std::optional<mlir::Value> step;
    if (mlir::getConstantIntValue(mixedLower) != 0) {
      auto value = materializeIndexValue(builder, forall.getLoc(), mixedLower);
      if (!value)
        return value.takeError();
      lower = *value;
    }
    if (mlir::getConstantIntValue(mixedStep) != 1) {
      auto value = materializeIndexValue(builder, forall.getLoc(), mixedStep);
      if (!value)
        return value.takeError();
      step = *value;
    }
    auto sourceInduction = materializeSourceInduction(builder, forall.getLoc(),
                                                      coordinates[dimension],
                                                      lower, step, *indexWidth);
    if (!sourceInduction)
      return sourceInduction.takeError();
    mapping.map(induction, *sourceInduction);
  }
  for (mlir::Operation &operation : forall.getBody()->without_terminator())
    builder.clone(operation, mapping);
  forall.erase();
  for (LaunchExtents &extents : launchExtents)
    extents.launch.getGridUpperBoundsMutable().append(extents.values);
  return llvm::Error::success();
}

llvm::Error
materializePreparedOperation(PreparedSpatialOwnershipSelection &prepared,
                             mlir::Operation *operation) {
  mlir::ModuleOp module = prepared.module.get();
  auto callable = eligibleOwningCallable(operation);
  if (!callable)
    return callable.takeError();
  auto activationSource = sourceBlockReference(prepared, operation->getBlock());
  if (!activationSource)
    return activationSource.takeError();
  OperationClosure closure = deriveOperationClosure(operation);
  mlir::Location location = operation->getLoc();
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);

  llvm::SmallVector<mlir::Type, 4> valueResultTypes;
  valueResultTypes.reserve(closure.liveOuts.size());
  for (mlir::Value liveOut : closure.liveOuts)
    valueResultTypes.push_back(liveOut.getType());
  auto resultSlots =
      createCallerOwnedResultStorage(*callable, valueResultTypes, location);
  if (!resultSlots)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  resultSlots.takeError());
  auto boundary = createSpatialThreadBoundary(
      module, *callable, closure.liveIns, *resultSlots, valueResultTypes,
      mlir::TypeRange{}, location);
  if (!boundary)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  boundary.takeError());
  prepared.materializedSpatialRegion = boundary->spatial.getOperation();

  mlir::IRMapping mapping;
  const std::size_t sourceBindingCount = prepared.sourceBlocks.size();
  for (auto [index, liveIn] : llvm::enumerate(closure.liveIns))
    mapping.map(liveIn, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::arith::ConstantOp constant : closure.constants)
    builder.clone(*constant, mapping);
  auto exactClosure = llvm::dyn_cast<mlir::scf::ExecuteRegionOp>(operation);
  if (exactClosure) {
    if (!exactClosure.getRegion().hasOneBlock())
      return invalid("direct-call exact closure is not single-block");
    for (mlir::Operation &nested :
         exactClosure.getRegion().front().without_terminator())
      builder.clone(nested, mapping);
  } else {
    builder.clone(*operation, mapping);
  }
  if (llvm::Error error =
          propagateClonedBlockBindings(prepared, mapping, sourceBindingCount))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, &boundary->thread.getBody().front(), *activationSource))
    return error;
  if (llvm::Error error = addSourceBlockBinding(
          prepared, boundary->spatialEntry, *activationSource))
    return error;
  llvm::SmallVector<mlir::Value, 4> yieldedValues;
  yieldedValues.reserve(closure.liveOuts.size());
  if (exactClosure) {
    auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(
        exactClosure.getRegion().front().getTerminator());
    if (!yield || yield.getNumOperands() != exactClosure.getNumResults())
      return invalid("direct-call exact closure has an invalid yield");
    for (mlir::Value liveOut : closure.liveOuts) {
      auto result = llvm::dyn_cast<mlir::OpResult>(liveOut);
      if (!result || result.getOwner() != exactClosure.getOperation() ||
          result.getResultNumber() >= yield.getNumOperands())
        return invalid("direct-call exact closure has an invalid live result");
      mlir::Value yielded = yield.getOperand(result.getResultNumber());
      if (!mapping.contains(yielded))
        return invalid("direct-call exact closure result was not cloned");
      yieldedValues.push_back(mapping.lookup(yielded));
    }
  } else {
    for (mlir::Value liveOut : closure.liveOuts)
      yieldedValues.push_back(mapping.lookup(liveOut));
  }
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
  llvm::append_range(launchOperands, *resultSlots);
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, launchOperands, mlir::ValueRange{},
      mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  for (auto [liveOut, slot] : llvm::zip_equal(closure.liveOuts, *resultSlots)) {
    mlir::Value loaded =
        mlir::LLVM::LoadOp::create(builder, location, liveOut.getType(), slot);
    liveOut.replaceAllUsesWith(loaded);
  }
  operation->erase();
  return llvm::Error::success();
}

/// Resolves the exact parent-local selection, clones the complete parent
/// candidate, and resolves the same reference in the clone.
struct PrivateSelection {
  mlir::OwningOpRef<mlir::ModuleOp> clone;
  mlir::Operation *operation;
  mlir::Operation *directCallSite;
  std::vector<PreparedSpatialOwnershipSelection::SourceBlockBinding>
      sourceBlocks;
};

llvm::Expected<PrivateSelection> cloneSelectedOperation(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &selection,
    const std::optional<DirectCallInliningDecision> &directCallInlining,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto parentView = parent.view();
  if (!parentView)
    return parentView.takeError();
  auto parentEntity = parentView->resolve(selection);
  if (!parentEntity)
    return parentEntity.takeError();
  if (!parentEntity->operation)
    return invalid("selected StructuredEntityRef is not an operation");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  mlir::Operation *clonedOperation =
      mapping.lookupOrNull(parentEntity->operation);
  if (!clonedOperation)
    return invalid("selected operation was not mapped into the private clone");
  mlir::Operation *clonedCallSite = nullptr;
  if (directCallInlining) {
    if (directCallInlining->callSite.parent != parent.identity() ||
        directCallInlining->callSite.kind != StructuredEntityKind::Operation)
      return invalid("direct-call inline site has the wrong parent or kind");
    auto callEntity = parentView->resolve(directCallInlining->callSite);
    if (!callEntity)
      return callEntity.takeError();
    if (!llvm::isa_and_nonnull<mlir::LLVM::CallOp>(callEntity->operation))
      return invalid("direct-call inline site does not resolve to llvm.call");
    clonedCallSite = mapping.lookupOrNull(callEntity->operation);
    if (!clonedCallSite)
      return invalid("direct-call inline site was not mapped into the clone");
  }
  std::vector<PreparedSpatialOwnershipSelection::SourceBlockBinding>
      sourceBlocks;
  sourceBlocks.reserve(
      parentView->entities(StructuredEntityKind::Block).size());
  for (const StructuredEntity &entity :
       parentView->entities(StructuredEntityKind::Block)) {
    mlir::Block *mapped = mapping.lookupOrNull(entity.block);
    if (!mapped)
      return invalid("parent block was not mapped into the private clone");
    sourceBlocks.push_back({mapped, entity.reference});
  }
  return PrivateSelection{std::move(clone), clonedOperation, clonedCallSite,
                          std::move(sourceBlocks)};
}

llvm::Error retainLiveBlockLineage(PrivateSelection &selection) {
  llvm::DenseSet<mlir::Block *> liveBlocks;
  selection.clone->walk([&](mlir::Operation *operation) {
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        liveBlocks.insert(&block);
  });

  llvm::DenseSet<mlir::Block *> trackedBlocks;
  std::vector<PreparedSpatialOwnershipSelection::SourceBlockBinding>
      retainedBindings;
  retainedBindings.reserve(selection.sourceBlocks.size());
  for (const auto &binding : selection.sourceBlocks) {
    if (!liveBlocks.contains(binding.candidateBlock))
      continue;
    if (!trackedBlocks.insert(binding.candidateBlock).second)
      return invalid("call specialization duplicated a live block lineage");
    retainedBindings.push_back(binding);
  }
  if (trackedBlocks.size() != liveBlocks.size())
    return invalid("call specialization created an untracked block lineage");
  selection.sourceBlocks = std::move(retainedBindings);
  return llvm::Error::success();
}

llvm::Expected<MaterializedStructuredOwnershipCandidate>
finalizeStructuredOwnershipCandidate(
    PreparedSpatialOwnershipSelection &prepared) {
  mlir::ModuleOp module = prepared.module.get();
  if (!prepared.materializedSpatialRegion)
    return invalid("ownership decision created no Spatial carrier");
  if (std::optional<std::string> rejection =
          lowering::explainSpatialCarrierParallelRejection(
              prepared.materializedSpatialRegion))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  *rejection);
  if (mlir::failed(mlir::verify(module)))
    return invalid("materialized Structured Program does not verify");

  llvm::DenseSet<mlir::Block *> liveBlocks;
  module->walk([&](mlir::Operation *operation) {
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        liveBlocks.insert(&block);
  });
  std::vector<mlir::Block *> trackedBlocks;
  std::vector<StructuredEntityRef> parentBlocks;
  trackedBlocks.reserve(prepared.sourceBlocks.size());
  parentBlocks.reserve(prepared.sourceBlocks.size());
  for (const PreparedSpatialOwnershipSelection::SourceBlockBinding &binding :
       prepared.sourceBlocks) {
    if (!liveBlocks.contains(binding.candidateBlock))
      continue;
    trackedBlocks.push_back(binding.candidateBlock);
    parentBlocks.push_back(binding.parentBlock);
  }
  auto structured = finalizeStructuredProgramWithTrackedEntities(
      module, trackedBlocks, {prepared.materializedSpatialRegion});
  if (!structured)
    return structured.takeError();
  if (structured->trackedBlocks.size() != parentBlocks.size())
    return invalid("finalized block activity lineage changed cardinality");
  auto structuredView = structured->artifact.view();
  if (!structuredView)
    return structuredView.takeError();
  if (structured->trackedBlocks.size() !=
      structuredView->entities(StructuredEntityKind::Block).size())
    return invalid("finalized block activity lineage is not total");
  if (structured->trackedOperations.size() != 1)
    return invalid("finalized ownership carrier projection is not singular");
  std::vector<StructuredBlockActivityLineage> blockActivityLineage;
  if (!prepared.requiresExactActivityObservations) {
    blockActivityLineage.reserve(parentBlocks.size());
    for (auto [child, parent] :
         llvm::zip_equal(structured->trackedBlocks, parentBlocks))
      blockActivityLineage.push_back({child, parent});
  }
  return MaterializedStructuredOwnershipCandidate{
      std::move(structured->artifact), structured->trackedOperations.front(),
      std::move(blockActivityLineage), std::move(structured->sourceProvenance)};
}

llvm::Expected<llvm::DenseSet<mlir::Operation *>>
deriveDirectCallableClosure(const StructuredProgramCandidate &parent,
                            llvm::ArrayRef<StructuredEntityRef> callableRoots) {
  if (callableRoots.empty())
    return invalid("protocol-rooted ownership requires a nonempty root set");
  auto view = parent.view();
  if (!view)
    return view.takeError();

  llvm::DenseSet<mlir::Operation *> closure;
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> pending;
  for (const StructuredEntityRef &reference : callableRoots) {
    auto entity = view->resolve(reference);
    if (!entity)
      return entity.takeError();
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity->operation);
    if (!function || function.isExternal())
      return invalid(
          "protocol root is not a defined LLVM callable in the parent");
    if (closure.insert(function.getOperation()).second)
      pending.push_back(function);
  }

  for (std::size_t index = 0; index < pending.size(); ++index) {
    pending[index].walk([&](mlir::Operation *operation) {
      mlir::FlatSymbolRefAttr callee;
      if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(operation))
        callee = call.getCalleeAttr();
      else if (auto invoke = llvm::dyn_cast<mlir::LLVM::InvokeOp>(operation))
        callee = invoke.getCalleeAttr();
      if (!callee)
        return;
      auto resolved =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              operation, callee);
      if (!resolved || resolved.isExternal())
        return;
      if (closure.insert(resolved.getOperation()).second)
        pending.push_back(resolved);
    });
  }
  return closure;
}

struct SpatialOwnershipScopeDomainStorage final {
  std::vector<SpatialOwnershipScopeDomainEntry> entries;
  std::vector<std::optional<std::uint64_t>> parentScopeOrdinals;
};

llvm::Expected<SpatialOwnershipScopeDomainStorage>
enumerateSpatialOwnershipScopeDomainImpl(
    const StructuredProgramCandidate &parent,
    const llvm::DenseSet<mlir::Operation *> *callableClosure) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  SpatialOwnershipScopeDomainStorage domain;
  std::vector<mlir::Operation *> scopeOperations;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    if (!entity.operation)
      continue;
    auto enclosingCallable =
        llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (!enclosingCallable)
      enclosingCallable =
          entity.operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (callableClosure &&
        (!enclosingCallable ||
         !callableClosure->contains(enclosingCallable.getOperation())))
      continue;
    if (auto callable =
            llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(entity.operation)) {
      if (callable.isExternal())
        continue;
      SpatialOwnershipScope scope{entity.reference};
      if (std::optional<std::string> rejection =
              detail::explainCallableOwnershipRejection(callable)) {
        domain.entries.push_back(
            RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
      } else if (std::optional<std::string> rejection =
                     detail::explainGraphStructuralOwnershipRejection(
                         parent.module(), entity.operation)) {
        domain.entries.push_back(
            RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
      } else if (std::optional<std::string> rejection =
                     detail::explainAddressStateNormalizationRejection(
                         entity.operation)) {
        domain.entries.push_back(
            RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
      } else {
        domain.entries.push_back(scope);
      }
      scopeOperations.push_back(entity.operation);
      continue;
    }
    OwnershipScopeAnalysis analysis = analyzeOwnershipScope(entity.operation);
    if (analysis.rejection != OwnershipScopeRejection::None &&
        !(analysis.rejection == OwnershipScopeRejection::NoRegion &&
          isExactDirectCallLeafScope(parent.module(), entity.operation)))
      continue;
    SpatialOwnershipScope scope{entity.reference};
    if (std::optional<std::string> rejection =
            detail::explainGraphStructuralOwnershipRejection(
                parent.module(), entity.operation)) {
      domain.entries.push_back(
          RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
    } else if (std::optional<std::string> rejection =
                   detail::explainAddressStateNormalizationRejection(
                       entity.operation)) {
      domain.entries.push_back(
          RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
    } else {
      domain.entries.push_back(scope);
    }
    scopeOperations.push_back(entity.operation);
  }

  llvm::DenseMap<mlir::Operation *, std::uint64_t> ordinalByOperation;
  for (auto [ordinal, operation] : llvm::enumerate(scopeOperations))
    ordinalByOperation.try_emplace(operation, ordinal);
  domain.parentScopeOrdinals.reserve(scopeOperations.size());
  for (mlir::Operation *operation : scopeOperations) {
    std::optional<std::uint64_t> parentScope;
    for (mlir::Operation *ancestor = operation->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      auto found = ordinalByOperation.find(ancestor);
      if (found == ordinalByOperation.end())
        continue;
      parentScope = found->second;
      break;
    }
    domain.parentScopeOrdinals.push_back(parentScope);
  }
  return domain;
}

} // namespace

llvm::Error
materializeOwnedSpatialForallThreadDomain(mlir::scf::ForallOp forall) {
  if (llvm::Error error = materializeOwnedSpatialForallThreadDomainImpl(forall))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  std::move(error));
  return llvm::Error::success();
}

llvm::Expected<std::vector<StructuredEntityRef>>
resolveDefinedLlvmCallables(const StructuredProgramCandidate &parent,
                            llvm::ArrayRef<llvm::StringRef> symbols) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  llvm::StringSet<> seen;
  std::vector<StructuredEntityRef> resolved;
  resolved.reserve(symbols.size());
  for (llvm::StringRef symbol : symbols) {
    if (!seen.insert(symbol).second)
      return invalid("callable symbol list contains a duplicate");
    std::optional<StructuredEntityRef> reference;
    for (const StructuredEntity &entity :
         view->entities(StructuredEntityKind::Operation)) {
      auto function =
          llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
      if (!function || function.getSymName() != symbol)
        continue;
      if (reference)
        return invalid("callable symbol is not unique in the parent");
      if (function.isExternal())
        return invalid("callable symbol resolves only to a declaration");
      reference = entity.reference;
    }
    if (!reference)
      return invalid("callable symbol '" + symbol +
                     "' does not resolve in the parent");
    resolved.push_back(*reference);
  }
  return resolved;
}

llvm::Expected<SpatialOwnershipScopeDomain>
enumerateSpatialOwnershipScopeDomain(const StructuredProgramCandidate &parent) {
  auto storage = enumerateSpatialOwnershipScopeDomainImpl(parent, nullptr);
  if (!storage)
    return storage.takeError();
  SpatialOwnershipScopeDomain domain;
  domain.entries_ = std::move(storage->entries);
  domain.parentScopeOrdinals_ = std::move(storage->parentScopeOrdinals);
  return domain;
}

llvm::Expected<SpatialOwnershipScopeDomain>
enumerateSpatialOwnershipScopeDomain(
    const StructuredProgramCandidate &parent,
    llvm::ArrayRef<StructuredEntityRef> callableRoots) {
  auto closure = deriveDirectCallableClosure(parent, callableRoots);
  if (!closure)
    return closure.takeError();
  auto storage = enumerateSpatialOwnershipScopeDomainImpl(parent, &*closure);
  if (!storage)
    return storage.takeError();
  SpatialOwnershipScopeDomain domain;
  domain.entries_ = std::move(storage->entries);
  domain.parentScopeOrdinals_ = std::move(storage->parentScopeOrdinals);
  return domain;
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
  } else if (!isExactDirectCallLeafScope(parent.module(), operation)) {
    if (auto callable = eligibleOwningCallable(operation); !callable)
      return callable.takeError();
  }

  llvm::SmallVector<std::optional<ForallOwnershipShape>, 2> forallShapes;
  if (llvm::isa<mlir::scf::ForallOp>(operation)) {
    forallShapes.push_back(ForallOwnershipShape::GraphParallel);
    forallShapes.push_back(ForallOwnershipShape::LogicalThreadDomain);
  } else {
    forallShapes.push_back(std::nullopt);
  }

  llvm::SmallVector<std::optional<DirectCallSpecializationShape>, 2>
      callSpecializations;
  callSpecializations.push_back(std::nullopt);
  auto hasCallSpecialization =
      detail::hasUniformExactCallArgumentSpecialization(parent.module(),
                                                        operation);
  if (!hasCallSpecialization)
    return hasCallSpecialization.takeError();
  if (*hasCallSpecialization)
    callSpecializations.push_back(
        DirectCallSpecializationShape::UniformExactConstants);

  llvm::SmallVector<std::optional<DirectCallInliningDecision>, 4> callInlinings;
  callInlinings.push_back(std::nullopt);
  if (std::optional<detail::ExactDirectCallSiteInliningCandidate> directCall =
          detail::findExactDirectCallSiteInliningCandidate(parent.module(),
                                                           operation)) {
    auto candidate =
        llvm::find_if(view->entities(StructuredEntityKind::Operation),
                      [&](const StructuredEntity &entity) {
                        return entity.operation == directCall->callSite;
                      });
    if (candidate == view->entities(StructuredEntityKind::Operation).end())
      return invalid("inlineable direct call has no StructuredEntityRef");
    callInlinings.push_back(DirectCallInliningDecision{candidate->reference});
  }

  std::vector<SpatialOwnershipDecisionPoint> result;
  result.reserve(3 * forallShapes.size() * callSpecializations.size() *
                 callInlinings.size());
  const bool scopeRequiresAddressDecision =
      detail::requiresCanonicalAddressIndexDecision(operation);
  for (const std::optional<DirectCallInliningDecision> &callInlining :
       callInlinings) {
    bool requiresAddressDecision = scopeRequiresAddressDecision;
    if (callInlining) {
      auto callEntity = view->resolve(callInlining->callSite);
      if (!callEntity)
        return callEntity.takeError();
      requiresAddressDecision |= detail::
          exactDirectCallSiteInliningRequiresCanonicalAddressIndexDecision(
              parent.module(), operation, callEntity->operation);
    }
    llvm::SmallVector<std::optional<SpatialAddressProjection>, 3>
        addressProjections;
    if (requiresAddressDecision) {
      std::optional<unsigned> fixedWidth =
          detail::getExplicitFixedAddressIndexWidth(parent.module());
      for (::fabric::ResolvedIndexWidth width :
           ::fabric::resolvedIndexWidthDomain) {
        const unsigned bitWidth = ::fabric::getResolvedIndexBitWidth(width);
        if (!fixedWidth || *fixedWidth == bitWidth)
          addressProjections.push_back(RootRelativeAddressProjection{bitWidth});
      }
      addressProjections.push_back(PointerAddressedAddressProjection{});
    } else {
      addressProjections.push_back(std::nullopt);
    }
    for (const std::optional<SpatialAddressProjection> &addressProjection :
         addressProjections)
      for (std::optional<ForallOwnershipShape> forallShape : forallShapes)
        for (std::optional<DirectCallSpecializationShape> callSpecialization :
             callSpecializations)
          result.push_back(
              SpatialOwnershipDecisionPoint{addressProjection, forallShape,
                                            callSpecialization, callInlining});
  }
  return result;
}

llvm::Expected<MaterializedStructuredOwnershipCandidate>
materializeStructuredSpatialOwnershipDecision(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto prepared = prepareSpatialOwnershipSelection(parent, scope, decision,
                                                   sourceProvenance);
  if (!prepared)
    return prepared.takeError();

  if (auto function =
          llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(prepared->operation)) {
    if (llvm::Error error = materializeThread(*prepared, function))
      return std::move(error);
  } else if (auto forall = llvm::dyn_cast_or_null<mlir::scf::ForallOp>(
                 prepared->operation);
             forall && decision.forallOwnershipShape ==
                           ForallOwnershipShape::LogicalThreadDomain) {
    if (llvm::Error error =
            materializePreparedForallThreadDomain(*prepared, forall))
      return std::move(error);
  } else {
    if (llvm::Error error =
            materializePreparedOperation(*prepared, prepared->operation))
      return std::move(error);
  }
  return finalizeStructuredOwnershipCandidate(*prepared);
}

llvm::Expected<PreparedSpatialOwnershipSelection>
prepareSpatialOwnershipSelection(
    const StructuredProgramCandidate &parent,
    const SpatialOwnershipScope &scope,
    const SpatialOwnershipDecisionPoint &decision,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto domain =
      enumerateSpatialOwnershipDecisionDomain(parent, scope.selection);
  if (!domain)
    return domain.takeError();
  if (llvm::find(*domain, decision) == domain->end()) {
    if (!decision.addressProjection &&
        llvm::all_of(*domain, [](const SpatialOwnershipDecisionPoint &point) {
          return point.addressProjection.has_value();
        }))
      return invalid("selected scope requires an explicit address projection");
    if (!decision.forallOwnershipShape &&
        llvm::all_of(*domain, [](const SpatialOwnershipDecisionPoint &point) {
          return point.forallOwnershipShape.has_value();
        }))
      return invalid("selected forall requires an explicit ownership shape");
    return invalid("decision is not in the selected scope's typed domain");
  }

  auto selection = cloneSelectedOperation(
      parent, scope.selection, decision.directCallInlining, sourceProvenance);
  if (!selection)
    return selection.takeError();
  mlir::Operation *operation = selection->operation;
  const bool directCallRoot =
      isExactDirectCallLeafScope(selection->clone.get(), operation);
  if (directCallRoot && !decision.directCallInlining)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  "direct-call root requires its exact inline coordinate");
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(operation)) {
    if (llvm::Error error = verifyEligibleCallable(function))
      return std::move(error);
  } else if (!directCallRoot) {
    if (auto callable = eligibleOwningCallable(operation); !callable)
      return callable.takeError();
  }
  if (auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(operation);
      forall && decision.forallOwnershipShape ==
                    ForallOwnershipShape::LogicalThreadDomain) {
    unsigned indexWidth = 0;
    const auto *rootRelative = decision.addressProjection
                                   ? std::get_if<RootRelativeAddressProjection>(
                                         &*decision.addressProjection)
                                   : nullptr;
    if (rootRelative) {
      indexWidth = rootRelative->canonicalIndexWidth;
    } else {
      auto resolved = ::loom::getIndexBitWidth(forall);
      if (!resolved)
        return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                      resolved.takeError());
      indexWidth = *resolved;
    }
    if (llvm::Error error = verifyDynamicThreadDomainWidth(forall, indexWidth))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
  }
  if (decision.directCallInlining) {
    const std::size_t sourceBindingCount = selection->sourceBlocks.size();
    auto inlined = detail::materializeExactDirectCallSiteInlining(
        selection->clone.get(), operation, selection->directCallSite);
    if (!inlined)
      return inlined.takeError();
    if (!*inlined)
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    "pinned MLIR inliner rejected the exact direct call");
    operation = (*inlined)->selection;
    if ((*inlined)->exactClosureBlock) {
      auto source = llvm::find_if(
          selection->sourceBlocks,
          [&](const PreparedSpatialOwnershipSelection::SourceBlockBinding
                  &binding) {
            return binding.candidateBlock == operation->getBlock();
          });
      if (source == selection->sourceBlocks.end())
        return invalid("direct-call root has no source block lineage");
      selection->sourceBlocks.push_back(
          {(*inlined)->exactClosureBlock, source->parentBlock});
    }
    if (llvm::Error error = propagateInlinedBlockBindings(
            selection->sourceBlocks, (*inlined)->clonedBlocks,
            sourceBindingCount))
      return std::move(error);
    if (llvm::Error error = retainLiveBlockLineage(*selection))
      return std::move(error);
  }
  if (decision.directCallSpecializationShape) {
    if (*decision.directCallSpecializationShape !=
        DirectCallSpecializationShape::UniformExactConstants)
      return invalid("unknown direct-call specialization shape");
    auto specialized =
        detail::materializeUniformExactCallArgumentSpecialization(
            selection->clone.get(), operation);
    if (!specialized)
      return specialized.takeError();
    if (!*specialized)
      return reject(
          SpatialOwnershipCandidateRejectionKind::NonFinalizable,
          "uniform exact call argument specialization removed the selected "
          "scope");
    operation = **specialized;
    if (llvm::Error error = retainLiveBlockLineage(*selection))
      return std::move(error);
  }
  if (detail::containsGeneralCall(operation))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  "selected scope contains an unresolved general call");
  std::optional<std::string> structuralRejection;
  if (llvm::isa<mlir::scf::ExecuteRegionOp>(operation))
    structuralRejection =
        lowering::explainGraphRegionStructuralRejection(operation, operation);
  else
    structuralRejection =
        lowering::explainGraphRegionStructuralRejection(operation);
  if (structuralRejection)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  *structuralRejection);
  if (llvm::Error error = detail::materializeDataLayoutEndiannessProjection(
          selection->clone.get()))
    return std::move(error);
  llvm::DenseMap<mlir::Block *, std::size_t> blockLineage;
  blockLineage.reserve(selection->sourceBlocks.size());
  for (auto item : llvm::enumerate(selection->sourceBlocks))
    blockLineage.try_emplace(item.value().candidateBlock, item.index());
  auto observeBlockReplacement = [&](mlir::Block *source,
                                     mlir::Block *replacement) -> llvm::Error {
    auto sourceBinding = blockLineage.find(source);
    if (sourceBinding == blockLineage.end())
      return invalid(
          "address normalization replaced a block without source lineage");
    const std::size_t bindingIndex = sourceBinding->second;
    if (blockLineage.contains(replacement))
      return invalid("address normalization reused a live block lineage");
    blockLineage.erase(sourceBinding);
    selection->sourceBlocks[bindingIndex].candidateBlock = replacement;
    blockLineage.try_emplace(replacement, bindingIndex);
    return llvm::Error::success();
  };
  const bool pointerAddressed =
      decision.addressProjection &&
      std::holds_alternative<PointerAddressedAddressProjection>(
          *decision.addressProjection);
  if (!pointerAddressed) {
    std::optional<unsigned> canonicalIndexWidth;
    if (decision.addressProjection)
      canonicalIndexWidth =
          std::get<RootRelativeAddressProjection>(*decision.addressProjection)
              .canonicalIndexWidth;
    auto normalized = detail::materializeAddressIndexContract(
        selection->clone.get(), operation, canonicalIndexWidth,
        observeBlockReplacement);
    if (!normalized) {
      llvm::Error error = normalized.takeError();
      std::optional<std::string> rejection;
      llvm::Error unhandled = llvm::handleErrors(
          std::move(error),
          [&](const detail::AddressIndexContractRejection &failure) {
            rejection = failure.message();
          });
      if (unhandled)
        return std::move(unhandled);
      if (!rejection)
        return invalid("address projection failed without a typed outcome");
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    *rejection);
    }
    operation = *normalized;
  }
  std::vector<mlir::Value> liveIns;
  std::vector<mlir::Value> liveOuts;
  std::vector<mlir::Operation *> callableSpatialBody;
  std::optional<
      std::vector<PreparedSpatialOwnershipSelection::SourceInductionBinding>>
      sourceInductions;
  std::optional<std::vector<mlir::Value>> threadExtents;
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(operation)) {
    detail::CallableOwnershipBoundary boundary =
        detail::deriveCallableOwnershipBoundary(function);
    for (mlir::LLVM::AddressOfOp address : llvm::reverse(boundary.addresses))
      if (&function.getBody().front().front() != address.getOperation())
        address->moveBefore(&function.getBody().front(),
                            function.getBody().front().begin());
    for (mlir::LLVM::UndefOp undef : llvm::reverse(boundary.undefs))
      if (&function.getBody().front().front() != undef.getOperation())
        undef->moveBefore(&function.getBody().front(),
                          function.getBody().front().begin());
    detail::CallableSpatialSlice slice =
        detail::deriveCallableSpatialSlice(function, boundary);
    callableSpatialBody.assign(slice.body.begin(), slice.body.end());
    liveIns.assign(slice.liveIns.begin(), slice.liveIns.end());
    liveOuts.assign(slice.liveOuts.begin(), slice.liveOuts.end());
  } else if (auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(operation);
             forall && decision.forallOwnershipShape ==
                           ForallOwnershipShape::LogicalThreadDomain) {
    auto boundary = deriveForallThreadBoundary(forall);
    if (!boundary)
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    boundary.takeError());
    liveIns.assign(boundary->liveIns.begin(), boundary->liveIns.end());
    sourceInductions = std::move(boundary->sourceInductions);
    auto indexWidth = ::loom::getIndexBitWidth(forall);
    if (!indexWidth)
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    indexWidth.takeError());
    mlir::OpBuilder builder(forall);
    std::vector<mlir::Value> extents;
    extents.reserve(forall.getRank());
    for (auto [lower, upper, step] :
         llvm::zip_equal(forall.getMixedLowerBound(),
                         forall.getMixedUpperBound(), forall.getMixedStep())) {
      auto projection = materializeThreadExtent(builder, forall.getLoc(), lower,
                                                upper, step, *indexWidth);
      if (!projection)
        return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                      projection.takeError());
      extents.push_back(*projection);
    }
    threadExtents = std::move(extents);
  } else {
    OperationClosure closure = deriveOperationClosure(operation);
    liveIns.assign(closure.liveIns.begin(), closure.liveIns.end());
    liveOuts.assign(closure.liveOuts.begin(), closure.liveOuts.end());
  }
  llvm::SmallVector<mlir::Operation *, 1> selectedOperation;
  llvm::ArrayRef<mlir::Operation *> memoryServiceBody = callableSpatialBody;
  if (memoryServiceBody.empty()) {
    selectedOperation.push_back(operation);
    memoryServiceBody = selectedOperation;
  }
  if (std::optional<std::string> rejection =
          detail::explainUnboundMemoryService(memoryServiceBody, liveIns))
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  *rejection);
  if (mlir::failed(mlir::verify(selection->clone.get())))
    return invalid("prepared ownership selection does not verify");
  return PreparedSpatialOwnershipSelection{
      std::move(selection->clone),
      operation,
      std::move(callableSpatialBody),
      std::move(liveIns),
      std::move(liveOuts),
      std::move(sourceInductions),
      std::move(threadExtents),
      nullptr,
      std::move(selection->sourceBlocks),
      decision.directCallInlining.has_value()};
}

} // namespace loom::frontend
