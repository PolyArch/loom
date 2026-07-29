#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "StructuredAddressIndexNarrowing.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/LoomOps.h"
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
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
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

std::optional<std::string>
callableOwnershipRejection(mlir::LLVM::LLVMFuncOp function) {
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

  mlir::Operation *nestedCall = nullptr;
  function.getBody().walk([&](mlir::Operation *operation) {
    if (llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::InvokeOp>(operation)) {
      nestedCall = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (nestedCall)
    return "selected callable contains an unresolved nested call";
  return std::nullopt;
}

llvm::Error verifyEligibleCallable(mlir::LLVM::LLVMFuncOp function) {
  if (std::optional<std::string> rejection =
          callableOwnershipRejection(function))
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

struct CallableOwnershipBoundary final {
  llvm::SmallVector<mlir::LLVM::AddressOfOp, 4> addresses;
  llvm::SmallVector<mlir::LLVM::UndefOp, 4> undefs;
  llvm::SmallVector<mlir::Value, 8> inputs;
  llvm::SmallVector<mlir::Value, 1> outputs;
};

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
  boundary.inputs.append(entry.getArguments().begin(),
                         entry.getArguments().end());
  for (mlir::LLVM::AddressOfOp address : boundary.addresses)
    boundary.inputs.push_back(address.getRes());
  for (mlir::LLVM::UndefOp undef : boundary.undefs)
    boundary.inputs.push_back(undef.getRes());
  auto returnOp = llvm::cast<mlir::LLVM::ReturnOp>(entry.getTerminator());
  boundary.outputs.append(returnOp.getOperands().begin(),
                          returnOp.getOperands().end());
  return boundary;
}

llvm::Error materializeThread(mlir::ModuleOp module,
                              mlir::LLVM::LLVMFuncOp function) {
  if (llvm::Error error = verifyEligibleCallable(function))
    return error;
  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = function.getLoc();
  mlir::Block &source = function.getBody().front();
  auto sourceReturn = llvm::cast<mlir::LLVM::ReturnOp>(source.getTerminator());

  // A global address is resolved by the stored-program launch owner and
  // crosses the thread ABI as a memory capability. It is not a SpatialCore
  // actor. Hoist nested address leaves to the wrapper entry so every retained
  // leaf has one explicit launch binding and can be removed from the cloned
  // graph body.
  CallableOwnershipBoundary callableBoundary =
      deriveCallableOwnershipBoundary(function);
  for (mlir::LLVM::AddressOfOp address :
       llvm::reverse(callableBoundary.addresses))
    if (&source.front() != address.getOperation())
      address->moveBefore(&source, source.begin());
  for (mlir::LLVM::UndefOp undef : llvm::reverse(callableBoundary.undefs))
    if (&source.front() != undef.getOperation())
      undef->moveBefore(&source, source.begin());

  llvm::SmallVector<mlir::Operation *, 16> selectedBody;
  for (mlir::Operation &operation : source.without_terminator())
    if (!llvm::isa<mlir::LLVM::AddressOfOp, mlir::LLVM::UndefOp>(operation))
      selectedBody.push_back(&operation);
  llvm::SmallVector<mlir::Value, 1> yieldedValues(
      callableBoundary.outputs.begin(), callableBoundary.outputs.end());
  llvm::SmallVector<mlir::Type, 1> resultTypes;
  for (mlir::Value value : yieldedValues)
    resultTypes.push_back(value.getType());
  auto resultSlots =
      createCallerOwnedResultStorage(function, resultTypes, location);
  if (!resultSlots)
    return resultSlots.takeError();

  llvm::SmallVector<mlir::Value, 8> captures(callableBoundary.inputs.begin(),
                                             callableBoundary.inputs.end());
  auto boundary =
      createSpatialThreadBoundary(module, function, captures, *resultSlots,
                                  resultTypes, mlir::TypeRange{}, location);
  if (!boundary)
    return boundary.takeError();

  mlir::OpBuilder builder(context);
  mlir::IRMapping mapping;
  for (auto [index, capture] : llvm::enumerate(captures))
    mapping.map(capture, boundary->captureArguments[index]);
  builder.setInsertionPointToEnd(boundary->spatialEntry);
  for (mlir::Operation *operation : selectedBody)
    builder.clone(*operation, mapping);
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
  llvm::SmallVector<mlir::Value, 1> returnedValues;
  for (auto [slot, type] : llvm::zip_equal(*resultSlots, resultTypes))
    returnedValues.push_back(
        mlir::LLVM::LoadOp::create(builder, location, type, slot));
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

  mlir::Location location = forall.getLoc();
  auto boundary = createSpatialThreadBoundary(
      module, *callable, prepared.liveIns, mlir::ValueRange{},
      mlir::TypeRange{}, mlir::TypeRange{}, location, forall.getRank());
  if (!boundary)
    return boundary.takeError();

  mlir::IRMapping mapping;
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

llvm::Error materializePreparedOperation(mlir::ModuleOp module,
                                         mlir::Operation *operation) {
  auto callable = eligibleOwningCallable(operation);
  if (!callable)
    return callable.takeError();
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
    return resultSlots.takeError();
  auto boundary = createSpatialThreadBoundary(
      module, *callable, closure.liveIns, *resultSlots, valueResultTypes,
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

llvm::Error requireExactFabricCapabilities(
    const dataflow::CanonicalDataflowArtifact &program,
    const fabric::FinalizedFabricRoot &fabric) {
  auto view = program.view();
  if (!view)
    return view.takeError();
  if (view->graphs().empty() || view->actors().empty())
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  "materialized candidate has no SpatialCore workload");

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

llvm::Expected<SpatialOwnershipScopeDomain>
enumerateSpatialOwnershipScopeDomain(const StructuredProgramCandidate &parent) {
  auto view = parent.view();
  if (!view)
    return view.takeError();

  SpatialOwnershipScopeDomain domain;
  std::vector<mlir::Operation *> scopeOperations;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    if (!entity.operation)
      continue;
    if (auto callable =
            llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(entity.operation)) {
      if (callable.isExternal())
        continue;
      SpatialOwnershipScope scope{entity.reference};
      if (std::optional<std::string> rejection =
              callableOwnershipRejection(callable)) {
        domain.entries_.push_back(
            RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
      } else if (std::optional<std::string> rejection =
                     detail::explainAddressStateNormalizationRejection(
                         entity.operation)) {
        domain.entries_.push_back(
            RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
      } else {
        domain.entries_.push_back(scope);
      }
      scopeOperations.push_back(entity.operation);
      continue;
    }
    if (analyzeOwnershipScope(entity.operation).rejection !=
        OwnershipScopeRejection::None)
      continue;
    SpatialOwnershipScope scope{entity.reference};
    if (std::optional<std::string> rejection =
            lowering::explainGraphRegionStructuralRejection(entity.operation)) {
      domain.entries_.push_back(
          RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
    } else if (std::optional<std::string> rejection =
                   detail::explainAddressStateNormalizationRejection(
                       entity.operation)) {
      domain.entries_.push_back(
          RejectedSpatialOwnershipScope{scope, std::move(*rejection)});
    } else {
      domain.entries_.push_back(scope);
    }
    scopeOperations.push_back(entity.operation);
  }

  llvm::DenseMap<mlir::Operation *, std::uint64_t> ordinalByOperation;
  for (auto [ordinal, operation] : llvm::enumerate(scopeOperations))
    ordinalByOperation.try_emplace(operation, ordinal);
  domain.parentScopeOrdinals_.reserve(scopeOperations.size());
  for (mlir::Operation *operation : scopeOperations) {
    std::optional<std::uint64_t> parent;
    for (mlir::Operation *ancestor = operation->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      auto found = ordinalByOperation.find(ancestor);
      if (found == ordinalByOperation.end())
        continue;
      parent = found->second;
      break;
    }
    domain.parentScopeOrdinals_.push_back(parent);
  }
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

  llvm::SmallVector<std::optional<ForallOwnershipShape>, 2> forallShapes;
  if (llvm::isa<mlir::scf::ForallOp>(operation)) {
    forallShapes.push_back(ForallOwnershipShape::GraphParallel);
    forallShapes.push_back(ForallOwnershipShape::LogicalThreadDomain);
  } else {
    forallShapes.push_back(std::nullopt);
  }

  std::vector<SpatialOwnershipDecisionPoint> result;
  result.reserve(indexWidths.size() * executionShapes.size() *
                 forallShapes.size());
  for (std::optional<unsigned> indexWidth : indexWidths)
    for (std::optional<raising::FMulAddExecutionShape> shape : executionShapes)
      for (std::optional<ForallOwnershipShape> forallShape : forallShapes)
        result.push_back(
            SpatialOwnershipDecisionPoint{shape, indexWidth, forallShape});
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

  if (auto function =
          llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(prepared->operation)) {
    if (llvm::Error error = materializeThread(prepared->module.get(), function))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
  } else if (auto forall = llvm::dyn_cast_or_null<mlir::scf::ForallOp>(
                 prepared->operation);
             forall && decision.forallOwnershipShape ==
                           ForallOwnershipShape::LogicalThreadDomain) {
    if (llvm::Error error =
            materializePreparedForallThreadDomain(*prepared, forall))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
  } else {
    if (llvm::Error error = materializePreparedOperation(prepared->module.get(),
                                                         prepared->operation))
      return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                    std::move(error));
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
    if (!decision.forallOwnershipShape &&
        llvm::all_of(*domain, [](const SpatialOwnershipDecisionPoint &point) {
          return point.forallOwnershipShape.has_value();
        }))
      return invalid("selected forall requires an explicit ownership shape");
    return invalid("decision is not in the selected scope's typed domain");
  }

  auto selection = cloneSelectedOperation(parent, scope.selection);
  if (!selection)
    return selection.takeError();
  mlir::Operation *operation = selection->operation;
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(operation)) {
    if (llvm::Error error = verifyEligibleCallable(function))
      return std::move(error);
  } else {
    if (auto callable = eligibleOwningCallable(operation); !callable)
      return callable.takeError();
  }
  if (auto forall = llvm::dyn_cast<mlir::scf::ForallOp>(operation);
      forall && decision.forallOwnershipShape ==
                    ForallOwnershipShape::LogicalThreadDomain) {
    unsigned indexWidth = 0;
    if (decision.canonicalIndexWidth) {
      indexWidth = *decision.canonicalIndexWidth;
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
  if (llvm::Error error = detail::materializeDataLayoutEndiannessProjection(
          selection->clone.get()))
    return std::move(error);
  auto normalized = detail::materializeAddressIndexContract(
      selection->clone.get(), operation, decision.canonicalIndexWidth);
  if (!normalized)
    return reject(SpatialOwnershipCandidateRejectionKind::NonFinalizable,
                  normalized.takeError());
  operation = *normalized;
  if (decision.fmuladdExecutionShape)
    raising::materializeFMulAddInOperation(*operation,
                                           *decision.fmuladdExecutionShape);
  std::vector<mlir::Value> liveIns;
  std::vector<mlir::Value> liveOuts;
  std::optional<
      std::vector<PreparedSpatialOwnershipSelection::SourceInductionBinding>>
      sourceInductions;
  std::optional<std::vector<mlir::Value>> threadExtents;
  if (auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(operation)) {
    CallableOwnershipBoundary boundary =
        deriveCallableOwnershipBoundary(function);
    liveIns.assign(boundary.inputs.begin(), boundary.inputs.end());
    liveOuts.assign(boundary.outputs.begin(), boundary.outputs.end());
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
  if (mlir::failed(mlir::verify(selection->clone.get())))
    return invalid("prepared ownership selection does not verify");
  return PreparedSpatialOwnershipSelection{
      std::move(selection->clone), operation,
      std::move(liveIns),          std::move(liveOuts),
      std::move(sourceInductions), std::move(threadExtents)};
}

llvm::Expected<MaterializedOwnershipCandidate>
materializeSpatialOwnership(const StructuredProgramCandidate &parent,
                            const StructuredEntityRef &selection,
                            const fabric::FinalizedFabricRoot &fabric,
                            const SpatialOwnershipOptions &options) {
  return materializeSpatialOwnershipDecision(parent, {selection},
                                             {options.fmuladdExecutionShape,
                                              options.canonicalIndexWidth,
                                              options.forallOwnershipShape},
                                             fabric, options.lowering);
}

} // namespace loom::frontend
