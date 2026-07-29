#include "StructuredProgramNativeExecutionInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"

#include <system_error>

namespace loom::sim::native_detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("native_structured_program_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("native_structured_program_unsupported: ") + message);
}

llvm::Error inlineSpatialOwnershipCarriers(mlir::ModuleOp module) {
  llvm::SmallVector<loom::SpatialRegionOp> regions;
  module.walk([&](loom::SpatialRegionOp region) { regions.push_back(region); });
  for (loom::SpatialRegionOp region : llvm::reverse(regions)) {
    if (!region.getStreamInputs().empty() || !region.getStreamOutputs().empty())
      return unsupported(
          "native selected execution does not support stream ownership "
          "carriers");
    if (!region.getBody().hasOneBlock())
      return invalid("selected spatial ownership carrier is not single-block");
    mlir::Block &body = region.getBody().front();
    auto yield = llvm::dyn_cast<loom::SpatialYieldOp>(body.getTerminator());
    if (!yield)
      return invalid("selected spatial ownership carrier has no typed yield");
    if (body.getNumArguments() != region->getNumOperands() ||
        yield->getNumOperands() != region->getNumResults())
      return invalid("selected spatial ownership carrier boundary is not "
                     "positional");

    mlir::IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip_equal(body.getArguments(), region->getOperands()))
      mapping.map(argument, operand);
    mlir::OpBuilder builder(region);
    for (mlir::Operation &operation : body.without_terminator())
      builder.clone(operation, mapping);

    llvm::SmallVector<mlir::Value> results;
    results.reserve(yield->getNumOperands());
    for (mlir::Value value : yield->getOperands())
      results.push_back(mapping.lookupOrDefault(value));
    region->replaceAllUsesWith(results);
    region.erase();
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::string>>
inlineDenseThreadOwnershipCarriers(mlir::ModuleOp module) {
  llvm::SmallVector<dataflow::ThreadLaunchOp> launches;
  module.walk(
      [&](dataflow::ThreadLaunchOp launch) { launches.push_back(launch); });
  std::optional<std::string> invalidExtentCallback;
  for (dataflow::ThreadLaunchOp launch : launches) {
    if (!launch.getAsyncDependencies().empty())
      return unsupported(
          "native selected execution does not project asynchronous thread "
          "dependencies");
    if (!launch.getAsyncToken().hasOneUse())
      return unsupported(
          "native selected execution requires one exact thread wait");
    auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
        *launch.getAsyncToken().getUsers().begin());
    if (!wait || wait->getNumOperands() != 1 ||
        launch->getNextNode() != wait.getOperation())
      return unsupported(
          "native selected execution requires an immediately joined thread "
          "launch");

    auto thread =
        mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr());
    if (!thread || thread.isExternal())
      return invalid("selected thread launch has no exact definition");
    if (thread.getDomain().getKind() !=
        dataflow::ThreadDomainKind::DenseRectangular)
      return unsupported(
          "native selected execution does not support dynamic-work threads");
    mlir::Block &body = thread.getBody().front();
    const std::size_t inputCount = thread.getFunctionType().getNumInputs();
    const std::size_t rank = launch.getGridUpperBounds().size();
    if (body.getNumArguments() != inputCount + 1 + rank ||
        launch.getBodyOperands().size() != inputCount)
      return invalid("selected dense thread boundary is malformed");
    if (!body.getArgument(inputCount).use_empty())
      return unsupported(
          "native selected execution cannot erase a used thread control "
          "token");
    auto yield = llvm::dyn_cast<dataflow::ThreadYieldOp>(body.getTerminator());
    if (!yield || !yield.getCompletionFrontier().empty())
      return unsupported(
          "native selected execution cannot erase a completion frontier");

    mlir::IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip_equal(body.getArguments().take_front(inputCount),
                         launch.getBodyOperands()))
      mapping.map(argument, operand);
    mlir::OpBuilder builder(launch);
    llvm::SmallVector<mlir::Value, 4> coordinates;
    coordinates.reserve(rank);
    if (rank != 0) {
      mlir::Value zero =
          mlir::arith::ConstantIndexOp::create(builder, launch.getLoc(), 0);
      mlir::Value one =
          mlir::arith::ConstantIndexOp::create(builder, launch.getLoc(), 1);
      mlir::Value anyNegative;
      for (mlir::Value extent : launch.getGridUpperBounds()) {
        llvm::APInt constant;
        if (mlir::matchPattern(extent, mlir::m_ConstantInt(&constant))) {
          if (constant.isNegative())
            return invalid("selected dense thread has a negative static "
                           "extent");
          continue;
        }
        mlir::Value negative = mlir::arith::CmpIOp::create(
            builder, launch.getLoc(), mlir::arith::CmpIPredicate::slt, extent,
            zero);
        anyNegative = anyNegative
                          ? mlir::arith::OrIOp::create(builder, launch.getLoc(),
                                                       anyNegative, negative)
                                .getResult()
                          : negative;
      }
      if (anyNegative) {
        if (!invalidExtentCallback) {
          invalidExtentCallback = uniqueMlirSymbolName(
              module, "__loom_invalid_logical_thread_extent");
          mlir::OpBuilder declarations(module.getContext());
          declarations.setInsertionPointToStart(module.getBody());
          mlir::Type type = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(module.getContext()), {});
          mlir::LLVM::LLVMFuncOp::create(declarations, launch.getLoc(),
                                         *invalidExtentCallback, type);
        }
        auto guard = mlir::scf::IfOp::create(
            builder, launch.getLoc(), anyNegative,
            [&](mlir::OpBuilder &bodyBuilder, mlir::Location location) {
              mlir::LLVM::CallOp::create(
                  bodyBuilder, location, mlir::TypeRange{},
                  *invalidExtentCallback, mlir::ValueRange{});
              mlir::scf::YieldOp::create(bodyBuilder, location);
            });
        (void)guard;
      }
      for (mlir::Value extent : launch.getGridUpperBounds()) {
        auto loop = mlir::scf::ForOp::create(builder, launch.getLoc(), zero,
                                             extent, one);
        coordinates.push_back(loop.getInductionVar());
        builder.setInsertionPointToStart(loop.getBody());
      }
    }
    for (auto [argument, coordinate] : llvm::zip_equal(
             body.getArguments().drop_front(inputCount + 1), coordinates))
      mapping.map(argument, coordinate);
    for (mlir::Operation &operation : body.without_terminator())
      builder.clone(operation, mapping);
    wait.erase();
    launch.erase();
  }

  bool residualLaunch = false;
  module.walk([&](dataflow::ThreadLaunchOp) { residualLaunch = true; });
  if (residualLaunch)
    return invalid("selected thread projection left a residual launch");
  llvm::SmallVector<dataflow::ThreadOp> threads;
  module.walk([&](dataflow::ThreadOp thread) { threads.push_back(thread); });
  for (dataflow::ThreadOp thread : threads)
    thread.erase();
  return invalidExtentCallback;
}

} // namespace

llvm::Expected<std::optional<std::string>>
projectSelectedWholeProgram(mlir::ModuleOp module) {
  if (llvm::Error error = inlineSpatialOwnershipCarriers(module))
    return error;
  auto invalidExtentCallback = inlineDenseThreadOwnershipCarriers(module);
  if (!invalidExtentCallback)
    return invalidExtentCallback.takeError();
  bool residualCarrier = false;
  module.walk([&](mlir::Operation *operation) {
    residualCarrier |=
        llvm::isa<loom::SpatialRegionOp, loom::SpatialYieldOp,
                  dataflow::ThreadOp, dataflow::ThreadLaunchOp,
                  dataflow::ThreadWaitOp, dataflow::ThreadYieldOp>(operation);
  });
  if (residualCarrier)
    return invalid("selected whole-program projection left an ownership "
                   "carrier");
  if (mlir::failed(mlir::verify(module)))
    return invalid("selected whole-program projection does not verify");
  return invalidExtentCallback;
}

} // namespace loom::sim::native_detail
