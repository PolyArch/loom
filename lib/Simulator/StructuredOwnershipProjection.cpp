#include "StructuredProgramNativeExecutionInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <system_error>

namespace loom::sim::native_detail {
namespace {

constexpr llvm::StringLiteral receiverOrdinalAttribute =
    "__loom_native_channel_receiver";

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

struct DenseThreadProjection final {
  std::optional<std::string> invalidExtentCallback;
  llvm::DenseMap<mlir::Value, std::uint64_t> receiverCounts;
  llvm::DenseMap<mlir::Value, std::uint64_t> producerMessageCounts;
  llvm::DenseMap<mlir::Value, std::vector<std::uint64_t>> consumerMessageCounts;
};

llvm::Expected<mlir::Value>
resolveDirectChannelActual(dataflow::ThreadLaunchOp launch, mlir::Block &body,
                           std::size_t inputCount, mlir::Value channel,
                           llvm::StringRef endpointKind) {
  auto formal = llvm::dyn_cast<mlir::BlockArgument>(channel);
  if (!formal || formal.getOwner() != &body ||
      formal.getArgNumber() >= inputCount)
    return invalid("thread channel " + endpointKind +
                   " is not bound to an input formal");
  mlir::Value actual = launch.getBodyOperands()[formal.getArgNumber()];
  auto create = actual.getDefiningOp<dataflow::ChannelCreateOp>();
  if (!llvm::isa<dataflow::ChannelType>(actual.getType()) || !create)
    return unsupported("native selected execution requires a direct logical "
                       "channel instance");
  if (create->getBlock() != launch->getBlock() ||
      !create->isBeforeInBlock(launch))
    return unsupported("native selected execution requires channel creation "
                       "and launch in one ordered block");
  return actual;
}

llvm::Error proveSerializedChannelLaunch(
    dataflow::ThreadLaunchOp launch, dataflow::ThreadOp thread,
    mlir::Block &body, std::size_t inputCount,
    llvm::DenseMap<mlir::Value, std::uint64_t> &producedMessages) {
  llvm::SmallVector<dataflow::ChannelSendOp> sends;
  llvm::SmallVector<dataflow::ChannelReceiveOp> receives;
  thread.walk([&](dataflow::ChannelSendOp send) { sends.push_back(send); });
  thread.walk(
      [&](dataflow::ChannelReceiveOp receive) { receives.push_back(receive); });
  if (sends.empty() && receives.empty())
    return llvm::Error::success();
  if (!launch.getGridUpperBounds().empty())
    return unsupported("native selected channel execution requires a "
                       "rank-zero thread launch");
  if (!sends.empty() && !receives.empty())
    return unsupported("native selected execution cannot serialize a thread "
                       "that both sends and receives channels");

  for (dataflow::ChannelSendOp send : sends) {
    if (send->getBlock() != &body)
      return unsupported("native selected execution cannot prove nested "
                         "channel send control flow nonblocking");
    auto actual = resolveDirectChannelActual(launch, body, inputCount,
                                             send.getChannel(), "send");
    if (!actual)
      return actual.takeError();
    ++producedMessages[*actual];
  }

  llvm::DenseMap<unsigned, std::uint64_t> receivesByFormal;
  for (dataflow::ChannelReceiveOp receive : receives) {
    if (receive->getBlock() != &body)
      return unsupported("native selected execution cannot prove nested "
                         "channel receive control flow nonblocking");
    auto actual = resolveDirectChannelActual(launch, body, inputCount,
                                             receive.getChannel(), "receive");
    if (!actual)
      return actual.takeError();
    auto formal = llvm::cast<mlir::BlockArgument>(receive.getChannel());
    ++receivesByFormal[formal.getArgNumber()];
  }
  for (const auto &[formalOrdinal, receiveCount] : receivesByFormal) {
    auto actual = resolveDirectChannelActual(
        launch, body, inputCount, body.getArgument(formalOrdinal), "receive");
    if (!actual)
      return actual.takeError();
    if (producedMessages.lookup(*actual) < receiveCount)
      return unsupported("native selected channel launch can block under its "
                         "serial projection");
  }
  return llvm::Error::success();
}

llvm::Expected<DenseThreadProjection>
inlineDenseThreadOwnershipCarriers(mlir::ModuleOp module) {
  llvm::SmallVector<dataflow::ThreadLaunchOp> launches;
  module.walk(
      [&](dataflow::ThreadLaunchOp launch) { launches.push_back(launch); });
  DenseThreadProjection result;
  llvm::SmallVector<dataflow::ThreadWaitOp> waits;
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
    if (!wait || wait->getBlock() != launch->getBlock())
      return unsupported(
          "native selected execution requires a same-block thread wait");
    for (mlir::Operation *operation = launch->getNextNode();
         operation != wait.getOperation(); operation = operation->getNextNode()) {
      if (!operation)
        return invalid("thread wait does not follow its launch");
      if (!llvm::isa<dataflow::ThreadLaunchOp, dataflow::ThreadWaitOp>(
              operation))
        return unsupported(
            "native selected execution cannot reorder work across a thread "
            "launch batch");
    }
    waits.push_back(wait);
  }

  llvm::SmallPtrSet<mlir::Operation *, 8> erasedWaits;
  for (dataflow::ThreadWaitOp wait : waits)
    if (erasedWaits.insert(wait.getOperation()).second)
      wait.erase();

  for (dataflow::ThreadLaunchOp launch : launches) {
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

    llvm::DenseMap<unsigned, std::uint64_t> receiverOrdinals;
    llvm::SmallVector<dataflow::ChannelReceiveOp> sourceReceives;
    thread.walk([&](dataflow::ChannelReceiveOp receive) {
      sourceReceives.push_back(receive);
    });
    if (llvm::Error error = proveSerializedChannelLaunch(
            launch, thread, body, inputCount, result.producerMessageCounts))
      return std::move(error);
    llvm::DenseMap<unsigned, std::uint64_t> receiveCounts;
    for (dataflow::ChannelReceiveOp receive : sourceReceives) {
      auto formal = llvm::cast<mlir::BlockArgument>(receive.getChannel());
      ++receiveCounts[formal.getArgNumber()];
    }
    for (dataflow::ChannelReceiveOp receive : sourceReceives) {
      auto formal = llvm::dyn_cast<mlir::BlockArgument>(receive.getChannel());
      if (!formal || formal.getOwner() != &body ||
          formal.getArgNumber() >= inputCount)
        return invalid("thread channel receive is not bound to an input formal");
      const unsigned formalOrdinal = formal.getArgNumber();
      if (receiverOrdinals.contains(formalOrdinal))
        continue;
      mlir::Value actual = launch.getBodyOperands()[formalOrdinal];
      if (!llvm::isa<dataflow::ChannelType>(actual.getType()) ||
          !actual.getDefiningOp<dataflow::ChannelCreateOp>())
        return unsupported(
            "native selected execution requires a direct logical channel "
            "instance");
      const std::uint64_t receiverOrdinal = result.receiverCounts[actual]++;
      receiverOrdinals.try_emplace(formalOrdinal, receiverOrdinal);
      std::vector<std::uint64_t> &counts = result.consumerMessageCounts[actual];
      if (counts.size() != receiverOrdinal)
        return invalid("logical channel receiver ordinals are not dense");
      counts.push_back(receiveCounts.lookup(formalOrdinal));
    }

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
        if (!result.invalidExtentCallback) {
          result.invalidExtentCallback = uniqueMlirSymbolName(
              module, "__loom_invalid_logical_thread_extent");
          mlir::OpBuilder declarations(module.getContext());
          declarations.setInsertionPointToStart(module.getBody());
          mlir::Type type = mlir::LLVM::LLVMFunctionType::get(
              mlir::LLVM::LLVMVoidType::get(module.getContext()), {});
          mlir::LLVM::LLVMFuncOp::create(declarations, launch.getLoc(),
                                         *result.invalidExtentCallback, type);
        }
        auto guard = mlir::scf::IfOp::create(
            builder, launch.getLoc(), anyNegative,
            [&](mlir::OpBuilder &bodyBuilder, mlir::Location location) {
              mlir::LLVM::CallOp::create(
                  bodyBuilder, location, mlir::TypeRange{},
                  *result.invalidExtentCallback, mlir::ValueRange{});
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
    for (dataflow::ChannelReceiveOp receive : sourceReceives) {
      auto formal = llvm::cast<mlir::BlockArgument>(receive.getChannel());
      mlir::Operation *cloned = mapping.lookupOrNull(receive.getOperation());
      auto ordinal = receiverOrdinals.find(formal.getArgNumber());
      if (!cloned || ordinal == receiverOrdinals.end())
        return invalid("thread channel receive was not mapped into the clone");
      cloned->setAttr(receiverOrdinalAttribute,
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(module.getContext(), 64),
                          ordinal->second));
    }
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
  return result;
}

llvm::Expected<std::uint64_t> fixedTypeByteCount(mlir::Operation *operation,
                                                 mlir::Type type) {
  llvm::TypeSize bytes = mlir::DataLayout::closest(operation).getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return unsupported("logical channel payload has no fixed storage size");
  return bytes.getFixedValue();
}

llvm::Expected<mlir::Value> allocateMessageSlot(mlir::Operation *operation,
                                                mlir::Type type) {
  auto function = operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!function || function.getBody().empty())
    return invalid("logical channel operation has no executable owner");
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(&function.getBody().front());
  mlir::Type i64 = builder.getI64Type();
  mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value one = mlir::LLVM::ConstantOp::create(
      builder, operation->getLoc(), i64, builder.getI64IntegerAttr(1));
  return mlir::LLVM::AllocaOp::create(builder, operation->getLoc(), pointer,
                                     type, one)
      .getRes();
}

llvm::Expected<std::optional<NativeChannelCallbackNames>>
lowerLogicalChannels(mlir::ModuleOp module,
                     const DenseThreadProjection &projection) {
  llvm::SmallVector<dataflow::ChannelCreateOp> creates;
  llvm::SmallVector<dataflow::ChannelSendOp> sends;
  llvm::SmallVector<dataflow::ChannelReceiveOp> receives;
  module.walk([&](dataflow::ChannelCreateOp op) { creates.push_back(op); });
  module.walk([&](dataflow::ChannelSendOp op) { sends.push_back(op); });
  module.walk([&](dataflow::ChannelReceiveOp op) { receives.push_back(op); });
  if (creates.empty()) {
    if (!sends.empty() || !receives.empty() ||
        !projection.receiverCounts.empty() ||
        !projection.producerMessageCounts.empty() ||
        !projection.consumerMessageCounts.empty())
      return invalid("logical channel endpoints have no channel instance");
    return std::optional<NativeChannelCallbackNames>{};
  }

  NativeChannelCallbackNames names{
      uniqueMlirSymbolName(module, "__loom_logical_channel_create"),
      uniqueMlirSymbolName(module, "__loom_logical_channel_rate"),
      uniqueMlirSymbolName(module, "__loom_logical_channel_send"),
      uniqueMlirSymbolName(module, "__loom_logical_channel_receive")};
  mlir::OpBuilder declarations(module.getContext());
  declarations.setInsertionPointToStart(module.getBody());
  mlir::Type i64 = declarations.getI64Type();
  mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(module.getContext());
  mlir::LLVM::LLVMFuncOp::create(
      declarations, module.getLoc(), names.create,
      mlir::LLVM::LLVMFunctionType::get(i64, {i64}));
  mlir::LLVM::LLVMFuncOp::create(
      declarations, module.getLoc(), names.rate,
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, i64, i64, i64}));
  mlir::LLVM::LLVMFuncOp::create(
      declarations, module.getLoc(), names.send,
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, pointer, i64}));
  mlir::LLVM::LLVMFuncOp::create(
      declarations, module.getLoc(), names.receive,
      mlir::LLVM::LLVMFunctionType::get(voidType,
                                        {i64, i64, pointer, i64}));

  llvm::DenseMap<mlir::Value, mlir::Value> handles;
  for (dataflow::ChannelCreateOp create : creates) {
    auto count = projection.receiverCounts.find(create.getChannel());
    auto producer = projection.producerMessageCounts.find(create.getChannel());
    auto consumers = projection.consumerMessageCounts.find(create.getChannel());
    if (count == projection.receiverCounts.end() || count->second == 0 ||
        producer == projection.producerMessageCounts.end() ||
        consumers == projection.consumerMessageCounts.end() ||
        consumers->second.size() != count->second)
      return invalid("logical channel has no receiver endpoint");
    mlir::OpBuilder builder(create);
    mlir::Value countValue = mlir::LLVM::ConstantOp::create(
        builder, create.getLoc(), i64,
        builder.getI64IntegerAttr(count->second));
    auto call = mlir::LLVM::CallOp::create(
        builder, create.getLoc(), mlir::TypeRange{i64}, names.create,
        mlir::ValueRange{countValue});
    handles.try_emplace(create.getChannel(), call.getResult());
    mlir::Value producerCount = mlir::LLVM::ConstantOp::create(
        builder, create.getLoc(), i64,
        builder.getI64IntegerAttr(producer->second));
    for (const auto indexed : llvm::enumerate(consumers->second)) {
      mlir::Value consumerOrdinal = mlir::LLVM::ConstantOp::create(
          builder, create.getLoc(), i64,
          builder.getI64IntegerAttr(indexed.index()));
      mlir::Value consumerCount = mlir::LLVM::ConstantOp::create(
          builder, create.getLoc(), i64,
          builder.getI64IntegerAttr(indexed.value()));
      mlir::LLVM::CallOp::create(
          builder, create.getLoc(), mlir::TypeRange{}, names.rate,
          mlir::ValueRange{call.getResult(), producerCount, consumerOrdinal,
                           consumerCount});
    }
  }

  for (dataflow::ChannelSendOp send : sends) {
    auto handle = handles.find(send.getChannel());
    if (handle == handles.end())
      return invalid("logical channel send has no exact channel handle");
    auto bytes = fixedTypeByteCount(send, send.getMessage().getType());
    if (!bytes)
      return bytes.takeError();
    auto slot = allocateMessageSlot(send, send.getMessage().getType());
    if (!slot)
      return slot.takeError();
    mlir::OpBuilder builder(send);
    mlir::LLVM::StoreOp::create(builder, send.getLoc(), send.getMessage(),
                                *slot);
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        builder, send.getLoc(), i64, builder.getI64IntegerAttr(*bytes));
    mlir::LLVM::CallOp::create(builder, send.getLoc(), mlir::TypeRange{},
                               names.send,
                               mlir::ValueRange{handle->second, *slot,
                                                byteCount});
    send.erase();
  }

  for (dataflow::ChannelReceiveOp receive : receives) {
    auto handle = handles.find(receive.getChannel());
    auto ordinal = receive->getAttrOfType<mlir::IntegerAttr>(
        receiverOrdinalAttribute);
    if (handle == handles.end() || !ordinal || ordinal.getValue().isNegative())
      return invalid("logical channel receive has no exact receiver endpoint");
    auto bytes = fixedTypeByteCount(receive, receive.getMessage().getType());
    if (!bytes)
      return bytes.takeError();
    auto slot = allocateMessageSlot(receive, receive.getMessage().getType());
    if (!slot)
      return slot.takeError();
    mlir::OpBuilder builder(receive);
    mlir::Value receiverOrdinal = mlir::LLVM::ConstantOp::create(
        builder, receive.getLoc(), i64, ordinal);
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        builder, receive.getLoc(), i64, builder.getI64IntegerAttr(*bytes));
    mlir::LLVM::CallOp::create(
        builder, receive.getLoc(), mlir::TypeRange{}, names.receive,
        mlir::ValueRange{handle->second, receiverOrdinal, *slot, byteCount});
    auto value = mlir::LLVM::LoadOp::create(
        builder, receive.getLoc(), receive.getMessage().getType(), *slot);
    receive.getMessage().replaceAllUsesWith(value.getRes());
    receive.erase();
  }
  for (dataflow::ChannelCreateOp create : creates) {
    if (!create.getChannel().use_empty())
      return invalid("logical channel instance retained an unlowered use");
    create.erase();
  }
  return std::optional<NativeChannelCallbackNames>(std::move(names));
}

} // namespace

llvm::Expected<SelectedWholeProgramProjection>
projectSelectedWholeProgram(mlir::ModuleOp module) {
  if (llvm::Error error = inlineSpatialOwnershipCarriers(module))
    return error;
  auto threads = inlineDenseThreadOwnershipCarriers(module);
  if (!threads)
    return threads.takeError();
  auto channels = lowerLogicalChannels(module, *threads);
  if (!channels)
    return channels.takeError();
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
  return SelectedWholeProgramProjection{
      std::move(threads->invalidExtentCallback), std::move(*channels)};
}

} // namespace loom::sim::native_detail
