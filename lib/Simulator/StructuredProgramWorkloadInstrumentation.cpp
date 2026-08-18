#include "StructuredProgramNativeExecutionInternal.h"

#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

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

struct ProgramObjectCaptureSite final {
  mlir::Value base;
  mlir::LLVM::GlobalOp global;
  struct ExtentFactor final {
    mlir::Value runtimeValue;
    std::uint64_t fixedValue = 1;
  } extentFactor0, extentFactor1;
  mlir::Operation *registration = nullptr;

  bool isGlobal() const { return static_cast<bool>(global); }
};

std::optional<std::uint64_t> constantUnsignedValue(mlir::Value value) {
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 64)
    return std::nullopt;
  return integer.getValue().getZExtValue();
}

std::optional<std::uint64_t>
fixedTypeByteCountForCapture(mlir::Operation *scope, mlir::Type type) {
  llvm::TypeSize bytes = mlir::DataLayout::closest(scope).getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return std::nullopt;
  return bytes.getFixedValue();
}

std::optional<std::uint64_t>
fixedAllocationByteCountForCapture(mlir::LLVM::AllocaOp allocation) {
  std::optional<std::uint64_t> count =
      constantUnsignedValue(allocation.getArraySize());
  std::optional<std::uint64_t> elementBytes = fixedTypeByteCountForCapture(
      allocation.getOperation(), allocation.getElemType());
  if (!count || *count == 0 || !elementBytes ||
      *count > std::numeric_limits<std::uint64_t>::max() / *elementBytes)
    return std::nullopt;
  return *count * *elementBytes;
}

llvm::Expected<std::vector<ProgramObjectCaptureSite>>
collectProgramObjectCaptureSites(mlir::ModuleOp module) {
  std::vector<mlir::LLVM::AllocaOp> allocations;
  std::vector<mlir::LLVM::GlobalOp> globals;
  std::vector<mlir::LLVM::CallOp> calls;
  module.walk([&](mlir::LLVM::AllocaOp allocation) {
    allocations.push_back(allocation);
  });
  module.walk([&](mlir::LLVM::GlobalOp global) { globals.push_back(global); });
  module.walk([&](mlir::LLVM::CallOp call) { calls.push_back(call); });

  std::vector<ProgramObjectCaptureSite> sites;
  sites.reserve(allocations.size() + globals.size() + calls.size());
  for (mlir::LLVM::AllocaOp allocation : allocations) {
    std::optional<std::uint64_t> byteCount =
        fixedAllocationByteCountForCapture(allocation);
    if (!byteCount)
      continue;
    ProgramObjectCaptureSite site;
    site.base = allocation.getRes();
    site.extentFactor0.fixedValue = *byteCount;
    sites.push_back(std::move(site));
  }
  for (mlir::LLVM::GlobalOp global : globals) {
    if (!global.getValueOrNull() && !global.getInitializerBlock())
      continue;
    std::optional<std::uint64_t> byteCount = fixedTypeByteCountForCapture(
        global.getOperation(), global.getGlobalType());
    if (!byteCount)
      continue;
    ProgramObjectCaptureSite site;
    site.global = global;
    site.extentFactor0.fixedValue = *byteCount;
    sites.push_back(std::move(site));
  }
  for (mlir::LLVM::CallOp call : calls) {
    std::optional<llvm::ArrayRef<std::int32_t>> allocsize = call.getAllocsize();
    if (!allocsize && call.getCalleeAttr()) {
      auto callee =
          mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
              call, call.getCalleeAttr());
      if (callee)
        allocsize = callee.getAllocsize();
    }
    if (!allocsize)
      continue;
    if (allocsize->empty() || allocsize->size() > 2)
      return invalid("allocator allocsize must name one or two operands");
    if (call->getNumResults() != 1 ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(call->getResult(0).getType()))
      continue;

    ProgramObjectCaptureSite site;
    site.base = call->getResult(0);
    ProgramObjectCaptureSite::ExtentFactor *factors[] = {&site.extentFactor0,
                                                         &site.extentFactor1};
    bool representable = true;
    for (auto [factorOrdinal, operandOrdinal] : llvm::enumerate(*allocsize)) {
      if (operandOrdinal < 0 || static_cast<std::size_t>(operandOrdinal) >=
                                    call.getCalleeOperands().size())
        return invalid("allocator allocsize operand is outside the call ABI");
      mlir::Value operand = call.getCalleeOperands()[operandOrdinal];
      auto type = llvm::dyn_cast<mlir::IntegerType>(operand.getType());
      if (!type || type.getWidth() > 64) {
        representable = false;
        break;
      }
      factors[factorOrdinal]->runtimeValue = operand;
    }
    if (representable)
      sites.push_back(std::move(site));
  }
  return sites;
}

mlir::Value programObjectBase(mlir::Value pointer) {
  while (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>())
    pointer = gep.getBase();
  if (pointer.getDefiningOp<mlir::LLVM::AllocaOp>() ||
      pointer.getDefiningOp<mlir::LLVM::AddressOfOp>() ||
      pointer.getDefiningOp<mlir::LLVM::CallOp>())
    return pointer;
  return {};
}

std::optional<std::uint64_t>
pointerRootHint(mlir::Value value,
                const WorkloadBackedSimulationInputCapturePlan &plan,
                llvm::DenseSet<mlir::Value> &visited) {
  if (!value || !visited.insert(value).second)
    return std::nullopt;
  for (const SimulationValueInputCapture &input : plan.valueInputs)
    if (input.pointerTarget && input.boundaryValue == value)
      return input.pointerTarget->memoryRootBindingOrdinal;
  for (auto [ordinal, root] : llvm::enumerate(plan.memoryRoots))
    if (root.boundaryPointer == value)
      return ordinal;

  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    auto spatial = llvm::dyn_cast_or_null<loom::SpatialRegionOp>(
        argument.getOwner()->getParentOp());
    if (spatial && argument.getArgNumber() < spatial->getNumOperands())
      return pointerRootHint(spatial->getOperand(argument.getArgNumber()), plan,
                             visited);
  }

  if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>())
    return pointerRootHint(gep.getBase(), plan, visited);
  if (auto cast = value.getDefiningOp<mlir::LLVM::BitcastOp>())
    return pointerRootHint(cast.getArg(), plan, visited);
  if (auto cast = value.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>())
    return pointerRootHint(cast.getArg(), plan, visited);
  auto commonHint = [&](mlir::Value lhs,
                        mlir::Value rhs) -> std::optional<std::uint64_t> {
    llvm::DenseSet<mlir::Value> lhsVisited(visited);
    llvm::DenseSet<mlir::Value> rhsVisited(visited);
    std::optional<std::uint64_t> left = pointerRootHint(lhs, plan, lhsVisited);
    std::optional<std::uint64_t> right = pointerRootHint(rhs, plan, rhsVisited);
    return left && right && *left == *right ? left : std::nullopt;
  };
  if (auto select = value.getDefiningOp<mlir::arith::SelectOp>())
    return commonHint(select.getTrueValue(), select.getFalseValue());
  if (auto select = value.getDefiningOp<mlir::LLVM::SelectOp>())
    return commonHint(select.getTrueValue(), select.getFalseValue());
  return std::nullopt;
}

std::optional<std::uint64_t>
pointerRootHint(mlir::Value value,
                const WorkloadBackedSimulationInputCapturePlan &plan) {
  llvm::DenseSet<mlir::Value> visited;
  return pointerRootHint(value, plan, visited);
}

} // namespace

llvm::Expected<WorkloadCaptureCallbackNames> instrumentWorkloadBackedCapture(
    mlir::ModuleOp module, mlir::Operation *selectedOperation,
    const WorkloadBackedSimulationInputCapturePlan &plan) {
  if (!selectedOperation ||
      selectedOperation->getParentOfType<mlir::ModuleOp>() != module)
    return invalid("selected capture boundary is not owned by its module");
  auto selectedFunction =
      llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(selectedOperation);
  auto selectedForall = llvm::dyn_cast<mlir::scf::ForallOp>(selectedOperation);
  auto selectedThread =
      selectedOperation->getParentOfType<dataflow::ThreadOp>();
  auto callable =
      selectedFunction
          ? selectedFunction
          : selectedOperation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  mlir::Block *captureEntry = nullptr;
  if (callable && !callable.getBody().empty())
    captureEntry = &callable.getBody().front();
  else if (selectedThread && !selectedThread.getBody().empty())
    captureEntry = &selectedThread.getBody().front();
  if (!captureEntry)
    return invalid("selected capture boundary has no executable owner");
  if (selectedFunction && !callable.getBody().hasOneBlock())
    return unsupported(
        "workload-backed callable capture requires one structured block");

  llvm::SmallVector<mlir::LLVM::StoreOp, 16> originalStores;
  llvm::SmallVector<mlir::LLVM::LoadOp, 8> selectedPointerLoads;
  if (!plan.memoryRoots.empty())
    module.walk(
        [&](mlir::LLVM::StoreOp store) { originalStores.push_back(store); });
  if (!plan.memoryRoots.empty()) {
    selectedOperation->walk([&](mlir::LLVM::LoadOp load) {
      if (llvm::isa<mlir::LLVM::LLVMPointerType>(load.getResult().getType()))
        selectedPointerLoads.push_back(load);
    });
  }

  llvm::Expected<std::vector<ProgramObjectCaptureSite>> objectSites =
      plan.memoryRoots.empty()
          ? llvm::Expected<std::vector<ProgramObjectCaptureSite>>(
                std::vector<ProgramObjectCaptureSite>{})
          : collectProgramObjectCaptureSites(module);
  if (!objectSites)
    return objectSites.takeError();

  mlir::MLIRContext *context = module.getContext();
  mlir::Location location = selectedOperation->getLoc();
  mlir::OpBuilder declarations(context);
  declarations.setInsertionPointToStart(module.getBody());
  const mlir::Type i64 = declarations.getI64Type();
  const mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type voidType = mlir::LLVM::LLVMVoidType::get(context);
  const mlir::Type lifecycleType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {});
  const mlir::Type memoryRootType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, pointer});
  const mlir::Type objectRegistrationType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {pointer, i64, i64});
  const mlir::Type valueType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {i64, pointer, i64});
  const mlir::Type memoryWriteType =
      mlir::LLVM::LLVMFunctionType::get(voidType, {pointer, i64});
  const mlir::Type pointerWriteType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {pointer, pointer, i64, i64, i64, i64});
  const mlir::Type pointerReadType = mlir::LLVM::LLVMFunctionType::get(
      voidType, {pointer, pointer, i64, i64, i64});

  WorkloadCaptureCallbackNames names;
  names.begin = uniqueMlirSymbolName(module, "__loom_workload_capture_begin");
  names.end = uniqueMlirSymbolName(module, "__loom_workload_capture_end");
  if (!objectSites->empty())
    names.registerObject =
        uniqueMlirSymbolName(module, "__loom_workload_capture_register_object");
  if (!plan.denseCoordinates.empty())
    names.coordinate =
        uniqueMlirSymbolName(module, "__loom_workload_capture_coordinate");
  if (!plan.memoryRoots.empty())
    names.memoryRoot =
        uniqueMlirSymbolName(module, "__loom_workload_capture_memory_root");
  if (llvm::any_of(plan.valueInputs, [](const auto &input) {
        return !input.fixedValue.has_value();
      }))
    names.value = uniqueMlirSymbolName(module, "__loom_workload_capture_value");
  if (!plan.streamInputs.empty())
    names.streamInput =
        uniqueMlirSymbolName(module, "__loom_workload_capture_stream_input");
  if (!plan.valueResults.empty())
    names.result =
        uniqueMlirSymbolName(module, "__loom_workload_capture_result");
  if (!plan.streamOutputs.empty())
    names.streamOutput =
        uniqueMlirSymbolName(module, "__loom_workload_capture_stream_output");
  if (llvm::any_of(originalStores, [](mlir::LLVM::StoreOp store) {
        return !llvm::isa<mlir::LLVM::LLVMPointerType>(
            store.getValue().getType());
      }))
    names.memoryWrite =
        uniqueMlirSymbolName(module, "__loom_workload_capture_memory_write");
  if (llvm::any_of(originalStores, [](mlir::LLVM::StoreOp store) {
        return llvm::isa<mlir::LLVM::LLVMPointerType>(
            store.getValue().getType());
      }))
    names.pointerWrite =
        uniqueMlirSymbolName(module, "__loom_workload_capture_pointer_write");
  if (!selectedPointerLoads.empty())
    names.pointerRead =
        uniqueMlirSymbolName(module, "__loom_workload_capture_pointer_read");

  mlir::LLVM::LLVMFuncOp::create(declarations, location, names.begin,
                                 lifecycleType);
  mlir::LLVM::LLVMFuncOp::create(declarations, location, names.end,
                                 lifecycleType);
  if (names.registerObject)
    mlir::LLVM::LLVMFuncOp::create(
        declarations, location, *names.registerObject, objectRegistrationType);
  if (names.coordinate)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.coordinate,
                                   valueType);
  if (names.memoryRoot)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.memoryRoot,
                                   memoryRootType);
  if (names.value)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.value,
                                   valueType);
  if (names.streamInput)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.streamInput,
                                   valueType);
  if (names.result)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.result,
                                   valueType);
  if (names.streamOutput)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.streamOutput,
                                   valueType);
  if (names.memoryWrite)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.memoryWrite,
                                   memoryWriteType);
  if (names.pointerWrite)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.pointerWrite,
                                   pointerWriteType);
  if (names.pointerRead)
    mlir::LLVM::LLVMFuncOp::create(declarations, location, *names.pointerRead,
                                   pointerReadType);

  if (names.registerObject) {
    auto materializeExtentFactor =
        [&](mlir::OpBuilder &builder, mlir::Location factorLocation,
            const ProgramObjectCaptureSite::ExtentFactor &factor) {
          if (!factor.runtimeValue)
            return mlir::Value(mlir::LLVM::ConstantOp::create(
                builder, factorLocation, i64,
                builder.getI64IntegerAttr(factor.fixedValue)));
          auto type =
              llvm::cast<mlir::IntegerType>(factor.runtimeValue.getType());
          if (type.getWidth() == 64)
            return factor.runtimeValue;
          return mlir::Value(mlir::LLVM::ZExtOp::create(
              builder, factorLocation, i64, factor.runtimeValue));
        };
    for (ProgramObjectCaptureSite &site : *objectSites) {
      mlir::OpBuilder registration(context);
      mlir::Location siteLocation = location;
      if (site.isGlobal()) {
        siteLocation = site.global.getLoc();
        registration.setInsertionPointToStart(captureEntry);
        site.base = mlir::LLVM::AddressOfOp::create(registration, siteLocation,
                                                    site.global);
      } else {
        siteLocation = site.base.getLoc();
        registration.setInsertionPointAfter(site.base.getDefiningOp());
      }
      mlir::Value extentFactor0 = materializeExtentFactor(
          registration, siteLocation, site.extentFactor0);
      mlir::Value extentFactor1 = materializeExtentFactor(
          registration, siteLocation, site.extentFactor1);
      auto call = mlir::LLVM::CallOp::create(
          registration, siteLocation, mlir::TypeRange{}, *names.registerObject,
          mlir::ValueRange{site.base, extentFactor0, extentFactor1});
      site.registration = call.getOperation();
    }
  }

  mlir::OpBuilder storage(context);
  storage.setInsertionPointToStart(captureEntry);
  mlir::Value one;
  mlir::Operation *lastStorage = nullptr;
  auto allocateSlot = [&](mlir::Type type) {
    if (!one) {
      one = mlir::LLVM::ConstantOp::create(storage, location, i64,
                                           storage.getI64IntegerAttr(1));
      lastStorage = one.getDefiningOp();
    }
    mlir::Value slot =
        mlir::LLVM::AllocaOp::create(storage, location, pointer, type, one)
            .getRes();
    lastStorage = slot.getDefiningOp();
    return slot;
  };
  auto storageTypeFor =
      [&](mlir::Type semanticType,
          std::uint64_t byteCount) -> llvm::Expected<mlir::Type> {
    if (!semanticType.isIndex())
      return semanticType;
    if (byteCount == 0 || byteCount > std::numeric_limits<unsigned>::max() / 8)
      return invalid("index capture has no finite storage width");
    return mlir::IntegerType::get(context,
                                  static_cast<unsigned>(byteCount * 8));
  };
  auto valueForStorage =
      [&](mlir::OpBuilder &builder, mlir::Value value,
          mlir::Type storageType) -> llvm::Expected<mlir::Value> {
    if (value.getType() == storageType)
      return value;
    if (value.getType().isIndex() && llvm::isa<mlir::IntegerType>(storageType))
      return mlir::arith::IndexCastOp::create(builder, location, storageType,
                                              value)
          .getResult();
    return invalid("capture value has no exact LLVM storage projection");
  };

  std::vector<mlir::Value> coordinateSlots;
  std::vector<mlir::Type> coordinateStorageTypes;
  coordinateSlots.reserve(plan.denseCoordinates.size());
  coordinateStorageTypes.reserve(plan.denseCoordinates.size());
  for (const WorkloadBackedDenseCoordinateCapture &coordinate :
       plan.denseCoordinates) {
    auto storageType = storageTypeFor(coordinate.boundaryValue.getType(),
                                      coordinate.byteCount);
    if (!storageType)
      return storageType.takeError();
    coordinateStorageTypes.push_back(*storageType);
    coordinateSlots.push_back(allocateSlot(*storageType));
  }
  std::vector<mlir::Value> inputSlots(plan.valueInputs.size());
  std::vector<mlir::Type> inputStorageTypes(plan.valueInputs.size());
  for (auto [ordinal, input] : llvm::enumerate(plan.valueInputs)) {
    if (input.fixedValue)
      continue;
    auto storageType =
        storageTypeFor(input.boundaryValue.getType(), input.byteCount);
    if (!storageType)
      return storageType.takeError();
    inputStorageTypes[ordinal] = *storageType;
    inputSlots[ordinal] = allocateSlot(*storageType);
  }
  std::vector<mlir::Value> resultSlots;
  std::vector<mlir::Type> resultStorageTypes;
  resultSlots.reserve(plan.valueResults.size());
  resultStorageTypes.reserve(plan.valueResults.size());
  for (const SimulationValueResultCapture &result : plan.valueResults) {
    auto storageType =
        storageTypeFor(result.boundaryValue.getType(), result.byteCount);
    if (!storageType)
      return storageType.takeError();
    resultStorageTypes.push_back(*storageType);
    resultSlots.push_back(allocateSlot(*storageType));
  }
  std::vector<mlir::Value> streamInputSlots;
  std::vector<mlir::Type> streamInputStorageTypes;
  mlir::OpBuilder streamStorage(context);
  mlir::Value streamStorageCount;
  auto allocateStreamSlot =
      [&](mlir::Type type) -> llvm::Expected<mlir::Value> {
    if (!streamStorageCount) {
      if (selectedFunction || selectedForall) {
        streamStorage.setInsertionPointToStart(captureEntry);
      } else {
        if (selectedOperation->getNumRegions() != 1 ||
            selectedOperation->getRegion(0).empty())
          return invalid("selected stream boundary has no executable region");
        streamStorage.setInsertionPointToStart(
            &selectedOperation->getRegion(0).front());
      }
      streamStorageCount = mlir::LLVM::ConstantOp::create(
          streamStorage, location, i64, streamStorage.getI64IntegerAttr(1));
    }
    return mlir::Value(mlir::LLVM::AllocaOp::create(streamStorage, location,
                                                    pointer, type,
                                                    streamStorageCount)
                           .getRes());
  };
  streamInputSlots.reserve(plan.streamInputs.size());
  streamInputStorageTypes.reserve(plan.streamInputs.size());
  for (const WorkloadBackedStreamCapture &stream : plan.streamInputs) {
    if (stream.endpoints.empty())
      return invalid("graph stream input has no selected endpoint");
    auto channel =
        llvm::dyn_cast<dataflow::ChannelReceiveOp>(stream.endpoints.front());
    if (!channel)
      return invalid("graph stream input endpoint is not a receive");
    auto storageType =
        storageTypeFor(channel.getMessage().getType(), stream.byteCount);
    if (!storageType)
      return storageType.takeError();
    auto slot = allocateStreamSlot(*storageType);
    if (!slot)
      return slot.takeError();
    streamInputStorageTypes.push_back(*storageType);
    streamInputSlots.push_back(*slot);
  }
  std::vector<mlir::Value> streamOutputSlots;
  std::vector<mlir::Type> streamOutputStorageTypes;
  streamOutputSlots.reserve(plan.streamOutputs.size());
  streamOutputStorageTypes.reserve(plan.streamOutputs.size());
  for (const WorkloadBackedStreamCapture &stream : plan.streamOutputs) {
    if (stream.endpoints.empty())
      return invalid("graph stream output has no selected endpoint");
    auto channel =
        llvm::dyn_cast<dataflow::ChannelSendOp>(stream.endpoints.front());
    if (!channel)
      return invalid("graph stream output endpoint is not a send");
    auto storageType =
        storageTypeFor(channel.getMessage().getType(), stream.byteCount);
    if (!storageType)
      return storageType.takeError();
    auto slot = allocateStreamSlot(*storageType);
    if (!slot)
      return slot.takeError();
    streamOutputStorageTypes.push_back(*storageType);
    streamOutputSlots.push_back(*slot);
  }

  mlir::OpBuilder before(context);
  if (selectedFunction) {
    mlir::Operation *lastPrelude = lastStorage;
    mlir::Block *entry = &callable.getBody().front();
    auto includePrelude = [&](mlir::Operation *operation) {
      if (!operation || operation->getBlock() != entry)
        return;
      if (!lastPrelude || lastPrelude->isBeforeInBlock(operation))
        lastPrelude = operation;
    };
    for (const SimulationValueInputCapture &input : plan.valueInputs)
      includePrelude(input.boundaryValue.getDefiningOp());
    for (const WorkloadBackedDenseCoordinateCapture &coordinate :
         plan.denseCoordinates)
      includePrelude(coordinate.boundaryValue.getDefiningOp());
    for (const ProgramObjectCaptureSite &site : *objectSites)
      if (site.isGlobal())
        includePrelude(site.registration);
    for (const WorkloadBackedMemoryRootCapture &root : plan.memoryRoots) {
      includePrelude(root.boundaryPointer.getDefiningOp());
      mlir::Value base = programObjectBase(root.boundaryPointer);
      if (!base)
        continue;
      for (const ProgramObjectCaptureSite &site : *objectSites)
        if (site.base == base)
          includePrelude(site.registration);
    }
    if (lastPrelude)
      before.setInsertionPointAfter(lastPrelude);
    else
      before.setInsertionPointToStart(entry);
  } else if (selectedForall) {
    mlir::Operation *lastPrelude = nullptr;
    mlir::Block *body = selectedForall.getBody();
    auto includePrelude = [&](mlir::Operation *operation) {
      if (!operation || operation->getBlock() != body)
        return;
      if (!lastPrelude || lastPrelude->isBeforeInBlock(operation))
        lastPrelude = operation;
    };
    for (const SimulationValueInputCapture &input : plan.valueInputs)
      includePrelude(input.boundaryValue.getDefiningOp());
    for (const WorkloadBackedDenseCoordinateCapture &coordinate :
         plan.denseCoordinates)
      includePrelude(coordinate.boundaryValue.getDefiningOp());
    for (const WorkloadBackedMemoryRootCapture &root : plan.memoryRoots) {
      includePrelude(root.boundaryPointer.getDefiningOp());
      mlir::Value base = programObjectBase(root.boundaryPointer);
      if (!base)
        continue;
      for (const ProgramObjectCaptureSite &site : *objectSites)
        if (site.base == base)
          includePrelude(site.registration);
    }
    if (lastPrelude)
      before.setInsertionPointAfter(lastPrelude);
    else
      before.setInsertionPointToStart(body);
  } else {
    before.setInsertionPoint(selectedOperation);
  }
  mlir::LLVM::CallOp::create(before, location, mlir::TypeRange{}, names.begin,
                             mlir::ValueRange{});
  for (auto [ordinal, coordinate] : llvm::enumerate(plan.denseCoordinates)) {
    if (!names.coordinate || !coordinate.boundaryValue ||
        coordinate.dimension != ordinal || coordinate.byteCount == 0)
      return invalid("dense coordinate has no finite selected boundary");
    auto storageValue = valueForStorage(before, coordinate.boundaryValue,
                                        coordinateStorageTypes[ordinal]);
    if (!storageValue)
      return storageValue.takeError();
    mlir::LLVM::StoreOp::create(before, location, *storageValue,
                                coordinateSlots[ordinal]);
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(ordinal));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(coordinate.byteCount));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *names.coordinate,
        mlir::ValueRange{ordinalValue, coordinateSlots[ordinal], byteCount});
  }
  for (auto [ordinal, root] : llvm::enumerate(plan.memoryRoots)) {
    if (!names.memoryRoot || !root.boundaryPointer ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(root.boundaryPointer.getType()))
      return invalid("memory root has no pointer-valued selected boundary");
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(ordinal));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *names.memoryRoot,
        mlir::ValueRange{ordinalValue, root.boundaryPointer});
  }
  std::uint64_t runtimeOrdinal = 0;
  for (auto [ordinal, input] : llvm::enumerate(plan.valueInputs)) {
    if (input.fixedValue)
      continue;
    if (!names.value || !input.boundaryValue || input.byteCount == 0)
      return invalid("runtime graph input has no finite selected boundary");
    auto storageValue = valueForStorage(before, input.boundaryValue,
                                        inputStorageTypes[ordinal]);
    if (!storageValue)
      return storageValue.takeError();
    mlir::LLVM::StoreOp::create(before, location, *storageValue,
                                inputSlots[ordinal]);
    mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(runtimeOrdinal));
    mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
        before, location, i64, before.getI64IntegerAttr(input.byteCount));
    mlir::LLVM::CallOp::create(
        before, location, mlir::TypeRange{}, *names.value,
        mlir::ValueRange{ordinalValue, inputSlots[ordinal], byteCount});
    ++runtimeOrdinal;
  }

  auto instrumentStreams =
      [&](llvm::ArrayRef<WorkloadBackedStreamCapture> streams,
          llvm::ArrayRef<mlir::Value> slots,
          llvm::ArrayRef<mlir::Type> storageTypes,
          const std::optional<std::string> &callback,
          bool input) -> llvm::Error {
    if (streams.empty())
      return llvm::Error::success();
    if (!callback)
      return invalid("graph stream has no capture callback");
    for (auto [ordinal, stream] : llvm::enumerate(streams)) {
      if (stream.graphOrdinal != ordinal || stream.byteCount == 0)
        return invalid("graph stream capture is not dense in ABI order");
      for (mlir::Operation *endpoint : stream.endpoints) {
        if (!endpoint || !selectedOperation->isAncestor(endpoint))
          return invalid(
              "graph stream endpoint is outside its selected region");
        mlir::Value message;
        if (input) {
          auto receive = llvm::dyn_cast<dataflow::ChannelReceiveOp>(endpoint);
          if (!receive)
            return invalid("graph stream input endpoint is not a receive");
          message = receive.getMessage();
        } else {
          auto send = llvm::dyn_cast<dataflow::ChannelSendOp>(endpoint);
          if (!send)
            return invalid("graph stream output endpoint is not a send");
          message = send.getMessage();
        }
        mlir::OpBuilder capture(endpoint);
        capture.setInsertionPointAfter(endpoint);
        auto storageValue =
            valueForStorage(capture, message, storageTypes[ordinal]);
        if (!storageValue)
          return storageValue.takeError();
        mlir::LLVM::StoreOp::create(capture, endpoint->getLoc(), *storageValue,
                                    slots[ordinal]);
        mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
            capture, endpoint->getLoc(), i64,
            capture.getI64IntegerAttr(stream.graphOrdinal));
        mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
            capture, endpoint->getLoc(), i64,
            capture.getI64IntegerAttr(stream.byteCount));
        mlir::LLVM::CallOp::create(
            capture, endpoint->getLoc(), mlir::TypeRange{}, *callback,
            mlir::ValueRange{ordinalValue, slots[ordinal], byteCount});
      }
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          instrumentStreams(plan.streamInputs, streamInputSlots,
                            streamInputStorageTypes, names.streamInput, true))
    return std::move(error);
  if (llvm::Error error = instrumentStreams(
          plan.streamOutputs, streamOutputSlots, streamOutputStorageTypes,
          names.streamOutput, false))
    return std::move(error);

  auto emitResultsAndEnd = [&](mlir::OpBuilder &after) -> llvm::Error {
    for (auto [ordinal, result] : llvm::enumerate(plan.valueResults)) {
      if (!names.result || !result.boundaryValue || result.byteCount == 0)
        return invalid("graph result has no finite selected boundary");
      auto storageValue = valueForStorage(after, result.boundaryValue,
                                          resultStorageTypes[ordinal]);
      if (!storageValue)
        return storageValue.takeError();
      mlir::LLVM::StoreOp::create(after, location, *storageValue,
                                  resultSlots[ordinal]);
      mlir::Value ordinalValue = mlir::LLVM::ConstantOp::create(
          after, location, i64, after.getI64IntegerAttr(ordinal));
      mlir::Value byteCount = mlir::LLVM::ConstantOp::create(
          after, location, i64, after.getI64IntegerAttr(result.byteCount));
      mlir::LLVM::CallOp::create(
          after, location, mlir::TypeRange{}, *names.result,
          mlir::ValueRange{ordinalValue, resultSlots[ordinal], byteCount});
    }
    mlir::LLVM::CallOp::create(after, location, mlir::TypeRange{}, names.end,
                               mlir::ValueRange{});
    return llvm::Error::success();
  };

  if (selectedFunction) {
    auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(
        callable.getBody().front().getTerminator());
    if (!returnOp)
      return unsupported(
          "workload-backed callable capture has no direct return");
    mlir::OpBuilder after(returnOp);
    if (llvm::Error error = emitResultsAndEnd(after))
      return std::move(error);
  } else if (selectedForall) {
    mlir::OpBuilder after(selectedForall.getTerminator());
    if (llvm::Error error = emitResultsAndEnd(after))
      return std::move(error);
  } else {
    mlir::OpBuilder after(selectedOperation);
    after.setInsertionPointAfter(selectedOperation);
    if (llvm::Error error = emitResultsAndEnd(after))
      return std::move(error);
  }

  for (mlir::LLVM::StoreOp store : originalStores) {
    auto storageType =
        llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(store.getAddr().getType());
    if (!storageType || storageType.getAddressSpace() != 0)
      return unsupported(
          "workload-backed pointer provenance requires address-space-zero "
          "storage");
    mlir::OpBuilder after(store);
    after.setInsertionPointAfter(store);
    if (auto pointerType = llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(
            store.getValue().getType())) {
      if (!names.pointerWrite || pointerType.getAddressSpace() != 0)
        return unsupported(
            "workload-backed pointer provenance requires address-space-zero "
            "pointer payloads");
      auto layout =
          ::loom::resolvePointerLayout(store, pointerType.getAddressSpace());
      if (!layout)
        return layout.takeError();
      if (layout->kind != ::loom::PointerLayoutKind::StableIntegral ||
          layout->representationBits % 8 != 0)
        return unsupported(
            "workload-backed pointer provenance has no stable-integral "
            "representation");
      mlir::Value addressSpace = mlir::LLVM::ConstantOp::create(
          after, store.getLoc(), i64,
          after.getI64IntegerAttr(pointerType.getAddressSpace()));
      mlir::Value representationBits = mlir::LLVM::ConstantOp::create(
          after, store.getLoc(), i64,
          after.getI64IntegerAttr(layout->representationBits));
      mlir::Value addressBits = mlir::LLVM::ConstantOp::create(
          after, store.getLoc(), i64,
          after.getI64IntegerAttr(layout->addressBits));
      mlir::Value rootHint = mlir::LLVM::ConstantOp::create(
          after, store.getLoc(), i64,
          after.getI64IntegerAttr(
              pointerRootHint(store.getValue(), plan)
                  .value_or(std::numeric_limits<std::uint64_t>::max())));
      mlir::LLVM::CallOp::create(
          after, store.getLoc(), mlir::TypeRange{}, *names.pointerWrite,
          mlir::ValueRange{store.getAddr(), store.getValue(), addressSpace,
                           representationBits, addressBits, rootHint});
      continue;
    }
    if (!names.memoryWrite)
      return invalid("non-pointer store has no memory-write callback");
    llvm::TypeSize byteCount = mlir::DataLayout::closest(store).getTypeSize(
        store.getValue().getType());
    if (byteCount.isScalable() || byteCount.getFixedValue() == 0)
      return unsupported(
          "workload-backed memory write has no fixed nonzero storage size");
    mlir::Value bytes = mlir::LLVM::ConstantOp::create(
        after, store.getLoc(), i64,
        after.getI64IntegerAttr(byteCount.getFixedValue()));
    mlir::LLVM::CallOp::create(after, store.getLoc(), mlir::TypeRange{},
                               *names.memoryWrite,
                               mlir::ValueRange{store.getAddr(), bytes});
  }
  for (mlir::LLVM::LoadOp load : selectedPointerLoads) {
    auto storageType =
        llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(load.getAddr().getType());
    auto pointerType =
        llvm::dyn_cast<mlir::LLVM::LLVMPointerType>(load.getResult().getType());
    if (!names.pointerRead || !storageType || !pointerType ||
        storageType.getAddressSpace() != 0 ||
        pointerType.getAddressSpace() != 0)
      return unsupported(
          "workload-backed pointer reads require address-space-zero storage "
          "and payloads");
    auto layout =
        ::loom::resolvePointerLayout(load, pointerType.getAddressSpace());
    if (!layout)
      return layout.takeError();
    if (layout->kind != ::loom::PointerLayoutKind::StableIntegral ||
        layout->representationBits % 8 != 0)
      return unsupported(
          "workload-backed pointer reads have no stable-integral "
          "representation");
    mlir::OpBuilder after(load);
    after.setInsertionPointAfter(load);
    mlir::Value addressSpace = mlir::LLVM::ConstantOp::create(
        after, load.getLoc(), i64,
        after.getI64IntegerAttr(pointerType.getAddressSpace()));
    mlir::Value representationBits = mlir::LLVM::ConstantOp::create(
        after, load.getLoc(), i64,
        after.getI64IntegerAttr(layout->representationBits));
    mlir::Value addressBits = mlir::LLVM::ConstantOp::create(
        after, load.getLoc(), i64,
        after.getI64IntegerAttr(layout->addressBits));
    mlir::LLVM::CallOp::create(
        after, load.getLoc(), mlir::TypeRange{}, *names.pointerRead,
        mlir::ValueRange{load.getAddr(), load.getResult(), addressSpace,
                         representationBits, addressBits});
  }
  if (mlir::failed(mlir::verify(module)))
    return invalid("workload-backed capture instrumentation does not verify");
  return names;
}

} // namespace loom::sim::native_detail
