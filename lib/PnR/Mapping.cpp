#include "PnR/Mapping.h"
#include "MappingInternal.h"

#include "Common/IndexWidth.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/Elaboration.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace loom::pnr::detail {

namespace {

struct HardwareSelection {
  mlir::Operation *module = nullptr;
  std::string moduleName;
  std::string rootKind;
  std::string systemName;
  std::string accCoreName;
  std::string summaryHardware;
};

std::string escapedIdentityPart(llvm::StringRef value) {
  std::string part;
  constexpr char hex[] = "0123456789ABCDEF";
  for (char ch : value) {
    unsigned char byte = static_cast<unsigned char>(ch);
    if (llvm::isAlnum(ch) || ch == '_') {
      part.push_back(ch);
      continue;
    }
    part.push_back('%');
    part.push_back(hex[(byte >> 4) & 0xF]);
    part.push_back(hex[byte & 0xF]);
  }
  return part;
}

std::string mappingId(llvm::StringRef workload, llvm::StringRef graph,
                      llvm::StringRef hardware) {
  std::string id = escapedIdentityPart(workload);
  id += "__";
  id += escapedIdentityPart(graph);
  id += "__";
  id += escapedIdentityPart(hardware);
  return id;
}

std::string systemCoreHardwareIdentity(llvm::StringRef systemName,
                                       llvm::StringRef accCoreName) {
  return (systemName + "::" + accCoreName).str();
}

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::ub::UBDialect>();
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(mlir::MLIRContext &context,
                                              llvm::StringRef path) {
  return mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
}

std::optional<std::string> symbolName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return std::nullopt;
}

mlir::Operation *findSymbolOp(mlir::ModuleOp module, llvm::StringRef opName,
                              llvm::StringRef symbol) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (found || op->getName().getStringRef() != opName)
      return;
    std::optional<std::string> name = symbolName(op);
    if (name && *name == symbol)
      found = op;
  });
  return found;
}

llvm::Expected<std::string> normalizeHardwareRootKind(llvm::StringRef kind) {
  if (kind.empty() || kind == "module" || kind == "fabric.module")
    return std::string("fabric.module");
  if (kind == "system" || kind == "fabric.system")
    return std::string("fabric.system");
  return llvm::createStringError(
      std::errc::invalid_argument,
      "hardware root kind must be module or system, got %s",
      kind.str().c_str());
}

llvm::Expected<HardwareSelection>
selectFabricHardware(mlir::ModuleOp module, const MappingOptions &options) {
  auto rootKindOrErr = normalizeHardwareRootKind(options.hardwareRootKind);
  if (!rootKindOrErr)
    return rootKindOrErr.takeError();

  if (*rootKindOrErr == "fabric.module") {
    if (!options.accCoreName.empty())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "--acc-core requires --hardware-root-kind system");
    mlir::Operation *hardwareOp =
        findSymbolOp(module, "fabric.module", options.hardwareName);
    if (!hardwareOp)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "could not find fabric hardware %s",
                                     options.hardwareName.c_str());
    return HardwareSelection{hardwareOp, options.hardwareName, *rootKindOrErr,
                             "", "", options.hardwareName};
  }

  if (options.accCoreName.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s requires --acc-core",
        options.hardwareName.c_str());

  mlir::Operation *systemOp =
      findSymbolOp(module, "fabric.system", options.hardwareName);
  if (!systemOp)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find fabric system %s",
                                   options.hardwareName.c_str());

  fabric::NodeOp selectedNode;
  systemOp->walk([&](fabric::NodeOp node) {
    if (selectedNode)
      return;
    if (node.getSymName() == options.accCoreName)
      selectedNode = node;
  });
  if (!selectedNode || selectedNode.getKind() != "acc_core")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s does not contain acc_core %s",
        options.hardwareName.c_str(), options.accCoreName.c_str());

  mlir::FlatSymbolRefAttr spatial = selectedNode.getSpatialAttr();
  if (!spatial)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s acc_core %s lacks spatial fabric.module reference",
        options.hardwareName.c_str(), options.accCoreName.c_str());

  std::string spatialName = spatial.getValue().str();
  mlir::Operation *moduleOp =
      findSymbolOp(module, "fabric.module", spatialName);
  if (!moduleOp)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "system hardware %s acc_core %s references missing fabric.module %s",
        options.hardwareName.c_str(), options.accCoreName.c_str(),
        spatialName.c_str());

  return HardwareSelection{
      moduleOp, spatialName, *rootKindOrErr, options.hardwareName,
      options.accCoreName,
      systemCoreHardwareIdentity(options.hardwareName, options.accCoreName)};
}

bool isForControlOperandUse(mlir::OpOperand &use) {
  return mlir::isa<mlir::scf::ForOp>(use.getOwner()) &&
         use.getOperandNumber() <= 2;
}

bool isStructuredControlConstant(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name != "dataflow.constant" && name != "arith.constant")
    return false;
  if (op->getNumResults() != 1)
    return false;
  mlir::Value result = op->getResult(0);
  if (result.use_empty())
    return false;
  return llvm::all_of(result.getUses(), isForControlOperandUse);
}

bool isIgnoredOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "dataflow.graph.return")
    return true;
  return isStructuredControlConstant(op);
}

} // namespace

bool isAdapterOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "builtin.unrealized_conversion_cast" ||
         name == "arith.index_cast" || name == "arith.index_castui";
}

namespace {

bool isLlvmPointerType(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

std::optional<unsigned> adapterBitWidth(mlir::Type type) {
  if (!mlir::isa<mlir::IntegerType, mlir::IndexType>(type))
    return std::nullopt;
  std::string error;
  auto width = fabric::getSemanticPayloadWidth(type, error);
  if (mlir::failed(width))
    return std::nullopt;
  return *width;
}

bool isDataflowMemoryAddressUse(mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  llvm::StringRef name = owner->getName().getStringRef();
  return (name == "dataflow.load" || name == "dataflow.store") &&
         use.getOperandNumber() == 1;
}

bool isAddressArithmeticOp(mlir::Operation *op) {
  if (!op || op->getNumResults() != 1)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.shli" ||
         name == "arith.shrsi" || name == "arith.shrui" ||
         name == "arith.andi" || name == "arith.ori" ||
         name == "arith.xori";
}

bool isAddressShiftOp(mlir::Operation *op) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == "arith.shli" || name == "arith.shrsi" ||
         name == "arith.shrui";
}

bool valueFeedsOnlyDirectMemoryAddress(mlir::Value value) {
  if (value.use_empty())
    return false;
  for (mlir::OpOperand &use : value.getUses())
    if (!isDataflowMemoryAddressUse(use))
      return false;
  return true;
}

bool valueFeedsOnlyShiftedMemoryAddress(
    mlir::Value value, llvm::DenseSet<mlir::Value> &active, bool &sawShift) {
  if (value.use_empty())
    return false;
  if (!active.insert(value).second)
    return false;

  bool sawAddressPath = false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (isDataflowMemoryAddressUse(use)) {
      sawAddressPath = true;
      continue;
    }
    mlir::Operation *owner = use.getOwner();
    if (!isAddressArithmeticOp(owner) ||
        !valueFeedsOnlyShiftedMemoryAddress(owner->getResult(0), active,
                                            sawShift)) {
      active.erase(value);
      return false;
    }
    sawShift |= isAddressShiftOp(owner);
    sawAddressPath = true;
  }

  active.erase(value);
  return sawAddressPath;
}

bool valueFeedsOnlyShiftedMemoryAddress(mlir::Value value) {
  llvm::DenseSet<mlir::Value> active;
  bool sawShift = false;
  return valueFeedsOnlyShiftedMemoryAddress(value, active, sawShift) &&
         sawShift;
}

bool valueFeedsOnlyComputedMemoryAddress(mlir::Value value,
                                         llvm::DenseSet<mlir::Value> &active) {
  if (value.use_empty())
    return false;
  if (!active.insert(value).second)
    return false;

  bool sawAddressPath = false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (isDataflowMemoryAddressUse(use)) {
      sawAddressPath = true;
      continue;
    }
    mlir::Operation *owner = use.getOwner();
    if (!isAddressArithmeticOp(owner) ||
        !valueFeedsOnlyComputedMemoryAddress(owner->getResult(0), active)) {
      active.erase(value);
      return false;
    }
    sawAddressPath = true;
  }

  active.erase(value);
  return sawAddressPath;
}

bool valueFeedsOnlyComputedMemoryAddress(mlir::Value value) {
  llvm::DenseSet<mlir::Value> active;
  return valueFeedsOnlyComputedMemoryAddress(value, active);
}

bool isDataflowStreamIndex(mlir::Value value) {
  auto stream = value.getDefiningOp<::dataflow::StreamOp>();
  return stream && stream.getIv() == value;
}

} // namespace

bool shouldMaterializeAdapterOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if ((name != "arith.index_cast" && name != "arith.index_castui") ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  bool directAddress = valueFeedsOnlyDirectMemoryAddress(op->getResult(0));
  bool shiftedAddress = valueFeedsOnlyShiftedMemoryAddress(op->getResult(0));
  bool computedAddress = valueFeedsOnlyComputedMemoryAddress(op->getResult(0));
  if (!directAddress && !shiftedAddress && !computedAddress)
    return false;
  if ((shiftedAddress || computedAddress) && !directAddress &&
      !op->getOperand(0).getDefiningOp())
    return false;
  if (isDataflowStreamIndex(op->getOperand(0)))
    return false;
  std::optional<unsigned> inputWidth =
      adapterBitWidth(op->getOperand(0).getType());
  std::optional<unsigned> resultWidth =
      adapterBitWidth(op->getResult(0).getType());
  return inputWidth && resultWidth && *inputWidth > *resultWidth;
}

namespace {

bool isPointerCarryOp(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.carry" ||
      op->getNumResults() != 1)
    return false;
  return isLlvmPointerType(op->getResult(0).getType());
}

bool isPointerGateOp(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.gate" ||
      op->getNumOperands() != 2 || op->getNumResults() != 2)
    return false;
  return isLlvmPointerType(op->getResult(1).getType());
}

bool isGraphReturnOp(mlir::Operation *op) {
  return op->getName().getStringRef() == "dataflow.graph.return";
}

bool isEffectFormForall(mlir::scf::ForallOp op) {
  if (!op.getOutputs().empty() || op->getNumResults() != 0)
    return false;
  auto inParallel = op.getTerminator();
  return !inParallel.getRegion().empty() &&
         inParallel.getRegion().front().empty();
}

} // namespace

bool isStructuredContainerOp(mlir::Operation *op) {
  if (auto forall = mlir::dyn_cast<mlir::scf::ForallOp>(op))
    return isEffectFormForall(forall);
  return mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::IndexSwitchOp,
                   mlir::scf::WhileOp>(op);
}

namespace {

bool isStructuredTerminatorOp(mlir::Operation *op) {
  if (mlir::isa<mlir::scf::YieldOp, mlir::scf::ConditionOp>(op))
    return true;
  auto inParallel = mlir::dyn_cast<mlir::scf::InParallelOp>(op);
  return inParallel && !inParallel.getRegion().empty() &&
         inParallel.getRegion().front().empty();
}

bool isDataflowMemoryBaseUse(mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  llvm::StringRef name = owner->getName().getStringRef();
  return (name == "dataflow.load" || name == "dataflow.store") &&
         use.getOperandNumber() == 0;
}

bool isPointerMemrefBaseAdapterOp(mlir::Operation *op) {
  if (op->getName().getStringRef() != "builtin.unrealized_conversion_cast" ||
      op->getNumOperands() != 1 || op->getNumResults() == 0 ||
      !isLlvmPointerType(op->getOperand(0).getType()))
    return false;
  for (mlir::Value result : op->getResults()) {
    if (!mlir::isa<mlir::MemRefType>(result.getType()))
      return false;
    for (mlir::OpOperand &use : result.getUses()) {
      if (!isDataflowMemoryBaseUse(use))
        return false;
    }
  }
  return true;
}

bool isLlvmPointerMemoryAddressUse(mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  if (mlir::isa<mlir::LLVM::LoadOp>(owner))
    return use.getOperandNumber() == 0;
  if (mlir::isa<mlir::LLVM::StoreOp>(owner))
    return use.getOperandNumber() == 1;
  return false;
}

bool isStructuredPointerValueOnlyBookkeeping(mlir::Value value) {
  if (!isLlvmPointerType(value.getType()))
    return false;
  if (value.use_empty())
    return true;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (mlir::isa<mlir::LLVM::GEPOp>(owner) ||
        isLlvmPointerMemoryAddressUse(use) || isPointerCarryOp(owner) ||
        isPointerGateOp(owner) || isPointerMemrefBaseAdapterOp(owner) ||
        isGraphReturnOp(owner))
      continue;
    mlir::Operation *parent = nullptr;
    unsigned resultIndex = 0;
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(owner)) {
      parent = yield->getParentOp();
      resultIndex = use.getOperandNumber();
    } else if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(owner)) {
      parent = condition->getParentOp();
      if (use.getOperandNumber() == 0)
        return false;
      resultIndex = use.getOperandNumber() - 1;
    } else if (mlir::isa<mlir::scf::WhileOp>(owner)) {
      parent = owner;
      resultIndex = use.getOperandNumber();
    } else if (mlir::isa<mlir::scf::ForOp>(owner)) {
      parent = owner;
      if (use.getOperandNumber() < 3)
        return false;
      resultIndex = use.getOperandNumber() - 3;
    } else {
      return false;
    }
    if (!parent || resultIndex >= parent->getNumResults())
      return false;
    if (!isStructuredPointerValueOnlyBookkeeping(
            parent->getResult(resultIndex)))
      return false;
  }
  return true;
}

bool isStructuredPointerForwardingUse(mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  mlir::Operation *parent = nullptr;
  unsigned resultIndex = 0;
  if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(owner)) {
    parent = yield->getParentOp();
    resultIndex = use.getOperandNumber();
  } else if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(owner)) {
    parent = condition->getParentOp();
    if (use.getOperandNumber() == 0)
      return false;
    resultIndex = use.getOperandNumber() - 1;
  } else if (mlir::isa<mlir::scf::WhileOp>(owner)) {
    parent = owner;
    resultIndex = use.getOperandNumber();
  } else if (mlir::isa<mlir::scf::ForOp>(owner)) {
    parent = owner;
    if (use.getOperandNumber() < 3)
      return false;
    resultIndex = use.getOperandNumber() - 3;
  } else {
    return false;
  }
  if (!parent || resultIndex >= parent->getNumResults())
    return false;
  return isStructuredPointerValueOnlyBookkeeping(
      parent->getResult(resultIndex));
}

} // namespace

bool isPointerBookkeepingOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name == "llvm.mlir.zero" || name == "llvm.mlir.addressof") {
    if (op->getNumResults() != 1 ||
        !isLlvmPointerType(op->getResult(0).getType()))
      return false;
    for (mlir::OpOperand &use : op->getResult(0).getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (mlir::isa<mlir::LLVM::ICmpOp>(owner) ||
          mlir::isa<mlir::LLVM::GEPOp>(owner) ||
          isLlvmPointerMemoryAddressUse(use) || isPointerCarryOp(owner) ||
          isPointerMemrefBaseAdapterOp(owner) || isGraphReturnOp(owner) ||
          isStructuredPointerForwardingUse(use))
        continue;
      return false;
    }
    return true;
  }

  if (name == "llvm.getelementptr") {
    if (op->getNumResults() != 1 ||
        !isLlvmPointerType(op->getResult(0).getType()))
      return false;
    for (mlir::OpOperand &use : op->getResult(0).getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (mlir::isa<mlir::LLVM::GEPOp>(owner) ||
          isLlvmPointerMemoryAddressUse(use) || isPointerCarryOp(owner) ||
          isPointerMemrefBaseAdapterOp(owner) || isGraphReturnOp(owner) ||
          isStructuredPointerForwardingUse(use))
        continue;
      return false;
    }
    return true;
  }

  if (!isPointerCarryOp(op) && !isPointerGateOp(op))
    return false;
  if (isPointerGateOp(op) && !op->getResult(0).use_empty())
    return false;
  unsigned pointerResultIndex = isPointerGateOp(op) ? 1 : 0;
  for (mlir::OpOperand &use : op->getResult(pointerResultIndex).getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (mlir::isa<mlir::LLVM::GEPOp>(owner) ||
        isLlvmPointerMemoryAddressUse(use) || isGraphReturnOp(owner))
      continue;
    if (isPointerGateOp(owner))
      continue;
    if (isPointerMemrefBaseAdapterOp(owner))
      continue;
    return false;
  }
  return true;
}

namespace {

bool isPointerBookkeepingReturnValue(mlir::Value value) {
  mlir::Operation *owner = value.getDefiningOp();
  if (!owner)
    return false;
  return isPointerBookkeepingOp(owner);
}

std::optional<llvm::StringRef> canonicalArmInlineAsmOperationName(
    mlir::Operation *op) {
  if (op->getName().getStringRef() != "llvm.inline_asm")
    return std::nullopt;
  auto asmString = op->getAttrOfType<mlir::StringAttr>("asm_string");
  if (!asmString)
    return std::nullopt;
  llvm::StringRef text = asmString.getValue();
  if (text == "pkhbt $0, $1, $2, lsl $3")
    return llvm::StringRef("llvm.arm.pkhbt");
  if (text == "pkhtb $0, $1, $2, asr $3")
    return llvm::StringRef("llvm.arm.pkhtb");
  if (text == "sxtab16 $0, $1, $2")
    return llvm::StringRef("llvm.arm.sxtab16");
  if (text == "sxtb16 $0, $1")
    return llvm::StringRef("llvm.arm.sxtb16");
  return std::nullopt;
}

std::optional<ResourceKind> resourceKindForSoftwareOp(mlir::Operation *op) {
  std::string nameStorage;
  llvm::StringRef name = op->getName().getStringRef();
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op)) {
    nameStorage = intrinsic.getIntrin().str();
    name = nameStorage;
  }
  if (std::optional<llvm::StringRef> asmName =
          canonicalArmInlineAsmOperationName(op))
    name = *asmName;
  if (name == "dataflow.load" || name == "llvm.load")
    return ResourceKind::MemLoad;
  if (name == "dataflow.store" || name == "llvm.store")
    return ResourceKind::MemStore;
  if (fabric::isFabricOpSupported(name))
    return ResourceKind::FabricOp;
  return std::nullopt;
}

std::string softwareOperationName(mlir::Operation *op) {
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin().str();
  if (std::optional<llvm::StringRef> asmName =
          canonicalArmInlineAsmOperationName(op))
    return asmName->str();
  return op->getName().getStringRef().str();
}

} // namespace

llvm::StringRef resourceKindName(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::FabricOp:
    return "fabric.op";
  case ResourceKind::MemLoad:
    return "fabric.mem.load";
  case ResourceKind::MemStore:
    return "fabric.mem.store";
  }
  llvm_unreachable("unknown resource kind");
}

namespace {

llvm::Error
collectSoftwareNodesInBlock(mlir::Block &block,
                            llvm::SmallVectorImpl<SoftwareNode> &nodes,
                            llvm::StringMap<unsigned> &counts) {
  for (mlir::Operation &op : block) {
    if (isStructuredContainerOp(&op)) {
      for (mlir::Region &region : op.getRegions())
        for (mlir::Block &nested : region)
          if (llvm::Error err =
                  collectSoftwareNodesInBlock(nested, nodes, counts))
            return err;
      continue;
    }
    if (isStructuredTerminatorOp(&op))
      continue;
    if (isIgnoredOp(&op) ||
        (isAdapterOp(&op) && !shouldMaterializeAdapterOp(&op)) ||
        isPointerBookkeepingOp(&op))
      continue;
    std::optional<ResourceKind> kind = resourceKindForSoftwareOp(&op);
    if (!kind) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "graph contains unsupported operation for PnR mapping: %s",
          op.getName().getStringRef().str().c_str());
    }
    std::string opName = softwareOperationName(&op);
    unsigned index = counts[opName]++;
    nodes.push_back(
        SoftwareNode{opName + "#" + std::to_string(index), opName, *kind, &op});
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<SoftwareNode>>
collectSoftwareNodes(mlir::Operation *graph) {
  llvm::SmallVector<SoftwareNode> nodes;
  llvm::StringMap<unsigned> counts;
  for (mlir::Operation &op : graph->getRegion(0).front()) {
    if (!isGraphReturnOp(&op))
      continue;
    for (mlir::Value value : op.getOperands()) {
      if (isLlvmPointerType(value.getType()) &&
          !isPointerBookkeepingReturnValue(value))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "graph returns unsupported pointer value for PnR mapping");
    }
  }
  if (llvm::Error err = collectSoftwareNodesInBlock(graph->getRegion(0).front(),
                                                    nodes, counts))
    return std::move(err);
  return nodes;
}

std::string apintToHexString(const llvm::APInt &value) {
  llvm::SmallString<32> hex;
  value.toString(hex, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/false);
  for (char &ch : hex) {
    if (ch >= 'A' && ch <= 'F')
      ch = static_cast<char>(ch - 'A' + 'a');
  }
  std::string out = "0x";
  out += hex.c_str();
  return out;
}

std::optional<std::string> encodeConstHex(mlir::Attribute attr) {
  if (auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attr))
    return apintToHexString(integer.getValue());
  if (auto fp = llvm::dyn_cast_if_present<mlir::FloatAttr>(attr))
    return apintToHexString(fp.getValue().bitcastToAPInt());
  if (auto stringAttr = llvm::dyn_cast_if_present<mlir::StringAttr>(attr)) {
    llvm::StringRef value = stringAttr.getValue();
    if (value.starts_with("0x") || value.starts_with("0X"))
      return value.str();
    return ("0x" + value).str();
  }
  return std::nullopt;
}

std::optional<std::string> canonicalHexValue(llvm::StringRef value) {
  if (!(value.starts_with("0x") || value.starts_with("0X")))
    return std::nullopt;
  llvm::StringRef digits = value.drop_front(2);
  if (digits.empty())
    return std::nullopt;
  std::string lowered;
  lowered.reserve(digits.size());
  for (char ch : digits) {
    if (!llvm::isHexDigit(ch))
      return std::nullopt;
    if (ch >= 'A' && ch <= 'F')
      ch = static_cast<char>(ch - 'A' + 'a');
    lowered.push_back(ch);
  }
  std::size_t firstNonZero = lowered.find_first_not_of('0');
  if (firstNonZero == std::string::npos)
    return std::string("0x0");
  return "0x" + lowered.substr(firstNonZero);
}

std::optional<std::string> predicateConfig(mlir::Operation *op) {
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op))
    return mlir::arith::stringifyCmpIPredicate(cmp.getPredicate()).str();
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(op))
    return mlir::arith::stringifyCmpFPredicate(cmp.getPredicate()).str();
  if (auto cmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(op))
    return mlir::LLVM::stringifyICmpPredicate(cmp.getPredicate()).str();
  return std::nullopt;
}

std::optional<std::string> constantConfig(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.constant")
    return std::nullopt;
  return encodeConstHex(op->getAttr("const_value"));
}

std::optional<std::string> gateValueKindConfig(mlir::Operation *op) {
  if (op->getName().getStringRef() != "dataflow.gate" ||
      op->getNumOperands() < 2)
    return std::nullopt;
  if (mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
    return "control";
  return "data";
}

} // namespace

std::map<std::string, std::string>
softwareConfigsFor(const SoftwareNode &node) {
  std::map<std::string, std::string> configs;
  if (std::optional<std::string> predicate = predicateConfig(node.op))
    configs.try_emplace("predicate", *predicate);
  if (std::optional<std::string> constant = constantConfig(node.op))
    configs.try_emplace("const_hex_value", *constant);
  if (std::optional<std::string> gateKind = gateValueKindConfig(node.op))
    configs.try_emplace("value_kind", *gateKind);
  return configs;
}

namespace {

bool configValuesMatch(llvm::StringRef key, llvm::StringRef hardwareValue,
                       llvm::StringRef softwareValue) {
  if (key == "const_hex_value") {
    std::optional<std::string> hardwareHex = canonicalHexValue(hardwareValue);
    std::optional<std::string> softwareHex = canonicalHexValue(softwareValue);
    if (hardwareHex && softwareHex)
      return *hardwareHex == *softwareHex;
  }
  return hardwareValue == softwareValue;
}

} // namespace

std::optional<std::string>
resolvedSoftwareConfigValue(const HardwareResource &resource,
                            llvm::StringRef key, llvm::StringRef value) {
  auto fixed = resource.swConfigs.find(key.str());
  if (fixed != resource.swConfigs.end()) {
    if (configValuesMatch(key, fixed->second, value))
      return fixed->second;
    return std::nullopt;
  }
  auto allowed = resource.hwParamOptions.find(key.str());
  if (allowed == resource.hwParamOptions.end() || allowed->second.empty())
    return value.str();
  for (const std::string &allowedValue : allowed->second) {
    if (configValuesMatch(key, allowedValue, value))
      return allowedValue;
  }
  return std::nullopt;
}

namespace {

bool resourceSupportsConfig(const HardwareResource &resource,
                            llvm::StringRef key, llvm::StringRef value) {
  return resolvedSoftwareConfigValue(resource, key, value).has_value();
}

bool resourceSupportsSoftwareConfigs(const SoftwareNode &node,
                                     const HardwareResource &resource) {
  if (auto stream = mlir::dyn_cast_or_null<dataflow::StreamOp>(node.op)) {
    if (!resource.streamConfiguration ||
        resource.streamConfiguration->stepKind != stream.getStepKind() ||
        !resource.streamConfiguration->supports(stream.getPredicate()))
      return false;
    if (resource.streamConfiguration->selectedPredicate &&
        *resource.streamConfiguration->selectedPredicate !=
            stream.getPredicate())
      return false;
    return true;
  }
  for (const auto &[key, value] : softwareConfigsFor(node)) {
    if (!resourceSupportsConfig(resource, key, value))
      return false;
  }
  return true;
}

} // namespace

std::optional<unsigned> softwareBitWidth(mlir::Type type) {
  std::string error;
  auto width = fabric::getSemanticPayloadWidth(type, error);
  if (mlir::failed(width))
    return std::nullopt;
  return *width;
}

namespace {

std::optional<unsigned> fabricBitWidth(mlir::Type type) {
  if (auto bits = mlir::dyn_cast<fabric::BitsType>(type))
    return bits.getWidth();
  return std::nullopt;
}

bool valueFeedsOnlyMemoryAddressThroughIndexCast(mlir::Value value) {
  bool sawAddressUse = false;
  for (mlir::OpOperand &use : value.getUses()) {
    auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(use.getOwner());
    if (!cast || use.getOperandNumber() != 0 ||
        !mlir::isa<mlir::IndexType>(cast.getType()))
      return false;
    for (mlir::OpOperand &castUse : cast.getResult().getUses()) {
      if (!isDataflowMemoryAddressUse(castUse))
        return false;
      sawAddressUse = true;
    }
  }
  return sawAddressUse;
}

bool isDataflowStoreDataUse(mlir::OpOperand &use) {
  return use.getOwner()->getName().getStringRef() == "dataflow.store" &&
         use.getOperandNumber() == 2;
}

bool valueFeedsOnlyStoreData(mlir::Value value) {
  if (value.use_empty())
    return false;
  for (mlir::OpOperand &use : value.getUses())
    if (!isDataflowStoreDataUse(use))
      return false;
  return true;
}

bool isFabricBitsWidth(mlir::Type type, unsigned width) {
  std::optional<unsigned> actual = fabricBitWidth(type);
  return actual && *actual == width;
}

bool softwareTypeFitsFabricType(mlir::Type softwareType,
                                mlir::Type hardwareType) {
  std::optional<unsigned> softwareWidth = softwareBitWidth(softwareType);
  std::optional<unsigned> hardwareWidth = fabricBitWidth(hardwareType);
  if (!softwareWidth || !hardwareWidth)
    return false;
  if (mlir::isa<mlir::NoneType>(softwareType))
    return *softwareWidth == *hardwareWidth;
  if (mlir::isa<mlir::FloatType>(softwareType))
    return *softwareWidth == *hardwareWidth;
  if (isLlvmPointerType(softwareType))
    return *softwareWidth <= *hardwareWidth;
  if (mlir::isa<mlir::IntegerType, mlir::IndexType>(softwareType))
    return *softwareWidth <= *hardwareWidth;
  return false;
}

bool resourceSupportsDataflowInvariantTransport(
    const SoftwareNode &node, const HardwareResource &resource) {
  if (node.operation != "dataflow.invariant" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 2 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 2 || resource.op->getNumResults() != 1)
    return false;

  std::optional<unsigned> softwareCondWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareValueWidth =
      softwareBitWidth(node.op->getOperand(1).getType());
  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  unsigned transportWidth = loom::getIndexWidth();
  if (!softwareCondWidth || *softwareCondWidth != 1 || !softwareValueWidth ||
      !softwareResultWidth || *softwareValueWidth != *softwareResultWidth ||
      *softwareValueWidth == 0 || *softwareValueWidth > transportWidth)
    return false;

  return isFabricBitsWidth(resource.op->getOperand(0).getType(), 1) &&
         isFabricBitsWidth(resource.op->getOperand(1).getType(),
                           transportWidth) &&
         isFabricBitsWidth(resource.op->getResult(0).getType(), transportWidth);
}

bool resourceSupportsDataflowConstantTransport(
    const SoftwareNode &node, const HardwareResource &resource) {
  if (node.operation != "dataflow.constant" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;

  std::optional<unsigned> softwareControlWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  unsigned transportWidth = loom::getIndexWidth();
  if (!softwareControlWidth || *softwareControlWidth != 0 ||
      !softwareResultWidth || *softwareResultWidth == 0 ||
      *softwareResultWidth > transportWidth)
    return false;

  return isFabricBitsWidth(resource.op->getOperand(0).getType(), 0) &&
         isFabricBitsWidth(resource.op->getResult(0).getType(), transportWidth);
}

bool isTransportablePayloadType(mlir::Type type) {
  if (isLlvmPointerType(type))
    return true;
  std::optional<unsigned> width = softwareBitWidth(type);
  return width && *width > 0 && *width <= loom::getIndexWidth();
}

bool resourceSupportsDataflowGateTransport(const SoftwareNode &node,
                                           const HardwareResource &resource) {
  if (node.operation != "dataflow.gate" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 2 || node.op->getNumResults() != 2 ||
      resource.op->getNumOperands() != 2 || resource.op->getNumResults() != 2)
    return false;

  std::optional<unsigned> softwareCondWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareCondResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  unsigned transportWidth = loom::getIndexWidth();
  if (!softwareCondWidth || *softwareCondWidth != 1 ||
      !softwareCondResultWidth || *softwareCondResultWidth != 1 ||
      !isTransportablePayloadType(node.op->getOperand(1).getType()) ||
      !isTransportablePayloadType(node.op->getResult(1).getType()))
    return false;

  return isFabricBitsWidth(resource.op->getOperand(0).getType(), 1) &&
         isFabricBitsWidth(resource.op->getOperand(1).getType(),
                           transportWidth) &&
         isFabricBitsWidth(resource.op->getResult(0).getType(), 1) &&
         isFabricBitsWidth(resource.op->getResult(1).getType(), transportWidth);
}

bool isIndexAdapterResult(mlir::Value value) {
  auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>();
  return cast && mlir::isa<mlir::IndexType>(cast.getIn().getType());
}

bool resourceSupportsIndexDomainZExt(const SoftwareNode &node,
                                     const HardwareResource &resource) {
  if (node.operation != "llvm.zext" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;

  std::optional<unsigned> softwareInputWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  auto softwareResultType =
      mlir::dyn_cast<mlir::IntegerType>(node.op->getResult(0).getType());
  std::optional<unsigned> hardwareInputWidth =
      fabricBitWidth(resource.op->getOperand(0).getType());
  std::optional<unsigned> hardwareResultWidth =
      fabricBitWidth(resource.op->getResult(0).getType());
  unsigned indexWidth = loom::getIndexWidth();
  if (!softwareInputWidth || *softwareInputWidth != indexWidth ||
      !softwareResultType || softwareResultType.getWidth() < indexWidth ||
      !hardwareInputWidth || *hardwareInputWidth != indexWidth ||
      !hardwareResultWidth || *hardwareResultWidth != indexWidth)
    return false;

  return valueFeedsOnlyMemoryAddressThroughIndexCast(node.op->getResult(0));
}

bool resourceSupportsIndexAdapterTrunc(const SoftwareNode &node,
                                       const HardwareResource &resource) {
  if (node.operation != "llvm.trunc" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;
  if (!isIndexAdapterResult(node.op->getOperand(0)))
    return false;

  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  unsigned indexWidth = loom::getIndexWidth();
  return softwareResultWidth && *softwareResultWidth == indexWidth &&
         isFabricBitsWidth(resource.op->getOperand(0).getType(), indexWidth) &&
         isFabricBitsWidth(resource.op->getResult(0).getType(), indexWidth);
}

bool resourceSupportsStreamIndexTrunc(const SoftwareNode &node,
                                      const HardwareResource &resource) {
  if (node.operation != "llvm.trunc" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;
  auto stream = node.op->getOperand(0).getDefiningOp<::dataflow::StreamOp>();
  if (!stream || stream.getIv() != node.op->getOperand(0))
    return false;

  std::optional<unsigned> softwareInputWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  std::optional<unsigned> hardwareInputWidth =
      fabricBitWidth(resource.op->getOperand(0).getType());
  std::optional<unsigned> hardwareResultWidth =
      fabricBitWidth(resource.op->getResult(0).getType());
  unsigned indexWidth = loom::getIndexWidth();
  return softwareInputWidth && *softwareInputWidth >= indexWidth &&
         softwareResultWidth && *softwareResultWidth == indexWidth &&
         hardwareInputWidth && *hardwareInputWidth == indexWidth &&
         hardwareResultWidth && *hardwareResultWidth == indexWidth;
}

bool resourceSupportsIntegerNarrowingTrunc(const SoftwareNode &node,
                                           const HardwareResource &resource) {
  if (node.operation != "llvm.trunc" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;

  std::optional<unsigned> softwareInputWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  std::optional<unsigned> hardwareInputWidth =
      fabricBitWidth(resource.op->getOperand(0).getType());
  std::optional<unsigned> hardwareResultWidth =
      fabricBitWidth(resource.op->getResult(0).getType());
  unsigned indexWidth = loom::getIndexWidth();
  if (!softwareInputWidth || !softwareResultWidth || !hardwareInputWidth ||
      !hardwareResultWidth)
    return false;
  if (*softwareInputWidth > indexWidth ||
      *softwareResultWidth >= *softwareInputWidth)
    return false;
  if (*hardwareInputWidth != indexWidth || *hardwareResultWidth != indexWidth)
    return false;
  return valueFeedsOnlyStoreData(node.op->getResult(0));
}

bool resourceSupportsIntegerWideningExtension(
    const SoftwareNode &node, const HardwareResource &resource) {
  if ((node.operation != "llvm.sext" && node.operation != "llvm.zext") ||
      !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 1 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 1 || resource.op->getNumResults() != 1)
    return false;

  std::optional<unsigned> softwareInputWidth =
      softwareBitWidth(node.op->getOperand(0).getType());
  std::optional<unsigned> softwareResultWidth =
      softwareBitWidth(node.op->getResult(0).getType());
  std::optional<unsigned> hardwareInputWidth =
      fabricBitWidth(resource.op->getOperand(0).getType());
  std::optional<unsigned> hardwareResultWidth =
      fabricBitWidth(resource.op->getResult(0).getType());
  unsigned indexWidth = loom::getIndexWidth();
  return softwareInputWidth && softwareResultWidth && hardwareInputWidth &&
         hardwareResultWidth && *softwareInputWidth < *softwareResultWidth &&
         *softwareResultWidth == indexWidth &&
         *hardwareInputWidth == indexWidth &&
         *hardwareResultWidth == indexWidth;
}

bool isPredicateConsumerUse(mlir::OpOperand &use) {
  mlir::Operation *owner = use.getOwner();
  llvm::StringRef name = owner->getName().getStringRef();
  unsigned operand = use.getOperandNumber();
  if (name == "arith.select" || name == "dataflow.mux" ||
      name == "dataflow.demux" || name == "dataflow.gate")
    return operand == 0;
  if (name == "dataflow.carry" || name == "dataflow.invariant")
    return operand == 0;
  return false;
}

bool valueFeedsOnlyPredicateConsumers(mlir::Value value) {
  if (value.use_empty())
    return false;
  for (mlir::OpOperand &use : value.getUses()) {
    if (!isPredicateConsumerUse(use))
      return false;
  }
  return true;
}

bool resourceSupportsPredicateTransportAndI(const SoftwareNode &node,
                                            const HardwareResource &resource) {
  if (node.operation != "arith.andi" || !node.op || !resource.op)
    return false;
  if (node.op->getNumOperands() != 2 || node.op->getNumResults() != 1 ||
      resource.op->getNumOperands() != 2 || resource.op->getNumResults() != 1)
    return false;
  auto isI1 = [](mlir::Type type) {
    auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
    return intType && intType.getWidth() == 1;
  };
  if (!llvm::all_of(node.op->getOperandTypes(), isI1) ||
      !llvm::all_of(node.op->getResultTypes(), isI1))
    return false;
  auto isTransport32 = [](mlir::Type type) {
    std::optional<unsigned> width = fabricBitWidth(type);
    return width && *width == loom::getIndexWidth();
  };
  if (!llvm::all_of(resource.op->getOperandTypes(), isTransport32) ||
      !llvm::all_of(resource.op->getResultTypes(), isTransport32))
    return false;
  return valueFeedsOnlyPredicateConsumers(node.op->getResult(0));
}

bool resourceSupportsSoftwarePortShape(const SoftwareNode &node,
                                       const HardwareResource &resource) {
  if (resource.kind != ResourceKind::FabricOp)
    return true;
  if (!resource.op)
    return false;

  if (node.operation == "dataflow.stream") {
    if (resource.op->getNumOperands() != 3 || resource.op->getNumResults() != 2)
      return false;
    for (auto [softwareType, hardwareType] : llvm::zip(
             node.op->getOperandTypes(), resource.op->getOperandTypes())) {
      if (!mlir::isa<mlir::IntegerType>(softwareType))
        return false;
      std::optional<unsigned> softwareWidth = softwareBitWidth(softwareType);
      std::optional<unsigned> hardwareWidth = fabricBitWidth(hardwareType);
      if (!softwareWidth || !hardwareWidth || *softwareWidth != *hardwareWidth)
        return false;
    }
    mlir::Type softwareIvType = node.op->getResult(0).getType();
    mlir::Type softwarePhaseType = node.op->getResult(1).getType();
    if (!mlir::isa<mlir::IntegerType>(softwareIvType) ||
        !mlir::isa<mlir::IntegerType>(softwarePhaseType))
      return false;
    if (softwarePhaseType.getIntOrFloatBitWidth() != 1)
      return false;
    std::optional<unsigned> softwareIvWidth = softwareBitWidth(softwareIvType);
    std::optional<unsigned> hardwareIvWidth =
        fabricBitWidth(resource.op->getResult(0).getType());
    std::optional<unsigned> hardwarePhaseWidth =
        fabricBitWidth(resource.op->getResult(1).getType());
    return softwareIvWidth && hardwareIvWidth &&
           *softwareIvWidth == *hardwareIvWidth && hardwarePhaseWidth &&
           *hardwarePhaseWidth == 1;
  }

  if (node.operation == "dataflow.sync") {
    if (node.op->getNumOperands() != node.op->getNumResults())
      return false;
    if (node.op->getNumOperands() > resource.op->getNumOperands() ||
        node.op->getNumResults() > resource.op->getNumResults())
      return false;
    auto isControlToken = [](mlir::Type type) {
      return mlir::isa<mlir::NoneType>(type);
    };
    auto isFabricControlToken = [](mlir::Type type) {
      std::optional<unsigned> width = fabricBitWidth(type);
      return width && *width == 0;
    };
    if (!llvm::all_of(node.op->getOperandTypes(), isControlToken) ||
        !llvm::all_of(node.op->getResultTypes(), isControlToken))
      return false;
    if (!llvm::all_of(resource.op->getOperandTypes(), isFabricControlToken) ||
        !llvm::all_of(resource.op->getResultTypes(), isFabricControlToken))
      return false;
    return true;
  }

  if (node.op->getNumOperands() != resource.op->getNumOperands() ||
      node.op->getNumResults() != resource.op->getNumResults())
    return false;

  if (resourceSupportsIndexDomainZExt(node, resource))
    return true;
  if (resourceSupportsIndexAdapterTrunc(node, resource))
    return true;
  if (resourceSupportsStreamIndexTrunc(node, resource))
    return true;
  if (resourceSupportsIntegerNarrowingTrunc(node, resource))
    return true;
  if (resourceSupportsIntegerWideningExtension(node, resource))
    return true;
  if (resourceSupportsPredicateTransportAndI(node, resource))
    return true;
  if (resourceSupportsDataflowConstantTransport(node, resource))
    return true;
  if (resourceSupportsDataflowInvariantTransport(node, resource))
    return true;
  if (resourceSupportsDataflowGateTransport(node, resource))
    return true;

  for (auto [softwareType, hardwareType] :
       llvm::zip(node.op->getOperandTypes(), resource.op->getOperandTypes())) {
    if (!softwareTypeFitsFabricType(softwareType, hardwareType))
      return false;
  }
  for (auto [softwareType, hardwareType] :
       llvm::zip(node.op->getResultTypes(), resource.op->getResultTypes())) {
    if (!softwareTypeFitsFabricType(softwareType, hardwareType))
      return false;
  }
  return true;
}

} // namespace

HardwareResource *
claimResource(SoftwareNode &node,
              llvm::MutableArrayRef<HardwareResource> resources) {
  for (HardwareResource &resource : resources) {
    if (resource.used || resource.kind != node.resourceKind)
      continue;
    if (resource.kind == ResourceKind::FabricOp &&
        !resource.supportedOps.contains(node.operation))
      continue;
    if (!resourceSupportsSoftwarePortShape(node, resource))
      continue;
    if (!resourceSupportsSoftwareConfigs(node, resource))
      continue;
    resource.used = true;
    return &resource;
  }
  return nullptr;
}

PlacementRecord makePlacementRecord(const SoftwareNode &node,
                                    const HardwareResource &resource) {
  return PlacementRecord{node.id, node.operation,
                         resourceKindName(node.resourceKind).str(),
                         resource.id, resource.schedule};
}

bool resourceIsCompatible(const SoftwareNode &node,
                          const HardwareResource &resource) {
  if (resource.kind != node.resourceKind)
    return false;
  if (resource.kind == ResourceKind::FabricOp &&
      !resource.supportedOps.contains(node.operation))
    return false;
  if (!resourceSupportsSoftwarePortShape(node, resource))
    return false;
  return resourceSupportsSoftwareConfigs(node, resource);
}

unsigned compatibleResourceCount(const SoftwareNode &node,
                                 llvm::ArrayRef<HardwareResource> resources) {
  unsigned count = 0;
  for (const HardwareResource &resource : resources)
    if (resourceIsCompatible(node, resource))
      ++count;
  return count;
}

namespace {

struct ResourcePressureKey {
  ResourceKind kind;
  std::string operation;

  bool operator<(const ResourcePressureKey &other) const {
    return std::tie(kind, operation) < std::tie(other.kind, other.operation);
  }
};

std::string resourcePressureDiagnostic(const ResourcePressureRecord &record) {
  return (llvm::Twine("resource pressure: resource_kind=") +
          record.resourceKind + " operation=" + record.operation +
          " required=" + llvm::Twine(record.required) +
          " available=" + llvm::Twine(record.available) +
          " placed=" + llvm::Twine(record.placed) +
          " missing=" + llvm::Twine(record.missing))
      .str();
}

void appendResourcePressureRecords(MappingSummary &summary,
                                   llvm::ArrayRef<SoftwareNode> nodes,
                                   llvm::ArrayRef<HardwareResource> resources) {
  std::map<ResourcePressureKey, std::uint64_t> requiredByKey;
  std::map<ResourcePressureKey, std::uint64_t> placedByKey;
  std::map<ResourcePressureKey, std::set<std::string>> availableByKey;
  llvm::StringMap<const SoftwareNode *> nodeById;

  for (const SoftwareNode &node : nodes) {
    ResourcePressureKey key{node.resourceKind, node.operation};
    ++requiredByKey[key];
    nodeById.try_emplace(node.id, &node);
    for (const HardwareResource &resource : resources)
      if (resourceIsCompatible(node, resource))
        availableByKey[key].insert(resource.id);
  }
  for (const PlacementRecord &placement : summary.placements) {
    auto nodeIt = nodeById.find(placement.softwareId);
    if (nodeIt == nodeById.end())
      continue;
    ResourcePressureKey key{nodeIt->second->resourceKind,
                            nodeIt->second->operation};
    ++placedByKey[key];
  }

  for (const auto &[key, required] : requiredByKey) {
    std::uint64_t placed = placedByKey[key];
    if (placed >= required)
      continue;
    std::uint64_t available = availableByKey[key].size();
    summary.resourcePressure.push_back(
        ResourcePressureRecord{resourceKindName(key.kind).str(), key.operation,
                               required, available, placed, required - placed});
  }
}

void appendResourcePressureDiagnostic(MappingSummary &summary) {
  if (summary.resourcePressure.empty())
    return;
  std::string details;
  for (const ResourcePressureRecord &record : summary.resourcePressure) {
    if (!details.empty())
      details += "; ";
    details += resourcePressureDiagnostic(record);
  }
  if (summary.diagnostic.empty()) {
    summary.diagnostic = details;
    return;
  }
  summary.diagnostic += " (" + details + ")";
}

} // namespace

std::optional<std::string> configFor(const HardwareResource &resource,
                                     llvm::StringRef key) {
  auto it = resource.swConfigs.find(key.str());
  if (it == resource.swConfigs.end())
    return std::nullopt;
  return it->second;
}

} // namespace loom::pnr::detail

llvm::Expected<MappingSummary>
loom::pnr::createMapping(const MappingOptions &options) {
  mlir::DialectRegistry registry = makeRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> dfg =
      parseModule(context, options.dfgMlirPath);
  if (!dfg)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse DFG MLIR");
  mlir::OwningOpRef<mlir::ModuleOp> hardware =
      parseModule(context, options.hardwareMlirPath);
  if (!hardware)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not parse hardware MLIR");

  mlir::Operation *graph =
      findSymbolOp(*dfg, "dataflow.graph.func", options.graphName);
  if (!graph)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "could not find dataflow graph %s",
                                   options.graphName.c_str());
  auto selectionOrErr = selectFabricHardware(*hardware, options);
  if (!selectionOrErr)
    return selectionOrErr.takeError();
  if (mlir::failed(fabric::elaborateInstances(
          mlir::cast<fabric::ModuleOp>(selectionOrErr->module))))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "PnR could not elaborate selected fabric.module @%s",
        selectionOrErr->moduleName.c_str());

  auto nodesOrErr = collectSoftwareNodes(graph);
  if (!nodesOrErr)
    return nodesOrErr.takeError();
  auto hardwareModelOrErr =
      collectHardwareModel(selectionOrErr->module, selectionOrErr->moduleName);
  if (!hardwareModelOrErr)
    return hardwareModelOrErr.takeError();

  loom::ResolvedConfig resolvedConfig = loom::defaultResolvedConfig();
  MappingSummary summary(loom::resolvedConfigIdentity(resolvedConfig));
  summary.workload =
      options.workload.empty() ? options.graphName : options.workload;
  summary.hardware = selectionOrErr->summaryHardware;
  summary.hardwareRootKind = selectionOrErr->rootKind;
  summary.hardwareSystem = selectionOrErr->systemName;
  summary.selectedAccCore = selectionOrErr->accCoreName;
  summary.spatialcoreTemplate = selectionOrErr->moduleName;
  summary.graph = options.graphName;
  summary.mappingId =
      mappingId(summary.workload, summary.graph, summary.hardware);
  summary.configId = resolvedConfig.configId;
  summary.status = "pass";

  auto routingProblemOrErr = buildRoutingProblem(*nodesOrErr, graph);
  if (!routingProblemOrErr)
    return routingProblemOrErr.takeError();
  RoutingProblem routingProblem = std::move(*routingProblemOrErr);
  RouteCache routeCache;
  if (!placeRouteFeasible(*nodesOrErr, routingProblem,
                          hardwareModelOrErr->resources,
                          hardwareModelOrErr->topology, summary.placements,
                          routeCache)) {
    for (HardwareResource &resource : hardwareModelOrErr->resources)
      resource.used = false;
    for (SoftwareNode &node : *nodesOrErr) {
      HardwareResource *resource =
          claimResource(node, hardwareModelOrErr->resources);
      if (!resource) {
        summary.status = "fail";
        summary.diagnostic =
            "missing hardware resource for software op " + node.operation;
        ++summary.unplacedRecords;
        continue;
      }
      summary.placements.push_back(makePlacementRecord(node, *resource));
    }
    appendResourcePressureRecords(summary, *nodesOrErr,
                                  hardwareModelOrErr->resources);
    appendResourcePressureDiagnostic(summary);
  }

  llvm::StringMap<const SoftwareNode *> nodeById;
  for (const SoftwareNode &node : *nodesOrErr)
    nodeById.try_emplace(node.id, &node);
  llvm::StringMap<const HardwareResource *> resourceById;
  for (const HardwareResource &resource : hardwareModelOrErr->resources)
    resourceById.try_emplace(resource.id, &resource);
  for (const PlacementRecord &placement : summary.placements) {
    auto nodeIt = nodeById.find(placement.softwareId);
    auto resourceIt = resourceById.find(placement.hardwareId);
    if (nodeIt == nodeById.end() || resourceIt == resourceById.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "mapping placement references unknown software or hardware id");
    if (llvm::Error err = appendPlacementConfig(summary, *nodeIt->second,
                                                *resourceIt->second))
      return std::move(err);
  }

  RouteCollection routeCollection =
      collectRoutes(routingProblem, summary.placements,
                    hardwareModelOrErr->topology, routeCache);
  summary.routes = std::move(routeCollection.routes);
  summary.unroutedEdgeDetails = std::move(routeCollection.unroutedEdgeDetails);
  summary.unroutedEdges = routeCollection.unroutedEdges;
  if (summary.status == "pass" && summary.unroutedEdges != 0) {
    summary.status = "fail";
    summary.diagnostic = "unrouted software edges lack Fabric ADG connectivity";
  }
  if (summary.status == "pass") {
    appendRouteConfig(summary);
    if (llvm::Error err = validateConfigBitstream(summary))
      return std::move(err);
    summary.diagnostic = "mapped software graph to fabric resources";
  } else {
    summary.configEntries.clear();
  }
  return summary;
}
