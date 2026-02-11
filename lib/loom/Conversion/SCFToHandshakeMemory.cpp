//===-- SCFToHandshakeMemory.cpp - Memory control builder -------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements memory control logic for the SCF-to-Handshake conversion.
// It builds memory interface operations (ExternalMemoryOp, MemoryOp), connects
// load/store operations to memory ports, and constructs control token chains
// that enforce memory ordering through alias analysis and SCF path tracking.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"
#include "loom/Hardware/Common/FabricError.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "loom-scf-to-handshake"

namespace loom {
namespace detail {

using loom::dataflow::CarryOp;

namespace {

constexpr int64_t kOnChipMemThresholdBytes = 4096;

enum class RootKind { BlockArg, Alloc, Alloca, Global, Other };

static RootKind classifyRoot(mlir::Value root, llvm::StringRef &globalName) {
  if (!root)
    return RootKind::Other;
  if (mlir::isa<mlir::BlockArgument>(root))
    return RootKind::BlockArg;
  if (root.getDefiningOp<mlir::memref::AllocOp>())
    return RootKind::Alloc;
  if (root.getDefiningOp<mlir::memref::AllocaOp>())
    return RootKind::Alloca;
  if (auto global = root.getDefiningOp<mlir::memref::GetGlobalOp>()) {
    globalName = global.getName();
    return RootKind::Global;
  }
  return RootKind::Other;
}

static bool areGuaranteedNoAlias(mlir::Value lhs, mlir::Value rhs) {
  mlir::Value lhsRoot = getMemrefRoot(lhs);
  mlir::Value rhsRoot = getMemrefRoot(rhs);
  if (!lhsRoot || !rhsRoot)
    return false;
  if (lhsRoot == rhsRoot)
    return false;

  llvm::StringRef lhsGlobal;
  llvm::StringRef rhsGlobal;
  RootKind lhsKind = classifyRoot(lhsRoot, lhsGlobal);
  RootKind rhsKind = classifyRoot(rhsRoot, rhsGlobal);

  auto isUniqueRoot = [](RootKind kind) {
    return kind == RootKind::BlockArg || kind == RootKind::Alloc ||
           kind == RootKind::Alloca || kind == RootKind::Global;
  };

  if (!isUniqueRoot(lhsKind) || !isUniqueRoot(rhsKind))
    return false;
  if (lhsKind == RootKind::Global && rhsKind == RootKind::Global &&
      lhsGlobal == rhsGlobal)
    return false;
  return true;
}

static std::optional<int64_t> getConstantIndexValue(mlir::Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return getConstantIndexValue(cast.getIn());
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return getConstantIndexValue(cast.getIn());
  return std::nullopt;
}

static std::optional<llvm::SmallVector<int64_t, 4>>
getConstantIndices(mlir::ValueRange indices) {
  llvm::SmallVector<int64_t, 4> values;
  values.reserve(indices.size());
  for (mlir::Value index : indices) {
    auto constant = getConstantIndexValue(index);
    if (!constant)
      return std::nullopt;
    values.push_back(*constant);
  }
  return values;
}

static bool getStaticStridesAndOffset(mlir::MemRefType type,
                                      llvm::SmallVector<int64_t, 4> &strides,
                                      int64_t &offset) {
  if (!type.hasStaticShape())
    return false;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  if (offset == mlir::ShapedType::kDynamic)
    return false;
  for (int64_t stride : strides) {
    if (stride == mlir::ShapedType::kDynamic || stride < 0)
      return false;
  }
  return true;
}

static bool getStaticLinearRange(mlir::Value memref, int64_t &min,
                                 int64_t &max) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape())
    return false;
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (!getStaticStridesAndOffset(type, strides, offset))
    return false;
  auto shape = type.getShape();
  if (shape.empty())
    return false;
  int64_t minOffset = offset;
  int64_t maxOffset = offset;
  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    int64_t size = shape[i];
    if (size <= 0)
      return false;
    maxOffset += (size - 1) * strides[i];
  }
  if (maxOffset < minOffset)
    std::swap(maxOffset, minOffset);
  min = minOffset;
  max = maxOffset;
  return true;
}

static bool areDisjointStaticRanges(const MemAccess &lhs,
                                    const MemAccess &rhs) {
  if (!lhs.origMemref || !rhs.origMemref)
    return false;
  if (getMemrefRoot(lhs.origMemref) != getMemrefRoot(rhs.origMemref))
    return false;
  int64_t lhsMin = 0;
  int64_t lhsMax = 0;
  int64_t rhsMin = 0;
  int64_t rhsMax = 0;
  if (!getStaticLinearRange(lhs.origMemref, lhsMin, lhsMax))
    return false;
  if (!getStaticLinearRange(rhs.origMemref, rhsMin, rhsMax))
    return false;
  return lhsMax < rhsMin || rhsMax < lhsMin;
}

static bool hasSafeStaticStrides(mlir::MemRefType type) {
  llvm::SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return false;
  for (int64_t stride : strides) {
    if (stride == mlir::ShapedType::kDynamic || stride == 0)
      return false;
  }
  return true;
}

static bool areDisjointConstantElements(const MemAccess &lhs,
                                        const MemAccess &rhs) {
  if (lhs.origMemref != rhs.origMemref)
    return false;
  auto lhsLoad = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(lhs.origOp);
  auto rhsLoad = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(rhs.origOp);
  auto lhsStore = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(lhs.origOp);
  auto rhsStore = mlir::dyn_cast_or_null<mlir::memref::StoreOp>(rhs.origOp);

  mlir::Value memref = lhs.origMemref;
  if (!memref)
    return false;
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!memrefType || !hasSafeStaticStrides(memrefType))
    return false;

  mlir::ValueRange lhsIndices =
      lhsLoad ? lhsLoad.getIndices() : lhsStore.getIndices();
  mlir::ValueRange rhsIndices =
      rhsLoad ? rhsLoad.getIndices() : rhsStore.getIndices();

  if (lhsIndices.size() != rhsIndices.size())
    return false;
  if (lhsIndices.empty())
    return false;

  auto lhsConst = getConstantIndices(lhsIndices);
  auto rhsConst = getConstantIndices(rhsIndices);
  if (!lhsConst || !rhsConst)
    return false;

  for (size_t i = 0, e = lhsConst->size(); i < e; ++i) {
    if ((*lhsConst)[i] != (*rhsConst)[i])
      return true;
  }

  return false;
}

static bool pathMatchesPrefix(const ScfPath &path, const ScfPath &prefix) {
  if (path.size() < prefix.size())
    return false;
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (path[i].op != prefix[i].op || path[i].region != prefix[i].region)
      return false;
  }
  return true;
}

static bool isMutuallyExclusive(const ScfPath &lhs, const ScfPath &rhs) {
  unsigned common = std::min(lhs.size(), rhs.size());
  for (unsigned i = 0; i < common; ++i) {
    if (lhs[i].op != rhs[i].op)
      break;
    if (lhs[i].region == rhs[i].region)
      continue;
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(lhs[i].op))
      return true;
  }
  return false;
}

static bool hasDataDependence(mlir::Operation *src, mlir::Operation *dst) {
  if (!src || !dst)
    return false;
  llvm::SmallVector<mlir::Value, 8> worklist;
  for (mlir::Value res : src->getResults())
    worklist.push_back(res);
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (mlir::OpOperand &use : current.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == dst)
        return true;
      for (mlir::Value res : user->getResults())
        worklist.push_back(res);
    }
  }
  return false;
}

static void setAccessCtrl(MemAccess &access, mlir::Value ctrl) {
  if (access.kind == AccessKind::Load) {
    auto load = access.loadOp;
    unsigned addrCount = load.getAddresses().size();
    load->setOperand(addrCount + 1, ctrl);
  } else {
    auto store = access.storeOp;
    unsigned addrCount = store.getAddresses().size();
    store->setOperand(addrCount + 1, ctrl);
  }
}

static void dumpPath(llvm::raw_ostream &os, const ScfPath &path) {
  for (const PathEntry &entry : path) {
    if (!entry.op)
      continue;
    os << "/" << entry.op->getName().getStringRef() << ":" << entry.region;
  }
}

static void dumpValue(llvm::raw_ostream &os, mlir::Value value) {
  if (!value) {
    os << "<null>";
    return;
  }
  value.print(os);
}

static void dumpAccess(const MemAccess &access) {
  llvm::dbgs() << "[loom.memctrl] access order=" << access.order << " kind="
               << (access.kind == AccessKind::Load ? "load" : "store");
  if (access.origOp)
    llvm::dbgs() << " op=" << access.origOp->getName().getStringRef();
  llvm::dbgs() << " path=";
  dumpPath(llvm::dbgs(), access.path);
  llvm::dbgs() << " memref=";
  dumpValue(llvm::dbgs(), access.origMemref);
  llvm::dbgs() << "\n";
}

static void collectMemorySources(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Operation *> &memTokens,
    llvm::SmallPtrSetImpl<mlir::Value> &visited,
    llvm::SmallPtrSetImpl<mlir::Operation *> &sources) {
  if (!value)
    return;
  if (!mlir::isa<mlir::NoneType>(value.getType()))
    return;
  if (!visited.insert(value).second)
    return;
  auto it = memTokens.find(value);
  if (it != memTokens.end())
    sources.insert(it->second);
  if (mlir::Operation *def = value.getDefiningOp()) {
    for (mlir::Value operand : def->getOperands()) {
      if (!mlir::isa<mlir::NoneType>(operand.getType()))
        continue;
      collectMemorySources(operand, memTokens, visited, sources);
    }
  }
}

static void attachGlobalConstant(mlir::ModuleOp module,
                                 circt::handshake::MemoryOp memOp,
                                 mlir::memref::GetGlobalOp getGlobal) {
  if (!module || !memOp || !getGlobal)
    return;
  auto global =
      module.lookupSymbol<mlir::memref::GlobalOp>(getGlobal.getName());
  if (!global)
    return;
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(global.getType());
  if (!memrefType || !memrefType.hasStaticShape())
    return;
  auto sizeBytes = getStaticMemrefByteSize(memrefType);
  if (!sizeBytes)
    return;

  MemTargetHint hint = getMemTargetHint(getGlobal);
  if (hint == MemTargetHint::None)
    hint = getMemTargetHint(global);

  bool isConst = global.getConstant();
  bool forceExt = hint == MemTargetHint::Extmemory;
  bool forceRom = hint == MemTargetHint::Rom;
  if (!isConst && forceRom)
    return;
  if (!isConst && !forceRom)
    return;

  if (!forceRom && (*sizeBytes > kOnChipMemThresholdBytes))
    return;
  if (forceExt)
    return;

  mlir::Builder builder(module.getContext());
  std::string name = getGlobal.getName().str();
  std::string attrName = std::string("loom.global_constant.") + name;
  if (!module->hasAttr(attrName)) {
    llvm::SmallVector<mlir::NamedAttribute, 2> fields;
    fields.push_back(builder.getNamedAttr(
        "type", mlir::TypeAttr::get(memrefType)));
    if (auto init = global.getInitialValueAttr())
      fields.push_back(builder.getNamedAttr("data", init));
    module->setAttr(attrName, builder.getDictionaryAttr(fields));
  }

  memOp->setAttr("loom.global_memref",
                 mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
  memOp->setAttr("loom.global_constant",
                 builder.getStringAttr(name));
}

} // namespace

void HandshakeConversion::finalizeMemory() {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation *returnOp = nullptr;
  handshakeFunc.walk([&](circt::handshake::ReturnOp ret) {
    returnOp = ret.getOperation();
    return mlir::WalkResult::interrupt();
  });
  if (returnOp)
    builder.setInsertionPoint(returnOp);
  else
    builder.setInsertionPointToEnd(handshakeFunc.getBodyBlock());

  llvm::DenseMap<mlir::Value, llvm::SmallVector<MemAccess *, 4>> loadsByMemref;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<MemAccess *, 4>> storesByMemref;

  for (MemAccess &access : memAccesses) {
    if (!access.memref)
      continue;
    if (access.kind == AccessKind::Load)
      loadsByMemref[access.memref].push_back(&access);
    else
      storesByMemref[access.memref].push_back(&access);
  }

  llvm::SmallVector<mlir::Value, 8> memrefs;
  llvm::DenseSet<mlir::Value> memrefSet;
  for (auto &entry : loadsByMemref) {
    if (memrefSet.insert(entry.first).second)
      memrefs.push_back(entry.first);
  }
  for (auto &entry : storesByMemref) {
    if (memrefSet.insert(entry.first).second)
      memrefs.push_back(entry.first);
  }
  for (mlir::BlockArgument arg : handshakeFunc.getArguments()) {
    if (!mlir::isa<mlir::MemRefType>(arg.getType()))
      continue;
    if (memrefSet.insert(arg).second)
      memrefs.push_back(arg);
  }

  for (mlir::Value memrefValue : memrefs) {
    llvm::SmallVector<MemAccess *, 4> &loads = loadsByMemref[memrefValue];
    llvm::SmallVector<MemAccess *, 4> &stores = storesByMemref[memrefValue];

    llvm::SmallVector<mlir::Value, 4> operands;
    operands.reserve(loads.size() + stores.size());
    for (MemAccess *access : stores) {
      operands.push_back(access->storeOp.getDataResult());
      for (mlir::Value addr : access->storeOp.getAddressResult())
        operands.push_back(addr);
    }
    for (MemAccess *access : loads) {
      for (mlir::Value addr : access->loadOp.getAddressResults())
        operands.push_back(addr);
    }

    unsigned ldCount = loads.size();
    unsigned stCount = stores.size();
    mlir::Location loc = memrefValue.getLoc();

    bool useExternal = mlir::isa<mlir::BlockArgument>(memrefValue);
    if (auto memRefType =
            mlir::dyn_cast<mlir::MemRefType>(memrefValue.getType())) {
      if (!memRefType.hasStaticShape())
        useExternal = true;
    } else {
      continue;
    }

    mlir::Operation *memOp = nullptr;
    if (useExternal) {
      memOp = circt::handshake::ExternalMemoryOp::create(
                  builder, loc, memrefValue, operands, ldCount, stCount,
                  memoryId++)
                  .getOperation();
    } else {
      auto created = circt::handshake::MemoryOp::create(
          builder, loc, operands, ldCount, ldCount + stCount, false,
          memoryId++, memrefValue);
      memOp = created.getOperation();
      if (auto getGlobal =
              memrefValue.getDefiningOp<mlir::memref::GetGlobalOp>()) {
        auto module = handshakeFunc->getParentOfType<mlir::ModuleOp>();
        attachGlobalConstant(module, created, getGlobal);
      }
    }

    auto memResults = memOp->getResults();

    for (unsigned i = 0, e = loads.size(); i < e; ++i) {
      MemAccess *access = loads[i];
      auto load = access->loadOp;
      unsigned addrCount = load.getAddresses().size();
      load->setOperand(addrCount, memResults[i]);
      access->doneToken = memResults[ldCount + stCount + i];
    }

    for (unsigned i = 0, e = stores.size(); i < e; ++i) {
      stores[i]->doneToken = memResults[ldCount + i];
    }
  }

  for (mlir::BlockArgument arg : handshakeFunc.getArguments()) {
    if (!mlir::isa<mlir::MemRefType>(arg.getType()))
      continue;
    llvm::SmallVector<mlir::OpOperand *, 8> uses;
    for (mlir::OpOperand &use : arg.getUses()) {
      if (mlir::isa<circt::handshake::ExternalMemoryOp>(use.getOwner()))
        continue;
      uses.push_back(&use);
    }
    if (uses.empty())
      continue;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(handshakeFunc.getBodyBlock());
    auto cast = mlir::memref::CastOp::create(builder,
        arg.getLoc(), arg.getType(), arg);
    for (mlir::OpOperand *use : uses)
      use->set(cast.getResult());
  }
}

namespace {

class MemoryCtrlBuilder {
public:
  MemoryCtrlBuilder(circt::handshake::FuncOp func,
                    llvm::ArrayRef<MemAccess *> accesses,
                    mlir::AliasAnalysis &aliasAnalysis,
                    llvm::DenseMap<mlir::Operation *, mlir::Value> &forConds,
                    llvm::DenseMap<mlir::Operation *, mlir::Value> &whileConds,
                    llvm::DenseMap<mlir::Operation *, mlir::Value> &ifConds,
                    llvm::DenseMap<mlir::Operation *, mlir::Value> &switchConds,
                    mlir::Value entryControl)
      : func(func), aliasAnalysis(aliasAnalysis), forConds(forConds),
        whileConds(whileConds), ifConds(ifConds), switchConds(switchConds),
        builder(func.getContext()), entryControl(entryControl) {
    sortedAccesses.append(accesses.begin(), accesses.end());
    llvm::sort(sortedAccesses, [](MemAccess *lhs, MemAccess *rhs) {
      return lhs->order < rhs->order;
    });

    mlir::Operation *returnOp = nullptr;
    func.walk([&](circt::handshake::ReturnOp ret) {
      returnOp = ret.getOperation();
      return mlir::WalkResult::interrupt();
    });
    if (returnOp)
      builder.setInsertionPoint(returnOp);
    else
      builder.setInsertionPointToEnd(func.getBodyBlock());
  }

  mlir::LogicalResult run() {
    if (sortedAccesses.empty())
      return mlir::success();
    cursor = 0;
    mlir::Value entry = makeEntryToken(sortedAccesses.front()->origOp->getLoc());
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] start\n");
    doneToken = processLevel(ScfPath{}, entry);
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] end cursor=" << cursor << "\n");
    if (!failed && cursor != sortedAccesses.size()) {
      sortedAccesses[cursor]->origOp->emitError(
          "memory control did not consume all accesses");
      failed = true;
    }
    return failed ? mlir::failure() : mlir::success();
  }

  mlir::Value getDoneToken() const { return doneToken; }

private:
  mlir::Value makeEntryToken(mlir::Location loc) {
    if (entryControl)
      return entryControl;
    return circt::handshake::SourceOp::create(builder, loc, builder.getNoneType())
        .getResult();
  }

  mlir::Value makeConstant(mlir::Location loc, mlir::Attribute value,
                           mlir::Type type, mlir::Value ctrlToken) {
    mlir::Value ctrl = ctrlToken ? ctrlToken : makeEntryToken(loc);
    mlir::OperationState constState(
        loc, circt::handshake::ConstantOp::getOperationName());
    constState.addOperands(ctrl);
    constState.addTypes(type);
    constState.addAttribute("value", value);
    mlir::Operation *created = builder.create(constState);
    return created->getResult(0);
  }

  bool isAtLevel(const MemAccess *access, const ScfPath &path) const {
    return access && access->path.size() == path.size() &&
           pathMatchesPrefix(access->path, path);
  }

  bool canParallelize(const MemAccess *lhs, const MemAccess *rhs) const {
    if (!lhs || !rhs)
      return false;
    if (lhs->kind == AccessKind::Load && rhs->kind == AccessKind::Load)
      return true;
    if (aliasAnalysis.alias(lhs->origMemref, rhs->origMemref) ==
        mlir::AliasResult::NoAlias)
      return true;
    if (areDisjointStaticRanges(*lhs, *rhs))
      return true;
    if (areDisjointConstantElements(*lhs, *rhs))
      return true;
    if (hasDataDependence(lhs->origOp, rhs->origOp))
      return true;
    if (hasDataDependence(rhs->origOp, lhs->origOp))
      return true;
    return false;
  }

  mlir::Value processBatch(llvm::ArrayRef<MemAccess *> batch,
                           mlir::Value entryToken) {
    if (batch.empty())
      return entryToken;
    mlir::Location loc = batch.front()->origOp->getLoc();
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] batch size=" << batch.size()
                            << " entry=");
    LLVM_DEBUG(dumpValue(llvm::dbgs(), entryToken));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    LLVM_DEBUG({
      for (MemAccess *access : batch)
        if (access)
          dumpAccess(*access);
    });
    if (batch.size() == 1) {
      setAccessCtrl(*batch.front(), entryToken);
      return batch.front()->doneToken ? batch.front()->doneToken : entryToken;
    }

    // Distribute the control token to all parallel accesses via wire fanout.
    llvm::SmallVector<mlir::Value, 4> doneTokens;
    doneTokens.reserve(batch.size());
    for (unsigned i = 0; i < batch.size(); ++i) {
      setAccessCtrl(*batch[i], entryToken);
      if (batch[i]->doneToken)
        doneTokens.push_back(batch[i]->doneToken);
    }

    if (doneTokens.empty())
      return entryToken;
    if (doneTokens.size() == 1)
      return doneTokens.front();

    auto join = circt::handshake::JoinOp::create(builder, loc, doneTokens);
    return join.getResult();
  }

  mlir::Value lookupCond(mlir::Operation *op,
                         llvm::DenseMap<mlir::Operation *, mlir::Value> &map,
                         llvm::StringRef label) {
    auto it = map.find(op);
    if (it != map.end())
      return it->second;
    if (op)
      op->emitError("missing control value for ") << label;
    failed = true;
    return mlir::Value();
  }

  mlir::Value processIf(mlir::scf::IfOp op, const ScfPath &parentPath,
                        mlir::Value entryToken) {
    mlir::Value cond = lookupCond(op, ifConds, "scf.if");
    if (!cond)
      return entryToken;
    if (!cond.getType().isInteger(1)) {
      op.emitError("expected i1 condition for scf.if");
      failed = true;
      return entryToken;
    }
    mlir::Location loc = op.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] enter scf.if path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, cond, entryToken);
    mlir::Value thenEntry = branch.getTrueResult();
    mlir::Value elseEntry = branch.getFalseResult();

    ScfPath thenPath = parentPath;
    thenPath.push_back(PathEntry{op, 0});
    mlir::Value thenDone = processLevel(thenPath, thenEntry);

    ScfPath elsePath = parentPath;
    elsePath.push_back(PathEntry{op, 1});
    mlir::Value elseDone = processLevel(elsePath, elseEntry);

    mlir::Value zero =
        makeConstant(loc, builder.getIndexAttr(0), builder.getIndexType(),
                     entryToken);
    mlir::Value one =
        makeConstant(loc, builder.getIndexAttr(1), builder.getIndexType(),
                     entryToken);
    mlir::Value sel = mlir::arith::SelectOp::create(builder, loc, cond, one, zero);
    auto mux = circt::handshake::MuxOp::create(builder,
        loc, sel, mlir::ValueRange{elseDone, thenDone});
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] exit scf.if path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << " done=");
    LLVM_DEBUG(dumpValue(llvm::dbgs(), mux.getResult()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return mux.getResult();
  }

  mlir::Value processFor(mlir::scf::ForOp op, const ScfPath &parentPath,
                         mlir::Value entryToken) {
    mlir::Value cond = lookupCond(op, forConds, "scf.for");
    if (!cond)
      return entryToken;
    if (!cond.getType().isInteger(1)) {
      op.emitError("expected i1 condition for scf.for");
      failed = true;
      return entryToken;
    }
    mlir::Location loc = op.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] enter scf.for path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    auto carry = CarryOp::create(builder, loc, entryToken.getType(), cond,
                                         entryToken, entryToken);
    mlir::Value loopToken = carry.getO();

    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, cond, loopToken);
    mlir::Value bodyEntry = branch.getTrueResult();
    mlir::Value loopExit = branch.getFalseResult();

    ScfPath bodyPath = parentPath;
    bodyPath.push_back(PathEntry{op, 0});
    mlir::Value bodyDone = processLevel(bodyPath, bodyEntry);

    carry->setOperand(2, bodyDone);
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] exit scf.for path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << " done=");
    LLVM_DEBUG(dumpValue(llvm::dbgs(), loopExit));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return loopExit;
  }

  mlir::Value processWhile(mlir::scf::WhileOp op, const ScfPath &parentPath,
                           mlir::Value entryToken) {
    mlir::Value cond = lookupCond(op, whileConds, "scf.while");
    if (!cond)
      return entryToken;
    if (!cond.getType().isInteger(1)) {
      op.emitError("expected i1 condition for scf.while");
      failed = true;
      return entryToken;
    }
    mlir::Location loc = op.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] enter scf.while path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    auto carry = CarryOp::create(builder, loc, entryToken.getType(), cond,
                                         entryToken, entryToken);
    mlir::Value beforeEntry = carry.getO();

    ScfPath beforePath = parentPath;
    beforePath.push_back(PathEntry{op, 0});
    mlir::Value beforeDone = processLevel(beforePath, beforeEntry);

    auto branch = circt::handshake::ConditionalBranchOp::create(builder,
        loc, cond, beforeDone);
    mlir::Value afterEntry = branch.getTrueResult();
    mlir::Value loopExit = branch.getFalseResult();

    ScfPath afterPath = parentPath;
    afterPath.push_back(PathEntry{op, 1});
    mlir::Value afterDone = processLevel(afterPath, afterEntry);

    carry->setOperand(2, afterDone);
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] exit scf.while path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << " done=");
    LLVM_DEBUG(dumpValue(llvm::dbgs(), loopExit));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return loopExit;
  }

  mlir::Value processIndexSwitch(mlir::scf::IndexSwitchOp op,
                                 const ScfPath &parentPath,
                                 mlir::Value entryToken) {
    mlir::Value indexValue = lookupCond(op, switchConds, "scf.index_switch");
    if (!indexValue)
      return entryToken;
    if (!indexValue.getType().isIndex() &&
        !mlir::isa<mlir::IntegerType>(indexValue.getType())) {
      op.emitError("expected integer or index value for scf.index_switch");
      failed = true;
      return entryToken;
    }
    mlir::Location loc = op.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] enter scf.index_switch path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << "\n");

    mlir::Value chainToken = entryToken;
    llvm::SmallVector<mlir::Value, 4> branchTokens;
    llvm::SmallVector<ScfPath, 4> branchPaths;
    llvm::SmallVector<mlir::Value, 4> caseConds;

    auto cases = op.getCases();
    for (unsigned i = 0; i < cases.size(); ++i) {
      mlir::Value caseConst =
          makeConstant(loc, builder.getIndexAttr(cases[i]),
                       builder.getIndexType(), entryToken);
      mlir::Value caseCond = mlir::arith::CmpIOp::create(builder,
          loc, mlir::arith::CmpIPredicate::eq, indexValue, caseConst);
      caseConds.push_back(caseCond);
      auto branch = circt::handshake::ConditionalBranchOp::create(builder,
          loc, caseCond, chainToken);
      branchTokens.push_back(branch.getTrueResult());
      chainToken = branch.getFalseResult();

      ScfPath casePath = parentPath;
      casePath.push_back(PathEntry{op, i});
      branchPaths.push_back(std::move(casePath));
    }

    branchTokens.push_back(chainToken);
    ScfPath defaultPath = parentPath;
    defaultPath.push_back(PathEntry{op, op.getNumRegions() - 1});
    branchPaths.push_back(std::move(defaultPath));

    llvm::SmallVector<mlir::Value, 4> doneTokens;
    doneTokens.reserve(branchTokens.size());
    for (unsigned i = 0; i < branchTokens.size(); ++i) {
      doneTokens.push_back(processLevel(branchPaths[i], branchTokens[i]));
    }

    mlir::Value select =
        makeConstant(loc, builder.getIndexAttr(cases.size()),
                     builder.getIndexType(), entryToken);
    for (int64_t i = static_cast<int64_t>(caseConds.size()) - 1; i >= 0; --i) {
      mlir::Value caseIndex =
          makeConstant(loc, builder.getIndexAttr(i),
                       builder.getIndexType(), entryToken);
      select = mlir::arith::SelectOp::create(builder,
          loc, caseConds[static_cast<size_t>(i)], caseIndex, select);
    }
    auto mux = circt::handshake::MuxOp::create(builder, loc, select, doneTokens);
    LLVM_DEBUG(llvm::dbgs() << "[loom.memctrl] exit scf.index_switch path=");
    LLVM_DEBUG(dumpPath(llvm::dbgs(), parentPath));
    LLVM_DEBUG(llvm::dbgs() << " done=");
    LLVM_DEBUG(dumpValue(llvm::dbgs(), mux.getResult()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
    return mux.getResult();
  }

  mlir::Value processChild(const ScfPath &parentPath, const PathEntry &child,
                           mlir::Value entryToken) {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(child.op))
      return processFor(forOp, parentPath, entryToken);
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(child.op))
      return processWhile(whileOp, parentPath, entryToken);
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(child.op))
      return processIf(ifOp, parentPath, entryToken);
    if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(child.op))
      return processIndexSwitch(switchOp, parentPath, entryToken);
    child.op->emitError("unsupported scf op in memory control");
    failed = true;
    return entryToken;
  }

  mlir::Value processLevel(const ScfPath &path, mlir::Value entryToken) {
    size_t guard = 0;
    while (cursor < sortedAccesses.size()) {
      MemAccess *access = sortedAccesses[cursor];
      if (!pathMatchesPrefix(access->path, path))
        break;

      if (access->path.size() > path.size()) {
        PathEntry child = access->path[path.size()];
        size_t beforeCursor = cursor;
        entryToken = processChild(path, child, entryToken);
        if (cursor == beforeCursor && failed)
          break;
        if (cursor == beforeCursor) {
          access->origOp->emitError("memory control did not advance cursor");
          failed = true;
          break;
        }
        continue;
      }

      llvm::SmallVector<MemAccess *, 4> batch;
      batch.push_back(access);
      size_t lookAhead = cursor + 1;
      while (lookAhead < sortedAccesses.size()) {
        MemAccess *next = sortedAccesses[lookAhead];
        if (!isAtLevel(next, path))
          break;
        bool ok = true;
        for (MemAccess *existing : batch) {
          if (!canParallelize(existing, next)) {
            ok = false;
            break;
          }
        }
        if (!ok)
          break;
        batch.push_back(next);
        ++lookAhead;
      }
      cursor = lookAhead;
      entryToken = processBatch(batch, entryToken);
      if (++guard > sortedAccesses.size() * 2) {
        access->origOp->emitError("memory control exceeded iteration guard");
        failed = true;
        break;
      }
    }
    return entryToken;
  }

  circt::handshake::FuncOp func;
  mlir::AliasAnalysis &aliasAnalysis;
  llvm::DenseMap<mlir::Operation *, mlir::Value> &forConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> &whileConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> &ifConds;
  llvm::DenseMap<mlir::Operation *, mlir::Value> &switchConds;
  mlir::OpBuilder builder;
  llvm::SmallVector<MemAccess *, 16> sortedAccesses;
  mlir::Value entryControl;
  mlir::Value doneToken;
  size_t cursor = 0;
  bool failed = false;
};

} // namespace

mlir::LogicalResult HandshakeConversion::buildMemoryControl() {
  memoryDoneToken = entryToken;
  if (memAccesses.empty())
    return mlir::success();

  for (MemAccess &access : memAccesses) {
    if (!access.controlToken) {
      access.origOp->emitError("missing memory control token");
      return mlir::failure();
    }
    if (!access.doneToken) {
      access.origOp->emitError("missing memory done token");
      return mlir::failure();
    }
  }

  auto requireCond = [&](mlir::Operation *op, auto &map,
                         llvm::StringRef label) -> bool {
    if (!op)
      return false;
    if (map.find(op) != map.end())
      return true;
    op->emitError("missing control value for ") << label;
    return false;
  };
  for (MemAccess &access : memAccesses) {
    for (const PathEntry &entry : access.path) {
      if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(entry.op)) {
        if (!requireCond(forOp, forConds, "scf.for"))
          return mlir::failure();
      } else if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(entry.op)) {
        if (!requireCond(whileOp, whileConds, "scf.while"))
          return mlir::failure();
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(entry.op)) {
        if (!requireCond(ifOp, ifConds, "scf.if"))
          return mlir::failure();
      } else if (auto switchOp =
                     mlir::dyn_cast<mlir::scf::IndexSwitchOp>(entry.op)) {
        if (!requireCond(switchOp, switchConds, "scf.index_switch"))
          return mlir::failure();
      }
    }
  }

  llvm::DenseMap<mlir::Value, unsigned> memrefIds;
  llvm::SmallVector<mlir::Value, 8> memrefs;
  memrefs.reserve(memAccesses.size());

  for (MemAccess &access : memAccesses) {
    if (!access.origMemref)
      continue;
    if (memrefIds.find(access.origMemref) == memrefIds.end()) {
      unsigned id = memrefs.size();
      memrefIds[access.origMemref] = id;
      memrefs.push_back(access.origMemref);
    }
  }

  if (memrefs.empty())
    return mlir::success();

  llvm::SmallVector<unsigned, 8> parent(memrefs.size());
  for (unsigned i = 0; i < parent.size(); ++i)
    parent[i] = i;

  auto findRoot = [&](auto self, unsigned v) -> unsigned {
    if (parent[v] == v)
      return v;
    parent[v] = self(self, parent[v]);
    return parent[v];
  };

  auto unite = [&](unsigned a, unsigned b) {
    unsigned ra = findRoot(findRoot, a);
    unsigned rb = findRoot(findRoot, b);
    if (ra != rb)
      parent[rb] = ra;
  };

  for (unsigned i = 0; i < memrefs.size(); ++i) {
    for (unsigned j = i + 1; j < memrefs.size(); ++j) {
      if (areGuaranteedNoAlias(memrefs[i], memrefs[j]))
        continue;
      if (aliasAnalysis.alias(memrefs[i], memrefs[j]) !=
          mlir::AliasResult::NoAlias)
        unite(i, j);
    }
  }

  llvm::DenseMap<unsigned, llvm::SmallVector<MemAccess *, 8>> groups;
  for (MemAccess &access : memAccesses) {
    auto it = memrefIds.find(access.origMemref);
    if (it == memrefIds.end())
      continue;
    unsigned root = findRoot(findRoot, it->second);
    groups[root].push_back(&access);
  }

  llvm::SmallVector<mlir::Value, 4> doneTokens;
  for (auto &entry : groups) {
    MemoryCtrlBuilder builder(handshakeFunc, entry.second, aliasAnalysis,
                              forConds, whileConds, ifConds, switchConds,
                              entryToken);
    if (mlir::failed(builder.run()))
      return mlir::failure();
    if (mlir::Value done = builder.getDoneToken())
      doneTokens.push_back(done);
  }

  if (!doneTokens.empty()) {
    mlir::OpBuilder joinBuilder(handshakeFunc.getContext());
    joinBuilder.setInsertionPointToEnd(handshakeFunc.getBodyBlock());
    if (doneTokens.size() == 1) {
      memoryDoneToken = doneTokens.front();
    } else {
      auto join = circt::handshake::JoinOp::create(joinBuilder,
          handshakeFunc.getLoc(), doneTokens);
      memoryDoneToken = join.getResult();
    }
  } else {
    memoryDoneToken = entryToken;
  }

  return mlir::success();
}

mlir::LogicalResult HandshakeConversion::verifyMemoryControl() {
  llvm::DenseMap<mlir::Value, mlir::Operation *> memTokens;
  memTokens.reserve(memAccesses.size());
  for (MemAccess &access : memAccesses) {
    if (!access.doneToken)
      continue;
    mlir::Operation *owner = access.doneToken.getDefiningOp();
    if (!owner)
      continue;
    if (!mlir::isa<circt::handshake::ExternalMemoryOp,
                   circt::handshake::MemoryOp>(owner))
      continue;
    memTokens[access.doneToken] = owner;
  }

  llvm::DenseMap<mlir::Operation *, mlir::Operation *> allowedMem;
  allowedMem.reserve(memAccesses.size());
  for (MemAccess &access : memAccesses) {
    mlir::Operation *op = nullptr;
    if (access.loadOp)
      op = access.loadOp.getOperation();
    else if (access.storeOp)
      op = access.storeOp.getOperation();
    if (!op || !access.doneToken)
      continue;
    mlir::Operation *owner = access.doneToken.getDefiningOp();
    if (!owner)
      continue;
    if (!mlir::isa<circt::handshake::ExternalMemoryOp,
                   circt::handshake::MemoryOp>(owner))
      continue;
    allowedMem[op] = owner;
  }

  bool failed = false;
  handshakeFunc.walk([&](mlir::Operation *op) {
    mlir::Value ctrl;
    mlir::Operation *expectedMem = nullptr;
    if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op)) {
      unsigned addrCount = load.getAddresses().size();
      ctrl = load->getOperand(addrCount + 1);
      expectedMem = allowedMem.lookup(op);
    } else if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op)) {
      unsigned addrCount = store.getAddresses().size();
      ctrl = store->getOperand(addrCount + 1);
      expectedMem = allowedMem.lookup(op);
    } else {
      return;
    }

    llvm::SmallPtrSet<mlir::Value, 32> visited;
    llvm::SmallPtrSet<mlir::Operation *, 4> sources;
    collectMemorySources(ctrl, memTokens, visited, sources);

    bool sawEntry = entryToken && visited.contains(entryToken);
    if (!sources.empty()) {
      for (mlir::Operation *source : sources) {
        if (expectedMem && source == expectedMem)
          continue;
        op->emitError(std::string(CompError::HANDSHAKE_CTRL_MULTI_MEM) +
                      ": control token depends on "
                      "a memory interface not associated with this access");
        failed = true;
        return;
      }
    }
    if (sources.empty() && !sawEntry) {
      op->emitError(std::string(CompError::HANDSHAKE_CTRL_MULTI_MEM) +
                    ": control token is not "
                    "rooted at start_token");
      failed = true;
      return;
    }
    if (!expectedMem && !sources.empty()) {
      op->emitError(std::string(CompError::HANDSHAKE_CTRL_MULTI_MEM) +
                    ": missing memory mapping for "
                    "access control check");
      failed = true;
      return;
    }
  });

  return failed ? mlir::failure() : mlir::success();
}

} // namespace detail
} // namespace loom
