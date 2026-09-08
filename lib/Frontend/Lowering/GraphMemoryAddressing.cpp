#include "Frontend/Lowering/GraphMemoryAddressing.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Analysis/MemoryProvenance.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace loom::lowering {
namespace {

mlir::Value resolveMemoryServiceBoundaryRootImpl(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot,
    llvm::DenseSet<mlir::Value> &visiting,
    const PointerServiceBindings &bindings) {
  if (!pointer || !visiting.insert(pointer).second)
    return {};
  if (isBoundaryRoot(pointer))
    return pointer;
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parent)) {
      if (argument.getOwner() == loop.getBody() &&
          argument.getArgNumber() > 0) {
        unsigned ordinal = argument.getArgNumber() - 1;
        if (ordinal < loop.getInitArgs().size())
          return resolveMemoryServiceBoundaryRootImpl(
              loop.getInitArgs()[ordinal], isBoundaryRoot, visiting, bindings);
      }
    }
    if (auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(parent)) {
      unsigned ordinal = argument.getArgNumber();
      if (argument.getOwner() == loop.getBeforeBody() &&
          ordinal < loop.getInits().size())
        return resolveMemoryServiceBoundaryRootImpl(
            loop.getInits()[ordinal], isBoundaryRoot, visiting, bindings);
      if (argument.getOwner() == loop.getAfterBody() &&
          ordinal < loop.getConditionOp().getArgs().size())
        return resolveMemoryServiceBoundaryRootImpl(
            loop.getConditionOp().getArgs()[ordinal], isBoundaryRoot, visiting,
            bindings);
    }
    return {};
  }
  mlir::Value descriptorAddress;
  if (auto read = pointer.getDefiningOp<mlir::LLVM::LoadOp>())
    descriptorAddress = read.getAddr();
  else if (auto read = pointer.getDefiningOp<dataflow::LoadOp>()) {
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(read.getAddr().getType())) {
      mlir::Value target = bindings.lookup(read.getMem());
      return target && isBoundaryRoot(target) ? target : mlir::Value{};
    }
    descriptorAddress = read.getAddr();
  }
  if (descriptorAddress) {
    mlir::Value descriptor = resolveMemoryServiceBoundaryRootImpl(
        descriptorAddress, isBoundaryRoot, visiting, bindings);
    mlir::Value target =
        descriptor ? bindings.lookup(descriptor) : mlir::Value{};
    return target && isBoundaryRoot(target) ? target : mlir::Value{};
  }
  if (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>())
    return resolveMemoryServiceBoundaryRootImpl(gep.getBase(), isBoundaryRoot,
                                                visiting, bindings);
  if (auto carry = pointer.getDefiningOp<dataflow::CarryOp>())
    return resolveMemoryServiceBoundaryRootImpl(carry.getInit(), isBoundaryRoot,
                                                visiting, bindings);
  if (auto invariant = pointer.getDefiningOp<dataflow::InvariantOp>())
    return resolveMemoryServiceBoundaryRootImpl(
        invariant.getInit(), isBoundaryRoot, visiting, bindings);
  if (auto gate = pointer.getDefiningOp<dataflow::GateOp>())
    return resolveMemoryServiceBoundaryRootImpl(
        gate.getBeforeValue(), isBoundaryRoot, visiting, bindings);
  if (auto sync = pointer.getDefiningOp<dataflow::SyncOp>()) {
    unsigned ordinal = llvm::cast<mlir::OpResult>(pointer).getResultNumber();
    if (ordinal < sync.getInputs().size())
      return resolveMemoryServiceBoundaryRootImpl(
          sync.getInputs()[ordinal], isBoundaryRoot, visiting, bindings);
  }
  if (auto select = pointer.getDefiningOp<mlir::arith::SelectOp>()) {
    llvm::DenseSet<mlir::Value> truePath = visiting;
    llvm::DenseSet<mlir::Value> falsePath = visiting;
    mlir::Value trueRoot = resolveMemoryServiceBoundaryRootImpl(
        select.getTrueValue(), isBoundaryRoot, truePath, bindings);
    mlir::Value falseRoot = resolveMemoryServiceBoundaryRootImpl(
        select.getFalseValue(), isBoundaryRoot, falsePath, bindings);
    if (trueRoot && trueRoot == falseRoot)
      return trueRoot;
  }
  if (auto result = llvm::dyn_cast<mlir::OpResult>(pointer)) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      if (ordinal >= loop.getInitArgs().size() ||
          ordinal >= loop.getYieldedValues().size())
        return {};
      llvm::DenseSet<mlir::Value> initialPath = visiting;
      llvm::DenseSet<mlir::Value> yieldedPath = visiting;
      mlir::Value initial = resolveMemoryServiceBoundaryRootImpl(
          loop.getInitArgs()[ordinal], isBoundaryRoot, initialPath, bindings);
      mlir::Value yielded = resolveMemoryServiceBoundaryRootImpl(
          loop.getYieldedValues()[ordinal], isBoundaryRoot, yieldedPath,
          bindings);
      if (initial && initial == yielded)
        return initial;
    }
    if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      if (ordinal >= loop.getInits().size() ||
          ordinal >= loop.getYieldOp().getNumOperands())
        return {};
      llvm::DenseSet<mlir::Value> initialPath = visiting;
      llvm::DenseSet<mlir::Value> yieldedPath = visiting;
      mlir::Value initial = resolveMemoryServiceBoundaryRootImpl(
          loop.getInits()[ordinal], isBoundaryRoot, initialPath, bindings);
      mlir::Value yielded = resolveMemoryServiceBoundaryRootImpl(
          loop.getYieldOp().getOperand(ordinal), isBoundaryRoot, yieldedPath,
          bindings);
      if (initial && initial == yielded)
        return initial;
    }
    if (auto branch = llvm::dyn_cast<mlir::scf::IfOp>(result.getOwner())) {
      unsigned ordinal = result.getResultNumber();
      auto thenYield = llvm::dyn_cast<mlir::scf::YieldOp>(
          branch.getThenRegion().front().getTerminator());
      auto elseYield =
          branch.getElseRegion().empty()
              ? mlir::scf::YieldOp{}
              : llvm::dyn_cast<mlir::scf::YieldOp>(
                    branch.getElseRegion().front().getTerminator());
      if (!thenYield || !elseYield || ordinal >= thenYield.getNumOperands() ||
          ordinal >= elseYield.getNumOperands())
        return {};
      llvm::DenseSet<mlir::Value> thenPath = visiting;
      llvm::DenseSet<mlir::Value> elsePath = visiting;
      mlir::Value thenRoot = resolveMemoryServiceBoundaryRootImpl(
          thenYield.getOperand(ordinal), isBoundaryRoot, thenPath, bindings);
      mlir::Value elseRoot = resolveMemoryServiceBoundaryRootImpl(
          elseYield.getOperand(ordinal), isBoundaryRoot, elsePath, bindings);
      if (thenRoot && thenRoot == elseRoot)
        return thenRoot;
    }
  }
  return {};
}

} // namespace

mlir::Value resolveMemoryServiceBoundaryRoot(
    mlir::Value pointer, llvm::function_ref<bool(mlir::Value)> isBoundaryRoot,
    const PointerServiceBindings &bindings) {
  llvm::DenseSet<mlir::Value> visiting;
  return resolveMemoryServiceBoundaryRootImpl(pointer, isBoundaryRoot, visiting,
                                              bindings);
}

bool usesLoadedPointerService(mlir::Value pointer) {
  return static_cast<bool>(resolveMemoryServiceBoundaryRoot(
      pointer, [](mlir::Value value) {
        auto read = value.getDefiningOp<mlir::LLVM::LoadOp>();
        return read && llvm::isa<mlir::LLVM::LLVMPointerType>(read.getType());
      }));
}

PointerServiceBindingsOutcome projectPointerServiceBindings(
    llvm::ArrayRef<mlir::Operation *> selectedBody,
    llvm::ArrayRef<mlir::Value> boundaryValues,
    frontend::analysis::StoredMemoryProvenance &provenance) {
  PointerServiceBindings bindings;
  std::optional<frontend::analysis::StoredPointerRefusal> refusal;
  auto isBoundary = [&](mlir::Value value) {
    return llvm::is_contained(boundaryValues, value);
  };
  for (mlir::Operation *topLevel : selectedBody) {
    topLevel->walk([&](mlir::LLVM::LoadOp read) {
      if (refusal || !llvm::isa<mlir::LLVM::LLVMPointerType>(read.getType()))
        return;
      auto projected =
          provenance.projectPointerTarget(read.getResult(), boundaryValues);
      if (auto failed = std::get_if<frontend::analysis::StoredPointerRefusal>(
              &projected)) {
        refusal = *failed;
        return;
      }
      mlir::Value target =
          std::get<frontend::analysis::StoredPointerTarget>(projected).root;
      mlir::Value descriptor = resolveMemoryServiceBoundaryRoot(
          read.getAddr(), isBoundary, bindings);
      if (!descriptor) {
        refusal = frontend::analysis::StoredPointerRefusal::OriginNotInBoundary;
        return;
      }
      auto [entry, inserted] = bindings.try_emplace(descriptor, target);
      if (!inserted && entry->second != target)
        refusal =
            frontend::analysis::StoredPointerRefusal::DistinctPointerOrigins;
    });
    if (refusal)
      return *refusal;
  }
  return bindings;
}

ExactPointerPointAccessOutcome projectExactPointerPointAccess(
    mlir::Operation *operation, mlir::Operation *enclosingRoot,
    llvm::function_ref<bool(mlir::Value)> isPointCoordinate) {
  mlir::Value pointer;
  mlir::Type accessType;
  bool writes = false;
  if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation)) {
    if (load.getVolatile_() ||
        load.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return ExactPointerPointAccessRefusal::UnsupportedEffect;
    pointer = load.getAddr();
    accessType = load.getType();
  } else if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation)) {
    if (store.getVolatile_() ||
        store.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return ExactPointerPointAccessRefusal::UnsupportedEffect;
    pointer = store.getAddr();
    accessType = store.getValue().getType();
    writes = true;
  } else {
    return ExactPointerPointAccessRefusal::NotMemoryAccess;
  }
  if (!accessType.isIntOrFloat())
    return ExactPointerPointAccessRefusal::UnsupportedElementType;

  auto address = pointer.getDefiningOp<mlir::LLVM::GEPOp>();
  if (!address ||
      !mlir::LLVM::bitEnumContainsAny(
          address.getNoWrapFlags(), mlir::LLVM::GEPNoWrapFlags::inboundsFlag) ||
      address.getRawConstantIndices().size() != 1 ||
      address.getRawConstantIndices().front() !=
          mlir::LLVM::GEPOp::kDynamicIndex ||
      !llvm::hasSingleElement(address.getDynamicIndices()))
    return ExactPointerPointAccessRefusal::NonDirectInboundsAddress;

  // A direct GEP may start at a row base computed by an enclosing loop. That
  // invariant pointer owns this loop's point partition; resolving through it
  // would mix the enclosing coordinate into an otherwise exact local proof.
  // Cross-root alias analysis still traces the base to its allocation owner.
  mlir::Value root = address.getBase();
  mlir::Region *rootRegion = root.getParentRegion();
  if (enclosingRoot && enclosingRoot->getNumRegions() != 0 && rootRegion) {
    mlir::Region &enclosingRegion = enclosingRoot->getRegion(0);
    if (rootRegion == &enclosingRegion ||
        enclosingRegion.isAncestor(rootRegion))
      return ExactPointerPointAccessRefusal::NonLocalRoot;
  }
  auto resolved = frontend::analysis::resolveLinearPointerAddress(
      pointer, accessType,
      [&](mlir::Value candidate) { return candidate == root; });
  if (!resolved || resolved->addressBitWidth != 64 ||
      resolved->terms.size() != 1 ||
      !isPointCoordinate(resolved->terms.front().index) ||
      resolved->terms.front().byteStride <= 0 || resolved->byteBias != 0 ||
      resolved->elementBias != 0 ||
      resolved->elementAllocByteCount != resolved->accessByteCount ||
      static_cast<std::uint64_t>(resolved->terms.front().byteStride) !=
          resolved->elementAllocByteCount)
    return ExactPointerPointAccessRefusal::AddressRelationNotEstablished;

  return ExactPointerPointAccess{operation, root, address, writes,
                                 resolved->elementAllocByteCount};
}

bool isSameSignedMemoryCoordinate(mlir::Value value, mlir::Value expected,
                                  mlir::Operation *anchor) {
  while (value != expected) {
    if (auto truncation = value.getDefiningOp<mlir::arith::TruncIOp>()) {
      if (!mlir::arith::bitEnumContainsAny(
              truncation.getOverflowFlags(),
              mlir::arith::IntegerOverflowFlags::nsw))
        return false;
      value = truncation.getIn();
      continue;
    }
    if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
      value = extension.getIn();
      continue;
    }
    auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>();
    if (!cast)
      return false;
    auto indexWidth = getIndexBitWidth(anchor);
    if (!indexWidth) {
      llvm::consumeError(indexWidth.takeError());
      return false;
    }
    const auto width = [&](mlir::Type type) {
      if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
        return integer.getWidth();
      return *indexWidth;
    };
    if (width(cast.getIn().getType()) > width(cast.getType()))
      return false;
    value = cast.getIn();
  }
  return true;
}

ExactPointerPointAccessPairKind
classifyExactPointerPointAccessPair(const ExactPointerPointAccess &lhs,
                                    const ExactPointerPointAccess &rhs) {
  if (!lhs.writes && !rhs.writes)
    return ExactPointerPointAccessPairKind::NoDependence;
  if (lhs.root == rhs.root)
    return lhs.elementBytes == rhs.elementBytes
               ? ExactPointerPointAccessPairKind::SameRootIterationLocal
               : ExactPointerPointAccessPairKind::ByteRelationNotEstablished;
  return frontend::analysis::haveProvenDistinctMemoryRoots(lhs.root, rhs.root)
             ? ExactPointerPointAccessPairKind::NoDependence
             : ExactPointerPointAccessPairKind::AliasNotEstablished;
}

} // namespace loom::lowering
