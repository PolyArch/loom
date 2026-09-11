#include "StructuredPointerMemRefPromotion.h"

#include "Common/MappingDebugLog.h"
#include "Frontend/Analysis/MemoryProvenance.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <system_error>

namespace loom::frontend {
namespace {

/// Every vector transport the structured schedules admit is at most this wide,
/// so a static global promoted for them is aligned once at this bound.
constexpr std::uint64_t kPromotedGlobalAlignmentBytes = 16;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "structured pointer promotion: " + message);
}

struct PromotableAccess {
  mlir::Operation *access;
  mlir::LLVM::GEPOp address;
  mlir::BlockArgument root;
  mlir::Value index;
  mlir::Type element;
};

struct PromotableRoot {
  mlir::BlockArgument argument;
  mlir::LLVM::GlobalOp global;
  mlir::MemRefType memory;
  mlir::Value aligned;
};

/// The single element access an LLVM pointer access performs through one
/// region pointer input, or nothing when the access is any other shape.
std::optional<PromotableAccess> promotableAccess(mlir::Operation *operation,
                                                 mlir::Block *entry) {
  mlir::Value pointer;
  mlir::Type element;
  if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation)) {
    if (load.getVolatile_() ||
        load.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return std::nullopt;
    pointer = load.getAddr();
    element = load.getResult().getType();
  } else if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation)) {
    if (store.getVolatile_() ||
        store.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return std::nullopt;
    pointer = store.getAddr();
    element = store.getValue().getType();
  } else {
    return std::nullopt;
  }
  if (!element.isIntOrFloat())
    return std::nullopt;
  auto address = pointer.getDefiningOp<mlir::LLVM::GEPOp>();
  if (!address || !address.getResult().hasOneUse() ||
      address.getIndices().size() != 1)
    return std::nullopt;
  // The raised address may step in a byte array of the element's size rather
  // than in the element type itself; both address the same element sequence.
  const mlir::DataLayout layout = mlir::DataLayout::closest(operation);
  const llvm::TypeSize stride = layout.getTypeSize(address.getElemType());
  const llvm::TypeSize size = layout.getTypeSize(element);
  if (stride.isScalable() || size.isScalable() || stride != size ||
      size.getFixedValue() == 0)
    return std::nullopt;
  auto index = llvm::dyn_cast_if_present<mlir::Value>(address.getIndices()[0]);
  if (!index || !llvm::isa<mlir::IntegerType>(index.getType()))
    return std::nullopt;
  auto root = llvm::dyn_cast<mlir::BlockArgument>(address.getBase());
  if (!root || root.getOwner() != entry ||
      !llvm::isa<mlir::LLVM::LLVMPointerType>(root.getType()))
    return std::nullopt;
  return PromotableAccess{operation, address, root, index, element};
}

/// The static array a promotable root addresses at every launch site, as the
/// memref type of the promoted view, or nothing when the root is any other
/// object or its element type differs from the loop's accesses.
std::optional<std::pair<mlir::LLVM::GlobalOp, mlir::MemRefType>>
promotableRoot(mlir::BlockArgument root, mlir::Type element) {
  auto global = analysis::resolveUniqueStaticGlobalRoot(root);
  if (!global)
    return std::nullopt;
  auto array = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(global->getGlobalType());
  if (!array || array.getElementType() != element ||
      array.getNumElements() == 0)
    return std::nullopt;
  return std::pair{*global,
                   mlir::MemRefType::get(
                       {static_cast<std::int64_t>(array.getNumElements())},
                       element)};
}

/// Why one loop kept its pointer form.
enum class PromotionRefusal : std::uint8_t {
  None,
  MultiBlockBody,
  ForeignEffect,
  NoAccess,
  MixedElementType,
  RootNotStaticArray,
  UnpromotedAddressing,
  RootsNotDistinct,
  LoopCarriedValues,
  NonIntegerInduction,
};
llvm::StringRef spelling(PromotionRefusal refusal) {
  switch (refusal) {
  case PromotionRefusal::None: return "promoted";
  case PromotionRefusal::MultiBlockBody: return "multi_block_body";
  case PromotionRefusal::ForeignEffect: return "foreign_effect";
  case PromotionRefusal::NoAccess: return "no_pointer_access";
  case PromotionRefusal::MixedElementType: return "mixed_element_type";
  case PromotionRefusal::RootNotStaticArray: return "root_not_static_array";
  case PromotionRefusal::UnpromotedAddressing: return "unpromoted_addressing";
  case PromotionRefusal::RootsNotDistinct: return "roots_not_distinct";
  case PromotionRefusal::LoopCarriedValues: return "loop_carried_values";
  case PromotionRefusal::NonIntegerInduction: return "non_integer_induction";
  }
  llvm_unreachable("unknown promotion refusal");
}
llvm::Expected<PromotionRefusal> promoteLoop(mlir::scf::ForOp loop,
                                             mlir::Block *entry) {
  if (!loop.getRegion().hasOneBlock())
    return PromotionRefusal::MultiBlockBody;
  if (loop.getNumRegionIterArgs() != 0)
    return PromotionRefusal::LoopCarriedValues;
  if (!llvm::isa<mlir::IntegerType, mlir::IndexType>(
          loop.getInductionVar().getType()))
    return PromotionRefusal::NonIntegerInduction;
  llvm::SmallVector<PromotableAccess, 8> accesses;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (auto access = promotableAccess(&operation, entry)) {
      accesses.push_back(*access);
      continue;
    }
    if (llvm::isa<mlir::LLVM::GEPOp>(operation))
      continue;
    if (operation.getNumRegions() != 0 || !mlir::isMemoryEffectFree(&operation))
      return PromotionRefusal::ForeignEffect;
  }
  if (accesses.empty())
    return PromotionRefusal::NoAccess;
  llvm::MapVector<mlir::BlockArgument, PromotableRoot> roots;
  for (const PromotableAccess &access : accesses) {
    auto known = roots.find(access.root);
    if (known != roots.end()) {
      if (known->second.memory.getElementType() != access.element)
        return PromotionRefusal::MixedElementType;
      continue;
    }
    auto resolved = promotableRoot(access.root, access.element);
    if (!resolved)
      return PromotionRefusal::RootNotStaticArray;
    roots.insert({access.root,
                  PromotableRoot{access.root, resolved->first,
                                 resolved->second, mlir::Value()}});
  }
  for (mlir::Operation &operation : loop.getBody()->without_terminator())
    if (auto address = llvm::dyn_cast<mlir::LLVM::GEPOp>(operation))
      if (auto base = llvm::dyn_cast<mlir::BlockArgument>(address.getBase()))
        if (roots.count(base) &&
            llvm::none_of(accesses, [&](const PromotableAccess &access) {
              return access.address == address;
            }))
          return PromotionRefusal::UnpromotedAddressing;
  for (auto first = roots.begin(); first != roots.end(); ++first)
    for (auto second = std::next(first); second != roots.end(); ++second)
      if (!analysis::haveProvenDistinctMemoryRoots(first->first, second->first))
        return PromotionRefusal::RootsNotDistinct;

  mlir::OpBuilder builder(loop);
  builder.setInsertionPoint(loop);
  const mlir::Location location = loop.getLoc();
  llvm::SmallVector<mlir::Value, 4> views;
  std::uint64_t alignment = kPromotedGlobalAlignmentBytes;
  for (auto &[argument, root] : roots) {
    const std::uint64_t current = root.global.getAlignment().value_or(0);
    if (current < kPromotedGlobalAlignmentBytes)
      root.global.setAlignment(kPromotedGlobalAlignmentBytes);
    alignment = std::min(alignment,
                         std::max(current, kPromotedGlobalAlignmentBytes));
  }
  for (auto &[argument, root] : roots)
    views.push_back(loom::PointerViewOp::create(
        builder, location, root.memory, argument,
        builder.getI64IntegerAttr(static_cast<std::int64_t>(alignment))));
  llvm::SmallVector<mlir::Value, 4> distinct(views.begin(), views.end());
  if (views.size() > 1) {
    llvm::SmallVector<mlir::Type, 4> types;
    for (mlir::Value view : views)
      types.push_back(view.getType());
    auto objects =
        mlir::memref::DistinctObjectsOp::create(builder, location, types, views);
    distinct.assign(objects.getResults().begin(), objects.getResults().end());
  }
  for (auto [ordinal, entry] : llvm::enumerate(roots))
    entry.second.aligned = mlir::memref::AssumeAlignmentOp::create(
        builder, location, entry.second.memory, distinct[ordinal],
        builder.getI32IntegerAttr(static_cast<std::int32_t>(alignment)));
  mlir::Value induction = loop.getInductionVar();
  /// The index of one promoted access: the induction variable when the raised
  /// address only narrowed or widened it, otherwise the raised index cast to
  /// index. The cast is placed before the access so the loop body keeps its
  /// statement order.
  const auto accessIndex = [&](const PromotableAccess &access) -> mlir::Value {
    mlir::Value value = access.index;
    while (auto narrowed = value.getDefiningOp<mlir::arith::TruncIOp>())
      value = narrowed.getIn();
    while (auto widened = value.getDefiningOp<mlir::arith::ExtSIOp>())
      value = widened.getIn();
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
      if (cast.getIn() == induction)
        return induction;
    if (value == induction)
      return induction;
    builder.setInsertionPoint(access.access);
    return mlir::arith::IndexCastOp::create(builder, access.access->getLoc(),
                                            builder.getIndexType(), access.index);
  };
  for (const PromotableAccess &access : accesses) {
    const PromotableRoot &root = roots.find(access.root)->second;
    mlir::Value index = accessIndex(access);
    builder.setInsertionPoint(access.access);
    if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(access.access)) {
      auto promoted = mlir::memref::LoadOp::create(
          builder, load.getLoc(), root.aligned, mlir::ValueRange{index});
      load.getResult().replaceAllUsesWith(promoted.getResult());
    } else {
      auto store = llvm::cast<mlir::LLVM::StoreOp>(access.access);
      mlir::memref::StoreOp::create(builder, store.getLoc(), store.getValue(),
                                    root.aligned, mlir::ValueRange{index});
    }
    access.access->erase();
    mlir::LLVM::GEPOp address = access.address;
    address.erase();
  }
  // The raised address arithmetic is now dead; retire it before deciding
  // whether the old induction variable has any remaining reader.
  for (bool erased = true; erased;) {
    erased = false;
    for (mlir::Operation &operation :
         llvm::make_early_inc_range(loop.getBody()->without_terminator()))
      if (operation.use_empty() && mlir::isMemoryEffectFree(&operation)) {
        operation.erase();
        erased = true;
      }
  }
  // The exact structured schedules own loops with an `index` induction
  // variable. Retype the induction variable in place so the body block, which
  // ownership lineage tracks by identity, survives; every other reader of the
  // old variable reads it back through one cast at the body entry.
  const mlir::Type oldInductionType = induction.getType();
  if (!llvm::isa<mlir::IndexType>(oldInductionType)) {
    const mlir::Type indexType = builder.getIndexType();
    builder.setInsertionPoint(loop);
    const auto castBound = [&](mlir::Value bound) -> mlir::Value {
      if (llvm::isa<mlir::IndexType>(bound.getType()))
        return bound;
      llvm::APInt constant;
      if (mlir::matchPattern(bound, mlir::m_ConstantInt(&constant)) &&
          constant.isNonNegative() && constant.getActiveBits() < 63)
        return mlir::arith::ConstantIndexOp::create(
            builder, location, static_cast<std::int64_t>(constant.getZExtValue()));
      return mlir::arith::IndexCastOp::create(builder, location, indexType,
                                              bound);
    };
    mlir::Value lower = castBound(loop.getLowerBound());
    mlir::Value upper = castBound(loop.getUpperBound());
    mlir::Value step = castBound(loop.getStep());
    auto retyped = mlir::scf::ForOp::create(builder, location, lower, upper,
                                            step, mlir::ValueRange{});
    retyped.getRegion().takeBody(loop.getRegion());
    mlir::Block *body = retyped.getBody();
    mlir::BlockArgument newInduction = body->getArgument(0);
    // Readers of the old integer induction variable keep their type through
    // one cast; promoted accesses already index by the variable itself.
    llvm::SmallVector<mlir::OpOperand *, 8> integerReaders;
    for (mlir::OpOperand &use : newInduction.getUses())
      if (!llvm::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(
              use.getOwner()))
        integerReaders.push_back(&use);
    newInduction.setType(indexType);
    if (!integerReaders.empty()) {
      mlir::OpBuilder entryBuilder(body, body->begin());
      mlir::Value readback = mlir::arith::IndexCastOp::create(
          entryBuilder, location, oldInductionType, newInduction);
      for (mlir::OpOperand *use : integerReaders)
        use->set(readback);
    }
    for (mlir::NamedAttribute attribute : loop->getAttrs())
      if (!retyped->hasAttr(attribute.getName()))
        retyped->setAttr(attribute.getName(), attribute.getValue());
    loop.erase();
  }
  return PromotionRefusal::None;
}

} // namespace

llvm::Error promoteStaticPointerLoopsToMemRef(mlir::ModuleOp module) {
  llvm::SmallVector<loom::SpatialRegionOp, 4> regions;
  module.walk([&](loom::SpatialRegionOp region) { regions.push_back(region); });
  for (loom::SpatialRegionOp region : regions) {
    if (region.getBody().empty())
      continue;
    mlir::Block &entry = region.getBody().front();
    llvm::SmallVector<mlir::scf::ForOp, 4> loops;
    for (mlir::Operation &operation : entry.without_terminator())
      if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
        loops.push_back(loop);
    for (auto indexed : llvm::enumerate(loops)) {
      auto outcome = promoteLoop(indexed.value(), &entry);
      if (!outcome)
        return outcome.takeError();
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "structured_pointer_promotion";
            fields["loop_ordinal"] = indexed.index();
            fields["disposition"] = spelling(*outcome);
          });
    }
    if (mapping_debug::enabled(mapping_debug::Level::Detail))
      mapping_debug::emit(
          mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            std::string text;
            llvm::raw_string_ostream stream(text);
            region.print(stream);
            fields["operation"] = "structured_pointer_promotion_region";
            fields["region"] = std::move(text);
          });
  }
  return llvm::Error::success();
}

} // namespace loom::frontend
