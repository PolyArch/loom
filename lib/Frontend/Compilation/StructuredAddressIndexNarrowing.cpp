#include "StructuredAddressIndexNarrowing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <limits>
#include <string>

namespace loom::frontend::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "ownership_candidate_invalid: canonical index materialization " +
          message);
}

mlir::IntegerAttr integerConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  return constant ? llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                  : mlir::IntegerAttr{};
}

bool fitsSignedWidth(mlir::IntegerAttr value, unsigned width) {
  return value && value.getValue().isSignedIntN(width);
}

/// Proves the exact post-tested counted-loop shape emitted by pinned
/// CFG-to-SCF structurization. The induction value used by the loop body is
/// start, start+step, ..., bound-step. A positive no-signed-wrap step and exact
/// landing on the constant bound make that complete domain finite and known.
bool provesPostTestedInductionDomain(mlir::Value value, unsigned targetWidth) {
  auto induction = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!induction)
    return false;
  auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(
      induction.getOwner()->getParentOp());
  if (!loop || induction.getOwner() != loop.getBeforeBody())
    return false;

  const unsigned lane = induction.getArgNumber();
  if (lane >= loop.getInits().size())
    return false;
  mlir::IntegerAttr startAttr = integerConstant(loop.getInits()[lane]);
  if (!startAttr || startAttr.getType() != induction.getType())
    return false;

  mlir::scf::ConditionOp condition = loop.getConditionOp();
  if (lane >= condition.getArgs().size())
    return false;
  auto update = condition.getArgs()[lane].getDefiningOp<mlir::arith::AddIOp>();
  if (!update || update->getParentRegion() != &loop.getBefore() ||
      !mlir::arith::bitEnumContainsAny(update.getOverflowFlags(),
                                       mlir::arith::IntegerOverflowFlags::nsw))
    return false;

  mlir::Value stepValue;
  if (update.getLhs() == induction)
    stepValue = update.getRhs();
  else if (update.getRhs() == induction)
    stepValue = update.getLhs();
  else
    return false;
  mlir::IntegerAttr stepAttr = integerConstant(stepValue);
  if (!stepAttr || stepAttr.getType() != induction.getType() ||
      !stepAttr.getValue().isStrictlyPositive())
    return false;

  auto compare = condition.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare->getParentRegion() != &loop.getBefore())
    return false;
  mlir::Value boundValue;
  if (compare.getLhs() == update.getResult())
    boundValue = compare.getRhs();
  else if (compare.getRhs() == update.getResult())
    boundValue = compare.getLhs();
  else
    return false;
  mlir::IntegerAttr boundAttr = integerConstant(boundValue);
  if (!boundAttr || boundAttr.getType() != induction.getType())
    return false;

  mlir::Block *afterBody = loop.getAfterBody();
  mlir::scf::YieldOp yield = loop.getYieldOp();
  if (lane >= afterBody->getNumArguments() ||
      lane >= yield.getResults().size() ||
      yield.getResults()[lane] != afterBody->getArgument(lane))
    return false;

  auto sourceType = llvm::dyn_cast<mlir::IntegerType>(induction.getType());
  if (!sourceType || sourceType.getWidth() <= targetWidth)
    return false;
  const unsigned arithmeticWidth = sourceType.getWidth() + 1;
  llvm::APInt start = startAttr.getValue().sext(arithmeticWidth);
  llvm::APInt step = stepAttr.getValue().sext(arithmeticWidth);
  llvm::APInt bound = boundAttr.getValue().sext(arithmeticWidth);
  if (!start.slt(bound))
    return false;
  llvm::APInt distance = bound - start;
  if (!distance.urem(step).isZero())
    return false;
  llvm::APInt last = bound - step;
  return start.isSignedIntN(targetWidth) && last.isSignedIntN(targetWidth);
}

bool provesSignedFit(mlir::Value value, unsigned targetWidth) {
  if (mlir::IntegerAttr constant = integerConstant(value))
    return fitsSignedWidth(constant, targetWidth);

  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
    auto sourceType =
        llvm::dyn_cast<mlir::IntegerType>(extension.getIn().getType());
    if (sourceType && sourceType.getWidth() <= targetWidth)
      return true;
  }

  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
    auto sourceType =
        llvm::dyn_cast<mlir::IntegerType>(extension.getIn().getType());
    if (sourceType &&
        (sourceType.getWidth() < targetWidth ||
         (sourceType.getWidth() == targetWidth && extension.getNonNeg())))
      return true;
  }

  return provesPostTestedInductionDomain(value, targetWidth);
}

llvm::Error materializeIndexLayout(mlir::ModuleOp module, unsigned width) {
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return invalid("requires a representable nonzero fixed width");

  mlir::DataLayoutSpecInterface current = module.getDataLayoutSpec();
  if (current) {
    mlir::DataLayoutEntryList indexEntries =
        current.getSpecForType<mlir::IndexType>();
    if (indexEntries.size() > 1)
      return invalid("found duplicate module index layout entries");
    if (!indexEntries.empty()) {
      auto declared =
          llvm::dyn_cast<mlir::IntegerAttr>(indexEntries.front().getValue());
      if (!declared || declared.getValue().getActiveBits() > 64 ||
          declared.getValue().getZExtValue() != width)
        return invalid("conflicts with the existing module index width");
      return llvm::Error::success();
    }
  }

  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries;
  if (current)
    llvm::append_range(entries, current.getEntries());
  mlir::MLIRContext *context = module.getContext();
  entries.push_back(mlir::DataLayoutEntryAttr::get(
      mlir::IndexType::get(context),
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), width)));
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                  mlir::DataLayoutSpecAttr::get(context, entries));
  return llvm::Error::success();
}

bool containsDynamicAddressIndex(mlir::Operation *operation) {
  bool found = false;
  operation->walk([&](mlir::LLVM::GEPOp gep) {
    for (mlir::Value index : gep.getDynamicIndices()) {
      if (integerConstant(index))
        continue;
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

std::optional<unsigned> explicitFixedIndexWidth(mlir::ModuleOp module) {
  mlir::DataLayoutSpecInterface spec = module.getDataLayoutSpec();
  if (!spec)
    return std::nullopt;
  mlir::DataLayoutEntryList entries = spec.getSpecForType<mlir::IndexType>();
  if (entries.size() != 1)
    return std::nullopt;
  auto declared = llvm::dyn_cast<mlir::IntegerAttr>(entries.front().getValue());
  if (!declared || declared.getValue().isZero() ||
      declared.getValue().getActiveBits() > 64 ||
      declared.getValue().getZExtValue() > mlir::IntegerType::kMaxWidth)
    return std::nullopt;
  return static_cast<unsigned>(declared.getValue().getZExtValue());
}

struct GepIndexUse final {
  mlir::LLVM::GEPOp operation;
  unsigned dynamicOrdinal;
  mlir::Value source;
};

struct PointerInductionLane final {
  unsigned ordinal;
  mlir::Value base;
  mlir::LLVM::GEPOp update;
  mlir::Type accessElementType;
  llvm::APInt stride;
};

struct PointerInductionLoop final {
  mlir::scf::WhileOp operation;
  unsigned iterationStateWidth;
  llvm::SmallVector<PointerInductionLane, 4> lanes;
};

std::optional<unsigned> proveUnitStepTerminationWidth(mlir::scf::WhileOp loop) {
  mlir::Block *before = loop.getBeforeBody();
  mlir::Block *after = loop.getAfterBody();
  mlir::scf::ConditionOp condition = loop.getConditionOp();
  mlir::scf::YieldOp yield = loop.getYieldOp();
  auto compare = condition.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!compare || compare->getParentRegion() != &loop.getBefore())
    return std::nullopt;

  for (unsigned lane = 0; lane < before->getNumArguments(); ++lane) {
    auto integer =
        llvm::dyn_cast<mlir::IntegerType>(before->getArgument(lane).getType());
    if (!integer || lane >= loop.getInits().size() ||
        lane >= condition.getArgs().size() ||
        lane >= after->getNumArguments() || lane >= yield.getResults().size() ||
        yield.getResults()[lane] != after->getArgument(lane))
      continue;

    auto update =
        condition.getArgs()[lane].getDefiningOp<mlir::arith::AddIOp>();
    if (!update || update->getParentRegion() != &loop.getBefore())
      continue;
    mlir::Value deltaValue;
    if (update.getLhs() == before->getArgument(lane))
      deltaValue = update.getRhs();
    else if (update.getRhs() == before->getArgument(lane))
      deltaValue = update.getLhs();
    else
      continue;
    mlir::IntegerAttr delta = integerConstant(deltaValue);
    if (!delta || delta.getType() != integer ||
        (!delta.getValue().isOne() && !delta.getValue().isAllOnes()))
      continue;

    mlir::Value comparisonValue;
    if (compare.getLhs() == update.getResult())
      comparisonValue = compare.getRhs();
    else if (compare.getRhs() == update.getResult())
      comparisonValue = compare.getLhs();
    if (compare.getPredicate() == mlir::arith::CmpIPredicate::ne &&
        comparisonValue) {
      mlir::IntegerAttr zero = integerConstant(comparisonValue);
      if (zero && zero.getType() == integer && zero.getValue().isZero())
        return integer.getWidth();
    }

    if (!delta.getValue().isAllOnes())
      continue;
    mlir::Value boundValue;
    bool countDown = false;
    switch (compare.getPredicate()) {
    case mlir::arith::CmpIPredicate::sgt:
    case mlir::arith::CmpIPredicate::ugt:
      countDown = compare.getLhs() == before->getArgument(lane);
      boundValue = compare.getRhs();
      break;
    case mlir::arith::CmpIPredicate::slt:
    case mlir::arith::CmpIPredicate::ult:
      countDown = compare.getRhs() == before->getArgument(lane);
      boundValue = compare.getLhs();
      break;
    default:
      break;
    }
    mlir::IntegerAttr one =
        boundValue ? integerConstant(boundValue) : mlir::IntegerAttr{};
    if (countDown && one && one.getType() == integer && one.getValue().isOne())
      return integer.getWidth();
  }
  return std::nullopt;
}

std::optional<llvm::APInt> positiveConstantGepIndex(mlir::LLVM::GEPOp gep) {
  auto indices = gep.getIndices();
  if (indices.size() != 1)
    return std::nullopt;
  auto index = indices[0];
  mlir::IntegerAttr constant =
      llvm::dyn_cast_if_present<mlir::IntegerAttr>(index);
  if (!constant)
    if (mlir::Value value = llvm::dyn_cast_if_present<mlir::Value>(index))
      constant = integerConstant(value);
  if (!constant || !constant.getValue().isStrictlyPositive())
    return std::nullopt;
  return constant.getValue();
}

std::optional<uint64_t> fixedByteSize(mlir::Operation *scope, mlir::Type type) {
  mlir::DataLayout layout = mlir::DataLayout::closest(scope);
  llvm::TypeSize size = layout.getTypeSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return std::nullopt;
  return size.getFixedValue();
}

std::optional<mlir::Type>
pointerLaneAccessElementType(mlir::BlockArgument pointer,
                             mlir::LLVM::GEPOp update) {
  mlir::Type accessType;
  for (mlir::OpOperand &use : pointer.getUses()) {
    if (use.getOwner() == update.getOperation())
      continue;
    mlir::Type current;
    if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(use.getOwner())) {
      if (load.getAddr() != pointer)
        return std::nullopt;
      current = load.getResult().getType();
    } else if (auto store =
                   llvm::dyn_cast<mlir::LLVM::StoreOp>(use.getOwner())) {
      if (store.getAddr() != pointer)
        return std::nullopt;
      current = store.getValue().getType();
    } else {
      return std::nullopt;
    }
    if (!accessType)
      accessType = current;
    else if (accessType != current)
      return std::nullopt;
  }
  return accessType ? std::optional<mlir::Type>(accessType) : std::nullopt;
}

std::optional<llvm::APInt> elementStride(mlir::LLVM::GEPOp update,
                                         const llvm::APInt &rawStride,
                                         mlir::Type accessElementType) {
  if (rawStride.getActiveBits() > 64)
    return std::nullopt;
  std::optional<uint64_t> gepBytes =
      fixedByteSize(update, update.getElemType());
  std::optional<uint64_t> accessBytes =
      fixedByteSize(update, accessElementType);
  if (!gepBytes || !accessBytes ||
      rawStride.getZExtValue() >
          std::numeric_limits<uint64_t>::max() / *gepBytes)
    return std::nullopt;
  uint64_t byteStride = rawStride.getZExtValue() * *gepBytes;
  if (byteStride == 0 || byteStride % *accessBytes != 0)
    return std::nullopt;
  uint64_t elements = byteStride / *accessBytes;
  return llvm::APInt(64, elements);
}

std::optional<PointerInductionLoop>
analyzePointerInductionLoop(mlir::scf::WhileOp loop) {
  std::optional<unsigned> iterationWidth = proveUnitStepTerminationWidth(loop);
  if (!iterationWidth)
    return std::nullopt;

  mlir::Block *before = loop.getBeforeBody();
  mlir::Block *after = loop.getAfterBody();
  mlir::scf::ConditionOp condition = loop.getConditionOp();
  mlir::scf::YieldOp yield = loop.getYieldOp();
  PointerInductionLoop result{loop, *iterationWidth, {}};
  unsigned pointerLanes = 0;
  for (unsigned lane = 0; lane < before->getNumArguments(); ++lane) {
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(
            before->getArgument(lane).getType()))
      continue;
    ++pointerLanes;
    if (lane >= loop.getInits().size() || lane >= condition.getArgs().size() ||
        lane >= after->getNumArguments() || lane >= yield.getResults().size() ||
        loop.getInits()[lane].getType() !=
            before->getArgument(lane).getType() ||
        condition.getArgs()[lane].getType() !=
            before->getArgument(lane).getType() ||
        after->getArgument(lane).getType() !=
            before->getArgument(lane).getType() ||
        yield.getResults()[lane] != after->getArgument(lane))
      return std::nullopt;

    auto update = condition.getArgs()[lane].getDefiningOp<mlir::LLVM::GEPOp>();
    if (!update || update->getParentRegion() != &loop.getBefore() ||
        update.getBase() != before->getArgument(lane) ||
        !update.getResult().hasOneUse() ||
        *update.getResult().getUsers().begin() != condition.getOperation())
      return std::nullopt;
    std::optional<llvm::APInt> rawStride = positiveConstantGepIndex(update);
    std::optional<mlir::Type> accessElementType =
        pointerLaneAccessElementType(before->getArgument(lane), update);
    if (!rawStride || !accessElementType)
      return std::nullopt;
    std::optional<llvm::APInt> stride =
        elementStride(update, *rawStride, *accessElementType);
    if (!stride)
      return std::nullopt;
    result.lanes.push_back(PointerInductionLane{lane, loop.getInits()[lane],
                                                update, *accessElementType,
                                                std::move(*stride)});
  }
  if (pointerLanes == 0 || result.lanes.size() != pointerLanes)
    return std::nullopt;
  return result;
}

llvm::SmallVector<PointerInductionLoop, 4>
collectPointerInductionLoops(mlir::Operation *operation) {
  llvm::SmallVector<PointerInductionLoop, 4> loops;
  operation->walk([&](mlir::scf::WhileOp loop) {
    if (std::optional<PointerInductionLoop> plan =
            analyzePointerInductionLoop(loop))
      loops.push_back(std::move(*plan));
  });
  return loops;
}

bool pointerOffsetsFit(const PointerInductionLoop &loop, unsigned width) {
  for (const PointerInductionLane &lane : loop.lanes) {
    const uint64_t required = static_cast<uint64_t>(loop.iterationStateWidth) +
                              lane.stride.getActiveBits() + 1;
    if (required > width)
      return false;
  }
  return true;
}

const PointerInductionLane *findPointerLane(const PointerInductionLoop &loop,
                                            unsigned ordinal) {
  auto found = llvm::find_if(loop.lanes, [&](const PointerInductionLane &lane) {
    return lane.ordinal == ordinal;
  });
  return found == loop.lanes.end() ? nullptr : &*found;
}

mlir::Value buildDerivedPointer(mlir::OpBuilder &builder,
                                const PointerInductionLane &lane,
                                mlir::Value offset, mlir::Location location) {
  mlir::LLVM::GEPOp update = lane.update;
  auto derived = mlir::LLVM::GEPOp::create(
      builder, location, update.getResult().getType(), lane.accessElementType,
      lane.base, mlir::ValueRange{offset}, update.getNoWrapFlags());
  derived->setDiscardableAttrs(update->getDiscardableAttrDictionary());
  return derived.getResult();
}

mlir::scf::WhileOp rewritePointerInductionLoop(const PointerInductionLoop &plan,
                                               unsigned width) {
  mlir::scf::WhileOp loop = plan.operation;
  mlir::OpBuilder builder(loop);
  mlir::IntegerType offsetType = builder.getIntegerType(width);
  auto zero =
      mlir::arith::ConstantOp::create(builder, loop.getLoc(), offsetType,
                                      builder.getIntegerAttr(offsetType, 0));

  llvm::SmallVector<mlir::Value, 4> inits(loop.getInits());
  llvm::SmallVector<mlir::Type, 4> resultTypes(loop.getResultTypes());
  for (const PointerInductionLane &lane : plan.lanes) {
    inits[lane.ordinal] = zero;
    resultTypes[lane.ordinal] = offsetType;
  }

  auto buildBefore = [&](mlir::OpBuilder &bodyBuilder, mlir::Location location,
                         mlir::ValueRange arguments) {
    mlir::IRMapping mapping;
    for (unsigned ordinal = 0; ordinal < arguments.size(); ++ordinal) {
      if (const PointerInductionLane *lane = findPointerLane(plan, ordinal)) {
        mapping.map(loop.getBeforeBody()->getArgument(ordinal),
                    buildDerivedPointer(bodyBuilder, *lane, arguments[ordinal],
                                        location));
      } else {
        mapping.map(loop.getBeforeBody()->getArgument(ordinal),
                    arguments[ordinal]);
      }
    }

    llvm::SmallPtrSet<mlir::Operation *, 4> skippedUpdates;
    for (const PointerInductionLane &lane : plan.lanes) {
      mlir::LLVM::GEPOp update = lane.update;
      skippedUpdates.insert(update.getOperation());
    }
    for (mlir::Operation &operation :
         loop.getBeforeBody()->without_terminator())
      if (!skippedUpdates.contains(&operation))
        bodyBuilder.clone(operation, mapping);

    llvm::SmallVector<mlir::Value, 4> nextOffsets(arguments.size());
    for (const PointerInductionLane &lane : plan.lanes) {
      mlir::LLVM::GEPOp update = lane.update;
      llvm::APInt strideBits = lane.stride.zextOrTrunc(width);
      auto stride = mlir::arith::ConstantOp::create(
          bodyBuilder, update.getLoc(), offsetType,
          mlir::IntegerAttr::get(offsetType, strideBits));
      nextOffsets[lane.ordinal] = mlir::arith::AddIOp::create(
          bodyBuilder, update.getLoc(), arguments[lane.ordinal], stride);
    }

    llvm::SmallVector<mlir::Value, 4> nextArguments;
    nextArguments.reserve(loop.getConditionOp().getArgs().size());
    for (auto [ordinal, value] :
         llvm::enumerate(loop.getConditionOp().getArgs())) {
      if (nextOffsets[ordinal])
        nextArguments.push_back(nextOffsets[ordinal]);
      else
        nextArguments.push_back(mapping.lookupOrDefault(value));
    }
    mlir::scf::ConditionOp::create(
        bodyBuilder, loop.getConditionOp().getLoc(),
        mapping.lookupOrDefault(loop.getConditionOp().getCondition()),
        nextArguments);
  };

  auto buildAfter = [&](mlir::OpBuilder &bodyBuilder, mlir::Location location,
                        mlir::ValueRange arguments) {
    mlir::IRMapping mapping;
    for (unsigned ordinal = 0; ordinal < arguments.size(); ++ordinal) {
      mlir::BlockArgument oldArgument =
          loop.getAfterBody()->getArgument(ordinal);
      if (const PointerInductionLane *lane = findPointerLane(plan, ordinal)) {
        bool hasPayloadUse =
            llvm::any_of(oldArgument.getUsers(), [&](mlir::Operation *user) {
              return user != loop.getYieldOp().getOperation();
            });
        if (hasPayloadUse)
          mapping.map(oldArgument,
                      buildDerivedPointer(bodyBuilder, *lane,
                                          arguments[ordinal], location));
      } else {
        mapping.map(oldArgument, arguments[ordinal]);
      }
    }
    for (mlir::Operation &operation : loop.getAfterBody()->without_terminator())
      bodyBuilder.clone(operation, mapping);

    llvm::SmallVector<mlir::Value, 4> yields;
    yields.reserve(loop.getYieldOp().getResults().size());
    for (auto [ordinal, value] :
         llvm::enumerate(loop.getYieldOp().getResults())) {
      if (findPointerLane(plan, ordinal))
        yields.push_back(arguments[ordinal]);
      else
        yields.push_back(mapping.lookupOrDefault(value));
    }
    mlir::scf::YieldOp::create(bodyBuilder, loop.getYieldOp().getLoc(), yields);
  };

  auto replacement = mlir::scf::WhileOp::create(
      builder, loop.getLoc(), resultTypes, inits, buildBefore, buildAfter);
  replacement->setDiscardableAttrs(loop->getDiscardableAttrDictionary());

  builder.setInsertionPointAfter(replacement);
  for (unsigned ordinal = 0; ordinal < loop.getNumResults(); ++ordinal) {
    if (const PointerInductionLane *lane = findPointerLane(plan, ordinal)) {
      if (!loop.getResult(ordinal).use_empty()) {
        mlir::Value pointer = buildDerivedPointer(
            builder, *lane, replacement.getResult(ordinal), loop.getLoc());
        loop.getResult(ordinal).replaceAllUsesWith(pointer);
      }
    } else {
      loop.getResult(ordinal).replaceAllUsesWith(
          replacement.getResult(ordinal));
    }
  }
  loop.erase();
  return replacement;
}

} // namespace

bool requiresCanonicalAddressIndexDecision(mlir::ModuleOp module,
                                           mlir::Operation *selectedOperation) {
  if (!selectedOperation || explicitFixedIndexWidth(module))
    return false;
  auto pointerLoops = collectPointerInductionLoops(selectedOperation);
  return containsDynamicAddressIndex(selectedOperation) ||
         !pointerLoops.empty();
}

llvm::Expected<mlir::Operation *>
materializeAddressIndexContract(mlir::ModuleOp module,
                                mlir::Operation *selectedOperation,
                                std::optional<unsigned> canonicalIndexWidth) {
  if (!selectedOperation)
    return invalid("requires a selected structured operation");
  std::optional<unsigned> effectiveWidth = canonicalIndexWidth;
  if (!canonicalIndexWidth) {
    if (requiresCanonicalAddressIndexDecision(module, selectedOperation))
      return invalid("requires an explicit canonical index width for LLVM "
                     "GEP operands");
    effectiveWidth = explicitFixedIndexWidth(module);
    if (!effectiveWidth)
      return selectedOperation;
  }
  if (*effectiveWidth == 0 || *effectiveWidth > mlir::IntegerType::kMaxWidth)
    return invalid("requires a representable nonzero fixed width");

  llvm::SmallVector<PointerInductionLoop, 4> pointerLoops =
      collectPointerInductionLoops(selectedOperation);
  for (const PointerInductionLoop &loop : pointerLoops)
    if (!pointerOffsetsFit(loop, *effectiveWidth))
      return invalid("cannot prove a pointer induction offset fits the "
                     "selected signed width");

  llvm::SmallVector<GepIndexUse> uses;
  llvm::SmallVector<mlir::Value> sources;
  llvm::SmallPtrSet<mlir::Value, 8> seenSources;
  std::string proofFailure;
  selectedOperation->walk([&](mlir::LLVM::GEPOp gep) {
    for (auto [ordinal, index] : llvm::enumerate(gep.getDynamicIndices())) {
      auto integer = llvm::dyn_cast<mlir::IntegerType>(index.getType());
      if (!integer) {
        proofFailure = "cannot prove a non-scalar GEP index narrowing";
        return mlir::WalkResult::interrupt();
      }
      if (integer.getWidth() <= *effectiveWidth)
        continue;
      if (!provesSignedFit(index, *effectiveWidth)) {
        proofFailure =
            "cannot prove a wide GEP index fits the selected signed width";
        return mlir::WalkResult::interrupt();
      }
      uses.push_back(GepIndexUse{gep, static_cast<unsigned>(ordinal), index});
      if (seenSources.insert(index).second)
        sources.push_back(index);
    }
    return mlir::WalkResult::advance();
  });
  if (!proofFailure.empty())
    return invalid(proofFailure);
  if (llvm::Error error = materializeIndexLayout(module, *effectiveWidth))
    return error;

  mlir::OpBuilder builder(module.getContext());
  mlir::IntegerType narrowedType =
      mlir::IntegerType::get(module.getContext(), *effectiveWidth);
  llvm::DenseMap<mlir::Value, mlir::Value> narrowed;
  for (mlir::Value source : sources) {
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(source))
      builder.setInsertionPointToStart(argument.getOwner());
    else
      builder.setInsertionPointAfter(source.getDefiningOp());
    if (mlir::IntegerAttr constant = integerConstant(source)) {
      llvm::APInt bits = constant.getValue().trunc(*effectiveWidth);
      auto attr = mlir::IntegerAttr::get(narrowedType, bits);
      auto narrowedConstant = mlir::arith::ConstantOp::create(
          builder, source.getLoc(), narrowedType, attr);
      narrowed.try_emplace(source, narrowedConstant.getResult());
      continue;
    }
    auto trunc = mlir::arith::TruncIOp::create(builder, source.getLoc(),
                                               narrowedType, source);
    trunc.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw);
    narrowed.try_emplace(source, trunc.getResult());
  }

  for (GepIndexUse &use : uses) {
    llvm::SmallVector<mlir::Value> indices(use.operation.getDynamicIndices());
    indices[use.dynamicOrdinal] = narrowed.lookup(use.source);
    use.operation.getDynamicIndicesMutable().assign(indices);
  }

  for (PointerInductionLoop &loop : llvm::reverse(pointerLoops)) {
    const bool replacesSelection =
        selectedOperation == loop.operation.getOperation();
    mlir::scf::WhileOp replacement =
        rewritePointerInductionLoop(loop, *effectiveWidth);
    if (replacesSelection)
      selectedOperation = replacement.getOperation();
  }
  if (mlir::failed(mlir::verify(module)))
    return invalid("produced an invalid pointer-induction normalization");
  return selectedOperation;
}

} // namespace loom::frontend::detail
