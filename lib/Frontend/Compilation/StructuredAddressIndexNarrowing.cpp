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
#include "llvm/IR/DataLayout.h"

#include <cstdint>
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

  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    auto sourceType = llvm::dyn_cast<mlir::IntegerType>(cast.getIn().getType());
    if (sourceType && sourceType.getWidth() <= targetWidth)
      return true;
    return provesSignedFit(cast.getIn(), targetWidth);
  }

  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
    auto sourceType = llvm::dyn_cast<mlir::IntegerType>(cast.getIn().getType());
    if (sourceType && sourceType.getWidth() < targetWidth)
      return true;
    return provesSignedFit(cast.getIn(), targetWidth);
  }

  return provesPostTestedInductionDomain(value, targetWidth);
}

struct ThreadDomainSignedRange final {
  __int128 minimum = 0;
  __int128 maximum = 0;
};

std::optional<unsigned> semanticIntegerWidth(mlir::Type type,
                                             unsigned indexWidth) {
  if (llvm::isa<mlir::IndexType>(type))
    return indexWidth;
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  return std::nullopt;
}

std::optional<ThreadDomainSignedRange>
fullThreadDomainSignedRange(unsigned width) {
  if (width == 0 || width > 64)
    return std::nullopt;
  const __int128 limit = static_cast<__int128>(1) << (width - 1);
  return ThreadDomainSignedRange{-limit, limit - 1};
}

std::optional<ThreadDomainSignedRange>
inferThreadDomainSignedRange(mlir::Value value, unsigned indexWidth);

std::optional<ThreadDomainSignedRange>
inferThreadDomainUnsignedExtensionRange(mlir::Value value,
                                        unsigned indexWidth,
                                        unsigned resultWidth) {
  auto sourceWidth = semanticIntegerWidth(value.getType(), indexWidth);
  if (!sourceWidth || *sourceWidth == 0 || *sourceWidth > 64 ||
      resultWidth == 0 || resultWidth > 64)
    return std::nullopt;
  __int128 minimum = 0;
  __int128 maximum = (static_cast<__int128>(1) << *sourceWidth) - 1;
  if (mlir::IntegerAttr constant = integerConstant(value)) {
    minimum = constant.getValue().getZExtValue();
    maximum = minimum;
  }
  auto destination = fullThreadDomainSignedRange(resultWidth);
  if (!destination || maximum > destination->maximum)
    return destination;
  return ThreadDomainSignedRange{minimum, maximum};
}

std::optional<ThreadDomainSignedRange>
projectThreadDomainSignedCast(ThreadDomainSignedRange source,
                              unsigned resultWidth) {
  auto destination = fullThreadDomainSignedRange(resultWidth);
  if (!destination)
    return std::nullopt;
  if (source.minimum < destination->minimum ||
      source.maximum > destination->maximum)
    return destination;
  return source;
}

std::optional<ThreadDomainSignedRange>
inferThreadDomainSignedRange(mlir::Value value, unsigned indexWidth) {
  auto resultWidth = semanticIntegerWidth(value.getType(), indexWidth);
  if (!resultWidth)
    return std::nullopt;
  if (mlir::IntegerAttr constant = integerConstant(value)) {
    if (constant.getValue().getBitWidth() > 64)
      return std::nullopt;
    const __int128 exact = constant.getValue().getSExtValue();
    return ThreadDomainSignedRange{exact, exact};
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    auto source = inferThreadDomainSignedRange(cast.getIn(), indexWidth);
    return source ? projectThreadDomainSignedCast(*source, *resultWidth)
                  : std::nullopt;
  }
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>())
    return inferThreadDomainUnsignedExtensionRange(cast.getIn(), indexWidth,
                                                   *resultWidth);
  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
    auto source =
        inferThreadDomainSignedRange(extension.getIn(), indexWidth);
    return source ? projectThreadDomainSignedCast(*source, *resultWidth)
                  : std::nullopt;
  }
  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>())
    return inferThreadDomainUnsignedExtensionRange(
        extension.getIn(), indexWidth, *resultWidth);
  if (auto maximum = value.getDefiningOp<mlir::arith::MaxSIOp>()) {
    auto lhs = inferThreadDomainSignedRange(maximum.getLhs(), indexWidth);
    auto rhs = inferThreadDomainSignedRange(maximum.getRhs(), indexWidth);
    if (!lhs || !rhs)
      return std::nullopt;
    return ThreadDomainSignedRange{std::max(lhs->minimum, rhs->minimum),
                                   std::max(lhs->maximum, rhs->maximum)};
  }
  if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>()) {
    auto lhs = inferThreadDomainSignedRange(minimum.getLhs(), indexWidth);
    auto rhs = inferThreadDomainSignedRange(minimum.getRhs(), indexWidth);
    if (!lhs || !rhs)
      return std::nullopt;
    return ThreadDomainSignedRange{std::min(lhs->minimum, rhs->minimum),
                                   std::min(lhs->maximum, rhs->maximum)};
  }
  if (auto select = value.getDefiningOp<mlir::arith::SelectOp>()) {
    auto trueRange =
        inferThreadDomainSignedRange(select.getTrueValue(), indexWidth);
    auto falseRange =
        inferThreadDomainSignedRange(select.getFalseValue(), indexWidth);
    if (!trueRange || !falseRange)
      return std::nullopt;
    return ThreadDomainSignedRange{
        std::min(trueRange->minimum, falseRange->minimum),
        std::max(trueRange->maximum, falseRange->maximum)};
  }
  return fullThreadDomainSignedRange(*resultWidth);
}

std::optional<ThreadDomainSignedRange>
inferThreadDomainSignedRange(mlir::OpFoldResult value, unsigned indexWidth) {
  if (auto dynamic = llvm::dyn_cast<mlir::Value>(value))
    return inferThreadDomainSignedRange(dynamic, indexWidth);
  auto integer = llvm::dyn_cast<mlir::IntegerAttr>(
      llvm::cast<mlir::Attribute>(value));
  if (!integer || integer.getValue().getBitWidth() > 64)
    return std::nullopt;
  const __int128 exact = integer.getValue().getSExtValue();
  return ThreadDomainSignedRange{exact, exact};
}

llvm::Error
materializeDataLayoutEndiannessProjectionImpl(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  mlir::StringAttr endiannessKey = mlir::StringAttr::get(
      context, mlir::DLTIDialect::kDataLayoutEndiannessKey);

  mlir::DataLayoutSpecInterface current = module.getDataLayoutSpec();
  bool hasEndianness = false;
  mlir::StringAttr expectedEndianness;
  if (current) {
    if (mlir::DataLayoutEntryInterface entry =
            current.getSpecForIdentifier(endiannessKey)) {
      auto declared = llvm::dyn_cast<mlir::StringAttr>(entry.getValue());
      if (!declared ||
          (declared.getValue() !=
               mlir::DLTIDialect::kDataLayoutEndiannessLittle &&
           declared.getValue() != mlir::DLTIDialect::kDataLayoutEndiannessBig))
        return invalid("has an unsupported explicit DLTI endianness");
      expectedEndianness = declared;
      hasEndianness = true;
    }
  }

  auto llvmLayout = module->getAttrOfType<mlir::StringAttr>("llvm.data_layout");
  if (!llvmLayout || llvmLayout.getValue().empty())
    return invalid("requires a nonempty LLVM DataLayout");
  auto parsedLayout = llvm::DataLayout::parse(llvmLayout.getValue());
  if (!parsedLayout)
    return invalid("cannot parse the LLVM DataLayout: " +
                   llvm::toString(parsedLayout.takeError()));
  mlir::StringAttr projectedEndianness = mlir::StringAttr::get(
      context, parsedLayout->isLittleEndian()
                   ? mlir::DLTIDialect::kDataLayoutEndiannessLittle
                   : mlir::DLTIDialect::kDataLayoutEndiannessBig);
  if (hasEndianness && expectedEndianness != projectedEndianness)
    return invalid("conflicts with the LLVM DataLayout endianness");
  expectedEndianness = projectedEndianness;

  if (hasEndianness)
    return llvm::Error::success();

  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries;
  if (current)
    llvm::append_range(entries, current.getEntries());
  entries.push_back(
      mlir::DataLayoutEntryAttr::get(endiannessKey, expectedEndianness));
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
                  mlir::DataLayoutSpecAttr::get(context, entries));
  return llvm::Error::success();
}

llvm::Error materializeIndexLayout(mlir::ModuleOp module, unsigned width) {
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return invalid("requires a representable nonzero fixed width");
  if (llvm::Error error = materializeDataLayoutEndiannessProjectionImpl(module))
    return error;

  mlir::MLIRContext *context = module.getContext();
  mlir::DataLayoutSpecInterface current = module.getDataLayoutSpec();
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

  llvm::SmallVector<mlir::DataLayoutEntryInterface> entries;
  llvm::append_range(entries, current.getEntries());
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

struct PointerInductionStride final {
  std::optional<llvm::APInt> constantElements;
  mlir::Value invariantIndex;
  uint64_t elementScale = 1;
};

struct PointerInductionLane final {
  unsigned ordinal;
  mlir::Value base;
  mlir::LLVM::GEPOp update;
  mlir::Type accessElementType;
  PointerInductionStride stride;
};

struct PointerInductionLoop final {
  mlir::scf::WhileOp operation;
  unsigned iterationStateWidth;
  llvm::SmallVector<PointerInductionLane, 4> lanes;
};

bool isDefinedOutsideLoop(mlir::Value value, mlir::scf::WhileOp loop) {
  if (mlir::Operation *definition = value.getDefiningOp())
    return !loop->isAncestor(definition);
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  mlir::Operation *owner =
      argument ? argument.getOwner()->getParentOp() : nullptr;
  return !owner || (owner != loop.getOperation() && !loop->isAncestor(owner));
}

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
        comparisonValue && isDefinedOutsideLoop(comparisonValue, loop))
      return integer.getWidth();

    if (delta.getValue().isOne() && comparisonValue &&
        isDefinedOutsideLoop(comparisonValue, loop)) {
      const bool updateOnLeft = compare.getLhs() == update.getResult();
      const bool updateOnRight = compare.getRhs() == update.getResult();
      const auto flags = update.getOverflowFlags();
      switch (compare.getPredicate()) {
      case mlir::arith::CmpIPredicate::slt:
        if (updateOnLeft && mlir::arith::bitEnumContainsAny(
                                flags, mlir::arith::IntegerOverflowFlags::nsw))
          return integer.getWidth();
        break;
      case mlir::arith::CmpIPredicate::sgt:
        if (updateOnRight && mlir::arith::bitEnumContainsAny(
                                 flags, mlir::arith::IntegerOverflowFlags::nsw))
          return integer.getWidth();
        break;
      case mlir::arith::CmpIPredicate::ult:
        if (updateOnLeft && mlir::arith::bitEnumContainsAny(
                                flags, mlir::arith::IntegerOverflowFlags::nuw))
          return integer.getWidth();
        break;
      case mlir::arith::CmpIPredicate::ugt:
        if (updateOnRight && mlir::arith::bitEnumContainsAny(
                                 flags, mlir::arith::IntegerOverflowFlags::nuw))
          return integer.getWidth();
        break;
      default:
        break;
      }
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

std::optional<llvm::APInt> constantGepIndex(mlir::LLVM::GEPOp gep) {
  auto indices = gep.getIndices();
  if (indices.size() != 1)
    return std::nullopt;
  auto index = indices[0];
  mlir::IntegerAttr constant =
      llvm::dyn_cast_if_present<mlir::IntegerAttr>(index);
  if (!constant)
    if (mlir::Value value = llvm::dyn_cast_if_present<mlir::Value>(index))
      constant = integerConstant(value);
  return constant ? std::optional<llvm::APInt>(constant.getValue())
                  : std::nullopt;
}

mlir::Value invariantGepIndex(mlir::LLVM::GEPOp gep, mlir::scf::WhileOp loop) {
  auto indices = gep.getIndices();
  if (indices.size() != 1)
    return {};
  mlir::Value value = llvm::dyn_cast_if_present<mlir::Value>(indices[0]);
  if (!value || integerConstant(value) ||
      !llvm::isa<mlir::IntegerType>(value.getType()))
    return {};
  return isDefinedOutsideLoop(value, loop) ? value : mlir::Value{};
}

std::optional<uint64_t> fixedByteSize(mlir::Operation *scope, mlir::Type type) {
  mlir::DataLayout layout = mlir::DataLayout::closest(scope);
  llvm::TypeSize size = layout.getTypeSize(type);
  if (size.isScalable() || size.getFixedValue() == 0)
    return std::nullopt;
  return size.getFixedValue();
}

bool collectPointerAccessElementType(mlir::Value pointer,
                                     mlir::Operation *ignored,
                                     mlir::Region *region,
                                     mlir::Type &accessType,
                                     bool &foundAccess) {
  for (mlir::OpOperand &use : pointer.getUses()) {
    if (use.getOwner() == ignored)
      continue;
    if (!region->isAncestor(use.getOwner()->getParentRegion()))
      return false;
    mlir::Type current;
    if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(use.getOwner())) {
      if (load.getAddr() != pointer)
        return false;
      current = load.getResult().getType();
    } else if (auto store =
                   llvm::dyn_cast<mlir::LLVM::StoreOp>(use.getOwner())) {
      if (store.getAddr() != pointer)
        return false;
      current = store.getValue().getType();
    } else if (auto gep = llvm::dyn_cast<mlir::LLVM::GEPOp>(use.getOwner())) {
      if (gep.getBase() != pointer ||
          !collectPointerAccessElementType(gep.getResult(), ignored, region,
                                           accessType, foundAccess))
        return false;
      continue;
    } else {
      return false;
    }
    if (!accessType)
      accessType = current;
    else if (accessType != current)
      return false;
    foundAccess = true;
  }
  return true;
}

std::optional<mlir::Type>
pointerLaneAccessElementType(mlir::BlockArgument pointer,
                             mlir::LLVM::GEPOp update,
                             mlir::scf::ConditionOp condition) {
  mlir::Type accessType;
  bool foundAccess = false;
  mlir::Region *region = pointer.getOwner()->getParent();
  if (!collectPointerAccessElementType(pointer, update.getOperation(), region,
                                       accessType, foundAccess) ||
      !collectPointerAccessElementType(update.getResult(),
                                       condition.getOperation(), region,
                                       accessType, foundAccess) ||
      !foundAccess)
    return std::nullopt;
  return accessType;
}

std::optional<llvm::APInt> constantElementStride(mlir::LLVM::GEPOp update,
                                                 const llvm::APInt &rawStride,
                                                 mlir::Type accessElementType) {
  std::optional<uint64_t> gepBytes =
      fixedByteSize(update, update.getElemType());
  std::optional<uint64_t> accessBytes =
      fixedByteSize(update, accessElementType);
  if (!gepBytes || !accessBytes)
    return std::nullopt;
  const unsigned width = rawStride.getBitWidth() + 65;
  llvm::APInt byteStride =
      rawStride.sext(width) * llvm::APInt(width, *gepBytes);
  llvm::APInt divisor(width, *accessBytes);
  if (!byteStride.srem(divisor).isZero())
    return std::nullopt;
  return byteStride.sdiv(divisor);
}

std::optional<uint64_t> dynamicElementScale(mlir::LLVM::GEPOp update,
                                            mlir::Type accessElementType) {
  std::optional<uint64_t> gepBytes =
      fixedByteSize(update, update.getElemType());
  std::optional<uint64_t> accessBytes =
      fixedByteSize(update, accessElementType);
  if (!gepBytes || !accessBytes || *gepBytes % *accessBytes != 0)
    return std::nullopt;
  return *gepBytes / *accessBytes;
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
        update.getBase() != before->getArgument(lane))
      return std::nullopt;
    unsigned feedbackUses = 0;
    for (mlir::OpOperand &use : update.getResult().getUses())
      if (use.getOwner() == condition.getOperation())
        ++feedbackUses;
    if (feedbackUses != 1)
      return std::nullopt;
    std::optional<mlir::Type> accessElementType = pointerLaneAccessElementType(
        before->getArgument(lane), update, condition);
    if (!accessElementType)
      return std::nullopt;

    PointerInductionStride stride;
    if (std::optional<llvm::APInt> rawStride = constantGepIndex(update)) {
      stride.constantElements =
          constantElementStride(update, *rawStride, *accessElementType);
      if (!stride.constantElements)
        return std::nullopt;
    } else {
      stride.invariantIndex = invariantGepIndex(update, loop);
      std::optional<uint64_t> scale =
          dynamicElementScale(update, *accessElementType);
      if (!stride.invariantIndex || !scale)
        return std::nullopt;
      stride.elementScale = *scale;
    }
    result.lanes.push_back(PointerInductionLane{lane, loop.getInits()[lane],
                                                update, *accessElementType,
                                                std::move(stride)});
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

struct SignedRange final {
  llvm::APInt minimum;
  llvm::APInt maximum;
};

std::optional<SignedRange> signedIndexRange(mlir::Value value) {
  auto type = llvm::dyn_cast<mlir::IntegerType>(value.getType());
  if (!type)
    return std::nullopt;
  const unsigned width = type.getWidth();
  if (mlir::IntegerAttr constant = integerConstant(value))
    return SignedRange{constant.getValue(), constant.getValue()};
  if (auto extension = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
    auto source =
        llvm::dyn_cast<mlir::IntegerType>(extension.getIn().getType());
    if (!source)
      return std::nullopt;
    return SignedRange{
        llvm::APInt::getSignedMinValue(source.getWidth()).sext(width),
        llvm::APInt::getSignedMaxValue(source.getWidth()).sext(width)};
  }
  if (auto extension = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
    auto source =
        llvm::dyn_cast<mlir::IntegerType>(extension.getIn().getType());
    if (!source)
      return std::nullopt;
    return SignedRange{llvm::APInt(width, 0),
                       llvm::APInt::getMaxValue(source.getWidth()).zext(width)};
  }
  return SignedRange{llvm::APInt::getSignedMinValue(width),
                     llvm::APInt::getSignedMaxValue(width)};
}

bool scaledAccumulationFits(const SignedRange &range, uint64_t scale,
                            unsigned iterationWidth, unsigned width) {
  if (iterationWidth >= width || scale == 0)
    return false;
  llvm::APInt scaleBits(width, scale);
  if (scaleBits.isNegative())
    return false;
  llvm::APInt maxIterations = llvm::APInt::getLowBitsSet(width, iterationWidth);
  for (const llvm::APInt *endpoint : {&range.minimum, &range.maximum}) {
    if (!endpoint->isSignedIntN(width))
      return false;
    bool overflow = false;
    llvm::APInt scaled =
        endpoint->sextOrTrunc(width).smul_ov(scaleBits, overflow);
    if (overflow)
      return false;
    (void)scaled.smul_ov(maxIterations, overflow);
    if (overflow)
      return false;
  }
  return true;
}

bool pointerOffsetsFit(const PointerInductionLoop &loop, unsigned width) {
  for (const PointerInductionLane &lane : loop.lanes) {
    SignedRange range{llvm::APInt(1, 0), llvm::APInt(1, 0)};
    uint64_t scale = lane.stride.elementScale;
    if (lane.stride.constantElements) {
      range = SignedRange{*lane.stride.constantElements,
                          *lane.stride.constantElements};
      scale = 1;
    } else {
      std::optional<SignedRange> dynamic =
          signedIndexRange(lane.stride.invariantIndex);
      if (!dynamic)
        return false;
      range = std::move(*dynamic);
    }
    if (!scaledAccumulationFits(range, scale, loop.iterationStateWidth, width))
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

mlir::Value materializeElementStride(mlir::OpBuilder &builder,
                                     const PointerInductionLane &lane,
                                     mlir::IntegerType offsetType,
                                     mlir::Location location) {
  if (lane.stride.constantElements) {
    llvm::APInt bits =
        lane.stride.constantElements->sextOrTrunc(offsetType.getWidth());
    return mlir::arith::ConstantOp::create(
        builder, location, offsetType,
        mlir::IntegerAttr::get(offsetType, bits));
  }

  mlir::Value stride = lane.stride.invariantIndex;
  auto sourceType = llvm::cast<mlir::IntegerType>(stride.getType());
  if (sourceType.getWidth() < offsetType.getWidth())
    stride =
        mlir::arith::ExtSIOp::create(builder, location, offsetType, stride);
  else if (sourceType.getWidth() > offsetType.getWidth()) {
    auto trunc =
        mlir::arith::TruncIOp::create(builder, location, offsetType, stride);
    trunc.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw);
    stride = trunc;
  }
  if (lane.stride.elementScale == 1)
    return stride;
  auto scale = mlir::arith::ConstantOp::create(
      builder, location, offsetType,
      builder.getIntegerAttr(offsetType, lane.stride.elementScale));
  return mlir::arith::MulIOp::create(builder, location, stride, scale);
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

    llvm::SmallVector<mlir::Value, 4> nextOffsets(arguments.size());
    llvm::SmallPtrSet<mlir::Operation *, 4> skippedUpdates;
    for (const PointerInductionLane &lane : plan.lanes) {
      mlir::LLVM::GEPOp update = lane.update;
      skippedUpdates.insert(update.getOperation());
      mlir::Value stride = materializeElementStride(
          bodyBuilder, lane, offsetType, update.getLoc());
      nextOffsets[lane.ordinal] = mlir::arith::AddIOp::create(
          bodyBuilder, update.getLoc(), arguments[lane.ordinal], stride);
      mapping.map(update.getResult(),
                  buildDerivedPointer(bodyBuilder, lane,
                                      nextOffsets[lane.ordinal],
                                      update.getLoc()));
    }
    for (mlir::Operation &operation :
         loop.getBeforeBody()->without_terminator())
      if (!skippedUpdates.contains(&operation))
        bodyBuilder.clone(operation, mapping);

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

llvm::Error materializeDataLayoutEndiannessProjection(mlir::ModuleOp module) {
  if (!module)
    return invalid("requires a Structured Program module");
  return materializeDataLayoutEndiannessProjectionImpl(module);
}

bool provesThreadDomainExtentFits(mlir::OpFoldResult lower,
                                  mlir::OpFoldResult upper,
                                  mlir::OpFoldResult step,
                                  unsigned targetWidth) {
  auto lowerRange = inferThreadDomainSignedRange(lower, targetWidth);
  auto upperRange = inferThreadDomainSignedRange(upper, targetWidth);
  auto stepRange = inferThreadDomainSignedRange(step, targetWidth);
  auto targetRange = fullThreadDomainSignedRange(targetWidth);
  if (!lowerRange || !upperRange || !stepRange || !targetRange ||
      stepRange->minimum <= 0 ||
      lowerRange->minimum < targetRange->minimum ||
      lowerRange->maximum > targetRange->maximum ||
      upperRange->minimum < targetRange->minimum ||
      upperRange->maximum > targetRange->maximum ||
      stepRange->maximum > targetRange->maximum)
    return false;
  const __int128 maximumDistance = upperRange->maximum - lowerRange->minimum;
  const __int128 maximumExtent =
      maximumDistance <= 0
          ? 0
          : (maximumDistance + stepRange->minimum - 1) / stepRange->minimum;
  return maximumExtent <= targetRange->maximum;
}

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

  while (!pointerLoops.empty()) {
    PointerInductionLoop &loop = pointerLoops.front();
    const bool replacesSelection =
        selectedOperation == loop.operation.getOperation();
    mlir::scf::WhileOp replacement =
        rewritePointerInductionLoop(loop, *effectiveWidth);
    if (replacesSelection)
      selectedOperation = replacement.getOperation();
    pointerLoops = collectPointerInductionLoops(selectedOperation);
    for (const PointerInductionLoop &loop : pointerLoops)
      if (!pointerOffsetsFit(loop, *effectiveWidth))
        return invalid("cannot prove a pointer induction offset fits the "
                       "selected signed width");
  }
  if (mlir::failed(mlir::verify(module)))
    return invalid("produced an invalid pointer-induction normalization");
  return selectedOperation;
}

} // namespace loom::frontend::detail
