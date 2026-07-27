#include "StructuredAddressIndexNarrowing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

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
  if (!compare || compare->getParentRegion() != &loop.getBefore() ||
      compare.getPredicate() != mlir::arith::CmpIPredicate::ne)
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
  operation->walk([&](mlir::LLVM::GEPOp) {
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

bool hasExplicitFixedIndexLayout(mlir::ModuleOp module) {
  mlir::DataLayoutSpecInterface spec = module.getDataLayoutSpec();
  if (!spec)
    return false;
  mlir::DataLayoutEntryList entries = spec.getSpecForType<mlir::IndexType>();
  if (entries.size() != 1)
    return false;
  auto declared = llvm::dyn_cast<mlir::IntegerAttr>(entries.front().getValue());
  return declared && !declared.getValue().isZero() &&
         declared.getValue().getActiveBits() <= 64 &&
         declared.getValue().getZExtValue() <= mlir::IntegerType::kMaxWidth;
}

struct GepIndexUse final {
  mlir::LLVM::GEPOp operation;
  unsigned dynamicOrdinal;
  mlir::Value source;
};

} // namespace

llvm::Error
materializeAddressIndexContract(mlir::ModuleOp module,
                                mlir::Operation *selectedOperation,
                                std::optional<unsigned> canonicalIndexWidth) {
  if (!selectedOperation)
    return invalid("requires a selected structured operation");
  if (!canonicalIndexWidth) {
    if (containsDynamicAddressIndex(selectedOperation) &&
        !hasExplicitFixedIndexLayout(module))
      return invalid("requires an explicit canonical index width for LLVM "
                     "GEP operands");
    return llvm::Error::success();
  }
  if (*canonicalIndexWidth == 0 ||
      *canonicalIndexWidth > mlir::IntegerType::kMaxWidth)
    return invalid("requires a representable nonzero fixed width");

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
      if (integer.getWidth() <= *canonicalIndexWidth)
        continue;
      if (!provesSignedFit(index, *canonicalIndexWidth)) {
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
  if (llvm::Error error = materializeIndexLayout(module, *canonicalIndexWidth))
    return error;

  mlir::OpBuilder builder(module.getContext());
  mlir::Type narrowedType =
      mlir::IntegerType::get(module.getContext(), *canonicalIndexWidth);
  llvm::DenseMap<mlir::Value, mlir::Value> narrowed;
  for (mlir::Value source : sources) {
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(source))
      builder.setInsertionPointToStart(argument.getOwner());
    else
      builder.setInsertionPointAfter(source.getDefiningOp());
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
  return llvm::Error::success();
}

} // namespace loom::frontend::detail
