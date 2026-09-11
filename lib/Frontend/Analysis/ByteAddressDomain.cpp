#include "Frontend/Analysis/ByteAddressDomain.h"
#include "Common/MappingDebugLog.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

namespace loom::frontend::analysis {
namespace {

constexpr std::int64_t signedMaximum = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t signedMinimum = std::numeric_limits<std::int64_t>::min();

/// Index expressions deeper than this contribute no bound of their own.
constexpr unsigned maximumIndexBoundDepth = 32;

std::optional<std::int64_t> signedConstant(mlir::Value value) {
  llvm::APInt literal;
  if (!value || !mlir::matchPattern(value, mlir::m_ConstantInt(&literal)) ||
      !literal.isSignedIntN(64))
    return std::nullopt;
  return literal.getSExtValue();
}

/// Bound of one index value. A known `first` also fixes the phase of `stride`,
/// so a stride above one is reported only together with `first`.
struct IndexValueBounds final {
  std::optional<std::int64_t> first;
  std::optional<std::int64_t> last;
  std::int64_t stride = 1;
};

IndexValueBounds projectIndexBounds(mlir::Value index,
                                    FiniteIndexValues finiteIndexValues,
                                    unsigned depth);

IndexValueBounds projectCountedLoopBounds(mlir::scf::ForOp loop,
                                          FiniteIndexValues finiteIndexValues,
                                          unsigned depth) {
  IndexValueBounds bounds;
  auto step = signedConstant(loop.getStep());
  IndexValueBounds lower =
      projectIndexBounds(loop.getLowerBound(), finiteIndexValues, depth + 1);
  if (!step || *step <= 0 || !lower.first)
    return bounds;
  bounds.first = *lower.first;
  bounds.stride = *step;
  IndexValueBounds upper =
      projectIndexBounds(loop.getUpperBound(), finiteIndexValues, depth + 1);
  std::int64_t highest = 0;
  if (upper.last && !llvm::SubOverflow<std::int64_t>(*upper.last, 1, highest))
    // An empty trip count reaches no index at all, so reporting the first
    // value keeps the progression well formed without admitting any other
    // offset.
    bounds.last = std::max(highest, *bounds.first);
  return bounds;
}

/// Operand range of one integer value, when its bound fits its own width.
std::optional<mlir::ConstantIntRanges>
operandRange(mlir::Value operand, const IndexValueBounds &bounds) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(operand.getType());
  if (!integer || !bounds.first || !bounds.last)
    return std::nullopt;
  llvm::APInt first(64, static_cast<std::uint64_t>(*bounds.first), true);
  llvm::APInt last(64, static_cast<std::uint64_t>(*bounds.last), true);
  if (!first.isSignedIntN(integer.getWidth()) ||
      !last.isSignedIntN(integer.getWidth()))
    return std::nullopt;
  return mlir::ConstantIntRanges::fromSigned(
      first.sextOrTrunc(integer.getWidth()),
      last.sextOrTrunc(integer.getWidth()));
}

IndexValueBounds projectIndexBounds(mlir::Value index,
                                    FiniteIndexValues finiteIndexValues,
                                    unsigned depth) {
  IndexValueBounds bounds;
  if (!index || depth > maximumIndexBoundDepth)
    return bounds;
  if (auto constant = signedConstant(index)) {
    bounds.first = *constant;
    bounds.last = *constant;
    return bounds;
  }
  if (finiteIndexValues)
    if (auto values = finiteIndexValues(index); values && !values->empty()) {
      bounds.first = *values->begin();
      bounds.last = *values->rbegin();
      return bounds;
    }
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(index)) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!loop || argument != loop.getInductionVar())
      return bounds;
    return projectCountedLoopBounds(loop, finiteIndexValues, depth);
  }
  auto operation = index.getDefiningOp();
  auto infer = llvm::dyn_cast_or_null<mlir::InferIntRangeInterface>(operation);
  if (!infer || !llvm::isa<mlir::IntegerType>(index.getType()))
    return bounds;
  llvm::SmallVector<mlir::IntegerValueRange> operands;
  for (mlir::Value input : operation->getOperands()) {
    auto range = operandRange(
        input, projectIndexBounds(input, finiteIndexValues, depth + 1));
    if (range)
      operands.emplace_back(*range);
    else
      operands.push_back(mlir::IntegerValueRange::getMaxRange(input));
  }
  infer.inferResultRangesFromOptional(
      operands, [&](mlir::Value output, const mlir::IntegerValueRange &range) {
        if (output != index || range.isUninitialized())
          return;
        const mlir::ConstantIntRanges &values = range.getValue();
        if (!values.smin().isSignedIntN(64) || !values.smax().isSignedIntN(64))
          return;
        bounds.first = values.smin().getSExtValue();
        bounds.last = values.smax().getSExtValue();
      });
  return bounds;
}

/// Solves the in-bounds containment window for one index. `known` carries the
/// offsets the remaining terms already denote, so the window bounds this index
/// alone.
bool containmentBounds(const LinearByteTerm &term,
                       const AddressByteDomainRequest &request,
                       const ByteAddressDomain &known, std::int64_t &first,
                       std::int64_t &last, bool &reachable) {
  reachable = true;
  const std::uint64_t window =
      *request.allocationByteCount - request.accessByteCount;
  if (window > static_cast<std::uint64_t>(signedMaximum))
    return false;
  std::int64_t lowestProduct = 0;
  std::int64_t highestProduct = 0;
  if (llvm::SubOverflow<std::int64_t>(0, known.highest(), lowestProduct) ||
      llvm::SubOverflow<std::int64_t>(static_cast<std::int64_t>(window),
                                      known.lowest(), highestProduct))
    return false;
  if (lowestProduct == signedMinimum || highestProduct == signedMinimum)
    return false;
  if (lowestProduct > highestProduct) {
    reachable = false;
    return true;
  }
  if (term.byteStride > 0) {
    first = llvm::divideCeilSigned(lowestProduct, term.byteStride);
    last = llvm::divideFloorSigned(highestProduct, term.byteStride);
  } else {
    first = llvm::divideCeilSigned(highestProduct, term.byteStride);
    last = llvm::divideFloorSigned(lowestProduct, term.byteStride);
  }
  return true;
}

/// Byte offsets one unproven index may contribute.
std::optional<ByteAddressDomain>
openIndexContribution(const LinearByteTerm &term,
                      const AddressByteDomainRequest &request,
                      const ByteAddressDomain &known,
                      FiniteIndexValues finiteIndexValues) {
  if (term.byteStride == 0)
    return ByteAddressDomain::singleOffset(0);
  if (term.byteStride == signedMinimum)
    return std::nullopt;
  const IndexValueBounds bounds =
      projectIndexBounds(term.index, finiteIndexValues, 0);
  std::optional<std::int64_t> first = bounds.first;
  std::optional<std::int64_t> last = bounds.last;
  if (request.allocationByteCount && request.inBoundsOfAllocation &&
      !known.isEmpty()) {
    if (request.accessByteCount > *request.allocationByteCount)
      return ByteAddressDomain();
    std::int64_t contained = 0;
    std::int64_t containedEnd = 0;
    bool reachable = true;
    if (!containmentBounds(term, request, known, contained, containedEnd,
                           reachable))
      return std::nullopt;
    if (!reachable)
      return ByteAddressDomain();
    first = first ? std::max(*first, contained) : contained;
    last = last ? std::min(*last, containedEnd) : containedEnd;
  }
  if (!first || !last)
    return std::nullopt;
  if (*first > *last)
    return ByteAddressDomain();
  const std::int64_t stride = bounds.first ? bounds.stride : 1;
  if (stride > 1) {
    // Only the phase fixed by the index's own first value is reachable.
    const std::uint64_t sinceFirst = static_cast<std::uint64_t>(*first) -
                                     static_cast<std::uint64_t>(*bounds.first);
    const std::uint64_t untilLast = static_cast<std::uint64_t>(*last) -
                                    static_cast<std::uint64_t>(*bounds.first);
    const std::uint64_t steps = static_cast<std::uint64_t>(stride);
    first = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(*bounds.first) +
        llvm::divideCeil(sinceFirst, steps) * steps);
    last = static_cast<std::int64_t>(
        static_cast<std::uint64_t>(*bounds.first) + untilLast -
        untilLast % steps);
    if (*first > *last)
      return ByteAddressDomain();
  }
  std::int64_t lowest = 0;
  std::int64_t highest = 0;
  std::int64_t byteStride = 0;
  if (llvm::MulOverflow<std::int64_t>(*first, term.byteStride, lowest) ||
      llvm::MulOverflow<std::int64_t>(*last, term.byteStride, highest) ||
      llvm::MulOverflow<std::int64_t>(
          stride, term.byteStride < 0 ? -term.byteStride : term.byteStride,
          byteStride))
    return std::nullopt;
  if (term.byteStride < 0)
    std::swap(lowest, highest);
  auto contribution =
      ByteAddressDomain::progression(lowest, highest, byteStride);
  if (!contribution)
    return std::nullopt;
  contribution->markApproximate();
  return contribution;
}

void reportUnknownDomain(const AddressByteDomainRequest &request,
                         FiniteIndexValues finiteIndexValues) {
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext,
      [&](llvm::json::Object &fields) {
        fields["context_kind"] = "stored_pointer_address_domain";
        fields["root"] = describeValue(request.root);
        fields["byte_bias"] = request.byteBias;
        fields["access_bytes"] = request.accessByteCount;
        fields["in_bounds_of_allocation"] = request.inBoundsOfAllocation;
        if (request.allocationByteCount)
          fields["allocation_bytes"] = *request.allocationByteCount;
        llvm::json::Array indices;
        for (const LinearByteTerm &term : request.terms) {
          auto values =
              finiteIndexValues ? finiteIndexValues(term.index) : std::nullopt;
          const IndexValueBounds bounds =
              projectIndexBounds(term.index, finiteIndexValues, 0);
          llvm::json::Object entry;
          entry["index"] = describeValue(term.index);
          entry["byte_stride"] = term.byteStride;
          entry["index_domain_known"] = values.has_value();
          if (values)
            entry["index_value_count"] =
                static_cast<std::uint64_t>(values->size());
          if (bounds.first)
            entry["index_lowest"] = *bounds.first;
          if (bounds.last)
            entry["index_highest"] = *bounds.last;
          entry["index_stride"] = bounds.stride;
          indices.push_back(std::move(entry));
        }
        fields["indices"] = std::move(indices);
        fields["progression_limit"] =
            static_cast<std::uint64_t>(ByteAddressDomain::maximumProgressions);
      });
}

} // namespace

std::string describeValue(mlir::Value value) {
  if (!value)
    return "<none>";
  std::string text;
  llvm::raw_string_ostream output(text);
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    value.printAsOperand(output, mlir::OpPrintingFlags());
    output << " in ";
    argument.getOwner()->getParentOp()->print(
        output, mlir::OpPrintingFlags().skipRegions());
  } else {
    value.print(output, mlir::OpPrintingFlags().skipRegions());
  }
  return text;
}

std::uint64_t ByteOffsetProgression::count() const {
  if (last < first || stride <= 0)
    return 0;
  const std::uint64_t span =
      static_cast<std::uint64_t>(last) - static_cast<std::uint64_t>(first);
  const std::uint64_t steps = span / static_cast<std::uint64_t>(stride);
  if (steps == std::numeric_limits<std::uint64_t>::max())
    return steps;
  return steps + 1;
}

ByteAddressDomain ByteAddressDomain::singleOffset(std::int64_t offset) {
  ByteAddressDomain domain;
  domain.progressions_.push_back({offset, offset, 1});
  return domain;
}

std::optional<ByteAddressDomain>
ByteAddressDomain::progression(std::int64_t first, std::int64_t last,
                               std::int64_t stride) {
  if (stride <= 0 || last < first)
    return std::nullopt;
  const std::uint64_t span =
      static_cast<std::uint64_t>(last) - static_cast<std::uint64_t>(first);
  const std::uint64_t steps = static_cast<std::uint64_t>(stride);
  ByteAddressDomain domain;
  domain.progressions_.push_back(
      {first,
       static_cast<std::int64_t>(static_cast<std::uint64_t>(first) + span -
                                 span % steps),
       span < steps ? 1 : stride});
  return domain;
}

std::optional<ByteAddressDomain>
ByteAddressDomain::explicitOffsets(const std::set<std::int64_t> &offsets) {
  if (offsets.empty())
    return std::nullopt;
  llvm::SmallVector<std::int64_t> ordered(offsets.begin(), offsets.end());
  ByteAddressDomain domain;
  std::size_t position = 0;
  while (position < ordered.size()) {
    if (position + 1 == ordered.size()) {
      domain.append({ordered[position], ordered[position], 1});
      break;
    }
    const std::uint64_t stride =
        static_cast<std::uint64_t>(ordered[position + 1]) -
        static_cast<std::uint64_t>(ordered[position]);
    if (stride > static_cast<std::uint64_t>(signedMaximum))
      return std::nullopt;
    std::size_t end = position + 1;
    while (end + 1 < ordered.size() &&
           static_cast<std::uint64_t>(ordered[end + 1]) -
                   static_cast<std::uint64_t>(ordered[end]) ==
               stride)
      ++end;
    domain.append({ordered[position], ordered[end],
                   static_cast<std::int64_t>(stride)});
    position = end + 1;
  }
  return domain;
}

std::uint64_t ByteAddressDomain::count() const {
  std::uint64_t total = 0;
  for (const ByteOffsetProgression &entry : progressions_) {
    const std::uint64_t size = entry.count();
    if (size > std::numeric_limits<std::uint64_t>::max() - total)
      return std::numeric_limits<std::uint64_t>::max();
    total += size;
  }
  return total;
}

std::int64_t ByteAddressDomain::lowest() const {
  std::int64_t result = signedMaximum;
  for (const ByteOffsetProgression &entry : progressions_)
    result = std::min(result, entry.first);
  return result;
}

std::int64_t ByteAddressDomain::highest() const {
  std::int64_t result = signedMinimum;
  for (const ByteOffsetProgression &entry : progressions_)
    result = std::max(result, entry.last);
  return result;
}

std::optional<llvm::SmallVector<std::int64_t>>
ByteAddressDomain::enumerate(std::uint64_t limit) const {
  if (count() > limit)
    return std::nullopt;
  std::set<std::int64_t> distinct;
  for (const ByteOffsetProgression &entry : progressions_) {
    const std::uint64_t size = entry.count();
    for (std::uint64_t step = 0; step != size; ++step)
      distinct.insert(static_cast<std::int64_t>(
          static_cast<std::uint64_t>(entry.first) +
          step * static_cast<std::uint64_t>(entry.stride)));
  }
  return llvm::SmallVector<std::int64_t>(distinct.begin(), distinct.end());
}

bool ByteAddressDomain::scale(std::int64_t byteStride) {
  if (byteStride == 0) {
    progressions_.assign(1, {0, 0, 1});
    return true;
  }
  if (byteStride == signedMinimum)
    return false;
  const std::int64_t magnitude = byteStride < 0 ? -byteStride : byteStride;
  for (ByteOffsetProgression &entry : progressions_) {
    std::int64_t first = 0;
    std::int64_t last = 0;
    std::int64_t stride = 0;
    if (llvm::MulOverflow<std::int64_t>(entry.first, byteStride, first) ||
        llvm::MulOverflow<std::int64_t>(entry.last, byteStride, last) ||
        llvm::MulOverflow<std::int64_t>(entry.stride, magnitude, stride))
      return false;
    if (byteStride < 0)
      std::swap(first, last);
    entry = {first, last, first == last ? 1 : stride};
  }
  return true;
}

bool ByteAddressDomain::addPointwise(const ByteAddressDomain &other) {
  if (progressions_.empty() || other.progressions_.empty()) {
    progressions_.clear();
    exact_ = exact_ && other.exact_;
    return true;
  }
  ByteAddressDomain left = *this;
  ByteAddressDomain right = other;
  while (left.progressions_.size() * right.progressions_.size() >
         maximumProgressions) {
    if (left.progressions_.size() >= right.progressions_.size())
      left.widen();
    else
      right.widen();
  }
  ByteAddressDomain result;
  result.exact_ = left.exact_ && right.exact_;
  for (const ByteOffsetProgression &lhs : left.progressions_)
    for (const ByteOffsetProgression &rhs : right.progressions_) {
      std::int64_t first = 0;
      std::int64_t last = 0;
      if (llvm::AddOverflow<std::int64_t>(lhs.first, rhs.first, first) ||
          llvm::AddOverflow<std::int64_t>(lhs.last, rhs.last, last))
        return false;
      std::int64_t stride = 1;
      if (lhs.isSingleOffset())
        stride = rhs.stride;
      else if (rhs.isSingleOffset() || lhs.stride == rhs.stride)
        stride = lhs.stride;
      else {
        stride = std::gcd(lhs.stride, rhs.stride);
        result.exact_ = false;
      }
      result.append({first, last, first == last ? 1 : stride});
    }
  *this = std::move(result);
  return true;
}

void ByteAddressDomain::restrictTo(std::int64_t lower, std::int64_t upper) {
  llvm::SmallVector<ByteOffsetProgression, 4> retained;
  for (const ByteOffsetProgression &entry : progressions_) {
    if (entry.last < lower || entry.first > upper)
      continue;
    ByteOffsetProgression kept = entry;
    const std::uint64_t stride = static_cast<std::uint64_t>(kept.stride);
    if (kept.first < lower) {
      const std::uint64_t missing = static_cast<std::uint64_t>(lower) -
                                    static_cast<std::uint64_t>(kept.first);
      kept.first = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(kept.first) +
          llvm::divideCeil(missing, stride) * stride);
    }
    if (kept.last > upper) {
      const std::uint64_t excess = static_cast<std::uint64_t>(kept.last) -
                                   static_cast<std::uint64_t>(upper);
      kept.last = static_cast<std::int64_t>(
          static_cast<std::uint64_t>(kept.last) -
          llvm::divideCeil(excess, stride) * stride);
    }
    if (kept.first > kept.last)
      continue;
    if (kept.first == kept.last)
      kept.stride = 1;
    retained.push_back(kept);
  }
  progressions_ = std::move(retained);
}

void ByteAddressDomain::append(ByteOffsetProgression progression) {
  progressions_.push_back(progression);
  if (progressions_.size() > maximumProgressions)
    widen();
}

void ByteAddressDomain::widen() {
  if (progressions_.size() < 2)
    return;
  const std::int64_t base = lowest();
  const std::int64_t bound = highest();
  std::uint64_t stride = 0;
  for (const ByteOffsetProgression &entry : progressions_) {
    if (!entry.isSingleOffset())
      stride = std::gcd(stride, static_cast<std::uint64_t>(entry.stride));
    stride = std::gcd(stride, static_cast<std::uint64_t>(entry.first) -
                                  static_cast<std::uint64_t>(base));
    stride = std::gcd(stride, static_cast<std::uint64_t>(entry.last) -
                                  static_cast<std::uint64_t>(base));
  }
  if (stride == 0 || stride > static_cast<std::uint64_t>(signedMaximum))
    stride = 1;
  const std::uint64_t span =
      static_cast<std::uint64_t>(bound) - static_cast<std::uint64_t>(base);
  progressions_.assign(
      1, {base,
          static_cast<std::int64_t>(static_cast<std::uint64_t>(base) + span -
                                    span % stride),
          span < stride ? 1 : static_cast<std::int64_t>(stride)});
  exact_ = false;
}

static std::optional<AddressByteDomain>
deriveAddressByteDomain(const AddressByteDomainRequest &request,
                        FiniteIndexValues finiteIndexValues) {
  AddressByteDomain result;
  result.offsets = ByteAddressDomain::singleOffset(request.byteBias);
  result.exhaustive = true;
  llvm::SmallDenseSet<mlir::Value, 4> seenIndices;
  const LinearByteTerm *openTerm = nullptr;
  for (const LinearByteTerm &term : request.terms) {
    // One index reached through two terms is scaled independently below, which
    // denotes more offsets than the address itself can reach.
    if (!seenIndices.insert(term.index).second)
      result.offsets.markApproximate();
    auto values =
        finiteIndexValues ? finiteIndexValues(term.index) : std::nullopt;
    if (!values || values->empty()) {
      if (openTerm)
        return std::nullopt;
      openTerm = &term;
      continue;
    }
    auto scaled = ByteAddressDomain::explicitOffsets(*values);
    if (!scaled || !scaled->scale(term.byteStride) ||
        !result.offsets.addPointwise(*scaled))
      return std::nullopt;
    result.exhaustive &=
        values->size() == 1 ||
        llvm::is_contained(request.completedIndices, term.index);
  }
  if (openTerm) {
    auto contribution = openIndexContribution(
        *openTerm, request, result.offsets, finiteIndexValues);
    if (!contribution || !result.offsets.addPointwise(*contribution))
      return std::nullopt;
  }
  if (!result.offsets.isExact() && request.inBoundsOfAllocation &&
      request.allocationByteCount) {
    if (request.accessByteCount > *request.allocationByteCount)
      return AddressByteDomain{};
    const std::uint64_t window =
        *request.allocationByteCount - request.accessByteCount;
    if (window > static_cast<std::uint64_t>(signedMaximum))
      return std::nullopt;
    result.offsets.restrictTo(0, static_cast<std::int64_t>(window));
  }
  result.exhaustive &= result.offsets.isExact();
  return result;
}

std::optional<AddressByteDomain>
projectAddressByteDomain(const AddressByteDomainRequest &request,
                         FiniteIndexValues finiteIndexValues) {
  auto domain = deriveAddressByteDomain(request, finiteIndexValues);
  if (!domain)
    reportUnknownDomain(request, finiteIndexValues);
  return domain;
}

bool removeContiguousIndex(llvm::SmallVectorImpl<LinearByteTerm> &terms,
                           mlir::Value index, std::uint64_t accessByteCount) {
  bool removed = false;
  llvm::SmallVector<LinearByteTerm, 4> retained;
  for (const LinearByteTerm &term : terms) {
    if (term.index != index) {
      retained.push_back(term);
      continue;
    }
    if (removed || term.byteStride <= 0 ||
        static_cast<std::uint64_t>(term.byteStride) != accessByteCount)
      return false;
    removed = true;
  }
  if (!removed)
    return false;
  terms.assign(retained.begin(), retained.end());
  return true;
}

} // namespace loom::frontend::analysis
