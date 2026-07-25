#include "Fabric/IR/MemoryCapabilityDomains.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <system_error>

using namespace dataflow::semantics;

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool isAdjacent(std::uint64_t lower, std::uint64_t upper) {
  return upper != std::numeric_limits<std::uint64_t>::max() &&
         lower == upper + 1;
}

llvm::Error validateAccessForm(MemoryAccessForm form) {
  switch (form) {
  case MemoryAccessForm::Element:
  case MemoryAccessForm::Contiguous:
  case MemoryAccessForm::Indexed:
    return llvm::Error::success();
  }
  return invalid("unknown memory access form");
}

llvm::Error validateMaskForm(MemoryMaskForm form) {
  switch (form) {
  case MemoryMaskForm::Absent:
  case MemoryMaskForm::Dynamic:
    return llvm::Error::success();
  }
  return invalid("unknown memory mask form");
}

bool domainsIntersect(const UnsignedDomain &lhs, const UnsignedDomain &rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < lhs.intervals().size() && right < rhs.intervals().size()) {
    const UnsignedInterval a = lhs.intervals()[left];
    const UnsignedInterval b = rhs.intervals()[right];
    if (a.upper < b.lower) {
      ++left;
      continue;
    }
    if (b.upper < a.lower) {
      ++right;
      continue;
    }
    return true;
  }
  return false;
}

template <typename Value>
bool domainsIntersect(llvm::ArrayRef<Value> lhs, llvm::ArrayRef<Value> rhs) {
  for (Value value : lhs)
    if (llvm::is_contained(rhs, value))
      return true;
  return false;
}

bool accessClassesOverlap(const MemoryAccessClass &lhs,
                          const MemoryAccessClass &rhs) {
  return lhs.accessForm() == rhs.accessForm() &&
         domainsIntersect(lhs.elementWidths(), rhs.elementWidths()) &&
         domainsIntersect(lhs.flattenedLaneCounts(),
                          rhs.flattenedLaneCounts()) &&
         domainsIntersect(lhs.maskInactivePairs(), rhs.maskInactivePairs()) &&
         domainsIntersect(lhs.sourceAlignments().exponents(),
                          rhs.sourceAlignments().exponents()) &&
         domainsIntersect(lhs.readSubwordSemantics().values(),
                          rhs.readSubwordSemantics().values()) &&
         domainsIntersect(lhs.writeSubwordSemantics().values(),
                          rhs.writeSubwordSemantics().values());
}

llvm::Expected<std::uint8_t> checkedTag(ReadSubwordSemantics semantics) {
  switch (semantics) {
  case ReadSubwordSemantics::NotApplicable:
    return 0;
  case ReadSubwordSemantics::Exact:
    return 1;
  case ReadSubwordSemantics::ZeroExtend:
    return 2;
  }
  return invalid("unknown read-subword semantics");
}

llvm::Expected<std::uint8_t> checkedTag(WriteSubwordSemantics semantics) {
  switch (semantics) {
  case WriteSubwordSemantics::NotApplicable:
    return 0;
  case WriteSubwordSemantics::Exact:
    return 1;
  case WriteSubwordSemantics::ByteEnable:
    return 2;
  }
  return invalid("unknown write-subword semantics");
}

llvm::Expected<std::uint8_t> checkedTag(InactiveLaneSemantics semantics) {
  switch (semantics) {
  case InactiveLaneSemantics::NotApplicable:
    return 0;
  case InactiveLaneSemantics::Suppress:
    return 1;
  case InactiveLaneSemantics::SuppressAndZeroFill:
    return 2;
  }
  return invalid("unknown inactive-lane semantics");
}

} // namespace

llvm::Error detail::domainError(const char *message) {
  return invalid(message);
}

llvm::Expected<std::uint8_t>
detail::ClosedEnumTraits<ReadSubwordSemantics>::tag(
    ReadSubwordSemantics value) {
  return checkedTag(value);
}

llvm::Expected<std::uint8_t>
detail::ClosedEnumTraits<WriteSubwordSemantics>::tag(
    WriteSubwordSemantics value) {
  return checkedTag(value);
}

llvm::Expected<std::uint8_t>
detail::ClosedEnumTraits<InactiveLaneSemantics>::tag(
    InactiveLaneSemantics value) {
  return checkedTag(value);
}

llvm::Expected<UnsignedDomain>
UnsignedDomain::normalize(llvm::ArrayRef<UnsignedInterval> intervals) {
  if (intervals.empty())
    return invalid("unsigned domain must not be empty");

  std::vector<UnsignedInterval> normalized(intervals.begin(), intervals.end());
  for (UnsignedInterval interval : normalized)
    if (interval.lower > interval.upper)
      return invalid("unsigned interval lower bound exceeds its upper bound");

  llvm::sort(normalized, [](UnsignedInterval lhs, UnsignedInterval rhs) {
    return std::tie(lhs.lower, lhs.upper) < std::tie(rhs.lower, rhs.upper);
  });

  std::vector<UnsignedInterval> merged;
  merged.reserve(normalized.size());
  for (UnsignedInterval interval : normalized) {
    if (merged.empty() || (interval.lower > merged.back().upper &&
                           !isAdjacent(interval.lower, merged.back().upper))) {
      merged.push_back(interval);
      continue;
    }
    merged.back().upper = std::max(merged.back().upper, interval.upper);
  }
  return UnsignedDomain(std::move(merged));
}

llvm::Expected<UnsignedDomain>
UnsignedDomain::fromCanonical(llvm::ArrayRef<UnsignedInterval> intervals) {
  if (intervals.empty())
    return invalid("unsigned domain must not be empty");

  for (std::size_t index = 0; index < intervals.size(); ++index) {
    const UnsignedInterval interval = intervals[index];
    if (interval.lower > interval.upper)
      return invalid("unsigned interval lower bound exceeds its upper bound");
    if (index == 0)
      continue;
    const UnsignedInterval previous = intervals[index - 1];
    if (interval.lower <= previous.upper ||
        isAdjacent(interval.lower, previous.upper))
      return invalid("unsigned domain is not canonical");
  }
  return UnsignedDomain(
      std::vector<UnsignedInterval>(intervals.begin(), intervals.end()));
}

bool UnsignedDomain::contains(std::uint64_t value) const {
  const auto candidate = llvm::upper_bound(
      intervals_, value, [](std::uint64_t needle, UnsignedInterval interval) {
        return needle < interval.lower;
      });
  if (candidate == intervals_.begin())
    return false;
  const UnsignedInterval interval = *std::prev(candidate);
  return value <= interval.upper;
}

llvm::Expected<AlignmentDomain>
AlignmentDomain::create(UnsignedDomain exponents) {
  for (UnsignedInterval interval : exponents.intervals())
    if (interval.upper > 63)
      return invalid("alignment exponent is outside [0, 63]");
  return AlignmentDomain(std::move(exponents));
}

bool AlignmentDomain::containsBytes(std::uint64_t bytes) const {
  return llvm::isPowerOf2_64(bytes) &&
         exponents_.contains(llvm::Log2_64(bytes));
}

std::uint8_t getCanonicalTag(ReadSubwordSemantics semantics) {
  switch (semantics) {
  case ReadSubwordSemantics::NotApplicable:
    return 0;
  case ReadSubwordSemantics::Exact:
    return 1;
  case ReadSubwordSemantics::ZeroExtend:
    return 2;
  }
  llvm_unreachable("unknown read-subword semantics");
}

std::uint8_t getCanonicalTag(WriteSubwordSemantics semantics) {
  switch (semantics) {
  case WriteSubwordSemantics::NotApplicable:
    return 0;
  case WriteSubwordSemantics::Exact:
    return 1;
  case WriteSubwordSemantics::ByteEnable:
    return 2;
  }
  llvm_unreachable("unknown write-subword semantics");
}

std::uint8_t getCanonicalTag(InactiveLaneSemantics semantics) {
  switch (semantics) {
  case InactiveLaneSemantics::NotApplicable:
    return 0;
  case InactiveLaneSemantics::Suppress:
    return 1;
  case InactiveLaneSemantics::SuppressAndZeroFill:
    return 2;
  }
  llvm_unreachable("unknown inactive-lane semantics");
}

llvm::Expected<ReadSubwordSemantics>
decodeReadSubwordSemantics(std::uint8_t tag) {
  switch (tag) {
  case 0:
    return ReadSubwordSemantics::NotApplicable;
  case 1:
    return ReadSubwordSemantics::Exact;
  case 2:
    return ReadSubwordSemantics::ZeroExtend;
  default:
    return invalid("unknown read-subword semantics tag");
  }
}

llvm::Expected<WriteSubwordSemantics>
decodeWriteSubwordSemantics(std::uint8_t tag) {
  switch (tag) {
  case 0:
    return WriteSubwordSemantics::NotApplicable;
  case 1:
    return WriteSubwordSemantics::Exact;
  case 2:
    return WriteSubwordSemantics::ByteEnable;
  default:
    return invalid("unknown write-subword semantics tag");
  }
}

llvm::Expected<InactiveLaneSemantics>
decodeInactiveLaneSemantics(std::uint8_t tag) {
  switch (tag) {
  case 0:
    return InactiveLaneSemantics::NotApplicable;
  case 1:
    return InactiveLaneSemantics::Suppress;
  case 2:
    return InactiveLaneSemantics::SuppressAndZeroFill;
  default:
    return invalid("unknown inactive-lane semantics tag");
  }
}

llvm::Expected<MemoryAccessClass> MemoryAccessClass::create(
    MemoryAccessForm accessForm, UnsignedDomain elementWidths,
    UnsignedDomain flattenedLaneCounts,
    llvm::ArrayRef<MaskInactivePair> maskInactivePairs,
    AlignmentDomain sourceAlignments,
    ClosedEnumDomain<ReadSubwordSemantics> readSubword,
    ClosedEnumDomain<WriteSubwordSemantics> writeSubword) {
  if (llvm::Error error = validateAccessForm(accessForm))
    return std::move(error);
  if (elementWidths.contains(0))
    return invalid(
        "memory element width domain must contain only positive values");
  if (flattenedLaneCounts.contains(0))
    return invalid(
        "memory lane-count domain must contain only positive values");
  if (accessForm == MemoryAccessForm::Element &&
      (flattenedLaneCounts.intervals().size() != 1 ||
       flattenedLaneCounts.intervals().front() != UnsignedInterval{1, 1}))
    return invalid("element access must have exactly one flattened lane");
  if (maskInactivePairs.empty())
    return invalid("mask and inactive-lane domain must not be empty");

  for (std::size_t index = 0; index < maskInactivePairs.size(); ++index) {
    const MaskInactivePair pair = maskInactivePairs[index];
    if (llvm::Error error = validateMaskForm(pair.mask))
      return std::move(error);
    llvm::Expected<std::uint8_t> inactive = checkedTag(pair.inactive);
    if (!inactive)
      return inactive.takeError();
    if ((pair.mask == MemoryMaskForm::Absent) !=
        (pair.inactive == InactiveLaneSemantics::NotApplicable))
      return invalid("mask and inactive-lane semantics are inconsistent");

    if (llvm::is_contained(maskInactivePairs.take_front(index), pair))
      return invalid("mask and inactive-lane domain contains a duplicate");
  }

  return MemoryAccessClass(
      accessForm, std::move(elementWidths), std::move(flattenedLaneCounts),
      std::vector<MaskInactivePair>(maskInactivePairs.begin(),
                                    maskInactivePairs.end()),
      std::move(sourceAlignments), std::move(readSubword),
      std::move(writeSubword));
}

bool MemoryAccessClass::contains(
    const CanonicalMemoryAccessView &access) const {
  // Plain access has the specification-defined alignment of one byte. Atomic
  // access stays fail-closed until its owner projection exposes exact source
  // alignment; type and payload width are not alignment evidence.
  if (access.contract().atomic)
    return false;
  if (access.form() != accessForm_ ||
      !elementWidths_.contains(access.elementBits()) ||
      !flattenedLaneCounts_.contains(access.laneCount()) ||
      !sourceAlignments_.containsBytes(1))
    return false;

  InactiveLaneSemantics required = InactiveLaneSemantics::NotApplicable;
  if (access.maskForm() == MemoryMaskForm::Dynamic) {
    required = access.operation() == MemoryAccessOperation::Store
                   ? InactiveLaneSemantics::Suppress
                   : InactiveLaneSemantics::SuppressAndZeroFill;
  }
  return llvm::is_contained(maskInactivePairs_,
                            MaskInactivePair{access.maskForm(), required});
}

llvm::Expected<ParameterizedMemoryAccessDomain>
ParameterizedMemoryAccessDomain::create(
    llvm::ArrayRef<MemoryAccessClass> accessClasses) {
  if (accessClasses.empty())
    return invalid("parameterized memory access domain must not be empty");

  for (std::size_t left = 0; left < accessClasses.size(); ++left)
    for (std::size_t right = left + 1; right < accessClasses.size(); ++right)
      if (accessClassesOverlap(accessClasses[left], accessClasses[right]))
        return invalid("parameterized memory access classes overlap");

  return ParameterizedMemoryAccessDomain(std::vector<MemoryAccessClass>(
      accessClasses.begin(), accessClasses.end()));
}

const MemoryAccessClass *ParameterizedMemoryAccessDomain::matchingClass(
    const CanonicalMemoryAccessView &access) const {
  for (const MemoryAccessClass &candidate : accessClasses_)
    if (candidate.contains(access))
      return &candidate;
  return nullptr;
}

} // namespace fabric
