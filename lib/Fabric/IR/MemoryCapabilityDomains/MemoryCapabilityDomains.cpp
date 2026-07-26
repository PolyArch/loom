#include "Fabric/IR/MemoryCapabilityDomains.h"

#include "Fabric/IR/MemoryCapabilityRelation.h"
#include "Fabric/IR/ReducedProductRelation.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

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

llvm::Expected<std::uint8_t> checkedTag(ReadSubwordSemantics semantics);
llvm::Expected<std::uint8_t> checkedTag(WriteSubwordSemantics semantics);
llvm::Expected<std::uint8_t> checkedTag(InactiveLaneSemantics semantics);

using FiniteAtom = detail::ReducedFiniteAtom;
using FiniteDomain = detail::ReducedFiniteDomain;
using RelationDomain = detail::ReducedProductDomain;
using RelationRow = detail::ReducedProductRow;

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64(bytes, field.size());
  bytes.insert(bytes.end(), field.begin(), field.end());
}

llvm::Expected<FiniteAtom> canonicalAtom(MemoryAccessForm value) {
  auto encoded = dataflow::encodeMemoryAccessForm(value);
  if (!encoded)
    return encoded.takeError();
  return FiniteAtom{std::vector<std::uint8_t>(encoded->bytes().begin(),
                                              encoded->bytes().end())};
}

llvm::Expected<FiniteAtom> canonicalAtom(MaskInactivePair value) {
  auto mask = dataflow::encodeMemoryMaskForm(value.mask);
  if (!mask)
    return mask.takeError();
  auto inactive = checkedTag(value.inactive);
  if (!inactive)
    return inactive.takeError();
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, mask->bytes());
  appendU32(bytes, *inactive);
  return FiniteAtom{std::move(bytes)};
}

llvm::Expected<FiniteAtom> canonicalAtom(ReadSubwordSemantics value) {
  auto tag = checkedTag(value);
  if (!tag)
    return tag.takeError();
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, *tag);
  return FiniteAtom{std::move(bytes)};
}

llvm::Expected<FiniteAtom> canonicalAtom(WriteSubwordSemantics value) {
  auto tag = checkedTag(value);
  if (!tag)
    return tag.takeError();
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, *tag);
  return FiniteAtom{std::move(bytes)};
}

template <typename Value>
llvm::Expected<FiniteDomain>
canonicalFiniteDomain(llvm::ArrayRef<Value> values) {
  if (values.empty())
    return invalid("finite relation domain must not be empty");
  std::vector<FiniteAtom> atoms;
  atoms.reserve(values.size());
  for (Value value : values) {
    auto atom = canonicalAtom(value);
    if (!atom)
      return atom.takeError();
    atoms.push_back(std::move(*atom));
  }
  llvm::sort(atoms, [](const FiniteAtom &lhs, const FiniteAtom &rhs) {
    return lhs.bytes < rhs.bytes;
  });
  for (std::size_t index = 1; index < atoms.size(); ++index)
    if (atoms[index - 1].bytes == atoms[index].bytes)
      return invalid("finite relation domain contains a duplicate");
  return FiniteDomain{std::move(atoms)};
}

llvm::Expected<RelationRow>
accessClassRelationRow(const MemoryAccessClass &accessClass) {
  const MemoryAccessForm formValues[] = {accessClass.accessForm()};
  auto forms = canonicalFiniteDomain<MemoryAccessForm>(formValues);
  if (!forms)
    return forms.takeError();
  auto masks =
      canonicalFiniteDomain<MaskInactivePair>(accessClass.maskInactivePairs());
  if (!masks)
    return masks.takeError();
  auto reads = canonicalFiniteDomain<ReadSubwordSemantics>(
      accessClass.readSubwordSemantics().values());
  if (!reads)
    return reads.takeError();
  auto writes = canonicalFiniteDomain<WriteSubwordSemantics>(
      accessClass.writeSubwordSemantics().values());
  if (!writes)
    return writes.takeError();
  RelationRow row;
  row.push_back(std::move(*forms));
  row.push_back(accessClass.elementWidths());
  row.push_back(accessClass.flattenedLaneCounts());
  row.push_back(std::move(*masks));
  row.push_back(accessClass.sourceAlignments().exponents());
  row.push_back(std::move(*reads));
  row.push_back(std::move(*writes));
  return row;
}

llvm::Expected<std::uint32_t> readU32(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - std::min(offset, bytes.size()) < 4)
    return invalid("finite relation atom is truncated");
  std::uint32_t value = 0;
  for (unsigned index = 0; index < 4; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - std::min(offset, bytes.size()) < 8)
    return invalid("finite relation atom is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<MemoryAccessForm>
decodeFiniteAtom(llvm::ArrayRef<std::uint8_t> bytes, MemoryAccessForm *) {
  return dataflow::decodeMemoryAccessForm(bytes);
}

llvm::Expected<MaskInactivePair>
decodeFiniteAtom(llvm::ArrayRef<std::uint8_t> bytes, MaskInactivePair *) {
  std::size_t offset = 0;
  auto maskLength = readU64(bytes, offset);
  if (!maskLength)
    return maskLength.takeError();
  if (*maskLength > bytes.size() - std::min(offset, bytes.size()))
    return invalid("mask/inactive relation atom is truncated");
  llvm::ArrayRef<std::uint8_t> maskBytes = bytes.slice(offset, *maskLength);
  offset += *maskLength;
  auto mask = dataflow::decodeMemoryMaskForm(maskBytes);
  if (!mask)
    return mask.takeError();
  auto inactiveTag = readU32(bytes, offset);
  if (!inactiveTag)
    return inactiveTag.takeError();
  if (offset != bytes.size() ||
      *inactiveTag > std::numeric_limits<std::uint8_t>::max())
    return invalid("mask/inactive relation atom is not canonical");
  auto inactive =
      decodeInactiveLaneSemantics(static_cast<std::uint8_t>(*inactiveTag));
  if (!inactive)
    return inactive.takeError();
  return MaskInactivePair{*mask, *inactive};
}

llvm::Expected<ReadSubwordSemantics>
decodeFiniteAtom(llvm::ArrayRef<std::uint8_t> bytes, ReadSubwordSemantics *) {
  std::size_t offset = 0;
  auto tag = readU32(bytes, offset);
  if (!tag)
    return tag.takeError();
  if (offset != bytes.size() || *tag > std::numeric_limits<std::uint8_t>::max())
    return invalid("read-subword relation atom is not canonical");
  return decodeReadSubwordSemantics(static_cast<std::uint8_t>(*tag));
}

llvm::Expected<WriteSubwordSemantics>
decodeFiniteAtom(llvm::ArrayRef<std::uint8_t> bytes, WriteSubwordSemantics *) {
  std::size_t offset = 0;
  auto tag = readU32(bytes, offset);
  if (!tag)
    return tag.takeError();
  if (offset != bytes.size() || *tag > std::numeric_limits<std::uint8_t>::max())
    return invalid("write-subword relation atom is not canonical");
  return decodeWriteSubwordSemantics(static_cast<std::uint8_t>(*tag));
}

template <typename Value>
llvm::Expected<std::vector<Value>> valuesOf(const RelationDomain &domain) {
  const auto *finite = std::get_if<FiniteDomain>(&domain);
  if (!finite)
    return invalid("canonical relation field has the wrong domain kind");
  std::vector<Value> values;
  values.reserve(finite->atoms.size());
  for (const FiniteAtom &atom : finite->atoms) {
    auto value = decodeFiniteAtom(atom.bytes, static_cast<Value *>(nullptr));
    if (!value)
      return value.takeError();
    values.push_back(*value);
  }
  return values;
}

llvm::Expected<std::vector<MemoryAccessClass>>
accessClassesFromRows(llvm::ArrayRef<RelationRow> rows) {
  std::vector<MemoryAccessClass> result;
  result.reserve(rows.size());
  for (const RelationRow &row : rows) {
    if (row.size() != 7)
      return invalid("canonical access relation has the wrong field count");
    auto forms = valuesOf<MemoryAccessForm>(row[0]);
    auto masks = valuesOf<MaskInactivePair>(row[3]);
    auto reads = valuesOf<ReadSubwordSemantics>(row[5]);
    auto writes = valuesOf<WriteSubwordSemantics>(row[6]);
    if (!forms)
      return forms.takeError();
    if (!masks)
      return masks.takeError();
    if (!reads)
      return reads.takeError();
    if (!writes)
      return writes.takeError();
    if (forms->size() != 1)
      return invalid("canonical access form partition is not a singleton");
    const auto *elementWidths = std::get_if<UnsignedDomain>(&row[1]);
    const auto *laneCounts = std::get_if<UnsignedDomain>(&row[2]);
    const auto *alignmentExponents = std::get_if<UnsignedDomain>(&row[4]);
    if (!elementWidths || !laneCounts || !alignmentExponents)
      return invalid("canonical access relation has a wrong unsigned field");
    auto alignments = AlignmentDomain::create(*alignmentExponents);
    if (!alignments)
      return alignments.takeError();
    auto readDomain =
        ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical(*reads);
    if (!readDomain)
      return readDomain.takeError();
    auto writeDomain =
        ClosedEnumDomain<WriteSubwordSemantics>::fromCanonical(*writes);
    if (!writeDomain)
      return writeDomain.takeError();
    auto accessClass = MemoryAccessClass::create(
        forms->front(), *elementWidths, *laneCounts, *masks,
        std::move(*alignments), std::move(*readDomain),
        std::move(*writeDomain));
    if (!accessClass)
      return accessClass.takeError();
    result.push_back(std::move(*accessClass));
  }
  return result;
}

llvm::Expected<std::vector<MemoryAccessClass>>
normalizeAccessClasses(llvm::ArrayRef<MemoryAccessClass> accessClasses) {
  if (accessClasses.empty())
    return invalid("parameterized memory access domain must not be empty");

  std::vector<RelationRow> rows;
  rows.reserve(accessClasses.size());
  for (const MemoryAccessClass &accessClass : accessClasses) {
    auto row = accessClassRelationRow(accessClass);
    if (!row)
      return row.takeError();
    rows.push_back(std::move(*row));
  }
  const bool groupFiniteFields[] = {false, true, true, true, true, true, true};
  auto normalized = detail::reduceProductRelation(rows, groupFiniteFields);
  if (!normalized)
    return normalized.takeError();
  return accessClassesFromRows(*normalized);
}

llvm::Expected<std::vector<std::uint8_t>>
encodeAccessRelation(llvm::ArrayRef<MemoryAccessClass> accessClasses) {
  std::vector<RelationRow> rows;
  rows.reserve(accessClasses.size());
  for (const MemoryAccessClass &accessClass : accessClasses) {
    auto row = accessClassRelationRow(accessClass);
    if (!row)
      return row.takeError();
    rows.push_back(std::move(*row));
  }
  return detail::encodeReducedProductRelation(rows);
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

  auto canonicalMasks =
      canonicalFiniteDomain<MaskInactivePair>(maskInactivePairs);
  if (!canonicalMasks)
    return canonicalMasks.takeError();
  std::vector<MaskInactivePair> sortedMasks;
  sortedMasks.reserve(canonicalMasks->atoms.size());
  for (const FiniteAtom &atom : canonicalMasks->atoms) {
    auto pair =
        decodeFiniteAtom(atom.bytes, static_cast<MaskInactivePair *>(nullptr));
    if (!pair)
      return pair.takeError();
    sortedMasks.push_back(*pair);
  }

  return MemoryAccessClass(accessForm, std::move(elementWidths),
                           std::move(flattenedLaneCounts),
                           std::move(sortedMasks), std::move(sourceAlignments),
                           std::move(readSubword), std::move(writeSubword));
}

bool MemoryAccessClass::contains(
    const CanonicalMemoryAccessView &access) const {
  const MemoryActorContract &contract = access.contract();
  const std::optional<std::uint64_t> sourceAlignment =
      contract.atomic ? contract.sourceAlignmentBytes
                      : std::optional<std::uint64_t>(1);
  if (access.form() != accessForm_ ||
      !elementWidths_.contains(access.elementBits()) ||
      !flattenedLaneCounts_.contains(access.laneCount()) || !sourceAlignment ||
      !sourceAlignments_.containsBytes(*sourceAlignment))
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
  auto normalized = normalizeAccessClasses(accessClasses);
  if (!normalized)
    return normalized.takeError();
  return ParameterizedMemoryAccessDomain(std::move(*normalized));
}

llvm::Expected<ParameterizedMemoryAccessDomain>
ParameterizedMemoryAccessDomain::fromCanonical(
    llvm::ArrayRef<MemoryAccessClass> accessClasses) {
  auto encoded = encodeAccessRelation(accessClasses);
  if (!encoded)
    return encoded.takeError();
  auto normalized = normalizeAccessClasses(accessClasses);
  if (!normalized)
    return normalized.takeError();
  auto canonical = encodeAccessRelation(*normalized);
  if (!canonical)
    return canonical.takeError();
  if (*encoded != *canonical)
    return invalid("parameterized memory access domain is not canonical");
  return ParameterizedMemoryAccessDomain(std::move(*normalized));
}

const MemoryAccessClass *ParameterizedMemoryAccessDomain::matchingClass(
    const CanonicalMemoryAccessView &access) const {
  for (const MemoryAccessClass &candidate : accessClasses_)
    if (candidate.contains(access))
      return &candidate;
  return nullptr;
}

llvm::Expected<std::vector<std::uint8_t>> encodeParameterizedMemoryAccessDomain(
    const ParameterizedMemoryAccessDomain &domain) {
  return encodeAccessRelation(domain.accessClasses());
}

llvm::Expected<ParameterizedMemoryAccessDomain>
decodeParameterizedMemoryAccessDomain(llvm::ArrayRef<std::uint8_t> bytes) {
  auto rows = detail::decodeReducedProductRelation(bytes);
  if (!rows)
    return rows.takeError();
  auto accessClasses = accessClassesFromRows(*rows);
  if (!accessClasses)
    return accessClasses.takeError();
  auto domain = ParameterizedMemoryAccessDomain::fromCanonical(*accessClasses);
  if (!domain)
    return domain.takeError();
  auto canonical = encodeParameterizedMemoryAccessDomain(*domain);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid(
        "parameterized memory access domain bytes are not canonical");
  return domain;
}

llvm::Expected<std::vector<std::uint8_t>>
encodeUnsignedDomain(const UnsignedDomain &domain) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, domain.intervals().size());
  for (UnsignedInterval interval : domain.intervals()) {
    appendU64(bytes, interval.lower);
    appendU64(bytes, interval.upper);
  }
  return bytes;
}

llvm::Expected<UnsignedDomain>
decodeUnsignedDomain(llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  auto count = readU64(bytes, offset);
  if (!count)
    return count.takeError();
  if (*count > (bytes.size() - offset) / 16)
    return invalid("unsigned domain interval count exceeds framing");
  std::vector<UnsignedInterval> intervals;
  intervals.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto lower = readU64(bytes, offset);
    if (!lower)
      return lower.takeError();
    auto upper = readU64(bytes, offset);
    if (!upper)
      return upper.takeError();
    intervals.push_back({*lower, *upper});
  }
  if (offset != bytes.size())
    return invalid("unsigned domain has trailing bytes");
  auto domain = UnsignedDomain::fromCanonical(intervals);
  if (!domain)
    return domain.takeError();
  auto canonical = encodeUnsignedDomain(*domain);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("unsigned domain bytes are not canonical");
  return domain;
}

llvm::Expected<detail::ReducedProductRow>
detail::projectMemoryAccessClass(const MemoryAccessClass &accessClass) {
  return accessClassRelationRow(accessClass);
}

llvm::Expected<MemoryAccessClass>
detail::importMemoryAccessClass(const ReducedProductRow &relation) {
  auto accessClasses = accessClassesFromRows({relation});
  if (!accessClasses)
    return accessClasses.takeError();
  if (accessClasses->size() != 1)
    return invalid("one access relation row did not import as one class");
  return std::move(accessClasses->front());
}

} // namespace fabric
