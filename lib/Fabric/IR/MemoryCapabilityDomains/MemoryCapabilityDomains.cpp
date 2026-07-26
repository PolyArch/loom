#include "Fabric/IR/MemoryCapabilityDomains.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <variant>

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

llvm::Expected<std::uint8_t> checkedTag(ReadSubwordSemantics semantics);
llvm::Expected<std::uint8_t> checkedTag(WriteSubwordSemantics semantics);
llvm::Expected<std::uint8_t> checkedTag(InactiveLaneSemantics semantics);

using AccessAtomValue =
    std::variant<MemoryAccessForm, MaskInactivePair, ReadSubwordSemantics,
                 WriteSubwordSemantics>;

struct FiniteAtom {
  std::vector<std::uint8_t> bytes;
  AccessAtomValue value;
};

struct FiniteDomain {
  std::vector<FiniteAtom> atoms;
};

using RelationDomain = std::variant<FiniteDomain, UnsignedDomain>;
using RelationRow = std::vector<RelationDomain>;

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
                                              encoded->bytes().end()),
                    value};
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
  return FiniteAtom{std::move(bytes), value};
}

llvm::Expected<FiniteAtom> canonicalAtom(ReadSubwordSemantics value) {
  auto tag = checkedTag(value);
  if (!tag)
    return tag.takeError();
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, *tag);
  return FiniteAtom{std::move(bytes), value};
}

llvm::Expected<FiniteAtom> canonicalAtom(WriteSubwordSemantics value) {
  auto tag = checkedTag(value);
  if (!tag)
    return tag.takeError();
  std::vector<std::uint8_t> bytes;
  appendU32(bytes, *tag);
  return FiniteAtom{std::move(bytes), value};
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

bool finiteDomainsIntersect(const FiniteDomain &lhs, const FiniteDomain &rhs) {
  std::size_t left = 0;
  std::size_t right = 0;
  while (left < lhs.atoms.size() && right < rhs.atoms.size()) {
    if (lhs.atoms[left].bytes < rhs.atoms[right].bytes) {
      ++left;
      continue;
    }
    if (rhs.atoms[right].bytes < lhs.atoms[left].bytes) {
      ++right;
      continue;
    }
    return true;
  }
  return false;
}

bool relationDomainsIntersect(const RelationDomain &lhs,
                              const RelationDomain &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (auto *left = std::get_if<FiniteDomain>(&lhs))
    return finiteDomainsIntersect(*left, std::get<FiniteDomain>(rhs));
  return domainsIntersect(std::get<UnsignedDomain>(lhs),
                          std::get<UnsignedDomain>(rhs));
}

bool relationRowsOverlap(const RelationRow &lhs, const RelationRow &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!relationDomainsIntersect(left, right))
      return false;
  return true;
}

std::vector<std::uint8_t> encodeRelationDomain(const RelationDomain &domain) {
  std::vector<std::uint8_t> bytes;
  if (auto *finite = std::get_if<FiniteDomain>(&domain)) {
    appendU32(bytes, 0);
    appendU64(bytes, finite->atoms.size());
    for (const FiniteAtom &atom : finite->atoms)
      appendFramed(bytes, atom.bytes);
    return bytes;
  }
  appendU32(bytes, 1);
  const UnsignedDomain &unsignedDomain = std::get<UnsignedDomain>(domain);
  appendU64(bytes, unsignedDomain.intervals().size());
  for (UnsignedInterval interval : unsignedDomain.intervals()) {
    appendU64(bytes, interval.lower);
    appendU64(bytes, interval.upper);
  }
  return bytes;
}

std::vector<std::uint8_t> encodeRelationRow(const RelationRow &row) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, row.size());
  for (const RelationDomain &domain : row) {
    std::vector<std::uint8_t> encoded = encodeRelationDomain(domain);
    appendFramed(bytes, encoded);
  }
  return bytes;
}

void sortRelationRows(std::vector<RelationRow> &rows) {
  llvm::sort(rows, [](const RelationRow &lhs, const RelationRow &rhs) {
    return encodeRelationRow(lhs) < encodeRelationRow(rhs);
  });
}

std::vector<std::uint8_t> encodeRelation(llvm::ArrayRef<RelationRow> rows) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, rows.size());
  for (const RelationRow &row : rows) {
    std::vector<std::uint8_t> encoded = encodeRelationRow(row);
    appendFramed(bytes, encoded);
  }
  return bytes;
}

llvm::Expected<std::vector<RelationRow>>
normalizeRelation(llvm::ArrayRef<RelationRow> rows, std::size_t field,
                  llvm::ArrayRef<bool> groupFiniteFields) {
  if (rows.empty())
    return invalid("relation partition must not be empty");
  const std::size_t fieldCount = rows.front().size();
  if (groupFiniteFields.size() != fieldCount)
    return invalid("relation field policy has the wrong size");
  for (const RelationRow &row : rows)
    if (row.size() != fieldCount)
      return invalid("relation rows have inconsistent field counts");
  if (field == fieldCount)
    return std::vector<RelationRow>{RelationRow{}};

  const std::size_t domainKind = rows.front()[field].index();
  for (const RelationRow &row : rows)
    if (row[field].index() != domainKind)
      return invalid("relation field mixes incompatible domain kinds");

  std::vector<RelationRow> normalized;
  if (domainKind == 0) {
    struct Cell {
      FiniteAtom atom;
      std::vector<RelationRow> rows;
    };
    std::map<std::vector<std::uint8_t>, Cell> cells;
    for (const RelationRow &row : rows) {
      for (const FiniteAtom &atom : std::get<FiniteDomain>(row[field]).atoms) {
        auto found = cells.find(atom.bytes);
        if (found == cells.end())
          found = cells.emplace(atom.bytes, Cell{atom, {}}).first;
        found->second.rows.push_back(row);
      }
    }

    struct Group {
      std::vector<FiniteAtom> atoms;
      std::vector<RelationRow> suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (auto &[atomBytes, cell] : cells) {
      auto suffix = normalizeRelation(cell.rows, field + 1, groupFiniteFields);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeRelation(*suffix);
      if (!groupFiniteFields[field])
        appendFramed(key, atomBytes);
      auto [group, inserted] = groups.try_emplace(key);
      if (inserted)
        group->second.suffix = *suffix;
      group->second.atoms.push_back(std::move(cell.atom));
    }

    for (auto &[key, group] : groups) {
      llvm::sort(group.atoms, [](const FiniteAtom &lhs, const FiniteAtom &rhs) {
        return lhs.bytes < rhs.bytes;
      });
      for (const RelationRow &suffix : group.suffix) {
        RelationRow row;
        row.reserve(1 + suffix.size());
        row.push_back(FiniteDomain{group.atoms});
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  } else {
    std::set<std::uint64_t> boundarySet;
    for (const RelationRow &row : rows)
      for (UnsignedInterval interval :
           std::get<UnsignedDomain>(row[field]).intervals()) {
        boundarySet.insert(interval.lower);
        if (interval.upper != std::numeric_limits<std::uint64_t>::max())
          boundarySet.insert(interval.upper + 1);
      }
    std::vector<std::uint64_t> boundaries(boundarySet.begin(),
                                          boundarySet.end());

    struct Group {
      std::vector<UnsignedInterval> intervals;
      std::vector<RelationRow> suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
      const std::uint64_t lower = boundaries[index];
      const std::uint64_t upper =
          index + 1 < boundaries.size()
              ? boundaries[index + 1] - 1
              : std::numeric_limits<std::uint64_t>::max();
      std::vector<RelationRow> selected;
      for (const RelationRow &row : rows)
        if (std::get<UnsignedDomain>(row[field]).contains(lower))
          selected.push_back(row);
      if (selected.empty())
        continue;
      auto suffix = normalizeRelation(selected, field + 1, groupFiniteFields);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeRelation(*suffix);
      auto [group, inserted] = groups.try_emplace(key);
      if (inserted)
        group->second.suffix = *suffix;
      group->second.intervals.push_back({lower, upper});
    }

    for (auto &[key, group] : groups) {
      auto domain = UnsignedDomain::normalize(group.intervals);
      if (!domain)
        return domain.takeError();
      for (const RelationRow &suffix : group.suffix) {
        RelationRow row;
        row.reserve(1 + suffix.size());
        row.push_back(*domain);
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  }

  sortRelationRows(normalized);
  return normalized;
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

template <typename Value>
llvm::Expected<std::vector<Value>> valuesOf(const RelationDomain &domain) {
  const auto *finite = std::get_if<FiniteDomain>(&domain);
  if (!finite)
    return invalid("canonical relation field has the wrong domain kind");
  std::vector<Value> values;
  values.reserve(finite->atoms.size());
  for (const FiniteAtom &atom : finite->atoms) {
    const Value *value = std::get_if<Value>(&atom.value);
    if (!value)
      return invalid("canonical relation atom has the wrong value kind");
    values.push_back(*value);
  }
  return values;
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
  for (std::size_t left = 0; left < rows.size(); ++left)
    for (std::size_t right = left + 1; right < rows.size(); ++right)
      if (relationRowsOverlap(rows[left], rows[right]))
        return invalid("parameterized memory access classes overlap");

  const bool groupFiniteFields[] = {false, true, true, true, true, true, true};
  auto normalized = normalizeRelation(rows, 0, groupFiniteFields);
  if (!normalized)
    return normalized.takeError();

  std::vector<MemoryAccessClass> result;
  result.reserve(normalized->size());
  for (const RelationRow &row : *normalized) {
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
  return encodeRelation(rows);
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
  for (const FiniteAtom &atom : canonicalMasks->atoms)
    sortedMasks.push_back(std::get<MaskInactivePair>(atom.value));

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

} // namespace fabric
