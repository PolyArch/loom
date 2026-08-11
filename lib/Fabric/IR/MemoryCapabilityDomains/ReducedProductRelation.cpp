#include "Fabric/IR/ReducedProductRelation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <system_error>

namespace fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  const std::size_t offset = bytes.size();
  bytes.resize(offset + 4);
  for (unsigned index = 0; index < 4; ++index)
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (8 * (3 - index)));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  const std::size_t offset = bytes.size();
  bytes.resize(offset + 8);
  for (unsigned index = 0; index < 8; ++index)
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (8 * (7 - index)));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64(bytes, field.size());
  bytes.insert(bytes.end(), field.begin(), field.end());
}

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> readU32(const llvm::Twine &field) {
    if (remaining() < 4)
      return invalid(field + " is truncated");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> readU64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  readFrame(const llvm::Twine &field) {
    auto size = readU64(field + " length");
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid(field + " is truncated");
    llvm::ArrayRef<std::uint8_t> result = bytes_.slice(offset_, *size);
    offset_ += *size;
    return result;
  }

  llvm::Error finish(const llvm::Twine &record) const {
    if (remaining() != 0)
      return invalid(record + " has trailing bytes");
    return llvm::Error::success();
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<ReducedProductDomain>
decodeDomain(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto tag = reader.readU32("relation domain tag");
  if (!tag)
    return tag.takeError();
  auto count = reader.readU64("relation domain element count");
  if (!count)
    return count.takeError();
  if (*count == 0)
    return invalid("relation domain must not be empty");

  if (*tag == 0) {
    if (*count > reader.remaining() / 8)
      return invalid("finite relation domain has an invalid atom count");
    std::vector<ReducedFiniteAtom> atoms;
    atoms.reserve(*count);
    for (std::uint64_t index = 0; index < *count; ++index) {
      auto atom = reader.readFrame("finite relation atom");
      if (!atom)
        return atom.takeError();
      if (!atoms.empty() &&
          !(llvm::ArrayRef<std::uint8_t>(atoms.back().bytes) < *atom))
        return invalid("finite relation atoms are not sorted and unique");
      atoms.push_back({std::vector<std::uint8_t>(atom->begin(), atom->end())});
    }
    if (llvm::Error error = reader.finish("finite relation domain"))
      return std::move(error);
    return ReducedProductDomain(ReducedFiniteDomain{std::move(atoms)});
  }

  if (*tag == 1) {
    if (*count > reader.remaining() / 16)
      return invalid("unsigned relation domain has an invalid interval count");
    std::vector<UnsignedInterval> intervals;
    intervals.reserve(*count);
    for (std::uint64_t index = 0; index < *count; ++index) {
      auto lower = reader.readU64("unsigned interval lower bound");
      if (!lower)
        return lower.takeError();
      auto upper = reader.readU64("unsigned interval upper bound");
      if (!upper)
        return upper.takeError();
      intervals.push_back({*lower, *upper});
    }
    if (llvm::Error error = reader.finish("unsigned relation domain"))
      return std::move(error);
    auto domain = UnsignedDomain::fromCanonical(intervals);
    if (!domain)
      return domain.takeError();
    return ReducedProductDomain(std::move(*domain));
  }

  return invalid("unknown reduced relation domain tag");
}

llvm::Expected<ReducedProductRow>
decodeRow(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto count = reader.readU64("relation row field count");
  if (!count)
    return count.takeError();
  if (*count == 0 || *count > reader.remaining() / 8)
    return invalid("relation row has an invalid field count");
  ReducedProductRow row;
  row.reserve(*count);
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto domainBytes = reader.readFrame("relation field domain");
    if (!domainBytes)
      return domainBytes.takeError();
    auto domain = decodeDomain(*domainBytes);
    if (!domain)
      return domain.takeError();
    row.push_back(std::move(*domain));
  }
  if (llvm::Error error = reader.finish("relation row"))
    return std::move(error);
  return row;
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

bool finiteDomainsIntersect(const ReducedFiniteDomain &lhs,
                            const ReducedFiniteDomain &rhs) {
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

bool relationDomainsIntersect(const ReducedProductDomain &lhs,
                              const ReducedProductDomain &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<ReducedFiniteDomain>(&lhs))
    return finiteDomainsIntersect(*left, std::get<ReducedFiniteDomain>(rhs));
  return domainsIntersect(std::get<UnsignedDomain>(lhs),
                          std::get<UnsignedDomain>(rhs));
}

bool relationRowsOverlap(const ReducedProductRow &lhs,
                         const ReducedProductRow &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (!relationDomainsIntersect(left, right))
      return false;
  return true;
}

llvm::Expected<bool>
rowCoveredFromField(const ReducedProductRow &subset,
                    llvm::ArrayRef<ReducedProductRow> candidates,
                    std::size_t field) {
  if (field == subset.size())
    return !candidates.empty();

  if (candidates.empty())
    return false;
  std::vector<ReducedProductRow> compatible;
  compatible.reserve(candidates.size());
  for (const ReducedProductRow &candidate : candidates) {
    if (candidate.size() != subset.size())
      return invalid("relation coverage compares incompatible row shapes");
    if (candidate[field].index() == subset[field].index())
      compatible.push_back(candidate);
  }
  if (compatible.empty())
    return false;

  if (const auto *finite = std::get_if<ReducedFiniteDomain>(&subset[field])) {
    for (const ReducedFiniteAtom &atom : finite->atoms) {
      std::vector<ReducedProductRow> selected;
      for (const ReducedProductRow &candidate : compatible) {
        const auto &domain = std::get<ReducedFiniteDomain>(candidate[field]);
        if (llvm::any_of(domain.atoms, [&](const ReducedFiniteAtom &value) {
              return value.bytes == atom.bytes;
            }))
          selected.push_back(candidate);
      }
      auto covered = rowCoveredFromField(subset, selected, field + 1);
      if (!covered || !*covered)
        return covered;
    }
    return true;
  }

  const UnsignedDomain &domain = std::get<UnsignedDomain>(subset[field]);
  for (UnsignedInterval interval : domain.intervals()) {
    std::vector<std::uint64_t> boundaries{interval.lower};
    if (interval.upper != std::numeric_limits<std::uint64_t>::max())
      boundaries.push_back(interval.upper + 1);
    for (const ReducedProductRow &candidate : compatible) {
      for (UnsignedInterval accepted :
           std::get<UnsignedDomain>(candidate[field]).intervals()) {
        if (accepted.upper < interval.lower || accepted.lower > interval.upper)
          continue;
        boundaries.push_back(std::max(accepted.lower, interval.lower));
        const std::uint64_t clippedUpper =
            std::min(accepted.upper, interval.upper);
        if (clippedUpper != std::numeric_limits<std::uint64_t>::max())
          boundaries.push_back(clippedUpper + 1);
      }
    }
    llvm::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                     boundaries.end());
    for (std::uint64_t lower : boundaries) {
      if (lower < interval.lower || lower > interval.upper)
        continue;
      std::vector<ReducedProductRow> selected;
      for (const ReducedProductRow &candidate : compatible)
        if (std::get<UnsignedDomain>(candidate[field]).contains(lower))
          selected.push_back(candidate);
      auto covered = rowCoveredFromField(subset, selected, field + 1);
      if (!covered || !*covered)
        return covered;
    }
  }
  return true;
}

std::size_t encodedDomainSize(const ReducedProductDomain &domain) {
  if (const auto *finite = std::get_if<ReducedFiniteDomain>(&domain)) {
    std::size_t size = 4 + 8;
    for (const ReducedFiniteAtom &atom : finite->atoms)
      size += 8 + atom.bytes.size();
    return size;
  }
  return 4 + 8 + 16 * std::get<UnsignedDomain>(domain).intervals().size();
}

void appendDomain(std::vector<std::uint8_t> &bytes,
                  const ReducedProductDomain &domain) {
  if (const auto *finite = std::get_if<ReducedFiniteDomain>(&domain)) {
    appendU32(bytes, 0);
    appendU64(bytes, finite->atoms.size());
    for (const ReducedFiniteAtom &atom : finite->atoms)
      appendFramed(bytes, atom.bytes);
    return;
  }
  const UnsignedDomain &unsignedDomain = std::get<UnsignedDomain>(domain);
  appendU32(bytes, 1);
  appendU64(bytes, unsignedDomain.intervals().size());
  for (UnsignedInterval interval : unsignedDomain.intervals()) {
    appendU64(bytes, interval.lower);
    appendU64(bytes, interval.upper);
  }
}

std::size_t encodedRowSize(const ReducedProductRow &row) {
  std::size_t size = 8;
  for (const ReducedProductDomain &domain : row)
    size += 8 + encodedDomainSize(domain);
  return size;
}

void appendRow(std::vector<std::uint8_t> &bytes, const ReducedProductRow &row) {
  appendU64(bytes, row.size());
  for (const ReducedProductDomain &domain : row) {
    appendU64(bytes, encodedDomainSize(domain));
    appendDomain(bytes, domain);
  }
}

std::vector<std::uint8_t> encodeRow(const ReducedProductRow &row) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(encodedRowSize(row));
  appendRow(bytes, row);
  return bytes;
}

void sortRows(std::vector<ReducedProductRow> &rows) {
  using KeyedRow = std::pair<std::vector<std::uint8_t>, ReducedProductRow>;
  std::vector<KeyedRow> keyedRows;
  keyedRows.reserve(rows.size());
  for (ReducedProductRow &row : rows)
    keyedRows.emplace_back(encodeRow(row), std::move(row));
  llvm::sort(keyedRows, [](const KeyedRow &lhs, const KeyedRow &rhs) {
    return lhs.first < rhs.first;
  });
  for (auto [destination, keyed] : llvm::zip(rows, keyedRows))
    destination = std::move(keyed.second);
}

using ReducedProductRowRef = const ReducedProductRow *;
using ReducedProductRows = std::vector<ReducedProductRow>;
using ReducedProductReduction = const ReducedProductRows *;

struct ReductionKey {
  std::size_t field = 0;
  std::vector<ReducedProductRowRef> rows;
};

struct ReductionKeyLess {
  bool operator()(const ReductionKey &lhs, const ReductionKey &rhs) const {
    if (lhs.field != rhs.field)
      return lhs.field < rhs.field;
    return std::lexicographical_compare(lhs.rows.begin(), lhs.rows.end(),
                                        rhs.rows.begin(), rhs.rows.end(),
                                        std::less<ReducedProductRowRef>());
  }
};

using ReductionCache =
    std::map<ReductionKey, ReducedProductRows, ReductionKeyLess>;

llvm::Expected<ReducedProductReduction>
reduceAtField(llvm::ArrayRef<ReducedProductRowRef> rows, std::size_t field,
              llvm::ArrayRef<bool> groupFiniteFields, ReductionCache &cache) {
  if (rows.empty())
    return invalid("relation partition must not be empty");
  ReductionKey cacheKey{field, {rows.begin(), rows.end()}};
  if (auto found = cache.find(cacheKey); found != cache.end())
    return &found->second;
  auto publish = [&](ReducedProductRows result) {
    auto position = cache.emplace(std::move(cacheKey), std::move(result)).first;
    return &position->second;
  };

  const std::size_t fieldCount = rows.front()->size();
  if (field == fieldCount)
    return publish({ReducedProductRow{}});

  if (rows.size() == 1) {
    bool canonicalSingleton = true;
    for (std::size_t index = field; index < fieldCount; ++index) {
      const auto *finite =
          std::get_if<ReducedFiniteDomain>(&(*rows.front())[index]);
      if (finite && !groupFiniteFields[index] && finite->atoms.size() != 1) {
        canonicalSingleton = false;
        break;
      }
    }
    if (canonicalSingleton)
      return publish({ReducedProductRow(rows.front()->begin() + field,
                                        rows.front()->end())});
  }

  const std::size_t domainKind = (*rows.front())[field].index();
  std::map<std::size_t, std::vector<ReducedProductRowRef>> kindPartitions;
  for (ReducedProductRowRef row : rows)
    kindPartitions[(*row)[field].index()].push_back(row);
  if (kindPartitions.size() != 1) {
    ReducedProductRows partitioned;
    for (const auto &[kind, partition] : kindPartitions) {
      auto reduced = reduceAtField(partition, field, groupFiniteFields, cache);
      if (!reduced)
        return reduced.takeError();
      partitioned.insert(partitioned.end(), (*reduced)->begin(),
                         (*reduced)->end());
    }
    sortRows(partitioned);
    return publish(std::move(partitioned));
  }

  ReducedProductRows normalized;
  if (domainKind == 0) {
    struct Cell {
      ReducedFiniteAtom atom;
      std::vector<ReducedProductRowRef> rows;
    };
    std::map<std::vector<std::uint8_t>, Cell> cells;
    for (ReducedProductRowRef row : rows) {
      for (const ReducedFiniteAtom &atom :
           std::get<ReducedFiniteDomain>((*row)[field]).atoms) {
        auto found = cells.find(atom.bytes);
        if (found == cells.end())
          found = cells.emplace(atom.bytes, Cell{atom, {}}).first;
        found->second.rows.push_back(row);
      }
    }

    struct Group {
      std::vector<ReducedFiniteAtom> atoms;
      ReducedProductReduction suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (auto &[atomBytes, cell] : cells) {
      auto suffix =
          reduceAtField(cell.rows, field + 1, groupFiniteFields, cache);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeReducedProductRelation(**suffix);
      if (!groupFiniteFields[field])
        appendFramed(key, atomBytes);
      auto [group, inserted] = groups.try_emplace(key);
      if (inserted)
        group->second.suffix = *suffix;
      group->second.atoms.push_back(std::move(cell.atom));
    }

    for (auto &[key, group] : groups) {
      llvm::sort(group.atoms, [](const ReducedFiniteAtom &lhs,
                                 const ReducedFiniteAtom &rhs) {
        return lhs.bytes < rhs.bytes;
      });
      for (const ReducedProductRow &suffix : *group.suffix) {
        ReducedProductRow row;
        row.reserve(1 + suffix.size());
        row.push_back(ReducedFiniteDomain{group.atoms});
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  } else {
    std::set<std::uint64_t> boundarySet;
    for (ReducedProductRowRef row : rows)
      for (UnsignedInterval interval :
           std::get<UnsignedDomain>((*row)[field]).intervals()) {
        boundarySet.insert(interval.lower);
        if (interval.upper != std::numeric_limits<std::uint64_t>::max())
          boundarySet.insert(interval.upper + 1);
      }
    std::vector<std::uint64_t> boundaries(boundarySet.begin(),
                                          boundarySet.end());

    struct Group {
      std::vector<UnsignedInterval> intervals;
      ReducedProductReduction suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
      const std::uint64_t lower = boundaries[index];
      const std::uint64_t upper =
          index + 1 < boundaries.size()
              ? boundaries[index + 1] - 1
              : std::numeric_limits<std::uint64_t>::max();
      std::vector<ReducedProductRowRef> selected;
      for (ReducedProductRowRef row : rows)
        if (std::get<UnsignedDomain>((*row)[field]).contains(lower))
          selected.push_back(row);
      if (selected.empty())
        continue;
      auto suffix =
          reduceAtField(selected, field + 1, groupFiniteFields, cache);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeReducedProductRelation(**suffix);
      auto [group, inserted] = groups.try_emplace(key);
      if (inserted)
        group->second.suffix = *suffix;
      group->second.intervals.push_back({lower, upper});
    }

    for (auto &[key, group] : groups) {
      auto domain = UnsignedDomain::normalize(group.intervals);
      if (!domain)
        return domain.takeError();
      for (const ReducedProductRow &suffix : *group.suffix) {
        ReducedProductRow row;
        row.reserve(1 + suffix.size());
        row.push_back(*domain);
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  }

  sortRows(normalized);
  return publish(std::move(normalized));
}

} // namespace

llvm::Expected<std::vector<ReducedProductRow>>
reduceProductRelation(llvm::ArrayRef<ReducedProductRow> rows,
                      llvm::ArrayRef<bool> groupFiniteFields) {
  if (rows.empty())
    return invalid("reduced product relation must not be empty");
  const std::size_t fieldCount = rows.front().size();
  if (groupFiniteFields.size() != fieldCount)
    return invalid("relation field policy has the wrong size");
  for (const ReducedProductRow &row : rows)
    if (row.size() != fieldCount)
      return invalid("relation rows have inconsistent field counts");
  for (std::size_t left = 0; left < rows.size(); ++left)
    for (std::size_t right = left + 1; right < rows.size(); ++right)
      if (relationRowsOverlap(rows[left], rows[right]))
        return invalid("reduced product relation rows overlap");
  std::vector<ReducedProductRowRef> rowRefs;
  rowRefs.reserve(rows.size());
  for (const ReducedProductRow &row : rows)
    rowRefs.push_back(&row);
  ReductionCache cache;
  auto reduced = reduceAtField(rowRefs, 0, groupFiniteFields, cache);
  if (!reduced)
    return reduced.takeError();
  return **reduced;
}

llvm::Expected<bool>
reducedProductRelationCovers(llvm::ArrayRef<ReducedProductRow> superset,
                             llvm::ArrayRef<ReducedProductRow> subset) {
  if (superset.empty() || subset.empty())
    return invalid("relation coverage requires two nonempty relations");
  const std::size_t fieldCount = superset.front().size();
  if (fieldCount == 0)
    return invalid("relation coverage requires nonempty product rows");
  auto validateShape = [&](llvm::ArrayRef<ReducedProductRow> rows) {
    for (const ReducedProductRow &row : rows)
      if (row.size() != fieldCount)
        return false;
    return true;
  };
  if (!validateShape(superset) || !validateShape(subset))
    return invalid("relation coverage compares incompatible relations");

  for (const ReducedProductRow &row : subset) {
    auto covered = rowCoveredFromField(row, superset, 0);
    if (!covered || !*covered)
      return covered;
  }
  return true;
}

llvm::Expected<bool>
reducedProductRelationsOverlap(llvm::ArrayRef<ReducedProductRow> left,
                               llvm::ArrayRef<ReducedProductRow> right) {
  if (left.empty() || right.empty())
    return invalid("relation overlap requires two nonempty relations");
  const std::size_t fieldCount = left.front().size();
  if (fieldCount == 0)
    return invalid("relation overlap requires nonempty product rows");
  for (const ReducedProductRow &row : left) {
    if (row.size() != fieldCount)
      return invalid("left relation has inconsistent row shapes");
  }
  for (const ReducedProductRow &row : right) {
    if (row.size() != fieldCount)
      return invalid("relation overlap compares incompatible row shapes");
  }
  for (const ReducedProductRow &lhs : left)
    for (const ReducedProductRow &rhs : right)
      if (relationRowsOverlap(lhs, rhs))
        return true;
  return false;
}

std::vector<std::uint8_t>
encodeReducedProductRelation(llvm::ArrayRef<ReducedProductRow> rows) {
  std::size_t size = 8;
  for (const ReducedProductRow &row : rows)
    size += 8 + encodedRowSize(row);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(size);
  appendU64(bytes, rows.size());
  for (const ReducedProductRow &row : rows) {
    appendU64(bytes, encodedRowSize(row));
    appendRow(bytes, row);
  }
  return bytes;
}

llvm::Expected<std::vector<ReducedProductRow>>
decodeReducedProductRelation(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto count = reader.readU64("reduced relation row count");
  if (!count)
    return count.takeError();
  if (*count == 0 || *count > reader.remaining() / 8)
    return invalid("reduced relation has an invalid row count");

  std::vector<ReducedProductRow> rows;
  rows.reserve(*count);
  std::vector<std::uint8_t> previousBytes;
  std::size_t fieldCount = 0;
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto rowBytes = reader.readFrame("reduced relation row");
    if (!rowBytes)
      return rowBytes.takeError();
    if (!previousBytes.empty() &&
        !(llvm::ArrayRef<std::uint8_t>(previousBytes) < *rowBytes))
      return invalid("reduced relation rows are not sorted and unique");
    auto row = decodeRow(*rowBytes);
    if (!row)
      return row.takeError();
    if (rows.empty())
      fieldCount = row->size();
    else if (row->size() != fieldCount)
      return invalid("reduced relation rows have inconsistent field counts");
    previousBytes.assign(rowBytes->begin(), rowBytes->end());
    rows.push_back(std::move(*row));
  }
  if (llvm::Error error = reader.finish("reduced relation"))
    return std::move(error);
  return rows;
}

} // namespace fabric::detail
