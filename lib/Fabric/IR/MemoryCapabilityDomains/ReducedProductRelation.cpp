#include "Fabric/IR/ReducedProductRelation.h"

#include "llvm/ADT/STLExtras.h"

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

std::vector<std::uint8_t> encodeDomain(const ReducedProductDomain &domain) {
  std::vector<std::uint8_t> bytes;
  if (const auto *finite = std::get_if<ReducedFiniteDomain>(&domain)) {
    appendU32(bytes, 0);
    appendU64(bytes, finite->atoms.size());
    for (const ReducedFiniteAtom &atom : finite->atoms)
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

std::vector<std::uint8_t> encodeRow(const ReducedProductRow &row) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, row.size());
  for (const ReducedProductDomain &domain : row) {
    std::vector<std::uint8_t> encoded = encodeDomain(domain);
    appendFramed(bytes, encoded);
  }
  return bytes;
}

void sortRows(std::vector<ReducedProductRow> &rows) {
  llvm::sort(rows,
             [](const ReducedProductRow &lhs, const ReducedProductRow &rhs) {
               return encodeRow(lhs) < encodeRow(rhs);
             });
}

llvm::Expected<std::vector<ReducedProductRow>>
reduceAtField(llvm::ArrayRef<ReducedProductRow> rows, std::size_t field,
              llvm::ArrayRef<bool> groupFiniteFields) {
  if (rows.empty())
    return invalid("relation partition must not be empty");
  const std::size_t fieldCount = rows.front().size();
  if (groupFiniteFields.size() != fieldCount)
    return invalid("relation field policy has the wrong size");
  for (const ReducedProductRow &row : rows)
    if (row.size() != fieldCount)
      return invalid("relation rows have inconsistent field counts");
  if (field == fieldCount)
    return std::vector<ReducedProductRow>{ReducedProductRow{}};

  const std::size_t domainKind = rows.front()[field].index();
  for (const ReducedProductRow &row : rows)
    if (row[field].index() != domainKind)
      return invalid("relation field mixes incompatible domain kinds");

  std::vector<ReducedProductRow> normalized;
  if (domainKind == 0) {
    struct Cell {
      ReducedFiniteAtom atom;
      std::vector<ReducedProductRow> rows;
    };
    std::map<std::vector<std::uint8_t>, Cell> cells;
    for (const ReducedProductRow &row : rows) {
      for (const ReducedFiniteAtom &atom :
           std::get<ReducedFiniteDomain>(row[field]).atoms) {
        auto found = cells.find(atom.bytes);
        if (found == cells.end())
          found = cells.emplace(atom.bytes, Cell{atom, {}}).first;
        found->second.rows.push_back(row);
      }
    }

    struct Group {
      std::vector<ReducedFiniteAtom> atoms;
      std::vector<ReducedProductRow> suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (auto &[atomBytes, cell] : cells) {
      auto suffix = reduceAtField(cell.rows, field + 1, groupFiniteFields);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeReducedProductRelation(*suffix);
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
      for (const ReducedProductRow &suffix : group.suffix) {
        ReducedProductRow row;
        row.reserve(1 + suffix.size());
        row.push_back(ReducedFiniteDomain{group.atoms});
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  } else {
    std::set<std::uint64_t> boundarySet;
    for (const ReducedProductRow &row : rows)
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
      std::vector<ReducedProductRow> suffix;
    };
    std::map<std::vector<std::uint8_t>, Group> groups;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
      const std::uint64_t lower = boundaries[index];
      const std::uint64_t upper =
          index + 1 < boundaries.size()
              ? boundaries[index + 1] - 1
              : std::numeric_limits<std::uint64_t>::max();
      std::vector<ReducedProductRow> selected;
      for (const ReducedProductRow &row : rows)
        if (std::get<UnsignedDomain>(row[field]).contains(lower))
          selected.push_back(row);
      if (selected.empty())
        continue;
      auto suffix = reduceAtField(selected, field + 1, groupFiniteFields);
      if (!suffix)
        return suffix.takeError();
      std::vector<std::uint8_t> key = encodeReducedProductRelation(*suffix);
      auto [group, inserted] = groups.try_emplace(key);
      if (inserted)
        group->second.suffix = *suffix;
      group->second.intervals.push_back({lower, upper});
    }

    for (auto &[key, group] : groups) {
      auto domain = UnsignedDomain::normalize(group.intervals);
      if (!domain)
        return domain.takeError();
      for (const ReducedProductRow &suffix : group.suffix) {
        ReducedProductRow row;
        row.reserve(1 + suffix.size());
        row.push_back(*domain);
        row.insert(row.end(), suffix.begin(), suffix.end());
        normalized.push_back(std::move(row));
      }
    }
  }

  sortRows(normalized);
  return normalized;
}

} // namespace

llvm::Expected<std::vector<ReducedProductRow>>
reduceProductRelation(llvm::ArrayRef<ReducedProductRow> rows,
                      llvm::ArrayRef<bool> groupFiniteFields) {
  if (rows.empty())
    return invalid("reduced product relation must not be empty");
  for (std::size_t left = 0; left < rows.size(); ++left)
    for (std::size_t right = left + 1; right < rows.size(); ++right)
      if (relationRowsOverlap(rows[left], rows[right]))
        return invalid("reduced product relation rows overlap");
  return reduceAtField(rows, 0, groupFiniteFields);
}

std::vector<std::uint8_t>
encodeReducedProductRelation(llvm::ArrayRef<ReducedProductRow> rows) {
  std::vector<std::uint8_t> bytes;
  appendU64(bytes, rows.size());
  for (const ReducedProductRow &row : rows) {
    std::vector<std::uint8_t> encoded = encodeRow(row);
    appendFramed(bytes, encoded);
  }
  return bytes;
}

} // namespace fabric::detail
