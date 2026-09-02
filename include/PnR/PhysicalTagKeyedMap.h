#ifndef LOOM_PNR_PHYSICALTAGKEYEDMAP_H
#define LOOM_PNR_PHYSICALTAGKEYEDMAP_H

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace loom::pnr {

/// Sorted-by-value flat map keyed by one Physical Tag value. Tag-keyed
/// domains hold few distinct values, so ordered flat storage beats hashing
/// and string keys and keeps iteration canonical in the Fabric tag order.
/// The entry shape mirrors the map interface its consumers use; equality is
/// the entry-wise Fabric tag comparison plus value equality.
template <typename Value> class PhysicalTagKeyedMap final {
public:
  using Entry = std::pair<llvm::APInt, Value>;
  using iterator = typename std::vector<Entry>::iterator;
  using const_iterator = typename std::vector<Entry>::const_iterator;

  iterator begin() { return entries_.begin(); }
  iterator end() { return entries_.end(); }
  const_iterator begin() const { return entries_.begin(); }
  const_iterator end() const { return entries_.end(); }
  std::size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

  iterator find(const llvm::APInt &value) {
    const auto found = lowerBound(value);
    if (found != entries_.end() &&
        ::fabric::comparePhysicalTagValues(found->first, value) == 0)
      return found;
    return entries_.end();
  }
  const_iterator find(const llvm::APInt &value) const {
    return const_cast<PhysicalTagKeyedMap *>(this)->find(value);
  }
  Value &operator[](const llvm::APInt &value) {
    auto found = lowerBound(value);
    if (found == entries_.end() ||
        ::fabric::comparePhysicalTagValues(found->first, value) != 0)
      found = entries_.insert(found, {value, Value{}});
    return found->second;
  }
  void erase(iterator entry) { entries_.erase(entry); }

  /// Bytes retained by the entry storage, including the heap words of tag
  /// values wider than one machine word.
  std::size_t retainedBytes() const {
    std::size_t bytes = entries_.capacity() * sizeof(Entry);
    for (const Entry &entry : entries_)
      if (!entry.first.isSingleWord())
        bytes += entry.first.getNumWords() * sizeof(std::uint64_t);
    return bytes;
  }

  friend bool operator==(const PhysicalTagKeyedMap &lhs,
                         const PhysicalTagKeyedMap &rhs) {
    return std::equal(lhs.entries_.begin(), lhs.entries_.end(),
                      rhs.entries_.begin(), rhs.entries_.end(),
                      [](const Entry &left, const Entry &right) {
                        return ::fabric::comparePhysicalTagValues(
                                   left.first, right.first) == 0 &&
                               left.second == right.second;
                      });
  }
  friend bool operator!=(const PhysicalTagKeyedMap &lhs,
                         const PhysicalTagKeyedMap &rhs) {
    return !(lhs == rhs);
  }

private:
  iterator lowerBound(const llvm::APInt &value) {
    return std::lower_bound(entries_.begin(), entries_.end(), value,
                            [](const Entry &entry, const llvm::APInt &target) {
                              return ::fabric::comparePhysicalTagValues(
                                         entry.first, target) < 0;
                            });
  }

  std::vector<Entry> entries_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_PHYSICALTAGKEYEDMAP_H
