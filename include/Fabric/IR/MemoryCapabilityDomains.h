#ifndef LOOM_FABRIC_IR_MEMORY_CAPABILITY_DOMAINS_H
#define LOOM_FABRIC_IR_MEMORY_CAPABILITY_DOMAINS_H

#include "Dataflow/IR/DataflowServiceSchema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace fabric {

struct UnsignedInterval {
  std::uint64_t lower;
  std::uint64_t upper;

  friend bool operator==(UnsignedInterval lhs, UnsignedInterval rhs) {
    return lhs.lower == rhs.lower && lhs.upper == rhs.upper;
  }
  friend bool operator!=(UnsignedInterval lhs, UnsignedInterval rhs) {
    return !(lhs == rhs);
  }
};

class UnsignedDomain {
public:
  static llvm::Expected<UnsignedDomain>
  normalize(llvm::ArrayRef<UnsignedInterval> intervals);

  static llvm::Expected<UnsignedDomain>
  fromCanonical(llvm::ArrayRef<UnsignedInterval> intervals);

  llvm::ArrayRef<UnsignedInterval> intervals() const { return intervals_; }
  bool contains(std::uint64_t value) const;

private:
  explicit UnsignedDomain(std::vector<UnsignedInterval> intervals)
      : intervals_(std::move(intervals)) {}

  std::vector<UnsignedInterval> intervals_;
};

class AlignmentDomain {
public:
  static llvm::Expected<AlignmentDomain> create(UnsignedDomain exponents);

  const UnsignedDomain &exponents() const { return exponents_; }
  bool containsBytes(std::uint64_t bytes) const;

private:
  explicit AlignmentDomain(UnsignedDomain exponents)
      : exponents_(std::move(exponents)) {}

  UnsignedDomain exponents_;
};

enum class ReadSubwordSemantics : std::uint8_t {
  NotApplicable,
  Exact,
  ZeroExtend,
};

enum class WriteSubwordSemantics : std::uint8_t {
  NotApplicable,
  Exact,
  ByteEnable,
};

enum class InactiveLaneSemantics : std::uint8_t {
  NotApplicable,
  Suppress,
  SuppressAndZeroFill,
};

std::uint8_t getCanonicalTag(ReadSubwordSemantics semantics);
std::uint8_t getCanonicalTag(WriteSubwordSemantics semantics);
std::uint8_t getCanonicalTag(InactiveLaneSemantics semantics);

llvm::Expected<ReadSubwordSemantics>
decodeReadSubwordSemantics(std::uint8_t tag);
llvm::Expected<WriteSubwordSemantics>
decodeWriteSubwordSemantics(std::uint8_t tag);
llvm::Expected<InactiveLaneSemantics>
decodeInactiveLaneSemantics(std::uint8_t tag);

namespace detail {

template <typename Enum> struct ClosedEnumTraits;

template <> struct ClosedEnumTraits<ReadSubwordSemantics> {
  static llvm::Expected<std::uint8_t> tag(ReadSubwordSemantics value);
};

template <> struct ClosedEnumTraits<WriteSubwordSemantics> {
  static llvm::Expected<std::uint8_t> tag(WriteSubwordSemantics value);
};

template <> struct ClosedEnumTraits<InactiveLaneSemantics> {
  static llvm::Expected<std::uint8_t> tag(InactiveLaneSemantics value);
};

llvm::Error domainError(const char *message);

} // namespace detail

template <typename Enum> class ClosedEnumDomain {
public:
  static llvm::Expected<ClosedEnumDomain>
  normalize(llvm::ArrayRef<Enum> values) {
    if (values.empty())
      return detail::domainError("closed enum domain must not be empty");

    std::vector<std::pair<std::uint8_t, Enum>> tagged;
    tagged.reserve(values.size());
    for (Enum value : values) {
      llvm::Expected<std::uint8_t> tag =
          detail::ClosedEnumTraits<Enum>::tag(value);
      if (!tag)
        return tag.takeError();
      tagged.emplace_back(*tag, value);
    }
    std::sort(
        tagged.begin(), tagged.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

    std::vector<Enum> normalizedValues;
    std::vector<std::uint8_t> normalizedTags;
    normalizedValues.reserve(tagged.size());
    normalizedTags.reserve(tagged.size());
    for (const auto &[tag, value] : tagged) {
      if (!normalizedTags.empty() && normalizedTags.back() == tag) {
        if (normalizedValues.back() != value)
          return detail::domainError(
              "closed enum codec assigns one tag to multiple values");
        continue;
      }
      normalizedValues.push_back(value);
      normalizedTags.push_back(tag);
    }
    return ClosedEnumDomain(std::move(normalizedValues),
                            std::move(normalizedTags));
  }

  static llvm::Expected<ClosedEnumDomain>
  fromCanonical(llvm::ArrayRef<Enum> values) {
    if (values.empty())
      return detail::domainError("closed enum domain must not be empty");

    std::vector<std::uint8_t> tags;
    tags.reserve(values.size());
    std::uint8_t previous = 0;
    bool hasPrevious = false;
    for (Enum value : values) {
      llvm::Expected<std::uint8_t> tag =
          detail::ClosedEnumTraits<Enum>::tag(value);
      if (!tag)
        return tag.takeError();
      if (hasPrevious && *tag <= previous)
        return detail::domainError(
            "closed enum domain is not sorted and unique");
      previous = *tag;
      hasPrevious = true;
      tags.push_back(*tag);
    }
    return ClosedEnumDomain(std::vector<Enum>(values.begin(), values.end()),
                            std::move(tags));
  }

  llvm::ArrayRef<Enum> values() const { return values_; }

  bool contains(Enum value) const {
    llvm::Expected<std::uint8_t> tag =
        detail::ClosedEnumTraits<Enum>::tag(value);
    if (!tag) {
      llvm::consumeError(tag.takeError());
      return false;
    }
    return std::binary_search(tags_.begin(), tags_.end(), *tag);
  }

private:
  ClosedEnumDomain(std::vector<Enum> values, std::vector<std::uint8_t> tags)
      : values_(std::move(values)), tags_(std::move(tags)) {}

  std::vector<Enum> values_;
  std::vector<std::uint8_t> tags_;
};

struct MaskInactivePair {
  dataflow::semantics::MemoryMaskForm mask;
  InactiveLaneSemantics inactive;

  friend bool operator==(MaskInactivePair lhs, MaskInactivePair rhs) {
    return lhs.mask == rhs.mask && lhs.inactive == rhs.inactive;
  }
};

class MemoryAccessClass {
public:
  static llvm::Expected<MemoryAccessClass>
  create(dataflow::semantics::MemoryAccessForm accessForm,
         UnsignedDomain elementWidths, UnsignedDomain flattenedLaneCounts,
         llvm::ArrayRef<MaskInactivePair> maskInactivePairs,
         AlignmentDomain sourceAlignments,
         ClosedEnumDomain<ReadSubwordSemantics> readSubword,
         ClosedEnumDomain<WriteSubwordSemantics> writeSubword);

  dataflow::semantics::MemoryAccessForm accessForm() const {
    return accessForm_;
  }
  const UnsignedDomain &elementWidths() const { return elementWidths_; }
  const UnsignedDomain &flattenedLaneCounts() const {
    return flattenedLaneCounts_;
  }
  llvm::ArrayRef<MaskInactivePair> maskInactivePairs() const {
    return maskInactivePairs_;
  }
  const AlignmentDomain &sourceAlignments() const { return sourceAlignments_; }
  const ClosedEnumDomain<ReadSubwordSemantics> &readSubwordSemantics() const {
    return readSubword_;
  }
  const ClosedEnumDomain<WriteSubwordSemantics> &writeSubwordSemantics() const {
    return writeSubword_;
  }

  bool
  contains(const dataflow::semantics::CanonicalMemoryAccessView &access) const;

private:
  MemoryAccessClass(dataflow::semantics::MemoryAccessForm accessForm,
                    UnsignedDomain elementWidths,
                    UnsignedDomain flattenedLaneCounts,
                    std::vector<MaskInactivePair> maskInactivePairs,
                    AlignmentDomain sourceAlignments,
                    ClosedEnumDomain<ReadSubwordSemantics> readSubword,
                    ClosedEnumDomain<WriteSubwordSemantics> writeSubword)
      : accessForm_(accessForm), elementWidths_(std::move(elementWidths)),
        flattenedLaneCounts_(std::move(flattenedLaneCounts)),
        maskInactivePairs_(std::move(maskInactivePairs)),
        sourceAlignments_(std::move(sourceAlignments)),
        readSubword_(std::move(readSubword)),
        writeSubword_(std::move(writeSubword)) {}

  dataflow::semantics::MemoryAccessForm accessForm_;
  UnsignedDomain elementWidths_;
  UnsignedDomain flattenedLaneCounts_;
  std::vector<MaskInactivePair> maskInactivePairs_;
  AlignmentDomain sourceAlignments_;
  ClosedEnumDomain<ReadSubwordSemantics> readSubword_;
  ClosedEnumDomain<WriteSubwordSemantics> writeSubword_;
};

class ParameterizedMemoryAccessDomain {
public:
  /// Normalizes an authoring relation into the unique reduced ordered product
  /// relation owned by the Fabric memory capability schema.
  static llvm::Expected<ParameterizedMemoryAccessDomain>
  create(llvm::ArrayRef<MemoryAccessClass> accessClasses);

  /// Strictly imports an already canonical relation. Equivalent but split,
  /// reordered, or otherwise noncanonical products are rejected.
  static llvm::Expected<ParameterizedMemoryAccessDomain>
  fromCanonical(llvm::ArrayRef<MemoryAccessClass> accessClasses);

  const MemoryAccessClass *matchingClass(
      const dataflow::semantics::CanonicalMemoryAccessView &access) const;

  llvm::ArrayRef<MemoryAccessClass> accessClasses() const {
    return accessClasses_;
  }

  bool
  contains(const dataflow::semantics::CanonicalMemoryAccessView &access) const {
    return matchingClass(access) != nullptr;
  }

private:
  explicit ParameterizedMemoryAccessDomain(
      std::vector<MemoryAccessClass> accessClasses)
      : accessClasses_(std::move(accessClasses)) {}

  std::vector<MemoryAccessClass> accessClasses_;
};

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_CAPABILITY_DOMAINS_H
