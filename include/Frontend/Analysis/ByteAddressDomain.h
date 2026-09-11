#ifndef LOOM_FRONTEND_ANALYSIS_BYTEADDRESSDOMAIN_H
#define LOOM_FRONTEND_ANALYSIS_BYTEADDRESSDOMAIN_H

#include "Frontend/Analysis/MemoryAddressProjection.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <set>

namespace loom::frontend::analysis {

/// One arithmetic progression of byte offsets: `first`, `first + stride`, ...,
/// up to and including `last`. A single offset has `first == last`.
struct ByteOffsetProgression final {
  std::int64_t first = 0;
  std::int64_t last = 0;
  std::int64_t stride = 1;

  bool isSingleOffset() const { return first == last; }
  /// Number of offsets, saturating at the unsigned maximum.
  std::uint64_t count() const;
};

/// Byte offsets one address may denote inside a single memory object. The
/// domain always contains every offset a defined execution can reach;
/// `isExact` additionally states that it contains nothing else. A fully static
/// address is one single-offset progression, so static geometry keeps the
/// exact finite-set behaviour of a plain offset enumeration.
class ByteAddressDomain final {
public:
  /// Progressions retained before a union widens to one progression that still
  /// contains every offset. This bounds the representation, not the number of
  /// offsets; `enumerate` bounds those separately.
  static constexpr unsigned maximumProgressions = 512;

  ByteAddressDomain() = default;

  static ByteAddressDomain singleOffset(std::int64_t offset);
  /// Every `first + stride * n` in `[first, last]`. The stride must be
  /// positive and `last` is lowered to the last reachable offset.
  static std::optional<ByteAddressDomain>
  progression(std::int64_t first, std::int64_t last, std::int64_t stride);
  /// Exact domain of an explicit non-empty offset set.
  static std::optional<ByteAddressDomain>
  explicitOffsets(const std::set<std::int64_t> &offsets);

  bool isEmpty() const { return progressions_.empty(); }
  bool isExact() const { return exact_; }
  std::int64_t lowest() const;
  std::int64_t highest() const;

  /// Ascending distinct offsets, when at most `limit` of them exist.
  std::optional<llvm::SmallVector<std::int64_t>>
  enumerate(std::uint64_t limit) const;

  /// Replaces every offset `value` by `value * byteStride`.
  [[nodiscard]] bool scale(std::int64_t byteStride);
  /// Replaces this domain by every `left + right` sum of the two domains.
  [[nodiscard]] bool addPointwise(const ByteAddressDomain &other);
  /// Drops every offset outside `[lower, upper]`.
  void restrictTo(std::int64_t lower, std::int64_t upper);
  /// Records that this domain is a strict over-approximation.
  void markApproximate() { exact_ = false; }

private:
  /// Sum of the progression sizes, saturating; overlapping progressions make
  /// this an upper bound on the number of distinct offsets.
  std::uint64_t count() const;
  void append(ByteOffsetProgression progression);
  void widen();

  llvm::SmallVector<ByteOffsetProgression, 4> progressions_;
  bool exact_ = true;
};

/// Removes the one index of `terms` whose byte stride equals `accessByteCount`.
/// The per-iteration accesses of a unit-stride counted loop over that index
/// then tile one contiguous range at the remaining base. `terms` is unchanged
/// when no such index exists.
bool removeContiguousIndex(llvm::SmallVectorImpl<LinearByteTerm> &terms,
                           mlir::Value index, std::uint64_t accessByteCount);

/// Byte offsets one address denotes, and whether one completed execution of
/// its enclosing counted loops denotes every one of them. Only an exhaustive
/// domain proves definite initialization; a merely possible domain still
/// contributes every payload its access may carry.
struct AddressByteDomain final {
  ByteAddressDomain offsets;
  bool exhaustive = false;
};

/// Geometry of one root-relative address: `byteBias` plus the scaled index
/// terms, accessed `accessByteCount` bytes wide.
struct AddressByteDomainRequest final {
  /// Base object of the address; identifies the access in diagnostics.
  mlir::Value root;
  llvm::ArrayRef<LinearByteTerm> terms;
  std::int64_t byteBias = 0;
  std::uint64_t accessByteCount = 0;
  /// Byte extent of the fixed allocation this address is rooted at.
  std::optional<std::uint64_t> allocationByteCount;
  /// Every index step from that allocation to this address is in bounds, so a
  /// defined access lies in `[0, allocationByteCount - accessByteCount]`.
  bool inBoundsOfAllocation = false;
  /// Indices whose every value is taken by one completed execution.
  llvm::ArrayRef<mlir::Value> completedIndices;
};

/// Proven finite value set of one index, when the caller owns one.
using FiniteIndexValues =
    llvm::function_ref<std::optional<std::set<std::int64_t>>(mlir::Value)>;

/// Byte domain of one address. Every index with a proven finite value set is
/// expanded exactly. At most one remaining index is admitted, because the
/// in-bounds containment window constrains the total offset rather than any
/// single term; that index is bounded by its own counted-loop, widening, or
/// inferred integer range intersected with the containment window. The result
/// is absent when no such bound exists.
std::optional<AddressByteDomain>
projectAddressByteDomain(const AddressByteDomainRequest &request,
                         FiniteIndexValues finiteIndexValues);

} // namespace loom::frontend::analysis

#endif
