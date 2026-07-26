#ifndef LOOM_FABRIC_IR_REDUCED_PRODUCT_RELATION_H
#define LOOM_FABRIC_IR_REDUCED_PRODUCT_RELATION_H

#include "Fabric/IR/MemoryCapabilityDomains.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace fabric::detail {

struct ReducedFiniteAtom {
  std::vector<std::uint8_t> bytes;
};

struct ReducedFiniteDomain {
  std::vector<ReducedFiniteAtom> atoms;
};

using ReducedProductDomain = std::variant<ReducedFiniteDomain, UnsignedDomain>;
using ReducedProductRow = std::vector<ReducedProductDomain>;

/// Reduces a disjoint union of product rows using maximal domains whose
/// recursively reduced suffix relations are byte-identical. Finite fields
/// marked false remain singleton structural partitions.
llvm::Expected<std::vector<ReducedProductRow>>
reduceProductRelation(llvm::ArrayRef<ReducedProductRow> rows,
                      llvm::ArrayRef<bool> groupFiniteFields);

/// Returns whether every point in `subset` belongs to `superset`. The check is
/// exact for the finite and unsigned-interval domains used by Fabric memory
/// capabilities; it partitions interval boundaries without enumerating values.
llvm::Expected<bool>
reducedProductRelationCovers(llvm::ArrayRef<ReducedProductRow> superset,
                             llvm::ArrayRef<ReducedProductRow> subset);

/// Returns whether the two relations contain at least one common point.
llvm::Expected<bool>
reducedProductRelationsOverlap(llvm::ArrayRef<ReducedProductRow> left,
                               llvm::ArrayRef<ReducedProductRow> right);

/// Encodes rows in their current order. Callers use this both for persistent
/// framing and to distinguish strict canonical import from authoring input.
std::vector<std::uint8_t>
encodeReducedProductRelation(llvm::ArrayRef<ReducedProductRow> rows);

/// Strictly decodes the shared relation framing and validates canonical finite
/// and unsigned domains. The semantic owner of each finite atom remains
/// responsible for decoding and validating the atom bytes.
llvm::Expected<std::vector<ReducedProductRow>>
decodeReducedProductRelation(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric::detail

#endif // LOOM_FABRIC_IR_REDUCED_PRODUCT_RELATION_H
