#ifndef FABRIC_IR_PHYSICALTAGRESOURCECONTRACT_H
#define FABRIC_IR_PHYSICALTAGRESOURCECONTRACT_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace fabric {

/// Appends one stateless sharing-assignment pattern per Physical Tag width.
/// Existing owner patterns, claims, transitions, and arbitration remain exact.
/// A null base creates the minimal valid owner contract for the assignments.
llvm::Expected<ResourceContract>
appendPhysicalTagAssignmentPatterns(const ResourceContract *base,
                                    llvm::ArrayRef<std::uint32_t> tagWidthBits);

/// Returns the exact Physical Tag width of a stateless assignment pattern.
/// Any pattern with claims, a commit, transactions, parameters, or a different
/// sharing schema is not an assignment pattern.
std::optional<std::uint32_t>
physicalTagAssignmentPatternWidth(const UsePattern &pattern);

} // namespace fabric

#endif // FABRIC_IR_PHYSICALTAGRESOURCECONTRACT_H
