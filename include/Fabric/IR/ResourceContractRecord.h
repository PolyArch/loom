#ifndef FABRIC_IR_RESOURCECONTRACTRECORD_H
#define FABRIC_IR_RESOURCECONTRACTRECORD_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace fabric {

inline constexpr char kResourceContractRecordAttrName[] =
    "fabric.resource_contract";

/// Encodes one validated ResourceContract as its complete embedded persistent
/// record. The surrounding Fabric owner supplies artifact identity and schema
/// versioning; these bytes contain only the canonical record fields.
llvm::Expected<std::vector<std::uint8_t>>
encodeResourceContractRecord(const ResourceContract &contract);

/// Strictly imports one complete canonical persistent record and projects it
/// through ResourceContract::create. Noncanonical, malformed, truncated, and
/// trailing encodings are rejected.
llvm::Expected<ResourceContract>
decodeResourceContractRecord(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace fabric

#endif // FABRIC_IR_RESOURCECONTRACTRECORD_H
