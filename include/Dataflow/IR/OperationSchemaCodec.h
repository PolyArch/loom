#ifndef DATAFLOW_IR_OPERATIONSCHEMACODEC_H
#define DATAFLOW_IR_OPERATIONSCHEMACODEC_H

#include "Common/Artifact.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace dataflow {

/// Encodes and strictly imports the registry-owned persistent identity of one
/// operation schema. Dense OperationSchemaId values never enter these bytes.
llvm::Expected<loom::CanonicalSemanticBytes>
encodeOperationSchemaId(OperationSchemaId schema);

llvm::Expected<OperationSchemaId>
decodeOperationSchemaId(llvm::ArrayRef<std::uint8_t> bytes);

/// Encodes and strictly imports one member of the closed semantic-case
/// vocabulary through its explicit registry wire tag.
llvm::Expected<loom::CanonicalSemanticBytes>
encodeOperationSemanticsCase(OperationSemanticsCase semanticCase);

llvm::Expected<OperationSemanticsCase>
decodeOperationSemanticsCase(llvm::ArrayRef<std::uint8_t> bytes);

/// Produces the complete stable bytes of one typed actor projection.
llvm::Expected<loom::CanonicalSemanticBytes>
encodeCanonicalActorSchemaProjection(
    const CanonicalActorSchemaProjection &projection);

/// Rejects unknown, malformed, noncanonical, truncated, or trailing projection
/// bytes without introducing a second decoded semantic object.
llvm::Error
validateCanonicalActorSchemaProjectionBytes(llvm::ArrayRef<std::uint8_t> bytes);

/// Projects a registered operation through the canonical actor interface and
/// immediately encodes the resulting typed projection.
llvm::Expected<loom::CanonicalSemanticBytes>
projectRegisteredActorSchemaProjectionBytes(::mlir::Operation *op);

} // namespace dataflow

#endif // DATAFLOW_IR_OPERATIONSCHEMACODEC_H
