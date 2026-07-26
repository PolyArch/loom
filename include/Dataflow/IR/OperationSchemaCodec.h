#ifndef DATAFLOW_IR_OPERATIONSCHEMACODEC_H
#define DATAFLOW_IR_OPERATIONSCHEMACODEC_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

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

/// Stable codecs for closed Dataflow-owned atoms embedded by downstream
/// capability records. Decoders reject wrong domains, unknown tags, malformed
/// payloads, truncation, and trailing bytes.
llvm::Expected<loom::CanonicalSemanticBytes>
encodeServiceValueRole(semantics::ServiceValueRole role);
llvm::Expected<semantics::ServiceValueRole>
decodeServiceValueRole(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeServiceKind(semantics::ServiceKind kind);
llvm::Expected<semantics::ServiceKind>
decodeServiceKind(llvm::ArrayRef<std::uint8_t> bytes);

/// The standalone production codec for exact service payload types. It is the
/// same closed type vocabulary used inside actor schema projections, with its
/// own domain framing so downstream records never depend on MLIR printing.
llvm::Expected<loom::CanonicalSemanticBytes>
encodeCanonicalType(::mlir::Type type);
llvm::Expected<::mlir::Type>
decodeCanonicalType(llvm::ArrayRef<std::uint8_t> bytes,
                    ::mlir::MLIRContext *context);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeMemoryAccessForm(semantics::MemoryAccessForm form);
llvm::Expected<semantics::MemoryAccessForm>
decodeMemoryAccessForm(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeMemoryMaskForm(semantics::MemoryMaskForm form);
llvm::Expected<semantics::MemoryMaskForm>
decodeMemoryMaskForm(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeAtomicOrdering(AtomicOrdering ordering);
llvm::Expected<AtomicOrdering>
decodeAtomicOrdering(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeAtomicRmwKind(AtomicRmwKind kind);
llvm::Expected<AtomicRmwKind>
decodeAtomicRmwKind(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeVectorAtomicGranularity(VectorAtomicGranularity granularity);
llvm::Expected<VectorAtomicGranularity>
decodeVectorAtomicGranularity(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeOptionalVectorAtomicGranularity(
    std::optional<VectorAtomicGranularity> granularity);
llvm::Expected<std::optional<VectorAtomicGranularity>>
decodeOptionalVectorAtomicGranularity(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<loom::CanonicalSemanticBytes>
encodeSyncScopeRef(const SyncScopeProjection &scope);
llvm::Expected<SyncScopeProjection>
decodeSyncScopeRef(llvm::ArrayRef<std::uint8_t> bytes,
                   ::mlir::MLIRContext *context);

llvm::Expected<loom::CanonicalSemanticBytes> encodeCanonicalBoolean(bool value);
llvm::Expected<bool> decodeCanonicalBoolean(llvm::ArrayRef<std::uint8_t> bytes);

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
