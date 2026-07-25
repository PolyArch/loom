#ifndef LOOM_DATAFLOW_IR_DATAFLOWREFERENCECODEC_H
#define LOOM_DATAFLOW_IR_DATAFLOWREFERENCECODEC_H

#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowStructuralRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dataflow {

enum class DataflowReferenceErrorKind : std::uint8_t {
  MalformedSyntax,
  ForeignArtifact,
  WrongKind,
  MissingArtifact,
  Noncanonical,
};

class DataflowReferenceError final
    : public llvm::ErrorInfo<DataflowReferenceError> {
public:
  static char ID;

  DataflowReferenceError(DataflowReferenceErrorKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  DataflowReferenceErrorKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  DataflowReferenceErrorKind kind_;
  std::string message_;
};

llvm::Error makeDataflowReferenceError(DataflowReferenceErrorKind kind,
                                       const llvm::Twine &message);

template <typename Ref> struct DataflowReferenceCodecTraits;

#define LOOM_DATAFLOW_REFERENCE_CODEC(Type)                                    \
  template <> struct DataflowReferenceCodecTraits<Type> {                      \
    static llvm::Expected<std::vector<std::uint8_t>>                           \
    encode(const Type &reference,                                              \
           const ::loom::ArtifactIdentity *expectedArtifact);                  \
    static llvm::Expected<Type>                                                \
    decode(llvm::ArrayRef<std::uint8_t> bytes,                                 \
           const ::loom::ArtifactIdentity &artifact);                          \
  };
#include "Dataflow/IR/DataflowRefs.def"

/// Canonical local comparison bytes. Typed entity fields emit only their
/// unsigned 64-bit ID because their static field type carries the entity kind.
/// Closed-union alternatives emit zero-based u32be discriminants and all
/// owner-relative ordinals emit u64be. The exact artifact identity is omitted
/// from this local wire but every nested entity reference must bind one common
/// artifact.
template <typename Ref>
llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowReference(const Ref &reference) {
  return DataflowReferenceCodecTraits<Ref>::encode(reference, nullptr);
}

/// The same local wire with an explicit containing artifact binding. This form
/// also admits references whose selected alternative has no nested entity
/// field, while still validating every nested owner against the binding.
template <typename Ref>
llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowReference(const ::loom::ArtifactIdentity &artifact,
                        const Ref &reference) {
  return DataflowReferenceCodecTraits<Ref>::encode(reference, &artifact);
}

template <typename Ref>
llvm::Expected<Ref>
decodeDataflowReference(llvm::ArrayRef<std::uint8_t> bytes,
                        const ::loom::ArtifactIdentity &artifact) {
  return DataflowReferenceCodecTraits<Ref>::decode(bytes, artifact);
}

bool eventLogicalInputSlotLess(const EventLogicalInputSlot &lhs,
                               const EventLogicalInputSlot &rhs);

/// Strict canonical projection wire:
/// u64be(slot_count), then u32be(kind) and u64be(ordinal) per slot.
llvm::Expected<::loom::CanonicalSemanticBytes>
encodeEventLogicalProjection(const EventLogicalProjection &projection);

llvm::Expected<EventLogicalProjection>
decodeEventLogicalProjection(llvm::ArrayRef<std::uint8_t> bytes);

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOWREFERENCECODEC_H
