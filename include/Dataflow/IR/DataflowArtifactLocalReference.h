#ifndef LOOM_DATAFLOW_IR_DATAFLOWARTIFACTLOCALREFERENCE_H
#define LOOM_DATAFLOW_IR_DATAFLOWARTIFACTLOCALREFERENCE_H

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace dataflow {

enum class DataflowArtifactLocalReferenceKind : std::uint32_t {
#define LOOM_DATAFLOW_LOCAL_REFERENCE_KIND(Ordinal, Type) Type = Ordinal,
#include "Dataflow/IR/DataflowRefs.def"
};

constexpr std::uint32_t dataflowArtifactLocalReferenceKindCount() {
  std::uint32_t count = 0;
#define LOOM_DATAFLOW_LOCAL_REFERENCE_KIND(Ordinal, Type) ++count;
#include "Dataflow/IR/DataflowRefs.def"
  return count;
}

constexpr std::uint32_t dataflowArtifactLocalReferenceKindOrdinal(
    DataflowArtifactLocalReferenceKind kind) {
  return static_cast<std::uint32_t>(kind);
}

struct DataflowArtifactLocalReferenceKindDescriptor {
  DataflowArtifactLocalReferenceKind kind;
  llvm::StringLiteral typedTarget;
};

llvm::ArrayRef<DataflowArtifactLocalReferenceKindDescriptor>
dataflowArtifactLocalReferenceKindCatalog();

template <typename Ref> struct DataflowArtifactLocalReferenceKindTraits;

#define LOOM_DATAFLOW_LOCAL_REFERENCE_KIND(Ordinal, Type)                      \
  template <> struct DataflowArtifactLocalReferenceKindTraits<Type> {          \
    static constexpr DataflowArtifactLocalReferenceKind kind =                 \
        DataflowArtifactLocalReferenceKind::Type;                              \
    static constexpr llvm::StringLiteral typedTarget = #Type;                  \
  };
#include "Dataflow/IR/DataflowRefs.def"

template <typename Ref, typename = void>
struct IsDataflowArtifactLocalReference : std::false_type {};

template <typename Ref>
struct IsDataflowArtifactLocalReference<
    Ref,
    std::void_t<decltype(DataflowArtifactLocalReferenceKindTraits<Ref>::kind)>>
    : std::true_type {};

template <typename Ref>
inline constexpr bool isDataflowArtifactLocalReference =
    IsDataflowArtifactLocalReference<Ref>::value;

template <typename Ref>
llvm::Expected<::loom::EncodedArtifactLocalReference>
encodeDataflowArtifactLocalReference(const ::loom::ArtifactIdentity &artifact,
                                     const Ref &reference) {
  static_assert(isDataflowArtifactLocalReference<Ref>,
                "Ref is not a Canonical Dataflow owner-local reference kind");
  llvm::Expected<std::vector<std::uint8_t>> payload =
      encodeDataflowReference(artifact, reference);
  if (!payload)
    return payload.takeError();
  return ::loom::EncodedArtifactLocalReference{
      ::loom::ArtifactRootReference{canonicalDataflowSchema.identity.str(),
                                    canonicalDataflowSchema.version, artifact},
      dataflowArtifactLocalReferenceKindOrdinal(
          DataflowArtifactLocalReferenceKindTraits<Ref>::kind),
      std::move(*payload)};
}

template <typename Ref>
llvm::Expected<Ref> decodeDataflowArtifactLocalReference(
    const ::loom::EncodedArtifactLocalReference &reference) {
  static_assert(isDataflowArtifactLocalReference<Ref>,
                "Ref is not a Canonical Dataflow owner-local reference kind");
  if (reference.artifact.schemaIdentity != canonicalDataflowSchema.identity ||
      reference.artifact.schemaVersion != canonicalDataflowSchema.version)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::ForeignArtifact,
        llvm::Twine("the local reference is not owned by ") +
            canonicalDataflowSchema.identity + " " +
            llvm::Twine(canonicalDataflowSchema.version.major) + "." +
            llvm::Twine(canonicalDataflowSchema.version.minor));

  const std::uint32_t expected = dataflowArtifactLocalReferenceKindOrdinal(
      DataflowArtifactLocalReferenceKindTraits<Ref>::kind);
  if (reference.ownerLocalKind != expected)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::WrongKind,
        llvm::Twine("owner-local kind ") +
            llvm::Twine(reference.ownerLocalKind) + " does not encode " +
            DataflowArtifactLocalReferenceKindTraits<Ref>::typedTarget);

  llvm::Expected<Ref> decoded = decodeDataflowReference<Ref>(
      reference.payload, reference.artifact.artifact);
  if (!decoded)
    return decoded.takeError();
  llvm::Expected<std::vector<std::uint8_t>> canonical =
      encodeDataflowReference(reference.artifact.artifact, *decoded);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != reference.payload)
    return makeDataflowReferenceError(
        DataflowReferenceErrorKind::Noncanonical,
        "noncanonical Canonical Dataflow reference payload");
  return std::move(*decoded);
}

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOWARTIFACTLOCALREFERENCE_H
