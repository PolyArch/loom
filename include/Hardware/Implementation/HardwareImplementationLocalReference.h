#ifndef LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONLOCALREFERENCE_H
#define LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONLOCALREFERENCE_H

#include "Common/ArtifactLocalReference.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::hardware {

/// Dense ordinals derived from the finalized canonical catalogs. They are
/// meaningful only with the exact owning HardwareImplementation identity.
struct HardwareImplementationInterfaceRef final {
  std::uint64_t ordinal = 0;

  friend bool operator==(HardwareImplementationInterfaceRef lhs,
                         HardwareImplementationInterfaceRef rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

struct HardwareImplementationActivityPointRef final {
  std::uint64_t ordinal = 0;

  friend bool operator==(HardwareImplementationActivityPointRef lhs,
                         HardwareImplementationActivityPointRef rhs) {
    return lhs.ordinal == rhs.ordinal;
  }
};

enum class HardwareImplementationLocalReferenceKind : std::uint32_t {
#define LOOM_HARDWARE_IMPLEMENTATION_LOCAL_REFERENCE_KIND(Ordinal, Kind, Type, \
                                                          CatalogAccessor)     \
  Kind = Ordinal,
#include "Hardware/Implementation/HardwareImplementationLocalReferences.def"
};

constexpr std::uint32_t hardwareImplementationLocalReferenceKindCount() {
  std::uint32_t count = 0;
#define LOOM_HARDWARE_IMPLEMENTATION_LOCAL_REFERENCE_KIND(Ordinal, Kind, Type, \
                                                          CatalogAccessor)     \
  ++count;
#include "Hardware/Implementation/HardwareImplementationLocalReferences.def"
  return count;
}

constexpr std::uint32_t hardwareImplementationLocalReferenceKindOrdinal(
    HardwareImplementationLocalReferenceKind kind) {
  return static_cast<std::uint32_t>(kind);
}

template <typename Ref> struct HardwareImplementationLocalReferenceKindTraits;

#define LOOM_HARDWARE_IMPLEMENTATION_LOCAL_REFERENCE_KIND(Ordinal, Kind, Type, \
                                                          CatalogAccessor)     \
  template <> struct HardwareImplementationLocalReferenceKindTraits<Type> {    \
    static constexpr HardwareImplementationLocalReferenceKind kind =           \
        HardwareImplementationLocalReferenceKind::Kind;                        \
    static constexpr llvm::StringLiteral typedTarget = #Type;                  \
  };
#include "Hardware/Implementation/HardwareImplementationLocalReferences.def"

template <typename Ref, typename = void>
struct IsHardwareImplementationLocalReference : std::false_type {};

template <typename Ref>
struct IsHardwareImplementationLocalReference<
    Ref, std::void_t<decltype(HardwareImplementationLocalReferenceKindTraits<
                              Ref>::kind)>> : std::true_type {};

template <typename Ref>
inline constexpr bool isHardwareImplementationLocalReference =
    IsHardwareImplementationLocalReference<Ref>::value;

namespace detail {

std::array<std::uint8_t, 8>
encodeHardwareImplementationLocalOrdinal(std::uint64_t ordinal);

llvm::Expected<std::uint64_t> decodeHardwareImplementationLocalReferenceOrdinal(
    const EncodedArtifactLocalReference &reference,
    HardwareImplementationLocalReferenceKind expectedKind,
    llvm::StringRef typedTarget);

} // namespace detail

template <typename Ref>
EncodedArtifactLocalReference encodeHardwareImplementationLocalReference(
    const ArtifactReference<Ref> &reference) {
  static_assert(isHardwareImplementationLocalReference<Ref>,
                "Ref is not a HardwareImplementation owner-local kind");
  const auto payload = detail::encodeHardwareImplementationLocalOrdinal(
      reference.entity.ordinal);
  return EncodedArtifactLocalReference{
      ArtifactRootReference{hardwareImplementationSchema.identity.str(),
                            hardwareImplementationSchema.version,
                            reference.artifact},
      hardwareImplementationLocalReferenceKindOrdinal(
          HardwareImplementationLocalReferenceKindTraits<Ref>::kind),
      std::vector<std::uint8_t>(payload.begin(), payload.end())};
}

template <typename Ref>
llvm::Expected<ArtifactReference<Ref>>
decodeHardwareImplementationLocalReference(
    const EncodedArtifactLocalReference &reference) {
  static_assert(isHardwareImplementationLocalReference<Ref>,
                "Ref is not a HardwareImplementation owner-local kind");
  auto ordinal = detail::decodeHardwareImplementationLocalReferenceOrdinal(
      reference, HardwareImplementationLocalReferenceKindTraits<Ref>::kind,
      HardwareImplementationLocalReferenceKindTraits<Ref>::typedTarget);
  if (!ordinal)
    return ordinal.takeError();
  return ArtifactReference<Ref>{reference.artifact.artifact, Ref{*ordinal}};
}

/// Validates the exact owner identity, structural typed payload, and canonical
/// catalog bound selected by the encoded owner-local kind.
llvm::Error validateHardwareImplementationLocalReference(
    const FinalizedHardwareImplementation &implementation,
    const EncodedArtifactLocalReference &reference);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_IMPLEMENTATION_HARDWAREIMPLEMENTATIONLOCALREFERENCE_H
