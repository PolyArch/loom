#ifndef LOOM_FRONTEND_PAYLOAD_RELOCATABLEACCELERATORPAYLOAD_H
#define LOOM_FRONTEND_PAYLOAD_RELOCATABLEACCELERATORPAYLOAD_H

#include "Common/Artifact.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"
#include "Frontend/Payload/FrontendConfigView.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace loom {

/// The closed LLVM provider the build selected. Both fields are complete: the
/// commit identity is the full pinned gitlink commit, never a release nickname
/// or an abbreviated hash.
struct LlvmProviderIdentity {
  std::string repositoryIdentity;
  std::string fullCommitIdentity;
};

/// The provider derived from the gitlink this build selected. It is the sole
/// production authority for the provider fields a payload records.
const LlvmProviderIdentity &buildSelectedLlvmProvider();

/// The complete `loom.relocatable_accelerator_payload` 1.0 Artifact value.
///
/// Its raw fields are authority: the normalized module bytes, the provider
/// identity, the canonical target facts, and the frontend view descriptor and
/// bytes. The bitcode digest, component-view digest, and ABI compatibility key
/// are mechanical projections that a reader recomputes and rejects on
/// disagreement.
class RelocatableAcceleratorPayload {
public:
  static constexpr ArtifactSchemaDescriptor artifactSchema{
      "loom.relocatable_accelerator_payload", SchemaVersion{1, 0}};

  /// Normalizes the source bitcode once, derives every projection from the
  /// result, and returns the complete payload. Creation is deterministic and
  /// failure-atomic: any rejection yields a typed error and no payload.
  static llvm::Expected<RelocatableAcceleratorPayload>
  create(llvm::ArrayRef<std::uint8_t> sourceBitcode,
         const ResolvedFrontendConfigView &frontendConfigView);

  const LlvmProviderIdentity &llvmProvider() const { return provider_; }
  llvm::StringRef targetTriple() const { return targetTriple_; }
  llvm::StringRef dataLayout() const { return dataLayout_; }
  const AbiCompatibilityKey &abiCompatibilityKey() const { return abiKey_; }
  const ResolvedFrontendConfigView &frontendConfigView() const { return view_; }
  llvm::ArrayRef<std::uint8_t> normalizedBitcode() const { return bitcode_; }
  llvm::ArrayRef<std::uint8_t> normalizedBitcodeDigest() const {
    return bitcodeDigest_;
  }

  /// The canonical root in its fixed field order. Fixed digests are raw 32-byte
  /// values; every string and variable byte sequence carries an unsigned 64-bit
  /// big-endian length followed by its exact bytes.
  CanonicalSemanticBytes canonicalSemanticBytes() const;

  /// The Common ArtifactIdentity over this payload's schema and canonical
  /// bytes.
  ArtifactIdentity identity() const;

private:
  RelocatableAcceleratorPayload(LlvmProviderIdentity provider,
                                std::string targetTriple,
                                std::string dataLayout,
                                AbiCompatibilityKey abiKey,
                                ResolvedFrontendConfigView view,
                                std::vector<std::uint8_t> bitcode,
                                std::array<std::uint8_t, 32> bitcodeDigest);

  LlvmProviderIdentity provider_;
  std::string targetTriple_;
  std::string dataLayout_;
  AbiCompatibilityKey abiKey_;
  ResolvedFrontendConfigView view_;
  std::vector<std::uint8_t> bitcode_;
  std::array<std::uint8_t, 32> bitcodeDigest_;

  friend llvm::Expected<RelocatableAcceleratorPayload>
  decodeRelocatableAcceleratorPayload(
      const ArtifactSchemaDescriptor &schema,
      llvm::ArrayRef<std::uint8_t> canonicalBytes);
};

/// Decodes and fully validates canonical payload bytes carrying the given
/// Artifact schema metadata.
///
/// The schema identity and version are checked before anything is parsed. The
/// stored module is reparsed, materialized, verified, and rewritten through the
/// same production writer, and the result must equal the stored bytes exactly.
/// Every stored projection is recomputed from the raw fields. A payload that
/// disagrees anywhere is rejected, never repaired.
llvm::Expected<RelocatableAcceleratorPayload>
decodeRelocatableAcceleratorPayload(
    const ArtifactSchemaDescriptor &schema,
    llvm::ArrayRef<std::uint8_t> canonicalBytes);

/// The raw-field cohort preflight over two already validated payloads: provider
/// repository and commit, canonical target triple and data layout, ABI
/// compatibility key, and the complete frontend config view.
///
/// This is a necessary condition, not proof that the LLVM modules link. It
/// inspects no module content; the pinned LLVM Linker and LTO libraries remain
/// the authority for symbol resolution, COMDAT, ODR, and module flags.
llvm::Error requireRelocatablePayloadCompatibility(
    const RelocatableAcceleratorPayload &lhs,
    const RelocatableAcceleratorPayload &rhs);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_RELOCATABLEACCELERATORPAYLOAD_H
