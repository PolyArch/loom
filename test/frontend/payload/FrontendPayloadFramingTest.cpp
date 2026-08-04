#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

/// Returns the typed rejection message, failing the test when the hostile input
/// was accepted instead.
template <typename T>
std::string rejectionMessage(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "a value that must fail closed was accepted");
  return llvm::toString(value.takeError());
}

llvm::ArrayRef<std::uint8_t> asBytes(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

/// Fixed provider input for the independently computed ABI-key known vector.
/// Production payloads derive their provider from the selected gitlink; this
/// vector intentionally stays independent of the current checkout.
constexpr llvm::StringRef closedRepositoryIdentity = "llvm-project";
constexpr llvm::StringRef closedCommitIdentity =
    "040a641988f6ed6f4fab250706ca2b620c1de2d8";

/// Canonical target triple of the fixed ABI-key vector below.
constexpr llvm::StringRef vectorTargetTriple = "x86_64-unknown-linux-gnu";

/// SHA-256 of the component-view preimage for the exact 1.0 frontend view,
/// computed independently of this project's encoders.
constexpr std::array<std::uint8_t, ComponentViewDigest::byteSize>
    knownFrontendViewDigest = {0x65, 0xec, 0x50, 0x54, 0xb5, 0xee, 0x7d, 0x71,
                               0xf8, 0x11, 0x78, 0xfd, 0xc2, 0xfb, 0x69, 0x58,
                               0x54, 0xc7, 0x06, 0x71, 0x67, 0x32, 0xa9, 0x20,
                               0x6a, 0x5b, 0x54, 0x08, 0xac, 0x8d, 0xd2, 0x4d};

/// SHA-256 of the ABI-key preimage for the closed provider, the canonical
/// target triple above, and the 1.0 frontend view, computed independently of
/// this project's encoders.
constexpr std::array<std::uint8_t, AbiCompatibilityKey::byteSize>
    knownAbiCompatibilityKey = {0xad, 0x60, 0xcf, 0x4c, 0x62, 0xc6, 0x49, 0xe6,
                                0x68, 0x71, 0x34, 0x1f, 0x55, 0x6e, 0x37, 0x31,
                                0xc3, 0x00, 0xea, 0xb0, 0x22, 0xbd, 0xec, 0xd8,
                                0xbb, 0xf0, 0xf1, 0x44, 0x23, 0x3a, 0x68, 0x34};

AbiCompatibilityKeyInputs
vectorKeyInputs(const ResolvedFrontendConfigView &view) {
  AbiCompatibilityKeyInputs inputs;
  inputs.repositoryIdentity = closedRepositoryIdentity;
  inputs.fullCommitIdentity = closedCommitIdentity;
  inputs.canonicalTargetTriple = vectorTargetTriple;
  inputs.viewSchemaDescriptorBytes = view.schemaDescriptorBytes();
  inputs.viewCanonicalBytes = view.canonicalViewBytes();
  return inputs;
}

void frontendViewIsTheClosedZeroFieldProjection() {
  const ResolvedFrontendConfigView view =
      projectResolvedFrontendConfigView(defaultResolvedConfig());

  const llvm::StringRef descriptor(
      reinterpret_cast<const char *>(view.schemaDescriptorBytes().data()),
      view.schemaDescriptorBytes().size());
  require(__func__, descriptor == "loom.config.view.frontend.1.0",
          "unexpected frontend view schema descriptor: " + descriptor.str());
  require(__func__, !descriptor.contains('\0'),
          "frontend view schema descriptor carries a zero byte");
  require(__func__, view.canonicalViewBytes().empty(),
          "frontend view 1.0 canonical bytes are not empty");
  require(__func__, view.digest().bytes() == knownFrontendViewDigest,
          "known frontend view digest changed: " +
              llvm::toHex(view.digest().bytes(), true));

  // The digest is the Common component-view value over exactly those bytes.
  require(__func__,
          view.digest() ==
              takeExpected(__func__, computeComponentViewDigest(
                                         view.schemaDescriptorBytes(),
                                         view.canonicalViewBytes())),
          "frontend view digest is not the Common component-view digest");
  static_assert(!std::is_default_constructible_v<ResolvedFrontendConfigView>);
}

void unrelatedResolvedConfigFieldsLeaveTheViewUnchanged() {
  const ResolvedConfig base = defaultResolvedConfig();
  const ResolvedFrontendConfigView baseView =
      projectResolvedFrontendConfigView(base);

  ResolvedConfig changed = base;
  ++changed.dse.structuredOwnership.scopeExpansionLimit;
  ++changed.dse.spatialPnr.search.routing.endpointExpansionLimit;
  require(__func__,
          resolvedConfigIdentity(base) != resolvedConfigIdentity(changed),
          "the mutated ResolvedConfig was not a semantic change");

  const ResolvedFrontendConfigView changedView =
      projectResolvedFrontendConfigView(changed);
  require(__func__,
          changedView.canonicalViewBytes() == baseView.canonicalViewBytes(),
          "an unconsumed ResolvedConfig field changed the view bytes");
  require(__func__, changedView.digest() == baseView.digest(),
          "an unconsumed ResolvedConfig field changed the view digest");
}

void adoptedFrontendViewFieldsFailClosed() {
  const ResolvedFrontendConfigView view =
      projectResolvedFrontendConfigView(defaultResolvedConfig());
  const ComponentViewDigest digest = view.digest();
  takeExpected(__func__, adoptResolvedFrontendConfigView(
                             view.schemaDescriptorBytes(),
                             view.canonicalViewBytes(), digest));

  const std::string wrongDescriptor = "loom.config.view.frontend.1.1";
  require(__func__,
          llvm::StringRef(rejectionMessage(
                              __func__, adoptResolvedFrontendConfigView(
                                            asBytes(wrongDescriptor),
                                            view.canonicalViewBytes(), digest)))
              .contains("frontend_config_view_descriptor"),
          "unexpected foreign frontend view descriptor rejection");

  std::array<std::uint8_t, ComponentViewDigest::byteSize> staleDigest =
      knownFrontendViewDigest;
  staleDigest.back() ^= 0x01;
  require(__func__,
          llvm::StringRef(
              rejectionMessage(
                  __func__,
                  adoptResolvedFrontendConfigView(
                      view.schemaDescriptorBytes(), view.canonicalViewBytes(),
                      takeExpected(__func__, ComponentViewDigest::fromBytes(
                                                 staleDigest)))))
              .contains("component_view_digest_mismatch"),
          "unexpected stale frontend view digest rejection");
}

void abiKeyMatchesKnownVector() {
  const ResolvedFrontendConfigView view =
      projectResolvedFrontendConfigView(defaultResolvedConfig());
  const AbiCompatibilityKey key =
      computeAbiCompatibilityKey(vectorKeyInputs(view));
  require(__func__, key.bytes() == knownAbiCompatibilityKey,
          "known ABI compatibility key changed: " +
              llvm::toHex(key.bytes(), true));
  require(__func__, key == computeAbiCompatibilityKey(vectorKeyInputs(view)),
          "identical ABI key inputs were not deterministic");
  require(
      __func__,
      llvm::toString(validateAbiCompatibilityKey(vectorKeyInputs(view), key))
          .empty(),
      "the exact ABI compatibility key was rejected");

  std::array<std::uint8_t, AbiCompatibilityKey::byteSize> staleKey =
      knownAbiCompatibilityKey;
  staleKey.back() ^= 0x01;
  const std::string message = llvm::toString(validateAbiCompatibilityKey(
      vectorKeyInputs(view),
      takeExpected(__func__, AbiCompatibilityKey::fromBytes(staleKey))));
  require(__func__,
          llvm::StringRef(message).contains("abi_compatibility_key_mismatch"),
          "unexpected stale ABI key validation result: " + message);
  static_assert(!std::is_default_constructible_v<AbiCompatibilityKey>);
  static_assert(!std::is_same_v<AbiCompatibilityKey, ComponentViewDigest>);
}

} // namespace

int main() {
  frontendViewIsTheClosedZeroFieldProjection();
  unrelatedResolvedConfigFieldsLeaveTheViewUnchanged();
  adoptedFrontendViewFieldsFailClosed();
  abiKeyMatchesKnownVector();
  return 0;
}
