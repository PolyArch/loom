#include "Common/ComponentViewDigest.h"
#include "Common/ResolvedConfig.h"
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
#include <vector>

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

/// The closed provider identity of payload version 1.0. The build derives the
/// production value from the selected gitlink; repeating it here checks that
/// the build still selects the provider this payload version closed over.
constexpr llvm::StringRef closedRepositoryIdentity = "llvm-project";
constexpr llvm::StringRef closedCommitIdentity =
    "040a641988f6ed6f4fab250706ca2b620c1de2d8";

/// Canonical target facts of the fixed ABI-key vector below.
constexpr llvm::StringRef vectorTargetTriple = "x86_64-unknown-linux-gnu";
constexpr llvm::StringRef vectorDataLayout =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-"
    "n8:16:32:64-S128";

/// SHA-256 of the component-view preimage for the exact 1.0 frontend view,
/// computed independently of this project's encoders.
constexpr std::array<std::uint8_t, ComponentViewDigest::byteSize>
    knownFrontendViewDigest = {0x65, 0xec, 0x50, 0x54, 0xb5, 0xee, 0x7d, 0x71,
                               0xf8, 0x11, 0x78, 0xfd, 0xc2, 0xfb, 0x69, 0x58,
                               0x54, 0xc7, 0x06, 0x71, 0x67, 0x32, 0xa9, 0x20,
                               0x6a, 0x5b, 0x54, 0x08, 0xac, 0x8d, 0xd2, 0x4d};

/// SHA-256 of the ABI-key preimage for the closed provider, the canonical
/// target facts above, and the 1.0 frontend view, computed independently of
/// this project's encoders.
constexpr std::array<std::uint8_t, AbiCompatibilityKey::byteSize>
    knownAbiCompatibilityKey = {0xda, 0xe2, 0xde, 0xf1, 0x2b, 0x68, 0x24, 0x44,
                                0x01, 0x3f, 0xf2, 0x2f, 0x16, 0x27, 0xe9, 0x3b,
                                0x3b, 0x8a, 0x65, 0xd7, 0xe6, 0x05, 0xb7, 0x9b,
                                0x0b, 0x02, 0x0e, 0x35, 0x94, 0x1a, 0x1b, 0x64};

AbiCompatibilityKeyInputs
vectorKeyInputs(const ResolvedFrontendConfigView &view) {
  AbiCompatibilityKeyInputs inputs;
  inputs.repositoryIdentity = closedRepositoryIdentity;
  inputs.fullCommitIdentity = closedCommitIdentity;
  inputs.canonicalTargetTriple = vectorTargetTriple;
  inputs.canonicalDataLayout = vectorDataLayout;
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
  changed.configId = "loom.other";
  changed.global.addrBits = base.global.addrBits + 1;
  changed.global.indexWidth = 64;
  changed.global.memBusWidth = base.global.memBusWidth * 2;
  changed.dse.rankingPolicy = "lexicographic";
  changed.dse.objectives.push_back(ResolvedDseObjective{"area", 0.25});
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

void abiKeyFollowsEverySourceField() {
  const ResolvedFrontendConfigView view =
      projectResolvedFrontendConfigView(defaultResolvedConfig());
  const AbiCompatibilityKey key =
      computeAbiCompatibilityKey(vectorKeyInputs(view));

  const std::string otherRepository = "llvm-project-fork";
  const std::string otherCommit(40, 'a');
  const std::string otherTriple = "aarch64-unknown-linux-gnu";
  const std::string otherDataLayout = vectorDataLayout.str() + "-P0";
  const std::string otherDescriptor = "loom.config.view.frontend.2.0";
  const std::array<std::uint8_t, 1> otherViewBytes = {0x00};

  struct Mutation {
    const char *name;
    AbiCompatibilityKeyInputs inputs;
  };
  std::vector<Mutation> mutations;
  auto add = [&](const char *name, auto &&mutate) {
    AbiCompatibilityKeyInputs inputs = vectorKeyInputs(view);
    mutate(inputs);
    mutations.push_back(Mutation{name, inputs});
  };
  add("repository_identity", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.repositoryIdentity = otherRepository;
  });
  add("full_commit_identity", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.fullCommitIdentity = otherCommit;
  });
  add("canonical_target_triple", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.canonicalTargetTriple = otherTriple;
  });
  add("canonical_data_layout", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.canonicalDataLayout = otherDataLayout;
  });
  add("view_schema_descriptor", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.viewSchemaDescriptorBytes = asBytes(otherDescriptor);
  });
  add("view_canonical_bytes", [&](AbiCompatibilityKeyInputs &inputs) {
    inputs.viewCanonicalBytes = otherViewBytes;
  });

  for (const Mutation &mutation : mutations) {
    require(__func__, computeAbiCompatibilityKey(mutation.inputs) != key,
            std::string("changing ") + mutation.name +
                " did not change the ABI compatibility key");
    require(__func__,
            llvm::StringRef(llvm::toString(validateAbiCompatibilityKey(
                                mutation.inputs, key)))
                .contains("abi_compatibility_key_mismatch"),
            std::string("a key derived from a different ") + mutation.name +
                " was still accepted");
  }

  // Adjacent fields must not be confusable: moving a byte across the
  // repository/commit boundary is a distinct framed preimage.
  AbiCompatibilityKeyInputs shifted = vectorKeyInputs(view);
  const std::string shortRepository =
      closedRepositoryIdentity.drop_back(1).str();
  const std::string extendedCommit =
      closedRepositoryIdentity.take_back(1).str() + closedCommitIdentity.str();
  shifted.repositoryIdentity = shortRepository;
  shifted.fullCommitIdentity = extendedCommit;
  require(__func__, computeAbiCompatibilityKey(shifted) != key,
          "the framed provider fields are not length delimited");

  rejectionMessage(__func__,
                   AbiCompatibilityKey::fromBytes(std::vector<std::uint8_t>(
                       AbiCompatibilityKey::byteSize - 1, 0)));
  rejectionMessage(__func__,
                   AbiCompatibilityKey::fromBytes(std::vector<std::uint8_t>(
                       AbiCompatibilityKey::byteSize + 1, 0)));
}

} // namespace

int main() {
  frontendViewIsTheClosedZeroFieldProjection();
  unrelatedResolvedConfigFieldsLeaveTheViewUnchanged();
  adoptedFrontendViewFieldsFailClosed();
  abiKeyMatchesKnownVector();
  abiKeyFollowsEverySourceField();
  return 0;
}
