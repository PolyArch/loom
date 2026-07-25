#include "Common/ArtifactLocalReference.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace loom;

namespace {

void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

void expectErrorContains(const char *test, llvm::Error error,
                         llvm::StringRef expected) {
  if (!error)
    fail(test, "expected an error");
  std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

constexpr ArtifactSchemaDescriptor testSchema{"loom.test.local_reference",
                                              SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor otherSchema{"loom.test.other_family",
                                               SchemaVersion{1, 0}};

ArtifactIdentity testArtifact() {
  ArtifactIdentity::Storage bytes{};
  bytes[0] = 0x5a;
  return llvm::cantFail(ArtifactIdentity::fromBytes(bytes));
}

llvm::Error strictDecodeEmpty(llvm::ArrayRef<std::uint8_t> payload) {
  if (!payload.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the test payload must be empty");
  return llvm::Error::success();
}

llvm::Error acceptAll(const EncodedArtifactLocalReference &) {
  return llvm::Error::success();
}

llvm::Error rejectAll(const EncodedArtifactLocalReference &) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "the test owner rejects every reference");
}

EncodedArtifactLocalReference emptyReference() {
  return EncodedArtifactLocalReference{
      ArtifactRootReference{testSchema, testArtifact()}, 0, {}};
}

/// Registration is idempotent for one owner codec and rejects a conflicting
/// codec for the same (owner schema, owner-local kind) pair.
void registrationIsIdempotentAndConflictChecked() {
  const ArtifactLocalReferenceCodec codec{&strictDecodeEmpty, &acceptAll};
  if (llvm::Error error =
          registerArtifactLocalReferenceKind(testSchema, 0, codec))
    fail(__func__, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerArtifactLocalReferenceKind(testSchema, 0, codec))
    fail(__func__, llvm::toString(std::move(error)));
  expectErrorContains(
      __func__,
      registerArtifactLocalReferenceKind(
          testSchema, 0,
          ArtifactLocalReferenceCodec{&strictDecodeEmpty, &rejectAll}),
      "conflicting registration");
  expectErrorContains(
      __func__,
      registerArtifactLocalReferenceKind(
          testSchema, 1, ArtifactLocalReferenceCodec{nullptr, &acceptAll}),
      "requires both a strict decoder and a validator");
}

/// Lookup results are value copies taken under the registry lock: no later
/// registration, however much the registry grows, can invalidate or mutate an
/// earlier result.
void lookupResultsSurviveLaterRegistrations() {
  const std::optional<ArtifactLocalReferenceCodec> earlyCodec =
      findArtifactLocalReferenceKind(testSchema, 0);
  const std::optional<ArtifactSchemaDescriptor> earlySchema =
      findArtifactLocalReferenceSchema(testSchema.identity, testSchema.version);
  require(__func__, earlyCodec && earlySchema,
          "a registered kind did not resolve");

  // Force repeated growth of the registry after the lookup.
  for (std::uint32_t kind = 1; kind <= 64; ++kind)
    if (llvm::Error error = registerArtifactLocalReferenceKind(
            otherSchema, kind,
            ArtifactLocalReferenceCodec{&strictDecodeEmpty, &rejectAll}))
      fail(__func__, llvm::toString(std::move(error)));

  require(__func__,
          earlyCodec->strictDecode == &strictDecodeEmpty &&
              earlyCodec->validate == &acceptAll,
          "a later registration mutated an earlier codec lookup result");
  require(__func__, *earlySchema == testSchema,
          "a later registration mutated an earlier schema lookup result");

  const std::optional<ArtifactLocalReferenceCodec> lateCodec =
      findArtifactLocalReferenceKind(testSchema, 0);
  require(__func__,
          lateCodec && lateCodec->strictDecode == earlyCodec->strictDecode &&
              lateCodec->validate == earlyCodec->validate,
          "the registered codec changed across registry growth");
  require(__func__,
          !findArtifactLocalReferenceKind(testSchema, 999) &&
              !findArtifactLocalReferenceSchema("loom.test.absent", {1, 0}),
          "an unregistered kind or schema resolved");
}

/// Dispatch strictly decodes through the owner codec, surfaces owner
/// rejections, and reports an unknown kind as a capability error.
void validationDispatchesToTheOwnerCodec() {
  if (llvm::Error error = validateArtifactLocalReference(emptyReference()))
    fail(__func__, llvm::toString(std::move(error)));

  EncodedArtifactLocalReference malformed = emptyReference();
  malformed.payload = {0x01};
  expectErrorContains(__func__, validateArtifactLocalReference(malformed),
                      "must be empty");

  EncodedArtifactLocalReference rejected = emptyReference();
  rejected.artifact.schema = otherSchema;
  rejected.ownerLocalKind = 7;
  expectErrorContains(__func__, validateArtifactLocalReference(rejected),
                      "rejects every reference");

  EncodedArtifactLocalReference unknown = emptyReference();
  unknown.ownerLocalKind = 999;
  expectErrorContains(__func__, validateArtifactLocalReference(unknown),
                      "no registered owner codec");
}

} // namespace

int main() {
  registrationIsIdempotentAndConflictChecked();
  lookupResultsSurviveLaterRegistrations();
  validationDispatchesToTheOwnerCodec();
  return 0;
}
