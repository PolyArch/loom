#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"

#include "ArtifactLocalReferenceRegistry.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  std::cerr << test << ": " << message << '\n';
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

void expectErrorContains(const char *test, llvm::Error error,
                         llvm::StringRef expected) {
  if (!error)
    fail(test, "expected an error");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "unexpected error: " + message);
}

void expectOwnerCodecUnavailable(const char *test, llvm::Error error) {
  if (!error)
    fail(test, "expected an unavailable owner codec");
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      std::move(error),
      [&](const ArtifactLocalReferenceError &failure) -> llvm::Error {
        matched = failure.kind() ==
                  ArtifactLocalReferenceErrorKind::OwnerCodecUnavailable;
        return llvm::Error::success();
      });
  if (remaining)
    fail(test, llvm::toString(std::move(remaining)));
  require(test, matched, "wrong local-reference capability failure");
}

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const char *test) {
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-artifact-local-reference", path_))
      fail(test, error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

constexpr ArtifactSchemaDescriptor testSchema{"loom.test.local_reference",
                                               {1, 0}};
constexpr ArtifactSchemaDescriptor otherSchema{"loom.test.other_reference",
                                                {1, 0}};
constexpr std::uint32_t testKind = 7;

CanonicalSemanticBytes semantic(std::initializer_list<std::uint8_t> bytes) {
  return CanonicalSemanticBytes(std::vector<std::uint8_t>(bytes));
}

llvm::Error validateCanonicalPayload(llvm::ArrayRef<std::uint8_t> payload) {
  if (payload.size() != 2)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid test target payload length");
  const std::vector<std::uint8_t> canonical{0, payload[1]};
  if (!std::equal(payload.begin(), payload.end(), canonical.begin()))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "noncanonical test target payload");
  return llvm::Error::success();
}

llvm::Error
validateTarget(const CanonicalSemanticBytes &artifactBytes,
               const EncodedArtifactLocalReference &reference) {
  if (reference.artifact.schemaIdentity != testSchema.identity ||
      reference.artifact.schemaVersion != testSchema.version ||
      reference.ownerLocalKind != testKind)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "wrong test local-reference owner");
  if (finalizeArtifactIdentity(testSchema, artifactBytes) !=
      reference.artifact.artifact)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test artifact identity does not match bytes");
  const llvm::ArrayRef<std::uint8_t> bytes = artifactBytes.bytes();
  if (bytes.empty() || bytes.size() != static_cast<std::size_t>(bytes[0]) + 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid test artifact bytes");
  if (!std::is_sorted(bytes.drop_front().begin(), bytes.drop_front().end()) ||
      std::adjacent_find(bytes.drop_front().begin(),
                         bytes.drop_front().end()) != bytes.drop_front().end())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "noncanonical test artifact bytes");
  const std::uint8_t target = reference.payload[1];
  if (std::find(bytes.drop_front().begin(), bytes.drop_front().end(), target) ==
      bytes.drop_front().end())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test target is not present in artifact");
  return llvm::Error::success();
}

ArtifactRootReference root(const ArtifactSchemaDescriptor &schema,
                           const ArtifactIdentity &identity) {
  return ArtifactRootReference{schema.identity.str(), schema.version, identity};
}

EncodedArtifactLocalReference reference(const ArtifactRootReference &artifact,
                                        std::uint8_t target) {
  return EncodedArtifactLocalReference{artifact, testKind, {0, target}};
}

void payloadValidationDoesNotResolveArtifact() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({1, 3});
  const ArtifactRootReference absent =
      root(testSchema, finalizeArtifactIdentity(testSchema, bytes));
  if (llvm::Error error =
          validateArtifactLocalReferencePayload(reference(absent, 3)))
    fail(__func__, llvm::toString(std::move(error)));

  EncodedArtifactLocalReference noncanonical = reference(absent, 3);
  noncanonical.payload[0] = 1;
  expectErrorContains(
      __func__, validateArtifactLocalReferencePayload(noncanonical),
      "noncanonical test target payload");
  expectErrorContains(__func__,
                      validateArtifactLocalReference(store, noncanonical),
                      "noncanonical test target payload");
}

void exactStoreBackedValidationSucceeds() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({2, 3, 5});
  const ArtifactIdentity identity =
      takeExpected(__func__, store.put(testSchema, bytes));

  if (llvm::Error error = validateArtifactLocalReference(
          store, reference(root(testSchema, identity), 5)))
    fail(__func__, llvm::toString(std::move(error)));
}

void missingArtifactPropagatesStoreFailure() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({1, 3});
  const ArtifactRootReference absent =
      root(testSchema, finalizeArtifactIdentity(testSchema, bytes));

  expectErrorContains(
      __func__, validateArtifactLocalReference(store, reference(absent, 3)),
      "artifact_store_missing");
}

void wrongArtifactTargetIsRejected() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const ArtifactIdentity identity =
      takeExpected(__func__, store.put(testSchema, semantic({1, 3})));

  expectErrorContains(__func__,
                      validateArtifactLocalReference(
                          store, reference(root(testSchema, identity), 5)),
                      "test target is not present in artifact");
}

void unavailableCodecPrecedesArtifactLookup() {
  TemporaryDirectory directory(__func__);
  ArtifactStore store(directory.path());
  const CanonicalSemanticBytes bytes = semantic({1, 3});
  EncodedArtifactLocalReference unknown{
      root(otherSchema, finalizeArtifactIdentity(otherSchema, bytes)), testKind,
      {0, 3}};

  expectOwnerCodecUnavailable(
      __func__, validateArtifactLocalReference(store, unknown));
}

} // namespace

int main() {
  llvm::cantFail(registerArtifactLocalReferenceKind(
      testSchema, testKind,
      ArtifactLocalReferenceCodec{validateCanonicalPayload, validateTarget}));
  payloadValidationDoesNotResolveArtifact();
  exactStoreBackedValidationSucceeds();
  missingArtifactPropagatesStoreFailure();
  wrongArtifactTargetIsRejected();
  unavailableCodecPrecedesArtifactLookup();
  return 0;
}
