#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

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

ArtifactIdentity identity(const char *test, std::uint8_t seed) {
  return takeExpected(
      test, ArtifactIdentity::fromBytes(
                std::vector<std::uint8_t>(ArtifactIdentity::byteSize, seed)));
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

/// The loom.fabric semantic domain is independent of the codec internals so
/// the rejection tests build malformed framing without trusting the encoder.
void appendSemanticDomain(std::vector<std::uint8_t> &bytes) {
  static constexpr char domain[] = "loom.fabric.semantic.v1\0";
  bytes.insert(bytes.end(), domain, domain + sizeof(domain) - 1);
}

void appendDependencyRow(std::vector<std::uint8_t> &bytes, std::uint32_t role,
                         llvm::StringRef schema, std::uint32_t major,
                         std::uint32_t minor,
                         const ArtifactIdentity &artifact) {
  appendU32Be(bytes, role);
  appendU32Be(bytes, static_cast<std::uint32_t>(schema.size()));
  bytes.insert(bytes.end(), schema.begin(), schema.end());
  appendU32Be(bytes, major);
  appendU32Be(bytes, minor);
  bytes.insert(bytes.end(), artifact.bytes().begin(), artifact.bytes().end());
}

/// Assembles one envelope from independent framing fields. `rowCount` may
/// disagree with the rows it is paired with so malformed framing can be fed to
/// the strict decoder.
std::vector<std::uint8_t> buildEnvelope(std::uint32_t variant,
                                        llvm::ArrayRef<std::uint8_t> rows,
                                        std::uint64_t rowCount,
                                        llvm::ArrayRef<std::uint8_t> payload) {
  std::vector<std::uint8_t> bytes;
  appendSemanticDomain(bytes);
  appendU32Be(bytes, variant);
  appendU64Be(bytes, rowCount);
  bytes.insert(bytes.end(), rows.begin(), rows.end());
  appendU64Be(bytes, payload.size());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

std::vector<std::uint8_t> moduleEnvelopeWithOneDependency() {
  FabricDirectDependency dependency{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity("envelope", 0x01)}};
  CanonicalSemanticBytes encoded = takeExpected(
      "envelope", encodeFabricArtifactEnvelope(FabricRootKind::Module,
                                               {dependency}, {0xaa, 0xbb}));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

void expectDecodeFails(const char *test, llvm::ArrayRef<std::uint8_t> bytes,
                       llvm::StringRef expectedReason) {
  llvm::Expected<DecodedFabricArtifact> result =
      decodeFabricArtifactEnvelope(bytes);
  if (result)
    fail(test, "decode accepted an envelope that must be rejected");
  std::string message = llvm::toString(result.takeError());
  if (message.find(expectedReason) == std::string::npos)
    fail(test, "decode rejected for the wrong reason: " + message);
}

void expectDecodeFails(const char *test, llvm::ArrayRef<std::uint8_t> bytes) {
  llvm::Expected<DecodedFabricArtifact> result =
      decodeFabricArtifactEnvelope(bytes);
  if (result)
    fail(test, "decode accepted an envelope that must be rejected");
  llvm::consumeError(result.takeError());
}

void rootReferenceIsAnOwningCommonValue() {
  const ArtifactIdentity expected = identity(__func__, 0x5a);
  std::string sourceIdentity = "loom.test.root";
  ArtifactRootReference reference{sourceIdentity, {3, 7}, expected};
  sourceIdentity.assign("changed.after.construction");
  require(__func__, reference.schemaIdentity == "loom.test.root",
          "root reference did not own its schema identity");
  require(__func__, reference.schemaVersion == SchemaVersion{3, 7},
          "root reference did not preserve its schema version");
  require(__func__, reference.artifact == expected,
          "root reference did not preserve its artifact identity");

  ArtifactRootReference copy = reference;
  require(__func__, copy == reference,
          "root reference copy changed its owned value");
  reference.schemaIdentity.assign("changed.after.copy");
  require(__func__, copy != reference,
          "root reference equality ignored its owned schema identity");
  require(__func__, copy.schemaIdentity == "loom.test.root",
          "root reference copy did not own its schema identity");
}

void envelopeFramesRootDependencyAndPayloadExactly() {
  const FabricDirectDependency dependency{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x01)}};
  const std::vector<std::uint8_t> payload{0xaa, 0xbb};

  const CanonicalSemanticBytes encoded = takeExpected(
      __func__, encodeFabricArtifactEnvelope(FabricRootKind::Module,
                                             {dependency}, payload));
  require(__func__,
          llvm::toHex(encoded.bytes(), true) ==
              "6c6f6f6d2e6661627269632e73656d616e7469632e763100"
              "00000000"
              "0000000000000001"
              "00000000"
              "0000000b6c6f6f6d2e666162726963"
              "0000000100000000"
              "0101010101010101010101010101010101010101010101010101010101010101"
              "0000000000000002aabb",
          "Fabric semantic envelope bytes changed");

  const DecodedFabricArtifact decoded =
      takeExpected(__func__, decodeFabricArtifactEnvelope(encoded.bytes()));
  require(__func__, decoded.rootKind == FabricRootKind::Module,
          "decoded root kind changed");
  require(__func__,
          decoded.dependencies ==
              std::vector<FabricDirectDependency>{dependency},
          "decoded dependency row changed");
  require(__func__, decoded.canonicalMlirBytecode == payload,
          "decoded canonical payload changed");
}

void envelopeFramesEveryRootOrdinal() {
  struct Case {
    FabricRootKind kind;
    std::uint32_t ordinal;
  };
  const Case cases[] = {
      {FabricRootKind::Module, 0},
      {FabricRootKind::System, 1},
      {FabricRootKind::InterconnectImplementation, 2},
  };
  for (const Case &one : cases) {
    const CanonicalSemanticBytes encoded = takeExpected(
        __func__, encodeFabricArtifactEnvelope(one.kind, {}, {0xcc}));
    llvm::ArrayRef<std::uint8_t> bytes = encoded.bytes();
    require(__func__,
            bytes[24] == 0 && bytes[25] == 0 && bytes[26] == 0 &&
                bytes[27] == static_cast<std::uint8_t>(one.ordinal),
            "root variant ordinal byte is wrong");
    const DecodedFabricArtifact decoded =
        takeExpected(__func__, decodeFabricArtifactEnvelope(bytes));
    require(__func__, decoded.rootKind == one.kind,
            "decoded root kind did not round trip");
    require(__func__, decoded.dependencies.empty(),
            "spurious dependency decoded for a root-only envelope");
  }
}

void emptyDependencyTableRoundTrips() {
  const std::vector<std::uint8_t> payload{0xde, 0xad, 0xbe, 0xef};
  const CanonicalSemanticBytes encoded = takeExpected(
      __func__,
      encodeFabricArtifactEnvelope(FabricRootKind::Module, {}, payload));
  const DecodedFabricArtifact decoded =
      takeExpected(__func__, decodeFabricArtifactEnvelope(encoded.bytes()));
  require(__func__, decoded.dependencies.empty(),
          "spurious dependency decoded for an empty table");
  require(__func__, decoded.canonicalMlirBytecode == payload,
          "decoded canonical payload changed");

  llvm::ArrayRef<std::uint8_t> bytes = encoded.bytes();
  bool countIsZero = true;
  for (std::size_t index = 28; index < 36; ++index)
    countIsZero = countIsZero && bytes[index] == 0;
  require(__func__, countIsZero,
          "zero dependency count was not encoded as a zero u64");
}

void encodeSortsDependenciesCanonically() {
  // Deliberately non-canonical input order spanning distinct roles and
  // identities so the sort exercises both the role and identity fields.
  const FabricDirectDependency implementationInput{
      FabricDependencyRole::ImplementationInput,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x02)}};
  const FabricDirectDependency importedHigher{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x03)}};
  const FabricDirectDependency importedLower{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x01)}};
  const FabricDirectDependency refinedSystem{
      FabricDependencyRole::RefinedSystem,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x04)}};

  const CanonicalSemanticBytes encoded = takeExpected(
      __func__,
      encodeFabricArtifactEnvelope(
          FabricRootKind::System,
          {implementationInput, importedHigher, refinedSystem, importedLower},
          {}));
  std::vector<std::uint8_t> canonicalRows;
  appendDependencyRow(canonicalRows, 0, "loom.fabric", 1, 0,
                      importedLower.root.artifact);
  appendDependencyRow(canonicalRows, 0, "loom.fabric", 1, 0,
                      importedHigher.root.artifact);
  appendDependencyRow(canonicalRows, 1, "loom.fabric", 1, 0,
                      refinedSystem.root.artifact);
  appendDependencyRow(canonicalRows, 2, "loom.fabric", 1, 0,
                      implementationInput.root.artifact);
  const std::vector<std::uint8_t> expected =
      buildEnvelope(1, canonicalRows, 4, {});
  require(__func__, encoded.bytes().equals(expected),
          "canonical dependency row bytes are wrong");

  const CanonicalSemanticBytes permuted = takeExpected(
      __func__,
      encodeFabricArtifactEnvelope(
          FabricRootKind::System,
          {refinedSystem, importedLower, implementationInput, importedHigher},
          {}));
  require(__func__, permuted.bytes().equals(encoded.bytes()),
          "dependency input order changed canonical envelope bytes");

  const DecodedFabricArtifact decoded =
      takeExpected(__func__, decodeFabricArtifactEnvelope(encoded.bytes()));
  require(__func__, decoded.dependencies.size() == 4,
          "canonical sort lost a dependency");
  require(__func__, decoded.dependencies[0] == importedLower,
          "first canonical dependency row is wrong");
  require(__func__, decoded.dependencies[1] == importedHigher,
          "second canonical dependency row is wrong");
  require(__func__, decoded.dependencies[2] == refinedSystem,
          "third canonical dependency row is wrong");
  require(__func__, decoded.dependencies[3] == implementationInput,
          "fourth canonical dependency row is wrong");
}

void encodeRejectsEmptySchemaIdentity() {
  const FabricDirectDependency dependency{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"", {1, 0}, identity(__func__, 0x01)}};
  llvm::Expected<CanonicalSemanticBytes> result =
      encodeFabricArtifactEnvelope(FabricRootKind::Module, {dependency}, {});
  if (result)
    fail(__func__, "encode accepted an empty schema identity");
  std::string message = llvm::toString(result.takeError());
  if (message.find("dependency schema identity is empty") == std::string::npos)
    fail(__func__, "encode rejected for the wrong reason: " + message);
}

void encodeRejectsDuplicateDependencyRow() {
  const FabricDirectDependency dependency{
      FabricDependencyRole::ImportedModule,
      ArtifactRootReference{"loom.fabric", {1, 0}, identity(__func__, 0x01)}};
  llvm::Expected<CanonicalSemanticBytes> result = encodeFabricArtifactEnvelope(
      FabricRootKind::Module, {dependency, dependency}, {});
  if (result)
    fail(__func__, "encode accepted a duplicate direct dependency row");
  std::string message = llvm::toString(result.takeError());
  if (message.find("duplicate direct dependency row") == std::string::npos)
    fail(__func__, "encode rejected for the wrong reason: " + message);
}

void decodeRejectsWrongDomain() {
  std::vector<std::uint8_t> bytes = moduleEnvelopeWithOneDependency();
  bytes[0] ^= 0xff;
  expectDecodeFails(__func__, bytes, "wrong semantic domain");
}

void decodeRejectsUnknownRootOrdinal() {
  std::vector<std::uint8_t> rows;
  appendDependencyRow(rows, 0, "loom.fabric", 1, 0, identity(__func__, 0x01));
  const std::vector<std::uint8_t> bytes = buildEnvelope(3, rows, 1, {});
  expectDecodeFails(__func__, bytes, "unknown root kind ordinal");
}

void decodeRejectsUnknownDependencyRole() {
  std::vector<std::uint8_t> rows;
  appendDependencyRow(rows, 3, "loom.fabric", 1, 0, identity(__func__, 0x01));
  const std::vector<std::uint8_t> bytes = buildEnvelope(0, rows, 1, {});
  expectDecodeFails(__func__, bytes, "unknown dependency role ordinal");
}

void decodeRejectsEmptySchemaIdentity() {
  std::vector<std::uint8_t> rows;
  appendU32Be(rows, 0); // role
  appendU32Be(rows, 0); // zero-length schema identity
  rows.insert(rows.end(), 41, 0x00);
  const std::vector<std::uint8_t> bytes = buildEnvelope(0, rows, 1, {});
  expectDecodeFails(__func__, bytes, "dependency schema identity is empty");
}

void decodeRejectsEveryTruncatedPrefix() {
  const std::vector<std::uint8_t> bytes = moduleEnvelopeWithOneDependency();
  for (std::size_t size = 0; size < bytes.size(); ++size)
    expectDecodeFails(__func__,
                      llvm::ArrayRef<std::uint8_t>(bytes).take_front(size));
}

void decodeRejectsTrailingBytes() {
  std::vector<std::uint8_t> bytes = moduleEnvelopeWithOneDependency();
  bytes.push_back(0x00);
  expectDecodeFails(__func__, bytes, "trailing bytes");
}

void decodeRejectsNoncanonicalDependencyOrder() {
  std::vector<std::uint8_t> rows;
  // Two distinct rows encoded in descending canonical order.
  appendDependencyRow(rows, 0, "loom.fabric", 1, 0, identity(__func__, 0x02));
  appendDependencyRow(rows, 0, "loom.fabric", 1, 0, identity(__func__, 0x01));
  const std::vector<std::uint8_t> bytes = buildEnvelope(0, rows, 2, {});
  expectDecodeFails(__func__, bytes, "noncanonical dependency order");
}

void decodeRejectsDuplicateDependencyRow() {
  std::vector<std::uint8_t> rows;
  appendDependencyRow(rows, 0, "loom.fabric", 1, 0, identity(__func__, 0x01));
  appendDependencyRow(rows, 0, "loom.fabric", 1, 0, identity(__func__, 0x01));
  const std::vector<std::uint8_t> bytes = buildEnvelope(0, rows, 2, {});
  expectDecodeFails(__func__, bytes, "duplicate direct dependency row");
}

void decodeRejectsImpossibleSchemaIdentityLength() {
  std::vector<std::uint8_t> rows;
  appendU32Be(rows, 0);
  appendU32Be(rows, 9);
  rows.insert(rows.end(), 41, 0x61);
  const std::vector<std::uint8_t> bytes = buildEnvelope(0, rows, 1, {});
  expectDecodeFails(
      __func__, bytes,
      "dependency schema identity length cannot fit the remaining envelope");
}

void decodeRejectsImpossiblePayloadLength() {
  std::vector<std::uint8_t> bytes = buildEnvelope(0, {}, 0, {});
  std::fill(bytes.end() - 8, bytes.end(), 0xff);
  expectDecodeFails(__func__, bytes, "truncated canonical MLIR bytecode");
}

void decodeRejectsImpossibleDependencyCount() {
  const std::vector<std::uint8_t> one = buildEnvelope(0, {}, 1, {});
  expectDecodeFails(__func__, one,
                    "dependency count cannot fit the remaining envelope");

  std::vector<std::uint8_t> noPayloadLength =
      buildEnvelope(0, std::vector<std::uint8_t>(49, 0x00), 1, {});
  noPayloadLength.resize(noPayloadLength.size() - 8);
  expectDecodeFails(__func__, noPayloadLength,
                    "dependency count cannot fit the remaining envelope");

  const std::vector<std::uint8_t> maximum =
      buildEnvelope(0, {}, 0xffffffffffffffffULL, {});
  expectDecodeFails(__func__, maximum,
                    "dependency count cannot fit the remaining envelope");
}

} // namespace

int main() {
  rootReferenceIsAnOwningCommonValue();
  envelopeFramesRootDependencyAndPayloadExactly();
  envelopeFramesEveryRootOrdinal();
  emptyDependencyTableRoundTrips();
  encodeSortsDependenciesCanonically();
  encodeRejectsEmptySchemaIdentity();
  encodeRejectsDuplicateDependencyRow();
  decodeRejectsWrongDomain();
  decodeRejectsUnknownRootOrdinal();
  decodeRejectsUnknownDependencyRole();
  decodeRejectsEmptySchemaIdentity();
  decodeRejectsEveryTruncatedPrefix();
  decodeRejectsTrailingBytes();
  decodeRejectsNoncanonicalDependencyOrder();
  decodeRejectsDuplicateDependencyRow();
  decodeRejectsImpossibleSchemaIdentityLength();
  decodeRejectsImpossiblePayloadLength();
  decodeRejectsImpossibleDependencyCount();
  llvm::outs() << "fabric artifact codec ok\n";
  return 0;
}
