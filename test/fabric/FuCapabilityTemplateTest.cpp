#include "Common/Artifact.h"
#include "Fabric/Artifact/FabricArtifactLocalReference.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"

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

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << message << "\n";
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireRejected(llvm::Expected<T> value, const std::string &message) {
  if (value)
    fail(message);
  llvm::consumeError(value.takeError());
}

void requireRejected(llvm::Error error, const std::string &message) {
  if (!error)
    fail(message);
  llvm::consumeError(std::move(error));
}

ArtifactIdentity identity(std::uint8_t seed) {
  return takeExpected(ArtifactIdentity::fromBytes(
      std::vector<std::uint8_t>(ArtifactIdentity::byteSize, seed)));
}

FabricFuTemplateNodeRef node(FabricFuTemplateRef fu, FabricFuNodeKind kind,
                             FabricOrdinal ordinal) {
  return FabricFuTemplateNodeRef{kind, fu, ordinal};
}

FabricFuCapabilityTemplateEndpointRef boundary(FabricFuTemplateRef fu,
                                               FabricPortDirection direction,
                                               FabricOrdinal ordinal) {
  return FabricFuCapabilityTemplateEndpointRef::boundaryPort(
      FabricFuTemplatePortRef{fu, direction, ordinal});
}

FabricFuCapabilityTemplateEndpointRef nodePort(FabricFuTemplateNodeRef owner,
                                               FabricPortDirection direction,
                                               FabricOrdinal ordinal) {
  return FabricFuCapabilityTemplateEndpointRef::nodePort(
      FabricFuNodePortRef{owner, direction, ordinal});
}

FabricFuCapabilityTemplateRecord twoNodeRecord(FabricFuTemplateRef fu) {
  const FabricFuTemplateNodeRef add = node(fu, FabricFuNodeKind::Op, 0);
  const FabricFuTemplateNodeRef mux = node(fu, FabricFuNodeKind::Mux, 1);
  return FabricFuCapabilityTemplateRecord{
      {mux, add},
      {{nodePort(mux, FabricPortDirection::Output, 0),
        boundary(fu, FabricPortDirection::Output, 0)},
       {boundary(fu, FabricPortDirection::Input, 0),
        nodePort(add, FabricPortDirection::Input, 0)},
       {nodePort(add, FabricPortDirection::Output, 0),
        nodePort(mux, FabricPortDirection::Input, 0)}}};
}

FabricFuCapabilityTemplateRecord oneNodeRecord(FabricFuTemplateRef fu,
                                               FabricOrdinal ordinal) {
  const FabricFuTemplateNodeRef active =
      node(fu, FabricFuNodeKind::Op, ordinal);
  return FabricFuCapabilityTemplateRecord{
      {active},
      {{boundary(fu, FabricPortDirection::Input, ordinal),
        nodePort(active, FabricPortDirection::Input, 0)},
       {nodePort(active, FabricPortDirection::Output, 0),
        boundary(fu, FabricPortDirection::Output, ordinal)}}};
}

void testRecordNormalizationAndCodec() {
  const FabricFuTemplateRef fu(7);
  const FabricFuCapabilityTemplateRecord normalized = takeExpected(
      normalizeFabricFuCapabilityTemplateRecord(twoNodeRecord(fu)));
  require(!normalized.activeNodes.empty(), "normalization lost active nodes");

  FabricFuCapabilityTemplateRecord permuted = normalized;
  std::reverse(permuted.activeNodes.begin(), permuted.activeNodes.end());
  std::reverse(permuted.activeEdges.begin(), permuted.activeEdges.end());
  require(takeExpected(normalizeFabricFuCapabilityTemplateRecord(permuted)) ==
              normalized,
          "record order changed canonical capability semantics");

  const std::vector<std::uint8_t> bytes =
      takeExpected(canonicalFabricFuCapabilityTemplateBytes(permuted));
  require(takeExpected(decodeFabricFuCapabilityTemplateRecord(bytes)) ==
              normalized,
          "canonical capability-template bytes did not round-trip");

  requireRejected(normalizeFabricFuCapabilityTemplateRecord({}),
                  "an empty active-node set was accepted");

  FabricFuCapabilityTemplateRecord duplicate = normalized;
  duplicate.activeNodes.push_back(duplicate.activeNodes.front());
  requireRejected(normalizeFabricFuCapabilityTemplateRecord(duplicate),
                  "a duplicate active node was accepted");
}

void testOwnerAndEndpointConstraints() {
  const FabricFuTemplateRef fu(11);
  FabricFuCapabilityTemplateRecord record = twoNodeRecord(fu);

  record.activeEdges.front().destination =
      boundary(FabricFuTemplateRef(12), FabricPortDirection::Output, 0);
  requireRejected(normalizeFabricFuCapabilityTemplateRecord(record),
                  "an edge endpoint with a foreign FU owner was accepted");

  record = twoNodeRecord(fu);
  record.activeEdges.front().source =
      boundary(fu, FabricPortDirection::Output, 0);
  requireRejected(normalizeFabricFuCapabilityTemplateRecord(record),
                  "an output boundary was accepted as an edge source");

  record = twoNodeRecord(fu);
  const FabricFuTemplateNodeRef inactive = node(fu, FabricFuNodeKind::Demux, 9);
  record.activeEdges.front().source =
      nodePort(inactive, FabricPortDirection::Output, 0);
  requireRejected(normalizeFabricFuCapabilityTemplateRecord(record),
                  "an edge naming an inactive node was accepted");
}

void testInventoryAndExactReference() {
  const FabricFuTemplateRef fu(19);
  std::vector<FabricFuCapabilityTemplateRecord> authored = {
      oneNodeRecord(fu, 3), oneNodeRecord(fu, 1)};
  const auto inventory =
      takeExpected(normalizeFabricFuCapabilityTemplateInventory(authored));
  require(inventory.size() == 2, "inventory normalization lost a record");

  const auto firstBytes =
      takeExpected(canonicalFabricFuCapabilityTemplateBytes(inventory[0]));
  const auto secondBytes =
      takeExpected(canonicalFabricFuCapabilityTemplateBytes(inventory[1]));
  require(firstBytes < secondBytes,
          "capability-template ordinals are not canonical byte order");

  const FabricFuCapabilityTemplateRef ref{fu, 1};
  if (llvm::Error error = validateFabricFuCapabilityTemplateRef(inventory, ref))
    fail("valid dense capability reference was rejected: " +
         llvm::toString(std::move(error)));
  requireRejected(validateFabricFuCapabilityTemplateRef(
                      inventory, FabricFuCapabilityTemplateRef{fu, 2}),
                  "an out-of-range capability ordinal was accepted");
  requireRejected(
      validateFabricFuCapabilityTemplateRef(
          inventory, FabricFuCapabilityTemplateRef{FabricFuTemplateRef(20), 0}),
      "a capability reference with the wrong owner was accepted");

  authored.push_back(authored.front());
  requireRejected(normalizeFabricFuCapabilityTemplateInventory(authored),
                  "a duplicate capability-template record was accepted");

  const ArtifactIdentity artifact = identity(0x55);
  const EncodedArtifactLocalReference encoded =
      encodeFabricArtifactLocalReference(
          ArtifactReference<FabricFuCapabilityTemplateRef>{artifact, ref});
  require(
      encoded.ownerLocalKind ==
          fabricArtifactLocalReferenceKindOrdinal(
              FabricArtifactLocalReferenceKind::FabricFuCapabilityTemplateRef),
      "capability reference used the wrong owner-local kind");
  require(takeExpected(
              decodeFabricArtifactLocalReference<FabricFuCapabilityTemplateRef>(
                  encoded)) ==
              ArtifactReference<FabricFuCapabilityTemplateRef>{artifact, ref},
          "capability reference did not round-trip through Common framing");
}

} // namespace

int main() {
  testRecordNormalizationAndCodec();
  testOwnerAndEndpointConstraints();
  testInventoryAndExactReference();
  return 0;
}
