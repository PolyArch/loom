#include "Dataflow/Transforms/DataflowRewrite.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace dataflow {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.dataflow_rewrite.decision.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "dataflow_rewrite_decision_invalid: " +
                                     message);
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

void appendActor(std::vector<std::uint8_t> &bytes, ActorRef actor) {
  bytes.insert(bytes.end(), actor.artifact.bytes().begin(),
               actor.artifact.bytes().end());
  appendU64(bytes, actor.entity.value());
}

llvm::Expected<ActorRef> readActor(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != loom::ArtifactIdentity::byteSize + 8)
    return invalid("actor reference has the wrong size");
  auto artifact = loom::ArtifactIdentity::fromBytes(
      bytes.take_front(loom::ArtifactIdentity::byteSize));
  if (!artifact)
    return artifact.takeError();
  return ActorRef{
      *artifact,
      ActorId(readU64(bytes.drop_front(loom::ArtifactIdentity::byteSize)))};
}

} // namespace

llvm::ArrayRef<std::uint8_t> dataflowRewriteDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeDataflowRewriteDecision(const DataflowRewriteDecision &decision) {
  std::vector<std::uint8_t> bytes;
  if (const auto *kind = std::get_if<DataflowRewriteKind>(&decision)) {
    if (static_cast<unsigned>(*kind) >
        static_cast<unsigned>(
            DataflowRewriteKind::ActivationPreservingConstantFold))
      return invalid("fixed rewrite has an unknown kind");
    bytes.push_back(0);
    bytes.push_back(static_cast<std::uint8_t>(*kind));
    return bytes;
  }
  if (const auto *chunk =
          std::get_if<ElementwiseVectorChunkRewrite>(&decision)) {
    if (chunk->leadingBlocksPerChunk <= 0)
      return invalid("vector chunk factor is invalid");
    bytes.push_back(1);
    appendActor(bytes, chunk->actor);
    appendU64(bytes, static_cast<std::uint64_t>(chunk->leadingBlocksPerChunk));
    return bytes;
  }
  const auto &scalar = std::get<ElementwiseVectorScalarizeRewrite>(decision);
  bytes.push_back(2);
  appendActor(bytes, scalar.actor);
  return bytes;
}

llvm::Expected<DataflowRewriteDecision>
adoptDataflowRewriteDecision(llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  if (canonicalBytes.empty())
    return invalid("decision payload is empty");
  DataflowRewriteDecision decision =
      DataflowRewriteKind::PackUnpackRoundTripEliminate;
  switch (canonicalBytes.front()) {
  case 0: {
    if (canonicalBytes.size() != 2 ||
        canonicalBytes[1] >
            static_cast<std::uint8_t>(
                DataflowRewriteKind::ActivationPreservingConstantFold))
      return invalid("fixed rewrite payload is invalid");
    decision = static_cast<DataflowRewriteKind>(canonicalBytes[1]);
    break;
  }
  case 1: {
    constexpr std::size_t actorSize = loom::ArtifactIdentity::byteSize + 8;
    if (canonicalBytes.size() != 1 + actorSize + 8)
      return invalid("vector chunk payload has the wrong size");
    auto actor = readActor(canonicalBytes.slice(1, actorSize));
    if (!actor)
      return actor.takeError();
    const std::uint64_t rawFactor =
        readU64(canonicalBytes.drop_front(1 + actorSize));
    if (rawFactor == 0 ||
        rawFactor > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()))
      return invalid("vector chunk factor is invalid");
    decision = ElementwiseVectorChunkRewrite{
        *actor, static_cast<std::int64_t>(rawFactor)};
    break;
  }
  case 2: {
    constexpr std::size_t actorSize = loom::ArtifactIdentity::byteSize + 8;
    if (canonicalBytes.size() != 1 + actorSize)
      return invalid("vector scalarization payload has the wrong size");
    auto actor = readActor(canonicalBytes.drop_front(1));
    if (!actor)
      return actor.takeError();
    decision = ElementwiseVectorScalarizeRewrite{*actor};
    break;
  }
  default:
    return invalid("decision payload has an unknown kind");
  }
  auto reencoded = encodeDataflowRewriteDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

} // namespace dataflow
