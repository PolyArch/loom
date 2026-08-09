#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactRootReference;
using loom::ArtifactSchemaDescriptor;
using loom::dse::DataflowRewriteDerivation;
using loom::dse::detail::StructuredOwnershipDataflowLineageIndex;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "structuredOwnershipLineageIndex: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireErrorContains(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error");
  const std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail("error did not contain the expected text");
}

ArtifactRootReference makeReference(const ArtifactSchemaDescriptor &schema,
                                    std::uint64_t ordinal) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  for (std::size_t byte = 0; byte < sizeof(ordinal); ++byte)
    bytes[byte] = static_cast<std::uint8_t>(ordinal >> (byte * 8));
  return {schema.identity.str(), schema.version,
          take(ArtifactIdentity::fromBytes(bytes))};
}

struct Fixture final {
  ArtifactRootReference structuredA =
      makeReference(loom::frontend::structuredProgramArtifactSchema, 1);
  ArtifactRootReference structuredB =
      makeReference(loom::frontend::structuredProgramArtifactSchema, 2);
  ArtifactRootReference rootA =
      makeReference(dataflow::canonicalDataflowSchema, 10);
  ArtifactRootReference rootB =
      makeReference(dataflow::canonicalDataflowSchema, 11);
  ArtifactRootReference first =
      makeReference(dataflow::canonicalDataflowSchema, 20);
  ArtifactRootReference second =
      makeReference(dataflow::canonicalDataflowSchema, 21);
  ArtifactRootReference joined =
      makeReference(dataflow::canonicalDataflowSchema, 22);
  ArtifactRootReference descendant =
      makeReference(dataflow::canonicalDataflowSchema, 23);
  ArtifactRootReference foreign =
      makeReference(dataflow::canonicalDataflowSchema, 24);
  ArtifactRootReference cycleA =
      makeReference(dataflow::canonicalDataflowSchema, 25);
  ArtifactRootReference cycleB =
      makeReference(dataflow::canonicalDataflowSchema, 26);
};

void populateReconvergent(StructuredOwnershipDataflowLineageIndex &index,
                          const Fixture &fixture, bool reverse) {
  requireSuccess(index.recordRoot(fixture.structuredA, fixture.rootA));
  requireSuccess(index.recordRoot(fixture.structuredB, fixture.rootB));
  const dataflow::DataflowRewriteDecision firstKind =
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)};
  const dataflow::DataflowRewriteDecision secondKind =
      dataflow::ParallelizeSerializeRoundTripRewrite{dataflow::ActorId(2)};
  const dataflow::DataflowRewriteDecision joinKind =
      dataflow::ActivationPreservingConstantFoldRewrite{dataflow::ActorId(3)};
  if (reverse) {
    requireSuccess(
        index.recordDecision(fixture.rootA, fixture.second, secondKind));
    requireSuccess(
        index.recordDecision(fixture.rootA, fixture.first, firstKind));
    requireSuccess(
        index.recordDecision(fixture.rootB, fixture.foreign, firstKind));
    requireSuccess(
        index.recordDecision(fixture.foreign, fixture.joined, joinKind));
    requireSuccess(
        index.recordDecision(fixture.second, fixture.joined, secondKind));
    requireSuccess(
        index.recordDecision(fixture.first, fixture.joined, firstKind));
  } else {
    requireSuccess(
        index.recordDecision(fixture.rootA, fixture.first, firstKind));
    requireSuccess(
        index.recordDecision(fixture.rootA, fixture.second, secondKind));
    requireSuccess(
        index.recordDecision(fixture.first, fixture.joined, firstKind));
    requireSuccess(
        index.recordDecision(fixture.second, fixture.joined, secondKind));
    requireSuccess(
        index.recordDecision(fixture.rootB, fixture.foreign, firstKind));
    requireSuccess(
        index.recordDecision(fixture.foreign, fixture.joined, joinKind));
  }
  requireSuccess(
      index.recordDecision(fixture.first, fixture.joined, firstKind));
}

std::vector<DataflowRewriteDerivation>
resolveForA(StructuredOwnershipDataflowLineageIndex &index,
            const Fixture &fixture) {
  auto lineage = take(index.tryResolve(fixture.structuredA, fixture.joined));
  if (!lineage || lineage->size() != 4)
    fail("reconvergent lineage did not retain exactly the rooted A edges");
  for (const DataflowRewriteDerivation &edge : *lineage)
    if (edge.parent == fixture.foreign || edge.parent == fixture.rootB)
      fail("root A resolution retained an edge owned by root B");
  return std::move(*lineage);
}

void reconvergenceIsRootedAndDeterministic() {
  const Fixture fixture;
  StructuredOwnershipDataflowLineageIndex forward;
  StructuredOwnershipDataflowLineageIndex reverse;
  populateReconvergent(forward, fixture, false);
  populateReconvergent(reverse, fixture, true);
  if (resolveForA(forward, fixture) != resolveForA(reverse, fixture))
    fail("insertion order changed canonical reconvergent lineage");

  auto lineageB = take(forward.tryResolve(fixture.structuredB, fixture.joined));
  if (!lineageB || lineageB->size() != 2)
    fail("root B resolution did not isolate its exact derivation path");
  auto unrelated =
      take(forward.tryResolve(fixture.structuredA, fixture.foreign));
  if (unrelated)
    fail("root membership admitted an unrelated candidate");
}

void cyclesFailClosed() {
  const Fixture fixture;
  StructuredOwnershipDataflowLineageIndex index;
  requireSuccess(index.recordRoot(fixture.structuredA, fixture.rootA));
  const dataflow::DataflowRewriteDecision firstKind =
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)};
  const dataflow::DataflowRewriteDecision secondKind =
      dataflow::ParallelizeSerializeRoundTripRewrite{dataflow::ActorId(2)};
  requireSuccess(
      index.recordDecision(fixture.rootA, fixture.cycleA, firstKind));
  requireSuccess(
      index.recordDecision(fixture.cycleA, fixture.cycleB, secondKind));
  requireSuccess(
      index.recordDecision(fixture.cycleB, fixture.cycleA, secondKind));
  auto cyclic = index.tryResolve(fixture.structuredA, fixture.cycleB);
  if (cyclic)
    fail("active invocation lineage accepted a cycle");
  requireErrorContains(cyclic.takeError(), "cycle");
}

void lateRootPathReachesExistingDescendants() {
  const Fixture fixture;
  StructuredOwnershipDataflowLineageIndex index;
  const dataflow::DataflowRewriteDecision firstKind =
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)};
  const dataflow::DataflowRewriteDecision secondKind =
      dataflow::ParallelizeSerializeRoundTripRewrite{dataflow::ActorId(2)};

  requireSuccess(index.recordRoot(fixture.structuredA, fixture.rootA));
  requireSuccess(
      index.recordDecision(fixture.rootA, fixture.joined, firstKind));
  requireSuccess(
      index.recordDecision(fixture.joined, fixture.descendant, secondKind));
  requireSuccess(index.recordRoot(fixture.structuredB, fixture.rootB));
  requireSuccess(
      index.recordDecision(fixture.rootB, fixture.joined, secondKind));

  auto lineage =
      take(index.tryResolve(fixture.structuredB, fixture.descendant));
  if (!lineage || lineage->size() != 2)
    fail("late root path did not reach an existing descendant");
  if (!std::any_of(lineage->begin(), lineage->end(),
                   [&](const DataflowRewriteDerivation &edge) {
                     return edge.parent == fixture.rootB &&
                            edge.child == fixture.joined;
                   }) ||
      !std::any_of(lineage->begin(), lineage->end(),
                   [&](const DataflowRewriteDerivation &edge) {
                     return edge.parent == fixture.joined &&
                            edge.child == fixture.descendant;
                   }))
    fail("late root resolution did not retain its exact canonical path");
}

void sharedDescendantScalesAcrossLateRoots() {
  constexpr std::uint64_t rootCount = 128;
  StructuredOwnershipDataflowLineageIndex index;
  const dataflow::DataflowRewriteDecision firstKind =
      dataflow::PackUnpackRoundTripRewrite{dataflow::ActorId(1)};
  const dataflow::DataflowRewriteDecision secondKind =
      dataflow::ParallelizeSerializeRoundTripRewrite{dataflow::ActorId(2)};
  const ArtifactRootReference shared =
      makeReference(dataflow::canonicalDataflowSchema, 1000);
  const ArtifactRootReference descendant =
      makeReference(dataflow::canonicalDataflowSchema, 1001);

  for (std::uint64_t ordinal = 0; ordinal < rootCount; ++ordinal) {
    const ArtifactRootReference structured = makeReference(
        loom::frontend::structuredProgramArtifactSchema, 2000 + ordinal);
    const ArtifactRootReference root =
        makeReference(dataflow::canonicalDataflowSchema, 3000 + ordinal);
    requireSuccess(index.recordRoot(structured, root));
    requireSuccess(index.recordDecision(root, shared, firstKind));
    if (ordinal == 0)
      requireSuccess(index.recordDecision(shared, descendant, secondKind));
  }

  for (std::uint64_t ordinal = 0; ordinal < rootCount; ++ordinal) {
    const ArtifactRootReference structured = makeReference(
        loom::frontend::structuredProgramArtifactSchema, 2000 + ordinal);
    auto lineage = take(index.tryResolve(structured, descendant));
    if (!lineage || lineage->size() != 2)
      fail("shared descendant lost one of its late rooted paths");
  }
}

} // namespace

int main() {
  reconvergenceIsRootedAndDeterministic();
  cyclesFailClosed();
  lateRootPathReachesExistingDescendants();
  sharedDescendantScalesAcrossLateRoots();
  return EXIT_SUCCESS;
}
