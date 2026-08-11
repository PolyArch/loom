#include "DSE/PortableSystemRtlCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/HardwareImplementation.h"

#include "MappedRtlSimulationTestSupport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "portable System RTL generator test failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T>
void requireError(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail("expected an error containing '" + expected + "'");
  std::string message = llvm::toString(value.takeError());
  if (!llvm::StringRef(message).contains(expected))
    fail("unexpected error: " + message);
}

const loom::dse::CompletedCandidateGeneratorResult &
completed(const loom::dse::CandidateGeneratorProviderResult &result) {
  const auto *value = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &result.outcome);
  if (!value)
    fail("provider did not complete");
  return *value;
}

} // namespace

int main() {
  const llvm::StringRef test = "portable_system_rtl_generator";
  loom::deployment::test::TemporaryTree tree(test);
  loom::ArtifactStore artifacts(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));

  auto fixture = loom::eda::test::buildMappedRtlRequestFixture(
      test, "verilator-test-build", artifacts, blobs, tree);
  const loom::hardware::HardwareImplementation &expected =
      fixture.implementation.implementation();

  auto config = take(loom::dse::resolvePortableSystemRtlConfig());
  auto binding = take(
      loom::dse::resolvePortableSystemRtlCandidateGeneratorBinding(config));
  auto inputs = take(loom::dse::bindPortableSystemRtlCandidateGeneratorInputs(
      expected.fabric(), expected.configurationAbi(),
      expected.interconnectImplementations()));

  auto first = take(
      loom::dse::invokeCandidateGenerator(inputs, binding, artifacts, blobs));
  const auto &firstCompleted = completed(first);
  require(firstCompleted.outputBindings.size() == 1 &&
              firstCompleted.outputBindings.front().artifacts.size() == 1,
          "generator did not publish exactly one implementation");
  require(firstCompleted.lineageEdges.size() == 1,
          "generator did not publish one mechanical lineage edge");
  const loom::ArtifactRootReference result =
      firstCompleted.outputBindings.front().artifacts.front();
  require(result == fixture.implementation.reference(),
          "portable derivation did not converge to the existing HImpl");

  auto imported = take(
      loom::hardware::importHardwareImplementation(result, artifacts, blobs));
  require(imported.implementation().fabric() == expected.fabric() &&
              imported.implementation().configurationAbi() ==
                  expected.configurationAbi(),
          "generated HImpl lost its exact System or ABI owner");

  auto repeated = take(
      loom::dse::invokeCandidateGenerator(inputs, binding, artifacts, blobs));
  require(completed(repeated).outputBindings.front().artifacts.front() ==
              result,
          "repeated portable derivation changed Artifact identity");
  require(repeated.workSummary == first.workSummary,
          "repeated portable derivation changed deterministic work");

  auto wrongRoot =
      take(loom::dse::bindPortableSystemRtlCandidateGeneratorInputs(
          fixture.module, expected.configurationAbi(), {}));
  requireError(
      loom::dse::invokeCandidateGenerator(wrongRoot, binding, artifacts, blobs),
      "complete System root");

  std::vector<std::uint8_t> nonemptyConfig = {0};
  auto digest = take(loom::computeComponentViewDigest(
      loom::dse::resolvedPortableSystemRtlConfigSchemaBytes(), nonemptyConfig));
  requireError(loom::dse::adoptResolvedPortableSystemRtlConfigView(
                   loom::dse::resolvedPortableSystemRtlConfigSchemaBytes(),
                   nonemptyConfig, digest),
               "must be empty");
  return EXIT_SUCCESS;
}
