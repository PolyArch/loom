#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "MappedRtlSimulationTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "portable SpatialCore RTL generator test failed: " << message
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

bool stopRequested(const void *context) {
  return *static_cast<const bool *>(context);
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

const loom::dse::IncompleteCandidateGeneratorResult &
incomplete(const loom::dse::CandidateGeneratorProviderResult &result) {
  const auto *value =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &result.outcome);
  if (!value)
    fail("provider did not return an incomplete outcome");
  return *value;
}

loom::fabric::FinalizedFabricRoot
makeUnsupportedResetSystem(const loom::fabric::FinalizedFabricRoot &module,
                           const loom::ArtifactStore &store) {
  loom::deployment::test::MappedSpatialSystemSpec spec;
  spec.accCoreCount = 1;
  spec.attachSystemMemory = true;
  spec.resetInitialState = loom::fabric::ResetInitialState::Deasserted;
  return loom::deployment::test::buildMappedSpatialSystem(
      "unsupported-reset-system", module, {}, store, spec);
}

} // namespace

int main() {
  const llvm::StringRef test = "portable_spatial_core_rtl_generator";
  loom::deployment::test::TemporaryTree tree(test);
  loom::ArtifactStore artifacts(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));

  auto fixture = loom::eda::test::buildMappedRtlRequestFixture(
      test, "verilator-test-build", artifacts, blobs, tree);
  const loom::hardware::HardwareImplementation &expected =
      fixture.implementation.implementation();
  auto system =
      take(loom::fabric::importEntireFabricRoot(expected.fabric(), artifacts));
  const auto accCores = system.view().accCoreOccurrences();

  auto config = take(loom::dse::resolvePortableSpatialCoreRtlConfig());
  auto binding =
      take(loom::dse::resolvePortableSpatialCoreRtlCandidateGeneratorBinding(
          config));
  auto inputs =
      take(loom::dse::bindPortableSpatialCoreRtlCandidateGeneratorInputs(
          expected.fabric(), expected.configurationAbi()));

  auto first = take(
      loom::dse::invokeCandidateGenerator(inputs, binding, artifacts, blobs));
  const auto &firstCompleted = completed(first);
  require(firstCompleted.outputBindings.size() == 1 &&
              firstCompleted.outputBindings.front().artifacts.size() ==
                  accCores.size(),
          "generator did not publish one implementation per SpatialCore");
  require(firstCompleted.lineageEdges.size() == accCores.size(),
          "generator did not publish one lineage edge per SpatialCore");
  const auto &results = firstCompleted.outputBindings.front().artifacts;
  bool retainedExistingImplementation = false;
  std::vector<loom::fabric::SpatialCoreOccurrenceRef> generatedSubjects;
  generatedSubjects.reserve(results.size());
  for (const loom::ArtifactRootReference &result : results) {
    retainedExistingImplementation |=
        result == fixture.implementation.reference();
    auto imported = take(
        loom::hardware::importHardwareImplementation(result, artifacts, blobs));
    require(imported.implementation().fabric() == expected.fabric() &&
                imported.implementation().configurationAbi() ==
                    expected.configurationAbi(),
            "generated HImpl lost its exact System or ABI owner");
    generatedSubjects.push_back(imported.implementation().subject());
  }
  require(retainedExistingImplementation,
          "portable derivation did not converge to the existing HImpl");
  for (loom::fabric::AccCoreOccurrenceRef accCore : accCores) {
    const loom::fabric::SpatialCoreOccurrenceRef expectedSubject{accCore};
    std::size_t matches = 0;
    for (loom::fabric::SpatialCoreOccurrenceRef subject : generatedSubjects)
      matches += subject == expectedSubject;
    require(matches == 1,
            "portable derivation did not exactly cover a SpatialCore");
  }

  auto repeated = take(
      loom::dse::invokeCandidateGenerator(inputs, binding, artifacts, blobs));
  require(completed(repeated).outputBindings.front().artifacts == results,
          "repeated portable derivation changed Artifact identity");
  require(repeated.workSummary == first.workSummary,
          "repeated portable derivation changed deterministic work");

  require(accCores.size() > 1,
          "portable RTL policy evidence requires multiple SpatialCores");
  const std::array<loom::dse::CandidateGeneratorOutputDemand, 1> limits = {{
      {loom::dse::CandidateGeneratorOutputSlotRef(0), 1},
  }};
  auto limitedResult = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, artifacts, blobs,
      loom::dse::CandidateGeneratorInvocationView({}, limits)));
  const auto &limited = incomplete(limitedResult);
  require(limited.reason == loom::dse::CandidateGeneratorIncompleteReason::
                                SemanticLimitReached &&
              limited.retainedOutputBindings.front().artifacts.size() == 1 &&
              limited.lineageEdges.size() == 1 &&
              limitedResult.workSummary.front().planned == accCores.size() &&
              limitedResult.workSummary.front().consumed == 1,
          "portable RTL ignored its output demand");

  const bool stop = true;
  auto cancelledResult = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, artifacts, blobs,
      loom::ExecutionControlView(&stop, stopRequested)));
  const auto &cancelled = incomplete(cancelledResult);
  require(cancelled.reason == loom::dse::CandidateGeneratorIncompleteReason::
                                  CancelledOrTimeout &&
              cancelled.retainedOutputBindings.front().artifacts.empty() &&
              cancelled.lineageEdges.empty() &&
              cancelledResult.workSummary.front().planned == accCores.size() &&
              cancelledResult.workSummary.front().consumed == 0,
          "portable RTL ignored cancellation before work");

  auto module =
      take(loom::fabric::importEntireFabricRoot(fixture.module, artifacts));
  auto unsupportedSystem = makeUnsupportedResetSystem(module, artifacts);
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto unsupportedAbiDraft =
      take(loom::hardware::derivePackedConfigurationABIDraft(unsupportedSystem,
                                                             context));
  auto unsupportedAbi = take(loom::hardware::finalizeConfigurationABI(
      std::move(unsupportedAbiDraft), artifacts));
  auto unsupportedInputs =
      take(loom::dse::bindPortableSpatialCoreRtlCandidateGeneratorInputs(
          unsupportedSystem.reference(), unsupportedAbi.reference()));
  auto unsupportedResult = take(loom::dse::invokeCandidateGenerator(
      unsupportedInputs, binding, artifacts, blobs));
  const auto &unsupported = incomplete(unsupportedResult);
  require(unsupported.reason ==
                  loom::dse::CandidateGeneratorIncompleteReason::Unsupported &&
              unsupported.retainedOutputBindings.front().artifacts.empty() &&
              unsupportedResult.workSummary.size() == 1 &&
              unsupportedResult.workSummary.front().planned == 1 &&
              unsupportedResult.workSummary.front().consumed == 1,
          "portable RTL did not retain typed unsupported work accounting");

  auto platform = take(loom::platform::finalizeImplementationPlatform(
      loom::platform::ImplementationPlatformDraft{
          loom::platform::AsicTarget{"portable-rtl-test", "2026.08"},
          {"typical"}},
      artifacts));
  auto platformInputs =
      take(loom::dse::bindPortableSpatialCoreRtlCandidateGeneratorInputs(
          expected.fabric(), expected.configurationAbi(),
          platform.reference()));
  auto platformResult = take(loom::dse::invokeCandidateGenerator(
      platformInputs, binding, artifacts, blobs));
  const auto &platformImplementations =
      completed(platformResult).outputBindings.front().artifacts;
  require(platformImplementations.size() == accCores.size() &&
              platformImplementations != results,
          "platform-bound derivation did not publish a distinct complete set");
  for (const loom::ArtifactRootReference &result : platformImplementations) {
    auto imported = take(
        loom::hardware::importHardwareImplementation(result, artifacts, blobs));
    require(imported.implementation().fabric() == expected.fabric() &&
                imported.implementation().configurationAbi() ==
                    expected.configurationAbi() &&
                imported.implementation().implementationPlatform() ==
                    std::optional<loom::ArtifactRootReference>(
                        platform.reference()),
            "platform-bound HImpl lost an exact semantic owner");
  }

  auto wrongRoot =
      take(loom::dse::bindPortableSpatialCoreRtlCandidateGeneratorInputs(
          fixture.module, expected.configurationAbi()));
  requireError(
      loom::dse::invokeCandidateGenerator(wrongRoot, binding, artifacts, blobs),
      "complete System root");

  std::vector<std::uint8_t> nonemptyConfig = {0};
  auto digest = take(loom::computeComponentViewDigest(
      loom::dse::resolvedPortableSpatialCoreRtlConfigSchemaBytes(),
      nonemptyConfig));
  requireError(loom::dse::adoptResolvedPortableSpatialCoreRtlConfigView(
                   loom::dse::resolvedPortableSpatialCoreRtlConfigSchemaBytes(),
                   nonemptyConfig, digest),
               "must be empty");
  return EXIT_SUCCESS;
}
