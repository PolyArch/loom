#include "DSE/FuReverseSynthesis.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactFinalizer.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr std::uint64_t reverseFuSystemClockPeriodFs = 1'000;
constexpr std::uint32_t reverseFuResidentRouteCapacity = 32;

llvm::Error failure(const llvm::Twine &message) {
  return llvm::make_error<FuReverseSynthesisError>(
      FuReverseSynthesisFailure::FabricFinalizationFailed, message.str());
}

llvm::Expected<::fabric::ResourceContract> messageTransportContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0),
         ::fabric::CapacityUnits(reverseFuResidentRouteCapacity),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return ::fabric::ResourceContract::create(std::move(declaration));
}

llvm::Expected<::loom::adg::DesignBuilder>
buildSystemDesign(const ::loom::fabric::FinalizedFabricRoot &module,
                  const ArtifactStore &store) {
  if (module.view().rootKind() != ::loom::fabric::FabricRootKind::Module)
    return failure("bounded reverse synthesis System requires a Module root");

  ::loom::adg::DesignBuilder design(store);
  auto system = design.createSystem("scalar-add-sub-synthesis-system");
  if (!system)
    return failure(llvm::toString(system.takeError()));
  auto imported = system->importSpatialCore(module);
  if (!imported)
    return failure(llvm::toString(imported.takeError()));
  auto architecture = ::loom::adg::getBuiltinInstructionCoreArchitecture();
  if (!architecture)
    return failure(llvm::toString(architecture.takeError()));
  auto microarchitecture =
      ::loom::adg::getBuiltinInOrderInstructionCoreMicroarchitecture();
  if (!microarchitecture)
    return failure(llvm::toString(microarchitecture.takeError()));
  auto host = system->addHostCore(*architecture, *microarchitecture);
  if (!host)
    return failure(llvm::toString(host.takeError()));
  auto accCore =
      system->addAccCore(*architecture, *microarchitecture, *imported);
  if (!accCore)
    return failure(llvm::toString(accCore.takeError()));

  auto clock = system->createHardwareDomain();
  if (!clock)
    return failure(llvm::toString(clock.takeError()));
  auto rate = system->createServiceRate(
      *clock, 1, 1, 1,
      ::loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>));
  if (!rate)
    return failure(llvm::toString(rate.takeError()));
  mlir::MLIRContext typeContext;
  const std::array<mlir::Type, 2> messageTypes = {
      mlir::NoneType::get(&typeContext),
      mlir::IntegerType::get(&typeContext, 32)};
  auto messageDomain =
      ::loom::fabric::MessageTransferCapabilityDomain::create(messageTypes);
  if (!messageDomain)
    return failure(llvm::toString(messageDomain.takeError()));
  auto initiate = ::loom::fabric::CanonicalServiceCapabilityRecord::create(
      ::dataflow::semantics::ServiceKind::MessageTransfer,
      ::loom::fabric::CanonicalServiceEndpointRole::Initiate, *messageDomain,
      *rate);
  if (!initiate)
    return failure(llvm::toString(initiate.takeError()));
  auto serve = ::loom::fabric::CanonicalServiceCapabilityRecord::create(
      ::dataflow::semantics::ServiceKind::MessageTransfer,
      ::loom::fabric::CanonicalServiceEndpointRole::Serve, *messageDomain,
      *rate);
  if (!serve)
    return failure(llvm::toString(serve.takeError()));
  auto initiateSet =
      ::loom::fabric::CanonicalServiceCapabilitySet::create({*initiate});
  if (!initiateSet)
    return failure(llvm::toString(initiateSet.takeError()));
  auto serveSet =
      ::loom::fabric::CanonicalServiceCapabilitySet::create({*serve});
  if (!serveSet)
    return failure(llvm::toString(serveSet.takeError()));
  auto carrier = ::loom::adg::PortType::bits(128);
  if (!carrier)
    return failure(llvm::toString(carrier.takeError()));
  auto hostSource = system->addServiceEndpoint(*host, *initiateSet, *carrier);
  if (!hostSource)
    return failure(llvm::toString(hostSource.takeError()));
  auto hostSink = system->addServiceEndpoint(*host, *serveSet, *carrier);
  if (!hostSink)
    return failure(llvm::toString(hostSink.takeError()));
  auto accSource = system->addServiceEndpoint(*accCore, *initiateSet, *carrier);
  if (!accSource)
    return failure(llvm::toString(accSource.takeError()));
  auto accSink = system->addServiceEndpoint(*accCore, *serveSet, *carrier);
  if (!accSink)
    return failure(llvm::toString(accSink.takeError()));
  auto transportContract = messageTransportContract();
  if (!transportContract)
    return failure(llvm::toString(transportContract.takeError()));

  const std::array sources = {*hostSource, *accSource};
  const std::array sinks = {*hostSink, *accSink};
  const std::array<std::vector<std::uint32_t>, 3> transferPatterns = {
      std::vector<std::uint32_t>{0}, std::vector<std::uint32_t>{1},
      std::vector<std::uint32_t>{0, 1}};
  std::vector<::loom::adg::SystemTransportResource> routers;
  std::vector<::loom::adg::HardwareDomainMember> clockMembers = {
      host->domainMember(), accCore->instructionCoreDomainMember(),
      accCore->spatialCoreDomainMember(
          ::loom::fabric::FabricClockResetKind::Clock)};
  std::vector<::loom::adg::HardwareDomainMember> resetMembers = {
      host->domainMember(), accCore->instructionCoreDomainMember(),
      accCore->spatialCoreResetDomainMember()};
  routers.reserve(sources.size());
  for (std::size_t ordinal = 0; ordinal != sources.size(); ++ordinal) {
    clockMembers.push_back(sources[ordinal].domainMember());
    clockMembers.push_back(sinks[ordinal].domainMember());
    resetMembers.push_back(sources[ordinal].domainMember());
    resetMembers.push_back(sinks[ordinal].domainMember());
    auto router = system->addTransportResource(
        {{*carrier, *carrier},
         {*carrier, *carrier},
         *transportContract,
         ::loom::adg::SystemTransferPatternSelection::Configuration});
    if (!router)
      return failure(llvm::toString(router.takeError()));
    routers.push_back(*router);
    clockMembers.push_back(router->domainMember());
    resetMembers.push_back(router->domainMember());
    for (std::size_t input = 0; input != 2; ++input)
      for (const auto &outputs : transferPatterns) {
        auto pattern = system->addTransferPattern(*router, input, outputs, 0);
        if (!pattern)
          return failure(llvm::toString(pattern.takeError()));
        clockMembers.push_back(pattern->domainMember());
        resetMembers.push_back(pattern->domainMember());
      }
    auto source = sources[ordinal].transport();
    auto input = router->input(0);
    auto output = router->output(0);
    auto sink = sinks[ordinal].transport();
    if (!source)
      return failure(llvm::toString(source.takeError()));
    if (!input)
      return failure(llvm::toString(input.takeError()));
    if (!output)
      return failure(llvm::toString(output.takeError()));
    if (!sink)
      return failure(llvm::toString(sink.takeError()));
    if (llvm::Error error = system->connect(*source, *input))
      return failure(llvm::toString(std::move(error)));
    if (llvm::Error error = system->connect(*output, *sink))
      return failure(llvm::toString(std::move(error)));
  }
  for (std::size_t ordinal = 0; ordinal != routers.size(); ++ordinal) {
    auto output = routers[ordinal].output(1);
    auto input = routers[(ordinal + 1) % routers.size()].input(1);
    if (!output)
      return failure(llvm::toString(output.takeError()));
    if (!input)
      return failure(llvm::toString(input.takeError()));
    if (llvm::Error error = system->connect(*output, *input))
      return failure(llvm::toString(std::move(error)));
  }

  auto clockContract = ::loom::fabric::ClockDomainContractRecord::create(
      reverseFuSystemClockPeriodFs, 0);
  if (!clockContract)
    return failure(llvm::toString(clockContract.takeError()));
  if (llvm::Error error = clock->close(clockMembers, *clockContract))
    return failure(llvm::toString(std::move(error)));

  auto reset = system->createHardwareDomain();
  if (!reset)
    return failure(llvm::toString(reset.takeError()));
  auto resetContract = ::loom::fabric::ResetDomainContractRecord::create(
      ::loom::fabric::ResetPolarity::ActiveHigh,
      ::loom::fabric::ResetTiming::Synchronous,
      ::loom::fabric::ResetTiming::Synchronous,
      ::loom::fabric::ResetInitialState::Asserted,
      ::loom::fabric::ClockDomainRef(clock->reference()), 0);
  if (!resetContract)
    return failure(llvm::toString(resetContract.takeError()));
  if (llvm::Error error = reset->close(resetMembers, *resetContract))
    return failure(llvm::toString(std::move(error)));
  if (llvm::Error error = system->close())
    return failure(llvm::toString(std::move(error)));
  return std::move(design);
}

mlir::DialectRegistry configurationDialects() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  return registry;
}

} // namespace

llvm::Expected<ScalarIntegerAddSubFuSystemArtifacts>
materializeScalarIntegerAddSubFuSystemArtifacts(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ArtifactStore &store) {
  auto design = buildSystemDesign(module, store);
  if (!design)
    return design.takeError();
  auto finalized = std::move(*design).finalize();
  if (!finalized)
    return failure(llvm::toString(finalized.takeError()));
  if (finalized->roots().size() != 1)
    return failure("bounded reverse synthesis did not publish one System");
  ::loom::fabric::FinalizedFabricRoot system = finalized->roots().front();
  auto systemView = ::loom::fabric::requireSystemRoot(system.view());
  if (!systemView)
    return failure(llvm::toString(systemView.takeError()));

  auto profiles = ::loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(
      *systemView);
  if (!profiles)
    return failure(llvm::toString(profiles.takeError()));
  if (profiles->size() != 1)
    return failure("bounded reverse synthesis System has no unique Module "
                   "timing profile");
  auto profile = ::loom::fabric::publishFabricPhysicalTimingProfile(
      profiles->front(), store);
  if (!profile)
    return failure(llvm::toString(profile.takeError()));

  mlir::DialectRegistry registry = configurationDialects();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto abiDraft =
      ::loom::hardware::derivePackedConfigurationABIDraft(system, context);
  if (!abiDraft)
    return failure(llvm::toString(abiDraft.takeError()));
  auto abi =
      ::loom::hardware::finalizeConfigurationABI(std::move(*abiDraft), store);
  if (!abi)
    return failure(llvm::toString(abi.takeError()));
  return ScalarIntegerAddSubFuSystemArtifacts(
      std::move(system), std::move(*profile), std::move(*abi));
}

llvm::Error verifyScalarIntegerAddSubFuSystemLineage(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ScalarIntegerAddSubFuSystemArtifacts &artifacts,
    const ArtifactStore &store) {
  if (llvm::Error error = verifyScalarIntegerAddSubFuSystemIdentity(
          module, artifacts.system(), store))
    return error;
  if (llvm::Error error = verifyScalarIntegerAddSubFuPhysicalTimingLineage(
          module, artifacts.physicalTimingProfile(), store))
    return error;
  return verifyScalarIntegerAddSubFuConfigurationAbiLineage(
      artifacts.system(), artifacts.configurationAbi().reference(), store);
}

llvm::Error verifyScalarIntegerAddSubFuSystemIdentity(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ::loom::fabric::FinalizedFabricRoot &system,
    const ArtifactStore &store) {
  auto expectedDesign = buildSystemDesign(module, store);
  if (!expectedDesign)
    return expectedDesign.takeError();
  auto expected = std::move(*expectedDesign).deriveRootIdentities();
  if (!expected)
    return expected.takeError();
  if (expected->size() != 1 || expected->front() != system.reference().artifact)
    return failure("System lineage is not the exact bounded execution shell");

  auto importedSystem =
      ::loom::fabric::importEntireFabricRoot(system.reference(), store);
  if (!importedSystem)
    return importedSystem.takeError();
  auto systemView = ::loom::fabric::requireSystemRoot(importedSystem->view());
  if (!systemView)
    return systemView.takeError();
  if (systemView->artifact().accCoreOccurrences().size() != 1)
    return failure("bounded System lineage has no unique AccCore");
  auto target = systemView->spatialCoreTarget(
      systemView->artifact().accCoreOccurrences().front());
  if (!target ||
      target->dependencyOrdinal >=
          systemView->artifact().importedModules().size() ||
      systemView->artifact()
              .importedModules()[target->dependencyOrdinal]
              .identity() != module.reference().artifact)
    return failure("bounded System lineage selects another SpatialCore Module");

  return llvm::Error::success();
}

llvm::Error verifyScalarIntegerAddSubFuPhysicalTimingLineage(
    const ::loom::fabric::FinalizedFabricRoot &module,
    const ArtifactRootReference &physicalTimingProfile,
    const ArtifactStore &store) {
  auto expected = ::loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
      module.view());
  if (!expected)
    return expected.takeError();
  const ArtifactRootReference expectedReference{
      ::loom::fabric::fabricPhysicalTimingProfileArtifactSchema.identity.str(),
      ::loom::fabric::fabricPhysicalTimingProfileArtifactSchema.version,
      finalizeArtifactIdentity(
          ::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
          CanonicalSemanticBytes(expected->canonicalViewBytes().vec()))};
  if (physicalTimingProfile != expectedReference)
    return failure("bounded System timing lineage is not the exact normalized "
                   "Module projection");
  auto imported = ::loom::fabric::importFabricPhysicalTimingProfile(
      physicalTimingProfile, module.view(), store);
  if (!imported)
    return imported.takeError();
  return llvm::Error::success();
}

llvm::Error verifyScalarIntegerAddSubFuConfigurationAbiLineage(
    const ::loom::fabric::FinalizedFabricRoot &system,
    const ArtifactRootReference &configurationAbi, const ArtifactStore &store) {
  mlir::DialectRegistry registry = configurationDialects();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto draft =
      ::loom::hardware::derivePackedConfigurationABIDraft(system, context);
  if (!draft)
    return draft.takeError();
  auto expected = ::loom::hardware::deriveConfigurationABIArtifactReference(
      std::move(*draft), store);
  if (!expected)
    return expected.takeError();
  if (configurationAbi != *expected)
    return failure("bounded ConfigurationABI lineage is not the exact packed "
                   "System projection");
  ::loom::hardware::ConfigurationABIImportSession abiSession(
      ::loom::hardware::ConfigurationABIImportSessionMode::Isolated);
  auto imported =
      ::loom::hardware::importConfigurationABI(configurationAbi, store);
  if (!imported)
    return imported.takeError();
  return llvm::Error::success();
}

} // namespace loom::dse
