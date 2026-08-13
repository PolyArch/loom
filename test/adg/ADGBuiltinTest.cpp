#include "ADGBuilderTestSupport.h"

#include "ADG/Builtin.h"
#include "ADG/FuLibrary.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace loom::adg::test {

template <typename Ref>
void requireExactCanonicalEntityRange(
    llvm::StringRef test, const loom::fabric::FabricArtifactView &view,
    llvm::ArrayRef<Ref> actual, loom::fabric::FabricEntityKind expectedKind) {
  std::vector<Ref> expected;
  for (std::uint64_t id = 0;; ++id) {
    const auto kind = view.entityKind(id);
    if (!kind)
      break;
    if (*kind == expectedKind)
      expected.emplace_back(id);
  }
  require(test, llvm::equal(actual, expected),
          "typed System range changed its exact canonical entity sequence");
}

void builtinPresetsExpandThroughPublicBuilder() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const auto architecture =
      take(test, loom::adg::getBuiltinInstructionCoreArchitecture());
  const std::array expectedExtensions{loom::fabric::RiscVExtension::M,
                                      loom::fabric::RiscVExtension::A,
                                      loom::fabric::RiscVExtension::F,
                                      loom::fabric::RiscVExtension::D,
                                      loom::fabric::RiscVExtension::C,
                                      loom::fabric::RiscVExtension::Zicsr,
                                      loom::fabric::RiscVExtension::Zifencei};
  require(test,
          llvm::equal(architecture.extensions(), expectedExtensions) &&
              llvm::equal(architecture.abiCapabilities(),
                          std::array{loom::fabric::RiscVAbi::Lp64d}),
          "builtin InstructionCore does not cover its exact compiler target");
  struct Expectation {
    loom::adg::BuiltinTargetPreset preset;
    std::uint32_t accCores;
    std::uint32_t spatialPes;
    std::uint32_t temporalPes;
    std::uint32_t spatialMemories;
    std::uint32_t temporalMemories;
    std::uint32_t meshDimension;
  };
  const std::array<Expectation, 3> expectations{{
      {loom::adg::BuiltinTargetPreset::Small, 4, 12, 4, 1, 1, 4},
      {loom::adg::BuiltinTargetPreset::Default, 8, 27, 9, 2, 2, 6},
      {loom::adg::BuiltinTargetPreset::Large, 16, 48, 16, 4, 4, 8},
  }};

  for (const Expectation &expected : expectations) {
    const auto &descriptor =
        loom::adg::getBuiltinTargetDescriptor(expected.preset);
    require(
        test,
        descriptor.scale.accCoreCount == expected.accCores &&
            descriptor.scale.spatialPeCount == expected.spatialPes &&
            descriptor.scale.temporalPeCount == expected.temporalPes &&
            descriptor.scale.spatialMemoryCount == expected.spatialMemories &&
            descriptor.scale.temporalMemoryCount == expected.temporalMemories,
        "builtin descriptor changed its scale contract");
    require(test, descriptor.schemaMajor == 3 && descriptor.schemaMinor == 0,
            "builtin descriptor did not select the buffered gateway recipe");

    auto target =
        take(test, loom::adg::buildBuiltinTarget(store, expected.preset));
    require(test, target.roots().size() == 1,
            "builtin expansion did not publish one System root");
    const auto &root = target.roots().front();
    require(
        test,
        root.view().rootKind() == loom::fabric::FabricRootKind::System &&
            root.directDependencies().size() == 1 &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::AccCoreOccurrence) ==
                expected.accCores &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::HostCoreOccurrence) ==
                1 &&
            entityCount(root.view(),
                        loom::fabric::FabricEntityKind::SystemMemoryService) ==
                1 &&
            entityCount(
                root.view(),
                loom::fabric::FabricEntityKind::SystemServiceEndpoint) ==
                1 + 2 * (expected.accCores + 1),
        "builtin lost its SpatialCore, AccCore, or System memory inventory");

    auto systemView = take(test, loom::fabric::requireSystemRoot(root.view()));
    std::optional<loom::fabric::HardwareDomainRef> clockDomain;
    std::optional<loom::fabric::HardwareDomainRef> resetDomain;
    const loom::fabric::ClockDomainContractRecord *clockContract = nullptr;
    const loom::fabric::ResetDomainContractRecord *resetContract = nullptr;
    for (const auto domain : systemView.hardwareDomains()) {
      const auto *contract = systemView.hardwareDomainContract(domain);
      require(test, contract != nullptr,
              "builtin hardware domain has no contract");
      if (const auto *clock =
              std::get_if<loom::fabric::ClockDomainContractRecord>(
                  &contract->contract())) {
        require(test, !clockDomain,
                "builtin declares more than one Clock domain");
        clockDomain = domain;
        clockContract = clock;
      } else if (const auto *reset =
                     std::get_if<loom::fabric::ResetDomainContractRecord>(
                         &contract->contract())) {
        require(test, !resetDomain,
                "builtin declares more than one Reset domain");
        resetDomain = domain;
        resetContract = reset;
      } else {
        fail(test, "builtin declares an unexpected hardware domain kind");
      }
    }
    require(test,
            clockDomain && resetDomain && clockContract && resetContract &&
                clockContract->periodFs() == 1'000 &&
                clockContract->phaseFs() == 0 &&
                resetContract->polarity() ==
                    loom::fabric::ResetPolarity::ActiveHigh &&
                resetContract->assertion() ==
                    loom::fabric::ResetTiming::Synchronous &&
                resetContract->deassertion() ==
                    loom::fabric::ResetTiming::Synchronous &&
                resetContract->initialState() ==
                    loom::fabric::ResetInitialState::Asserted &&
                resetContract->synchronousTo() ==
                    std::optional<loom::fabric::ClockDomainRef>(
                        loom::fabric::ClockDomainRef(*clockDomain)) &&
                resetContract->releaseLatencyCycles() == 0 &&
                llvm::equal(systemView.hardwareDomainMembers(*clockDomain),
                            systemView.hardwareDomainMembers(*resetDomain)),
            "builtin lost its exact Clock and Reset contract");
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().hostCoreOccurrences(),
        loom::fabric::FabricEntityKind::HostCoreOccurrence);
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().accCoreOccurrences(),
        loom::fabric::FabricEntityKind::AccCoreOccurrence);
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().systemMemoryServices(),
        loom::fabric::FabricEntityKind::SystemMemoryService);
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().systemServiceEndpoints(),
        loom::fabric::FabricEntityKind::SystemServiceEndpoint);
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().systemServiceTransforms(),
        loom::fabric::FabricEntityKind::SystemServiceTransform);
    requireExactCanonicalEntityRange(
        test, root.view(), systemView.artifact().externalBoundaries(),
        loom::fabric::FabricEntityKind::ExternalBoundary);
    const auto systemMemory =
        systemView.artifact().systemMemoryServices().front();
    const auto serviceEndpoint =
        systemView.artifact().systemServiceEndpoints().front();
    const auto *memoryService = systemView.memoryService(systemMemory);
    const auto *endpointOwner =
        systemView.serviceEndpointOwner(serviceEndpoint);
    const auto *endpointCapabilities =
        systemView.serviceEndpointCapabilities(serviceEndpoint);
    bool supportsRead = false;
    bool supportsWrite = false;
    if (endpointCapabilities)
      for (const auto &capability : endpointCapabilities->capabilities()) {
        supportsRead |=
            capability.kind() == dataflow::semantics::ServiceKind::MemoryRead;
        supportsWrite |=
            capability.kind() == dataflow::semantics::ServiceKind::MemoryWrite;
      }
    require(
        test,
        memoryService && memoryService->regions().size() == 1 &&
            memoryService->capabilities().size() == 2 && endpointOwner &&
            endpointOwner->owner() ==
                loom::fabric::FabricInventoryOwnerRef::of(
                    loom::fabric::FabricMemoryServiceRef::system(
                        systemMemory)) &&
            endpointCapabilities &&
            endpointCapabilities->role() ==
                loom::fabric::CanonicalServiceEndpointRole::Serve &&
            endpointCapabilities->plane() ==
                loom::fabric::CanonicalServiceEndpointPlane::Memory &&
            endpointCapabilities->capabilities().size() == 2 && supportsRead &&
            supportsWrite &&
            !systemView.memoryService(
                loom::fabric::SystemMemoryServiceRef(serviceEndpoint.id())) &&
            !systemView.serviceEndpointOwner(
                loom::fabric::SystemServiceEndpointRef(systemMemory.id())),
        "builtin System view lost its exact memory service capability owner");
    const loom::fabric::HostCoreOccurrenceRef host(uniqueEntity(
        test, root.view(), loom::fabric::FabricEntityKind::HostCoreOccurrence));
    const auto *hostArchitecture = systemView.instructionCoreArchitecture(host);
    const auto *hostMicroarchitecture =
        systemView.instructionCoreMicroarchitecture(host);
    require(test,
            hostArchitecture && hostMicroarchitecture &&
                hostArchitecture->xlen() == loom::fabric::RiscVXLen::X64 &&
                llvm::is_contained(hostArchitecture->abiCapabilities(),
                                   loom::fabric::RiscVAbi::Lp64d),
            "builtin HostCore lost its exact InstructionCore contracts");
    std::size_t projectedAccCores = 0;
    for (std::uint64_t id = 0;; ++id) {
      const auto kind = root.view().entityKind(id);
      if (!kind)
        break;
      if (*kind != loom::fabric::FabricEntityKind::AccCoreOccurrence)
        continue;
      const loom::fabric::InstructionCoreContextRef instruction{
          loom::fabric::AccCoreOccurrenceRef(id)};
      const auto *coreArchitecture =
          systemView.instructionCoreArchitecture(instruction);
      const auto *coreMicroarchitecture =
          systemView.instructionCoreMicroarchitecture(instruction);
      require(test,
              coreArchitecture && coreMicroarchitecture &&
                  coreArchitecture->xlen() == hostArchitecture->xlen() &&
                  coreArchitecture->endianness() ==
                      hostArchitecture->endianness() &&
                  llvm::is_contained(coreArchitecture->abiCapabilities(),
                                     loom::fabric::RiscVAbi::Lp64d),
              "builtin AccCore left the HostCore ISA and ABI cohort");
      ++projectedAccCores;
    }
    require(test, projectedAccCores == expected.accCores,
            "builtin System view lost an AccCore InstructionCore contract");
    std::size_t hostMessageEndpoints = 0;
    std::size_t accMessageEndpoints = 0;
    for (const auto endpoint : systemView.artifact().systemServiceEndpoints()) {
      const auto *capabilities =
          systemView.serviceEndpointCapabilities(endpoint);
      if (!capabilities ||
          capabilities->plane() !=
              loom::fabric::CanonicalServiceEndpointPlane::Transport)
        continue;
      require(test,
              capabilities->capabilities().size() == 1 &&
                  capabilities->capabilities().front().kind() ==
                      dataflow::semantics::ServiceKind::MessageTransfer,
              "builtin message endpoint has a foreign capability");
      const auto *owner = systemView.serviceEndpointOwner(endpoint);
      require(test, owner, "builtin message endpoint has no owner");
      if (std::holds_alternative<loom::fabric::HostCoreOccurrenceRef>(
              owner->owner().payload))
        ++hostMessageEndpoints;
      else if (std::holds_alternative<loom::fabric::AccCoreOccurrenceRef>(
                   owner->owner().payload))
        ++accMessageEndpoints;
      else
        fail(test, "builtin message endpoint has a nonexecution owner");
    }
    require(test,
            hostMessageEndpoints == 2 &&
                accMessageEndpoints == 2 * expected.accCores,
            "builtin execution owners lack paired message endpoints");
    std::size_t memoryAttachments = 0;
    for (const auto &attachment : systemView.spatialAttachments()) {
      if (attachment.spatialEndpoint.plane() !=
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory) {
        require(test, !attachment.serviceEndpoint,
                "transport attachment gained a service endpoint");
        continue;
      }
      ++memoryAttachments;
      require(test, attachment.serviceEndpoint == serviceEndpoint,
              "memory attachment lost its exact System service endpoint");
    }
    require(test, memoryAttachments == expected.accCores,
            "builtin did not attach one manager capability per AccCore");

    auto module =
        take(test, loom::fabric::importEntireFabricRoot(
                       root.directDependencies().front().root, store));
    require(test,
            module.view().hostCoreOccurrences().empty() &&
                module.view().accCoreOccurrences().empty() &&
                module.view().systemMemoryServices().empty() &&
                module.view().systemServiceEndpoints().empty() &&
                module.view().systemServiceTransforms().empty() &&
                module.view().externalBoundaries().empty(),
            "Module root exposed a System-only typed range");
    const loom::fabric::FabricModuleTemplateRef moduleTemplate(
        uniqueEntity(test, module.view(),
                     loom::fabric::FabricEntityKind::FabricModuleTemplate));
    require(test,
            entityCount(module.view(),
                        loom::fabric::FabricEntityKind::FabricPeOccurrence) ==
                    expected.spatialPes + expected.temporalPes &&
                entityCount(
                    module.view(),
                    loom::fabric::FabricEntityKind::FabricMemoryOccurrence) ==
                    expected.spatialMemories + expected.temporalMemories,
            "builtin SpatialCore lost its PE or memory scale");
    const std::uint64_t expectedMeshLinkFifos =
        16 * expected.meshDimension * (expected.meshDimension - 1);
    const std::uint64_t expectedAdapterFifos =
        3 * (expected.spatialMemories + expected.temporalMemories) +
        2 * descriptor.scale.gatewayCount;
    require(test,
            module.view().fifoOccurrences().size() ==
                expectedMeshLinkFifos + expectedAdapterFifos,
            "builtin did not emit one FIFO per mesh link and width adapter");
    const std::uint32_t expectedTagWidth = std::max(
        1U, llvm::Log2_64_Ceil(descriptor.scale.temporalResidentContexts));
    std::size_t taggedEndpoints = 0;
    for (const auto &endpoint : module.view().transportEndpoints()) {
      const auto dataPath = module.view().transportEndpointDataPath(endpoint);
      require(test, dataPath.has_value(),
              "builtin transport endpoint lost its data path");
      if (dataPath->kind != ::fabric::DataPathKind::BitsTag)
        continue;
      ++taggedEndpoints;
      require(test, dataPath->tagWidthBits == expectedTagWidth,
              "builtin temporal endpoint does not use the minimum tag width");
    }
    require(test, taggedEndpoints != 0,
            "builtin emitted no tagged temporal endpoint");

    std::size_t configuredWriters = 0;
    std::size_t tagRemovers = 0;
    for (const auto boundary : module.view().boundaryOccurrences()) {
      const auto continuity =
          module.view().boundaryTagContinuityPoint(boundary);
      require(test, continuity.has_value(),
              "builtin boundary lost its tag continuity kind");
      configuredWriters +=
          continuity->kind == loom::fabric::
                                  FabricBoundaryTagContinuityKind::
                                      ConfigurableWriter;
      tagRemovers += continuity->kind ==
                     loom::fabric::FabricBoundaryTagContinuityKind::Remover;
      const auto boundaryOwner =
          loom::fabric::FabricTransportEndpointOwnerRef::of(boundary);
      std::optional<loom::fabric::FabricFifoOccurrenceRef> outputFifo;
      for (const auto &connection : module.view().pointConnections()) {
        if (connection.source.owner != boundaryOwner)
          continue;
        const auto *fifo =
            std::get_if<loom::fabric::FabricFifoOccurrenceRef>(
                &connection.destination.owner.payload);
        require(test, fifo != nullptr && !outputFifo.has_value(),
                "builtin gateway boundary does not feed one exact FIFO");
        outputFifo = *fifo;
      }
      require(test, outputFifo.has_value(),
              "builtin gateway boundary has no output FIFO");
      for (const auto &traversal : module.view().admittedTraversals()) {
        const auto *fifo =
            std::get_if<loom::fabric::FabricFifoTraversalPayload>(
                &traversal.payload);
        require(test,
                !fifo || fifo->owner != *outputFifo ||
                    fifo->mode !=
                        loom::fabric::FabricFifoTraversalMode::Bypass,
                "builtin gateway FIFO admits a combinational bypass");
      }
    }
    require(test,
            configuredWriters == descriptor.scale.gatewayCount &&
                tagRemovers == descriptor.scale.gatewayCount,
            "builtin cross-schedule gateway inventory changed unexpectedly");
    std::size_t interiorTransitSwitches = 0;
    for (const auto occurrence : module.view().switchOccurrences()) {
      const auto owner =
          loom::fabric::FabricTransportEndpointOwnerRef::of(occurrence);
      std::size_t inputs = 0;
      std::size_t outputs = 0;
      for (std::uint64_t ordinal = 0;
           ordinal != module.view().transportEndpointCount(owner); ++ordinal) {
        const loom::fabric::FabricTransportEndpointRef endpoint{
            owner, loom::fabric::FabricOrdinal(ordinal)};
        const auto direction =
            module.view().transportEndpointDirection(endpoint);
        require(test, direction.has_value(),
                "builtin switch endpoint lost its direction");
        if (*direction == loom::fabric::FabricPortDirection::Input)
          ++inputs;
        else
          ++outputs;
      }
      require(test, inputs <= 8 && outputs <= 8,
              "builtin mesh emitted a switch dimension larger than eight");
      interiorTransitSwitches += inputs == 8 && outputs == 8;
    }
    require(test,
            interiorTransitSwitches >=
                2 * (expected.meshDimension - 2) *
                    (expected.meshDimension - 2),
            "builtin lost an interior 8x8 transit switch");
    for (const auto memory : module.view().memoryOccurrences()) {
      const auto memoryOwner =
          loom::fabric::FabricTransportEndpointOwnerRef::of(memory);
      llvm::SmallDenseSet<std::uint64_t, 4> ingressSwitches;
      for (const auto &connection : module.view().pointConnections()) {
        if (connection.destination.owner != memoryOwner)
          continue;
        const auto *source =
            std::get_if<loom::fabric::FabricSwitchOccurrenceRef>(
                &connection.source.owner.payload);
        if (source)
          ingressSwitches.insert(source->id());
      }
      require(test, ingressSwitches.size() >= 2,
              "builtin memory transport inputs share one local switch");
    }
    std::size_t wideScalarPorts = 0;
    for (std::uint64_t id = 0;; ++id) {
      const auto kind = module.view().entityKind(id);
      if (!kind)
        break;
      if (*kind != loom::fabric::FabricEntityKind::FabricMemoryOccurrence)
        continue;
      const loom::fabric::FabricMemoryOccurrenceRef memory(id);
      for (const loom::fabric::FabricMemoryOperationPortRef port :
           module.view().memoryOperationPorts(memory)) {
        const auto *record = module.view().memoryOperationPort(port);
        require(test, record != nullptr,
                "builtin memory lost its operation-port record");
        bool sawScalar64 = false;
        bool sawIndexed = false;
        for (const auto &alternative : record->capabilityAlternatives()) {
          require(test, alternative.accessDomain.has_value(),
                  "builtin memory lost its typed access domain");
          for (const auto &access : alternative.accessDomain->accessClasses()) {
            sawScalar64 |=
                access.accessForm() ==
                    ::dataflow::semantics::MemoryAccessForm::Element &&
                access.elementWidths().contains(64);
            sawIndexed |= access.accessForm() ==
                          ::dataflow::semantics::MemoryAccessForm::Indexed;
          }
        }
        require(test, sawScalar64,
                "builtin memory does not cover the common 64-bit scalar floor");
        require(test, sawIndexed,
                "builtin memory does not expose its indexed access domain");
        ++wideScalarPorts;
      }
    }
    require(test,
            wideScalarPorts ==
                2 * (expected.spatialMemories + expected.temporalMemories),
            "builtin memory operation-port inventory changed unexpectedly");
    require(test,
            module.view().moduleBoundaryEndpointCount(
                moduleTemplate, loom::fabric::FabricPortDirection::Input) ==
                descriptor.scale.gatewayCount + 1,
            "builtin SpatialCore did not expose one shared manager capability");
    const loom::fabric::FabricModuleBoundaryEndpointRef memoryBoundary{
        moduleTemplate, loom::fabric::FabricPortDirection::Input,
        descriptor.scale.gatewayCount};
    require(test,
            module.view().moduleBoundaryEndpointPlane(memoryBoundary) ==
                loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory,
            "builtin manager capability is not on the memory plane");
  }

  const auto preset = loom::adg::BuiltinTargetPreset::Small;
  auto canonical = take(test, loom::adg::buildBuiltinTarget(store, preset));

  DesignBuilder moduleDesign(store);
  auto moduleExpansion =
      take(test, loom::adg::expandBuiltinSpatialCore(moduleDesign, preset));
  if (llvm::Error error =
          moduleExpansion.spatialCore.close(moduleExpansion.outputs))
    fail(test, llvm::toString(std::move(error)));
  auto modules = take(test, std::move(moduleDesign).finalize());
  require(test, modules.roots().size() == 1,
          "public builtin expansion did not publish one SpatialCore");

  DesignBuilder systemDesign(store);
  auto system = take(test, loom::adg::expandBuiltinSystem(
                               systemDesign, preset, modules.roots().front()));
  if (llvm::Error error = system.close())
    fail(test, llvm::toString(std::move(error)));
  auto direct = take(test, std::move(systemDesign).finalize());
  require(test,
          direct.roots().size() == 1 && canonical.roots().size() == 1 &&
              direct.roots().front().reference() ==
                  canonical.roots().front().reference(),
          "public builtin expansion changed the canonical preset identity");

  DesignBuilder customModuleDesign(store);
  auto customExpansion = take(
      test, loom::adg::expandBuiltinSpatialCore(customModuleDesign, preset));
  std::vector<SpatialValue> customOutputs = customExpansion.outputs;
  customOutputs.front() =
      take(test, customExpansion.spatialCore.addFifo(
                     customOutputs.front(),
                     FifoSpec{take(test, PortType::bits(128)), 3, false}))
          .value();
  if (llvm::Error error = customExpansion.spatialCore.close(customOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto customModules = take(test, std::move(customModuleDesign).finalize());
  require(test,
          customModules.roots().front().reference() !=
              modules.roots().front().reference(),
          "typed builtin extension did not change the custom Fabric identity");
}

void publicFuLibraryBuildsTypedGraphs() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto spatial =
      take(test, design.createSpatialCore("fu-library",
                                          {bits128, bits128, bits128, bits128},
                                          {bits128, bits128, bits128}));
  auto pe = take(
      test, spatial.addPe(
                {take(test, spatial.input(0)), take(test, spatial.input(1)),
                 take(test, spatial.input(2)), take(test, spatial.input(3))},
                PeSpec::spatial({bits128, bits128, bits128, bits128},
                                {bits128, bits128, bits128})));
  std::vector<loom::adg::PeValue> inputs;
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal)
    inputs.push_back(take(test, pe.input(ordinal)));
  if (llvm::Error error = loom::adg::addCoreAluFu(
          pe, llvm::ArrayRef<loom::adg::PeValue>(inputs).take_front(3),
          ::fabric::ResolvedIndexWidthSet::get(
              {::fabric::ResolvedIndexWidth::I64})))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addMacFu(pe, inputs))
    fail(test, llvm::toString(std::move(error)));
  expectError(test,
              loom::adg::addLoopControlFu(pe, inputs,
                                          ::dataflow::StreamStepKind::Add,
                                          ::dataflow::StreamStepKind::Add),
              "distinct step kinds");
  if (llvm::Error error = loom::adg::addLoopControlFu(
          pe, inputs, ::dataflow::StreamStepKind::Add,
          ::dataflow::StreamStepKind::Sub))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addVectorComputeFu(pe, inputs, {128, 128}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::adg::addSpecialMathFu(
          pe, llvm::ArrayRef<loom::adg::PeValue>(inputs).take_front(2)))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatial.close({take(test, pe.output(0)), take(test, pe.output(1)),
                         take(test, pe.output(2))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  require(test,
          entityCount(finalized.roots().front().view(),
                      loom::fabric::FabricEntityKind::FabricFuOccurrence) == 5,
          "public FU helpers did not create five ordinary FU occurrences");
  bool sawMacDomain = false;
  bool sawLoopControlDomain = false;
  bool sawExactLoopControlContracts = false;
  bool sawStreamSemanticConfiguration = false;
  bool sawVectorSelectSemanticConfiguration = false;
  std::uint32_t floatToIntegerResources = 0;
  bool sawCompleteFloatToIntegerSchemas = false;
  for (std::uint64_t id = 0;; ++id) {
    auto kind = finalized.roots().front().view().entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    const loom::fabric::FabricFuTemplateRef fu(id);
    auto templates = finalized.roots().front().view().fuCapabilityTemplates(fu);
    if (templates.size() == 8) {
      bool hasRecurrence = false;
      for (const auto &record : templates) {
        unsigned activeOperations = 0;
        for (const auto &node : record.activeNodes)
          activeOperations += node.node == loom::fabric::FabricFuNodeKind::Op;
        hasRecurrence |= activeOperations == 3;
      }
      sawMacDomain |= hasRecurrence;
    }
    if (templates.size() == 7) {
      unsigned fusedTemplates = 0;
      for (const auto &record : templates) {
        unsigned activeOperations = 0;
        for (const auto &node : record.activeNodes)
          activeOperations += node.node == loom::fabric::FabricFuNodeKind::Op;
        fusedTemplates += activeOperations == 2;
      }
      sawLoopControlDomain |= fusedTemplates == 2;
    }
    unsigned exactLoopContracts = 0;
    for (const auto &capability :
         finalized.roots().front().view().resolvedFabricOpCapabilities(fu)) {
      if (capability.implementationFamily ==
          ::fabric::ImplementationFamilyId::ScalarFloatToInteger) {
        ++floatToIntegerResources;
        sawCompleteFloatToIntegerSchemas |=
            capability.enabledOperationSchemas ==
            std::vector<::dataflow::OperationSchemaId>{
                ::dataflow::OperationSchemaId::ArithFPToSI,
                ::dataflow::OperationSchemaId::ArithFPToUI,
                ::dataflow::OperationSchemaId::LLVMFPToSISat,
                ::dataflow::OperationSchemaId::LLVMFPToUISat};
      }
      std::uint32_t expectedPatterns = 0;
      switch (capability.implementationFamily) {
      case ::fabric::ImplementationFamilyId::LoopStream:
        sawStreamSemanticConfiguration |=
            capability.configurationFieldSchema.size() == 1;
        [[fallthrough]];
      case ::fabric::ImplementationFamilyId::LoopGate:
        expectedPatterns = 4;
        break;
      case ::fabric::ImplementationFamilyId::LoopCarry:
      case ::fabric::ImplementationFamilyId::LoopInvariant:
        expectedPatterns = 3;
        break;
      default:
        if (capability.implementationFamily ==
            ::fabric::ImplementationFamilyId::FixedVectorValueSelect)
          sawVectorSelectSemanticConfiguration |=
              capability.configurationFieldSchema.size() == 1;
        continue;
      }
      exactLoopContracts +=
          capability.resourceStateAndTimingContract.usePatternCount() ==
          expectedPatterns;
    }
    sawExactLoopControlContracts |= exactLoopContracts == 5;
  }
  require(test, sawMacDomain,
          "MacFu did not expose its complete carry-recurrence domain");
  require(test, sawLoopControlDomain,
          "LoopControlFu did not expose its seven coherent templates");
  require(test, sawExactLoopControlContracts,
          "loop-control resources lost their schema-case use patterns");
  require(test, sawStreamSemanticConfiguration,
          "stream capability lost its typed semantic configuration field");
  require(test, sawVectorSelectSemanticConfiguration,
          "vector select lost its lane-width configuration field");
  require(test,
          floatToIntegerResources == 1 && sawCompleteFloatToIntegerSchemas,
          "CoreAluFu did not model saturating conversion as one "
          "float-to-integer resource add-on");
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (llvm::Error error =
          loom::fabric::writeFabricMlir(finalized.roots().front(), stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();
  require(test,
          llvm::StringRef(text).contains("ScalarIntegerAddSub") &&
              llvm::StringRef(text).contains("LoopCarry") &&
              llvm::StringRef(text).contains("LoopStream") &&
              llvm::StringRef(text).contains("LoopInvariant") &&
              llvm::StringRef(text).contains("LoopGate") &&
              llvm::StringRef(text).contains("FixedVectorFloatFma") &&
              llvm::StringRef(text).contains("ScalarMathSqrt") &&
              llvm::StringRef(text).contains("ScalarMathPow"),
          "public FU helpers lost generated implementation-family bindings");
}

void vectorStructuralFuUsesTypedRecipeWidths() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType outer = take(test, PortType::bits(257));

  std::vector<PortType> boundary(5, outer);
  auto spatial = take(
      test, design.createSpatialCore("vector-structure", boundary, {outer}));
  std::vector<SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundary.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs, PeSpec::spatial(boundary, {outer})));
  std::vector<PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundary.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));

  const loom::adg::VectorStructuralFuParameters parameters{
      257, 192, 64,
      ::fabric::FixedVectorSliceAlignMergeParams{
          ::fabric::IntegerWidthSet::get(
              {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
               ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64}),
          ::fabric::FloatFormatSet::get(
              {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
               ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64}),
          192, 96, 3,
          ::fabric::ResolvedIndexWidthSet::get(
              {::fabric::ResolvedIndexWidth::I32,
               ::fabric::ResolvedIndexWidth::I64})},
      ::fabric::FixedVectorShuffleParams{
          ::fabric::IntegerWidthSet::get(
              {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
               ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64}),
          ::fabric::FloatFormatSet::get(
              {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
               ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64}),
          192, 192, 96, 12, 6}};
  if (llvm::Error error =
          loom::adg::addVectorStructuralFu(pe, peInputs, parameters))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  bool sawSlice = false;
  bool sawShuffle = false;
  for (std::uint64_t id = 0;; ++id) {
    const auto kind = finalized.roots().front().view().entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    const loom::fabric::FabricFuTemplateRef fu(id);
    for (const auto &capability :
         finalized.roots().front().view().resolvedFabricOpCapabilities(fu)) {
      std::vector<std::uint32_t> inputs;
      std::vector<std::uint32_t> outputs;
      for (const auto &port : capability.physicalPorts)
        (port.reference.direction == loom::fabric::FabricPortDirection::Input
             ? inputs
             : outputs)
            .push_back(port.payloadWidthBits);
      if (capability.implementationFamily ==
          ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge) {
        sawSlice = inputs == std::vector<std::uint32_t>{192, 192, 64, 64, 64} &&
                   outputs == std::vector<std::uint32_t>{192} &&
                   capability.configurationFieldSchema.size() == 1;
      }
      if (capability.implementationFamily ==
          ::fabric::ImplementationFamilyId::FixedVectorShuffle) {
        sawShuffle = inputs == std::vector<std::uint32_t>{192, 192} &&
                     outputs == std::vector<std::uint32_t>{192} &&
                     capability.configurationFieldSchema.size() == 1;
      }
    }
  }
  require(test, sawSlice && sawShuffle,
          "VectorStructuralFu did not preserve its exact typed recipe widths");
}

void resolvedCapabilityPreservesTypedVectorGeometry() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));

  auto spatial = take(test, design.createSpatialCore(
                                "f32-vector", {bits128, bits128}, {bits128}));
  auto pe =
      take(test, spatial.addPe({take(test, spatial.input(0)),
                                take(test, spatial.input(1))},
                               PeSpec::spatial({bits128, bits128}, {bits128})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits128, bits128}, {bits128}}));
  auto operation = take(
      test,
      fu.addOperation(
          {take(test, fu.input(0)), take(test, fu.input(1))},
          OperationCapabilitySpec{
              ::fabric::ImplementationFamilyId::FixedVectorFloatAddSub,
              ::fabric::FixedVectorFloatParams{
                  ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F32}),
                  ::fabric::FloatBehaviorProfile::strictIEEE(), 128},
              {::dataflow::OperationSchemaId::ArithAddF,
               ::dataflow::OperationSchemaId::ArithSubF},
              {bits128},
              ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto finalized = take(test, std::move(design).finalize());
  const loom::fabric::FabricFuTemplateRef fuRef =
      uniqueFuTemplate(test, finalized.roots().front().view());
  auto templates =
      finalized.roots().front().view().fuCapabilityTemplates(fuRef);
  require(test,
          templates.size() == 1 && templates.front().activeNodes.size() == 1,
          "custom vector FU changed its capability template");
  const auto *capability =
      finalized.roots().front().view().resolvedFabricOpCapability(
          templates.front().activeNodes.front());
  require(test, capability != nullptr,
          "custom vector FU lost its concrete capability");
  const auto &typedCapability = std::get<::fabric::FixedVectorFloatParams>(
      capability->parameterizedCapability);
  require(
      test,
      typedCapability.elementFormats.contains(::fabric::FloatFormat::F32) &&
          !typedCapability.elementFormats.contains(::fabric::FloatFormat::F64),
      "custom vector FU changed its typed floating format domain");

  mlir::MLIRContext actorContext(mlir::MLIRContext::Threading::DISABLED);
  auto vectorActor = [&](mlir::Type elementType, std::int64_t lanes) {
    mlir::Type vector = mlir::VectorType::get({lanes}, elementType);
    require(test,
            mlir::cast<mlir::VectorType>(vector).getElementType() ==
                elementType,
            "vector actor changed its element type");
    return ::dataflow::CanonicalActorSchemaProjection{
        ::dataflow::OperationSchemaId::ArithAddF,
        mlir::FunctionType::get(&actorContext, {vector, vector}, {vector}),
        ::dataflow::FloatingPointPayload{}};
  };
  auto f32Actor = vectorActor(mlir::Float32Type::get(&actorContext), 4);
  require(test,
          mlir::cast<mlir::VectorType>(f32Actor.type.getInput(0))
              .getElementType()
              .isF32(),
          "f32 vector actor changed its semantic element type");
  if (llvm::Error error = capability->admit(f32Actor, 32))
    fail(test, llvm::toString(std::move(error)));
  auto f64Actor = vectorActor(mlir::Float64Type::get(&actorContext), 2);
  expectError(test, capability->admit(f64Actor, 32),
              "element type is not admitted");

  loom::frontend::FabricCapabilityIndex index(finalized.roots().front().view());
  require(test, index.admittingOperationResources(f32Actor, 32).size() == 1,
          "Fabric capability index lost the admitted vector resource");
  require(test, index.admittingOperationResources(f64Actor, 32).empty(),
          "Fabric capability index treated equal payload width as semantics");
}

void builtinCoreCapabilitiesCoverTypedDomains() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto system = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  auto systemView = take(
      test, loom::fabric::requireSystemRoot(system.roots().front().view()));
  const auto attachments = systemView.serviceLegCarrierAttachments();
  const auto &descriptor = loom::adg::getBuiltinTargetDescriptor(
      loom::adg::BuiltinTargetPreset::Small);
  const std::size_t expectedAttachmentCount =
      4 * (descriptor.scale.accCoreCount + 1);
  require(test, attachments.size() == expectedAttachmentCount,
          "builtin System does not publish every pair-member service leg");
  const auto findAttachment =
      [&](const loom::fabric::FabricMemoryEndpointRef &endpoint,
          dataflow::semantics::ServiceKind kind,
          dataflow::StructuralOrdinal leg)
      -> const loom::fabric::ServiceLegCarrierAttachmentRecord * {
    const auto found = llvm::find_if(attachments, [&](const auto &attachment) {
      return attachment.endpoint() == endpoint && attachment.kind() == kind &&
             attachment.legOrdinal() == leg;
    });
    return found == attachments.end() ? nullptr : &*found;
  };
  const auto checkEndpointRows =
      [&](const loom::fabric::FabricMemoryEndpointRef &endpoint,
          loom::fabric::FabricTransportEndpointOwnerKind carrierOwner,
          loom::fabric::FabricPortDirection requestDirection,
          loom::fabric::FabricPortDirection responseDirection) {
        const auto *readRequest = findAttachment(
            endpoint, dataflow::semantics::ServiceKind::MemoryRead, 0);
        const auto *readResponse = findAttachment(
            endpoint, dataflow::semantics::ServiceKind::MemoryRead, 1);
        const auto *writeRequest = findAttachment(
            endpoint, dataflow::semantics::ServiceKind::MemoryWrite, 0);
        const auto *writeResponse = findAttachment(
            endpoint, dataflow::semantics::ServiceKind::MemoryWrite, 1);
        require(test,
                readRequest && readResponse && writeRequest && writeResponse,
                "builtin System lost a pair-member service leg");
        require(
            test,
            llvm::equal(readRequest->carriers(), writeRequest->carriers()) &&
                llvm::equal(readResponse->carriers(),
                            writeResponse->carriers()),
            "builtin System did not reuse one endpoint carrier domain");
        const std::size_t expectedCarrierCount =
            descriptor.scale.gatewayCount *
            (carrierOwner == loom::fabric::FabricTransportEndpointOwnerKind::
                                 SystemTransportResource
                 ? descriptor.scale.accCoreCount
                 : 1);
        require(test,
                readRequest->carriers().size() == expectedCarrierCount &&
                    readResponse->carriers().size() == expectedCarrierCount,
                "builtin endpoint carrier domain lost a gateway");
        for (const auto &carrier : readRequest->carriers())
          require(test,
                  carrier.owner.kind() == carrierOwner &&
                      systemView.artifact().transportEndpointDirection(
                          carrier) == requestDirection,
                  "builtin request carrier has the wrong owner or direction");
        for (const auto &carrier : readResponse->carriers())
          require(test,
                  carrier.owner.kind() == carrierOwner &&
                      systemView.artifact().transportEndpointDirection(
                          carrier) == responseDirection,
                  "builtin response carrier has the wrong owner or direction");
      };

  std::size_t memoryAttachmentCount = 0;
  for (const auto &spatialAttachment : systemView.spatialAttachments()) {
    const auto *occurrenceEndpoint = spatialAttachment.spatialEndpoint.memory();
    if (!occurrenceEndpoint)
      continue;
    require(test, spatialAttachment.serviceEndpoint.has_value(),
            "builtin memory attachment lost its System endpoint");
    const loom::fabric::FabricMemoryEndpointRef serviceEndpoint{
        loom::fabric::FabricMemoryEndpointOwnerRef::of(
            *spatialAttachment.serviceEndpoint),
        0};
    checkEndpointRows(
        serviceEndpoint,
        loom::fabric::FabricTransportEndpointOwnerKind::SystemTransportResource,
        loom::fabric::FabricPortDirection::Input,
        loom::fabric::FabricPortDirection::Output);
    checkEndpointRows(
        *occurrenceEndpoint,
        loom::fabric::FabricTransportEndpointOwnerKind::SpatialCoreOccurrence,
        loom::fabric::FabricPortDirection::Output,
        loom::fabric::FabricPortDirection::Input);
    ++memoryAttachmentCount;
  }
  require(test, memoryAttachmentCount == descriptor.scale.accCoreCount,
          "builtin System lost an AccCore memory attachment");
  require(
      test,
      llvm::none_of(attachments,
                    [](const auto &attachment) {
                      return attachment.kind() ==
                             dataflow::semantics::ServiceKind::MessageTransfer;
                    }),
      "builtin System attached a transport-plane MessageTransfer leg");
  auto module =
      take(test, loom::fabric::importEntireFabricRoot(
                     system.roots().front().directDependencies().front().root,
                     store));

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  const auto actor = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::ArithIndexCast,
      mlir::FunctionType::get(&context, {mlir::IntegerType::get(&context, 32)},
                              {mlir::IndexType::get(&context)}),
      ::dataflow::NoPayload{}};
  loom::frontend::FabricCapabilityIndex index(module.view());
  require(test, !index.admittingOperationResources(actor, 32).empty(),
          "builtin Fabric rejected its 32-bit resolved index cast");
  require(test, !index.admittingOperationResources(actor, 64).empty(),
          "builtin Fabric rejected its 64-bit resolved index cast");

  mlir::Type pointer = mlir::LLVM::LLVMPointerType::get(&context);
  const auto gep = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::LLVMGetElementPtr,
      mlir::FunctionType::get(
          &context, {pointer, mlir::IntegerType::get(&context, 64)}, {pointer}),
      ::dataflow::GetElementPtrPayload{mlir::IntegerType::get(&context, 32),
                                       {mlir::LLVM::GEPOp::kDynamicIndex},
                                       mlir::LLVM::GEPNoWrapFlags::none}};
  require(test, index.admittingOperationResources(gep, 64).empty(),
          "builtin Fabric inferred pointer support without DataLayout");
  const loom::PointerLayout pointerLayout{
      0, 64, 64, loom::PointerLayoutKind::StableIntegral};
  require(test,
          index.admittingOperationResources(gep, 64, &pointerLayout).empty(),
          "builtin Fabric admitted GEP before address normalization");
  const loom::PointerLayout narrowPointerLayout{
      0, 32, 32, loom::PointerLayoutKind::StableIntegral};
  require(
      test,
      index.admittingOperationResources(gep, 32, &narrowPointerLayout).empty(),
      "builtin Fabric admitted narrow GEP before address normalization");

  mlir::Type f32 = mlir::Float32Type::get(&context);
  const auto floatMultiply = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::ArithMulF,
      mlir::FunctionType::get(&context, {f32, f32}, {f32}),
      ::dataflow::FloatingPointPayload{mlir::arith::FastMathFlags::nnan,
                                       std::nullopt}};
  require(test, !index.admittingOperationResources(floatMultiply, 32).empty(),
          "strict builtin Fabric did not refine relaxed scalar f32 multiply");

  const auto saturatingAdd = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::LLVMSAddSat,
      mlir::FunctionType::get(&context,
                              {mlir::IntegerType::get(&context, 32),
                               mlir::IntegerType::get(&context, 32)},
                              {mlir::IntegerType::get(&context, 32)}),
      ::dataflow::NoPayload{}};
  require(test, !index.admittingOperationResources(saturatingAdd, 32).empty(),
          "builtin Fabric has no scalar saturating arithmetic resource");

  const auto countTrailingZeros = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::MathCountTrailingZeros,
      mlir::FunctionType::get(&context, {mlir::IntegerType::get(&context, 32)},
                              {mlir::IntegerType::get(&context, 32)}),
      ::dataflow::NoPayload{}};
  require(test,
          !index.admittingOperationResources(countTrailingZeros, 32).empty(),
          "builtin Fabric has no scalar zero-count resource");

  mlir::Type vectorI16 =
      mlir::VectorType::get({4}, mlir::IntegerType::get(&context, 16));
  const auto vectorSaturatingAdd = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::LLVMUAddSat,
      mlir::FunctionType::get(&context, {vectorI16, vectorI16}, {vectorI16}),
      ::dataflow::NoPayload{}};
  require(test,
          !index.admittingOperationResources(vectorSaturatingAdd, 32).empty(),
          "builtin Fabric has no fixed-vector saturating arithmetic resource");

  const auto vectorCountLeadingZeros =
      ::dataflow::CanonicalActorSchemaProjection{
          ::dataflow::OperationSchemaId::LLVMCountLeadingZeros,
          mlir::FunctionType::get(&context, {vectorI16}, {vectorI16}),
          ::dataflow::ZeroPoisonPayload{true}};
  require(
      test,
      !index.admittingOperationResources(vectorCountLeadingZeros, 32).empty(),
      "builtin Fabric has no fixed-vector zero-count resource");

  mlir::Type container =
      mlir::VectorType::get({4, 2}, mlir::IntegerType::get(&context, 16));
  mlir::Type slice =
      mlir::VectorType::get({2}, mlir::IntegerType::get(&context, 16));
  const auto vectorExtract = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::VectorExtract,
      mlir::FunctionType::get(&context, {container}, {slice}),
      ::dataflow::VectorStaticPositionPayload{{2}}};
  require(test, !index.admittingOperationResources(vectorExtract, 64).empty(),
          "builtin Fabric has no fixed-vector slice resource");

  mlir::Type lhs =
      mlir::VectorType::get({2, 2}, mlir::IntegerType::get(&context, 16));
  mlir::Type rhs =
      mlir::VectorType::get({1, 2}, mlir::IntegerType::get(&context, 16));
  mlir::Type shuffled =
      mlir::VectorType::get({3, 2}, mlir::IntegerType::get(&context, 16));
  const auto vectorShuffle = ::dataflow::CanonicalActorSchemaProjection{
      ::dataflow::OperationSchemaId::VectorShuffle,
      mlir::FunctionType::get(&context, {lhs, rhs}, {shuffled}),
      ::dataflow::VectorShuffleMaskPayload{{0, 2, -1}}};
  require(test, !index.admittingOperationResources(vectorShuffle, 64).empty(),
          "builtin Fabric has no fixed-vector shuffle resource");
}

void builtinMemoryCapabilitiesAdmitScalarAccess() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  auto system = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));

  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  constexpr llvm::StringLiteral source = R"mlir(
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<
    "dlti.endianness" = "little",
    index = 64 : i64
  >
} {
  func.func @load(%memory: memref<?xf32>, %index: index, %ctrl: none)
      -> (f32, none) {
    %value, %done = dataflow.load %memory[%index] %ctrl : memref<?xf32>
    return %value, %done : f32, none
  }

  func.func @pointer_load(%memory: memref<?xi32>, %address: !llvm.ptr,
                          %ctrl: none) -> (i32, none) {
    %value, %done = dataflow.load %memory[%address] %ctrl
        : memref<?xi32>, !llvm.ptr
    return %value, %done : i32, none
  }

  func.func @contiguous_f64(%memory: memref<?xf64>, %address: index,
                            %ctrl: none) -> (vector<2xf64>, none) {
    %value, %done = dataflow.load %memory[%address] %ctrl
        : memref<?xf64>, vector<2xf64>
    return %value, %done : vector<2xf64>, none
  }

  func.func @indexed_f64(%memory: memref<?xf64>,
                         %address: vector<2xindex>, %ctrl: none)
      -> (vector<2xf64>, none) {
    %value, %done = dataflow.load %memory[%address] %ctrl
        : memref<?xf64>, vector<2xindex>, vector<2xf64>
    return %value, %done : vector<2xf64>, none
  }

  func.func @indexed_f32(%memory: memref<?xf32>,
                         %address: vector<4xindex>, %ctrl: none)
      -> (vector<4xf32>, none) {
    %value, %done = dataflow.load %memory[%address] %ctrl
        : memref<?xf32>, vector<4xindex>, vector<4xf32>
    return %value, %done : vector<4xf32>, none
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  require(test, static_cast<bool>(module),
          "cannot parse the scalar memory actor anchor");
  const auto findLoad = [&](mlir::ModuleOp owner,
                            llvm::StringRef symbol) -> mlir::Operation * {
    mlir::Operation *result = nullptr;
    owner.walk([&](mlir::func::FuncOp function) {
      if (function.getSymName() != symbol)
        return;
      function.walk([&](::dataflow::LoadOp actor) { result = actor; });
    });
    require(test, result != nullptr,
            "builtin memory anchor omitted its load actor");
    return result;
  };

  loom::frontend::FabricCapabilityIndex index(system.roots().front().view());
  auto indexed =
      take(test, index.admittingMemoryResources(findLoad(*module, "load")));
  require(test, !indexed.empty(),
          "builtin Fabric has no scalar load memory resource");
  auto pointer = take(
      test, index.admittingMemoryResources(findLoad(*module, "pointer_load")));
  require(test, !pointer.empty(),
          "builtin Fabric has no P64 pointer-addressed load resource");
  require(test,
          !take(test, index.admittingMemoryResources(
                          findLoad(*module, "contiguous_f64")))
               .empty(),
          "builtin Fabric rejected contiguous vector<2xf64>");
  require(test,
          !take(test, index.admittingMemoryResources(
                          findLoad(*module, "indexed_f64")))
               .empty(),
          "builtin Fabric rejected a fitting indexed address token");
  require(test,
          take(test,
               index.admittingMemoryResources(findLoad(*module, "indexed_f32")))
              .empty(),
          "builtin Fabric admitted an indexed address token wider than its "
          "endpoint");

  (*module)->setAttr("llvm.data_layout",
                     mlir::StringAttr::get(&context, "e-p:32:32"));
  auto narrowPointer = take(
      test, index.admittingMemoryResources(findLoad(*module, "pointer_load")));
  require(test, !narrowPointer.empty(),
          "builtin Fabric has no P32 pointer-addressed load resource");

  constexpr llvm::StringLiteral narrowIndexSource = R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>
} {
  func.func @indexed_f32(%memory: memref<?xf32>,
                         %address: vector<4xindex>, %ctrl: none)
      -> (vector<4xf32>, none) {
    %value, %done = dataflow.load %memory[%address] %ctrl
        : memref<?xf32>, vector<4xindex>, vector<4xf32>
    return %value, %done : vector<4xf32>, none
  }
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> narrowIndexModule =
      mlir::parseSourceString<mlir::ModuleOp>(narrowIndexSource, &context);
  require(test, static_cast<bool>(narrowIndexModule),
          "cannot parse the 32-bit indexed memory anchor");
  require(test,
          !take(test, index.admittingMemoryResources(
                          findLoad(*narrowIndexModule, "indexed_f32")))
               .empty(),
          "builtin Fabric rejected a fitting 32-bit indexed address token");
}

void runBuiltinTests() {
  builtinPresetsExpandThroughPublicBuilder();
  publicFuLibraryBuildsTypedGraphs();
  vectorStructuralFuUsesTypedRecipeWidths();
  resolvedCapabilityPreservesTypedVectorGeometry();
  builtinCoreCapabilitiesCoverTypedDomains();
  builtinMemoryCapabilitiesAdmitScalarAccess();
}

} // namespace loom::adg::test
