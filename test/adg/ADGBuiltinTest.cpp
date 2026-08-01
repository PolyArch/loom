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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <vector>

namespace loom::adg::test {

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
  };
  const std::array<Expectation, 3> expectations{{
      {loom::adg::BuiltinTargetPreset::Small, 4, 12, 4, 1, 1},
      {loom::adg::BuiltinTargetPreset::Default, 8, 27, 9, 2, 2},
      {loom::adg::BuiltinTargetPreset::Large, 16, 48, 16, 4, 4},
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
                loom::fabric::FabricEntityKind::SystemServiceEndpoint) == 1,
        "builtin lost its SpatialCore, AccCore, or System memory inventory");

    auto systemView = take(test, loom::fabric::requireSystemRoot(root.view()));
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
    std::size_t memoryAttachments = 0;
    for (const auto &attachment : systemView.spatialAttachments())
      memoryAttachments +=
          attachment.spatialEndpoint.plane() ==
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory;
    require(test, memoryAttachments == expected.accCores,
            "builtin did not attach one manager capability per AccCore");

    auto module =
        take(test, loom::fabric::importEntireFabricRoot(
                       root.directDependencies().front().root, store));
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
        const auto *alternative =
            module.view().memoryCapabilityAlternative({port, 0});
        require(test, alternative && alternative->accessDomain,
                "builtin memory lost its typed access domain");
        const auto element = llvm::find_if(
            alternative->accessDomain->accessClasses(), [](const auto &access) {
              return access.accessForm() ==
                     ::dataflow::semantics::MemoryAccessForm::Element;
            });
        require(test,
                element != alternative->accessDomain->accessClasses().end() &&
                    element->elementWidths().contains(64),
                "builtin memory does not cover the common 64-bit scalar floor");
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
                     FifoSpec{take(test, PortType::bits(128)), 3, false}));
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
  if (llvm::Error error = loom::adg::addVectorComputeFu(pe, inputs))
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
              llvm::StringRef(text).contains("ScalarMathSqrt"),
          "public FU helpers lost generated implementation-family bindings");
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
          !index.admittingOperationResources(gep, 64, &pointerLayout).empty(),
          "builtin Fabric lost its exact stable-integral GEP add-on");
  const loom::PointerLayout narrowPointerLayout{
      0, 32, 32, loom::PointerLayoutKind::StableIntegral};
  require(
      test,
      !index.admittingOperationResources(gep, 32, &narrowPointerLayout).empty(),
      "builtin Fabric lost its explicit P32 GEP capability");

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
}
)mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  require(test, static_cast<bool>(module),
          "cannot parse the scalar memory actor anchor");
  llvm::SmallVector<mlir::Operation *, 2> loads;
  module->walk([&](::dataflow::LoadOp actor) { loads.push_back(actor); });
  require(test, loads.size() == 2,
          "scalar memory actor anchor does not have both loads");

  loom::frontend::FabricCapabilityIndex index(system.roots().front().view());
  auto indexed = take(test, index.admittingMemoryResources(loads[0]));
  require(test, !indexed.empty(),
          "builtin Fabric has no scalar load memory resource");
  auto pointer = take(test, index.admittingMemoryResources(loads[1]));
  require(test, !pointer.empty(),
          "builtin Fabric has no P64 pointer-addressed load resource");

  (*module)->setAttr("llvm.data_layout",
                     mlir::StringAttr::get(&context, "e-p:32:32"));
  auto narrowPointer = take(test, index.admittingMemoryResources(loads[1]));
  require(test, !narrowPointer.empty(),
          "builtin Fabric has no P32 pointer-addressed load resource");
}

void runBuiltinTests() {
  builtinPresetsExpandThroughPublicBuilder();
  publicFuLibraryBuildsTypedGraphs();
  resolvedCapabilityPreservesTypedVectorGeometry();
  builtinCoreCapabilitiesCoverTypedDomains();
  builtinMemoryCapabilitiesAdmitScalarAccess();
}

} // namespace loom::adg::test
