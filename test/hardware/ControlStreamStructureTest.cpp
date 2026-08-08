#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ConfigurationABI2TestSupport.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/LoopInvariant.h"
#include "Hardware/RTL/Transport.h"
#include "PortableProviderTestSupport.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::fabric::ResolvedFabricOpCapabilityView;
using loom::hardware::rtl::FabricOperationLeafProtocol;
using loom::hardware::rtl::FabricOperationLeafStateFieldKind;
using loom::hardware::rtl::ResolvedFabricPhysicalOperation;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted an incomplete control/stream leaf contract");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted an incomplete control/stream leaf contract");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

const ResolvedFabricPhysicalOperation &
findOperation(llvm::StringRef test,
              llvm::ArrayRef<ResolvedFabricPhysicalOperation> operations,
              ::fabric::ImplementationFamilyId family) {
  const auto found = llvm::find_if(operations, [&](const auto &operation) {
    return operation.capability->implementationFamily == family;
  });
  if (found == operations.end())
    fail(test, "builtin System is missing the requested operation family");
  return *found;
}

bool hasPort(llvm::ArrayRef<circt::hw::PortInfo> ports, llvm::StringRef name) {
  return llvm::any_of(ports,
                      [&](const auto &port) { return port.getName() == name; });
}

bool dependsOnRegisterNamed(mlir::Value value, llvm::StringRef prefix,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  if (auto reg = mlir::dyn_cast<circt::seq::FirRegOp>(definition))
    return reg.getName().starts_with(prefix);
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return dependsOnRegisterNamed(operand, prefix, visited);
  });
}

const ResolvedFabricOpCapabilityView *
findCapability(const loom::fabric::FinalizedFabricRoot &module,
               ::fabric::ImplementationFamilyId family) {
  for (const auto fu : module.view().fuOccurrences()) {
    const auto definition = module.view().fuTemplateOf(fu);
    if (!definition)
      continue;
    const auto capabilities =
        module.view().resolvedFabricOpCapabilities(*definition);
    const auto found = llvm::find_if(capabilities, [&](const auto &candidate) {
      return candidate.implementationFamily == family;
    });
    if (found != capabilities.end())
      return &*found;
  }
  return nullptr;
}

const loom::hardware::rtl::FabricOperationLeafStateFieldLayout &
requireField(llvm::StringRef test,
             const loom::hardware::rtl::FabricOperationLeafStateLayout &layout,
             FabricOperationLeafStateFieldKind kind) {
  const auto *field = layout.find(kind);
  if (!field)
    fail(test, "operation state layout omitted a required field");
  return *field;
}

const loom::fabric::FabricTransportEndpointRef &
boundaryEndpoint(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &module,
                 loom::fabric::FabricPortDirection direction,
                 loom::fabric::FabricOrdinal ordinal) {
  const loom::fabric::FabricTransportEndpointRef *result = nullptr;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    if (attachment.boundary.direction != direction ||
        attachment.boundary.ordinal != ordinal)
      continue;
    require(test, result == nullptr,
            "Module boundary endpoint has duplicate attachments");
    result = &attachment.endpoint;
  }
  if (!result)
    fail(test, "Module boundary endpoint is unattached");
  return *result;
}

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

struct RouteConfiguration final {
  std::string portName;
  std::uint64_t bitCount = 0;
  std::vector<std::uint8_t> payload;
};

RouteConfiguration
makeRouteConfiguration(llvm::StringRef test,
                       const loom::fabric::FinalizedFabricRoot &module,
                       loom::fabric::SpatialCoreOccurrenceRef spatialCore,
                       const loom::hardware::ConfigurationABI &abi) {
  require(test,
          module.view().peOccurrences().size() == 1 &&
              module.view().fuOccurrences().size() == 1,
          "route configuration fixture changed its PE/FU shape");
  const auto pe = module.view().peOccurrences().front();
  const auto fu = module.view().fuOccurrences().front();
  auto schema = take(test, module.view().spatialPeConfigurationSchema(pe));
  const loom::hardware::ProgrammingUnit *owner = nullptr;
  std::vector<loom::hardware::SemanticConfigurationValue> values;
  for (const auto &descriptor : schema.fields()) {
    loom::fabric::FabricPeConfigurationValue value;
    if (descriptor.kind ==
        loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      value = loom::fabric::FabricPeActive{fu};
    } else {
      require(test, descriptor.port.has_value(),
              "PE selector field has no FU port");
      value = loom::fabric::FabricPeRoute{
          boundaryEndpoint(test, module.view(), descriptor.port->direction,
                           descriptor.port->ordinal)};
    }
    const auto physical =
        qualifyConfigurationField(test, spatialCore, descriptor.reference);
    const loom::hardware::ProgrammingUnit *fieldOwner = nullptr;
    for (const auto &unit : abi.programmingUnits())
      for (const auto &field : unit.fields)
        if (field.field == physical) {
          require(test, fieldOwner == nullptr,
                  "configuration field has duplicate programming owners");
          fieldOwner = &unit;
        }
    if (!fieldOwner)
      fail(test, "configuration field has no programming owner");
    if (owner)
      require(test, owner->id == fieldOwner->id,
              "route configuration spans programming units");
    else
      owner = fieldOwner;
    auto encoded = take(test, schema.encode(descriptor.reference, value));
    values.push_back(
        {physical, std::vector<std::uint8_t>(encoded.bytes().begin(),
                                             encoded.bytes().end())});
  }
  if (!owner)
    fail(test, "route configuration has no programming unit");
  return RouteConfiguration{"configuration_" + std::to_string(owner->id),
                            owner->payloadBitCount,
                            take(test, abi.encode(owner->id, values))};
}

std::string bitLiteral(llvm::ArrayRef<std::uint8_t> bytes,
                       std::uint64_t bitCount) {
  std::string result;
  result.reserve(static_cast<std::size_t>(bitCount));
  for (std::uint64_t bit = bitCount; bit > 0; --bit) {
    const std::uint64_t index = bit - 1;
    result.push_back(
        ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) != 0
            ? '1'
            : '0');
  }
  return result;
}

loom::fabric::FinalizedFabricRoot
makeTokenSyncModule(llvm::StringRef test, loom::ArtifactStore &store) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits128 = take(test, PortType::bits(128));
  const std::vector<PortType> types(4, bits128);
  DesignBuilder builder(store);
  auto spatial = take(test, builder.createSpatialCore(
                                "control-stream-token-sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (unsigned ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe =
      take(test, spatial.addPe(spatialInputs, PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (unsigned ordinal = 0; ordinal != types.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{types, types}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (unsigned ordinal = 0; ordinal != types.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  auto operation = take(
      test,
      fu.addOperation(
          fuInputs, OperationCapabilitySpec{
                        ::fabric::ImplementationFamilyId::TokenSync,
                        ::fabric::RoutedTokenParams{128, 4},
                        {::dataflow::OperationSchemaId::DataflowSync},
                        types,
                        ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> operationOutputs;
  for (unsigned ordinal = 0; ordinal != types.size(); ++ordinal)
    operationOutputs.push_back(take(test, operation.output(ordinal)));
  if (llvm::Error error = fu.close(operationOutputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (unsigned ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialOutputs.push_back(take(test, pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "TokenSync fixture did not publish one Module root");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
makeLoopInvariantModule(llvm::StringRef test, loom::ArtifactStore &store,
                        const ::fabric::ResourceContract &resourceContract) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType phase = take(test, PortType::bits(1));
  const PortType value = take(test, PortType::bits(8));
  const std::vector<PortType> outerInputs{value, value};
  const std::vector<PortType> outerOutputs{value};
  const std::vector<PortType> operationInputs{phase, value};
  const std::vector<PortType> operationOutputs{value};
  DesignBuilder builder(store);
  auto spatial =
      take(test, builder.createSpatialCore("control-stream-loop-invariant",
                                           outerInputs, outerOutputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (unsigned ordinal = 0; ordinal != outerInputs.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe =
      take(test, spatial.addPe(spatialInputs,
                               PeSpec::spatial(outerInputs, outerOutputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (unsigned ordinal = 0; ordinal != outerInputs.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu =
      take(test, pe.addFu(peInputs, FuSpec{operationInputs, operationOutputs}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (unsigned ordinal = 0; ordinal != operationInputs.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  auto operation =
      take(test,
           fu.addOperation(
               fuInputs, OperationCapabilitySpec{
                             ::fabric::ImplementationFamilyId::LoopInvariant,
                             ::fabric::TokenPlaneParams{},
                             {::dataflow::OperationSchemaId::DataflowInvariant},
                             operationOutputs,
                             resourceContract}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "LoopInvariant fixture did not publish one Module root");
  return design.roots().front();
}

void tokenSyncUsesOneDerivedHandshakeContract(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  loom::fabric::FinalizedFabricRoot module = makeTokenSyncModule(test, store);
  loom::fabric::FinalizedFabricRoot system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(module, store));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  std::vector<ResolvedFabricPhysicalOperation> operations = take(
      test, loom::hardware::rtl::enumerateFabricPhysicalOperations(systemView));
  const ResolvedFabricPhysicalOperation &sync = findOperation(
      test, operations, ::fabric::ImplementationFamilyId::TokenSync);
  auto abi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test,
               loom::hardware::test::makeCompleteConfigurationABIDraft(system)),
          store));

  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  mlir::OpBuilder builder(&context);
  std::vector<circt::hw::PortInfo> ports = take(
      test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                builder, sync.physicalOccurrence, *sync.capability, abi.abi()));
  for (unsigned ordinal = 0; ordinal != 4; ++ordinal) {
    const std::string suffix = std::to_string(ordinal);
    require(test, hasPort(ports, "valid_input_" + suffix),
            "TokenSync leaf omitted an input valid port");
    require(test, hasPort(ports, "ready_output_" + suffix),
            "TokenSync leaf omitted an output ready port");
    require(test, hasPort(ports, "ready_input_" + suffix),
            "TokenSync leaf omitted an input ready port");
    require(test, hasPort(ports, "valid_output_" + suffix),
            "TokenSync leaf omitted an output valid port");
  }
  require(test, !sync.capability->configurationFieldSchema.empty(),
          "TokenSync fixture did not expose its sealed lane configuration");
  for (const auto &field : sync.capability->configurationFieldSchema)
    require(test, hasPort(ports, "config_" + std::to_string(field.ordinal)),
            "TokenSync leaf omitted an exact operation configuration field");
  require(test,
          !hasPort(ports, "state_current") && !hasPort(ports, "state_next") &&
              !hasPort(ports, "state_write"),
          "stateless TokenSync acquired a state transform");

  mlir::OwningOpRef<mlir::ModuleOp> circtModule =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(circtModule->getBody());
  llvm::erase_if(ports, [](const auto &port) {
    return port.getName() == "valid_output_3";
  });
  auto leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, builder.getUnknownLoc(),
      mlir::FlatSymbolRefAttr::get(&context, "test_generator"),
      builder.getStringAttr("incomplete_token_sync"), ports);
  std::string beforeRejection;
  llvm::raw_string_ostream(beforeRejection) << *circtModule;
  expectError(test,
              loom::hardware::rtl::verifyFabricOperationLeafPorts(
                  leaf, sync.physicalOccurrence, *sync.capability, abi.abi()),
              "port count");
  std::string afterRejection;
  llvm::raw_string_ostream(afterRejection) << *circtModule;
  require(test, beforeRejection == afterRejection,
          "malformed leaf rejection partially mutated its CIRCT module");

  require(test, systemView.artifact().accCoreOccurrences().size() == 1,
          "TokenSync System did not publish one SpatialCore");
  context.loadDialect<circt::comb::CombDialect, circt::seq::SeqDialect,
                      circt::sv::SVDialect>();
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, spatialCore, abi.abi()));
  require(test, skeleton.operationLeaves.size() == 1,
          "TokenSync did not traverse the CommonSkeleton operation path");
  circt::hw::InstanceOp operationInstance;
  skeleton.module->walk([&](circt::hw::InstanceOp candidate) {
    if (candidate.getInstanceName() == "operation")
      operationInstance = candidate;
  });
  require(test, static_cast<bool>(operationInstance),
          "TokenSync skeleton omitted its operation instance");
  std::optional<std::size_t> validInputOrdinal;
  std::size_t inputOrdinal = 0;
  for (const circt::hw::PortInfo &port :
       skeleton.operationLeaves.front().module.getPortList()) {
    if (port.isOutput())
      continue;
    if (port.getName() == "valid_input_0")
      validInputOrdinal = inputOrdinal;
    ++inputOrdinal;
  }
  require(test,
          validInputOrdinal &&
              *validInputOrdinal < operationInstance.getInputs().size(),
          "TokenSync instance omitted its first input-valid operand");
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  require(
      test,
      dependsOnRegisterNamed(operationInstance.getInputs()[*validInputOrdinal],
                             "result_valid_", visited),
      "elastic input-valid did not depend on common result capacity");
  std::string skeletonText;
  llvm::raw_string_ostream(skeletonText) << *skeleton.module;
  require(test,
          llvm::StringRef(skeletonText).contains("result_valid_0_reg") &&
              llvm::StringRef(skeletonText).contains("result_valid_3_reg") &&
              llvm::StringRef(skeletonText).contains("valid_input_0") &&
              llvm::StringRef(skeletonText).contains("ready_output_3") &&
              llvm::StringRef(skeletonText).contains("config_0"),
          "TokenSync skeleton omitted its common elastic tuple boundary");
}

void builtinCapabilitiesOwnProtocolAndState(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  const auto dependencies = design.roots().front().directDependencies();
  require(test, dependencies.size() == 1,
          "Small builtin did not publish one Module dependency");
  auto module = take(test, loom::fabric::importEntireFabricRoot(
                               dependencies.front().root, store));

  const auto capability = [&](::fabric::ImplementationFamilyId family)
      -> const ResolvedFabricOpCapabilityView & {
    const auto *result = findCapability(module, family);
    if (!result)
      fail(test, "Small builtin is missing a control/stream family");
    return *result;
  };
  for (const auto family :
       {::fabric::ImplementationFamilyId::FixedVectorParallelize,
        ::fabric::ImplementationFamilyId::FixedVectorSerialize,
        ::fabric::ImplementationFamilyId::TokenConstant,
        ::fabric::ImplementationFamilyId::TokenSync,
        ::fabric::ImplementationFamilyId::TokenMux,
        ::fabric::ImplementationFamilyId::TokenDemux}) {
    const auto interface =
        take(test, loom::hardware::rtl::deriveFabricOperationLeafInterface(
                       capability(family)));
    require(test,
            interface.protocol == FabricOperationLeafProtocol::ElasticToken,
            "control/stream family did not derive an elastic token boundary");
  }
  const auto streamInterface =
      take(test, loom::hardware::rtl::deriveFabricOperationLeafInterface(
                     capability(::fabric::ImplementationFamilyId::LoopStream)));
  require(test,
          streamInterface.protocol == FabricOperationLeafProtocol::ManagedToken,
          "LoopStream did not preserve its provider-managed timing contract");
  for (const auto family : {::fabric::ImplementationFamilyId::LoopCarry,
                            ::fabric::ImplementationFamilyId::LoopInvariant,
                            ::fabric::ImplementationFamilyId::LoopGate}) {
    const auto interface =
        take(test, loom::hardware::rtl::deriveFabricOperationLeafInterface(
                       capability(family)));
    require(test,
            interface.protocol == FabricOperationLeafProtocol::TransparentToken,
            "transparent loop acquired an elastic result boundary");
  }
  const auto ordinary = take(
      test, loom::hardware::rtl::deriveFabricOperationLeafInterface(capability(
                ::fabric::ImplementationFamilyId::ScalarIntegerMultiply)));
  require(test, ordinary.protocol == FabricOperationLeafProtocol::Combinational,
          "ordinary combinational family acquired a token leaf protocol");

  const auto parallelize =
      take(test,
           loom::hardware::rtl::deriveFabricOperationLeafStateLayout(capability(
               ::fabric::ImplementationFamilyId::FixedVectorParallelize)));
  const auto serialize =
      take(test,
           loom::hardware::rtl::deriveFabricOperationLeafStateLayout(capability(
               ::fabric::ImplementationFamilyId::FixedVectorSerialize)));
  const auto stream =
      take(test, loom::hardware::rtl::deriveFabricOperationLeafStateLayout(
                     capability(::fabric::ImplementationFamilyId::LoopStream)));
  const auto invariant = take(
      test, loom::hardware::rtl::deriveFabricOperationLeafStateLayout(
                capability(::fabric::ImplementationFamilyId::LoopInvariant)));
  const auto sync =
      take(test, loom::hardware::rtl::deriveFabricOperationLeafStateLayout(
                     capability(::fabric::ImplementationFamilyId::TokenSync)));
  require(test, parallelize && serialize && stream && invariant && !sync,
          "stateful and stateless control families were not distinguished");

  for (const auto *layout : {&*parallelize, &*serialize}) {
    const auto &value = requireField(
        test, *layout, FabricOperationLeafStateFieldKind::BufferedValue);
    const auto &mask = requireField(
        test, *layout, FabricOperationLeafStateFieldKind::BufferedMask);
    require(test,
            value.bitOffset == 0 && value.bitCount == 128 &&
                mask.bitOffset == 128 && mask.bitCount == 128 &&
                layout->encodedBitCount() == 256 &&
                layout->resetValue().isZero(),
            "vector adapter state did not follow its exact carrier widths");
  }

  const auto &mode =
      requireField(test, *stream, FabricOperationLeafStateFieldKind::Mode);
  const auto &current =
      requireField(test, *stream, FabricOperationLeafStateFieldKind::Current);
  const auto &limit =
      requireField(test, *stream, FabricOperationLeafStateFieldKind::Limit);
  const auto &step =
      requireField(test, *stream, FabricOperationLeafStateFieldKind::Step);
  require(test,
          mode.bitOffset == 0 && mode.bitCount == 1 && current.bitOffset == 1 &&
              current.bitCount == 64 && limit.bitOffset == 65 &&
              limit.bitCount == 64 && step.bitOffset == 129 &&
              step.bitCount == 64 && stream->encodedBitCount() == 193 &&
              stream->resetValue().isZero(),
          "LoopStream state did not follow its reachable integer domain");
  require(test,
          invariant->encodedBitCount() == 129 &&
              invariant->resetValue().isZero(),
          "transparent invariant state layout changed");
}

void writeControlStreamToolArtifacts(llvm::StringRef test,
                                     const std::filesystem::path &root,
                                     llvm::StringRef systemVerilog,
                                     const RouteConfiguration &configuration) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "control_stream_module.sv") << systemVerilog.str();
  std::ofstream testbench(root / "control_stream_testbench.sv");
  testbench << R"sv(
module control_stream_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic       output_0_valid;
  logic       output_0_ready;
)sv";
  testbench << "  logic [" << configuration.bitCount - 1 << ":0] "
            << configuration.portName << ";\n\n";
  testbench << R"sv(  loom_module dut(.*);

  always #5 clock = ~clock;

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 8'h01;
    input_0_valid = 1;
    input_1_data = 8'h3c;
    input_1_valid = 0;
    output_0_ready = 1;
)sv";
  testbench << "    " << configuration.portName << " = "
            << configuration.bitCount << "'b"
            << bitLiteral(configuration.payload, configuration.bitCount)
            << ";\n";
  testbench << R"sv(    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    #1;
    check(!input_0_ready && !output_0_valid,
          "Incomplete Init consumed phase or published a result");

    input_1_valid = 1;
    #1;
    check(!input_0_ready && input_1_ready && output_0_valid &&
              output_0_data == 8'h3c,
          "Init did not consume and publish its complete semantic tuple");
    @(posedge clock);
    #1;

    input_1_data = 8'ha5;
    output_0_ready = 0;
    repeat (3) begin
      @(negedge clock);
      #1;
      check(!input_0_ready && !input_1_ready && output_0_valid &&
                output_0_data == 8'h3c,
            "Stalled Replay changed valid, payload, or input consumption");
    end

    output_0_ready = 1;
    #1;
    check(input_0_ready && !input_1_ready && output_0_valid &&
              output_0_data == 8'h3c,
          "Replay did not progress atomically");
    @(posedge clock);
    #1;
    check(input_0_ready && output_0_valid && output_0_data == 8'h3c,
          "Progress did not admit a bubble-free replacement");

    input_0_data = 8'h00;
    #1;
    check(input_0_ready && !output_0_valid,
          "Close published an inactive result");
    @(posedge clock);
    @(negedge clock);
    input_0_valid = 0;
    input_1_data = 8'h7e;
    #1;
    check(input_1_ready && output_0_valid && output_0_data == 8'h7e,
          "Closed state did not accept a replacement Init");
    @(posedge clock);

    @(negedge clock);
    reset = 1;
    #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid,
          "Reset did not quiesce the token boundary");
    reset = 0;
    input_1_data = 8'h55;
    #1;
    check(input_1_ready && output_0_valid && output_0_data == 8'h55,
          "Reset did not restore the initial operation state");
    $finish;
  end
endmodule
)sv";
  testbench.close();
  require(test, static_cast<bool>(testbench),
          "could not write the control/stream testbench");

  std::ofstream(root / "control_stream.ys") << R"ys(
read_verilog -sv control_stream_module.sv
hierarchy -check -top loom_module
check -assert
proc
select -assert-count 1 loom_module/t:$adff
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
}

void writeAtomicTransportToolArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());

  const auto port = [&](llvm::StringRef name,
                        circt::hw::ModulePort::Direction direction) {
    return circt::hw::PortInfo{
        {builder.getStringAttr(name), builder.getI1Type(), direction}};
  };
  const auto input = circt::hw::ModulePort::Direction::Input;
  const auto output = circt::hw::ModulePort::Direction::Output;
  llvm::SmallVector<circt::hw::PortInfo, 7> inputs{
      port("input_valid_0", input),      port("input_valid_1", input),
      port("capacity_available", input), port("held_valid_0", input),
      port("held_valid_1", input),       port("result_ready_0", input),
      port("result_ready_1", input)};
  llvm::SmallVector<circt::hw::PortInfo, 7> outputs{
      port("input_ready_0", output),     port("input_ready_1", output),
      port("published_valid_0", output), port("published_valid_1", output),
      port("occupied", output),          port("released", output),
      port("available", output)};
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("atomic_transport_contract"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        llvm::SmallVector<mlir::Value, 2> inputValids{
            accessor.getInput("input_valid_0"),
            accessor.getInput("input_valid_1")};
        auto inputReady =
            take(test, loom::hardware::rtl::deriveAtomicInputReadiness(
                           bodyBuilder, location, inputValids,
                           accessor.getInput("capacity_available")));
        llvm::SmallVector<mlir::Value, 2> heldValids{
            accessor.getInput("held_valid_0"),
            accessor.getInput("held_valid_1")};
        llvm::SmallVector<mlir::Value, 2> resultReady{
            accessor.getInput("result_ready_0"),
            accessor.getInput("result_ready_1")};
        auto tuple =
            take(test, loom::hardware::rtl::deriveAtomicResultTupleSignals(
                           bodyBuilder, location, heldValids, resultReady));

        expectError(test,
                    loom::hardware::rtl::deriveAtomicInputReadiness(
                        bodyBuilder, location, {},
                        accessor.getInput("capacity_available")),
                    "empty");
        expectError(test,
                    loom::hardware::rtl::deriveAtomicResultTupleSignals(
                        bodyBuilder, location, heldValids,
                        llvm::ArrayRef<mlir::Value>{resultReady}.drop_back()),
                    "arity");

        accessor.setOutput("input_ready_0", inputReady[0]);
        accessor.setOutput("input_ready_1", inputReady[1]);
        accessor.setOutput("published_valid_0", tuple.publishedValids[0]);
        accessor.setOutput("published_valid_1", tuple.publishedValids[1]);
        accessor.setOutput("occupied", tuple.occupied);
        accessor.setOutput("released", tuple.released);
        accessor.setOutput("available", tuple.available);
      });

  const std::string systemVerilog = take(
      test,
      loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(*module));
  std::ofstream moduleFile(root / "atomic_transport_module.sv");
  moduleFile << systemVerilog;
  moduleFile.close();
  require(test, static_cast<bool>(moduleFile),
          "could not write the atomic Transport module");

  std::ofstream testbench(root / "atomic_transport_testbench.sv");
  testbench << R"sv(
module atomic_transport_testbench;
  logic input_valid_0;
  logic input_valid_1;
  logic capacity_available;
  logic held_valid_0;
  logic held_valid_1;
  logic result_ready_0;
  logic result_ready_1;
  logic input_ready_0;
  logic input_ready_1;
  logic published_valid_0;
  logic published_valid_1;
  logic occupied;
  logic released;
  logic available;

  atomic_transport_contract dut(.*);

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  initial begin
    for (int sample = 0; sample < 128; sample++) begin
      {result_ready_1, result_ready_0, held_valid_1, held_valid_0,
       capacity_available, input_valid_1, input_valid_0} = sample[6:0];
      #1;
      check(input_ready_0 == (capacity_available && input_valid_1),
            "Atomic join partially admitted input 0");
      check(input_ready_1 == (capacity_available && input_valid_0),
            "Atomic join partially admitted input 1");
      check(published_valid_0 ==
                (held_valid_0 && (!held_valid_1 || result_ready_1)),
            "Atomic fork published result 0 without tuple readiness");
      check(published_valid_1 ==
                (held_valid_1 && (!held_valid_0 || result_ready_0)),
            "Atomic fork published result 1 without tuple readiness");
      check(occupied == (held_valid_0 || held_valid_1),
            "Tuple occupancy did not follow held results");
      check(released ==
                ((held_valid_0 || held_valid_1) &&
                 (!held_valid_0 || result_ready_0) &&
                 (!held_valid_1 || result_ready_1)),
            "Tuple release was not one complete handoff");
      check(available ==
                (!(held_valid_0 || held_valid_1) || released),
            "Tuple capacity did not permit exact replacement");
    end
    $finish;
  end
endmodule
)sv";
  testbench.close();
  require(test, static_cast<bool>(testbench),
          "could not write the atomic Transport testbench");

  std::ofstream script(root / "atomic_transport.ys");
  script << R"ys(
read_verilog -sv atomic_transport_module.sv
hierarchy -check -top atomic_transport_contract
check -assert
proc
select -assert-none atomic_transport_contract/t:$adff
select -assert-none atomic_transport_contract/t:$dff
select -assert-none atomic_transport_contract/t:$dlatch
synth -top atomic_transport_contract
check -assert
)ys";
  script.close();
  require(test, static_cast<bool>(script),
          "could not write the atomic Transport Yosys script");
}

void commonSkeletonOwnsTransparentState(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  loom::ArtifactStore store(root.string());
  auto module = makeLoopInvariantModule(
      test, store, ::fabric::loopInvariantOperationResourceContract());
  auto system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(module, store));
  auto abi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test,
               loom::hardware::test::makeCompleteConfigurationABIDraft(system)),
          store));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test, systemView.artifact().accCoreOccurrences().size() == 1,
          "LoopInvariant System did not publish one SpatialCore");
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, spatialCore, abi.abi()));
  require(test, skeleton.operationLeaves.size() == 1,
          "LoopInvariant skeleton did not expose one operation leaf");
  const RouteConfiguration configuration =
      makeRouteConfiguration(test, module, spatialCore, abi.abi());
  loom::hardware::rtl::FabricOperationProviderRegistry providers;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableLoopInvariantProvider(providers))
    fail(test, llvm::toString(std::move(error)));
  loom::hardware::ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(skeleton), abi, providers, externalContracts));
  const llvm::StringRef rtl(conformance.systemVerilog);
  require(test,
          rtl.contains("operation_state_reg") && rtl.contains("clock") &&
              rtl.contains("reset") && rtl.contains("input_0_ready") &&
              rtl.contains("output_0_valid"),
          "transparent operation RTL omitted its structural state boundary");
  writeControlStreamToolArtifacts(test, root, rtl, configuration);

  auto wrongModule = makeLoopInvariantModule(
      test, store, ::fabric::oneCycleElasticOperationResourceContract());
  auto wrongSystem =
      take(test, loom::hardware::test::makeSingleSpatialCoreSystem(wrongModule,
                                                                   store));
  auto wrongAbi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         wrongSystem)),
          store));
  auto wrongView =
      take(test, loom::fabric::requireSystemRoot(wrongSystem.view()));
  const loom::fabric::SpatialCoreOccurrenceRef wrongSpatialCore{
      wrongView.artifact().accCoreOccurrences().front()};
  expectError(test,
              loom::hardware::rtl::buildModuleRootCirctSkeleton(
                  context, wrongSpatialCore, wrongAbi.abi()),
              "resource contract");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  tokenSyncUsesOneDerivedHandshakeContract(argv[1]);
  builtinCapabilitiesOwnProtocolAndState(std::filesystem::path(argv[1]) /
                                         "builtin");
  writeAtomicTransportToolArtifacts(std::filesystem::path(argv[1]) /
                                    "transport");
  commonSkeletonOwnsTransparentState(std::filesystem::path(argv[1]) /
                                     "skeleton");
  return 0;
}
