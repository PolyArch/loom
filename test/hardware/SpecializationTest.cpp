#include "Hardware/RTL/Specialization.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Fabric/Artifact/FabricArtifactLocalReference.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactRootReference;
using loom::ArtifactStore;
using loom::BlobStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

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
    fail(test, "accepted invalid specialization input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid specialization input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

struct FabricFixture final {
  FinalizedFabricRoot system;
  FinalizedFabricRoot module;
};

llvm::Error requireImplementationPlatform(
    const ExternalImplementationBinding &, HardwareRepresentation,
    const loom::platform::ImplementationPlatform *platform) {
  if (!platform)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "vendor binding requires an implementation "
                                   "platform");
  return llvm::Error::success();
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store) {
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "builtin target did not produce one System root");
  FinalizedFabricRoot system = design.roots().front();
  const auto dependencies = system.directDependencies();
  require(test, dependencies.size() == 1,
          "builtin System root did not name one module dependency");
  FinalizedFabricRoot module =
      take(test, loom::fabric::importEntireFabricRoot(dependencies.front().root,
                                                      store));
  return FabricFixture{std::move(system), std::move(module)};
}

ExternalImplementationContractCatalog
vendorContractCatalog(llvm::StringRef test,
                      ExternalDependencyKind dependencyKind =
                          ExternalDependencyKind::ToolBundledResource) {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = catalog.add(
          ExternalImplementationContract{"synopsys.designware@1",
                                         {{"implementation", {dependencyKind}}},
                                         {HardwareRepresentation::Rtl},
                                         true,
                                         false,
                                         requireImplementationPlatform}))
    fail(test, llvm::toString(std::move(error)));
  return catalog;
}

ExternalImplementationContractCatalog
multiSlotVendorContractCatalog(llvm::StringRef test) {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error = catalog.add(ExternalImplementationContract{
          "example.multi_input@1",
          {{"implementation", {ExternalDependencyKind::ToolBundledResource}},
           {"simulation", {ExternalDependencyKind::ToolBundledResource}}},
          {HardwareRepresentation::Rtl},
          true,
          false}))
    fail(test, llvm::toString(std::move(error)));
  return catalog;
}

std::vector<FabricFuOccurrenceNodeRef>
findOperationOccurrences(llvm::StringRef test,
                         const loom::fabric::FabricArtifactView &view,
                         std::size_t count) {
  std::vector<FabricFuOccurrenceNodeRef> result;
  for (const auto occurrence : view.fuOccurrences()) {
    const auto definition = view.fuTemplateOf(occurrence);
    if (!definition)
      continue;
    const auto capabilities = view.resolvedFabricOpCapabilities(*definition);
    for (const auto &capability : capabilities) {
      result.push_back(
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         view, capability.occurrence, occurrence)));
      if (result.size() == count)
        return result;
    }
  }
  fail(test, "builtin Fabric has too few concrete operation occurrences");
}

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
};

SkeletonFixture makeSkeleton(mlir::MLIRContext &context,
                             std::size_t leafCount = 1) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
  leaves.reserve(leafCount);
  for (std::size_t index = 0; index < leafCount; ++index)
    leaves.push_back(circt::hw::HWModuleGeneratedOp::create(
        builder, location,
        mlir::FlatSymbolRefAttr::get(&context,
                                     fabricOperationGeneratorSchemaSymbol),
        builder.getStringAttr("loom_fabric_operation_" + std::to_string(index)),
        llvm::ArrayRef<circt::hw::PortInfo>{}));
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("specialized_top"),
      circt::hw::ModulePortInfo({}, {}),
      [](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &) {});
  return SkeletonFixture{std::move(module), std::move(leaves)};
}

SkeletonFixture makeConnectedSkeleton(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  const mlir::Type valueType = builder.getIntegerType(32);
  const auto port = [&](llvm::StringRef name,
                        circt::hw::ModulePort::Direction direction) {
    return circt::hw::PortInfo{
        {builder.getStringAttr(name), valueType, direction}};
  };
  const std::vector<circt::hw::PortInfo> leafPorts = {
      port("operand", circt::hw::ModulePort::Direction::Input),
      port("result", circt::hw::ModulePort::Direction::Output)};
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_0"), leafPorts);
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("connected_top"),
      circt::hw::ModulePortInfo({leafPorts.front()}, {leafPorts.back()}),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
            bodyBuilder, location, leaf.getOperation(), "operation",
            {accessor.getInput("operand")});
        accessor.setOutput("result", instance.getResult(0));
      });
  return SkeletonFixture{std::move(module), {leaf}};
}

struct ProviderObservation final {
  FabricFuOccurrenceNodeRef occurrence;
  ::fabric::ImplementationFamilyId family;
  ArtifactRootReference configurationFabric;
  BackendRecipeKey recipe;
  bool isolatedFragment = false;
  bool hasImplementationPlatform = false;
  std::string externalContractRef;
  std::vector<ExternalInputBinding> externalInputs;
};

std::optional<ProviderObservation> observation;
unsigned providerInvocationCount = 0;

void materializeConcreteModule(FabricOperationProviderRequest request) {
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  circt::hw::HWModuleOp::create(
      builder, request.leaf.getLoc(), request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      request.leaf.getParametersAttr());
  request.leaf.erase();
}

llvm::Expected<FabricOperationProviderOutput>
materializeProvider(FabricOperationProviderRequest request) {
  ++providerInvocationCount;
  observation = ProviderObservation{
      request.occurrence,
      request.capability.implementationFamily,
      request.configurationAbi.fabric(),
      request.recipe,
      request.fragment->getParentOp() == nullptr,
      request.implementationPlatform != nullptr,
      request.externalImplementationContractRef.str(),
      std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                        request.externalInputs.end())};
  materializeConcreteModule(request);
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
passthroughProvider(FabricOperationProviderRequest request) {
  ++providerInvocationCount;
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  circt::hw::HWModuleOp::create(
      builder, request.leaf.getLoc(), request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &accessor) {
        accessor.setOutput("result", accessor.getInput("operand"));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
incompleteProvider(FabricOperationProviderRequest request) {
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
unboundExternalProvider(FabricOperationProviderRequest request) {
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  circt::hw::HWModuleExternOp::create(
      builder, request.leaf.getLoc(), request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      request.leaf.getSymName(), request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
failingProvider(FabricOperationProviderRequest request) {
  ++providerInvocationCount;
  materializeConcreteModule(request);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "provider preparation failed");
}

llvm::Expected<FabricOperationProviderOutput>
vendorProvider(FabricOperationProviderRequest request) {
  ++providerInvocationCount;
  observation = ProviderObservation{
      request.occurrence,
      request.capability.implementationFamily,
      request.configurationAbi.fabric(),
      request.recipe,
      request.fragment->getParentOp() == nullptr,
      request.implementationPlatform != nullptr,
      request.externalImplementationContractRef.str(),
      std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                        request.externalInputs.end())};

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const std::string moduleName = request.leaf.getSymName().str();
  circt::hw::HWModuleExternOp::create(
      builder, request.leaf.getLoc(), request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()), moduleName,
      request.leaf.getParametersAttr());
  request.leaf.erase();

  FabricOperationProviderOutput output;
  output.payloads.push_back({HardwarePayloadRole::BlackBoxContract,
                             moduleName + ".json",
                             "application/json",
                             {'{', '}'}});
  output.externalImplementationBindings.push_back(
      {moduleName,
       request.externalImplementationContractRef.str(),
       std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                         request.externalInputs.end()),
       {},
       {{RepresentationObjectKind::Module, moduleName}},
       HardwarePayloadRef{HardwarePayloadRole::BlackBoxContract,
                          moduleName + ".json"}});
  output.activityPoints.push_back(
      {moduleName + ".result",
       {RepresentationObjectKind::Port, moduleName + ".result"},
       std::nullopt});
  return output;
}

llvm::Expected<FabricOperationProviderOutput>
collidingProvider(FabricOperationProviderRequest request) {
  ++providerInvocationCount;
  materializeConcreteModule(request);
  mlir::OpBuilder builder(request.fragment.getContext());
  builder.setInsertionPointToEnd(request.fragment.getBody());
  circt::hw::HWModuleOp::create(
      builder, request.fragment.getLoc(),
      builder.getStringAttr("specialized_top"),
      circt::hw::ModulePortInfo({}, {}),
      [](mlir::OpBuilder &, circt::hw::HWModulePortAccessor &) {});
  return FabricOperationProviderOutput{};
}

void registryAndSpecializationAreExact(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  ArtifactStore store(root);
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  const FabricFuOccurrenceNodeRef occurrence =
      findOperationOccurrences(test, fabric.module.view(), 1).front();
  const auto *capability =
      fabric.module.view().resolvedFabricOpCapability(occurrence);
  require(test, capability != nullptr,
          "operation occurrence has no resolved capability");

  FabricOperationProviderRegistry registry;
  require(test,
          !registry.add({capability->implementationFamily,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializeProvider}),
          "registry rejected a valid provider");
  expectError(test,
              registry.add({capability->implementationFamily,
                            BackendRecipeKey::PortableSystemVerilog,
                            {},
                            materializeProvider}),
              "duplicate");
  expectError(test,
              registry.add({static_cast<::fabric::ImplementationFamilyId>(
                                ::fabric::implementationFamilyCount()),
                            BackendRecipeKey::PortableSystemVerilog,
                            {},
                            materializeProvider}),
              "family");
  expectError(test,
              registry.add({capability->implementationFamily,
                            static_cast<BackendRecipeKey>(1000),
                            {},
                            materializeProvider}),
              "recipe");
  expectError(test,
              registry.add({capability->implementationFamily,
                            BackendRecipeKey::SynopsysDesignWare,
                            "synopsys.designware@1", nullptr}),
              "callback");

  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "coverage did not derive the complete normative family inventory");
  const auto &selected =
      coverage[static_cast<std::size_t>(capability->implementationFamily)];
  require(test,
          selected.implementationFamily == capability->implementationFamily &&
              selected.recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "coverage did not project the registered recipe");

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  SkeletonFixture skeleton = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves.front(), occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  ExternalImplementationContractCatalog contracts;
  observation.reset();
  providerInvocationCount = 0;
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(
                     *skeleton.module, fabric.module, abi, associations,
                     recipes, registry, contracts));
  require(test,
          observation.has_value() && observation->occurrence == occurrence &&
              observation->family == capability->implementationFamily &&
              observation->configurationFabric == fabric.module.reference() &&
              observation->recipe == BackendRecipeKey::PortableSystemVerilog &&
              observation->isolatedFragment &&
              !observation->hasImplementationPlatform &&
              observation->externalContractRef.empty() &&
              observation->externalInputs.empty() && output.payloads.empty() &&
              output.activityPoints.empty() &&
              output.externalImplementationBindings.empty(),
          "provider did not receive the exact occurrence contract");
  require(test, providerInvocationCount == 1,
          "specialization invoked the provider an unexpected number of times");
  const std::string systemVerilog =
      take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
  require(test,
          llvm::StringRef(systemVerilog).contains("module specialized_top"),
          "specialized module did not export SystemVerilog");
}

void connectedLeafKeepsItsInstanceContract(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  ArtifactStore store(root);
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  const FabricFuOccurrenceNodeRef occurrence =
      findOperationOccurrences(test, fabric.module.view(), 1).front();
  const auto *capability =
      fabric.module.view().resolvedFabricOpCapability(occurrence);
  require(test, capability != nullptr,
          "operation occurrence has no resolved capability");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add({capability->implementationFamily,
                                        BackendRecipeKey::PortableSystemVerilog,
                                        {},
                                        passthroughProvider}))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog contracts;
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  SkeletonFixture skeleton = makeConnectedSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves.front(), occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};

  providerInvocationCount = 0;
  take(test, specializeFabricOperationLeaves(*skeleton.module, fabric.module,
                                             abi, associations, recipes,
                                             registry, contracts));
  require(test, providerInvocationCount == 1,
          "connected specialization invoked the provider unexpectedly");
  auto top =
      skeleton.module->lookupSymbol<circt::hw::HWModuleOp>("connected_top");
  require(test, static_cast<bool>(top),
          "connected specialization removed the surrounding module");
  auto instances = llvm::to_vector(top.getOps<circt::hw::InstanceOp>());
  require(test,
          instances.size() == 1 &&
              instances.front().getModuleName() == "loom_fabric_operation_0" &&
              instances.front().getNumOperands() == 1 &&
              instances.front().getNumResults() == 1,
          "connected specialization changed the exact instance contract");
  require(test, mlir::succeeded(mlir::verify(*skeleton.module)),
          "connected specialization produced invalid CIRCT");
  const std::string systemVerilog =
      take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
  require(test,
          llvm::StringRef(systemVerilog).contains("module connected_top") &&
              llvm::StringRef(systemVerilog)
                  .contains("loom_fabric_operation_0 operation"),
          "connected specialization did not preserve the exported instance");
}

void providerInputsHaveCanonicalOrder(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  ArtifactStore store(root);
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  const FabricFuOccurrenceNodeRef occurrence =
      findOperationOccurrences(test, fabric.module.view(), 1).front();
  const auto *capability =
      fabric.module.view().resolvedFabricOpCapability(occurrence);
  require(test, capability != nullptr,
          "operation occurrence has no resolved capability");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({capability->implementationFamily,
                        BackendRecipeKey::SynopsysDesignWare,
                        "example.multi_input@1", vendorProvider}))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog contracts =
      multiSlotVendorContractCatalog(test);
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  SkeletonFixture skeleton = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves.front(), occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {occurrence,
       BackendRecipeKey::SynopsysDesignWare,
       {{"simulation",
         ToolBundledResourceDependency{"example.tool:1", "simulation:model"}},
        {"implementation",
         ToolBundledResourceDependency{"example.tool:1",
                                       "implementation:cell"}}}},
  };

  observation.reset();
  take(test, specializeFabricOperationLeaves(*skeleton.module, fabric.module,
                                             abi, associations, recipes,
                                             registry, contracts));
  require(test,
          observation && observation->externalInputs.size() == 2 &&
              observation->externalInputs[0].providerInputSlotRef ==
                  "implementation" &&
              observation->externalInputs[1].providerInputSlotRef ==
                  "simulation",
          "provider observed authoring order instead of canonical slot order");
}

void specializationPreflightIsFailClosed(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  ArtifactStore store(root);
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  FinalizedConfigurationABI foreignAbi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.system.reference(), {}}, store));
  std::vector<FabricFuOccurrenceNodeRef> occurrences =
      findOperationOccurrences(test, fabric.module.view(), 2);
  std::sort(occurrences.begin(), occurrences.end(),
            [](const auto &lhs, const auto &rhs) {
              return loom::fabric::canonicalFabricBytes(lhs) <
                     loom::fabric::canonicalFabricBytes(rhs);
            });
  const FabricFuOccurrenceNodeRef occurrence = occurrences.front();
  const auto *capability =
      fabric.module.view().resolvedFabricOpCapability(occurrence);
  require(test, capability != nullptr,
          "operation occurrence has no resolved capability");

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  const std::vector<FabricOperationRecipeBinding> recipe = {
      {occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  ExternalImplementationContractCatalog contracts;

  FabricOperationProviderRegistry empty;
  SkeletonFixture missing = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> missingAssociation = {
      {missing.leaves.front(), occurrence}};
  auto unsupportedResult = specializeFabricOperationLeaves(
      *missing.module, fabric.module, abi, missingAssociation, recipe, empty,
      contracts);
  require(test, !unsupportedResult,
          "missing provider produced a successful specialization");
  llvm::Error unsupported = unsupportedResult.takeError();
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      std::move(unsupported),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() == capability->implementationFamily &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
        require(test,
                llvm::StringRef(error.message())
                    .contains("cannot implement the exact capability with "
                              "recipe 'portable_system_verilog'"),
                "Unsupported diagnostic misclassified the exact capability");
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "missing provider returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "missing provider lost its typed Unsupported classification");
  require(test, static_cast<bool>(missing.leaves.front()),
          "provider preflight mutated the skeleton on Unsupported");

  FabricOperationProviderRegistry partial;
  if (llvm::Error error = partial.add({capability->implementationFamily,
                                       BackendRecipeKey::PortableSystemVerilog,
                                       {},
                                       materializeProvider}))
    fail(test, llvm::toString(std::move(error)));
  SkeletonFixture partiallyCovered = makeSkeleton(context, 2);
  const std::vector<FabricOperationLeafAssociation> partialAssociations = {
      {partiallyCovered.leaves[0], occurrences[0]},
      {partiallyCovered.leaves[1], occurrences[1]},
  };
  const std::vector<FabricOperationRecipeBinding> partialRecipes = {
      {occurrences[0], BackendRecipeKey::PortableSystemVerilog, {}},
      {occurrences[1], BackendRecipeKey::SynopsysDesignWare, {}},
  };
  observation.reset();
  expectError(test,
              specializeFabricOperationLeaves(
                  *partiallyCovered.module, fabric.module, abi,
                  partialAssociations, partialRecipes, partial, contracts),
              "provider_unsupported");
  require(test,
          !observation && static_cast<bool>(partiallyCovered.leaves[0]) &&
              static_cast<bool>(partiallyCovered.leaves[1]),
          "complete provider preflight invoked or mutated an earlier leaf");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add({capability->implementationFamily,
                                        BackendRecipeKey::PortableSystemVerilog,
                                        {},
                                        materializeProvider}))
    fail(test, llvm::toString(std::move(error)));
  SkeletonFixture wrongAbi = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> wrongAbiAssociation = {
      {wrongAbi.leaves.front(), occurrence}};
  observation.reset();
  expectError(test,
              specializeFabricOperationLeaves(*wrongAbi.module, fabric.module,
                                              foreignAbi, wrongAbiAssociation,
                                              recipe, registry, contracts),
              "ConfigurationABI");
  require(test, !observation && static_cast<bool>(wrongAbi.leaves.front()),
          "ABI preflight invoked or mutated a provider");

  SkeletonFixture unexpectedInputs = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> unexpectedInputAssociation =
      {{unexpectedInputs.leaves.front(), occurrence}};
  const std::vector<FabricOperationRecipeBinding> unexpectedInputRecipe = {
      {occurrence,
       BackendRecipeKey::PortableSystemVerilog,
       {{"implementation",
         ToolBundledResourceDependency{"unexpected:build",
                                       "unexpected:resource"}}}},
  };
  providerInvocationCount = 0;
  expectError(test,
              specializeFabricOperationLeaves(
                  *unexpectedInputs.module, fabric.module, abi,
                  unexpectedInputAssociation, unexpectedInputRecipe, registry,
                  contracts),
              "self-contained provider");
  require(test,
          providerInvocationCount == 0 &&
              static_cast<bool>(unexpectedInputs.leaves.front()),
          "external-input preflight invoked or mutated a provider");

  FabricOperationProviderRegistry vendorRegistry;
  if (llvm::Error error =
          vendorRegistry.add({capability->implementationFamily,
                              BackendRecipeKey::SynopsysDesignWare,
                              "synopsys.designware@1", vendorProvider}))
    fail(test, llvm::toString(std::move(error)));
  const std::vector<FabricOperationRecipeBinding> vendorRecipe = {
      {occurrence,
       BackendRecipeKey::SynopsysDesignWare,
       {{"implementation",
         ToolBundledResourceDependency{"synopsys.vcs:Y-2026.03-SP1",
                                       "designware:arithmetic"}}}},
  };
  SkeletonFixture unknownContract = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> vendorAssociation = {
      {unknownContract.leaves.front(), occurrence}};
  providerInvocationCount = 0;
  expectError(test,
              specializeFabricOperationLeaves(
                  *unknownContract.module, fabric.module, abi,
                  vendorAssociation, vendorRecipe, vendorRegistry, contracts),
              "provider contract is not registered");
  require(test,
          providerInvocationCount == 0 &&
              static_cast<bool>(unknownContract.leaves.front()),
          "unknown provider contract was checked after callback invocation");

  SkeletonFixture incompatibleInput = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> incompatibleAssociation = {
      {incompatibleInput.leaves.front(), occurrence}};
  ExternalImplementationContractCatalog explicitOnly =
      vendorContractCatalog(test, ExternalDependencyKind::ExplicitFile);
  providerInvocationCount = 0;
  expectError(test,
              specializeFabricOperationLeaves(
                  *incompatibleInput.module, fabric.module, abi,
                  incompatibleAssociation, vendorRecipe, vendorRegistry,
                  explicitOnly),
              "dependency kind is incompatible");
  require(test,
          providerInvocationCount == 0 &&
              static_cast<bool>(incompatibleInput.leaves.front()),
          "incompatible provider input was checked after callback invocation");

  SkeletonFixture missingPlatform = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> platformAssociation = {
      {missingPlatform.leaves.front(), occurrence}};
  ExternalImplementationContractCatalog platformRequired =
      vendorContractCatalog(test);
  const std::string beforeMissingPlatform = moduleText(*missingPlatform.module);
  providerInvocationCount = 0;
  expectError(test,
              specializeFabricOperationLeaves(*missingPlatform.module,
                                              fabric.module, abi,
                                              platformAssociation, vendorRecipe,
                                              vendorRegistry, platformRequired),
              "requires an implementation platform");
  require(test,
          providerInvocationCount == 1 &&
              moduleText(*missingPlatform.module) == beforeMissingPlatform,
          "catalog output validation partially committed specialization");

  SkeletonFixture missingRecipe = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> missingRecipeAssociation = {
      {missingRecipe.leaves.front(), occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *missingRecipe.module, fabric.module, abi,
                  missingRecipeAssociation, {}, registry, contracts),
              "recipe");
  require(test, static_cast<bool>(missingRecipe.leaves.front()),
          "recipe preflight mutated the skeleton");

  FabricOperationProviderRegistry incomplete;
  if (llvm::Error error =
          incomplete.add({capability->implementationFamily,
                          BackendRecipeKey::PortableSystemVerilog,
                          {},
                          incompleteProvider}))
    fail(test, llvm::toString(std::move(error)));
  SkeletonFixture unresolved = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> unresolvedAssociation = {
      {unresolved.leaves.front(), occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(*unresolved.module, fabric.module,
                                              abi, unresolvedAssociation,
                                              recipe, incomplete, contracts),
              "concrete replacement");
  require(test, static_cast<bool>(unresolved.leaves.front()),
          "invalid provider output mutated the caller module");

  FabricOperationProviderRegistry unboundExternal;
  if (llvm::Error error =
          unboundExternal.add({capability->implementationFamily,
                               BackendRecipeKey::PortableSystemVerilog,
                               {},
                               unboundExternalProvider}))
    fail(test, llvm::toString(std::move(error)));
  SkeletonFixture external = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> externalAssociation = {
      {external.leaves.front(), occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(*external.module, fabric.module,
                                              abi, externalAssociation, recipe,
                                              unboundExternal, contracts),
              "external module has no exact binding");
  require(test, static_cast<bool>(external.leaves.front()),
          "unbound external module mutated the caller module");

  FabricOperationProviderRegistry colliding;
  if (llvm::Error error =
          colliding.add({capability->implementationFamily,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         collidingProvider}))
    fail(test, llvm::toString(std::move(error)));
  SkeletonFixture collision = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> collisionAssociation = {
      {collision.leaves.front(), occurrence}};
  const std::string beforeCollision = moduleText(*collision.module);
  expectError(test,
              specializeFabricOperationLeaves(*collision.module, fabric.module,
                                              abi, collisionAssociation, recipe,
                                              colliding, contracts),
              "collides with the common skeleton");
  require(test, moduleText(*collision.module) == beforeCollision,
          "provider helper collision mutated the caller module");
}

void providerFailureIsTransactional(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  ArtifactStore store(root);
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  std::vector<FabricFuOccurrenceNodeRef> occurrences =
      findOperationOccurrences(test, fabric.module.view(), 2);
  std::sort(occurrences.begin(), occurrences.end(),
            [](const auto &lhs, const auto &rhs) {
              return loom::fabric::canonicalFabricBytes(lhs) <
                     loom::fabric::canonicalFabricBytes(rhs);
            });
  const auto *firstCapability =
      fabric.module.view().resolvedFabricOpCapability(occurrences[0]);
  const auto *secondCapability =
      fabric.module.view().resolvedFabricOpCapability(occurrences[1]);
  require(test, firstCapability && secondCapability,
          "operation occurrence has no resolved capability");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add({firstCapability->implementationFamily,
                                        BackendRecipeKey::PortableSystemVerilog,
                                        {},
                                        materializeProvider}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registry.add({secondCapability->implementationFamily,
                        BackendRecipeKey::SynopsysDesignWare,
                        "synopsys.designware@1", failingProvider}))
    fail(test, llvm::toString(std::move(error)));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  SkeletonFixture skeleton = makeSkeleton(context, 2);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves[0], occurrences[0]},
      {skeleton.leaves[1], occurrences[1]},
  };
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {occurrences[0], BackendRecipeKey::PortableSystemVerilog, {}},
      {occurrences[1],
       BackendRecipeKey::SynopsysDesignWare,
       {{"implementation",
         ToolBundledResourceDependency{"synopsys.vcs:Y-2026.03-SP1",
                                       "designware:arithmetic"}}}},
  };
  ExternalImplementationContractCatalog contracts = vendorContractCatalog(test);
  const std::string before = moduleText(*skeleton.module);
  providerInvocationCount = 0;
  expectError(test,
              specializeFabricOperationLeaves(*skeleton.module, fabric.module,
                                              abi, associations, recipes,
                                              registry, contracts),
              "provider preparation failed");
  require(test, moduleText(*skeleton.module) == before,
          "provider failure partially committed specialization");
  require(test, providerInvocationCount == 2,
          "transaction test did not fail after an earlier prepared provider");
}

void vendorBindingIsExplicit(llvm::StringRef root) {
  const llvm::StringRef test = __func__;
  const std::filesystem::path testRoot(root.str());
  std::filesystem::create_directories(testRoot / "vendor-artifacts");
  std::filesystem::create_directories(testRoot / "vendor-blobs");
  ArtifactStore store((testRoot / "vendor-artifacts").string());
  BlobStore blobs((testRoot / "vendor-blobs").string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fabric.module.reference(), {}}, store));
  auto platform =
      take(test, loom::platform::finalizeImplementationPlatform(
                     loom::platform::ImplementationPlatformDraft{
                         loom::platform::AsicTarget{"saed14", "EDK_08_2025"},
                         {"tt_0p80v_25c"}},
                     store));
  const FabricFuOccurrenceNodeRef occurrence =
      findOperationOccurrences(test, fabric.module.view(), 1).front();
  const auto *capability =
      fabric.module.view().resolvedFabricOpCapability(occurrence);
  require(test, capability != nullptr,
          "operation occurrence has no resolved capability");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({capability->implementationFamily,
                        BackendRecipeKey::SynopsysDesignWare,
                        "synopsys.designware@1", vendorProvider}))
    fail(test, llvm::toString(std::move(error)));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  SkeletonFixture skeleton = makeSkeleton(context);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves.front(), occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {occurrence,
       BackendRecipeKey::SynopsysDesignWare,
       {{"implementation",
         ToolBundledResourceDependency{"synopsys.vcs:Y-2026.03-SP1",
                                       "designware:arithmetic"}}}},
  };
  ExternalImplementationContractCatalog contracts = vendorContractCatalog(test);
  observation.reset();
  providerInvocationCount = 0;
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(
                     *skeleton.module, fabric.module, abi, associations,
                     recipes, registry, contracts, &platform.platform()));

  require(test,
          observation && observation->isolatedFragment &&
              observation->hasImplementationPlatform &&
              observation->externalContractRef == "synopsys.designware@1" &&
              observation->externalInputs.size() == 1,
          "vendor provider did not receive its exact environment binding");
  require(test, providerInvocationCount == 1,
          "vendor specialization invoked the provider unexpectedly");
  const auto *bundled = std::get_if<ToolBundledResourceDependency>(
      &observation->externalInputs.front().dependencyIdentity);
  require(test,
          bundled &&
              bundled->stableProviderBuildIdentity ==
                  "synopsys.vcs:Y-2026.03-SP1" &&
              bundled->resourceKey == "designware:arithmetic",
          "vendor provider did not receive its typed bundled resource");
  require(
      test,
      output.payloads.size() == 1 && output.activityPoints.size() == 1 &&
          output.externalImplementationBindings.size() == 1 &&
          output.externalImplementationBindings.front().providerContractRef ==
              "synopsys.designware@1" &&
          output.externalImplementationBindings.front().externalInputs.size() ==
              1,
      "vendor provider contribution was not preserved structurally");
  require(test,
          output.payloads.front().bytes ==
                  std::vector<std::uint8_t>{'{', '}'} &&
              output.payloads.front().descriptor().content ==
                  loom::computeBlobDigest(output.payloads.front().bytes),
          "vendor provider did not return digestible payload material");
  const auto expectedOccurrence =
      loom::fabric::encodeFabricArtifactLocalReference(
          loom::ArtifactReference<FabricFuOccurrenceNodeRef>{
              fabric.module.reference().artifact, occurrence});
  require(
      test,
      output.externalImplementationBindings.front().fabricResourceRefs ==
          std::vector<loom::EncodedArtifactLocalReference>{expectedOccurrence},
      "specialization did not derive the exact Fabric occurrence relation");
  require(test,
          skeleton.module->lookupSymbol<circt::hw::HWModuleExternOp>(
              "loom_fabric_operation_0"),
          "vendor provider did not commit its external replacement");

  const std::string systemVerilog =
      take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
  const std::vector<std::uint8_t> rtlBytes(systemVerilog.begin(),
                                           systemVerilog.end());
  const loom::BlobDigest rtlDigest = take(test, blobs.put(rtlBytes));
  std::vector<HardwarePayload> payloads = {{HardwarePayloadRole::RtlSource,
                                            "rtl/specialized.sv",
                                            "text/x-systemverilog", rtlDigest}};
  for (const FabricOperationProviderPayload &payload : output.payloads) {
    const HardwarePayload descriptor = payload.descriptor();
    require(test, take(test, blobs.put(payload.bytes)) == descriptor.content,
            "provider payload publication changed its content identity");
    payloads.push_back(descriptor);
  }
  HardwareImplementationDraft draft{
      fabric.module.reference(),
      abi.reference(),
      {},
      HardwareRepresentation::Rtl,
      platform.reference(),
      std::move(payloads),
      {},
      output.activityPoints,
      {},
      output.externalImplementationBindings,
  };
  const FinalizedHardwareImplementation implementation =
      take(test, finalizeHardwareImplementation(std::move(draft), contracts,
                                                store, blobs));
  require(
      test,
      implementation.implementation().externalImplementationBindings().size() ==
          1,
      "provider contribution did not finalize as HardwareImplementation");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  registryAndSpecializationAreExact(argv[1]);
  connectedLeafKeepsItsInstanceContract(argv[1]);
  providerInputsHaveCanonicalOrder(argv[1]);
  specializationPreflightIsFailClosed(argv[1]);
  providerFailureIsTransactional(argv[1]);
  vendorBindingIsExplicit(argv[1]);
  return 0;
}
