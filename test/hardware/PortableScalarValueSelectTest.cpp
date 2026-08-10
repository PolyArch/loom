#include "ADG/Builder.h"
#include "ADG/FuLibrary.h"
#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarValueSelect.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::adg::DesignBuilder;
using loom::adg::FinalizedFabricDesign;
using loom::adg::FuCapabilityTemplateSpec;
using loom::adg::FuSpec;
using loom::adg::OperationCapabilitySpec;
using loom::adg::PeSpec;
using loom::adg::PortType;
using loom::fabric::FabricFuOccurrenceNodeRef;
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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid scalar value select input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid scalar value select input");
  expectError(test, value.takeError(), expected);
}

template <typename T>
void expectTypedUnsupported(llvm::StringRef test, llvm::Expected<T> value,
                            ::fabric::ImplementationFamilyId family,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() == family &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
}

::fabric::IntegerWidthSet completeIntegerWidths() {
  ::fabric::IntegerWidthSet result;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    result.insert(width);
  return result;
}

::fabric::IntegerWidthSet ordinaryIntegerWidths() {
  ::fabric::IntegerWidthSet result;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (::fabric::getBitWidth(width) != 1)
      result.insert(width);
  return result;
}

::fabric::FloatFormatSet completeFloatFormats() {
  ::fabric::FloatFormatSet result;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
    result.insert(format);
  return result;
}

::fabric::ScalarValueSelectParams scalarValueSelectParams() {
  return {completeIntegerWidths(), completeFloatFormats()};
}

struct FabricFixture final {
  FinalizedFabricDesign design;
  FabricFuOccurrenceNodeRef occurrence;
  loom::fabric::FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;

  const loom::fabric::FinalizedFabricRoot &root() const {
    return design.roots().front();
  }
};

FabricFixture findFixture(llvm::StringRef test, const ArtifactStore &store,
                          FinalizedFabricDesign design,
                          ::fabric::ImplementationFamilyId family) {
  require(test, design.roots().size() == 1,
          "Fabric fixture did not finalize one root");
  const auto &root = design.roots().front();
  for (const auto fuOccurrence : root.view().fuOccurrences()) {
    const auto definition = root.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &candidate :
         root.view().resolvedFabricOpCapabilities(*definition)) {
      if (candidate.implementationFamily != family)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         root.view(), candidate.occurrence, fuOccurrence));
      loom::fabric::FinalizedFabricRoot system = take(
          test, loom::hardware::test::makeSingleSpatialCoreSystem(root, store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical scalar value select occurrence");
      return FabricFixture{std::move(design), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no requested operation occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.root().view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

FabricFixture makeOperationFabric(llvm::StringRef test,
                                  const ArtifactStore &store,
                                  llvm::StringRef label,
                                  ::fabric::ImplementationFamilyId family,
                                  ::fabric::FamilyCapabilityParams parameters,
                                  llvm::ArrayRef<unsigned> inputWidths,
                                  unsigned outputWidth,
                                  bool unsupportedContract = false) {
  DesignBuilder design(store);
  std::vector<PortType> operationInputTypes;
  operationInputTypes.reserve(inputWidths.size());
  for (unsigned width : inputWidths)
    operationInputTypes.push_back(take(test, PortType::bits(width)));
  const PortType operationOutputType = take(test, PortType::bits(outputWidth));
  const unsigned boundaryWidth = std::max(
      outputWidth, *std::max_element(inputWidths.begin(), inputWidths.end()));
  const PortType boundaryType = take(test, PortType::bits(boundaryWidth));
  const std::vector<PortType> boundaryInputTypes(inputWidths.size(),
                                                 boundaryType);

  auto spatial = take(test, design.createSpatialCore(label, boundaryInputTypes,
                                                     {boundaryType}));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs,
                          PeSpec::spatial(boundaryInputTypes, {boundaryType})));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(
      test, pe.addFu(peInputs, FuSpec{operationInputTypes, {boundaryType}}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != operationInputTypes.size();
       ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));

  const auto &descriptor = ::fabric::implementationFamily(family);
  const ::fabric::ResourceContract contract =
      unsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  auto operation =
      take(test, fu.addOperation(fuInputs,
                                 OperationCapabilitySpec{
                                     family,
                                     std::move(parameters),
                                     std::vector<::dataflow::OperationSchemaId>(
                                         descriptor.admittedSchemas.begin(),
                                         descriptor.admittedSchemas.end()),
                                     {operationOutputType},
                                     contract}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  return findFixture(test, store, take(test, std::move(design).finalize()),
                     family);
}

FabricFixture makeCoreAluFabric(llvm::StringRef test,
                                const ArtifactStore &store) {
  DesignBuilder design(store);
  const PortType bits128 = take(test, PortType::bits(128));
  auto spatial = take(
      test, design.createSpatialCore("scalar-value-select-core-alu",
                                     {bits128, bits128, bits128}, {bits128}));
  auto pe = take(
      test,
      spatial.addPe({take(test, spatial.input(0)), take(test, spatial.input(1)),
                     take(test, spatial.input(2))},
                    PeSpec::spatial({bits128, bits128, bits128}, {bits128})));
  std::vector<loom::adg::PeValue> inputs;
  for (std::size_t ordinal = 0; ordinal != 3; ++ordinal)
    inputs.push_back(take(test, pe.input(ordinal)));
  if (llvm::Error error =
          loom::adg::addCoreAluFu(pe, inputs,
                                  ::fabric::ResolvedIndexWidthSet::get(
                                      {::fabric::ResolvedIndexWidth::I64})))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  return findFixture(test, store, take(test, std::move(design).finalize()),
                     ::fabric::ImplementationFamilyId::ScalarValueSelect);
}

void requirePhysicalShape(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
    llvm::ArrayRef<unsigned> expectedInputs, unsigned expectedOutput) {
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : resolved.physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  require(test, inputs.size() == expectedInputs.size() && outputs.size() == 1,
          "select capability changed its physical port arity");
  for (auto [ordinal, expectedWidth] : llvm::enumerate(expectedInputs))
    require(test,
            inputs[ordinal]->reference.ordinal == ordinal &&
                inputs[ordinal]->payloadWidthBits == expectedWidth,
            "select capability changed an input physical role");
  require(test,
          outputs.front()->reference.ordinal == 0 &&
              outputs.front()->payloadWidthBits == expectedOutput,
          "select capability changed its result physical role");
}

::dataflow::OperationSchemaId selectSchema(llvm::StringRef test) {
  const auto schema = ::dataflow::findOperationSchema("arith.select");
  require(test, schema.has_value(),
          "generated operation registry has no arith.select schema");
  return *schema;
}

::dataflow::CanonicalActorSchemaProjection
selectActor(llvm::StringRef test, mlir::MLIRContext &context,
            mlir::Type valueType) {
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir::Block block;
  const mlir::Location location = mlir::UnknownLoc::get(&context);
  mlir::Value condition =
      block.addArgument(mlir::IntegerType::get(&context, 1), location);
  mlir::Value trueValue = block.addArgument(valueType, location);
  mlir::Value falseValue = block.addArgument(valueType, location);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&block);
  mlir::arith::SelectOp operation = mlir::arith::SelectOp::create(
      builder, location, condition, trueValue, falseValue);
  return take(test,
              ::dataflow::projectRegisteredActorSchemaProjection(operation));
}

void generatedRegistryAndCoreAluAreAuthoritative(const FabricFixture &coreAlu) {
  const llvm::StringRef test = __func__;
  const auto schema = selectSchema(test);
  const auto &descriptor = ::fabric::implementationFamily(
      ::fabric::ImplementationFamilyId::ScalarValueSelect);
  require(
      test,
      descriptor.familyId ==
              ::fabric::ImplementationFamilyId::ScalarValueSelect &&
          descriptor.capabilityParamsSchema ==
              ::fabric::CapabilityParamsSchemaId::ScalarValueSelectParams &&
          descriptor.typedAdmissionProvider ==
              ::fabric::TypedAdmissionProviderId::ScalarValueSelectAdmission &&
          descriptor.admittedSchemas.size() == 1 &&
          descriptor.admittedSchemas.front() == schema,
      "generated scalar value select descriptor changed");
  require(test,
          llvm::is_contained(::fabric::implementationFamiliesFor(schema),
                             descriptor.familyId),
          "generated family relation lost arith.select");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
    return candidate.implementationFamily == descriptor.familyId;
  });
  require(test,
          coverage.size() == ::fabric::implementationFamilyCount() &&
              entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "scalar value select provider coverage changed");

  const auto &resolved = capability(test, coreAlu);
  require(test,
          resolved.implementationFamily == descriptor.familyId &&
              llvm::equal(resolved.enabledOperationSchemas,
                          descriptor.admittedSchemas) &&
              resolved.configurationFieldSchema.empty(),
          "CoreAlu select escaped its generated family contract");
  const auto *parameters = std::get_if<::fabric::ScalarValueSelectParams>(
      &resolved.parameterizedCapability);
  require(test, parameters != nullptr,
          "CoreAlu select changed its parameter schema");
  require(test,
          parameters->integerWidths.valid() &&
              parameters->integerWidths.size() ==
                  ::fabric::integerWidthDomain.size() &&
              llvm::all_of(::fabric::integerWidthDomain,
                           [&](auto width) {
                             return parameters->integerWidths.contains(width);
                           }) &&
              parameters->floatFormats.valid() &&
              parameters->floatFormats.size() ==
                  ::fabric::floatFormatDomain.size() &&
              llvm::all_of(::fabric::floatFormatDomain,
                           [&](auto format) {
                             return parameters->floatFormats.contains(format);
                           }),
          "CoreAlu select lost its canonical scalar value domains");
  requirePhysicalShape(test, resolved, {1, 64, 64}, 64);

  const auto actualContract =
      take(test, ::fabric::encodeResourceContractRecord(
                     resolved.resourceStateAndTimingContract));
  const auto expectedContract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  require(test, actualContract == expectedContract,
          "CoreAlu select changed its one-cycle elastic contract");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  auto actor = selectActor(test, context, mlir::IntegerType::get(&context, 64));
  if (llvm::Error error =
          resolved.admitCorrespondence(actor, 64, {0, 1, 2}, {0}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, resolved.admitCorrespondence(actor, 64, {1, 0, 2}, {0}),
              "fixed physical role");
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.empty(),
          "configuration-free fixture created a semantic field");
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system)),
          store));
}

std::unique_ptr<mlir::MLIRContext> makeCirctContext() {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

enum class LeafMutation {
  None,
  WrongConditionWidth,
  ExtraConfigurationPort,
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             llvm::StringRef symbol,
                             LeafMutation mutation = LeafMutation::None) {
  const auto &resolved = capability(test, fabric);
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, resolved, abi));
  if (mutation != LeafMutation::None)
    require(test, ports.size() == 4,
            "select leaf did not derive three inputs and one result");
  if (mutation == LeafMutation::WrongConditionWidth) {
    ports.front().type = builder.getIntegerType(
        mlir::cast<mlir::IntegerType>(ports.front().type).getWidth() == 1 ? 2
                                                                          : 1);
  } else if (mutation == LeafMutation::ExtraConfigurationPort) {
    ports.insert(ports.begin() + 3,
                 circt::hw::PortInfo{
                     {builder.getStringAttr("config_0"), builder.getI1Type(),
                      circt::hw::ModulePort::Direction::Input}});
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(symbol), ports);
  return SkeletonFixture{std::move(module), leaf};
}

struct SpecializedRtl final {
  std::string text;
  unsigned muxCount = 0;
};

SpecializedRtl specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                          const FabricFixture &fabric,
                          const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(module), abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable select emitted external implementation state");
  const unsigned muxCount =
      llvm::StringRef(conformance.systemVerilog).count(" ? ");
  return {std::move(conformance.systemVerilog), muxCount};
}

struct EmittedRtl final {
  std::string dense;
  std::string wideCondition;
};

EmittedRtl mixedCapabilityUsesOneMux(llvm::StringRef test,
                                     const ArtifactStore &store,
                                     const FabricFixture &coreAlu,
                                     const FabricFixture &dense,
                                     const FabricFixture &wideCondition) {
  const auto &core = capability(test, coreAlu);
  const auto &mixed = capability(test, dense);
  const auto &wide = capability(test, wideCondition);
  require(test,
          mixed.implementationFamily ==
                  ::fabric::ImplementationFamilyId::ScalarValueSelect &&
              llvm::equal(mixed.enabledOperationSchemas,
                          core.enabledOperationSchemas) &&
              mixed.configurationFieldSchema.empty(),
          "mixed select fixture changed the generated semantic contract");
  const auto &mixedParameters = std::get<::fabric::ScalarValueSelectParams>(
      mixed.parameterizedCapability);
  require(test,
          mixedParameters.integerWidths.size() ==
                  ::fabric::integerWidthDomain.size() &&
              mixedParameters.floatFormats.size() ==
                  ::fabric::floatFormatDomain.size(),
          "mixed select fixture did not retain both scalar domains");
  requirePhysicalShape(test, mixed, {1, 64, 64}, 64);
  requirePhysicalShape(test, wide, {64, 64, 64}, 64);

  FinalizedConfigurationABI denseAbi = makeConfigurationAbi(test, store, dense);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, dense,
                                       denseAbi.abi(), "scalar_value_select");
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(
      test,
      ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
          ports.atInput(1).getName() == "data_input_1" &&
          ports.atInput(2).getName() == "data_input_2" &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              1 &&
          mlir::cast<mlir::IntegerType>(ports.atInput(1).type).getWidth() ==
              64 &&
          mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() ==
              64 &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              64,
      "derived select leaf ports are not canonical");
  for (const auto &port : ports)
    require(test, !port.getName().starts_with("config_"),
            "mixed select leaf retained a configuration port");
  const SpecializedRtl firstRtl = specialize(test, first, dense, denseAbi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second = makeSkeleton(test, *secondContext, dense,
                                        denseAbi.abi(), "scalar_value_select");
  const SpecializedRtl secondRtl = specialize(test, second, dense, denseAbi);
  require(test,
          firstRtl.muxCount == 1 && secondRtl.muxCount == 1 &&
              firstRtl.text == secondRtl.text &&
              !llvm::StringRef(firstRtl.text).contains("config_"),
          "mixed select did not emit one deterministic configuration-free mux");

  FinalizedConfigurationABI wideAbi =
      makeConfigurationAbi(test, store, wideCondition);
  std::unique_ptr<mlir::MLIRContext> wideContext = makeCirctContext();
  SkeletonFixture wideSkeleton =
      makeSkeleton(test, *wideContext, wideCondition, wideAbi.abi(),
                   "scalar_value_select_wide_condition");
  const SpecializedRtl wideRtl =
      specialize(test, wideSkeleton, wideCondition, wideAbi);
  require(test,
          wideRtl.muxCount == 1 &&
              llvm::StringRef(wideRtl.text).contains("data_input_0[0]") &&
              !llvm::StringRef(wideRtl.text).contains("config_"),
          "wide condition did not select through its low bit");
  return {firstRtl.text, wideRtl.text};
}

void malformedLeavesAreTransactional(llvm::StringRef test,
                                     const ArtifactStore &store,
                                     const FabricFixture &fabric) {
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  for (LeafMutation mutation : {LeafMutation::WrongConditionWidth,
                                LeafMutation::ExtraConfigurationPort}) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, fabric, abi.abi(),
                     "malformed_scalar_value_select", mutation);
    ModuleRootCirctSkeleton module{
        std::move(skeleton.module),
        {{skeleton.leaf, fabric.physicalOccurrence}}};
    expectError(test,
                loom::hardware::test::specializeAndExportPortableProvider(
                    std::move(module), abi, registry, externalContracts),
                "leaf port");
  }
}

void unsupportedResourceContractIsTransactional(llvm::StringRef test,
                                                const ArtifactStore &store) {
  FabricFixture fabric = makeOperationFabric(
      test, store, "scalar-value-select-unsupported-contract",
      ::fabric::ImplementationFamilyId::ScalarValueSelect,
      scalarValueSelectParams(), {1, 64, 64}, 64, true);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi(),
                                          "unsupported_contract_value_select");
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  expectTypedUnsupported(
      test,
      loom::hardware::test::specializeAndExportPortableProvider(
          std::move(module), abi, registry, externalContracts),
      ::fabric::ImplementationFamilyId::ScalarValueSelect,
      "unsupported select resource contract");
}

void unsupportedPhysicalShapeIsTransactional(llvm::StringRef test,
                                             const ArtifactStore &store) {
  FabricFixture fabric =
      makeOperationFabric(test, store, "scalar-value-select-unsupported-shape",
                          ::fabric::ImplementationFamilyId::ScalarValueSelect,
                          scalarValueSelectParams(), {1, 64, 64, 64}, 64);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi(),
                                          "unsupported_shape_value_select");
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  expectTypedUnsupported(
      test,
      loom::hardware::test::specializeAndExportPortableProvider(
          std::move(module), abi, registry, externalContracts),
      ::fabric::ImplementationFamilyId::ScalarValueSelect,
      "unsupported select physical shape");
}

void anotherFamilyIsTypedUnsupported(llvm::StringRef test,
                                     const ArtifactStore &store,
                                     const FabricFixture &other) {
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, other);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(
      test, *context, other, abi.abi(), "unsupported_scalar_integer_multiply");
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, other.physicalOccurrence}}};
  expectTypedUnsupported(
      test,
      loom::hardware::test::specializeAndExportPortableProvider(
          std::move(module), abi, registry, externalContracts),
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      "wrong-family input");
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? std::numeric_limits<std::uint64_t>::max()
                     : (std::uint64_t{1} << width) - 1;
}

std::string hex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

void writeToolInputs(const std::filesystem::path &root,
                     const EmittedRtl &emitted,
                     const ::fabric::ScalarValueSelectParams &parameters) {
  std::ostringstream testbench;
  testbench << R"sv(module testbench;
  logic condition;
  logic [63:0] true_value;
  logic [63:0] false_value;
  logic [63:0] result;

  scalar_value_select dense(
    .data_input_0(condition),
    .data_input_1(true_value),
    .data_input_2(false_value),
    .data_output_0(result));

  initial begin
)sv";
  for (::fabric::IntegerWidth integerWidth : ::fabric::integerWidthDomain) {
    if (!parameters.integerWidths.contains(integerWidth))
      continue;
    const unsigned width = ::fabric::getBitWidth(integerWidth);
    const std::uint64_t mask = widthMask(width);
    const std::uint64_t trueBits = UINT64_C(0xa5a5a5a5a5a5a5a5) & mask;
    const std::uint64_t falseBits = UINT64_C(0x5a5a5a5a5a5a5a5a) & mask;
    testbench << "    true_value = 64'h" << hex64(trueBits) << ";\n"
              << "    false_value = 64'h" << hex64(falseBits) << ";\n"
              << "    condition = 1'b1;\n"
              << "    #1;\n"
              << "    if (result !== 64'h" << hex64(trueBits)
              << ") $fatal(1, \"integer " << width
              << "-bit true branch failed\");\n"
              << "    condition = 1'b0;\n"
              << "    #1;\n"
              << "    if (result !== 64'h" << hex64(falseBits)
              << ") $fatal(1, \"integer " << width
              << "-bit false branch failed\");\n\n";
  }
  std::uint64_t formatOrdinal = 0;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    if (!parameters.floatFormats.contains(format))
      continue;
    const unsigned width = ::fabric::getBitWidth(format);
    const std::uint64_t mask = widthMask(width);
    const std::uint64_t salt = ++formatOrdinal * UINT64_C(0x0101010101010101);
    const std::uint64_t trueBits = (UINT64_C(0x7ff8000000000001) ^ salt) & mask;
    const std::uint64_t falseBits =
        (UINT64_C(0x8000000000000000) ^ (salt << 1)) & mask;
    testbench << "    true_value = 64'h" << hex64(trueBits) << ";\n"
              << "    false_value = 64'h" << hex64(falseBits) << ";\n"
              << "    condition = 1'b1;\n"
              << "    #1;\n"
              << "    if (result !== 64'h" << hex64(trueBits)
              << ") $fatal(1, \"floating " << width
              << "-bit true pattern failed\");\n"
              << "    condition = 1'b0;\n"
              << "    #1;\n"
              << "    if (result !== 64'h" << hex64(falseBits)
              << ") $fatal(1, \"floating " << width
              << "-bit false pattern failed\");\n\n";
  }
  testbench << R"sv(    $finish;
  end
endmodule
)sv";

  const std::string synthesisTop = R"sv(
module scalar_value_select_synthesis_top(
  input logic condition,
  input logic [63:0] wide_condition,
  input logic [63:0] true_value,
  input logic [63:0] false_value,
  output logic [63:0] dense_result,
  output logic [63:0] wide_result);
  scalar_value_select dense(
    .data_input_0(condition), .data_input_1(true_value),
    .data_input_2(false_value), .data_output_0(dense_result));
  scalar_value_select_wide_condition wide(
    .data_input_0(wide_condition), .data_input_1(true_value),
    .data_input_2(false_value), .data_output_0(wide_result));
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv scalar_value_select.sv scalar_value_select_wide_condition.sv synthesis_top.sv
hierarchy -check -top scalar_value_select_synthesis_top
proc
opt
check -assert
synth -top scalar_value_select_synthesis_top
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "provider_artifacts",
          {{"scalar_value_select.sv", emitted.dense},
           {"scalar_value_select_wide_condition.sv", emitted.wideCondition},
           {"testbench.sv", testbench.str()},
           {"synthesis_top.sv", synthesisTop},
           {"portable_scalar_value_select.ys", yosysScript}}))
    fail("writeToolInputs", llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const llvm::StringRef test = "portableScalarValueSelect";
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());

  FabricFixture coreAlu = makeCoreAluFabric(test, store);
  generatedRegistryAndCoreAluAreAuthoritative(coreAlu);
  FabricFixture dense =
      makeOperationFabric(test, store, "scalar-value-select-dense",
                          ::fabric::ImplementationFamilyId::ScalarValueSelect,
                          scalarValueSelectParams(), {1, 64, 64}, 64);
  FabricFixture wideCondition =
      makeOperationFabric(test, store, "scalar-value-select-wide-condition",
                          ::fabric::ImplementationFamilyId::ScalarValueSelect,
                          scalarValueSelectParams(), {64, 64, 64}, 64);
  const EmittedRtl emitted =
      mixedCapabilityUsesOneMux(test, store, coreAlu, dense, wideCondition);
  malformedLeavesAreTransactional(test, store, dense);
  unsupportedResourceContractIsTransactional(test, store);
  unsupportedPhysicalShapeIsTransactional(test, store);

  FabricFixture other = makeOperationFabric(
      test, store, "scalar-integer-multiply-unsupported",
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      ::fabric::ScalarIntegerParams{ordinaryIntegerWidths()}, {64, 64}, 64);
  anotherFamilyIsTypedUnsupported(test, store, other);
  writeToolInputs(root, emitted,
                  std::get<::fabric::ScalarValueSelectParams>(
                      capability(test, dense).parameterizedCapability));
  return 0;
}
