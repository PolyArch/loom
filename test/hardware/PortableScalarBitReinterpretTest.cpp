#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/ScalarBitReinterpret.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
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
    fail(test, "accepted invalid portable scalar bit reinterpret input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
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

mlir::MLIRContext &fabricContext() {
  static mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *context;
}

const ::fabric::ImplementationFamilyDescriptor &bitReinterpretDescriptor() {
  return ::fabric::implementationFamily(
      ::fabric::ImplementationFamilyId::ScalarBitReinterpret);
}

::dataflow::CanonicalActorSchemaProjection
endpointActor(llvm::StringRef test, mlir::Type source, mlir::Type destination) {
  mlir::Block block;
  const mlir::Location location = mlir::UnknownLoc::get(&fabricContext());
  mlir::Value input = block.addArgument(source, location);
  mlir::OpBuilder builder(&fabricContext());
  builder.setInsertionPointToEnd(&block);
  mlir::arith::BitcastOp operation =
      mlir::arith::BitcastOp::create(builder, location, destination, input);
  return take(test,
              ::dataflow::projectRegisteredActorSchemaProjection(operation));
}

bool admitsSingletonEndpoint(llvm::StringRef test,
                             const ::fabric::FamilyCapabilityParams &params,
                             mlir::Type type) {
  llvm::Error error = ::fabric::verifyImplementationFamilyAdmission(
      ::fabric::ImplementationFamilyId::ScalarBitReinterpret, &params,
      endpointActor(test, type, type));
  if (!error)
    return true;
  llvm::consumeError(std::move(error));
  return false;
}

struct CanonicalFloatEndpoint final {
  mlir::Type type;
  std::string spelling;
};

CanonicalFloatEndpoint canonicalFloatEndpoint(llvm::StringRef test,
                                              ::fabric::FloatFormat format) {
  ::fabric::FamilyCapabilityParams singleton =
      ::fabric::ScalarBitReinterpretParams{
          {}, ::fabric::FloatFormatSet::get({format})};
  mlir::DictionaryAttr encoded =
      ::fabric::getFamilyCapabilityParamsAttr(&fabricContext(), singleton);
  auto formats = mlir::dyn_cast<mlir::ArrayAttr>(encoded.get("float_formats"));
  require(test, formats && formats.size() == 1,
          "canonical floating endpoint did not encode one format");
  auto spelling = mlir::dyn_cast<mlir::StringAttr>(formats[0]);
  require(test, static_cast<bool>(spelling),
          "canonical floating endpoint spelling is not a string");
  const std::string source = "module { func.func private @endpoint(" +
                             spelling.getValue().str() + ") -> " +
                             spelling.getValue().str() + " }";
  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(source, &fabricContext());
  require(test, static_cast<bool>(parsed),
          "canonical floating endpoint spelling did not parse as a type");
  mlir::func::FuncOp endpoint;
  parsed->walk([&](mlir::func::FuncOp candidate) { endpoint = candidate; });
  require(test, static_cast<bool>(endpoint),
          "canonical floating endpoint fixture has no function");
  return {endpoint.getFunctionType().getInput(0), spelling.getValue().str()};
}

::fabric::ScalarBitReinterpretParams fullEndpointParams(llvm::StringRef test) {
  const auto &descriptor = bitReinterpretDescriptor();
  require(test, !descriptor.admittedSchemas.empty(),
          "generated bit reinterpret family has no admitted schema");

  ::fabric::IntegerWidthSet integerWidths;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain) {
    ::fabric::FamilyCapabilityParams singleton =
        ::fabric::ScalarBitReinterpretParams{
            ::fabric::IntegerWidthSet::get({width}), {}};
    mlir::Type type =
        mlir::IntegerType::get(&fabricContext(), ::fabric::getBitWidth(width));
    if (admitsSingletonEndpoint(test, singleton, type))
      integerWidths.insert(width);
  }

  ::fabric::FloatFormatSet floatFormats;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    ::fabric::FamilyCapabilityParams singleton =
        ::fabric::ScalarBitReinterpretParams{
            {}, ::fabric::FloatFormatSet::get({format})};
    CanonicalFloatEndpoint endpoint = canonicalFloatEndpoint(test, format);
    if (admitsSingletonEndpoint(test, singleton, endpoint.type))
      floatFormats.insert(format);
  }
  require(test, !integerWidths.empty() && !floatFormats.empty(),
          "canonical admission produced an empty endpoint domain");
  return {integerWidths, floatFormats};
}

std::string attributeText(mlir::Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

enum class ContractKind { OneCycleElastic, Wrong };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  unsigned inputWidth = 0;
  unsigned outputWidth = 0;
};

void attachContract(llvm::StringRef test, mlir::ModuleOp module,
                    ContractKind kind) {
  const ::fabric::ResourceContract contract =
      kind == ContractKind::OneCycleElastic
          ? ::fabric::oneCycleElasticOperationResourceContract()
          : ::fabric::loopCarryOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedContract(encoded.begin(), encoded.end());
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });
}

FabricFixture finalizeFixture(llvm::StringRef test, const ArtifactStore &store,
                              llvm::StringRef sourceText,
                              ::fabric::ImplementationFamilyId family,
                              unsigned inputWidth, unsigned outputWidth,
                              ContractKind contract) {
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  if (!source)
    fail(test, "could not parse Fabric fixture:\n" + sourceText.str());
  attachContract(test, *source, contract);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != family)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return {std::move(fabric), occurrence, inputWidth, outputWidth};
    }
  }
  fail(test, "Fabric fixture has no expected operation occurrence");
}

FabricFixture makeBitReinterpretFabric(
    llvm::StringRef test, const ArtifactStore &store, unsigned inputWidth,
    unsigned outputWidth,
    ContractKind contract = ContractKind::OneCycleElastic) {
  const auto &descriptor = bitReinterpretDescriptor();
  const ::fabric::FamilyCapabilityParams params = fullEndpointParams(test);
  const std::string paramsText = attributeText(
      ::fabric::getFamilyCapabilityParamsAttr(&fabricContext(), params));
  const unsigned containerWidth = std::max(inputWidth, outputWidth);
  std::string sourceText;
  llvm::raw_string_ostream source(sourceText);
  source << "module { fabric.module @scalar_bit_reinterpret_" << inputWidth
         << "_to_" << outputWidth << "(%a: !fabric.bits<" << containerWidth
         << ">) -> !fabric.bits<" << containerWidth
         << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<"
         << containerWidth << ">) -> !fabric.bits<" << containerWidth
         << "> { %fu = fabric.fu (%fa = %pa : !fabric.bits<" << containerWidth
         << ">";
  if (inputWidth != containerWidth)
    source << " to !fabric.bits<" << inputWidth << ">";
  source << ") -> !fabric.bits<" << containerWidth
         << "> { %value = fabric.op [@"
         << ::dataflow::operationSchemaSpelling(
                descriptor.admittedSchemas.front())
         << "] (%fa) {implementation_family = "
            "#fabric.implementation_family<"
         << ::fabric::implementationFamilyKeyword(descriptor.familyId)
         << ">, hw_params = " << paramsText << "} : (!fabric.bits<"
         << inputWidth << ">) -> !fabric.bits<" << outputWidth
         << "> fabric.yield %value : !fabric.bits<" << outputWidth << ">";
  if (outputWidth != containerWidth)
    source << " to !fabric.bits<" << containerWidth << ">";
  source << " } } fabric.yield %pe : !fabric.bits<" << containerWidth
         << "> } }";
  return finalizeFixture(test, store, source.str(), descriptor.familyId,
                         inputWidth, outputWidth, contract);
}

::fabric::IntegerWidth firstAdmittedInteger(llvm::StringRef test) {
  const auto parameters = fullEndpointParams(test);
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (parameters.integerWidths.contains(width))
      return width;
  fail(test, "bit reinterpret domain has no admitted integer width");
}

FabricFixture makeOtherFamilyFabric(llvm::StringRef test,
                                    const ArtifactStore &store) {
  const auto family = ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
  const auto &descriptor = ::fabric::implementationFamily(family);
  const ::fabric::IntegerWidth width = firstAdmittedInteger(test);
  const unsigned bits = ::fabric::getBitWidth(width);
  const ::fabric::FamilyCapabilityParams params =
      ::fabric::ScalarIntegerParams{::fabric::IntegerWidthSet::get({width})};
  const std::string paramsText = attributeText(
      ::fabric::getFamilyCapabilityParamsAttr(&fabricContext(), params));
  std::string sourceText;
  llvm::raw_string_ostream source(sourceText);
  source << "module { fabric.module @other_family(%a: !fabric.bits<" << bits
         << ">, %b: !fabric.bits<" << bits << ">) -> !fabric.bits<" << bits
         << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<" << bits
         << ">, %pb = %b : !fabric.bits<" << bits << ">) -> !fabric.bits<"
         << bits << "> { %fu = fabric.fu (%fa = %pa : !fabric.bits<" << bits
         << ">, %fb = %pb : !fabric.bits<" << bits << ">) -> !fabric.bits<"
         << bits << "> { %value = fabric.op [@"
         << ::dataflow::operationSchemaSpelling(
                descriptor.admittedSchemas.front())
         << "] (%fa, %fb) {implementation_family = "
            "#fabric.implementation_family<"
         << ::fabric::implementationFamilyKeyword(family)
         << ">, hw_params = " << paramsText << "} : (!fabric.bits<" << bits
         << ">, !fabric.bits<" << bits << ">) -> !fabric.bits<" << bits
         << "> fabric.yield %value : !fabric.bits<" << bits
         << "> } } fabric.yield %pe : !fabric.bits<" << bits << "> } }";
  return finalizeFixture(test, store, source.str(), family, bits, bits,
                         ContractKind::OneCycleElastic);
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  require(test, capability->configurationFieldSchema.empty(),
          "bit reinterpret capability created a configuration field");
  FinalizedConfigurationABI abi = take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fixture.fabric.reference(), {}}, store));
  require(test, abi.abi().programmingUnits().empty(),
          "configuration-free capability created a programming unit");
  return abi;
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

enum class LeafMutation { None, WrongInputWidth, ExtraConfigurationPort };

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             llvm::StringRef symbol,
                             LeafMutation mutation = LeafMutation::None) {
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(builder, *capability, abi));
  if (mutation == LeafMutation::WrongInputWidth) {
    require(test, !ports.empty(), "operation leaf has no input port");
    ports.front().type = builder.getIntegerType(fabric.inputWidth + 1);
  } else if (mutation == LeafMutation::ExtraConfigurationPort) {
    require(test, ports.size() >= 2, "operation leaf has no output port");
    ports.insert(ports.end() - 1,
                 circt::hw::PortInfo{
                     {builder.getStringAttr("config_0"), builder.getI1Type(),
                      circt::hw::ModulePort::Direction::Input}});
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(symbol), ports);
  return {std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarBitReinterpretProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(
                     *skeleton.module, fabric.fabric, abi, associations,
                     recipes, registry, externalContracts));
  require(test,
          output.payloads.empty() && output.activityPoints.empty() &&
              output.externalImplementationBindings.empty(),
          "bit reinterpret provider emitted external implementation state");
  bool unresolved = false;
  skeleton.module->walk(
      [&](circt::hw::HWModuleGeneratedOp) { unresolved = true; });
  require(test, !unresolved,
          "bit reinterpret provider left an unresolved operation leaf");
  if (llvm::Error error = verifySpecializedCirctModule(*skeleton.module))
    fail(test, llvm::toString(std::move(error)));
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void generatedRegistryAndProviderCoverage() {
  const llvm::StringRef test = __func__;
  const auto &descriptor = bitReinterpretDescriptor();
  require(
      test,
      descriptor.familyId ==
              ::fabric::ImplementationFamilyId::ScalarBitReinterpret &&
          descriptor.capabilityParamsSchema ==
              ::fabric::CapabilityParamsSchemaId::ScalarBitReinterpretParams &&
          descriptor.typedAdmissionProvider ==
              ::fabric::TypedAdmissionProviderId::
                  ScalarBitReinterpretAdmission &&
          !descriptor.admittedSchemas.empty(),
      "generated bit reinterpret family descriptor changed");
  require(test,
          llvm::all_of(descriptor.admittedSchemas,
                       [&](auto schema) {
                         return ::fabric::admitsOperationSchema(
                             descriptor.familyId, schema);
                       }),
          "generated family admission relation is inconsistent");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarBitReinterpretProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
    return candidate.implementationFamily == descriptor.familyId;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "bit reinterpret provider registration changed its exact coverage");
}

void fabricAdmissionOwnsSemanticLegality() {
  const llvm::StringRef test = __func__;
  const ::fabric::ScalarBitReinterpretParams parameters =
      fullEndpointParams(test);
  std::vector<::fabric::IntegerWidth> widths;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (parameters.integerWidths.contains(width))
      widths.push_back(width);
  require(test, widths.size() >= 2,
          "canonical bit reinterpret integer domain has fewer than two widths");
  mlir::Type source = mlir::IntegerType::get(
      &fabricContext(), ::fabric::getBitWidth(widths.front()));
  mlir::Type destination = mlir::IntegerType::get(
      &fabricContext(), ::fabric::getBitWidth(widths.back()));
  ::fabric::FamilyCapabilityParams capability = parameters;
  llvm::Error error = ::fabric::verifyImplementationFamilyAdmission(
      ::fabric::ImplementationFamilyId::ScalarBitReinterpret, &capability,
      endpointActor(test, source, destination));
  require(test, static_cast<bool>(error),
          "Fabric admission accepted unequal semantic widths");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains("equal semantic width"),
          message);
}

void checkCapability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  const auto &descriptor = bitReinterpretDescriptor();
  require(test, capability->implementationFamily == descriptor.familyId,
          "capability changed implementation family");
  require(test,
          capability->enabledOperationSchemas.size() ==
                  descriptor.admittedSchemas.size() &&
              llvm::all_of(capability->enabledOperationSchemas,
                           [&](auto schema) {
                             return llvm::is_contained(
                                 descriptor.admittedSchemas, schema);
                           }),
          "capability escaped its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarBitReinterpretParams>(
      &capability->parameterizedCapability);
  require(test, parameters != nullptr,
          "capability changed its typed parameter schema");
  const auto expected = fullEndpointParams(test);
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    require(test,
            parameters->integerWidths.contains(width) ==
                expected.integerWidths.contains(width),
            "capability changed its admitted integer endpoint domain");
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
    require(test,
            parameters->floatFormats.contains(format) ==
                expected.floatFormats.contains(format),
            "capability changed its admitted floating endpoint domain");
  require(test, capability->configurationFieldSchema.empty(),
          "bit reinterpret capability gained a configuration field");
  require(
      test,
      !take(test, ::fabric::requiresSemanticConfigurationField(
                      descriptor.familyId, capability->parameterizedCapability,
                      capability->enabledOperationSchemas, 1, 1)),
      "canonical Fabric query requires a bit reinterpret selector");

  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : capability->physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  require(test,
          inputs.size() == 1 && outputs.size() == 1 &&
              inputs.front()->reference.ordinal == 0 &&
              outputs.front()->reference.ordinal == 0 &&
              inputs.front()->payloadWidthBits == fixture.inputWidth &&
              outputs.front()->payloadWidthBits == fixture.outputWidth,
          "capability changed its unary physical port shape");
  const std::vector<std::uint8_t> actual =
      take(test, ::fabric::encodeResourceContractRecord(
                     capability->resourceStateAndTimingContract));
  const std::vector<std::uint8_t> expectedContract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  require(test, actual == expectedContract,
          "capability changed its one-cycle elastic contract");
}

struct EmittedRtl final {
  std::string truncate;
  std::string extend;
};

std::string deterministicRtl(llvm::StringRef test, const ArtifactStore &store,
                             unsigned inputWidth, unsigned outputWidth,
                             llvm::StringRef symbol) {
  FabricFixture fabric =
      makeBitReinterpretFabric(test, store, inputWidth, outputWidth);
  checkCapability(test, fabric);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), symbol);
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(
      test,
      ports.size() == 2 && ports.atInput(0).getName() == "data_input_0" &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              inputWidth &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              outputWidth,
      "derived bit reinterpret leaf ports are not canonical");
  for (const auto &port : ports)
    require(test, !port.getName().starts_with("config_"),
            "bit reinterpret leaf retained a configuration port");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), symbol);
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical bit reinterpret inputs produced different SystemVerilog");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.count("assign data_output_0") == 1 && !rtl.contains("config_") &&
              !rtl.contains("always") && !rtl.contains("case"),
          "bit reinterpret RTL is not one configuration-free wire");
  return firstRtl;
}

EmittedRtl fullDomainUsesOnePhysicalResize(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  return {
      deterministicRtl(test, store, 80, 72, "scalar_bit_reinterpret_truncate"),
      deterministicRtl(test, store, 72, 80, "scalar_bit_reinterpret_extend")};
}

struct TestPattern final {
  std::string label;
  llvm::APInt bits;
};

std::vector<TestPattern> representativePatterns(llvm::StringRef test) {
  std::vector<TestPattern> patterns;
  const auto parameters = fullEndpointParams(test);
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    if (!parameters.floatFormats.contains(format))
      continue;
    CanonicalFloatEndpoint endpoint = canonicalFloatEndpoint(test, format);
    const unsigned width = ::fabric::getBitWidth(format);
    llvm::APInt negativeZero(width, 0);
    negativeZero.setBit(width - 1);
    patterns.push_back(
        {endpoint.spelling + " NaN payload", llvm::APInt::getAllOnes(width)});
    patterns.push_back(
        {endpoint.spelling + " negative zero", std::move(negativeZero)});
  }
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain) {
    if (!parameters.integerWidths.contains(width))
      continue;
    const unsigned bitWidth = ::fabric::getBitWidth(width);
    llvm::APInt bits(bitWidth, 0);
    for (unsigned bit = 0; bit < bitWidth; ++bit)
      if (((0xa5U >> (bit % 8)) & 1U) != 0)
        bits.setBit(bit);
    patterns.push_back(
        {"i" + std::to_string(bitWidth) + " integer pattern", std::move(bits)});
  }
  require(test, !patterns.empty(), "no representative bit patterns derived");
  return patterns;
}

std::string svLiteral(const llvm::APInt &value) {
  llvm::SmallString<32> digits;
  value.toString(digits, 16, false, false);
  return std::to_string(value.getBitWidth()) + "'h" + digits.str().str();
}

void writeToolInputs(const std::filesystem::path &root,
                     const EmittedRtl &emitted) {
  std::ofstream(root / "scalar_bit_reinterpret_truncate.sv")
      << emitted.truncate;
  std::ofstream(root / "scalar_bit_reinterpret_extend.sv") << emitted.extend;

  std::ofstream testbench(root / "testbench.sv");
  testbench << R"sv(module testbench;
  logic [79:0] truncate_input;
  logic [71:0] truncate_output;
  logic [71:0] extend_input;
  logic [79:0] extend_output;

  scalar_bit_reinterpret_truncate truncate_dut(
    .data_input_0(truncate_input), .data_output_0(truncate_output));
  scalar_bit_reinterpret_extend extend_dut(
    .data_input_0(extend_input), .data_output_0(extend_output));

  initial begin
)sv";
  for (const TestPattern &pattern : representativePatterns(__func__)) {
    llvm::APInt physical(80, 0);
    physical.insertBits(pattern.bits, 0);
    physical.insertBits(llvm::APInt(16, 0xa55a), 64);
    const llvm::APInt expected = physical.trunc(72);
    testbench << "    truncate_input = " << svLiteral(physical) << ";\n"
              << "    #1;\n"
              << "    if (truncate_output !== " << svLiteral(expected)
              << ") $fatal(1, \"" << pattern.label
              << " was not preserved\");\n";
  }
  const llvm::APInt extendInput = llvm::APInt::getAllOnes(72);
  const llvm::APInt extendExpected = extendInput.zext(80);
  testbench << "    extend_input = " << svLiteral(extendInput) << ";\n"
            << "    #1;\n"
            << "    if (extend_output !== " << svLiteral(extendExpected)
            << ") $fatal(1, \"physical zero extension was not low aligned\");\n"
            << R"sv(    $finish;
  end
endmodule
)sv";

  std::ofstream(root / "synthesis_top.sv") << R"sv(
module scalar_bit_reinterpret_synthesis_top(
  input logic [79:0] truncate_input,
  input logic [71:0] extend_input,
  output logic [71:0] truncate_output,
  output logic [79:0] extend_output);
  scalar_bit_reinterpret_truncate truncate_dut(
    .data_input_0(truncate_input), .data_output_0(truncate_output));
  scalar_bit_reinterpret_extend extend_dut(
    .data_input_0(extend_input), .data_output_0(extend_output));
endmodule
)sv";
  std::ofstream(root / "portable_scalar_bit_reinterpret.ys") << R"ys(
read_verilog -sv scalar_bit_reinterpret_truncate.sv scalar_bit_reinterpret_extend.sv synthesis_top.sv
hierarchy -check -top scalar_bit_reinterpret_synthesis_top
proc
opt
check -assert
synth -top scalar_bit_reinterpret_synthesis_top
check -assert
stat
)ys";
}

void invalidInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarBitReinterpretProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;

  FabricFixture valid = makeBitReinterpretFabric(test, store, 80, 72);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  const std::vector<FabricOperationRecipeBinding> validRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  for (LeafMutation mutation :
       {LeafMutation::WrongInputWidth, LeafMutation::ExtraConfigurationPort}) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, valid, validAbi.abi(),
                     "malformed_bit_reinterpret", mutation);
    const std::string before = moduleText(*skeleton.module);
    const std::vector<FabricOperationLeafAssociation> associations = {
        {skeleton.leaf, valid.occurrence}};
    expectError(test,
                specializeFabricOperationLeaves(
                    *skeleton.module, valid.fabric, validAbi, associations,
                    validRecipes, registry, externalContracts),
                "leaf port");
    require(test, moduleText(*skeleton.module) == before,
            "malformed leaf partially mutated the caller module");
  }

  FabricFixture wrongContract =
      makeBitReinterpretFabric(test, store, 80, 72, ContractKind::Wrong);
  FinalizedConfigurationABI wrongAbi =
      makeConfigurationAbi(test, store, wrongContract);
  std::unique_ptr<mlir::MLIRContext> wrongContext = makeCirctContext();
  SkeletonFixture wrongSkeleton =
      makeSkeleton(test, *wrongContext, wrongContract, wrongAbi.abi(),
                   "wrong_contract_bit_reinterpret");
  const std::string wrongBefore = moduleText(*wrongSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> wrongAssociations = {
      {wrongSkeleton.leaf, wrongContract.occurrence}};
  const std::vector<FabricOperationRecipeBinding> wrongRecipes = {
      {wrongContract.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(test,
                         specializeFabricOperationLeaves(
                             *wrongSkeleton.module, wrongContract.fabric,
                             wrongAbi, wrongAssociations, wrongRecipes,
                             registry, externalContracts),
                         ::fabric::ImplementationFamilyId::ScalarBitReinterpret,
                         "unsupported resource contract");
  require(test, moduleText(*wrongSkeleton.module) == wrongBefore,
          "wrong resource contract partially mutated the caller module");

  FabricFixture zeroWidth = makeBitReinterpretFabric(test, store, 0, 0);
  FinalizedConfigurationABI zeroWidthAbi =
      makeConfigurationAbi(test, store, zeroWidth);
  std::unique_ptr<mlir::MLIRContext> zeroWidthContext = makeCirctContext();
  SkeletonFixture zeroWidthSkeleton =
      makeSkeleton(test, *zeroWidthContext, zeroWidth, zeroWidthAbi.abi(),
                   "unsupported_zero_width_bit_reinterpret");
  const std::string zeroWidthBefore = moduleText(*zeroWidthSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> zeroWidthAssociations = {
      {zeroWidthSkeleton.leaf, zeroWidth.occurrence}};
  const std::vector<FabricOperationRecipeBinding> zeroWidthRecipes = {
      {zeroWidth.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(test,
                         specializeFabricOperationLeaves(
                             *zeroWidthSkeleton.module, zeroWidth.fabric,
                             zeroWidthAbi, zeroWidthAssociations,
                             zeroWidthRecipes, registry, externalContracts),
                         ::fabric::ImplementationFamilyId::ScalarBitReinterpret,
                         "unsupported zero-width physical shape");
  require(test, moduleText(*zeroWidthSkeleton.module) == zeroWidthBefore,
          "unsupported physical shape partially mutated the caller module");

  FabricFixture other = makeOtherFamilyFabric(test, store);
  FinalizedConfigurationABI otherAbi = makeConfigurationAbi(test, store, other);
  std::unique_ptr<mlir::MLIRContext> otherContext = makeCirctContext();
  SkeletonFixture otherSkeleton =
      makeSkeleton(test, *otherContext, other, otherAbi.abi(), "other_family");
  const std::string otherBefore = moduleText(*otherSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> otherAssociations = {
      {otherSkeleton.leaf, other.occurrence}};
  const std::vector<FabricOperationRecipeBinding> otherRecipes = {
      {other.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(
      test,
      specializeFabricOperationLeaves(*otherSkeleton.module, other.fabric,
                                      otherAbi, otherAssociations, otherRecipes,
                                      registry, externalContracts),
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      "other-family input");
  require(test, moduleText(*otherSkeleton.module) == otherBefore,
          "other-family input partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  generatedRegistryAndProviderCoverage();
  fabricAdmissionOwnsSemanticLegality();
  const EmittedRtl emitted = fullDomainUsesOnePhysicalResize(root);
  writeToolInputs(root, emitted);
  invalidInputsAreTransactional(root / "invalid");
  return 0;
}
