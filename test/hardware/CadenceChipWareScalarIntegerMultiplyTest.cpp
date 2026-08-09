#include "ConfigurationABI2TestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/Native/ChipWare.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::BlobStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FabricPhysicalOccurrenceOwnerRef;
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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  require(test, static_cast<bool>(error), "expected operation to fail");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected),
          "error did not identify the rejected contract closure");
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
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

enum class FixtureKind { Supported, MultipleWidths, UnsupportedContract };

struct FabricFixture final {
  FinalizedFabricRoot module;
  FabricFuOccurrenceNodeRef localOccurrence;
  FinalizedFabricRoot system;
  std::vector<FabricPhysicalOccurrenceOwnerRef> physicalOccurrences;
};

std::string fabricSource(unsigned width, FixtureKind kind,
                         std::uint64_t operationCount) {
  const std::string bits = std::to_string(width);
  const std::string widths = kind == FixtureKind::MultipleWidths
                                 ? "8 : i32, 16 : i32"
                                 : bits + " : i32";
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @chipware_multiply_" << bits
         << "(%a: !fabric.bits<" << bits << ">, %b: !fabric.bits<" << bits
         << ">) -> !fabric.bits<" << bits
         << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<" << bits
         << ">, %pb = %b : !fabric.bits<" << bits << ">) -> !fabric.bits<"
         << bits << "> { ";
  for (std::uint64_t index = 0; index < operationCount; ++index) {
    stream << "%fu" << index << " = fabric.fu (%fa = %pa : !fabric.bits<"
           << bits << ">, %fb = %pb : !fabric.bits<" << bits
           << ">) -> !fabric.bits<" << bits << "> { %value" << index
           << " = fabric.op [@arith.muli] (%fa, %fb) "
              "{implementation_family = "
              "#fabric.implementation_family<ScalarIntegerMultiply>, "
              "hw_params = {integer_widths = ["
           << widths << "]}} : (!fabric.bits<" << bits << ">, !fabric.bits<"
           << bits << ">) -> !fabric.bits<" << bits << "> fabric.yield %value"
           << index << " : !fabric.bits<" << bits << "> } ";
  }
  stream << "} fabric.yield %pe : !fabric.bits<" << bits << "> } }";
  return source;
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         unsigned width, FixtureKind kind,
                         std::uint64_t spatialCoreCount = 1,
                         std::uint64_t operationCount = 1) {
  require(test, operationCount != 0, "Fabric fixture has no operations");
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(width, kind, operationCount), &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const ::fabric::ResourceContract &resourceContract =
      kind == FixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(resourceContract));
  const std::vector<std::int8_t> signedBytes(encoded.begin(), encoded.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedBytes));
  });

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no module root");
  FinalizedFabricRoot module =
      take(test, loom::fabric::finalizeFabricRoot(root, store));

  std::optional<FabricFuOccurrenceNodeRef> localOccurrence;
  for (const auto fuOccurrence : module.view().fuOccurrences()) {
    const auto definition = module.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         module.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarIntegerMultiply)
        continue;
      localOccurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         module.view(), capability.occurrence, fuOccurrence));
      break;
    }
    if (localOccurrence)
      break;
  }
  require(test, localOccurrence.has_value(),
          "Fabric fixture has no scalar multiply occurrence");

  FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSpatialCoreSystem(module, store,
                                                             spatialCoreCount));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  auto operations = take(test, enumerateFabricPhysicalOperations(systemView));
  std::vector<FabricPhysicalOccurrenceOwnerRef> physicalOccurrences;
  for (const auto &operation : operations)
    if (operation.capability &&
        operation.capability->implementationFamily ==
            ::fabric::ImplementationFamilyId::ScalarIntegerMultiply)
      physicalOccurrences.push_back(operation.physicalOccurrence);
  llvm::sort(physicalOccurrences, [](const auto &lhs, const auto &rhs) {
    return loom::fabric::canonicalFabricBytes(lhs) <
           loom::fabric::canonicalFabricBytes(rhs);
  });
  require(test, physicalOccurrences.size() == spatialCoreCount * operationCount,
          "System has the wrong scalar multiply occurrence count");
  return FabricFixture{std::move(module), *localOccurrence, std::move(system),
                       std::move(physicalOccurrences)};
}

FinalizedConfigurationABI makeAbi(llvm::StringRef test,
                                  const ArtifactStore &store,
                                  const FabricFixture &fixture) {
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

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<FabricOperationLeafAssociation> associations;
  std::vector<std::string> symbols;
};

SkeletonFixture
makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
             const FabricFixture &fixture, const ConfigurationABI &abi,
             std::optional<std::size_t> onlyOccurrence = std::nullopt) {
  require(test,
          !onlyOccurrence ||
              *onlyOccurrence < fixture.physicalOccurrences.size(),
          "selected occurrence is outside the fixture");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));

  std::vector<FabricOperationLeafAssociation> associations;
  std::vector<std::string> symbols;
  const std::size_t begin = onlyOccurrence.value_or(0);
  const std::size_t end =
      onlyOccurrence ? begin + 1 : fixture.physicalOccurrences.size();
  const std::size_t selectedCount = end - begin;
  for (std::size_t occurrenceIndex = begin; occurrenceIndex < end;
       ++occurrenceIndex) {
    const std::size_t index = occurrenceIndex - begin;
    const auto &occurrence = fixture.physicalOccurrences[occurrenceIndex];
    const ResolvedFabricPhysicalOperation operation = take(
        test, resolveFabricPhysicalOperation(abi.fabricSystem(), occurrence));
    std::vector<circt::hw::PortInfo> ports =
        take(test, deriveFabricOperationLeafPorts(builder, occurrence,
                                                  *operation.capability, abi));
    std::string symbol =
        selectedCount == 1
            ? "chipware_scalar_integer_multiply"
            : "chipware_scalar_integer_multiply_" + std::to_string(index);
    circt::hw::HWModuleGeneratedOp leaf =
        circt::hw::HWModuleGeneratedOp::create(
            builder, location,
            mlir::FlatSymbolRefAttr::get(&context,
                                         fabricOperationGeneratorSchemaSymbol),
            builder.getStringAttr(symbol), ports);
    associations.push_back({leaf, occurrence});
    symbols.push_back(std::move(symbol));
  }
  return SkeletonFixture{std::move(module), std::move(associations),
                         std::move(symbols)};
}

ExternalInputBinding
modelInput(llvm::StringRef resourceKey = cadenceChipWareCwMultResourceKey) {
  return {cadenceChipWareComponentModelSlotRef.str(),
          ToolBundledResourceDependency{"example.cadence-ddi:synthetic",
                                        resourceKey.str()}};
}

void registerNative(llvm::StringRef test,
                    FabricOperationProviderRegistry &providers,
                    ExternalImplementationContractCatalog &contracts) {
  if (llvm::Error error =
          registerCadenceChipWareExternalImplementationContract(contracts))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerCadenceChipWareScalarIntegerMultiplyProvider(providers))
    fail(test, llvm::toString(std::move(error)));
}

struct SpecializationResult final {
  FabricOperationProviderOutput output;
  std::string systemVerilog;
  std::string top;
};

SpecializationResult
runSpecialization(llvm::StringRef test, SkeletonFixture skeleton,
                  const FinalizedConfigurationABI &abi,
                  llvm::ArrayRef<BackendRecipeKey> recipeKeys,
                  const FabricOperationProviderRegistry &providers,
                  const ExternalImplementationContractCatalog &contracts) {
  require(test, recipeKeys.size() == skeleton.associations.size(),
          "recipe count does not match occurrence count");
  std::vector<FabricOperationRecipeBinding> recipes;
  recipes.reserve(recipeKeys.size());
  for (const auto &[index, recipe] : llvm::enumerate(recipeKeys)) {
    std::vector<ExternalInputBinding> inputs;
    if (recipe == BackendRecipeKey::CadenceChipWare)
      inputs.push_back(modelInput());
    recipes.push_back(
        {skeleton.associations[index].occurrence, recipe, std::move(inputs)});
  }
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(*skeleton.module, abi,
                                                 skeleton.associations, recipes,
                                                 providers, contracts));
  const std::string top = skeleton.symbols.front();
  std::string systemVerilog =
      take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
  return {std::move(output), std::move(systemVerilog), top};
}

void requireExactCapability(llvm::StringRef test,
                            const FabricFixture &fixture) {
  const auto *capability =
      fixture.module.view().resolvedFabricOpCapability(fixture.localOccurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  require(test,
          capability->implementationFamily ==
                  ::fabric::ImplementationFamilyId::ScalarIntegerMultiply &&
              capability->enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::ArithMulI} &&
              capability->configurationFieldSchema.empty(),
          "ChipWare recipe changed the generated capability identity");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &capability->parameterizedCapability);
  require(test,
          parameters && parameters->integerWidths.size() == 1 &&
              parameters->integerWidths.contains(::fabric::IntegerWidth::I8) &&
              parameters->pointerFormats.empty(),
          "ChipWare recipe capability is outside the fixed 8-bit subset");

  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : capability->physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  require(test,
          inputs.size() == 2 && outputs.size() == 1 &&
              inputs[0]->reference.ordinal == 0 &&
              inputs[1]->reference.ordinal == 1 &&
              outputs[0]->reference.ordinal == 0 &&
              inputs[0]->payloadWidthBits == 8 &&
              inputs[1]->payloadWidthBits == 8 &&
              outputs[0]->payloadWidthBits == 8,
          "ChipWare recipe capability has the wrong physical ports");
  const auto actual =
      take(test, ::fabric::encodeResourceContractRecord(
                     capability->resourceStateAndTimingContract));
  const auto expected =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  require(test, actual == expected,
          "ChipWare recipe capability has the wrong progress contract");
}

void registrationIsExplicit() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry providers;
  ExternalImplementationContractCatalog contracts;
  registerNative(test, providers, contracts);
  const auto coverage = providers.coverage();
  const auto nativeCoverage =
      std::find_if(coverage.begin(), coverage.end(), [](const auto &entry) {
        return entry.implementationFamily ==
               ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
      });
  require(test,
          nativeCoverage != coverage.end() &&
              nativeCoverage->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::CadenceChipWare} &&
              std::count_if(coverage.begin(), coverage.end(),
                            [](const auto &entry) {
                              return !entry.recipes.empty();
                            }) == 1,
          "ChipWare provider owns unexpected recipe coverage");
  const auto contract = contracts.find(cadenceChipWareExternalContractRef);
  require(test,
          contract && contract->inputSlots.size() == 1 &&
              contract->inputSlots.front().providerInputSlotRef ==
                  cadenceChipWareComponentModelSlotRef &&
              contract->inputSlots.front().acceptedDependencyKinds ==
                  std::vector<ExternalDependencyKind>{
                      ExternalDependencyKind::ToolBundledResource} &&
              contract->supportedRepresentations ==
                  std::vector<RepresentationRootVariant>{
                      RepresentationRootVariant::Rtl} &&
              contract->blackBoxContractRequired &&
              !contract->memoryMacroCapable && contract->validator,
          "ChipWare external contract is not exact");
}

void writeText(llvm::StringRef test, const std::filesystem::path &path,
               llvm::StringRef contents) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  require(test, !error, "could not create tool artifact directory");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  output.close();
  require(test, static_cast<bool>(output), "could not write tool RTL");
}

ImplementationRepresentationRoot
makeRtlRepresentation(llvm::StringRef test, const std::filesystem::path &root,
                      const SpecializationResult &result) {
  const std::filesystem::path blobRoot = root / "blobs";
  std::filesystem::create_directories(blobRoot);
  BlobStore blobs(blobRoot.string());
  const std::vector<std::uint8_t> rtlBytes(result.systemVerilog.begin(),
                                           result.systemVerilog.end());
  std::vector<ImplementationPayload> payloads{
      {PayloadRole::RtlSource, "rtl/chipware_multiply.sv",
       take(test, blobs.put(rtlBytes))}};
  for (const auto &payload : result.output.payloads) {
    const ImplementationPayload descriptor = payload.descriptor();
    require(test, take(test, blobs.put(payload.bytes)) == descriptor.blobDigest,
            "BlackBox payload digest changed during publication");
    payloads.push_back(descriptor);
  }
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  return take(test, createImplementationRepresentationRoot(
                        RepresentationRootVariant::Rtl, std::nullopt, format,
                        {RepresentationObjectKind::Module, result.top},
                        std::move(payloads)));
}

void finalizeImplementation(
    llvm::StringRef test, const std::filesystem::path &root,
    const FabricFixture &fixture, const FinalizedConfigurationABI &abi,
    const ExternalImplementationContractCatalog &contracts,
    const SpecializationResult &result) {
  const std::filesystem::path blobRoot = root / "blobs";
  BlobStore blobs(blobRoot.string());
  HardwareImplementationDraft draft{
      fixture.system.reference(),
      abi.reference(),
      {},
      makeRtlRepresentation(test, root, result),
      std::nullopt,
      {},
      result.output.activityPoints,
      {},
      result.output.externalImplementationBindings,
  };
  const FinalizedHardwareImplementation implementation =
      take(test,
           finalizeHardwareImplementation(std::move(draft), contracts,
                                          ArtifactStore(root.string()), blobs));
  require(
      test,
      implementation.implementation().externalImplementationBindings().size() ==
          1,
      "ChipWare external binding did not finalize");
}

void exactExternalContractRejectsDifferentClosure(
    const std::filesystem::path &root, const SpecializationResult &result,
    const ExternalImplementationContractCatalog &contracts) {
  const llvm::StringRef test = __func__;
  const auto contract = contracts.find(cadenceChipWareExternalContractRef);
  require(test, contract && contract->validator,
          "ChipWare external contract has no exact binding validator");
  if (!contract || !contract->validator)
    return;

  const ImplementationRepresentationRoot representation =
      makeRtlRepresentation(test, root, result);
  const ExternalImplementationBindingDraft exact =
      result.output.externalImplementationBindings.front();
  if (llvm::Error error = contract->validator(exact, representation, nullptr))
    fail(test, llvm::toString(std::move(error)));

  std::vector<ExternalImplementationBindingDraft> malformed;
  malformed.push_back(exact);
  std::get<ToolBundledResourceDependency>(
      malformed.back().externalInputs.front().dependencyIdentity)
      .resourceKey = "chipware:CW_addsub";
  malformed.push_back(exact);
  malformed.back().representationLocators.front().canonicalName = "CW_addsub";
  malformed.push_back(exact);
  malformed.back().blackBoxContractPayload->canonicalLogicalName =
      "blackbox/wrong.txt";
  malformed.push_back(exact);
  malformed.back().fabricResourceRefs.clear();
  for (const ExternalImplementationBindingDraft &binding : malformed)
    expectError(test, contract->validator(binding, representation, nullptr),
                "verified CW_mult closure");
}

void nativeOccurrenceIsDeterministic(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fixture = makeFabric(test, store, 8, FixtureKind::Supported);
  requireExactCapability(test, fixture);
  FinalizedConfigurationABI abi = makeAbi(test, store, fixture);
  require(test,
          abi.reference().schemaIdentity == configurationAbiSchema.identity &&
              abi.reference().schemaVersion == configurationAbiSchema.version,
          "ChipWare specialization did not use ConfigurationABI 2.0");

  FabricOperationProviderRegistry providers;
  ExternalImplementationContractCatalog contracts;
  registerNative(test, providers, contracts);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SpecializationResult first = runSpecialization(
      test, makeSkeleton(test, *firstContext, fixture, abi.abi()), abi,
      {BackendRecipeKey::CadenceChipWare}, providers, contracts);
  SpecializationResult second = runSpecialization(
      test, makeSkeleton(test, *secondContext, fixture, abi.abi()), abi,
      {BackendRecipeKey::CadenceChipWare}, providers, contracts);
  require(test, first.systemVerilog == second.systemVerilog,
          "ChipWare wrapper emission is nondeterministic");
  require(test,
          llvm::StringRef(first.systemVerilog)
                  .contains("module chipware_scalar_integer_multiply") &&
              llvm::StringRef(first.systemVerilog)
                  .contains("wire [15:0] chipware_product;") &&
              llvm::StringRef(first.systemVerilog).contains("CW_mult #(") &&
              llvm::StringRef(first.systemVerilog).contains(".wA(8)") &&
              llvm::StringRef(first.systemVerilog).contains(".wB(8)") &&
              llvm::StringRef(first.systemVerilog).contains(".TC(1'b0)") &&
              llvm::StringRef(first.systemVerilog)
                  .contains("chipware_product[7:0]") &&
              !llvm::StringRef(first.systemVerilog)
                   .contains("data_input_0 * data_input_1"),
          "ChipWare wrapper does not expose the verified component contract");
  require(test,
          first.output.payloads.size() == 1 &&
              first.output.activityPoints.empty() &&
              first.output.externalImplementationBindings.size() == 1,
          "ChipWare provider emitted unexpected implementation state");
  const auto &payload = first.output.payloads.front();
  require(
      test,
      payload.role == PayloadRole::BlackBoxContract &&
          payload.canonicalLogicalName ==
              "blackbox/cadence-chipware-cw-mult-i8.txt" &&
          llvm::StringRef(reinterpret_cast<const char *>(payload.bytes.data()),
                          payload.bytes.size())
              .contains("component=CW_mult"),
      "ChipWare provider omitted its authored BlackBox contract");
  const auto &binding = first.output.externalImplementationBindings.front();
  require(test,
          binding.providerContractRef == cadenceChipWareExternalContractRef &&
              binding.externalInputs ==
                  std::vector<ExternalInputBinding>{modelInput()} &&
              binding.fabricResourceRefs ==
                  std::vector<FabricPhysicalOccurrenceOwnerRef>{
                      fixture.physicalOccurrences.front()} &&
              binding.representationLocators ==
                  std::vector<RepresentationLocator>{
                      {RepresentationObjectKind::Module,
                       cadenceChipWareCwMultModuleName.str()}} &&
              binding.blackBoxContractPayload ==
                  ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                           payload.canonicalLogicalName},
          "ChipWare provider did not preserve its exact occurrence binding");
  requireExactCapability(test, fixture);
  exactExternalContractRejectsDifferentClosure(root / "contract", first,
                                               contracts);
  finalizeImplementation(test, root, fixture, abi, contracts, first);
  writeText(test,
            root / "provider_artifacts" /
                "cadence_chipware_scalar_integer_multiply.sv",
            first.systemVerilog);
}

void portableAndNativeAreOccurrenceScoped(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fixture = makeFabric(test, store, 8, FixtureKind::Supported, 2);
  FinalizedConfigurationABI abi = makeAbi(test, store, fixture);
  FabricOperationProviderRegistry providers;
  ExternalImplementationContractCatalog contracts;
  registerNative(test, providers, contracts);
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(providers))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = providers.coverage();
  const auto multiplyCoverage =
      std::find_if(coverage.begin(), coverage.end(), [](const auto &entry) {
        return entry.implementationFamily ==
               ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
      });
  require(test,
          multiplyCoverage != coverage.end() &&
              multiplyCoverage->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog,
                      BackendRecipeKey::CadenceChipWare},
          "portable and ChipWare recipes are not independent coverage");

  std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
  SpecializationResult native = runSpecialization(
      test, makeSkeleton(test, *nativeContext, fixture, abi.abi(), 0), abi,
      {BackendRecipeKey::CadenceChipWare}, providers, contracts);
  std::unique_ptr<mlir::MLIRContext> portableContext = makeCirctContext();
  SpecializationResult portable = runSpecialization(
      test, makeSkeleton(test, *portableContext, fixture, abi.abi(), 1), abi,
      {BackendRecipeKey::PortableSystemVerilog}, providers, contracts);
  require(test,
          llvm::StringRef(native.systemVerilog).count("CW_mult #(") == 1 &&
              !llvm::StringRef(native.systemVerilog)
                   .contains("data_input_0 * data_input_1") &&
              !llvm::StringRef(portable.systemVerilog).contains("CW_mult #(") &&
              llvm::StringRef(portable.systemVerilog)
                      .count("data_input_0 * data_input_1") == 1 &&
              native.output.externalImplementationBindings.size() == 1 &&
              native.output.externalImplementationBindings.front()
                      .fabricResourceRefs ==
                  std::vector<FabricPhysicalOccurrenceOwnerRef>{
                      fixture.physicalOccurrences.front()} &&
              portable.output.externalImplementationBindings.empty(),
          "recipe selection escaped its physical occurrence");
}

void expectUnsupported(llvm::StringRef test,
                       llvm::Expected<FabricOperationProviderOutput> result,
                       mlir::ModuleOp module, llvm::StringRef before) {
  require(test, !result, "ChipWare provider accepted an unsupported subset");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::ScalarIntegerMultiply &&
            error.recipe() == BackendRecipeKey::CadenceChipWare;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported subset returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "unsupported subset lost its typed classification");
  require(test, moduleText(module) == before,
          "unsupported ChipWare specialization mutated the caller module");
}

void typedUnsupportedAndRollback(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry providers;
  ExternalImplementationContractCatalog contracts;
  registerNative(test, providers, contracts);

  for (const auto &[width, kind] :
       std::vector<std::pair<unsigned, FixtureKind>>{
           {16, FixtureKind::Supported},
           {8, FixtureKind::MultipleWidths},
           {8, FixtureKind::UnsupportedContract}}) {
    FabricFixture fixture = makeFabric(test, store, width, kind);
    FinalizedConfigurationABI abi = makeAbi(test, store, fixture);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fixture, abi.abi());
    const std::string before = moduleText(*skeleton.module);
    const std::vector<FabricOperationRecipeBinding> recipes = {
        {fixture.physicalOccurrences.front(),
         BackendRecipeKey::CadenceChipWare,
         {modelInput()}},
    };
    auto result = specializeFabricOperationLeaves(
        *skeleton.module, abi, skeleton.associations, recipes, providers,
        contracts);
    expectUnsupported(test, std::move(result), *skeleton.module, before);
  }

  FabricFixture supported = makeFabric(test, store, 8, FixtureKind::Supported);
  FinalizedConfigurationABI supportedAbi = makeAbi(test, store, supported);
  std::unique_ptr<mlir::MLIRContext> resourceContext = makeCirctContext();
  SkeletonFixture resourceSkeleton =
      makeSkeleton(test, *resourceContext, supported, supportedAbi.abi());
  const std::string resourceBefore = moduleText(*resourceSkeleton.module);
  const std::vector<FabricOperationRecipeBinding> wrongResource = {
      {supported.physicalOccurrences.front(),
       BackendRecipeKey::CadenceChipWare,
       {modelInput("chipware:CW_addsub")}},
  };
  expectUnsupported(
      test,
      specializeFabricOperationLeaves(*resourceSkeleton.module, supportedAbi,
                                      resourceSkeleton.associations,
                                      wrongResource, providers, contracts),
      *resourceSkeleton.module, resourceBefore);

  std::unique_ptr<mlir::MLIRContext> missingContext = makeCirctContext();
  SkeletonFixture missingSkeleton =
      makeSkeleton(test, *missingContext, supported, supportedAbi.abi());
  const std::string missingBefore = moduleText(*missingSkeleton.module);
  const std::vector<FabricOperationRecipeBinding> missingInput = {
      {supported.physicalOccurrences.front(),
       BackendRecipeKey::CadenceChipWare,
       {}},
  };
  auto missing = specializeFabricOperationLeaves(
      *missingSkeleton.module, supportedAbi, missingSkeleton.associations,
      missingInput, providers, contracts);
  require(test, !missing, "ChipWare provider accepted a missing model input");
  const std::string missingMessage = llvm::toString(missing.takeError());
  require(test,
          llvm::StringRef(missingMessage)
                  .contains("provider input slot closure is incomplete") &&
              moduleText(*missingSkeleton.module) == missingBefore,
          "missing ChipWare input was not rejected transactionally");

  FabricFixture twoOccurrences =
      makeFabric(test, store, 8, FixtureKind::Supported, 1, 2);
  FinalizedConfigurationABI twoAbi = makeAbi(test, store, twoOccurrences);
  std::unique_ptr<mlir::MLIRContext> rollbackContext = makeCirctContext();
  SkeletonFixture rollbackSkeleton =
      makeSkeleton(test, *rollbackContext, twoOccurrences, twoAbi.abi());
  const std::string rollbackBefore = moduleText(*rollbackSkeleton.module);
  const std::vector<FabricOperationRecipeBinding> rollbackRecipes = {
      {twoOccurrences.physicalOccurrences[0],
       BackendRecipeKey::CadenceChipWare,
       {modelInput()}},
      {twoOccurrences.physicalOccurrences[1],
       BackendRecipeKey::CadenceChipWare,
       {modelInput("chipware:CW_addsub")}},
  };
  auto rollback = specializeFabricOperationLeaves(
      *rollbackSkeleton.module, twoAbi, rollbackSkeleton.associations,
      rollbackRecipes, providers, contracts);
  expectUnsupported(test, std::move(rollback), *rollbackSkeleton.module,
                    rollbackBefore);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  registrationIsExplicit();
  nativeOccurrenceIsDeterministic(root / "native");
  portableAndNativeAreOccurrenceScoped(root / "occurrence");
  typedUnsupportedAndRollback(root / "negative");
  return 0;
}
