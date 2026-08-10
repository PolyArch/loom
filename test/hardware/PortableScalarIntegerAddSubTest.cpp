#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
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
    fail(test, "accepted invalid portable add/sub input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

mlir::MLIRContext &fabricContext();

std::vector<std::uint8_t>
operationValue(llvm::StringRef test,
               const loom::fabric::ResolvedFabricOpCapabilityView &capability,
               dataflow::OperationSchemaId schema) {
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "add/sub capability does not have a finite relation");
  const auto point = llvm::find_if(
      relation.finiteBehaviorDomain(), [&](const auto &candidate) {
        return candidate.representativeActor.schema == schema;
      });
  require(test,
          point != relation.finiteBehaviorDomain().end() &&
              point->semanticConfiguration.has_value(),
          "add/sub schema has no finite behavior key");
  return std::vector<std::uint8_t>(
      point->semanticConfiguration->bytes().begin(),
      point->semanticConfiguration->bytes().end());
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

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

enum class ConfigurationAbiKind {
  Complete,
  MissingSubtract,
  ExtraSemanticValue,
  SubtractInactive,
};

enum class FabricFixtureKind {
  ConfiguredAddSub,
  SingletonAdd,
  UnsupportedContract,
};

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           FabricFixtureKind kind = FabricFixtureKind::ConfiguredAddSub) {
  llvm::StringRef sourceText;
  switch (kind) {
  case FabricFixtureKind::SingletonAdd:
    sourceText = R"mlir(
    module {
      fabric.module @integer_add(%a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";
    break;
  case FabricFixtureKind::ConfiguredAddSub:
  case FabricFixtureKind::UnsupportedContract:
    sourceText = R"mlir(
    module {
      fabric.module @integer_add_sub(
          %a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";
    break;
  }
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const ::fabric::ResourceContract &resourceContract =
      kind == FabricFixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(resourceContract));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });

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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      FinalizedFabricRoot system =
          take(test, loom::hardware::test::makeSingleSpatialCoreSystem(fabric,
                                                                       store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical add/sub occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no scalar integer add/sub occurrence");
}

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, capability->configurationFieldSchema.size() == 1,
          "add/sub fixture has an unexpected configuration field count");
  const auto fieldReference = capability->configurationFieldSchema.front();
  const std::vector<std::uint8_t> addValue =
      operationValue(test, *capability, dataflow::OperationSchemaId::ArithAddI);
  const std::vector<std::uint8_t> subtractValue =
      operationValue(test, *capability, dataflow::OperationSchemaId::ArithSubI);

  std::vector<FiniteCodebookEntry> entries{
      {addValue, {0x02}},
      {kind == ConfigurationAbiKind::MissingSubtract
           ? std::vector<std::uint8_t>{0xff}
           : subtractValue,
       {0x01}}};
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x03}});
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{2, std::move(entries)},
      kind == ConfigurationAbiKind::SubtractInactive ? subtractValue
                                                     : addValue};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  return take(
      test, finalizeConfigurationABI(
                makeConfigurationAbiDraft(test, store, fixture, kind), store));
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
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             bool wrongConfigurationWidth = false) {
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
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, *capability, abi));
  if (wrongConfigurationWidth) {
    require(test, ports.size() == 4,
            "multi-member add/sub leaf did not have four ports");
    ports[2].type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("scalar_integer_add_sub"), ports);
  return SkeletonFixture{std::move(module), leaf};
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
  if (llvm::Error error = registerPortableScalarIntegerAddSubProvider(registry))
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
          "portable add/sub provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

std::string emitConfiguredAddSub(llvm::StringRef test,
                                 const ArtifactStore &store) {
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(context.get(), 2) &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived operation leaf ports are not canonical");
  return specialize(test, skeleton, fabric, abi);
}

void configuredCodebookAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const std::string first = emitConfiguredAddSub(test, store);
  const std::string second = emitConfiguredAddSub(test, store);
  require(test, first == second,
          "identical add/sub inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("config_0") && rtl.contains(" + ") &&
              !rtl.contains(" - "),
          "portable provider did not share its configured add/sub datapath");

  const std::string testbench = R"sv(
module testbench;
  logic [7:0] data_input_0;
  logic [7:0] data_input_1;
  logic [1:0] config_0;
  logic [7:0] data_output_0;

  scalar_integer_add_sub dut(.*);

  initial begin
    data_input_0 = 8'hff;
    data_input_1 = 8'h01;
    config_0 = 2'b10;
    #1;
    if (data_output_0 !== 8'h00) $fatal(1, "modular add failed");

    data_input_0 = 8'h00;
    data_input_1 = 8'h01;
    config_0 = 2'b01;
    #1;
    if (data_output_0 !== 8'hff) $fatal(1, "modular subtract failed");

    data_input_0 = 8'h37;
    data_input_1 = 8'h12;
    config_0 = 2'b10;
    #1;
    if (data_output_0 !== 8'h49) $fatal(1, "add codebook entry failed");

    config_0 = 2'b01;
    #1;
    if (data_output_0 !== 8'h25) $fatal(1, "subtract codebook entry failed");
    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog scalar_integer_add_sub.sv
hierarchy -check -top scalar_integer_add_sub
proc
opt
check
select -assert-none t:$sub
select -assert-count 2 t:$add
synth -top scalar_integer_add_sub
check
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_add_sub.sv", first},
           {"testbench.sv", testbench},
           {"portable_scalar_integer_add_sub.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::SingletonAdd);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton add leaf retained a redundant selector");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains(" + ") &&
              !llvm::StringRef(rtl).contains(" - ") &&
              !llvm::StringRef(rtl).contains("config_0"),
          "singleton add provider emitted configurable subtract logic");
}

void subtractInactiveControlsUnassignedCode(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(
      test, store, fabric, ConfigurationAbiKind::SubtractInactive);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("config_0") &&
              llvm::StringRef(rtl).contains(" + ") &&
              !llvm::StringRef(rtl).contains(" - "),
          "subtract-inactive provider did not share its add/sub datapath");

  const std::string testbench = R"sv(
module testbench;
  logic [7:0] data_input_0;
  logic [7:0] data_input_1;
  logic [1:0] config_0;
  logic [7:0] data_output_0;

  scalar_integer_add_sub dut(.*);

  initial begin
    data_input_0 = 8'h37;
    data_input_1 = 8'h12;
    config_0 = 2'b00;
    #1;
    if (data_output_0 !== 8'h25)
      $fatal(1, "unassigned code did not preserve subtract inactive behavior");

    config_0 = 2'b10;
    #1;
    if (data_output_0 !== 8'h49) $fatal(1, "add codebook entry failed");

    config_0 = 2'b01;
    #1;
    if (data_output_0 !== 8'h25) $fatal(1, "subtract codebook entry failed");
    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog scalar_integer_add_sub.sv
hierarchy -check -top scalar_integer_add_sub
proc
opt
check
select -assert-none t:$sub
select -assert-count 2 t:$add
synth -top scalar_integer_add_sub
check
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_add_sub.sv", rtl},
           {"testbench.sv", testbench},
           {"portable_scalar_integer_add_sub.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void unsupportedResourceContractIsTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::UnsupportedContract);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarIntegerAddSubProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto result = loom::hardware::test::specializeAndExportPortableProvider(
      std::move(module), abi, registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::ScalarIntegerAddSub &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "resource contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "resource contract lost its typed Unsupported classification");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *context, fabric, abi.abi(), true);
  const std::string before = moduleText(*wrongPorts.module);

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarIntegerAddSubProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {wrongPorts.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(*wrongPorts.module, abi,
                                              associations, recipes, registry,
                                              externalContracts),
              "leaf port");
  require(test, moduleText(*wrongPorts.module) == before,
          "invalid leaf ports partially mutated the common skeleton");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, store, fabric,
                                    ConfigurationAbiKind::MissingSubtract),
          store),
      "semantic");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, store, fabric,
                                    ConfigurationAbiKind::ExtraSemanticValue),
          store),
      "semantic");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredCodebookAndDeterminism(root);
  singletonNeedsNoSelector(root / "singleton");
  subtractInactiveControlsUnassignedCode(root / "subtract_inactive");
  unsupportedResourceContractIsTransactional(root / "unsupported_contract");
  malformedInputsFailClosed(root / "malformed");
  return 0;
}
