#include "ConfigurationABI3TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/IntegerSaturatingAddSub.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
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

enum class Geometry {
  Scalar,
  FixedVector,
};

enum class ConfigurationAbiKind {
  Complete,
  MissingBehavior,
  ExtraBehavior,
};

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
    fail(test, "accepted malformed saturating add/sub input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
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

::fabric::ImplementationFamilyId familyFor(Geometry geometry) {
  return geometry == Geometry::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarIntegerSaturatingAddSub
             : ::fabric::ImplementationFamilyId::
                   FixedVectorIntegerSaturatingAddSub;
}

llvm::StringRef moduleName(Geometry geometry, bool singleton) {
  if (singleton)
    return "scalar_integer_saturating_add";
  return geometry == Geometry::Scalar
             ? "scalar_integer_saturating_add_sub"
             : "fixed_vector_integer_saturating_add_sub";
}

llvm::StringRef configuredFabricSource(Geometry geometry) {
  if (geometry == Geometry::Scalar)
    return R"mlir(
    module {
      fabric.module @scalar_integer_saturating_add_sub(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
            %value = fabric.op [
                @llvm.intr.sadd.sat, @llvm.intr.uadd.sat,
                @llvm.intr.ssub.sat, @llvm.intr.usub.sat] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<
                   ScalarIntegerSaturatingAddSub>,
               hw_params = {integer_widths = [
                 8 : i32, 16 : i32, 32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir";

  return R"mlir(
    module {
      fabric.module @fixed_vector_integer_saturating_add_sub(
          %a: !fabric.bits<48>, %b: !fabric.bits<48>)
          -> !fabric.bits<48> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<48>, %pb = %b : !fabric.bits<48>)
            -> !fabric.bits<48> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<48>,
               %fb = %pb : !fabric.bits<48>) -> !fabric.bits<48> {
            %value = fabric.op [
                @llvm.intr.sadd.sat, @llvm.intr.uadd.sat,
                @llvm.intr.ssub.sat, @llvm.intr.usub.sat] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<
                   FixedVectorIntegerSaturatingAddSub>,
               hw_params = {
                 element_widths = [8 : i32, 16 : i32],
                 max_payload_bits = 48 : i32}}
              : (!fabric.bits<48>, !fabric.bits<48>) -> !fabric.bits<48>
            fabric.yield %value : !fabric.bits<48>
          }
        }
        fabric.yield %pe : !fabric.bits<48>
      }
    }
  )mlir";
}

llvm::StringRef singletonFabricSource() {
  return R"mlir(
    module {
      fabric.module @scalar_integer_saturating_add(
          %a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>,
               %fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> {
            %value = fabric.op [@llvm.intr.sadd.sat] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<
                   ScalarIntegerSaturatingAddSub>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";
}

struct FabricFixture final {
  Geometry geometry;
  bool singleton;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         Geometry geometry, bool singleton = false,
                         bool supportedContract = true) {
  require(test, !singleton || geometry == Geometry::Scalar,
          "only the scalar singleton fixture is defined");
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      singleton ? singletonFabricSource() : configuredFabricSource(geometry),
      &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const ::fabric::ResourceContract &resourceContract =
      supportedContract ? ::fabric::oneCycleElasticOperationResourceContract()
                        : ::fabric::loopCarryOperationResourceContract();
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
      if (capability.implementationFamily != familyFor(geometry))
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
              "System has no physical saturating add/sub occurrence");
      return FabricFixture{geometry,          singleton,
                           std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no saturating add/sub occurrence");
}

unsigned
modeWidth(llvm::StringRef test,
          const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  mlir::Type input = point.representativeActor.type.getInput(0);
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(input))
    return integer.getWidth();
  auto vector = llvm::dyn_cast<mlir::VectorType>(input);
  require(test, static_cast<bool>(vector),
          "behavior witness has no integer geometry");
  auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  require(test, static_cast<bool>(element),
          "vector behavior witness has no integer element");
  return element.getWidth();
}

std::uint8_t physicalCode(llvm::StringRef test,
                          dataflow::OperationSchemaId schema, unsigned width) {
  unsigned operation = 0;
  switch (schema) {
  case dataflow::OperationSchemaId::LLVMSAddSat:
    operation = 0;
    break;
  case dataflow::OperationSchemaId::LLVMUAddSat:
    operation = 1;
    break;
  case dataflow::OperationSchemaId::LLVMSSubSat:
    operation = 2;
    break;
  case dataflow::OperationSchemaId::LLVMUSubSat:
    operation = 3;
    break;
  default:
    fail(test, "Fabric relation exposed a foreign behavior");
  }

  unsigned widthOrdinal = 0;
  if (width == 8)
    widthOrdinal = 0;
  else if (width == 16)
    widthOrdinal = 1;
  else if (width == 32)
    widthOrdinal = 2;
  else
    fail(test, "Fabric relation exposed an unexpected active width");
  return static_cast<std::uint8_t>(1 + widthOrdinal * 4 + operation);
}

std::vector<FiniteCodebookEntry>
completeEntries(llvm::StringRef test,
                const loom::fabric::ResolvedFabricOpCapabilityView &capability,
                Geometry geometry) {
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured saturating add/sub relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == (geometry == Geometry::Scalar ? 12 : 8),
          "Fabric projected the wrong saturating behavior count");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured behavior has no semantic value");
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {physicalCode(test, point.representativeActor.schema,
                       modeWidth(test, point))}});
  }
  return entries;
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
          "saturating fixture has an unexpected field count");

  std::vector<FiniteCodebookEntry> entries =
      completeEntries(test, *capability, fixture.geometry);
  if (kind == ConfigurationAbiKind::MissingBehavior)
    entries.front().semanticValue = {0xfd};
  if (kind == ConfigurationAbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x0f}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{0x01};
  });
  require(test, inactive != entries.end(),
          "signed i8 add is absent from the physical codebook");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;
  const auto fieldReference = capability->configurationFieldSchema.front();
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{4, std::move(entries)},
      inactiveValue};
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
            "configured saturating leaf did not have four ports");
    ports[2].type = mlir::IntegerType::get(&context, 3);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName(fabric.geometry, fabric.singleton)),
      ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableIntegerSaturatingAddSubProviders(registry))
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
          "portable saturating provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

llvm::StringRef scalarTestbench() {
  return R"sv(
module testbench;
  logic [31:0] data_input_0;
  logic [31:0] data_input_1;
  logic [3:0] config_0;
  logic [31:0] data_output_0;

  scalar_integer_saturating_add_sub dut(.*);

  task automatic check(
      input logic [3:0] mode,
      input logic [31:0] lhs,
      input logic [31:0] rhs,
      input logic [31:0] expected,
      input string label);
    config_0 = mode;
    data_input_0 = lhs;
    data_input_1 = rhs;
    #1;
    if (data_output_0 !== expected)
      $fatal(1, "%s: got %h expected %h", label, data_output_0, expected);
  endtask

  initial begin
    check(4'h1, 32'hdead007f, 32'hbeef0001, 32'h0000007f,
          "signed i8 add positive saturation");
    check(4'h1, 32'h00000080, 32'h000000ff, 32'h00000080,
          "signed i8 add negative saturation");
    check(4'h1, 32'h000000fb, 32'h00000003, 32'h000000fe,
          "signed i8 add normal");
    check(4'h2, 32'h000000ff, 32'h00000001, 32'h000000ff,
          "unsigned i8 add saturation");
    check(4'h2, 32'h0000007f, 32'h00000001, 32'h00000080,
          "unsigned i8 add normal");
    check(4'h3, 32'h00000080, 32'h00000001, 32'h00000080,
          "signed i8 sub negative saturation");
    check(4'h3, 32'h0000007f, 32'h000000ff, 32'h0000007f,
          "signed i8 sub positive saturation");
    check(4'h3, 32'h00000003, 32'h00000005, 32'h000000fe,
          "signed i8 sub normal");
    check(4'h4, 32'h00000000, 32'h00000001, 32'h00000000,
          "unsigned i8 sub saturation");
    check(4'h4, 32'h00000009, 32'h00000004, 32'h00000005,
          "unsigned i8 sub normal");
    check(4'h5, 32'h00007fff, 32'h00000001, 32'h00007fff,
          "signed i16 add positive saturation");
    check(4'h5, 32'h00008000, 32'h0000ffff, 32'h00008000,
          "signed i16 add negative saturation");
    check(4'h6, 32'h0000ffff, 32'h00000001, 32'h0000ffff,
          "unsigned i16 add saturation");
    check(4'h7, 32'h00008000, 32'h00000001, 32'h00008000,
          "signed i16 sub negative saturation");
    check(4'h7, 32'h00007fff, 32'h0000ffff, 32'h00007fff,
          "signed i16 sub positive saturation");
    check(4'h8, 32'h00000000, 32'h00000001, 32'h00000000,
          "unsigned i16 sub saturation");
    check(4'h9, 32'h7fffffff, 32'h00000001, 32'h7fffffff,
          "signed i32 add positive saturation");
    check(4'h9, 32'h80000000, 32'hffffffff, 32'h80000000,
          "signed i32 add negative saturation");
    check(4'ha, 32'hffffffff, 32'h00000001, 32'hffffffff,
          "unsigned i32 add saturation");
    check(4'hb, 32'h80000000, 32'h00000001, 32'h80000000,
          "signed i32 sub negative saturation");
    check(4'hb, 32'h7fffffff, 32'hffffffff, 32'h7fffffff,
          "signed i32 sub positive saturation");
    check(4'hc, 32'h00000000, 32'h00000001, 32'h00000000,
          "unsigned i32 sub saturation");
    check(4'h0, 32'h0000007f, 32'h00000001, 32'h0000007f,
          "unassigned code preserves inactive behavior");
    $finish;
  end
endmodule
)sv";
}

llvm::StringRef vectorTestbench() {
  return R"sv(
module testbench;
  logic [47:0] data_input_0;
  logic [47:0] data_input_1;
  logic [3:0] config_0;
  logic [47:0] data_output_0;

  fixed_vector_integer_saturating_add_sub dut(.*);

  task automatic check(
      input logic [3:0] mode,
      input logic [47:0] lhs,
      input logic [47:0] rhs,
      input logic [47:0] expected,
      input string label);
    config_0 = mode;
    data_input_0 = lhs;
    data_input_1 = rhs;
    #1;
    if (data_output_0 !== expected)
      $fatal(1, "%s: got %h expected %h", label, data_output_0, expected);
  endtask

  initial begin
    check(4'h1, 48'hc001ff7e807f, 48'hc0ffff0101ff,
          48'h8000fe7f817e, "signed i8 add lanes");
    check(4'h2, 48'h8001fe007fff, 48'h80ff05010101,
          48'hffffff0180ff, "unsigned i8 add lanes");
    check(4'h3, 48'h057f80007f80, 48'h0780ff01ff01,
          48'hfe7f81ff7f80, "signed i8 sub lanes");
    check(4'h4, 48'h018010ff0500, 48'h017f20010301,
          48'h000100fe0200, "unsigned i8 sub lanes");
    check(4'h5, 48'h123480007fff, 48'h1111ffff0001,
          48'h234580007fff, "signed i16 add lanes");
    check(4'h6, 48'hff000001ffff, 48'h020000020001,
          48'hffff0003ffff, "unsigned i16 add lanes");
    check(4'h7, 48'h00017fff8000, 48'h0002ffff0001,
          48'hffff7fff8000, "signed i16 sub lanes");
    check(4'h8, 48'hffff00100000, 48'h000100200001,
          48'hfffe00000000, "unsigned i16 sub lanes");
    $finish;
  end
endmodule
)sv";
}

std::string yosysScript(llvm::StringRef top, llvm::StringRef source) {
  return ("read_verilog -sv " + source +
          "\n"
          "hierarchy -check -top " +
          top +
          "\n"
          "proc\n"
          "opt\n"
          "check\n"
          "select -assert-none " +
          top + "/t:$*ff* " + top + "/t:$*latch* " + top + "/t:$_*FF* " + top +
          "/t:$_*LATCH* " + top + "/t:$mem* " + top +
          "/m:*\n"
          "synth -top " +
          top +
          "\n"
          "check\n"
          "stat\n")
      .str();
}

void configuredBehaviorAndDeterminism(const std::filesystem::path &root,
                                      Geometry geometry) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, geometry);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 4),
          "derived saturating leaf ports are not canonical");
  const std::string firstRtl = specialize(test, std::move(first), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl =
      specialize(test, std::move(second), fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical saturating inputs produced different SystemVerilog");
  require(test,
          llvm::StringRef(firstRtl).contains(moduleName(geometry, false)) &&
              llvm::StringRef(firstRtl).contains("config_0"),
          "portable provider did not materialize its configured datapath");

  const bool scalar = geometry == Geometry::Scalar;
  const std::string source = scalar
                                 ? "scalar_integer_saturating_add_sub.sv"
                                 : "fixed_vector_integer_saturating_add_sub.sv";
  const std::string top = moduleName(geometry, false).str();
  const std::string script =
      scalar ? "portable_scalar_integer_saturating_add_sub.ys"
             : "portable_fixed_vector_integer_saturating_add_sub.ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{source, firstRtl},
           {"testbench.sv",
            (scalar ? scalarTestbench() : vectorTestbench()).str()},
           {script, yosysScript(top, source)}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, Geometry::Scalar, true, true);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton saturating add retained a selector");
  const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains(moduleName(Geometry::Scalar, true)) &&
              !llvm::StringRef(rtl).contains("config_0"),
          "singleton saturating add did not lower directly");
}

bool sameCoverage(llvm::ArrayRef<FabricOperationProviderCoverage> lhs,
                  llvm::ArrayRef<FabricOperationProviderCoverage> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index < lhs.size(); ++index)
    if (lhs[index].implementationFamily != rhs[index].implementationFamily ||
        lhs[index].recipes != rhs[index].recipes)
      return false;
  return true;
}

llvm::Expected<FabricOperationProviderOutput>
standInProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

void registrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add(
          {::fabric::ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           standInProvider}))
    fail(test, llvm::toString(std::move(error)));
  const auto before = registry.coverage();
  llvm::Error error =
      registerPortableIntegerSaturatingAddSubProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate package registration succeeded");
  llvm::consumeError(std::move(error));
  require(test, sameCoverage(before, registry.coverage()),
          "failed package registration changed the registry");
}

void unsupportedContractIsTypedAndTransactional(
    const std::filesystem::path &root, Geometry geometry) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, geometry, false, false);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string before = moduleText(*skeleton.module);

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableIntegerSaturatingAddSubProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  auto result =
      specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                      recipes, registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() == familyFor(geometry) &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "resource contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "resource contract lost typed Unsupported classification");
  require(test, moduleText(*skeleton.module) == before,
          "unsupported resource contract mutated the common skeleton");
}

void malformedInputsFailClosed(const std::filesystem::path &root,
                               Geometry geometry) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, geometry);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableIntegerSaturatingAddSubProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};

  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *context, fabric, abi.abi(), true);
  const std::string before = moduleText(*wrongPorts.module);
  const std::vector<FabricOperationLeafAssociation> associations = {
      {wrongPorts.leaf, fabric.physicalOccurrence}};
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
                                    ConfigurationAbiKind::MissingBehavior),
          store),
      "semantic");
  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, store, fabric, ConfigurationAbiKind::ExtraBehavior),
                  store),
              "semantic");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndDeterminism(root / "scalar", Geometry::Scalar);
  configuredBehaviorAndDeterminism(root / "vector", Geometry::FixedVector);
  singletonNeedsNoSelector(root / "singleton");
  registrationIsTransactional();
  unsupportedContractIsTypedAndTransactional(root / "unsupported_scalar",
                                             Geometry::Scalar);
  unsupportedContractIsTypedAndTransactional(root / "unsupported_vector",
                                             Geometry::FixedVector);
  malformedInputsFailClosed(root / "malformed_scalar", Geometry::Scalar);
  malformedInputsFailClosed(root / "malformed_vector", Geometry::FixedVector);
  return EXIT_SUCCESS;
}
