#include "ConfigurationABI3TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorPackUnpack.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ImplementationFamily.h"
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

void expectSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted an invalid fixed-vector adapter actor");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid portable fixed-vector adapter input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
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

enum class AdapterKind { Pack, Unpack };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  AdapterKind kind;
};

::fabric::ImplementationFamilyId familyId(AdapterKind kind) {
  return kind == AdapterKind::Pack
             ? ::fabric::ImplementationFamilyId::FixedVectorPack
             : ::fabric::ImplementationFamilyId::FixedVectorUnpack;
}

::dataflow::OperationSchemaId operationSchema(AdapterKind kind) {
  return kind == AdapterKind::Pack
             ? ::dataflow::OperationSchemaId::DataflowPack
             : ::dataflow::OperationSchemaId::DataflowUnpack;
}

llvm::StringRef moduleName(AdapterKind kind) {
  return kind == AdapterKind::Pack ? "fixed_vector_pack"
                                   : "fixed_vector_unpack";
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         AdapterKind kind, bool wrongContract = false) {
  const unsigned inputWidth = kind == AdapterKind::Pack ? 128 : 64;
  const unsigned outputWidth = kind == AdapterKind::Pack ? 64 : 128;
  const llvm::StringRef schema =
      kind == AdapterKind::Pack ? "dataflow.pack" : "dataflow.unpack";
  const llvm::StringRef family =
      kind == AdapterKind::Pack ? "FixedVectorPack" : "FixedVectorUnpack";

  std::string sourceText;
  llvm::raw_string_ostream source(sourceText);
  source << "module {\n"
         << "  fabric.module @" << moduleName(kind) << "(\n"
         << "      %input: !fabric.bits<128>) -> !fabric.bits<128> {\n"
         << "    %pe = fabric.pe [spatial]\n"
         << "        (%pe_input = %input : !fabric.bits<128>)\n"
         << "        -> !fabric.bits<128> {\n"
         << "      %fu = fabric.fu\n"
         << "          (%fu_input = %pe_input : !fabric.bits<128>";
  if (inputWidth != 128)
    source << " to !fabric.bits<" << inputWidth << ">";
  source << ") -> !fabric.bits<128> {\n"
         << "        %value = fabric.op [@" << schema << "] (%fu_input)\n"
         << "          {implementation_family =\n"
         << "             #fabric.implementation_family<" << family << ">,\n"
         << "           hw_params = {\n"
         << "             integer_element_widths = [8 : i32, 16 : i32],\n"
         << "             float_element_formats = [\"f16\", \"f32\"],\n"
         << "             max_payload_bits = 64 : i32}}\n"
         << "          : (!fabric.bits<" << inputWidth << ">) -> !fabric.bits<"
         << outputWidth << ">\n"
         << "        fabric.yield %value : !fabric.bits<" << outputWidth;
  if (outputWidth != 128)
    source << "> to !fabric.bits<128";
  source << ">\n"
         << "      }\n"
         << "    }\n"
         << "    fabric.yield %pe : !fabric.bits<128>\n"
         << "  }\n"
         << "}\n";

  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(source.str(), &fabricContext());
  require(test, static_cast<bool>(parsed), "could not parse Fabric fixture");

  const ::fabric::ResourceContract contract =
      wrongContract ? ::fabric::loopCarryOperationResourceContract()
                    : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedContract(encoded.begin(), encoded.end());
  parsed->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });

  ::fabric::ModuleOp root;
  parsed->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != familyId(kind))
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
              "System has no physical fixed-vector adapter occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence, kind};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector adapter occurrence");
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system)),
          store));
}

dataflow::CanonicalActorSchemaProjection
actor(AdapterKind kind, llvm::ArrayRef<std::int64_t> shape, mlir::Type element,
      unsigned packedWidth) {
  mlir::MLIRContext &context = fabricContext();
  mlir::VectorType vector = mlir::VectorType::get(shape, element);
  mlir::Type packed = mlir::IntegerType::get(&context, packedWidth);
  mlir::FunctionType type =
      kind == AdapterKind::Pack
          ? mlir::FunctionType::get(&context, {vector}, {packed})
          : mlir::FunctionType::get(&context, {packed}, {vector});
  return {operationSchema(kind), type, dataflow::NoPayload{}};
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

void checkGeneratedRegistration() {
  const llvm::StringRef test = __func__;
  for (AdapterKind kind : {AdapterKind::Pack, AdapterKind::Unpack}) {
    const auto family = familyId(kind);
    const ::fabric::ImplementationFamilyDescriptor &descriptor =
        ::fabric::implementationFamily(family);
    require(
        test,
        descriptor.familyId == family &&
            descriptor.capabilityParamsSchema ==
                ::fabric::CapabilityParamsSchemaId::FixedVectorAdapterParams &&
            descriptor.typedAdmissionProvider ==
                ::fabric::TypedAdmissionProviderId::
                    FixedVectorAdapterAdmission &&
            descriptor.admittedSchemas.size() == 1 &&
            descriptor.admittedSchemas.front() == operationSchema(kind),
        "generated fixed-vector adapter family descriptor changed");
  }

  FabricOperationProviderRegistry registry;
  expectSuccess(test, registerPortableFixedVectorPackProvider(registry));
  expectSuccess(test, registerPortableFixedVectorUnpackProvider(registry));
  const auto coverage = registry.coverage();
  std::size_t providedFamilies = 0;
  for (const FabricOperationProviderCoverage &entry : coverage) {
    if (!entry.recipes.empty())
      ++providedFamilies;
    if (entry.implementationFamily !=
            ::fabric::ImplementationFamilyId::FixedVectorPack &&
        entry.implementationFamily !=
            ::fabric::ImplementationFamilyId::FixedVectorUnpack)
      continue;
    require(test,
            entry.recipes ==
                std::vector<BackendRecipeKey>{
                    BackendRecipeKey::PortableSystemVerilog},
            "fixed-vector adapter provider coverage changed");
  }
  require(test, providedFamilies == 2,
          "adapter registration escaped its two exact family keys");
}

void checkCanonicalAdmission(const FabricFixture &pack,
                             const FabricFixture &unpack) {
  const llvm::StringRef test = __func__;
  mlir::MLIRContext &context = fabricContext();
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);
  mlir::Type i16 = mlir::IntegerType::get(&context, 16);
  mlir::Type i32 = mlir::IntegerType::get(&context, 32);
  mlir::Type f16 = mlir::Float16Type::get(&context);
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type f64 = mlir::Float64Type::get(&context);

  for (const FabricFixture *fixture : {&pack, &unpack}) {
    const auto &resolved = capability(test, *fixture);
    require(test,
            resolved.implementationFamily == familyId(fixture->kind) &&
                resolved.enabledOperationSchemas ==
                    std::vector<::dataflow::OperationSchemaId>{
                        operationSchema(fixture->kind)} &&
                resolved.configurationFieldSchema.empty(),
            "finalized adapter capability changed identity or configuration");

    std::vector<std::uint32_t> physicalInputWidths;
    std::vector<std::uint32_t> physicalResultWidths;
    for (const auto &port : resolved.physicalPorts) {
      auto &widths =
          port.reference.direction == loom::fabric::FabricPortDirection::Input
              ? physicalInputWidths
              : physicalResultWidths;
      widths.push_back(port.payloadWidthBits);
    }
    require(test,
            physicalInputWidths.size() == 1 && physicalResultWidths.size() == 1,
            "adapter capability changed its unary physical shape");
    const auto relation =
        take(test, ::fabric::resolveFabricOpSemanticFieldRelation(
                       resolved.implementationFamily,
                       resolved.parameterizedCapability,
                       resolved.enabledOperationSchemas, physicalInputWidths,
                       physicalResultWidths, fabricContext()));
    require(test,
            relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                !relation.hasConfigurationField() &&
                relation.finiteBehaviorDomain().size() == 1 &&
                !relation.finiteBehaviorDomain().front().semanticConfiguration,
            "multi-domain pack/unpack capability gained configuration");

    expectSuccess(test, resolved.admit(actor(fixture->kind, {8}, i8, 64), 64));
    expectSuccess(test,
                  resolved.admit(actor(fixture->kind, {2, 2}, i16, 64), 64));
    expectSuccess(test, resolved.admit(actor(fixture->kind, {2}, f32, 64), 64));
    expectSuccess(test,
                  resolved.admit(actor(fixture->kind, {2, 2}, f16, 64), 64));

    expectError(test, resolved.admit(actor(fixture->kind, {2}, i32, 64), 64),
                "element width");
    expectError(test, resolved.admit(actor(fixture->kind, {1}, f64, 64), 64),
                "element type");
    expectError(test, resolved.admit(actor(fixture->kind, {16}, i8, 128), 64),
                "payload capacity");
    expectError(test, resolved.admit(actor(fixture->kind, {8}, i8, 32), 64),
                "packed width");
  }

  const auto &packParams = capability(test, pack).parameterizedCapability;
  const auto &unpackParams = capability(test, unpack).parameterizedCapability;
  expectError(test,
              ::fabric::verifyImplementationFamilyAdmission(
                  ::fabric::ImplementationFamilyId::FixedVectorPack,
                  &packParams, actor(AdapterKind::Unpack, {8}, i8, 64)),
              "not admitted");
  expectError(test,
              ::fabric::verifyImplementationFamilyAdmission(
                  ::fabric::ImplementationFamilyId::FixedVectorUnpack,
                  &unpackParams, actor(AdapterKind::Pack, {8}, i8, 64)),
              "not admitted");
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
                             bool malformedInput = false) {
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
  require(test, ports.size() == 2,
          "configuration-free adapter leaf did not have two ports");
  if (malformedInput) {
    const unsigned width =
        mlir::cast<mlir::IntegerType>(ports.front().type).getWidth();
    ports.front().type = builder.getIntegerType(width - 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName(fabric.kind)), ports);
  return SkeletonFixture{std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  expectSuccess(test, registerPortableFixedVectorPackProvider(registry));
  expectSuccess(test, registerPortableFixedVectorUnpackProvider(registry));
  return registry;
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     {std::move(skeleton.module),
                      {{skeleton.leaf, fabric.physicalOccurrence}}},
                     abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable adapter emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

std::string deterministicRtl(llvm::StringRef test, const FabricFixture &fabric,
                             const FinalizedConfigurationABI &abi) {
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 2 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived adapter leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical adapter inputs produced different RTL");
  require(test, !llvm::StringRef(firstRtl).contains("config_"),
          "configuration-free adapter emitted a configuration port");
  return firstRtl;
}

struct EmittedRtl final {
  std::string pack;
  std::string unpack;
};

EmittedRtl validAdapters(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture pack = makeFabric(test, store, AdapterKind::Pack);
  FabricFixture unpack = makeFabric(test, store, AdapterKind::Unpack);
  checkCanonicalAdmission(pack, unpack);
  FinalizedConfigurationABI packAbi = makeConfigurationAbi(test, store, pack);
  FinalizedConfigurationABI unpackAbi =
      makeConfigurationAbi(test, store, unpack);
  return {deterministicRtl(test, pack, packAbi),
          deterministicRtl(test, unpack, unpackAbi)};
}

void writeToolInputs(const std::filesystem::path &root, const EmittedRtl &rtl) {
  const llvm::StringRef test = __func__;
  const std::string testbench = R"sv(
module testbench;
  logic [127:0] pack_input;
  logic [63:0] pack_output;
  logic [63:0] unpack_input;
  logic [127:0] unpack_output;

  fixed_vector_pack pack_dut(
    .data_input_0(pack_input),
    .data_output_0(pack_output)
  );
  fixed_vector_unpack unpack_dut(
    .data_input_0(unpack_input),
    .data_output_0(unpack_output)
  );

  initial begin
    pack_input = 128'hfedcba9876543210_4030201004030201;
    unpack_input = 64'h4030201004030201;
    #1;
    if (pack_output !== 64'h4030201004030201)
      $fatal(1, "pack did not preserve row-major lane-zero-low bits");
    if (unpack_output !== 128'h0000000000000000_4030201004030201)
      $fatal(1, "unpack did not zero-extend row-major lane-zero-low bits");
    $finish;
  end
endmodule
)sv";
  const std::string synthesisTop = R"sv(
module fixed_vector_adapter_synthesis_top(
  input logic [127:0] pack_input,
  input logic [63:0] unpack_input,
  output logic [63:0] pack_output,
  output logic [127:0] unpack_output
);
  fixed_vector_pack pack_dut(
    .data_input_0(pack_input),
    .data_output_0(pack_output)
  );
  fixed_vector_unpack unpack_dut(
    .data_input_0(unpack_input),
    .data_output_0(unpack_output)
  );
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv fixed_vector_pack.sv fixed_vector_unpack.sv synthesis_top.sv
hierarchy -check -top fixed_vector_adapter_synthesis_top
proc
opt
check -assert
synth -top fixed_vector_adapter_synthesis_top
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "tool-artifacts",
          {{"fixed_vector_pack.sv", rtl.pack},
           {"fixed_vector_unpack.sv", rtl.unpack},
           {"testbench.sv", testbench},
           {"synthesis_top.sv", synthesisTop},
           {"portable_fixed_vector_pack_unpack.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void invalidInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  ExternalImplementationContractCatalog externalContracts;

  FabricFixture pack = makeFabric(test, store, AdapterKind::Pack);
  FinalizedConfigurationABI packAbi = makeConfigurationAbi(test, store, pack);
  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, pack, packAbi.abi(), true);
  expectError(test,
              loom::hardware::test::specializeAndExportPortableProvider(
                  {std::move(malformed.module),
                   {{malformed.leaf, pack.physicalOccurrence}}},
                  packAbi, registry, externalContracts),
              "leaf port");

  FabricFixture unpack = makeFabric(test, store, AdapterKind::Unpack, true);
  FinalizedConfigurationABI unpackAbi =
      makeConfigurationAbi(test, store, unpack);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture wrongContract =
      makeSkeleton(test, *contractContext, unpack, unpackAbi.abi());
  expectTypedUnsupported(
      test,
      loom::hardware::test::specializeAndExportPortableProvider(
          {std::move(wrongContract.module),
           {{wrongContract.leaf, unpack.physicalOccurrence}}},
          unpackAbi, registry, externalContracts),
      ::fabric::ImplementationFamilyId::FixedVectorUnpack,
      "unsupported adapter resource contract");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  checkGeneratedRegistration();
  const EmittedRtl rtl = validAdapters(root);
  writeToolInputs(root, rtl);
  invalidInputsAreTransactional(root / "invalid");
  return 0;
}
