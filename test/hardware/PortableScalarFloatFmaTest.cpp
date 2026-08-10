#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/CirctConformance.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarFloatFma.h"
#include "PortableProviderTestSupport.h"

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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
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
    fail(test,
         "accepted invalid portable scalar FMA input expected to report " +
             expected.str());
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

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

enum class FabricFixtureKind {
  Canonical,
  NarrowPorts,
  WrongContract,
  SingletonF32,
};

enum class ConfigurationAbiKind {
  Complete,
  MissingBF16,
  ExtraSemanticValue,
};

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           FabricFixtureKind kind = FabricFixtureKind::Canonical) {
  const unsigned portWidth = kind == FabricFixtureKind::NarrowPorts ? 32 : 64;
  const llvm::StringRef floatFormats =
      kind == FabricFixtureKind::SingletonF32
          ? R"mlir(["f32"])mlir"
          : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  std::string sourceText;
  llvm::raw_string_ostream source(sourceText);
  source << R"mlir(
    module {
      fabric.module @scalar_float_fma(
          %a: !fabric.bits<)mlir"
         << portWidth << R"mlir(>, %b: !fabric.bits<)mlir" << portWidth
         << R"mlir(>, %c: !fabric.bits<)mlir" << portWidth
         << R"mlir(>) -> !fabric.bits<)mlir" << portWidth << R"mlir(> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<)mlir"
         << portWidth << R"mlir(>, %pb = %b : !fabric.bits<)mlir" << portWidth
         << R"mlir(>, %pc = %c : !fabric.bits<)mlir" << portWidth
         << R"mlir(>) -> !fabric.bits<)mlir" << portWidth << R"mlir(> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<)mlir"
         << portWidth << R"mlir(>, %fb = %pb : !fabric.bits<)mlir" << portWidth
         << R"mlir(>, %fc = %pc : !fabric.bits<)mlir" << portWidth
         << R"mlir(>) -> !fabric.bits<)mlir" << portWidth << R"mlir(> {
            %value = fabric.op [@math.fma] (%fa, %fb, %fc)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatFma>,
               hw_params = {
                 float_formats = )mlir"
         << floatFormats << R"mlir(,
                 behavior = {
                   rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"}}}
              : (!fabric.bits<)mlir"
         << portWidth << R"mlir(>, !fabric.bits<)mlir" << portWidth
         << R"mlir(>, !fabric.bits<)mlir" << portWidth
         << R"mlir(>) -> !fabric.bits<)mlir" << portWidth << R"mlir(>
            fabric.yield %value : !fabric.bits<)mlir"
         << portWidth << R"mlir(>
          }
        }
        fabric.yield %pe : !fabric.bits<)mlir"
         << portWidth << R"mlir(>
      }
    }
  )mlir";

  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(source.str(), &fabricContext());
  require(test, static_cast<bool>(parsed), "could not parse Fabric fixture");

  const ::fabric::ResourceContract contract =
      kind == FabricFixtureKind::WrongContract
          ? ::fabric::loopCarryOperationResourceContract()
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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarFloatFma)
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
              "System has no physical scalar float FMA occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no scalar float FMA occurrence");
}

::fabric::FloatFormat
behaviorFormat(llvm::StringRef test,
               const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  const mlir::Type type = point.representativeActor.type.getInput(0);
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "Fabric projected an unsupported FMA format");
}

std::uint8_t physicalCode(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 1;
  case ::fabric::FloatFormat::BF16:
    return 2;
  case ::fabric::FloatFormat::F32:
    return 4;
  case ::fabric::FloatFormat::F64:
    return 6;
  }
  llvm_unreachable("unknown floating format");
}

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, capability->configurationFieldSchema.size() == 1,
          "scalar FMA fixture has an unexpected field count");
  const auto fieldReference = capability->configurationFieldSchema.front();
  auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "scalar FMA semantic field relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, !domain.empty(), "Fabric projected no scalar FMA formats");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured FMA behavior has no semantic value");
    const ::fabric::FloatFormat format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F32)
      inactive = semantic;
    if (kind == ConfigurationAbiKind::MissingBF16 &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {physicalCode(format)}});
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x05}});
  require(test, !inactive.empty(), "FMA domain has no f32 inactive value");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{3, std::move(entries)},
      std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture), store));
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
                             bool wrongConfigurationWidth = false,
                             llvm::StringRef moduleName = "scalar_float_fma") {
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
    const auto field = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, field != ports.end(), "configured FMA leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarFloatFmaProvider(registry))
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
          "portable FMA provider emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

struct TestVector final {
  ::fabric::FloatFormat format;
  std::uint64_t lhs;
  std::uint64_t rhs;
  std::uint64_t addend;
};

const llvm::fltSemantics &semantics(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return llvm::APFloat::IEEEhalf();
  case ::fabric::FloatFormat::BF16:
    return llvm::APFloat::BFloat();
  case ::fabric::FloatFormat::F32:
    return llvm::APFloat::IEEEsingle();
  case ::fabric::FloatFormat::F64:
    return llvm::APFloat::IEEEdouble();
  }
  llvm_unreachable("unknown floating format");
}

unsigned bitWidth(::fabric::FloatFormat format) {
  return ::fabric::getBitWidth(format);
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint64_t bits) {
  return llvm::APFloat(semantics(format), llvm::APInt(bitWidth(format), bits));
}

llvm::APFloat fused(const TestVector &vector) {
  llvm::APFloat result = floating(vector.format, vector.lhs);
  const llvm::APFloat rhs = floating(vector.format, vector.rhs);
  const llvm::APFloat addend = floating(vector.format, vector.addend);
  (void)result.fusedMultiplyAdd(rhs, addend,
                                llvm::RoundingMode::NearestTiesToEven);
  return result;
}

llvm::APFloat split(const TestVector &vector) {
  llvm::APFloat result = floating(vector.format, vector.lhs);
  const llvm::APFloat rhs = floating(vector.format, vector.rhs);
  const llvm::APFloat addend = floating(vector.format, vector.addend);
  (void)result.multiply(rhs, llvm::RoundingMode::NearestTiesToEven);
  (void)result.add(addend, llvm::RoundingMode::NearestTiesToEven);
  return result;
}

std::uint64_t bits(const llvm::APFloat &value) {
  return value.bitcastToAPInt().getZExtValue();
}

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<TestVector> testVectors() {
  std::vector<TestVector> result;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    const unsigned width = bitWidth(format);
    const std::uint64_t mask = widthMask(width);
    const unsigned fractionBits = format == ::fabric::FloatFormat::F64    ? 52
                                  : format == ::fabric::FloatFormat::F32  ? 23
                                  : format == ::fabric::FloatFormat::BF16 ? 7
                                                                          : 10;
    const unsigned exponentBits = width - fractionBits - 1;
    const std::uint64_t one =
        std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
        << fractionBits;
    const std::uint64_t half = one - (std::uint64_t{1} << fractionBits);
    const std::uint64_t two = one + (std::uint64_t{1} << fractionBits);
    const std::uint64_t infinity = ((std::uint64_t{1} << exponentBits) - 1)
                                   << fractionBits;
    const std::uint64_t maximumFinite = infinity - 1;
    const std::uint64_t quietNaN =
        infinity | (std::uint64_t{1} << (fractionBits - 1));
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    const std::array curated = {
        TestVector{format, 0, 0, 0},
        TestVector{format, sign, 0, sign},
        TestVector{format, one, one, one},
        TestVector{format, one | sign, one, one},
        TestVector{format, 1, one, 0},
        TestVector{format, (std::uint64_t{1} << fractionBits) - 1, one, 1},
        TestVector{format, infinity, 0, one},
        TestVector{format, infinity, one, infinity | sign},
        TestVector{format, quietNaN, one, one},
        TestVector{format, infinity | 1, one, one},
    };
    result.insert(result.end(), curated.begin(), curated.end());
    result.push_back({format, sign | quietNaN | 5, one, quietNaN | 3});
    result.push_back({format, one, sign | quietNaN | 7, quietNaN | 3});
    result.push_back({format, sign | quietNaN | 5, infinity | 3, one});
    result.push_back({format, one, one, sign | infinity | 5});
    result.push_back({format, sign | infinity | 3, one, quietNaN | 5});
    result.push_back({format, 0, 0, sign});
    result.push_back({format, sign, 0, 0});
    const TestVector singleRound{format, one | 1, one | 1, sign | one | 2};
    require("testVectors", bits(fused(singleRound)) != bits(split(singleRound)),
            "single-rounding witness did not distinguish fused evaluation");
    if (format == ::fabric::FloatFormat::F32) {
      require("testVectors", bits(fused(singleRound)) == 0x28800000,
              "f32 single-rounding witness changed");
      require("testVectors", bits(split(singleRound)) == 0,
              "f32 split witness no longer rounds to zero");
    }
    result.push_back(singleRound);
    result.push_back({format, 1, half, (std::uint64_t{1} << fractionBits) - 1});
    result.push_back({format, sign | 1, half, 0});
    result.push_back({format, maximumFinite, two, sign | maximumFinite});
    if (format == ::fabric::FloatFormat::F32) {
      const TestVector cancellation{format, 0x00000001, 0x7f7fffff, 0xb5000000};
      require("testVectors", bits(fused(cancellation)) == 0xa9000000,
              "f32 deep-cancellation witness changed");
      result.push_back(cancellation);
    }

    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ width;
    unsigned fusedSensitive = 0;
    for (unsigned index = 0; index < 20000 && fusedSensitive < 4; ++index) {
      TestVector vector{format, nextRandom(state) & mask,
                        nextRandom(state) & mask, nextRandom(state) & mask};
      const llvm::APFloat fusedValue = fused(vector);
      const llvm::APFloat splitValue = split(vector);
      if (!fusedValue.isNaN() && !splitValue.isNaN() &&
          bits(fusedValue) != bits(splitValue)) {
        result.push_back(vector);
        ++fusedSensitive;
      }
    }
    require("testVectors", fusedSensitive == 4,
            "deterministic search did not find four fused-sensitive values");
    for (unsigned index = 0; index < 32; ++index)
      result.push_back({format, nextRandom(state) & mask,
                        nextRandom(state) & mask, nextRandom(state) & mask});
  }
  return result;
}

std::string hex64(std::uint64_t value) {
  std::ostringstream stream;
  stream << "64'h" << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

std::string testbench() {
  std::ostringstream output;
  output << R"sv(module testbench;
  logic [63:0] data_input_0;
  logic [63:0] data_input_1;
  logic [63:0] data_input_2;
  logic [2:0] config_0;
  logic [63:0] data_output_0;

  scalar_float_fma dut(.*);

  task automatic check_value(
      input logic [2:0] mode,
      input logic [63:0] lhs,
      input logic [63:0] rhs,
      input logic [63:0] addend,
      input logic [63:0] expected);
    begin
      config_0 = mode;
      data_input_0 = lhs;
      data_input_1 = rhs;
      data_input_2 = addend;
      #1;
      if (data_output_0 !== expected) begin
        $fatal(1, "FMA mismatch mode=%0d lhs=%h rhs=%h addend=%h got=%h expected=%h",
               mode, lhs, rhs, addend, data_output_0, expected);
      end
      case (mode)
        3'd1, 3'd2: if (data_output_0[63:16] !== 48'd0)
          $fatal(1, "16-bit result was not low aligned");
        3'd6: ;
        default: if (data_output_0[63:32] !== 32'd0)
          $fatal(1, "32-bit result was not low aligned");
      endcase
    end
  endtask

  initial begin
)sv";
  for (const TestVector &vector : testVectors()) {
    const llvm::APFloat expected = fused(vector);
    const std::uint64_t expectedBits = bits(expected);
    output << "    check_value(3'd" << unsigned(physicalCode(vector.format))
           << ", " << hex64(vector.lhs) << ", " << hex64(vector.rhs) << ", "
           << hex64(vector.addend) << ", " << hex64(expectedBits) << ");\n";
  }
  const TestVector inactive{::fabric::FloatFormat::F32, 0x3f800000, 0x40000000,
                            0x3f800000};
  output << "    check_value(3'd0, " << hex64(inactive.lhs) << ", "
         << hex64(inactive.rhs) << ", " << hex64(inactive.addend) << ", "
         << hex64(bits(fused(inactive))) << ");\n";
  output << R"sv(    $finish;
  end
endmodule
)sv";
  return output.str();
}

void configuredBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture firstSkeleton =
      makeSkeleton(test, *context, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(firstSkeleton.leaf.getPortList());
  require(test,
          ports.size() == 5 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "data_input_2" &&
              ports.atInput(3).getName() == "config_0" &&
              ports.atInput(3).type ==
                  mlir::IntegerType::get(context.get(), 3) &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived scalar FMA leaf ports are not canonical");
  const std::string first = specialize(test, firstSkeleton, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second = specialize(test, secondSkeleton, fabric, abi);
  require(test, first == second,
          "identical scalar FMA inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              !rtl.contains("shortreal") && !rtl.contains("real") &&
              !rtl.contains("DPI"),
          "portable scalar FMA is not self-contained synthesizable RTL");

  const std::string yosysScript = R"ys(
read_verilog -sv scalar_float_fma.sv
hierarchy -check -top scalar_float_fma
proc
opt
check -assert
select -assert-none scalar_float_fma/t:$*ff* scalar_float_fma/t:$*latch* scalar_float_fma/t:$_*FF* scalar_float_fma/t:$_*LATCH* scalar_float_fma/t:$mem* scalar_float_fma/m:*
synth -noabc -top scalar_float_fma
check -assert
select -assert-none scalar_float_fma/t:$*ff* scalar_float_fma/t:$*latch* scalar_float_fma/t:$_*FF* scalar_float_fma/t:$_*LATCH* scalar_float_fma/t:$mem* scalar_float_fma/m:*
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts", {{"scalar_float_fma.sv", first},
                               {"testbench.sv", testbench()},
                               {"portable_scalar_float_fma.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::SingletonF32);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture firstSkeleton =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), false,
                   "scalar_float_fma_singleton");
  const circt::hw::ModulePortInfo ports(firstSkeleton.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "data_input_2" &&
              ports.atOutput(0).getName() == "data_output_0",
          "singleton scalar FMA leaf retained a redundant selector");
  const std::string first = specialize(test, firstSkeleton, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), false,
                   "scalar_float_fma_singleton");
  const std::string second = specialize(test, secondSkeleton, fabric, abi);
  require(test, first == second,
          "singleton scalar FMA generation is not deterministic");
  require(test,
          llvm::StringRef(first).contains("loom_fma_e8_f23") &&
              !llvm::StringRef(first).contains("config_0"),
          "singleton scalar FMA emitted configurable or non-f32 logic");

  const std::string singletonTestbench = R"sv(
module testbench_singleton;
  logic [63:0] data_input_0;
  logic [63:0] data_input_1;
  logic [63:0] data_input_2;
  logic [63:0] data_output_0;

  scalar_float_fma_singleton dut(.*);

  initial begin
    data_input_0 = 64'h000000003fc00000;
    data_input_1 = 64'h0000000040000000;
    data_input_2 = 64'h000000003f800000;
    #1;
    if (data_output_0 !== 64'h0000000040800000)
      $fatal(1, "singleton f32 FMA produced the wrong result");
    $finish;
  end
endmodule
)sv";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts", {{"scalar_float_fma_singleton.sv", first},
                               {"testbench_singleton.sv", singletonTestbench}}))
    fail(test, llvm::toString(std::move(error)));
}

void invalidInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI valid = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarFloatFmaProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto rejects = [&](const FinalizedConfigurationABI &abi,
                     bool wrongConfigurationWidth, llvm::StringRef message) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi(),
                                            wrongConfigurationWidth);
    ModuleRootCirctSkeleton module{
        std::move(skeleton.module),
        {{skeleton.leaf, fabric.physicalOccurrence}}};
    expectError(test,
                loom::hardware::test::specializeAndExportPortableProvider(
                    std::move(module), abi, registry, externalContracts),
                message);
  };
  rejects(valid, true, "leaf port");
  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(test, fabric,
                                            ConfigurationAbiKind::MissingBF16),
                  store),
              "semantic");
  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, fabric, ConfigurationAbiKind::ExtraSemanticValue),
                  store),
              "semantic");

  FabricFixture wrongContract =
      makeFabric(test, store, FabricFixtureKind::WrongContract);
  FinalizedConfigurationABI wrongContractAbi =
      makeConfigurationAbi(test, store, wrongContract);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, wrongContract, wrongContractAbi.abi());
  ModuleRootCirctSkeleton module{
      std::move(skeleton.module),
      {{skeleton.leaf, wrongContract.physicalOccurrence}}};
  auto unsupportedContract =
      loom::hardware::test::specializeAndExportPortableProvider(
          std::move(module), wrongContractAbi, registry, externalContracts);
  require(test, !unsupportedContract,
          "FMA provider accepted an unsupported resource contract");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      unsupportedContract.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::ScalarFloatFma &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "resource contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "resource contract lost its typed Unsupported classification");
}

void physicalCapacityNarrowsTheFormatDomain(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture narrow =
      makeFabric(test, store, FabricFixtureKind::NarrowPorts);
  const auto *capability =
      narrow.fabric.view().resolvedFabricOpCapability(narrow.occurrence);
  require(test, capability != nullptr, "narrow FMA capability did not resolve");
  auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "narrow FMA semantic field relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 3,
          "32-bit physical ports did not remove only the f64 behavior");
  bool sawF16 = false;
  bool sawBF16 = false;
  bool sawF32 = false;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "reachable FMA behavior has no semantic value");
    const loom::CanonicalSemanticBytes expected =
        take(test, relation.projectSemanticValue(
                       point.representativeActor, point.operandPorts,
                       point.resultPorts, point.resolvedIndexWidth));
    require(test, point.semanticConfiguration->bytes().equals(expected.bytes()),
            "FMA behavior disagrees with its sealed semantic relation");
    switch (behaviorFormat(test, point)) {
    case ::fabric::FloatFormat::F16:
      sawF16 = true;
      break;
    case ::fabric::FloatFormat::BF16:
      sawBF16 = true;
      break;
    case ::fabric::FloatFormat::F32:
      sawF32 = true;
      break;
    case ::fabric::FloatFormat::F64:
      fail(test, "32-bit physical ports retained an f64 behavior");
    }
  }
  require(test, sawF16 && sawBF16 && sawF32,
          "32-bit physical ports removed a reachable format");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, narrow);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, narrow, abi.abi(),
                                          false, "scalar_float_fma_narrow");
  const std::string rtl = specialize(test, skeleton, narrow, abi);
  const llvm::StringRef text(rtl);
  require(test,
          text.contains("loom_fma_e5_f10") && text.contains("loom_fma_e8_f7") &&
              text.contains("loom_fma_e8_f23") &&
              !text.contains("loom_fma_e11_f52"),
          "narrow FMA RTL does not match the reachable format domain");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndDeterminism(root);
  singletonNeedsNoSelector(root / "singleton");
  physicalCapacityNarrowsTheFormatDomain(root / "narrow");
  invalidInputsFailClosed(root / "invalid");
  return 0;
}
