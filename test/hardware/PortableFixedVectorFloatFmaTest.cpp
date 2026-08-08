#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorFloatFma.h"
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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
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

enum class FixtureKind {
  Configured,
  Singleton,
  NonRne,
  UnsupportedContract,
};

enum class AbiKind {
  Complete,
  MissingBehavior,
  ExtraBehavior,
  DirectEncoding,
};

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

struct FloatLayout final {
  unsigned exponentBits;
  unsigned fractionBits;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  std::uint64_t sign() const { return std::uint64_t{1} << (width() - 1); }
  std::uint64_t exponentMask() const {
    return (std::uint64_t{1} << exponentBits) - 1;
  }
  std::uint64_t infinity() const { return exponentMask() << fractionBits; }
  std::uint64_t quietBit() const {
    return std::uint64_t{1} << (fractionBits - 1);
  }
  std::uint64_t one() const {
    return std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
           << fractionBits;
  }
};

struct TestVector final {
  std::uint64_t lhs;
  std::uint64_t rhs;
  std::uint64_t addend;
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
    fail(test, "accepted malformed fixed-vector FMA input");
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

std::string fabricSource(FixtureKind kind,
                         llvm::StringRef subnormal = "preserve",
                         llvm::StringRef schema = "math.fma") {
  const llvm::StringRef formats =
      kind == FixtureKind::Singleton || kind == FixtureKind::NonRne
          ? R"mlir(["f32"])mlir"
          : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const llvm::StringRef rounding = kind == FixtureKind::NonRne
                                       ? R"mlir(["upward"])mlir"
                                       : R"mlir(["to_nearest_even"])mlir";
  std::string text;
  llvm::raw_string_ostream source(text);
  source << R"mlir(module {
    fabric.module @fixed_vector_float_fma(
        %a: !fabric.bits<80>, %b: !fabric.bits<80>,
        %c: !fabric.bits<80>) -> !fabric.bits<80> {
      %pe = fabric.pe [spatial]
          (%pa = %a : !fabric.bits<80>, %pb = %b : !fabric.bits<80>,
           %pc = %c : !fabric.bits<80>) -> !fabric.bits<80> {
        %fu = fabric.fu
            (%fa = %pa : !fabric.bits<80>, %fb = %pb : !fabric.bits<80>,
             %fc = %pc : !fabric.bits<80>) -> !fabric.bits<80> {
          %value = fabric.op [@)mlir"
         << schema << R"mlir(] (%fa, %fb, %fc)
            {implementation_family =
               #fabric.implementation_family<FixedVectorFloatFma>,
             hw_params = {
               element_formats = )mlir"
         << formats << R"mlir(,
               behavior = {
                 rounding_modes = )mlir"
         << rounding << R"mlir(,
                 nan_behaviors = ["ieee"],
                 subnormal_behaviors = [")mlir"
         << subnormal << R"mlir("],
                 signed_zero_behaviors = ["preserve"],
                 fastmath = "none"},
               max_payload_bits = 80 : i32}}
            : (!fabric.bits<80>, !fabric.bits<80>, !fabric.bits<80>)
                -> !fabric.bits<80>
          fabric.yield %value : !fabric.bits<80>
        }
      }
      fabric.yield %pe : !fabric.bits<80>
    }
  })mlir";
  return source.str();
}

void attachContract(llvm::StringRef test, mlir::ModuleOp module,
                    FixtureKind kind) {
  const ::fabric::ResourceContract &contract =
      kind == FixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedBytes(encoded.begin(), encoded.end());
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedBytes));
  });
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FixtureKind kind = FixtureKind::Configured) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(kind),
                                                        &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  attachContract(test, *source, kind);
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
          ::fabric::ImplementationFamilyId::FixedVectorFloatFma)
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
              "System has no physical fixed-vector FMA occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector FMA occurrence");
}

void expectFabricRejected(llvm::StringRef test, const ArtifactStore &store,
                          llvm::StringRef subnormal, llvm::StringRef schema,
                          llvm::StringRef expected) {
  mlir::ParserConfig parserConfig(&fabricContext(), false);
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(FixtureKind::Configured, subnormal, schema), parserConfig);
  require(test, static_cast<bool>(source),
          "negative Fabric fixture did not parse");
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  if (mlir::failed(mlir::verify(*source))) {
    require(
        test,
        llvm::any_of(diagnostics,
                     [&](const std::string &diagnostic) {
                       return llvm::StringRef(diagnostic).contains(expected);
                     }),
        diagnostics.empty() ? "Fabric verifier produced no diagnostic"
                            : diagnostics.front());
    return;
  }
  attachContract(test, *source, FixtureKind::Configured);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  expectError(test, loom::fabric::finalizeFabricRoot(root, store), expected);
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

::fabric::FloatFormat
behaviorFormat(llvm::StringRef test,
               const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  auto vector = llvm::dyn_cast<mlir::VectorType>(
      point.representativeActor.type.getInput(0));
  require(test, vector && !vector.isScalable(),
          "Fabric projected a non-fixed-vector FMA behavior");
  mlir::Type type = vector.getElementType();
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "Fabric projected an unsupported FMA element format");
}

unsigned behaviorLaneCount(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  return static_cast<unsigned>(
      mlir::cast<mlir::VectorType>(point.representativeActor.type.getInput(0))
          .getNumElements());
}

std::uint8_t modeCode(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 5;
  case ::fabric::FloatFormat::BF16:
    return 1;
  case ::fabric::FloatFormat::F32:
    return 7;
  case ::fabric::FloatFormat::F64:
    return 3;
  }
  llvm_unreachable("unknown floating format");
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "fixed-vector FMA fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "fixed-vector FMA relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 4,
          "configured FMA domain is not the admitted format set");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured FMA behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F32)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {modeCode(format)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {6}});
  require(test, !inactive.empty(), "FMA domain has no inactive behavior");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{3}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{3, std::move(entries)}};
  if (kind == AbiKind::DirectEncoding)
    inactive = {0};
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, std::move(encoding), std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI
makeConfigurationAbi(llvm::StringRef test, const ArtifactStore &store,
                     const FabricFixture &fixture,
                     AbiKind kind = AbiKind::Complete) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture, kind), store));
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

SkeletonFixture
makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
             const FabricFixture &fabric, const ConfigurationABI &abi,
             bool wrongConfigurationWidth = false,
             llvm::StringRef moduleName = "fixed_vector_float_fma") {
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
  if (wrongConfigurationWidth) {
    auto field = llvm::find_if(
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

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFixedVectorFloatFmaProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

llvm::Expected<FabricOperationProviderOutput> trySpecialize(
    SkeletonFixture &skeleton, const FabricFixture &fabric,
    const FinalizedConfigurationABI &abi,
    FabricOperationProviderRegistry &registry,
    BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
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
          "portable FMA emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

FloatLayout layout(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return {5, 10};
  case ::fabric::FloatFormat::BF16:
    return {8, 7};
  case ::fabric::FloatFormat::F32:
    return {8, 23};
  case ::fabric::FloatFormat::F64:
    return {11, 52};
  }
  llvm_unreachable("unknown floating format");
}

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

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? std::numeric_limits<std::uint64_t>::max()
                     : (std::uint64_t{1} << width) - 1;
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint64_t bits) {
  return llvm::APFloat(semantics(format),
                       llvm::APInt(layout(format).width(), bits));
}

llvm::APFloat fused(::fabric::FloatFormat format, const TestVector &vector) {
  llvm::APFloat result = floating(format, vector.lhs);
  const llvm::APFloat rhs = floating(format, vector.rhs);
  const llvm::APFloat addend = floating(format, vector.addend);
  (void)result.fusedMultiplyAdd(rhs, addend,
                                llvm::RoundingMode::NearestTiesToEven);
  return result;
}

llvm::APFloat split(::fabric::FloatFormat format, const TestVector &vector) {
  llvm::APFloat result = floating(format, vector.lhs);
  const llvm::APFloat rhs = floating(format, vector.rhs);
  const llvm::APFloat addend = floating(format, vector.addend);
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

std::vector<TestVector> testVectors(::fabric::FloatFormat format) {
  const FloatLayout shape = layout(format);
  const std::uint64_t sign = shape.sign();
  const std::uint64_t one = shape.one();
  const std::uint64_t half = one - (std::uint64_t{1} << shape.fractionBits);
  const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
  const std::uint64_t infinity = shape.infinity();
  const std::uint64_t maximumFinite = infinity - 1;
  const std::uint64_t quietNaN = infinity | shape.quietBit() | 0x5;
  const std::uint64_t signalingNaN = infinity | 0x3;
  std::vector<TestVector> result = {
      {0, 0, 0},
      {sign, 0, sign},
      {one, one, one},
      {sign | one, one, one},
      {1, one, 0},
      {(std::uint64_t{1} << shape.fractionBits) - 1, one, 1},
      {infinity, 0, one},
      {infinity, one, sign | infinity},
      {sign | quietNaN, one, one},
      {one, signalingNaN, one},
      {one, one, sign | quietNaN},
      {sign | quietNaN, one, signalingNaN},
      {0, infinity, quietNaN},
      {1, half, (std::uint64_t{1} << shape.fractionBits) - 1},
      {sign | 1, half, 0},
      {maximumFinite, two, sign | maximumFinite},
  };
  const TestVector singleRound{one | 1, one | 1, sign | one | 2};
  require("testVectors",
          bits(fused(format, singleRound)) != bits(split(format, singleRound)),
          "FMA witness no longer distinguishes single rounding");
  result.push_back(singleRound);
  if (format == ::fabric::FloatFormat::F32) {
    require("testVectors", bits(fused(format, singleRound)) == 0x28800000,
            "f32 single-rounding witness changed");
    require("testVectors", bits(split(format, singleRound)) == 0,
            "f32 split witness no longer rounds to zero");
    result.push_back({0x00000001, 0x7f7fffff, 0xb5000000});
  }

  const std::uint64_t mask = widthMask(shape.width());
  std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ shape.width();
  unsigned fusedSensitive = 0;
  for (unsigned index = 0; index < 20000 && fusedSensitive < 4; ++index) {
    TestVector vector{nextRandom(state) & mask, nextRandom(state) & mask,
                      nextRandom(state) & mask};
    const llvm::APFloat fusedValue = fused(format, vector);
    const llvm::APFloat splitValue = split(format, vector);
    if (!fusedValue.isNaN() && !splitValue.isNaN() &&
        bits(fusedValue) != bits(splitValue)) {
      result.push_back(vector);
      ++fusedSensitive;
    }
  }
  require("testVectors", fusedSensitive == 4,
          "deterministic search did not find fused-sensitive vectors");
  for (unsigned index = 0; index < 16; ++index)
    result.push_back({nextRandom(state) & mask, nextRandom(state) & mask,
                      nextRandom(state) & mask});
  return result;
}

std::string hexLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned hexDigits = (value.getBitWidth() + 3) / 4;
  std::string padded(hexDigits > digits.size() ? hexDigits - digits.size() : 0,
                     '0');
  padded += digits.str();
  return std::to_string(value.getBitWidth()) + "'h" + padded;
}

std::string testbench() {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(module testbench;
  logic [79:0] data_input_0;
  logic [79:0] data_input_1;
  logic [79:0] data_input_2;
  logic [2:0] config_0;
  logic [79:0] data_output_0;
  logic [79:0] sibling_result;
  integer lane;

  fixed_vector_float_fma dut(.*);

  task automatic check_value(
      input logic [2:0] mode,
      input logic [79:0] lhs,
      input logic [79:0] rhs,
      input logic [79:0] addend,
      input logic [79:0] expected);
    begin
      config_0 = mode;
      data_input_0 = lhs;
      data_input_1 = rhs;
      data_input_2 = addend;
      #1;
      if (data_output_0 !== expected)
        $fatal(1, "fixed-vector FMA mismatch mode=%0d lhs=%h rhs=%h addend=%h got=%h expected=%h",
               mode, lhs, rhs, addend, data_output_0, expected);
    end
  endtask

  initial begin
)sv";

  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    const FloatLayout shape = layout(format);
    const unsigned lanes = 80 / shape.width();
    const unsigned activeBits = lanes * shape.width();
    const std::vector<TestVector> vectors = testVectors(format);
    for (std::size_t start = 0; start < vectors.size(); start += lanes) {
      llvm::APInt lhs(80, 0);
      llvm::APInt rhs(80, 0);
      llvm::APInt addend(80, 0);
      llvm::APInt expected(80, 0);
      for (unsigned lane = 0; lane < lanes; ++lane) {
        const TestVector &vector = vectors[(start + lane) % vectors.size()];
        const unsigned offset = lane * shape.width();
        lhs |= llvm::APInt(80, vector.lhs).shl(offset);
        rhs |= llvm::APInt(80, vector.rhs).shl(offset);
        addend |= llvm::APInt(80, vector.addend).shl(offset);
        expected |= llvm::APInt(80, bits(fused(format, vector))).shl(offset);
      }
      if (activeBits < 80) {
        const llvm::APInt padding =
            llvm::APInt::getHighBitsSet(80, 80 - activeBits);
        lhs |= padding;
        rhs |= padding;
        addend |= padding;
      }
      output << "    check_value(3'd" << unsigned(modeCode(format)) << ", "
             << hexLiteral(lhs) << ", " << hexLiteral(rhs) << ", "
             << hexLiteral(addend) << ", " << hexLiteral(expected) << ");\n";
    }
  }

  output << R"sv(
    config_0 = 3'd5;
    data_input_0 = {5{16'h3c00}};
    data_input_1 = {5{16'h3c00}};
    data_input_2 = {5{16'h3c00}};
    #1;
    sibling_result = data_output_0;
    data_input_0[32 +: 16] = 16'hffff;
    data_input_1[32 +: 16] = 16'hffff;
    data_input_2[32 +: 16] = 16'hffff;
    #1;
    for (lane = 0; lane < 5; lane = lane + 1)
      if (lane != 2 && data_output_0[lane * 16 +: 16] !==
                           sibling_result[lane * 16 +: 16])
        $fatal(1, "defined sibling FMA lane changed");

    config_0 = 3'd7;
    data_input_0 = {16'hffff, 32'h3f800000, 32'h3f800000};
    data_input_1 = {16'hffff, 32'h3f800000, 32'h3f800000};
    data_input_2 = {16'hffff, 32'h3f800000, 32'h3f800000};
    #1;
    if (data_output_0 !== {16'h0000, 32'h40000000, 32'h40000000})
      $fatal(1, "unused physical padding affected defined FMA lanes");
    config_0 = 3'd0;
    #1;
    if (data_output_0 !== {16'h0000, 32'h40000000, 32'h40000000})
      $fatal(1, "unused physical code did not select the inactive behavior");
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string yosysScript() {
  return R"ys(read_verilog -sv fixed_vector_float_fma.sv
hierarchy -check -top fixed_vector_float_fma
proc
opt
check -assert
select -assert-none fixed_vector_float_fma/t:$*ff* fixed_vector_float_fma/t:$*latch* fixed_vector_float_fma/t:$_*FF* fixed_vector_float_fma/t:$_*LATCH* fixed_vector_float_fma/t:$mem* fixed_vector_float_fma/m:*
synth -noabc -top fixed_vector_float_fma
check -assert
select -assert-none fixed_vector_float_fma/t:$*ff* fixed_vector_float_fma/t:$*latch* fixed_vector_float_fma/t:$_*FF* fixed_vector_float_fma/t:$_*LATCH* fixed_vector_float_fma/t:$mem* fixed_vector_float_fma/m:*
stat
)ys";
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorFloatFma;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog} &&
              llvm::count_if(coverage,
                             [](const auto &candidate) {
                               return !candidate.recipes.empty();
                             }) == 1,
          "fixed-vector FMA registration is not exact");

  FabricFixture fabric = makeFabric(test, store);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily ==
                  ::fabric::ImplementationFamilyId::FixedVectorFloatFma &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::MathFma},
          "FMA escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "FMA did not consume the exact sealed behavior domain");
  std::array<bool, 4> sawFormat{};
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test,
            point.representativeActor.schema ==
                    ::dataflow::OperationSchemaId::MathFma &&
                point.semanticConfiguration.has_value() &&
                point.operandPorts == std::vector<std::uint64_t>{0, 1, 2} &&
                point.resultPorts == std::vector<std::uint64_t>{0},
            "FMA relation contains a malformed behavior witness");
    const loom::CanonicalSemanticBytes projected =
        take(test, relation.projectSemanticValue(
                       point.representativeActor, point.operandPorts,
                       point.resultPorts, point.resolvedIndexWidth));
    require(test,
            projected.bytes().equals(point.semanticConfiguration->bytes()),
            "FMA behavior disagrees with its sealed semantic relation");
    const auto *payload = std::get_if<::dataflow::FloatingPointPayload>(
        &point.representativeActor.payload);
    require(test,
            payload && payload->flags == mlir::arith::FastMathFlags::none &&
                payload->roundingMode.value_or(
                    mlir::arith::RoundingMode::to_nearest_even) ==
                    mlir::arith::RoundingMode::to_nearest_even,
            "FMA relation escaped the strict RNE profile");
    const auto format = behaviorFormat(test, point);
    sawFormat[static_cast<std::size_t>(format)] = true;
    require(test, behaviorLaneCount(point) == 80 / layout(format).width(),
            "FMA relation changed fixed-vector lane geometry");
  }
  require(test, llvm::all_of(sawFormat, [](bool saw) { return saw; }),
          "FMA relation omitted an admitted element format");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  require(test, abi.abi().fabric() == fabric.system.reference(),
          "FMA ABI does not reference the exact System root");
  require(test,
          abi.abi().findOperationField(
              fabric.physicalOccurrence,
              resolved.configurationFieldSchema.front().ordinal) != nullptr,
          "FMA ABI has no occurrence-qualified operation field");
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture firstSkeleton =
      makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(firstSkeleton.leaf.getPortList());
  require(test,
          ports.size() == 5 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "data_input_2" &&
              ports.atInput(3).getName() == "config_0" &&
              ports.atInput(3).type ==
                  mlir::IntegerType::get(firstContext.get(), 3) &&
              ports.atOutput(0).getName() == "data_output_0",
          "FMA leaf ports do not follow ConfigurationABI 2.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical FMA inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("loom_fixed_vector_fma_e5_f10") &&
              rtl.contains("loom_fixed_vector_fma_e8_f7") &&
              rtl.contains("loom_fixed_vector_fma_e8_f23") &&
              rtl.contains("loom_fixed_vector_fma_e11_f52") &&
              !rtl.contains("shortreal") && !rtl.contains("real") &&
              !rtl.contains("DPI") && !rtl.contains_insensitive("poison") &&
              !rtl.contains_insensitive("trap") && !rtl.contains("valid_") &&
              !rtl.contains("ready_") && !rtl.contains("clock") &&
              !rtl.contains("reset"),
          "FMA RTL is incomplete or adds non-payload semantics");

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"fixed_vector_float_fma.sv", first},
           {"testbench.sv", testbench()},
           {"portable_fixed_vector_float_fma.ys", yosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture fabric = makeFabric(test, store, FixtureKind::Singleton);
  auto relation = take(
      test,
      capability(test, fabric).resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              relation.finiteBehaviorDomain().size() == 1 &&
              !relation.finiteBehaviorDomain().front().semanticConfiguration &&
              behaviorFormat(test, relation.finiteBehaviorDomain().front()) ==
                  ::fabric::FloatFormat::F32 &&
              behaviorLaneCount(relation.finiteBehaviorDomain().front()) == 2,
          "singleton FMA retained a configuration authority");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), false,
                   "fixed_vector_float_fma_singleton");
  require(test, skeleton.leaf.getPortList().size() == 4,
          "singleton FMA leaf retained a selector port");
  const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("loom_fixed_vector_fma_e8_f23") &&
              !llvm::StringRef(rtl).contains("config_0"),
          "singleton FMA emitted configurable or non-f32 logic");
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", {{"fixed_vector_float_fma_singleton.sv", rtl}}))
    fail(test, llvm::toString(std::move(error)));
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    BackendRecipeKey recipe) {
  require(test, !result, "unsupported FMA combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::FixedVectorFloatFma &&
            error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported FMA returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "FMA lost its typed Unsupported classification");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture valid = makeFabric(test, store);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *context, valid, validAbi.abi(), true);
  const std::string before = moduleText(*malformed.module);
  expectError(test, trySpecialize(malformed, valid, validAbi, registry),
              "leaf port");
  require(test, moduleText(*malformed.module) == before,
          "malformed FMA input partially mutated the skeleton");

  for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                       AbiKind::DirectEncoding})
    expectError(
        test,
        finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                 store),
        kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

  FabricFixture unsupported =
      makeFabric(test, store, FixtureKind::UnsupportedContract);
  FinalizedConfigurationABI unsupportedAbi =
      makeConfigurationAbi(test, store, unsupported);
  std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
  SkeletonFixture unsupportedSkeleton = makeSkeleton(
      test, *unsupportedContext, unsupported, unsupportedAbi.abi());
  const std::string unsupportedBefore = moduleText(*unsupportedSkeleton.module);
  expectTypedUnsupported(
      test,
      trySpecialize(unsupportedSkeleton, unsupported, unsupportedAbi, registry),
      BackendRecipeKey::PortableSystemVerilog);
  require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
          "Unsupported FMA mutated the caller skeleton");

  FabricFixture nonRne = makeFabric(test, store, FixtureKind::NonRne);
  FinalizedConfigurationABI nonRneAbi =
      makeConfigurationAbi(test, store, nonRne);
  std::unique_ptr<mlir::MLIRContext> nonRneContext = makeCirctContext();
  SkeletonFixture nonRneSkeleton =
      makeSkeleton(test, *nonRneContext, nonRne, nonRneAbi.abi());
  const std::string nonRneBefore = moduleText(*nonRneSkeleton.module);
  expectTypedUnsupported(
      test, trySpecialize(nonRneSkeleton, nonRne, nonRneAbi, registry),
      BackendRecipeKey::PortableSystemVerilog);
  require(test, moduleText(*nonRneSkeleton.module) == nonRneBefore,
          "unsupported non-RNE FMA mutated the caller skeleton");

  for (BackendRecipeKey recipe :
       {BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
        BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera}) {
    std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
    SkeletonFixture native =
        makeSkeleton(test, *nativeContext, valid, validAbi.abi());
    const std::string nativeBefore = moduleText(*native.module);
    expectTypedUnsupported(
        test, trySpecialize(native, valid, validAbi, registry, recipe), recipe);
    require(test, moduleText(*native.module) == nativeBefore,
            "unsupported native recipe mutated the caller skeleton");
  }

  expectFabricRejected(test, store, "flush_to_zero", "math.fma", "subnormal");
  expectFabricRejected(test, store, "preserve", "arith.mulf", "not admitted");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndArtifacts(root);
  singletonNeedsNoSelector(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}
