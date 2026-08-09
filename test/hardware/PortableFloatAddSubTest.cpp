#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatAddSub.h"
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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
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

enum class FamilyKind { Scalar, Vector };

enum class FixtureKind { Configured, Singleton, UnsupportedContract };

enum class ConfigurationAbiKind { Complete, MissingBehavior, ExtraBehavior };

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
    fail(test, "accepted invalid portable float add/sub input");
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
  FamilyKind family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarFloatAddSub
             : ::fabric::ImplementationFamilyId::FixedVectorFloatAddSub;
}

llvm::StringRef configuredFabricSource(FamilyKind family) {
  if (family == FamilyKind::Scalar)
    return R"mlir(
    module {
      fabric.module @scalar_float_add_sub(
          %a: !fabric.bits<64>, %b: !fabric.bits<64>)
          -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
            -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<64>,
               %fb = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
            %value = fabric.op [@arith.addf, @arith.subf] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatAddSub>,
               hw_params = {
                 float_formats = ["f16", "bf16", "f32", "f64"],
                 behavior = {
                   rounding_modes = ["to_nearest_even", "downward",
                                     "upward", "toward_zero",
                                     "to_nearest_away"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"}}}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir";
  return R"mlir(
    module {
      fabric.module @fixed_vector_float_add_sub(
          %a: !fabric.bits<96>, %b: !fabric.bits<96>)
          -> !fabric.bits<96> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<96>, %pb = %b : !fabric.bits<96>)
            -> !fabric.bits<96> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<96>,
               %fb = %pb : !fabric.bits<96>) -> !fabric.bits<96> {
            %value = fabric.op [@arith.addf, @arith.subf] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorFloatAddSub>,
               hw_params = {
                 element_formats = ["f16", "bf16", "f32", "f64"],
                 behavior = {
                   rounding_modes = ["to_nearest_even", "downward",
                                     "upward", "toward_zero",
                                     "to_nearest_away"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"},
                 max_payload_bits = 96 : i32}}
              : (!fabric.bits<96>, !fabric.bits<96>) -> !fabric.bits<96>
            fabric.yield %value : !fabric.bits<96>
          }
        }
        fabric.yield %pe : !fabric.bits<96>
      }
    }
  )mlir";
}

llvm::StringRef singletonFabricSource() {
  return R"mlir(
    module {
      fabric.module @scalar_float_add(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
            %value = fabric.op [@arith.addf] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatAddSub>,
               hw_params = {
                 float_formats = ["f32"],
                 behavior = {
                   rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"}}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir";
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FamilyKind family,
                         FixtureKind kind = FixtureKind::Configured) {
  require(test, kind != FixtureKind::Singleton || family == FamilyKind::Scalar,
          "only the scalar fixture has a singleton form");
  llvm::StringRef sourceText = kind == FixtureKind::Singleton
                                   ? singletonFabricSource()
                                   : configuredFabricSource(family);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const ::fabric::ResourceContract &resourceContract =
      kind == FixtureKind::UnsupportedContract
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
      if (capability.implementationFamily != familyId(family))
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
              "System has no physical float add/sub occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no float add/sub occurrence");
}

struct Mode final {
  ::fabric::FloatFormat format;
  mlir::arith::RoundingMode rounding;
  bool subtract;
  unsigned laneCount;
};

::fabric::FloatFormat formatOf(llvm::StringRef test, mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "Fabric projected an unsupported floating format");
}

Mode modeOf(llvm::StringRef test, FamilyKind family,
            const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  const bool add = point.representativeActor.schema ==
                   dataflow::OperationSchemaId::ArithAddF;
  const bool subtract = point.representativeActor.schema ==
                        dataflow::OperationSchemaId::ArithSubF;
  require(test, add || subtract,
          "Fabric projected a non-add/sub behavior point");
  const auto *payload = std::get_if<dataflow::FloatingPointPayload>(
      &point.representativeActor.payload);
  require(test, payload != nullptr,
          "Fabric projected add/sub without a floating payload");
  mlir::Type operand = point.representativeActor.type.getInput(0);
  unsigned laneCount = 1;
  mlir::Type element = operand;
  if (family == FamilyKind::Vector) {
    auto vector = mlir::dyn_cast<mlir::VectorType>(operand);
    require(test, static_cast<bool>(vector),
            "Fabric projected a non-vector behavior");
    require(test,
            vector.getNumElements() > 0 &&
                vector.getNumElements() <= std::numeric_limits<unsigned>::max(),
            "Fabric projected an invalid lane count");
    laneCount = static_cast<unsigned>(vector.getNumElements());
    element = vector.getElementType();
  }
  return Mode{formatOf(test, element),
              payload->roundingMode.value_or(
                  mlir::arith::RoundingMode::to_nearest_even),
              subtract, laneCount};
}

unsigned formatOrdinal(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 0;
  case ::fabric::FloatFormat::BF16:
    return 1;
  case ::fabric::FloatFormat::F32:
    return 2;
  case ::fabric::FloatFormat::F64:
    return 3;
  }
  llvm_unreachable("unknown float format");
}

unsigned roundingOrdinal(mlir::arith::RoundingMode rounding) {
  switch (rounding) {
  case mlir::arith::RoundingMode::to_nearest_even:
    return 0;
  case mlir::arith::RoundingMode::downward:
    return 1;
  case mlir::arith::RoundingMode::upward:
    return 2;
  case mlir::arith::RoundingMode::toward_zero:
    return 3;
  case mlir::arith::RoundingMode::to_nearest_away:
    return 4;
  }
  llvm_unreachable("unknown rounding mode");
}

std::uint8_t physicalCode(const Mode &mode) {
  const unsigned semanticIndex = (mode.subtract ? 20 : 0) +
                                 formatOrdinal(mode.format) * 5 +
                                 roundingOrdinal(mode.rounding);
  return static_cast<std::uint8_t>((semanticIndex * 17 + 11) & 0x3f);
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
          "float add/sub fixture has an unexpected field count");
  auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured float add/sub relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 40,
          "configured float add/sub domain is not the exact role product");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  entries.reserve(domain.size() + 1);
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured behavior has no semantic value");
    const Mode mode = modeOf(test, fixture.family, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (!mode.subtract && mode.format == ::fabric::FloatFormat::F32 &&
        mode.rounding == mlir::arith::RoundingMode::to_nearest_even)
      inactive = semantic;
    if (kind == ConfigurationAbiKind::MissingBehavior && mode.subtract &&
        mode.format == ::fabric::FloatFormat::BF16 &&
        mode.rounding == mlir::arith::RoundingMode::upward)
      continue;
    entries.push_back({std::move(semantic), {physicalCode(mode)}});
  }
  require(test,
          llvm::none_of(entries,
                        [](const FiniteCodebookEntry &entry) {
                          return entry.physicalCode.size() == 1 &&
                                 entry.physicalCode.front() == 0x3f;
                        }),
          "configured float add/sub codebook occupies its inactive fallback");
  if (kind == ConfigurationAbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x3f}});
  if (kind == ConfigurationAbiKind::MissingBehavior)
    require(test, entries.size() + 1 == domain.size(),
            "missing-behavior ABI did not omit exactly one domain point");
  if (kind == ConfigurationAbiKind::ExtraBehavior)
    require(test, entries.size() == domain.size() + 1,
            "extra-behavior ABI did not add exactly one foreign point");
  require(test, !inactive.empty(), "float add/sub domain has no inactive mode");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     capability->configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{6, std::move(entries)},
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
                             llvm::StringRef moduleName,
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
    const auto field = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, field != ports.end(), "configured leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName), ports);
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
  if (llvm::Error error = registerPortableFloatAddSubProviders(registry))
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
          "portable float add/sub emitted external implementation state");
  return std::move(conformance.systemVerilog);
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
  llvm_unreachable("unknown float format");
}

unsigned bitWidth(::fabric::FloatFormat format) {
  return ::fabric::getBitWidth(format);
}

unsigned fractionBits(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 10;
  case ::fabric::FloatFormat::BF16:
    return 7;
  case ::fabric::FloatFormat::F32:
    return 23;
  case ::fabric::FloatFormat::F64:
    return 52;
  }
  llvm_unreachable("unknown float format");
}

llvm::RoundingMode llvmRounding(mlir::arith::RoundingMode rounding) {
  switch (rounding) {
  case mlir::arith::RoundingMode::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case mlir::arith::RoundingMode::downward:
    return llvm::RoundingMode::TowardNegative;
  case mlir::arith::RoundingMode::upward:
    return llvm::RoundingMode::TowardPositive;
  case mlir::arith::RoundingMode::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case mlir::arith::RoundingMode::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  }
  llvm_unreachable("unknown rounding mode");
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint64_t value) {
  return llvm::APFloat(semantics(format), llvm::APInt(bitWidth(format), value));
}

std::uint64_t expected(const Mode &mode, std::uint64_t lhs, std::uint64_t rhs) {
  llvm::APFloat result = floating(mode.format, lhs);
  llvm::APFloat operand = floating(mode.format, rhs);
  if (mode.subtract)
    (void)result.subtract(operand, llvmRounding(mode.rounding));
  else
    (void)result.add(operand, llvmRounding(mode.rounding));
  return result.bitcastToAPInt().getZExtValue();
}

struct TestVector final {
  std::uint64_t lhs;
  std::uint64_t rhs;
};

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<TestVector> testVectors(::fabric::FloatFormat format) {
  const unsigned width = bitWidth(format);
  const unsigned fraction = fractionBits(format);
  const unsigned exponent = width - fraction - 1;
  const std::uint64_t sign = std::uint64_t{1} << (width - 1);
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponent) - 1;
  const std::uint64_t one = ((std::uint64_t{1} << (exponent - 1)) - 1)
                            << fraction;
  const std::uint64_t halfUlpAtOne =
      (std::uint64_t(((std::uint64_t{1} << (exponent - 1)) - 1) - fraction - 1)
       << fraction);
  const std::uint64_t infinity = exponentMask << fraction;
  const std::uint64_t maximumFinite = infinity - 1;
  const std::uint64_t maximumSubnormal = (std::uint64_t{1} << fraction) - 1;
  const std::uint64_t minimumNormal = std::uint64_t{1} << fraction;
  const std::uint64_t quietBit = std::uint64_t{1} << (fraction - 1);
  const std::uint64_t quietNaN = infinity | quietBit | 0x5;
  const std::uint64_t negativeQuietNaN = sign | quietNaN;
  const std::uint64_t signalingNaN = infinity | 0x3;
  const std::uint64_t negativeSignalingNaN = sign | signalingNaN;
  std::vector<TestVector> result = {
      {0, 0},
      {sign, sign},
      {0, sign},
      {sign, 0},
      {one, one},
      {one, sign | one},
      {one, halfUlpAtOne},
      {one | 1, halfUlpAtOne},
      {one, 1},
      {one, sign | 1},
      {1, 1},
      {minimumNormal, sign | maximumSubnormal},
      {maximumSubnormal, 1},
      {maximumFinite, maximumFinite},
      {sign | maximumFinite, sign | maximumFinite},
      {maximumFinite, sign | maximumFinite},
      {sign | maximumFinite, maximumFinite},
      {infinity, one},
      {one, infinity},
      {infinity, infinity},
      {infinity, sign | infinity},
      {quietNaN, one},
      {negativeQuietNaN, one},
      {one, negativeQuietNaN},
      {signalingNaN, one},
      {one, negativeSignalingNaN},
      {negativeQuietNaN, signalingNaN},
  };
  std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ width;
  const std::uint64_t mask = widthMask(width);
  for (unsigned index = 0; index < 8; ++index)
    result.push_back({nextRandom(state) & mask, nextRandom(state) & mask});
  return result;
}

std::string hexLiteral(unsigned width, const llvm::APInt &value) {
  llvm::SmallString<32> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned digitCount = (width + 3) / 4;
  return std::to_string(width) + "'h" +
         std::string(digitCount - digits.size(), '0') + digits.str().str();
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  return hexLiteral(width, llvm::APInt(width, value));
}

std::uint64_t paddedScalarInput(::fabric::FloatFormat format,
                                std::uint64_t value, std::uint64_t padding) {
  const unsigned width = bitWidth(format);
  return value | (padding & ~widthMask(width));
}

std::vector<Mode> projectedModes(llvm::StringRef test,
                                 const FabricFixture &fixture) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  std::vector<Mode> modes;
  modes.reserve(relation.finiteBehaviorDomain().size());
  for (const auto &point : relation.finiteBehaviorDomain()) {
    if (relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::None) {
      require(test,
              !point.semanticConfiguration &&
                  relation.finiteBehaviorDomain().size() == 1,
              "fieldless relation is not a carrier-free singleton");
    } else {
      const loom::CanonicalSemanticBytes projected =
          take(test, relation.projectSemanticValue(
                         point.representativeActor, point.operandPorts,
                         point.resultPorts, point.resolvedIndexWidth));
      require(
          test,
          point.semanticConfiguration &&
              point.semanticConfiguration->bytes().equals(projected.bytes()),
          "sealed relation disagrees with its projected behavior key");
    }
    modes.push_back(modeOf(test, fixture.family, point));
  }
  return modes;
}

struct EmittedProvider final {
  std::string systemVerilog;
  std::vector<Mode> modes;
};

EmittedProvider emitProvider(llvm::StringRef test, const ArtifactStore &store,
                             FamilyKind family, FixtureKind kind,
                             llvm::StringRef moduleName) {
  FabricFixture fabric = makeFabric(test, store, family, kind);
  std::vector<Mode> modes = projectedModes(test, fabric);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), moduleName);
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  const bool configured = kind != FixtureKind::Singleton;
  require(test,
          ports.size() == (configured ? 4 : 3) &&
              ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived float add/sub leaf ports are not canonical");
  if (configured)
    require(test,
            ports.atInput(2).getName() == "config_0" &&
                ports.atInput(2).type ==
                    mlir::IntegerType::get(context.get(), 6),
            "configured float add/sub selector is not the finalized ABI field");
  std::string first = specialize(test, skeleton, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), moduleName);
  const std::string second = specialize(test, secondSkeleton, fabric, abi);
  require(test, first == second,
          "identical float add/sub inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") &&
              rtl.contains("loom_float_add_sub") &&
              rtl.contains("config_0") == configured &&
              !rtl.contains("shortreal") && !rtl.contains("real") &&
              !rtl.contains("DPI"),
          "portable float add/sub is not self-contained synthesizable RTL");
  return EmittedProvider{std::move(first), std::move(modes)};
}

std::string buildTestbench(llvm::ArrayRef<Mode> scalarModes,
                           llvm::ArrayRef<Mode> vectorModes) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(
module testbench;
  logic [63:0] scalar_lhs;
  logic [63:0] scalar_rhs;
  logic [5:0] scalar_config;
  logic [63:0] scalar_result;
  logic [95:0] vector_lhs;
  logic [95:0] vector_rhs;
  logic [5:0] vector_config;
  logic [95:0] vector_result;
  logic [31:0] singleton_lhs;
  logic [31:0] singleton_rhs;
  logic [31:0] singleton_result;

`ifdef TEST_SCALAR
  scalar_float_add_sub scalar_dut(
      .data_input_0(scalar_lhs),
      .data_input_1(scalar_rhs),
      .config_0(scalar_config),
      .data_output_0(scalar_result));
`endif
`ifdef TEST_VECTOR
  fixed_vector_float_add_sub vector_dut(
      .data_input_0(vector_lhs),
      .data_input_1(vector_rhs),
      .config_0(vector_config),
      .data_output_0(vector_result));
`endif
`ifdef TEST_SINGLETON
  scalar_float_add_singleton singleton_dut(
      .data_input_0(singleton_lhs),
      .data_input_1(singleton_rhs),
      .data_output_0(singleton_result));
`endif

`ifdef TEST_SCALAR
  task automatic check_scalar(
      input logic [5:0] mode,
      input logic [63:0] lhs,
      input logic [63:0] rhs,
      input logic [63:0] expected);
    begin
      scalar_config = mode;
      scalar_lhs = lhs;
      scalar_rhs = rhs;
      #1;
      if (scalar_result !== expected)
        $fatal(1, "scalar mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h",
               mode, lhs, rhs, scalar_result, expected);
    end
  endtask
`endif

`ifdef TEST_VECTOR
  task automatic check_vector(
      input logic [5:0] mode,
      input logic [95:0] lhs,
      input logic [95:0] rhs,
      input logic [95:0] expected);
    begin
      vector_config = mode;
      vector_lhs = lhs;
      vector_rhs = rhs;
      #1;
      if (vector_result !== expected)
        $fatal(1, "vector mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h",
               mode, lhs, rhs, vector_result, expected);
    end
  endtask
`endif

  initial begin
)sv";

  output << "`ifdef TEST_SCALAR\n";
  for (const Mode &mode : scalarModes) {
    for (const TestVector &vector : testVectors(mode.format)) {
      output << "    check_scalar(6'd" << unsigned(physicalCode(mode)) << ", "
             << hexLiteral(64, paddedScalarInput(mode.format, vector.lhs,
                                                 0xa5a5a5a5a5a5a5a5ULL))
             << ", "
             << hexLiteral(64, paddedScalarInput(mode.format, vector.rhs,
                                                 0x5a5a5a5a5a5a5a5aULL))
             << ", " << hexLiteral(64, expected(mode, vector.lhs, vector.rhs))
             << ");\n";
    }
  }
  output << "`endif\n`ifdef TEST_VECTOR\n";

  for (const Mode &mode : vectorModes) {
    const unsigned width = bitWidth(mode.format);
    const std::vector<TestVector> vectors = testVectors(mode.format);
    llvm::APInt lhs(96, 0);
    llvm::APInt rhs(96, 0);
    llvm::APInt result(96, 0);
    for (unsigned lane = 0; lane < mode.laneCount; ++lane) {
      const TestVector vector =
          vectors[(lane * 5 + physicalCode(mode)) % vectors.size()];
      lhs.insertBits(llvm::APInt(width, vector.lhs), lane * width);
      rhs.insertBits(llvm::APInt(width, vector.rhs), lane * width);
      result.insertBits(
          llvm::APInt(width, expected(mode, vector.lhs, vector.rhs)),
          lane * width);
    }
    const unsigned payloadWidth = mode.laneCount * width;
    if (payloadWidth < 96) {
      lhs.setBitsFrom(payloadWidth);
      rhs.setBitsFrom(payloadWidth);
    }
    output << "    check_vector(6'd" << unsigned(physicalCode(mode)) << ", "
           << hexLiteral(96, lhs) << ", " << hexLiteral(96, rhs) << ", "
           << hexLiteral(96, result) << ");\n";
  }
  output << "`endif\n";

  const Mode inactive{::fabric::FloatFormat::F32,
                      mlir::arith::RoundingMode::to_nearest_even, false, 1};
  const std::vector<TestVector> inactiveVectors = {
      {0x3f800000, 0x33800000},
      {0xbf800000, 0xb3800000},
      {0x3f800000, 0x33800001},
  };
  llvm::APInt inactiveLhs(96, 0);
  llvm::APInt inactiveRhs(96, 0);
  llvm::APInt inactiveResult(96, 0);
  output << "`ifdef TEST_SCALAR\n";
  for (unsigned lane = 0; lane < inactiveVectors.size(); ++lane) {
    const TestVector vector = inactiveVectors[lane];
    const std::uint64_t result = expected(inactive, vector.lhs, vector.rhs);
    output << "    check_scalar(6'd63, " << hexLiteral(64, vector.lhs) << ", "
           << hexLiteral(64, vector.rhs) << ", " << hexLiteral(64, result)
           << ");\n";
    inactiveLhs.insertBits(llvm::APInt(32, vector.lhs), lane * 32);
    inactiveRhs.insertBits(llvm::APInt(32, vector.rhs), lane * 32);
    inactiveResult.insertBits(llvm::APInt(32, result), lane * 32);
  }
  output << "`endif\n`ifdef TEST_VECTOR\n";
  output << "    check_vector(6'd63, " << hexLiteral(96, inactiveLhs) << ", "
         << hexLiteral(96, inactiveRhs) << ", "
         << hexLiteral(96, inactiveResult) << ");\n";
  output << "`endif\n";
  output << R"sv(
`ifdef TEST_SINGLETON
    singleton_lhs = 32'h3f800000;
    singleton_rhs = 32'h40000000;
    #1;
    if (singleton_result !== 32'h40400000)
      $fatal(1, "configuration-free f32 add produced the wrong result");
`endif
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string yosysScript(llvm::StringRef top, llvm::StringRef source) {
  std::string script;
  llvm::raw_string_ostream output(script);
  output << "read_verilog -sv " << source << '\n'
         << "hierarchy -check -top " << top << '\n'
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none t:$*ff* t:$*latch* t:$_*FF* "
            "t:$_*LATCH* t:$mem*\n"
         << "synth -noabc -top " << top << '\n'
         << "check -assert\n"
         << "select -assert-none t:$*ff* t:$*latch* t:$_*FF* "
            "t:$_*LATCH* t:$mem*\n"
         << "stat\n";
  return output.str();
}

template <typename T>
void requireUnsupported(llvm::StringRef test, llvm::Expected<T> result,
                        ::fabric::ImplementationFamilyId family) {
  require(test, !result, "unsupported provider request succeeded");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == family &&
                     error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported request returned the wrong error class: " +
                       error.message());
      });
  require(test, classified,
          "unsupported request lost its typed classification");
}

llvm::Expected<FabricOperationProviderOutput>
unusedProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

void providerRegistrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add(
          {::fabric::ImplementationFamilyId::FixedVectorFloatAddSub,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           unusedProvider}))
    fail(test, llvm::toString(std::move(error)));
  const auto before = registry.coverage();
  llvm::Error error = registerPortableFloatAddSubProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate vector provider registration succeeded");
  llvm::consumeError(std::move(error));
  const auto after = registry.coverage();
  require(test, before.size() == after.size(),
          "failed aggregate registration changed the coverage domain");
  for (auto [oldEntry, newEntry] : llvm::zip(before, after))
    require(test,
            oldEntry.implementationFamily == newEntry.implementationFamily &&
                oldEntry.recipes == newEntry.recipes,
            "failed aggregate registration changed provider coverage");
  const auto scalar = llvm::find_if(after, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarFloatAddSub;
  });
  const auto vector = llvm::find_if(after, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorFloatAddSub;
  });
  require(test,
          scalar != after.end() && scalar->recipes.empty() &&
              vector != after.end() &&
              vector->recipes ==
                  std::vector{BackendRecipeKey::PortableSystemVerilog},
          "failed aggregate registration leaked the scalar provider");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());

  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::Vector}) {
    FabricFixture fixture = makeFabric(test, store, family);
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture wrongPorts = makeSkeleton(
        test, *context, fixture, abi.abi(),
        family == FamilyKind::Scalar ? "invalid_scalar" : "invalid_vector",
        true);
    const std::string before = moduleText(*wrongPorts.module);
    FabricOperationProviderRegistry registry;
    if (llvm::Error error = registerPortableFloatAddSubProviders(registry))
      fail(test, llvm::toString(std::move(error)));
    ExternalImplementationContractCatalog externalContracts;
    const std::vector<FabricOperationLeafAssociation> associations = {
        {wrongPorts.leaf, fixture.physicalOccurrence}};
    const std::vector<FabricOperationRecipeBinding> recipes = {
        {fixture.physicalOccurrence,
         BackendRecipeKey::PortableSystemVerilog,
         {}}};
    expectError(test,
                specializeFabricOperationLeaves(*wrongPorts.module, abi,
                                                associations, recipes, registry,
                                                externalContracts),
                "leaf port");
    require(test, moduleText(*wrongPorts.module) == before,
            "malformed leaf partially mutated the caller skeleton");

    expectError(test,
                finalizeConfigurationABI(
                    makeConfigurationAbiDraft(
                        test, fixture, ConfigurationAbiKind::MissingBehavior),
                    store),
                "finite codebook does not equal its Fabric relation");
    expectError(test,
                finalizeConfigurationABI(
                    makeConfigurationAbiDraft(
                        test, fixture, ConfigurationAbiKind::ExtraBehavior),
                    store),
                "semantic value is outside the finite behavior domain");
  }

  mlir::Type f32 = mlir::Float32Type::get(&fabricContext());
  const ::dataflow::CanonicalActorSchemaProjection actor{
      ::dataflow::OperationSchemaId::ArithAddF,
      mlir::FunctionType::get(&fabricContext(), {f32, f32}, {f32}),
      ::dataflow::FloatingPointPayload{mlir::arith::FastMathFlags::none,
                                       std::nullopt}};
  ::fabric::FloatBehaviorProfile behavior =
      ::fabric::FloatBehaviorProfile::strictIEEE();
  behavior.subnormalBehaviors = ::fabric::FloatSubnormalBehaviorSet::get(
      {::fabric::FloatSubnormalBehavior::FlushToZero});
  const ::fabric::FamilyCapabilityParams parameters =
      ::fabric::ScalarFloatParams{
          ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F32}),
          behavior};
  llvm::Error profileError = ::fabric::verifyImplementationFamilyAdmission(
      ::fabric::ImplementationFamilyId::ScalarFloatAddSub, &parameters, actor);
  require(test, static_cast<bool>(profileError),
          "flush-to-zero profile escaped Fabric finalization");
  const std::string profileMessage = llvm::toString(std::move(profileError));
  require(test, llvm::StringRef(profileMessage).contains("subnormal"),
          profileMessage);
}

void unsupportedRequestsRollBack(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());

  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::Vector}) {
    FabricFixture fixture =
        makeFabric(test, store, family, FixtureKind::UnsupportedContract);
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, fixture, abi.abi(),
                     family == FamilyKind::Scalar ? "unsupported_scalar"
                                                  : "unsupported_vector");
    FabricOperationProviderRegistry registry;
    if (llvm::Error error = registerPortableFloatAddSubProviders(registry))
      fail(test, llvm::toString(std::move(error)));
    ExternalImplementationContractCatalog externalContracts;
    ModuleRootCirctSkeleton module{
        std::move(skeleton.module),
        {{skeleton.leaf, fixture.physicalOccurrence}}};
    auto result = loom::hardware::test::specializeAndExportPortableProvider(
        std::move(module), abi, registry, externalContracts);
    requireUnsupported(test, std::move(result), familyId(family));
  }

  FabricFixture fixture = makeFabric(test, store, FamilyKind::Scalar);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fixture, abi.abi(), "unsupported_recipe");
  const std::string before = moduleText(*skeleton.module);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatAddSubProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fixture.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fixture.physicalOccurrence, BackendRecipeKey::SynopsysDesignWare, {}}};
  auto result =
      specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                      recipes, registry, externalContracts);
  require(test, !result, "unregistered native recipe specialized");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() ==
                         ::fabric::ImplementationFamilyId::ScalarFloatAddSub &&
                     error.recipe() == BackendRecipeKey::SynopsysDesignWare;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "native recipe returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "native recipe lost its typed classification");
  require(test, moduleText(*skeleton.module) == before,
          "unsupported recipe partially mutated the caller skeleton");
}

void emitConformanceArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  EmittedProvider scalar =
      emitProvider(test, store, FamilyKind::Scalar, FixtureKind::Configured,
                   "scalar_float_add_sub");
  EmittedProvider vector =
      emitProvider(test, store, FamilyKind::Vector, FixtureKind::Configured,
                   "fixed_vector_float_add_sub");
  EmittedProvider singleton =
      emitProvider(test, store, FamilyKind::Scalar, FixtureKind::Singleton,
                   "scalar_float_add_singleton");
  require(test,
          scalar.modes.size() == 40 && vector.modes.size() == 40 &&
              singleton.modes.size() == 1,
          "provider did not consume the complete sealed behavior domains");
  const Mode singletonMode = singleton.modes.front();
  require(test,
          !singletonMode.subtract &&
              singletonMode.format == ::fabric::FloatFormat::F32 &&
              singletonMode.rounding ==
                  mlir::arith::RoundingMode::to_nearest_even,
          "configuration-free provider consumed the wrong sealed behavior");
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_float_add_sub.sv", std::move(scalar.systemVerilog)},
           {"fixed_vector_float_add_sub.sv", std::move(vector.systemVerilog)},
           {"scalar_float_add_singleton.sv",
            std::move(singleton.systemVerilog)},
           {"testbench.sv", buildTestbench(scalar.modes, vector.modes)},
           {"portable_scalar_float_add_sub.ys",
            yosysScript("scalar_float_add_sub", "scalar_float_add_sub.sv")},
           {"portable_fixed_vector_float_add_sub.ys",
            yosysScript("fixed_vector_float_add_sub",
                        "fixed_vector_float_add_sub.sv")},
           {"portable_scalar_float_add_singleton.ys",
            yosysScript("scalar_float_add_singleton",
                        "scalar_float_add_singleton.sv")}}))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  emitConformanceArtifacts(root);
  malformedInputsFailClosed(root / "malformed");
  unsupportedRequestsRollBack(root / "unsupported");
  providerRegistrationIsTransactional();
  return EXIT_SUCCESS;
}
