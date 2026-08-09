#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatConversions.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "Simulator/OperationSemantics.h"

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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

enum class ConversionFamily { WidthCast, IntegerToFloat, FloatToInteger };
enum class FixtureKind {
  Configured,
  Singleton,
  OrdinarySignedSingleton,
  OrdinaryUnsignedSingleton,
  TransactionPair,
  UnsupportedContract
};

struct FabricFixture final {
  ConversionFamily family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  std::vector<FabricFuOccurrenceNodeRef> occurrences;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>
      physicalOccurrences;
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
  std::uint64_t one() const {
    return std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
           << fractionBits;
  }
};

struct ModeInfo final {
  ConversionFamily family;
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::uint8_t physicalCode;
  unsigned inputWidth;
  unsigned outputWidth;
  std::optional<::fabric::FloatFormat> sourceFormat;
  std::optional<::fabric::FloatFormat> destinationFormat;
  bool signedInteger = false;
  mlir::arith::RoundingMode rounding =
      mlir::arith::RoundingMode::to_nearest_even;
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
void expectInvalid(llvm::StringRef test, llvm::Expected<T> value,
                   llvm::StringRef expected) {
  require(test, !value, "accepted malformed float conversion input");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed float conversion became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        classified = true;
      });
  require(test, classified, "malformed float conversion lost its error");
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            ::fabric::ImplementationFamilyId expectedFamily,
                            BackendRecipeKey expectedRecipe,
                            llvm::StringRef description) {
  require(test, !value, description);
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == expectedFamily &&
                     error.recipe() == expectedRecipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, classified,
          description.str() + " lost typed Unsupported classification");
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

::fabric::ImplementationFamilyId familyId(ConversionFamily family) {
  switch (family) {
  case ConversionFamily::WidthCast:
    return ::fabric::ImplementationFamilyId::ScalarFloatWidthCast;
  case ConversionFamily::IntegerToFloat:
    return ::fabric::ImplementationFamilyId::ScalarIntegerToFloat;
  case ConversionFamily::FloatToInteger:
    return ::fabric::ImplementationFamilyId::ScalarFloatToInteger;
  }
  llvm_unreachable("unknown conversion family");
}

llvm::StringRef familyKeyword(ConversionFamily family) {
  switch (family) {
  case ConversionFamily::WidthCast:
    return "ScalarFloatWidthCast";
  case ConversionFamily::IntegerToFloat:
    return "ScalarIntegerToFloat";
  case ConversionFamily::FloatToInteger:
    return "ScalarFloatToInteger";
  }
  llvm_unreachable("unknown conversion family");
}

llvm::StringRef moduleName(ConversionFamily family) {
  switch (family) {
  case ConversionFamily::WidthCast:
    return "scalar_float_width_cast";
  case ConversionFamily::IntegerToFloat:
    return "scalar_integer_to_float";
  case ConversionFamily::FloatToInteger:
    return "scalar_float_to_integer";
  }
  llvm_unreachable("unknown conversion family");
}

std::string fabricSource(ConversionFamily family, FixtureKind kind) {
  if (kind == FixtureKind::TransactionPair)
    return R"mlir(module {
      fabric.module @two_float_width_casts(
          %input0: !fabric.bits<67>, %input1: !fabric.bits<67>)
          -> (!fabric.bits<67>, !fabric.bits<67>) {
        %pe0 = fabric.pe [spatial]
            (%pin0 = %input0 : !fabric.bits<67>) -> !fabric.bits<67> {
          %fu0 = fabric.fu
              (%fin0 = %pin0 : !fabric.bits<67>) -> !fabric.bits<67> {
            %value0 = fabric.op [@arith.extf] (%fin0)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatWidthCast>,
               hw_params = {format_pairs = [["f16", "f32"]],
                 behavior = {rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"], fastmath = "none"}}}
              : (!fabric.bits<67>) -> !fabric.bits<67>
            fabric.yield %value0 : !fabric.bits<67>
          }
        }
        %pe1 = fabric.pe [spatial]
            (%pin1 = %input1 : !fabric.bits<67>) -> !fabric.bits<67> {
          %fu1 = fabric.fu
              (%fin1 = %pin1 : !fabric.bits<67>) -> !fabric.bits<67> {
            %value1 = fabric.op [@arith.extf] (%fin1)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatWidthCast>,
               hw_params = {format_pairs = [["f16", "f32"]],
                 behavior = {rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"], fastmath = "none"}}}
              : (!fabric.bits<67>) -> !fabric.bits<67>
            fabric.yield %value1 : !fabric.bits<67>
          }
        }
        fabric.yield %pe0, %pe1 : !fabric.bits<67>, !fabric.bits<67>
      }
    })mlir";
  const bool singleton = kind == FixtureKind::Singleton ||
                         kind == FixtureKind::OrdinarySignedSingleton ||
                         kind == FixtureKind::OrdinaryUnsignedSingleton;
  llvm::StringRef schemas;
  llvm::StringRef params;
  switch (family) {
  case ConversionFamily::WidthCast:
    schemas = singleton ? "@arith.extf" : "@arith.extf, @arith.truncf";
    params =
        singleton
            ? R"mlir(format_pairs = [["f16", "f32"]], behavior = {rounding_modes = ["to_nearest_even"], nan_behaviors = ["ieee"], subnormal_behaviors = ["preserve"], signed_zero_behaviors = ["preserve"], fastmath = "none"})mlir"
            : R"mlir(format_pairs = [["f16", "f32"], ["bf16", "f32"], ["bf16", "f64"], ["f32", "f16"], ["f32", "bf16"], ["f64", "bf16"]], behavior = {rounding_modes = ["to_nearest_even", "downward", "upward", "toward_zero", "to_nearest_away"], nan_behaviors = ["ieee"], subnormal_behaviors = ["preserve"], signed_zero_behaviors = ["preserve"], fastmath = "none"})mlir";
    break;
  case ConversionFamily::IntegerToFloat:
    schemas = singleton ? "@arith.sitofp" : "@arith.sitofp, @arith.uitofp";
    params =
        singleton
            ? R"mlir(format_pairs = [[8 : i32, "f16"]])mlir"
            : R"mlir(format_pairs = [[8 : i32, "f16"], [16 : i32, "bf16"], [32 : i32, "f16"], [32 : i32, "f32"], [64 : i32, "f64"]])mlir";
    break;
  case ConversionFamily::FloatToInteger:
    if (kind == FixtureKind::OrdinarySignedSingleton)
      schemas = "@arith.fptosi";
    else if (kind == FixtureKind::OrdinaryUnsignedSingleton)
      schemas = "@arith.fptoui";
    else
      schemas = singleton ? "@llvm.fptosi.sat"
                          : "@arith.fptosi, @arith.fptoui, @llvm.fptosi.sat, "
                            "@llvm.fptoui.sat";
    params =
        singleton
            ? R"mlir(format_pairs = [[16 : i32, "f32"]])mlir"
            : R"mlir(format_pairs = [[8 : i32, "f16"], [8 : i32, "f64"], [16 : i32, "bf16"], [32 : i32, "f32"], [64 : i32, "f64"]])mlir";
    break;
  }

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%input: !fabric.bits<67>) -> !fabric.bits<67> { "
            "%pe = fabric.pe [spatial](%pin = %input : !fabric.bits<67>) "
            "-> !fabric.bits<67> { %fu = fabric.fu"
            "(%fin = %pin : !fabric.bits<67>) -> !fabric.bits<67> { "
            "%value = fabric.op ["
         << schemas << "] (%fin) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {" << params
         << "}} : (!fabric.bits<67>) -> !fabric.bits<67> "
            "fabric.yield %value : !fabric.bits<67> } } "
            "fabric.yield %pe : !fabric.bits<67> } }";
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
                         ConversionFamily family,
                         FixtureKind kind = FixtureKind::Configured) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(family, kind), &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  attachContract(test, *source, kind);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  std::vector<FabricFuOccurrenceNodeRef> occurrences;
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != familyId(family))
        continue;
      occurrences.push_back(
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence)));
    }
  }
  const std::size_t expectedCount =
      kind == FixtureKind::TransactionPair ? 2 : 1;
  require(test, occurrences.size() == expectedCount,
          "Fabric fixture has the wrong conversion occurrence count");
  FinalizedFabricRoot system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(fabric, store));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  auto operations = take(test, enumerateFabricPhysicalOperations(systemView));
  std::vector<FabricFuOccurrenceNodeRef> physicalLocalOccurrences;
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>
      physicalOccurrences;
  for (const auto &operation : operations) {
    if (operation.capability->implementationFamily == familyId(family)) {
      physicalLocalOccurrences.push_back(operation.localOccurrence);
      physicalOccurrences.push_back(operation.physicalOccurrence);
    }
  }
  require(test, physicalOccurrences.size() == expectedCount,
          "System has the wrong physical conversion occurrence count");
  FabricFuOccurrenceNodeRef occurrence = physicalLocalOccurrences.front();
  auto physicalOccurrence = physicalOccurrences.front();
  return FabricFixture{family,
                       std::move(fabric),
                       occurrence,
                       std::move(physicalLocalOccurrences),
                       std::move(system),
                       physicalOccurrence,
                       std::move(physicalOccurrences)};
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

std::uint8_t physicalCode(std::size_t index) {
  return static_cast<std::uint8_t>((7 + 13 * index) & 0x3f);
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

::fabric::FloatFormat floatFormat(llvm::StringRef test, mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "sealed conversion behavior has an unsupported float type");
}

unsigned integerWidth(llvm::StringRef test, mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  require(test, static_cast<bool>(integer),
          "sealed conversion behavior has a non-integer endpoint");
  return integer.getWidth();
}

llvm::RoundingMode llvmRounding(mlir::arith::RoundingMode rounding) {
  using Arith = mlir::arith::RoundingMode;
  switch (rounding) {
  case Arith::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case Arith::downward:
    return llvm::RoundingMode::TowardNegative;
  case Arith::upward:
    return llvm::RoundingMode::TowardPositive;
  case Arith::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case Arith::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  }
  llvm_unreachable("unknown floating rounding mode");
}

ModeInfo
describeMode(llvm::StringRef test, ConversionFamily family,
             const ::fabric::FiniteImplementationFamilyBehaviorPoint &point,
             std::uint8_t code) {
  const auto &actor = point.representativeActor;
  require(test,
          actor.type.getNumInputs() == 1 && actor.type.getNumResults() == 1,
          "sealed conversion behavior is not unary");
  ModeInfo mode{family, actor, code, 0, 0, std::nullopt, std::nullopt};
  using Schema = ::dataflow::OperationSchemaId;
  switch (family) {
  case ConversionFamily::WidthCast: {
    mode.sourceFormat = floatFormat(test, actor.type.getInput(0));
    mode.destinationFormat = floatFormat(test, actor.type.getResult(0));
    mode.inputWidth = layout(*mode.sourceFormat).width();
    mode.outputWidth = layout(*mode.destinationFormat).width();
    const auto *payload =
        std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
    require(test, payload != nullptr,
            "sealed width cast has no floating-point payload");
    mode.rounding = payload->roundingMode.value_or(
        mlir::arith::RoundingMode::to_nearest_even);
    break;
  }
  case ConversionFamily::IntegerToFloat:
    mode.inputWidth = integerWidth(test, actor.type.getInput(0));
    mode.destinationFormat = floatFormat(test, actor.type.getResult(0));
    mode.outputWidth = layout(*mode.destinationFormat).width();
    mode.signedInteger = actor.schema == Schema::ArithSIToFP;
    break;
  case ConversionFamily::FloatToInteger:
    mode.sourceFormat = floatFormat(test, actor.type.getInput(0));
    mode.inputWidth = layout(*mode.sourceFormat).width();
    mode.outputWidth = integerWidth(test, actor.type.getResult(0));
    mode.signedInteger = actor.schema == Schema::ArithFPToSI ||
                         actor.schema == Schema::LLVMFPToSISat;
    break;
  }
  return mode;
}

std::vector<ModeInfo> configuredModes(llvm::StringRef test,
                                      const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured conversion relation is not finite");
  std::vector<ModeInfo> modes;
  for (auto [index, point] : llvm::enumerate(relation.finiteBehaviorDomain()))
    modes.push_back(
        describeMode(test, fixture.family, point, physicalCode(index)));
  return modes;
}

ModeInfo singletonMode(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto domain = relation.finiteBehaviorDomain();
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              domain.size() == 1 && !domain.front().semanticConfiguration,
          "singleton conversion relation is not fieldless");
  return describeMode(test, fixture.family, domain.front(), 0);
}

llvm::APInt evaluateMode(llvm::StringRef test, const ModeInfo &mode,
                         const llvm::APInt &input) {
  require(test, input.getBitWidth() == mode.inputWidth,
          "oracle input has the wrong semantic width");
  switch (mode.family) {
  case ConversionFamily::WidthCast: {
    llvm::APFloat value(semantics(*mode.sourceFormat), input);
    bool losesInfo = false;
    (void)value.convert(semantics(*mode.destinationFormat),
                        llvmRounding(mode.rounding), &losesInfo);
    return value.bitcastToAPInt();
  }
  case ConversionFamily::IntegerToFloat: {
    llvm::APFloat value =
        llvm::APFloat::getZero(semantics(*mode.destinationFormat));
    (void)value.convertFromAPInt(input, mode.signedInteger,
                                 llvm::RoundingMode::NearestTiesToEven);
    return value.bitcastToAPInt();
  }
  case ConversionFamily::FloatToInteger: {
    loom::sim::PrimitiveOperationDescriptor descriptor{
        mode.actor, mode.outputWidth, mode.inputWidth};
    llvm::APFloat floating(semantics(*mode.sourceFormat), input);
    loom::sim::PrimitiveValue result = take(
        test, loom::sim::evaluatePrimitiveOperation(
                  descriptor, {loom::sim::PrimitiveValue::floating(floating)}));
    require(test, result.isDefined(),
            "saturating simulator oracle did not return a defined integer");
    return *result.bits;
  }
  }
  llvm_unreachable("unknown conversion family");
}

void appendUnique(std::vector<llvm::APInt> &values, llvm::APInt value) {
  if (!llvm::is_contained(values, value))
    values.push_back(std::move(value));
}

llvm::APInt floatBitsFromInteger(::fabric::FloatFormat format,
                                 const llvm::APInt &value, bool isSigned) {
  llvm::APFloat floating = llvm::APFloat::getZero(semantics(format));
  (void)floating.convertFromAPInt(value, isSigned,
                                  llvm::RoundingMode::NearestTiesToEven);
  return floating.bitcastToAPInt();
}

std::vector<llvm::APInt> floatingInputs(const ModeInfo &mode) {
  const ::fabric::FloatFormat format = *mode.sourceFormat;
  const FloatLayout shape = layout(format);
  const unsigned width = shape.width();
  const std::uint64_t sign = shape.sign();
  const std::uint64_t infinity = shape.infinity();
  const std::uint64_t fractionMask =
      (std::uint64_t{1} << shape.fractionBits) - 1;
  const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
  std::vector<llvm::APInt> values;
  for (std::uint64_t bits : {
           std::uint64_t{0},
           sign,
           shape.one(),
           sign | shape.one(),
           shape.one() | (std::uint64_t{1} << (shape.fractionBits - 1)),
           sign | shape.one() | (std::uint64_t{1} << (shape.fractionBits - 1)),
           infinity,
           sign | infinity,
           infinity | quiet | 0x5,
           infinity | 0x3,
           std::uint64_t{1},
           fractionMask,
           std::uint64_t{1} << shape.fractionBits,
           infinity - 1,
       })
    appendUnique(values, llvm::APInt(width, bits));

  if (mode.family == ConversionFamily::WidthCast) {
    if (format == ::fabric::FloatFormat::F32 &&
        *mode.destinationFormat == ::fabric::FloatFormat::F16) {
      appendUnique(values, llvm::APInt(32, 0x3f801000U));
      appendUnique(values, llvm::APInt(32, 0xbf801000U));
      appendUnique(values, llvm::APInt(32, 0x33000000U));
      appendUnique(values,
                   floatBitsFromInteger(format, llvm::APInt(32, 65519), false));
      appendUnique(values,
                   floatBitsFromInteger(format, llvm::APInt(32, 65520), false));
    }
    if (format == ::fabric::FloatFormat::F32 &&
        *mode.destinationFormat == ::fabric::FloatFormat::BF16) {
      appendUnique(values, llvm::APInt(32, 0x3f808000U));
      appendUnique(values, llvm::APInt(32, 0xbf808000U));
      appendUnique(values, llvm::APInt(32, 0x00008000U));
      appendUnique(values, llvm::APInt(32, 0x80008000U));
    }
    if (format == ::fabric::FloatFormat::F64 &&
        *mode.destinationFormat == ::fabric::FloatFormat::BF16) {
      appendUnique(values, llvm::APInt(64, 0x3ff0100000000000ULL));
      appendUnique(values, llvm::APInt(64, 0xbff0100000000000ULL));
    }
  }

  if (mode.family == ConversionFamily::FloatToInteger) {
    const unsigned limitBit = mode.outputWidth - (mode.signedInteger ? 1U : 0U);
    llvm::APInt positiveLimit(limitBit + 2, 1);
    positiveLimit <<= limitBit;
    appendUnique(values, floatBitsFromInteger(format, positiveLimit, false));
    appendUnique(values,
                 floatBitsFromInteger(format, positiveLimit - 1, false));
    llvm::APFloat insideUpperLimit = llvm::APFloat::getZero(semantics(format));
    (void)insideUpperLimit.convertFromAPInt(
        positiveLimit, false, llvm::RoundingMode::NearestTiesToEven);
    (void)insideUpperLimit.next(true);
    appendUnique(values, insideUpperLimit.bitcastToAPInt());
    llvm::APInt negativeOne(mode.outputWidth + 1, -1, true);
    appendUnique(values, floatBitsFromInteger(format, negativeOne, true));
    if (mode.signedInteger) {
      llvm::APInt negativeLimit =
          llvm::APInt::getOneBitSet(mode.outputWidth, mode.outputWidth - 1);
      appendUnique(values, floatBitsFromInteger(format, negativeLimit, true));
      llvm::APFloat outsideLowerLimit =
          llvm::APFloat::getZero(semantics(format));
      (void)outsideLowerLimit.convertFromAPInt(
          negativeLimit, true, llvm::RoundingMode::NearestTiesToEven);
      (void)outsideLowerLimit.next(true);
      appendUnique(values, outsideLowerLimit.bitcastToAPInt());
    }
  }
  return values;
}

std::vector<llvm::APInt> integerInputs(const ModeInfo &mode) {
  const unsigned width = mode.inputWidth;
  std::vector<llvm::APInt> values = {
      llvm::APInt(width, 0), llvm::APInt(width, 1),
      llvm::APInt::getAllOnes(width), llvm::APInt::getSignedMaxValue(width),
      llvm::APInt::getSignedMinValue(width)};
  if (width == 32) {
    appendUnique(values, llvm::APInt(32, 0x01000001U));
    appendUnique(values, llvm::APInt(32, 65519));
    appendUnique(values, llvm::APInt(32, 65520));
    appendUnique(values, llvm::APInt(32, 1000000));
  }
  if (width == 64)
    appendUnique(values, llvm::APInt(64, 0x0020000000000001ULL));
  return values;
}

std::vector<llvm::APInt> inputsForMode(const ModeInfo &mode) {
  return mode.family == ConversionFamily::IntegerToFloat ? integerInputs(mode)
                                                         : floatingInputs(mode);
}

std::vector<llvm::APInt> ordinaryFloatToIntegerInputs(const ModeInfo &mode) {
  require("ordinaryFloatToIntegerInputs",
          mode.family == ConversionFamily::FloatToInteger &&
              mode.sourceFormat == ::fabric::FloatFormat::F32 &&
              mode.outputWidth == 16,
          "ordinary conversion fixture has unexpected endpoints");
  std::vector<llvm::APInt> values = {
      llvm::APInt(32, 0x00000000U), llvm::APInt(32, 0x80000000U),
      llvm::APInt(32, 0x3f800000U), llvm::APInt(32, 0x3fc00000U)};
  if (mode.signedInteger) {
    values.push_back(llvm::APInt(32, 0xbf800000U));
    values.push_back(llvm::APInt(32, 0xbfc00000U));
    values.push_back(
        floatBitsFromInteger(*mode.sourceFormat, llvm::APInt(16, 32767), true));
    values.push_back(floatBitsFromInteger(
        *mode.sourceFormat, llvm::APInt::getSignedMinValue(16), true));
  } else {
    values.push_back(floatBitsFromInteger(*mode.sourceFormat,
                                          llvm::APInt(16, 65535), false));
  }
  return values;
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

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          bool omitLastSemanticValue = false) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "conversion fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "conversion relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() < 64,
          "conversion test codebook does not fit its physical carrier");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (auto [index, point] : llvm::enumerate(domain)) {
    require(test, point.semanticConfiguration.has_value(),
            "configured conversion behavior has no semantic value");
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (index == 0)
      inactive = semantic;
    entries.push_back({std::move(semantic), {physicalCode(index)}});
  }
  if (omitLastSemanticValue) {
    require(test, entries.size() > 1,
            "cannot truncate a singleton conversion codebook");
    entries.pop_back();
  }
  auto field =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride override{
      field, FiniteCodebookEncoding{6, std::move(entries)},
      std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(override)}));
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
                             llvm::StringRef leafName = {}) {
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
    const auto configuration = llvm::find_if(ports, [](const auto &port) {
      return port.getName().starts_with("config_");
    });
    require(test, configuration != ports.end(),
            "configured conversion leaf has no selector");
    configuration->type = builder.getIntegerType(5);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(leafName.empty() ? moduleName(fabric.family)
                                             : leafName),
      ports);
  return SkeletonFixture{std::move(module), leaf};
}

struct MultiSkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
};

MultiSkeletonFixture makeMultiSkeleton(llvm::StringRef test,
                                       mlir::MLIRContext &context,
                                       const FabricFixture &fabric,
                                       const ConfigurationABI &abi) {
  require(test, fabric.occurrences.size() == fabric.physicalOccurrences.size(),
          "multi-leaf fixture has mismatched occurrence inventories");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
  for (auto [index, physicalOccurrence] :
       llvm::enumerate(fabric.physicalOccurrences)) {
    const auto *resolved = fabric.fabric.view().resolvedFabricOpCapability(
        fabric.occurrences[index]);
    require(test, resolved != nullptr, "multi-leaf capability did not resolve");
    std::vector<circt::hw::PortInfo> ports =
        take(test, deriveFabricOperationLeafPorts(builder, physicalOccurrence,
                                                  *resolved, abi));
    leaves.push_back(circt::hw::HWModuleGeneratedOp::create(
        builder, location,
        mlir::FlatSymbolRefAttr::get(&context,
                                     fabricOperationGeneratorSchemaSymbol),
        builder.getStringAttr("transaction_float_width_cast_" +
                              std::to_string(index)),
        ports));
  }
  return MultiSkeletonFixture{std::move(module), std::move(leaves)};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output);
  return text;
}

llvm::Expected<FabricOperationProviderOutput> trySpecialize(
    SkeletonFixture &skeleton, const FabricFixture &fabric,
    const FinalizedConfigurationABI &abi,
    const FabricOperationProviderRegistry &registry,
    BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatConversionProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<FabricOperationProviderOutput>
lateFailingProvider(FabricOperationProviderRequest) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "late provider failure");
}

void registrationIsTransactionalAndComplete() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  require(test,
          llvm::count_if(
              coverage,
              [](const auto &entry) { return !entry.recipes.empty(); }) == 3,
          "conversion package registered an unexpected family");
  for (ConversionFamily family :
       {ConversionFamily::WidthCast, ConversionFamily::IntegerToFloat,
        ConversionFamily::FloatToInteger}) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "conversion package registration is incomplete or non-portable");
  }
  const auto snapshot = [](const FabricOperationProviderRegistry &value) {
    std::vector<std::pair<::fabric::ImplementationFamilyId,
                          std::vector<BackendRecipeKey>>>
        result;
    for (const auto &entry : value.coverage())
      result.emplace_back(entry.implementationFamily, entry.recipes);
    return result;
  };
  FabricOperationProviderRegistry preseeded;
  if (llvm::Error error =
          preseeded.add({::fabric::ImplementationFamilyId::ScalarFloatToInteger,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         lateFailingProvider}))
    fail(test, llvm::toString(std::move(error)));
  const auto before = snapshot(preseeded);
  llvm::Error duplicate = registerPortableFloatConversionProviders(preseeded);
  require(test, static_cast<bool>(duplicate),
          "duplicate conversion registration was accepted");
  llvm::consumeError(std::move(duplicate));
  require(test, snapshot(preseeded) == before,
          "failed conversion registration partially mutated registry");
}

std::string makeTestbench(llvm::ArrayRef<ModeInfo> widthModes,
                          llvm::ArrayRef<ModeInfo> integerToFloatModes,
                          llvm::ArrayRef<ModeInfo> floatToIntegerModes,
                          const ModeInfo &ordinarySignedMode,
                          const ModeInfo &ordinaryUnsignedMode) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(module testbench;
  logic [66:0] width_input;
  logic [5:0] width_mode;
  logic [66:0] width_output;
  logic [66:0] integer_to_float_input;
  logic [5:0] integer_to_float_mode;
  logic [66:0] integer_to_float_output;
  logic [66:0] float_to_integer_input;
  logic [5:0] float_to_integer_mode;
  logic [66:0] float_to_integer_output;
  logic [66:0] ordinary_signed_input;
  logic [66:0] ordinary_signed_output;
  logic [66:0] ordinary_unsigned_input;
  logic [66:0] ordinary_unsigned_output;

`ifdef TEST_WIDTH_CAST
  scalar_float_width_cast width_dut(
      .data_input_0(width_input), .config_0(width_mode),
      .data_output_0(width_output));
`endif
`ifdef TEST_INTEGER_TO_FLOAT
  scalar_integer_to_float integer_to_float_dut(
      .data_input_0(integer_to_float_input),
      .config_0(integer_to_float_mode),
      .data_output_0(integer_to_float_output));
`endif
`ifdef TEST_FLOAT_TO_INTEGER
  scalar_float_to_integer float_to_integer_dut(
      .data_input_0(float_to_integer_input),
      .config_0(float_to_integer_mode),
      .data_output_0(float_to_integer_output));
`endif
`ifdef TEST_SIGNED_SINGLETON
  scalar_float_to_signed_integer_singleton ordinary_signed_dut(
      .data_input_0(ordinary_signed_input),
      .data_output_0(ordinary_signed_output));
`endif
`ifdef TEST_UNSIGNED_SINGLETON
  scalar_float_to_unsigned_integer_singleton ordinary_unsigned_dut(
      .data_input_0(ordinary_unsigned_input),
      .data_output_0(ordinary_unsigned_output));
`endif

`ifdef TEST_WIDTH_CAST
  task automatic check_width(
      input logic [5:0] mode, input logic [66:0] value,
      input logic [66:0] expected);
    begin
      width_mode = mode;
      width_input = value;
      #1;
      if (width_output !== expected)
        $fatal(1, "width cast mismatch mode=%0d input=%h got=%h expected=%h",
               mode, value, width_output, expected);
    end
  endtask
`endif

`ifdef TEST_INTEGER_TO_FLOAT
  task automatic check_integer_to_float(
      input logic [5:0] mode, input logic [66:0] value,
      input logic [66:0] expected);
    begin
      integer_to_float_mode = mode;
      integer_to_float_input = value;
      #1;
      if (integer_to_float_output !== expected)
        $fatal(1, "integer-to-float mismatch mode=%0d input=%h got=%h expected=%h",
               mode, value, integer_to_float_output, expected);
    end
  endtask
`endif

`ifdef TEST_FLOAT_TO_INTEGER
  task automatic check_float_to_integer(
      input logic [5:0] mode, input logic [66:0] value,
      input logic [66:0] expected);
    begin
      float_to_integer_mode = mode;
      float_to_integer_input = value;
      #1;
      if (float_to_integer_output !== expected)
        $fatal(1, "float-to-integer mismatch mode=%0d input=%h got=%h expected=%h",
               mode, value, float_to_integer_output, expected);
    end
  endtask
`endif

`ifdef TEST_SIGNED_SINGLETON
  task automatic check_ordinary_signed(
      input logic [66:0] value, input logic [66:0] expected);
    begin
      ordinary_signed_input = value;
      #1;
      if (ordinary_signed_output !== expected)
        $fatal(1, "ordinary signed float-to-integer mismatch input=%h got=%h expected=%h",
               value, ordinary_signed_output, expected);
    end
  endtask
`endif

`ifdef TEST_UNSIGNED_SINGLETON
  task automatic check_ordinary_unsigned(
      input logic [66:0] value, input logic [66:0] expected);
    begin
      ordinary_unsigned_input = value;
      #1;
      if (ordinary_unsigned_output !== expected)
        $fatal(1, "ordinary unsigned float-to-integer mismatch input=%h got=%h expected=%h",
               value, ordinary_unsigned_output, expected);
    end
  endtask
`endif

  initial begin
)sv";

  const auto physicalInput = [](const ModeInfo &mode,
                                const llvm::APInt &input) {
    llvm::APInt result = input.zext(67);
    if (mode.inputWidth < 67)
      result |= llvm::APInt::getHighBitsSet(67, 67 - mode.inputWidth);
    return result;
  };
  const auto emit = [&](llvm::StringRef task, llvm::ArrayRef<ModeInfo> modes) {
    for (const ModeInfo &mode : modes) {
      for (const llvm::APInt &input : inputsForMode(mode)) {
        const llvm::APInt padded = physicalInput(mode, input);
        const llvm::APInt expected =
            evaluateMode("makeTestbench", mode, input).zext(67);
        output << "    " << task << "(6'd"
               << static_cast<unsigned>(mode.physicalCode) << ", "
               << hexLiteral(padded) << ", " << hexLiteral(expected) << ");\n";
      }
    }
  };
  const auto emitInactive = [&](llvm::StringRef task,
                                llvm::ArrayRef<ModeInfo> modes) {
    require("makeTestbench", !modes.empty(), "conversion mode set is empty");
    const ModeInfo &mode = modes.front();
    const std::vector<llvm::APInt> inputs = inputsForMode(mode);
    const llvm::APInt &input = inputs.back();
    output << "    " << task << "(6'd0, "
           << hexLiteral(physicalInput(mode, input)) << ", "
           << hexLiteral(evaluateMode("makeTestbench", mode, input).zext(67))
           << ");\n";
  };
  output << "`ifdef TEST_WIDTH_CAST\n";
  emit("check_width", widthModes);
  emitInactive("check_width", widthModes);
  output << "`endif\n`ifdef TEST_INTEGER_TO_FLOAT\n";
  emit("check_integer_to_float", integerToFloatModes);
  emitInactive("check_integer_to_float", integerToFloatModes);
  output << "`endif\n`ifdef TEST_FLOAT_TO_INTEGER\n";
  emit("check_float_to_integer", floatToIntegerModes);
  emitInactive("check_float_to_integer", floatToIntegerModes);
  output << "`endif\n";
  const auto emitOrdinary = [&](llvm::StringRef task, const ModeInfo &mode) {
    for (const llvm::APInt &input : ordinaryFloatToIntegerInputs(mode)) {
      output << "    " << task << "(" << hexLiteral(physicalInput(mode, input))
             << ", "
             << hexLiteral(evaluateMode("makeTestbench", mode, input).zext(67))
             << ");\n";
    }
  };
  output << "`ifdef TEST_SIGNED_SINGLETON\n";
  emitOrdinary("check_ordinary_signed", ordinarySignedMode);
  output << "`endif\n`ifdef TEST_UNSIGNED_SINGLETON\n";
  emitOrdinary("check_ordinary_unsigned", ordinaryUnsignedMode);
  output << "`endif\n";
  output << R"sv(    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string makeYosysScript(llvm::StringRef top, llvm::StringRef source) {
  std::string script;
  llvm::raw_string_ostream output(script);
  output << "read_verilog -sv " << source << '\n'
         << "hierarchy -check -top " << top << '\n'
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none " << top << "/t:$*ff* " << top
         << "/t:$*latch* " << top << "/t:$_*FF* " << top << "/t:$_*LATCH* "
         << top << "/t:$mem* " << top << "/m:*\n"
         << "synth -noabc -top " << top << '\n'
         << "check -assert\n"
         << "select -assert-none " << top << "/t:$*ff* " << top
         << "/t:$*latch* " << top << "/t:$_*FF* " << top << "/t:$_*LATCH* "
         << top << "/t:$mem* " << top << "/m:*\n"
         << "stat\n";
  return output.str();
}

struct EmittedFamily final {
  std::string systemVerilog;
  std::vector<ModeInfo> modes;
};

std::string emitSpecialized(llvm::StringRef test, const FabricFixture &fabric,
                            const FinalizedConfigurationABI &abi,
                            const FabricOperationProviderRegistry &registry,
                            llvm::StringRef leafName = {}) {
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), false, leafName);
  const bool configured =
      !capability(test, fabric).configurationFieldSchema.empty();
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  require(test,
          ports.size() == (configured ? 3 : 2) &&
              ports.atInput(0).getName() == "data_input_0" &&
              ports.atOutput(0).getName() == "data_output_0",
          "conversion leaf ports do not follow ConfigurationABI 2.0");
  if (configured)
    require(test,
            ports.atInput(1).getName() == "config_0" &&
                ports.atInput(1).type ==
                    mlir::IntegerType::get(context.get(), 6),
            "configured conversion selector is not the finalized ABI field");

  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(module), abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable conversion emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

EmittedFamily emitConfiguredFamily(llvm::StringRef test,
                                   const ArtifactStore &store,
                                   ConversionFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  std::vector<ModeInfo> modes = configuredModes(test, fabric);
  const std::size_t expectedModes =
      family == ConversionFamily::WidthCast ? 18 : 10;
  require(test, modes.size() == expectedModes,
          "configured fixture has the wrong sealed behavior count");
  if (family == ConversionFamily::FloatToInteger) {
    const auto &schemas = capability(test, fabric).enabledOperationSchemas;
    require(test,
            llvm::is_contained(schemas,
                               ::dataflow::OperationSchemaId::ArithFPToSI) &&
                llvm::is_contained(
                    schemas, ::dataflow::OperationSchemaId::ArithFPToUI) &&
                llvm::is_contained(
                    schemas, ::dataflow::OperationSchemaId::LLVMFPToSISat) &&
                llvm::is_contained(
                    schemas, ::dataflow::OperationSchemaId::LLVMFPToUISat) &&
                llvm::all_of(
                    modes,
                    [](const ModeInfo &mode) {
                      return mode.actor.schema ==
                                 ::dataflow::OperationSchemaId::LLVMFPToSISat ||
                             mode.actor.schema ==
                                 ::dataflow::OperationSchemaId::LLVMFPToUISat;
                    }),
            "ordinary and saturating float-to-integer schemas diverged");
  }
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  std::string first = emitSpecialized(test, fabric, abi, registry);
  const std::string second = emitSpecialized(test, fabric, abi, registry);
  require(test, first == second,
          "identical conversion inputs produced different SystemVerilog");
  const std::string expectedModule = "module " + moduleName(family).str();
  require(test,
          llvm::StringRef(first).contains(expectedModule) &&
              llvm::StringRef(first).contains("config_0") &&
              llvm::StringRef(first).contains("function automatic") &&
              !llvm::StringRef(first).contains("shortreal") &&
              !llvm::StringRef(first).contains("DPI"),
          "configured conversion did not materialize its sealed selector");
  return {std::move(first), std::move(modes)};
}

EmittedFamily
emitOrdinaryFloatToInteger(llvm::StringRef test, const ArtifactStore &store,
                           FixtureKind kind, llvm::StringRef emittedModuleName,
                           ::dataflow::OperationSchemaId expectedSchema) {
  FabricFixture fabric =
      makeFabric(test, store, ConversionFamily::FloatToInteger, kind);
  ModeInfo mode = singletonMode(test, fabric);
  require(test, mode.actor.schema == expectedSchema,
          "ordinary float-to-integer fixture selected the wrong schema");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  std::string first =
      emitSpecialized(test, fabric, abi, registry, emittedModuleName);
  const std::string second =
      emitSpecialized(test, fabric, abi, registry, emittedModuleName);
  require(test,
          first == second &&
              llvm::StringRef(first).contains("module " +
                                              emittedModuleName.str()) &&
              !llvm::StringRef(first).contains("config_") &&
              llvm::StringRef(first).contains("function automatic"),
          "ordinary float-to-integer singleton did not lower directly");
  return {std::move(first), {std::move(mode)}};
}

void fieldlessSingletonsNeedNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  for (ConversionFamily family :
       {ConversionFamily::WidthCast, ConversionFamily::IntegerToFloat,
        ConversionFamily::FloatToInteger}) {
    FabricFixture fabric =
        makeFabric(test, store, family, FixtureKind::Singleton);
    const auto &resolved = capability(test, fabric);
    auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    const auto domain = relation.finiteBehaviorDomain();
    require(test,
            resolved.configurationFieldSchema.empty() &&
                relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                domain.size() == 1 && !domain.front().semanticConfiguration,
            "singleton conversion retained a semantic carrier");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::string first = emitSpecialized(test, fabric, abi, registry);
    const std::string second = emitSpecialized(test, fabric, abi, registry);
    require(test,
            first == second && !llvm::StringRef(first).contains("config_") &&
                llvm::StringRef(first).contains("function automatic"),
            "fieldless singleton conversion did not lower directly");
  }
}

void failuresAreTypedAndTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  for (ConversionFamily family :
       {ConversionFamily::WidthCast, ConversionFamily::IntegerToFloat,
        ConversionFamily::FloatToInteger}) {
    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);

    std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
    SkeletonFixture malformed =
        makeSkeleton(test, *malformedContext, valid, validAbi.abi(), true);
    const std::string malformedBefore = moduleText(*malformed.module);
    expectInvalid(test, trySpecialize(malformed, valid, validAbi, registry),
                  "leaf port");
    require(test, moduleText(*malformed.module) == malformedBefore,
            "malformed conversion leaf mutated the caller module");

    expectInvalid(test,
                  finalizeConfigurationABI(
                      makeConfigurationAbiDraft(test, valid, true), store),
                  "finite codebook does not equal its Fabric relation");

    FabricFixture unsupported =
        makeFabric(test, store, family, FixtureKind::UnsupportedContract);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton = makeSkeleton(
        test, *unsupportedContext, unsupported, unsupportedAbi.abi());
    const std::string unsupportedBefore =
        moduleText(*unsupportedSkeleton.module);
    expectTypedUnsupported(test,
                           trySpecialize(unsupportedSkeleton, unsupported,
                                         unsupportedAbi, registry),
                           familyId(family),
                           BackendRecipeKey::PortableSystemVerilog,
                           "unsupported conversion resource contract");
    require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
            "unsupported conversion contract mutated the caller module");

    for (BackendRecipeKey recipe :
         {BackendRecipeKey::SynopsysDesignWare,
          BackendRecipeKey::CadenceChipWare, BackendRecipeKey::AmdXilinx,
          BackendRecipeKey::IntelAltera}) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native =
          makeSkeleton(test, *nativeContext, valid, validAbi.abi());
      const std::string nativeBefore = moduleText(*native.module);
      expectTypedUnsupported(
          test, trySpecialize(native, valid, validAbi, registry, recipe),
          familyId(family), recipe, "unsupported native conversion recipe");
      require(test, moduleText(*native.module) == nativeBefore,
              "unsupported native recipe mutated the caller module");
    }
  }
}

void lateProviderFailureRollsBack(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, ConversionFamily::WidthCast,
                                    FixtureKind::TransactionPair);
  require(test, fabric.physicalOccurrences.size() == 2,
          "transaction fixture is not multi-occurrence");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  if (llvm::Error error = registry.add({familyId(ConversionFamily::WidthCast),
                                        BackendRecipeKey::IntelAltera,
                                        {},
                                        lateFailingProvider}))
    fail(test, llvm::toString(std::move(error)));

  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  MultiSkeletonFixture skeleton =
      makeMultiSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaves.size() == 2,
          "transaction skeleton has the wrong leaf count");
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaves[0], fabric.physicalOccurrences[0]},
      {skeleton.leaves[1], fabric.physicalOccurrences[1]}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrences[0],
       BackendRecipeKey::PortableSystemVerilog,
       {}},
      {fabric.physicalOccurrences[1], BackendRecipeKey::IntelAltera, {}}};
  ExternalImplementationContractCatalog externalContracts;
  const std::string before = moduleText(*skeleton.module);
  expectInvalid(test,
                specializeFabricOperationLeaves(*skeleton.module, abi,
                                                associations, recipes, registry,
                                                externalContracts),
                "late provider failure");
  require(test, moduleText(*skeleton.module) == before,
          "late provider failure partially committed specialization");
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  EmittedFamily width =
      emitConfiguredFamily(test, store, ConversionFamily::WidthCast);
  EmittedFamily integerToFloat =
      emitConfiguredFamily(test, store, ConversionFamily::IntegerToFloat);
  EmittedFamily floatToInteger =
      emitConfiguredFamily(test, store, ConversionFamily::FloatToInteger);
  EmittedFamily ordinarySigned = emitOrdinaryFloatToInteger(
      test, store, FixtureKind::OrdinarySignedSingleton,
      "scalar_float_to_signed_integer_singleton",
      ::dataflow::OperationSchemaId::ArithFPToSI);
  EmittedFamily ordinaryUnsigned = emitOrdinaryFloatToInteger(
      test, store, FixtureKind::OrdinaryUnsignedSingleton,
      "scalar_float_to_unsigned_integer_singleton",
      ::dataflow::OperationSchemaId::ArithFPToUI);
  const std::string testbench = makeTestbench(
      width.modes, integerToFloat.modes, floatToInteger.modes,
      ordinarySigned.modes.front(), ordinaryUnsigned.modes.front());
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_float_width_cast.sv", width.systemVerilog},
           {"scalar_integer_to_float.sv", integerToFloat.systemVerilog},
           {"scalar_float_to_integer.sv", floatToInteger.systemVerilog},
           {"scalar_float_to_signed_integer_singleton.sv",
            ordinarySigned.systemVerilog},
           {"scalar_float_to_unsigned_integer_singleton.sv",
            ordinaryUnsigned.systemVerilog},
           {"testbench.sv", testbench},
           {"portable_float_width_cast.ys",
            makeYosysScript("scalar_float_width_cast",
                            "scalar_float_width_cast.sv")},
           {"portable_integer_to_float.ys",
            makeYosysScript("scalar_integer_to_float",
                            "scalar_integer_to_float.sv")},
           {"portable_float_to_integer.ys",
            makeYosysScript("scalar_float_to_integer",
                            "scalar_float_to_integer.sv")},
           {"portable_float_to_signed_integer_singleton.ys",
            makeYosysScript("scalar_float_to_signed_integer_singleton",
                            "scalar_float_to_signed_integer_singleton.sv")},
           {"portable_float_to_unsigned_integer_singleton.ys",
            makeYosysScript("scalar_float_to_unsigned_integer_singleton",
                            "scalar_float_to_unsigned_integer_singleton.sv")}}))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  registrationIsTransactionalAndComplete();
  fieldlessSingletonsNeedNoSelector(root / "singletons");
  failuresAreTypedAndTransactional(root / "failures");
  lateProviderFailureRollsBack(root / "late-failure");
  configuredBehaviorAndArtifacts(root);
  return EXIT_SUCCESS;
}
