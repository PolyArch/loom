#include "Hardware/RTL/Providers/AmdXilinxScalarIntegerMultiply.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

constexpr llvm::StringLiteral kContractRef = "amd.xilinx.unisim.dsp58@1";
constexpr llvm::StringLiteral kInputSlot = "primitive";
constexpr llvm::StringLiteral kProviderBuild =
    "amd.vivado:2024.2.2-build.6060944-ip.6050500-shared.6060542";
constexpr llvm::StringLiteral kResourceKey = "unisim:versal:DSP58";
constexpr llvm::StringLiteral kPart = "xcvp1802-vsva5601-3HP-e-S";
constexpr llvm::StringLiteral kPrimitive = "DSP58";
constexpr llvm::StringLiteral kPayloadName =
    "contracts/amd_xilinx_unisim_dsp58.json";
constexpr llvm::StringLiteral kBlackBoxContract =
    "{\"contract\":\"amd.xilinx.unisim.dsp58@1\","
    "\"device\":\"xcvp1802-vsva5601-3HP-e-S\","
    "\"latency\":\"combinational\",\"module\":\"DSP58\","
    "\"operation\":\"i16_mul_mod\","
    "\"resource\":\"unisim:versal:DSP58\","
    "\"tool_build\":"
    "\"amd.vivado:2024.2.2-build.6060944-ip.6050500-shared.6060542\"}\n";

llvm::Error unsupported() {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      BackendRecipeKey::AmdXilinx);
}

bool isExactPlatform(const platform::ImplementationPlatform *platform) {
  if (!platform)
    return false;
  const auto *target = std::get_if<platform::FpgaTarget>(&platform->target());
  return target && target->vendor == platform::FpgaVendor::AmdXilinx &&
         target->deviceOrderingCode == kPart;
}

bool isExactExternalInput(llvm::ArrayRef<ExternalInputBinding> inputs) {
  if (inputs.size() != 1 || inputs.front().providerInputSlotRef != kInputSlot)
    return false;
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &inputs.front().dependencyIdentity);
  return resource && resource->stableProviderBuildIdentity == kProviderBuild &&
         resource->resourceKey == kResourceKey;
}

bool hasExactCapability(const fabric::ResolvedFabricOpCapabilityView &view) {
  if (view.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarIntegerMultiply ||
      view.enabledOperationSchemas.size() != 1 ||
      view.enabledOperationSchemas.front() !=
          ::dataflow::OperationSchemaId::ArithMulI ||
      !view.configurationFieldSchema.empty())
    return false;
  const auto *parameters =
      std::get_if<::fabric::ScalarIntegerParams>(&view.parameterizedCapability);
  if (!parameters || !parameters->integerWidths.valid() ||
      parameters->integerWidths.size() != 1 ||
      !parameters->integerWidths.contains(::fabric::IntegerWidth::I16) ||
      !parameters->pointerFormats.empty())
    return false;

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       view.physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input ? inputs
                                                                    : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  return inputs.size() == 2 && outputs.size() == 1 &&
         inputs[0]->reference.ordinal == 0 &&
         inputs[1]->reference.ordinal == 1 &&
         outputs[0]->reference.ordinal == 0 &&
         inputs[0]->payloadWidthBits == 16 &&
         inputs[1]->payloadWidthBits == 16 &&
         outputs[0]->payloadWidthBits == 16;
}

llvm::Expected<bool>
hasExactResourceContract(const fabric::ResolvedFabricOpCapabilityView &view) {
  auto actual = ::fabric::encodeResourceContractRecord(
      view.resourceStateAndTimingContract);
  if (!actual)
    return actual.takeError();
  auto expected = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!expected)
    return expected.takeError();
  return *actual == *expected;
}

circt::hw::PortInfo port(mlir::OpBuilder &builder, llvm::StringRef name,
                         unsigned width,
                         circt::hw::ModulePort::Direction direction) {
  return {
      {builder.getStringAttr(name), builder.getIntegerType(width), direction}};
}

std::vector<circt::hw::PortInfo> dsp58Ports(mlir::OpBuilder &builder) {
  using Direction = circt::hw::ModulePort::Direction;
  return {
      port(builder, "ACOUT", 34, Direction::Output),
      port(builder, "BCOUT", 24, Direction::Output),
      port(builder, "CARRYCASCOUT", 1, Direction::Output),
      port(builder, "CARRYOUT", 4, Direction::Output),
      port(builder, "MULTSIGNOUT", 1, Direction::Output),
      port(builder, "OVERFLOW", 1, Direction::Output),
      port(builder, "P", 58, Direction::Output),
      port(builder, "PATTERNBDETECT", 1, Direction::Output),
      port(builder, "PATTERNDETECT", 1, Direction::Output),
      port(builder, "PCOUT", 58, Direction::Output),
      port(builder, "UNDERFLOW", 1, Direction::Output),
      port(builder, "XOROUT", 8, Direction::Output),
      port(builder, "A", 34, Direction::Input),
      port(builder, "ACIN", 34, Direction::Input),
      port(builder, "ALUMODE", 4, Direction::Input),
      port(builder, "ASYNC_RST", 1, Direction::Input),
      port(builder, "B", 24, Direction::Input),
      port(builder, "BCIN", 24, Direction::Input),
      port(builder, "C", 58, Direction::Input),
      port(builder, "CARRYCASCIN", 1, Direction::Input),
      port(builder, "CARRYIN", 1, Direction::Input),
      port(builder, "CARRYINSEL", 3, Direction::Input),
      port(builder, "CEA1", 1, Direction::Input),
      port(builder, "CEA2", 1, Direction::Input),
      port(builder, "CEAD", 1, Direction::Input),
      port(builder, "CEALUMODE", 1, Direction::Input),
      port(builder, "CEB1", 1, Direction::Input),
      port(builder, "CEB2", 1, Direction::Input),
      port(builder, "CEC", 1, Direction::Input),
      port(builder, "CECARRYIN", 1, Direction::Input),
      port(builder, "CECTRL", 1, Direction::Input),
      port(builder, "CED", 1, Direction::Input),
      port(builder, "CEINMODE", 1, Direction::Input),
      port(builder, "CEM", 1, Direction::Input),
      port(builder, "CEP", 1, Direction::Input),
      port(builder, "CLK", 1, Direction::Input),
      port(builder, "D", 27, Direction::Input),
      port(builder, "INMODE", 5, Direction::Input),
      port(builder, "MULTSIGNIN", 1, Direction::Input),
      port(builder, "NEGATE", 3, Direction::Input),
      port(builder, "OPMODE", 9, Direction::Input),
      port(builder, "PCIN", 58, Direction::Input),
      port(builder, "RSTA", 1, Direction::Input),
      port(builder, "RSTALLCARRYIN", 1, Direction::Input),
      port(builder, "RSTALUMODE", 1, Direction::Input),
      port(builder, "RSTB", 1, Direction::Input),
      port(builder, "RSTC", 1, Direction::Input),
      port(builder, "RSTCTRL", 1, Direction::Input),
      port(builder, "RSTD", 1, Direction::Input),
      port(builder, "RSTINMODE", 1, Direction::Input),
      port(builder, "RSTM", 1, Direction::Input),
      port(builder, "RSTP", 1, Direction::Input),
  };
}

struct Dsp58Parameters final {
  mlir::ArrayAttr declarations;
  mlir::ArrayAttr values;
};

Dsp58Parameters dsp58Parameters(mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Attribute> declarations;
  llvm::SmallVector<mlir::Attribute> values;
  const auto add = [&](llvm::StringRef name, mlir::TypedAttr value) {
    declarations.push_back(
        circt::hw::ParamDeclAttr::get(name, value.getType()));
    values.push_back(circt::hw::ParamDeclAttr::get(name, value));
  };
  const auto integer = [&](llvm::StringRef name, unsigned width,
                           const llvm::APInt &value) {
    add(name, builder.getIntegerAttr(builder.getIntegerType(width), value));
  };
  const auto integer32 = [&](llvm::StringRef name, std::uint64_t value) {
    integer(name, 32, llvm::APInt(32, value));
  };
  const auto string = [&](llvm::StringRef name, llvm::StringRef value) {
    add(name, builder.getStringAttr(value));
  };

  integer32("ACASCREG", 0);
  integer32("ADREG", 0);
  integer32("ALUMODEREG", 0);
  string("AMULTSEL", "A");
  integer32("AREG", 0);
  string("AUTORESET_PATDET", "NO_RESET");
  string("AUTORESET_PRIORITY", "RESET");
  string("A_INPUT", "DIRECT");
  integer32("BCASCREG", 0);
  string("BMULTSEL", "B");
  integer32("BREG", 0);
  string("B_INPUT", "DIRECT");
  integer32("CARRYINREG", 0);
  integer32("CARRYINSELREG", 0);
  integer32("CREG", 0);
  integer32("DREG", 0);
  string("DSP_MODE", "INT24");
  integer32("INMODEREG", 0);
  integer("IS_ALUMODE_INVERTED", 4, llvm::APInt(4, 0));
  integer("IS_ASYNC_RST_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_CARRYIN_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_CLK_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_INMODE_INVERTED", 5, llvm::APInt(5, 0));
  integer("IS_NEGATE_INVERTED", 3, llvm::APInt(3, 0));
  integer("IS_OPMODE_INVERTED", 9, llvm::APInt(9, 0));
  integer("IS_RSTALLCARRYIN_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTALUMODE_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTA_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTB_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTCTRL_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTC_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTD_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTINMODE_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTM_INVERTED", 1, llvm::APInt(1, 0));
  integer("IS_RSTP_INVERTED", 1, llvm::APInt(1, 0));
  integer("MASK", 58, llvm::APInt::getLowBitsSet(58, 56));
  integer32("MREG", 0);
  integer32("OPMODEREG", 0);
  integer("PATTERN", 58, llvm::APInt(58, 0));
  string("PREADDINSEL", "A");
  integer32("PREG", 0);
  string("RESET_MODE", "SYNC");
  integer("RND", 58, llvm::APInt(58, 0));
  string("SEL_MASK", "MASK");
  string("SEL_PATTERN", "PATTERN");
  string("USE_MULT", "MULTIPLY");
  string("USE_PATTERN_DETECT", "NO_PATDET");
  string("USE_SIMD", "ONE58");
  string("USE_WIDEXOR", "FALSE");
  string("XORSIMD", "XOR24_34_58_116");
  return {builder.getArrayAttr(declarations), builder.getArrayAttr(values)};
}

mlir::Value zero(mlir::OpBuilder &builder, mlir::Location location,
                 unsigned width) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

void materializeDsp58(FabricOperationProviderRequest request) {
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  const std::vector<circt::hw::PortInfo> primitivePorts = dsp58Ports(builder);
  const Dsp58Parameters parameters = dsp58Parameters(builder);
  circt::hw::HWModuleExternOp primitive = circt::hw::HWModuleExternOp::create(
      builder, location, builder.getStringAttr(kPrimitive),
      circt::hw::ModulePortInfo(primitivePorts), kPrimitive,
      parameters.declarations);

  const auto wrapperPortStorage = request.leaf.getPortList();
  const circt::hw::ModulePortInfo wrapperPorts(wrapperPortStorage);
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(), wrapperPorts,
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::vector<mlir::Value> operands;
        operands.reserve(primitive.getHWModuleType().getNumInputs());
        for (const circt::hw::PortInfo &input : primitivePorts) {
          if (input.isOutput())
            continue;
          mlir::Value value;
          if (input.getName() == "A")
            value = detail::resizeUnsigned(
                bodyBuilder, location, accessor.getInput("data_input_0"), 34);
          else if (input.getName() == "B")
            value = detail::resizeUnsigned(
                bodyBuilder, location, accessor.getInput("data_input_1"), 24);
          else if (input.getName() == "OPMODE")
            value = circt::hw::ConstantOp::create(bodyBuilder, location,
                                                  llvm::APInt(9, 5));
          else
            value = zero(bodyBuilder, location,
                         mlir::cast<mlir::IntegerType>(input.type).getWidth());
          operands.push_back(value);
        }
        circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
            bodyBuilder, location, primitive.getOperation(), "dsp58", operands,
            parameters.values);
        const auto output = llvm::find_if(
            primitivePorts, [](const circt::hw::PortInfo &candidate) {
              return candidate.isOutput() && candidate.getName() == "P";
            });
        const auto outputIndex = static_cast<unsigned>(
            std::distance(primitivePorts.begin(), output));
        unsigned resultIndex = 0;
        for (unsigned index = 0; index != outputIndex; ++index)
          if (primitivePorts[index].isOutput())
            ++resultIndex;
        accessor.setOutput(
            "data_output_0",
            circt::comb::ExtractOp::create(
                bodyBuilder, location, instance.getResult(resultIndex), 0, 16));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
}

llvm::Expected<FabricOperationProviderOutput>
materializeAmdXilinxScalarIntegerMultiply(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::AmdXilinx ||
      request.externalImplementationContractRef != kContractRef ||
      !isExactPlatform(request.implementationPlatform) ||
      !isExactExternalInput(request.externalInputs) ||
      !hasExactCapability(request.capability))
    return unsupported();
  auto exactContract = hasExactResourceContract(request.capability);
  if (!exactContract)
    return exactContract.takeError();
  if (!*exactContract)
    return unsupported();
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  materializeDsp58(request);
  FabricOperationProviderOutput output;
  output.payloads.push_back(
      {PayloadRole::BlackBoxContract, kPayloadName.str(),
       std::vector<std::uint8_t>(kBlackBoxContract.bytes_begin(),
                                 kBlackBoxContract.bytes_end())});
  output.externalImplementationBindings.push_back(
      {kContractRef.str(),
       std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                         request.externalInputs.end()),
       {},
       {{RepresentationObjectKind::Module, kPrimitive.str()}},
       ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                kPayloadName.str()}});
  return output;
}

llvm::Error
validateDsp58Binding(const ExternalImplementationBindingDraft &binding,
                     const ImplementationRepresentationRoot &representation,
                     const platform::ImplementationPlatform *platform) {
  if (representation.variant != RepresentationRootVariant::Rtl ||
      !isExactPlatform(platform) ||
      binding.providerContractRef != kContractRef ||
      !isExactExternalInput(binding.externalInputs) ||
      binding.fabricResourceRefs.size() != 1 ||
      binding.representationLocators !=
          std::vector<RepresentationLocator>{
              {RepresentationObjectKind::Module, kPrimitive.str()}} ||
      !binding.blackBoxContractPayload ||
      !(*binding.blackBoxContractPayload ==
        ImplementationPayloadKey{PayloadRole::BlackBoxContract,
                                 kPayloadName.str()}))
    return unsupported();
  return llvm::Error::success();
}

} // namespace

llvm::Error registerAmdXilinxScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                       BackendRecipeKey::AmdXilinx, kContractRef.str(),
                       materializeAmdXilinxScalarIntegerMultiply});
}

llvm::Error registerAmdXilinxDsp58ExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog) {
  return catalog.add(ExternalImplementationContract{
      kContractRef.str(),
      {{kInputSlot.str(), {ExternalDependencyKind::ToolBundledResource}}},
      {RepresentationRootVariant::Rtl},
      true,
      false,
      validateDsp58Binding});
}

} // namespace loom::hardware::rtl
