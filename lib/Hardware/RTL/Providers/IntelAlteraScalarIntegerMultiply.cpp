#include "Hardware/RTL/Providers/IntelAlteraScalarIntegerMultiply.h"

#include "Hardware/Implementation/FpgaNativeExternalContracts.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

const FpgaNativeExternalModuleContract &nativeContract() {
  return intelAlteraLpmMultExternalModuleContract();
}

llvm::Error unsupported() {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
      BackendRecipeKey::IntelAltera);
}

bool isExactPlatform(const platform::ImplementationPlatform *platform) {
  if (!platform)
    return false;
  const auto *target = std::get_if<platform::FpgaTarget>(&platform->target());
  return target && target->vendor == nativeContract().vendor &&
         target->deviceOrderingCode == nativeContract().deviceOrderingCode;
}

bool isExactExternalInput(llvm::ArrayRef<ExternalInputBinding> inputs) {
  if (inputs.size() != 1 || inputs.front().providerInputSlotRef !=
                                nativeContract().providerInputSlotRef)
    return false;
  const auto *resource = std::get_if<ToolBundledResourceDependency>(
      &inputs.front().dependencyIdentity);
  return resource &&
         resource->stableProviderBuildIdentity ==
             nativeContract().stableProviderBuildIdentity &&
         resource->resourceKey == nativeContract().resourceKey;
}

bool hasExactCapability(const fabric::ResolvedFabricOpCapabilityView &view) {
  if (view.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarIntegerMultiply ||
      view.enabledOperationSchemas !=
          std::vector<::dataflow::OperationSchemaId>{
              ::dataflow::OperationSchemaId::ArithMulI} ||
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

std::string lpmMultInstantiation() {
  return R"sv(lpm_mult #(
  .lpm_widtha(16),
  .lpm_widthb(16),
  .lpm_widthp(32),
  .lpm_widths(1),
  .lpm_representation("UNSIGNED"),
  .lpm_pipeline(0),
  .lpm_type("LPM_MULT"),
  .lpm_hint("DEDICATED_MULTIPLIER_CIRCUITRY=YES")
) lpm_multiplier (
  .dataa({{0}}),
  .datab({{1}}),
  .sum(1'b0),
  .aclr(1'b0),
  .sclr(1'b0),
  .clock(1'b0),
  .clken(1'b1),
  .result({{2}})
);)sv";
}

void materializeLpmMult(FabricOperationProviderRequest request) {
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  const auto wrapperPortStorage = request.leaf.getPortList();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(wrapperPortStorage),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::sv::WireOp product = circt::sv::WireOp::create(
            bodyBuilder, location, bodyBuilder.getIntegerType(32),
            "lpm_product");
        llvm::SmallVector<mlir::Value, 3> substitutions{
            accessor.getInput("data_input_0"),
            accessor.getInput("data_input_1"), product};
        circt::sv::VerbatimOp::create(
            bodyBuilder, location,
            bodyBuilder.getStringAttr(lpmMultInstantiation()), substitutions,
            bodyBuilder.getArrayAttr({}));
        mlir::Value fullProduct =
            circt::sv::ReadInOutOp::create(bodyBuilder, location, product);
        accessor.setOutput("data_output_0",
                           circt::comb::ExtractOp::create(bodyBuilder, location,
                                                          fullProduct, 0, 16));
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
}

llvm::Expected<FabricOperationProviderOutput>
materializeIntelAlteraScalarIntegerMultiply(
    FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::IntelAltera ||
      request.externalImplementationContractRef !=
          nativeContract().contractRef ||
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

  materializeLpmMult(request);
  FabricOperationProviderOutput output;
  output.payloads.push_back(
      {PayloadRole::BlackBoxContract,
       nativeContract().blackBoxPayloadLogicalName.str(),
       std::vector<std::uint8_t>(
           nativeContract().blackBoxContractBytes.bytes_begin(),
           nativeContract().blackBoxContractBytes.bytes_end())});
  output.externalImplementationBindings.push_back(
      {nativeContract().contractRef.str(),
       std::vector<ExternalInputBinding>(request.externalInputs.begin(),
                                         request.externalInputs.end()),
       {},
       {{RepresentationObjectKind::Module, nativeContract().moduleName.str()}},
       ImplementationPayloadKey{
           PayloadRole::BlackBoxContract,
           nativeContract().blackBoxPayloadLogicalName.str()}});
  return output;
}

} // namespace

llvm::Error registerIntelAlteraScalarIntegerMultiplyProvider(
    FabricOperationProviderRegistry &registry) {
  return registry.add({::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                       BackendRecipeKey::IntelAltera,
                       nativeContract().contractRef.str(),
                       materializeIntelAlteraScalarIntegerMultiply});
}

llvm::Error registerIntelAlteraLpmMultExternalImplementationContract(
    ExternalImplementationContractCatalog &catalog) {
  auto builtins = makeFpgaNativeExternalImplementationContractCatalog();
  if (!builtins)
    return builtins.takeError();
  auto contract = builtins->find(nativeContract().contractRef);
  if (!contract)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Intel LPM_MULT contract is not registered");
  return catalog.add(std::move(*contract));
}

} // namespace loom::hardware::rtl
