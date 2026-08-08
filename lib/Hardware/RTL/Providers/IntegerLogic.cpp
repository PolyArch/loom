#include "Hardware/RTL/Providers/IntegerLogic.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

enum class LogicOperation { And, Or, Xor };

struct Mode final {
  LogicOperation operation;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_integer_logic_invalid: " + message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

llvm::Expected<LogicOperation>
lowerOperation(::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithAndI:
    return LogicOperation::And;
  case Schema::ArithOrI:
  case Schema::LLVMOrDisjoint:
    return LogicOperation::Or;
  case Schema::ArithXOrI:
    return LogicOperation::Xor;
  default:
    return invalid("Fabric returned a non-logic behavior witness");
  }
}

mlir::Value materializeOperation(mlir::OpBuilder &builder,
                                 mlir::Location location,
                                 LogicOperation operation, mlir::Value lhs,
                                 mlir::Value rhs) {
  switch (operation) {
  case LogicOperation::And:
    return circt::comb::AndOp::create(builder, location, lhs, rhs);
  case LogicOperation::Or:
    return circt::comb::OrOp::create(builder, location, lhs, rhs);
  case LogicOperation::Xor:
    return circt::comb::XorOp::create(builder, location, lhs, rhs);
  }
  llvm_unreachable("unknown integer logic operation");
}

bool hasExpectedParameterSchema(
    const fabric::ResolvedFabricOpCapabilityView &capability,
    ::fabric::ImplementationFamilyId family) {
  if (family == ::fabric::ImplementationFamilyId::ScalarIntegerLogic)
    return std::holds_alternative<::fabric::ScalarIntegerParams>(
        capability.parameterizedCapability);
  return std::holds_alternative<::fabric::FixedVectorIntegerParams>(
      capability.parameterizedCapability);
}

llvm::Expected<FabricOperationProviderOutput> materializePortableIntegerLogic(
    FabricOperationProviderRequest request,
    ::fabric::ImplementationFamilyId expectedFamily) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily != expectedFamily)
    return invalid("provider received a different implementation family");
  if (!hasExpectedParameterSchema(request.capability, expectedFamily))
    return invalid("capability has the wrong parameter schema");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       request.capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(&port);
    else
      return invalid("capability has a physical port with unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  if (inputs.size() != 2 || outputs.size() != 1 ||
      inputs[0]->reference.ordinal != 0 || inputs[1]->reference.ordinal != 1 ||
      outputs[0]->reference.ordinal != 0 || inputs[0]->payloadWidthBits == 0 ||
      inputs[1]->payloadWidthBits == 0 || outputs[0]->payloadWidthBits == 0)
    return unsupported(request);

  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("Fabric returned an empty behavior domain");

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<Mode> modes;
  modes.reserve(domain.size());
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid(
          "configuration-free capability has a non-singleton behavior domain");
    auto operation = lowerOperation(domain.front().representativeActor.schema);
    if (!operation)
      return operation.takeError();
    modes.push_back({*operation, nullptr});
  } else {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Finite)
      return invalid("configured logic relation is not finite");
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured logic capability requires one field");
    field = request.configurationAbi.findOperationField(
        request.occurrence,
        request.capability.configurationFieldSchema.front().ordinal);
    if (!field)
      return invalid("configured field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid(
          "codebook does not exactly cover the configuration domain");
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook has no entry for an admitted semantic value");
      auto operation = lowerOperation(point.representativeActor.schema);
      if (!operation)
        return operation.takeError();
      modes.push_back({*operation, entry});
    }
  }

  std::size_t inactiveMode = 0;
  if (field) {
    const auto inactive = llvm::find_if(modes, [&](const Mode &mode) {
      return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
          .equals(field->inactiveValue);
    });
    if (inactive == modes.end())
      return invalid("ABI inactive value is outside the behavior domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        const unsigned datapathWidth =
            std::max({inputs[0]->payloadWidthBits, inputs[1]->payloadWidthBits,
                      outputs[0]->payloadWidthBits});
        mlir::Value lhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_0"),
            datapathWidth);
        mlir::Value rhs = detail::resizeUnsigned(
            bodyBuilder, location, accessor.getInput("data_input_1"),
            datapathWidth);

        std::vector<mlir::Value> results;
        results.reserve(modes.size());
        for (const Mode &mode : modes) {
          mlir::Value result = materializeOperation(bodyBuilder, location,
                                                    mode.operation, lhs, rhs);
          results.push_back(detail::resizeUnsigned(
              bodyBuilder, location, result, outputs[0]->payloadWidthBits));
        }

        mlir::Value selectedResult = results[inactiveMode];
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          for (std::size_t index = 0; index < modes.size(); ++index) {
            if (index == inactiveMode)
              continue;
            mlir::Value code = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                detail::decodePhysicalCode(
                    modes[index].codebookEntry->physicalCode,
                    codebook->encodedBitCount));
            mlir::Value selected = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
            selectedResult = circt::comb::MuxOp::create(
                bodyBuilder, location, selected, results[index], selectedResult,
                true);
          }
        }
        accessor.setOutput("data_output_0", selectedResult);
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableScalarIntegerLogic(FabricOperationProviderRequest request) {
  return materializePortableIntegerLogic(
      request, ::fabric::ImplementationFamilyId::ScalarIntegerLogic);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableFixedVectorIntegerLogic(
    FabricOperationProviderRequest request) {
  return materializePortableIntegerLogic(
      request, ::fabric::ImplementationFamilyId::FixedVectorIntegerLogic);
}

} // namespace

llvm::Error registerPortableIntegerLogicProviders(
    FabricOperationProviderRegistry &registry) {
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        materializePortableScalarIntegerLogic}))
    return error;
  return registry.add(
      {::fabric::ImplementationFamilyId::FixedVectorIntegerLogic,
       BackendRecipeKey::PortableSystemVerilog,
       {},
       materializePortableFixedVectorIntegerLogic});
}

} // namespace loom::hardware::rtl
