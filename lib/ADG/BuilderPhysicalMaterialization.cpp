#include "BuilderInternal.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <type_traits>

namespace loom::adg::detail {
namespace {

llvm::Expected<mlir::IntegerAttr> positiveI32(mlir::MLIRContext &context,
                                              std::uint32_t value,
                                              llvm::StringRef field) {
  if (value == 0 || value > static_cast<std::uint32_t>(
                                std::numeric_limits<std::int32_t>::max()))
    return invalid(field + " must fit positive i32");
  return mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), value);
}

llvm::Expected<mlir::IntegerAttr> nonNegativeI32(mlir::MLIRContext &context,
                                                 std::uint32_t value,
                                                 llvm::StringRef field) {
  if (value >
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return invalid(field + " must fit non-negative i32");
  return mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), value);
}

::fabric::FuConfigMode
materializeFuConfigurationMode(FuConfigurationMode mode) {
  switch (mode) {
  case FuConfigurationMode::PerInstruction:
    return ::fabric::FuConfigMode::PerInstructionFuConfig;
  case FuConfigurationMode::PerFu:
    return ::fabric::FuConfigMode::PerFuConfig;
  }
  llvm_unreachable("all FU configuration modes are handled");
}

llvm::Expected<mlir::DenseI32ArrayAttr>
encodeOrdinals(mlir::MLIRContext &context,
               llvm::ArrayRef<std::uint32_t> ordinals,
               std::size_t endpointCount, llvm::StringRef role) {
  llvm::SmallVector<std::int32_t, 4> encoded;
  encoded.reserve(ordinals.size());
  std::optional<std::uint32_t> previous;
  for (std::uint32_t ordinal : ordinals) {
    if (ordinal >= endpointCount)
      return invalid(role + " memory endpoint ordinal is out of range");
    if (ordinal >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
      return invalid(role + " memory endpoint ordinal does not fit i32");
    if (previous && ordinal <= *previous)
      return invalid(role +
                     " memory endpoint ordinals must be strictly increasing");
    previous = ordinal;
    encoded.push_back(static_cast<std::int32_t>(ordinal));
  }
  return mlir::DenseI32ArrayAttr::get(&context, encoded);
}

mlir::DenseI8ArrayAttr encodeBytes(mlir::MLIRContext &context,
                                   llvm::ArrayRef<std::uint8_t> bytes) {
  llvm::SmallVector<std::int8_t, 64> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(&context, signedBytes);
}

} // namespace

bool BuilderSpecMaterializer::samePortKind(mlir::Type left, mlir::Type right) {
  return (mlir::isa<::fabric::BitsType>(left) &&
          mlir::isa<::fabric::BitsType>(right)) ||
         (mlir::isa<::fabric::BitsTagType>(left) &&
          mlir::isa<::fabric::BitsTagType>(right)) ||
         (mlir::isa<mlir::MemRefType>(left) &&
          mlir::isa<mlir::MemRefType>(right));
}

llvm::Expected<PeMaterialization>
BuilderSpecMaterializer::pe(mlir::MLIRContext &context,
                            llvm::ArrayRef<mlir::Type> boundaryInputTypes,
                            const PeSpec &spec, bool namedTemplate) {
  if (boundaryInputTypes.empty() || spec.outputTypes_.empty() ||
      boundaryInputTypes.size() != spec.inputTypes_.size())
    return invalid("PE requires matching non-empty input and output port sets");

  PeMaterialization result;
  result.boundaryInputTypes.append(boundaryInputTypes.begin(),
                                   boundaryInputTypes.end());
  for (const PortType &type : spec.inputTypes_)
    result.bodyInputTypes.push_back(materializePortType(context, type));
  for (const PortType &type : spec.outputTypes_)
    result.outputTypes.push_back(materializePortType(context, type));

  if (spec.schedule_ == ::fabric::Schedule::Spatial) {
    if (spec.temporal_)
      return invalid("spatial PE cannot carry temporal hardware parameters");
    if (namedTemplate && result.boundaryInputTypes != result.bodyInputTypes)
      return invalid("spatial PE template boundary and body inputs differ");
    auto width = ::fabric::getFabricBitsWidth(result.bodyInputTypes.front());
    if (!width)
      return invalid("spatial PE body ports must be untagged Fabric bits");
    for (auto [boundary, inner] :
         llvm::zip(result.boundaryInputTypes, result.bodyInputTypes)) {
      if (!mlir::isa<::fabric::BitsType>(boundary) ||
          ::fabric::getFabricBitsWidth(inner) != width)
        return invalid("spatial PE inputs require one uniform width");
    }
    for (mlir::Type output : result.outputTypes)
      if (::fabric::getFabricBitsWidth(output) != width)
        return invalid("spatial PE outputs require the uniform input width");
    result.instructionContexts = 1;
    return result;
  }

  if (!spec.temporal_)
    return invalid("temporal PE requires temporal hardware parameters");
  auto firstBoundary =
      mlir::dyn_cast<::fabric::BitsTagType>(result.boundaryInputTypes.front());
  if (!firstBoundary)
    return invalid("temporal PE boundary must be tagged Fabric bits");
  const std::uint32_t dataWidth = firstBoundary.getWidth();
  const std::uint32_t tagBits = firstBoundary.getTagWidth();
  for (auto [boundaryType, innerType] :
       llvm::zip(result.boundaryInputTypes, result.bodyInputTypes)) {
    auto boundary = mlir::dyn_cast<::fabric::BitsTagType>(boundaryType);
    auto inner = mlir::dyn_cast<::fabric::BitsType>(innerType);
    if (!boundary || boundary.getWidth() != dataWidth ||
        boundary.getTagWidth() != tagBits || !inner ||
        inner.getWidth() > dataWidth)
      return invalid("temporal PE inputs violate its uniform tagged boundary");
  }
  for (mlir::Type outputType : result.outputTypes) {
    auto output = mlir::dyn_cast<::fabric::BitsTagType>(outputType);
    if (!output || output.getWidth() != dataWidth ||
        output.getTagWidth() != tagBits)
      return invalid("temporal PE outputs violate its uniform tagged boundary");
  }

  const TemporalPeParameters &parameters = *spec.temporal_;
  auto tag = positiveI32(context, tagBits, "PE tag width");
  if (!tag)
    return tag.takeError();
  result.tagWidth = *tag;
  auto instructions = positiveI32(context, parameters.instructionCapacity,
                                  "PE instruction capacity");
  if (!instructions)
    return instructions.takeError();
  result.instructionCapacity = *instructions;
  auto buffer = positiveI32(context, parameters.operandBufferSize,
                            "PE operand-buffer size");
  if (!buffer)
    return buffer.takeError();
  result.operandBufferSize = *buffer;
  result.fuConfigurationMode = ::fabric::FuConfigModeAttr::get(
      &context, materializeFuConfigurationMode(parameters.fuConfigurationMode));
  result.operandBufferMode = ::fabric::OperandBufferModeAttr::get(
      &context, parameters.operandBufferMode);
  if (parameters.registerFifos) {
    const TemporalRegisterFifoParameters &fifos = *parameters.registerFifos;
    auto count = positiveI32(context, fifos.count, "PE register-FIFO count");
    if (!count)
      return count.takeError();
    auto depth = positiveI32(context, fifos.depth, "PE register-FIFO depth");
    if (!depth)
      return depth.takeError();
    if (fifos.ports != 1 && fifos.ports != 2)
      return invalid("PE register-FIFO ports must be one or two");
    auto ports = nonNegativeI32(context, fifos.ports, "PE register-FIFO ports");
    if (!ports)
      return ports.takeError();
    result.registerFifoCount = *count;
    result.registerFifoDepth = *depth;
    result.registerFifoPorts = *ports;
  }
  result.instructionContexts = parameters.instructionCapacity;
  return result;
}

llvm::Expected<SwitchMaterialization>
BuilderSpecMaterializer::switchSpec(mlir::MLIRContext &context,
                                    const SwitchSpec &spec) {
  if (spec.inputTypes.empty() || spec.outputTypes.empty())
    return invalid("Switch requires non-empty input and output sets");
  if (spec.sourcesByOutput.size() != spec.outputTypes.size())
    return invalid("Switch connectivity row count does not match its outputs");
  if (spec.schedule == ::fabric::Schedule::Spatial &&
      (spec.routeTableSize || spec.grantPolicy))
    return invalid("Spatial switch cannot declare temporal state");
  if (spec.schedule == ::fabric::Schedule::Temporal &&
      (!spec.routeTableSize || *spec.routeTableSize == 0))
    return invalid("Temporal switch requires a positive route-table capacity");
  if (spec.inputTypes.size() > std::numeric_limits<std::uint32_t>::max() ||
      spec.outputTypes.size() > std::numeric_limits<std::uint32_t>::max())
    return invalid("Switch port domain exceeds u32");

  auto resources = ::fabric::SwitchResourceContract::create(
      {spec.schedule, static_cast<std::uint32_t>(spec.inputTypes.size()),
       static_cast<std::uint32_t>(spec.outputTypes.size()),
       spec.sourcesByOutput, spec.grantPolicy});
  if (!resources)
    return resources.takeError();

  SwitchMaterialization result;
  for (const PortType &type : spec.inputTypes)
    result.inputTypes.push_back(materializePortType(context, type));
  for (const PortType &type : spec.outputTypes)
    result.outputTypes.push_back(materializePortType(context, type));

  std::vector<bool> inputCovered(result.inputTypes.size(), false);
  llvm::SmallVector<mlir::Attribute, 8> rows;
  for (llvm::ArrayRef<std::uint32_t> sources : spec.sourcesByOutput) {
    if (sources.empty())
      return invalid("Switch output has no physical input source");
    std::string row(result.inputTypes.size(), '0');
    for (std::uint32_t inputOrdinal : sources) {
      if (inputOrdinal >= result.inputTypes.size())
        return invalid("Switch connectivity input ordinal is out of range");
      const std::size_t position = result.inputTypes.size() - 1 - inputOrdinal;
      if (row[position] == '1')
        return invalid("Switch connectivity row contains a duplicate input");
      row[position] = '1';
      inputCovered[inputOrdinal] = true;
    }
    rows.push_back(mlir::StringAttr::get(&context, row));
  }
  if (llvm::any_of(inputCovered, [](bool covered) { return !covered; }))
    return invalid("Switch input has no physical destination");

  mlir::NamedAttrList hardware;
  hardware.set("connectivity_table", mlir::ArrayAttr::get(&context, rows));
  if (spec.routeTableSize)
    hardware.set("route_table_size",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32),
                                        *spec.routeTableSize));
  if (spec.grantPolicy) {
    mlir::Attribute policy = std::visit(
        [&](auto &&selected) -> mlir::Attribute {
          using Policy = std::decay_t<decltype(selected)>;
          std::vector<std::int64_t> requesters;
          if constexpr (std::is_same_v<Policy,
                                       ::fabric::TemporalSwitchFixedPriority>) {
            requesters.assign(selected.requesterOrder.begin(),
                              selected.requesterOrder.end());
            return ::fabric::SwitchFixedPriorityAttr::get(
                &context, mlir::DenseI64ArrayAttr::get(&context, requesters));
          } else {
            requesters.assign(selected.requesterCycle.begin(),
                              selected.requesterCycle.end());
            return ::fabric::SwitchRoundRobinAttr::get(
                &context, mlir::DenseI64ArrayAttr::get(&context, requesters),
                selected.resetRequester);
          }
        },
        *spec.grantPolicy);
    hardware.set(::fabric::kSwitchGrantPolicyParameterName, policy);
  }
  result.hardwareParameters =
      mlir::ArrayAttr::get(&context, {hardware.getDictionary(&context)});
  return result;
}

llvm::Expected<MemoryMaterialization>
BuilderSpecMaterializer::memory(mlir::MLIRContext &context,
                                const MemorySpec &spec) {
  MemoryMaterialization result;
  for (const PortType &type : spec.inputTypes_)
    result.inputTypes.push_back(materializePortType(context, type));
  for (const PortType &type : spec.outputTypes_)
    result.outputTypes.push_back(materializePortType(context, type));

  auto managers = encodeOrdinals(context, spec.managerInputOrdinals_,
                                 result.inputTypes.size(), "manager");
  if (!managers)
    return managers.takeError();
  auto subordinates = encodeOrdinals(context, spec.subordinateOutputOrdinals_,
                                     result.outputTypes.size(), "subordinate");
  if (!subordinates)
    return subordinates.takeError();
  mlir::FunctionType signature =
      mlir::FunctionType::get(&context, result.inputTypes, result.outputTypes);
  auto endpoints = ::fabric::deriveMemoryTransportEndpointInventory(signature);
  if (!endpoints)
    return endpoints.takeError();

  ::fabric::MemoryEngineAttr engineAttr;
  if (spec.engine_) {
    ::fabric::MemoryResidentContextsAttr residentContexts;
    if (spec.engine_->residentContextCount_)
      residentContexts = ::fabric::MemoryResidentContextsAttr::get(
          &context, *spec.engine_->residentContextCount_);
    engineAttr = ::fabric::MemoryEngineAttr::get(
        &context, spec.engine_->schedule_, residentContexts);
    llvm::SmallVector<mlir::Attribute, 4> encodedPorts;
    encodedPorts.reserve(spec.engine_->operationPorts_.size());
    for (const ::fabric::MemoryOperationPortDeclaration &declaration :
         spec.engine_->operationPorts_) {
      auto record = ::fabric::MemoryOperationPortRecord::fromCanonical(
          &context, spec.engine_->schedule_, *endpoints, declaration);
      if (!record)
        return record.takeError();
      auto bytes = ::fabric::encodeMemoryOperationPortRecord(*record);
      if (!bytes)
        return bytes.takeError();
      encodedPorts.push_back(encodeBytes(context, *bytes));
    }
    result.operationPortCount = encodedPorts.size();
    result.operationPorts = mlir::ArrayAttr::get(&context, encodedPorts);
  }

  ::fabric::LocalMemoryServiceAttr localServiceAttr;
  if (spec.localService_) {
    auto serviceContract = ::fabric::MemoryServiceContractAttr::get(
        &context, encodeBytes(context, spec.localService_->contractBytes_));
    localServiceAttr = ::fabric::LocalMemoryServiceAttr::get(
        &context, spec.localService_->capacityBytes_, serviceContract);
    result.hasLocalService = true;
  }
  auto connectivity = ::fabric::MemoryConnectivityContractAttr::get(
      &context, encodeBytes(context, spec.connectivity_.canonicalBytes_));
  result.contract =
      ::fabric::MemoryContractAttr::get(&context, engineAttr, localServiceAttr,
                                        connectivity, *managers, *subordinates);
  return result;
}

} // namespace loom::adg::detail
