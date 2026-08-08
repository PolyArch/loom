#include "ADG/Builder.h"

#include "Fabric/IR/ResourceContractRecord.h"

#include "BuilderInternal.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/FuCapabilityDomain.h"
#include "Fabric/IR/TemporalSwitchResourceContract.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

namespace loom::adg {
namespace detail {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_builder_invalid: " + message);
}

llvm::Expected<std::shared_ptr<DesignState>>
activeState(const std::weak_ptr<DesignState> &weak) {
  std::shared_ptr<DesignState> state = weak.lock();
  if (!state || state->consumed)
    return invalid("ADG Builder view is stale");
  return state;
}

llvm::Error checkDomainHandleOwner(
    const std::shared_ptr<DesignState> &state, std::size_t rootOrdinal,
    const std::weak_ptr<DesignState> &owner, std::size_t handleRootOrdinal,
    llvm::StringRef description) {
  std::shared_ptr<DesignState> handleState = owner.lock();
  if (!handleState)
    return invalid("SpatialCore " + description + " handle is stale");
  if (handleState.get() != state.get() || handleRootOrdinal != rootOrdinal)
    return invalid(description + " belongs to a foreign SpatialCore");
  return llvm::Error::success();
}

mlir::Type materializePortType(mlir::MLIRContext &context,
                               const PortType &type) {
  switch (type.kind()) {
  case PortType::Kind::Bits:
    return ::fabric::BitsType::get(&context, type.width());
  case PortType::Kind::TaggedBits:
    return ::fabric::BitsTagType::get(&context, type.width(), type.tagWidth());
  case PortType::Kind::Memory: {
    mlir::Type element =
        type.tagWidth() == 0
            ? mlir::Type(::fabric::BitsType::get(&context, type.width()))
            : mlir::Type(::fabric::BitsTagType::get(&context, type.width(),
                                                    type.tagWidth()));
    return mlir::MemRefType::get(type.shape(), element);
  }
  }
  llvm_unreachable("all PortType kinds are handled");
}

} // namespace detail

namespace {

using detail::activeState;
using detail::checkDomainHandleOwner;
using detail::invalid;
using detail::materializePortType;

using DomainMemberRole =
    ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;

bool sameFabricKind(mlir::Type left, mlir::Type right) {
  return (mlir::isa<::fabric::BitsType>(left) &&
          mlir::isa<::fabric::BitsType>(right)) ||
         (mlir::isa<::fabric::BitsTagType>(left) &&
          mlir::isa<::fabric::BitsTagType>(right)) ||
         (mlir::isa<mlir::MemRefType>(left) &&
          mlir::isa<mlir::MemRefType>(right));
}

} // namespace

detail::DesignState::DesignState(const loom::ArtifactStore &store)
    : context(mlir::MLIRContext::Threading::DISABLED), store(store) {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  draft = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
}

llvm::Expected<PortType> PortType::bits(std::uint32_t width) {
  return PortType(Kind::Bits, width, 0, {});
}

llvm::Expected<PortType> PortType::taggedBits(std::uint32_t width,
                                              std::uint32_t tagWidth) {
  if (tagWidth == 0)
    return invalid("tagged Fabric port requires a positive tag width");
  return PortType(Kind::TaggedBits, width, tagWidth, {});
}

llvm::Expected<PortType> PortType::memory(llvm::ArrayRef<std::int64_t> shape,
                                          const PortType &elementType) {
  if (elementType.kind() == Kind::Memory)
    return invalid("Fabric memory element cannot itself be a memory port");
  if (elementType.kind() == Kind::TaggedBits)
    return invalid("Fabric memref element must be untagged bits");
  if (elementType.width() == 0)
    return invalid("Fabric memref element requires a positive data width");
  for (std::int64_t extent : shape)
    if (extent <= 0 && extent != PortType::kDynamicExtent)
      return invalid("Fabric memory shape contains an invalid extent");
  return PortType(Kind::Memory, elementType.width(), elementType.tagWidth(),
                  std::vector<std::int64_t>(shape.begin(), shape.end()));
}

PeSpec PeSpec::spatial(std::vector<PortType> inputTypes,
                       std::vector<PortType> outputTypes) {
  return PeSpec(::fabric::Schedule::Spatial, std::move(inputTypes),
                std::move(outputTypes), std::nullopt);
}

PeSpec PeSpec::temporal(std::vector<PortType> inputTypes,
                        std::vector<PortType> outputTypes,
                        TemporalPeParameters parameters) {
  return PeSpec(::fabric::Schedule::Temporal, std::move(inputTypes),
                std::move(outputTypes), std::move(parameters));
}

BoundarySpec BoundarySpec::s2t(const PortType &dataInput,
                               const PortType &tagInput,
                               const PortType &taggedOutput) {
  return {
      ::fabric::BoundaryDirection::S2t, {dataInput, tagInput}, {taggedOutput}};
}

BoundarySpec BoundarySpec::t2s(const PortType &taggedInput,
                               llvm::ArrayRef<PortType> outputs) {
  return {::fabric::BoundaryDirection::T2s,
          {taggedInput},
          std::vector<PortType>(outputs.begin(), outputs.end())};
}

SwitchSpec
SwitchSpec::spatial(std::vector<PortType> inputTypes,
                    std::vector<PortType> outputTypes,
                    std::vector<std::vector<std::uint32_t>> sourcesByOutput) {
  return {::fabric::Schedule::Spatial,
          std::move(inputTypes),
          std::move(outputTypes),
          std::move(sourcesByOutput),
          std::nullopt,
          std::nullopt};
}

SwitchSpec SwitchSpec::temporal(
    std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
    std::vector<std::vector<std::uint32_t>> sourcesByOutput,
    std::uint32_t routeTableSize,
    std::optional<::fabric::TemporalSwitchGrantPolicy> grantPolicy) {
  return {::fabric::Schedule::Temporal,
          std::move(inputTypes),
          std::move(outputTypes),
          std::move(sourcesByOutput),
          routeTableSize,
          std::move(grantPolicy)};
}

MemoryEngineSpec MemoryEngineSpec::spatial(
    std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts) {
  return MemoryEngineSpec(::fabric::Schedule::Spatial, std::nullopt,
                          std::move(operationPorts));
}

MemoryEngineSpec MemoryEngineSpec::temporal(
    std::uint64_t residentContextCount,
    std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts) {
  return MemoryEngineSpec(::fabric::Schedule::Temporal, residentContextCount,
                          std::move(operationPorts));
}

llvm::Expected<MemoryConnectivitySpec> MemoryConnectivitySpec::create(
    ::fabric::MemoryConnectivityDeclaration declaration) {
  auto record = ::fabric::MemoryConnectivityContractRecord::create(
      std::move(declaration));
  if (!record)
    return record.takeError();
  auto bytes = ::fabric::encodeMemoryConnectivityContractRecord(*record);
  if (!bytes)
    return bytes.takeError();
  return MemoryConnectivitySpec(std::move(*bytes));
}

llvm::Expected<LocalMemoryServiceSpec> LocalMemoryServiceSpec::create(
    std::uint64_t capacityBytes,
    const ::fabric::MemoryServiceContractRecord &contract) {
  if (capacityBytes == 0)
    return invalid("local memory service requires a positive capacity");
  if (llvm::Error error =
          ::fabric::validateLocalMemoryServiceCapacity(contract, capacityBytes))
    return std::move(error);
  auto bytes = ::fabric::encodeMemoryServiceContractRecord(contract);
  if (!bytes)
    return bytes.takeError();
  return LocalMemoryServiceSpec(capacityBytes, std::move(*bytes));
}

llvm::Expected<MemorySpec>
MemorySpec::create(std::vector<PortType> inputTypes,
                   std::vector<PortType> outputTypes,
                   std::vector<std::uint32_t> managerInputOrdinals,
                   std::vector<std::uint32_t> subordinateOutputOrdinals,
                   std::optional<MemoryEngineSpec> engine,
                   std::optional<LocalMemoryServiceSpec> localService,
                   MemoryConnectivitySpec connectivity) {
  if (!engine && !localService)
    return invalid(
        "memory requires an Operation Engine or Local Memory Service");
  if (engine && engine->operationPorts_.empty())
    return invalid("memory Operation Engine requires an operation port");
  if (engine && engine->schedule_ == ::fabric::Schedule::Temporal &&
      (!engine->residentContextCount_ || *engine->residentContextCount_ == 0))
    return invalid(
        "temporal memory Operation Engine requires resident contexts");
  if (!engine && !managerInputOrdinals.empty())
    return invalid("manager endpoint requires a memory Operation Engine");
  if (!engine && !inputTypes.empty())
    return invalid("storage-only memory must have zero input ports");
  if (!engine && outputTypes.size() != subordinateOutputOrdinals.size())
    return invalid(
        "storage-only memory results must match its subordinate endpoints");
  if (engine && !localService && managerInputOrdinals.empty())
    return invalid("operation-engine-only memory requires a manager endpoint");
  return MemorySpec(std::move(inputTypes), std::move(outputTypes),
                    std::move(managerInputOrdinals),
                    std::move(subordinateOutputOrdinals), std::move(engine),
                    std::move(localService), std::move(connectivity));
}

namespace {

llvm::Error verifyNewOperation(mlir::Operation *operation,
                               llvm::StringRef description) {
  if (mlir::failed(mlir::verify(operation))) {
    operation->erase();
    return invalid("Fabric rejected the typed " + description + " operation");
  }
  return llvm::Error::success();
}

llvm::Expected<detail::PeState *>
activePe(const std::shared_ptr<detail::DesignState> &state,
         std::size_t rootOrdinal, std::size_t peOrdinal) {
  if (peOrdinal >= state->pes.size())
    return invalid("PE handle has an invalid owner ordinal");
  detail::PeState &pe = state->pes[peOrdinal];
  if (pe.rootOrdinal != rootOrdinal || !pe.operation)
    return invalid("PE handle has a foreign SpatialCore owner");
  return &pe;
}

llvm::Expected<detail::FuState *>
activeFu(const std::shared_ptr<detail::DesignState> &state,
         std::size_t rootOrdinal, std::size_t peOrdinal,
         std::size_t fuOrdinal) {
  if (fuOrdinal >= state->fus.size())
    return invalid("FU handle has an invalid owner ordinal");
  detail::FuState &fu = state->fus[fuOrdinal];
  if (fu.rootOrdinal != rootOrdinal || fu.peOrdinal != peOrdinal ||
      !fu.operation)
    return invalid("FU handle has a foreign PE owner");
  return &fu;
}

llvm::Expected<mlir::IntegerAttr> positiveI32Attr(mlir::MLIRContext &context,
                                                  std::uint32_t value,
                                                  llvm::StringRef field) {
  if (value == 0 || value > static_cast<std::uint32_t>(
                                std::numeric_limits<std::int32_t>::max()))
    return invalid(field + " must fit positive i32");
  return mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), value);
}

llvm::Expected<mlir::IntegerAttr> nonNegativeI32Attr(mlir::MLIRContext &context,
                                                     std::uint32_t value,
                                                     llvm::StringRef field) {
  if (value >
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return invalid(field + " must fit non-negative i32");
  return mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), value);
}

llvm::StringRef
fuConfigurationModeSpelling(FuConfigurationMode configurationMode) {
  switch (configurationMode) {
  case FuConfigurationMode::PerInstruction:
    return "per_instruction_fu_config";
  case FuConfigurationMode::PerFu:
    return "per_fu_config";
  }
  llvm_unreachable("all FU configuration modes are handled");
}

} // namespace

llvm::Expected<mlir::Value>
FuBuilder::resolveValue(const std::shared_ptr<detail::DesignState> &state,
                        const FuValue &value) const {
  std::shared_ptr<detail::DesignState> valueState = value.state_.lock();
  if (!valueState || valueState.get() != state.get() ||
      value.rootOrdinal_ != rootOrdinal_ || value.peOrdinal_ != peOrdinal_ ||
      value.fuOrdinal_ != fuOrdinal_ || !value.value_)
    return invalid("foreign FuValue cannot cross FU owners");
  return value.value_;
}

llvm::Expected<mlir::Operation *>
FuBuilder::resolveNode(const std::shared_ptr<detail::DesignState> &state,
                       const FuNode &node) const {
  std::shared_ptr<detail::DesignState> nodeState = node.state_.lock();
  if (!nodeState || nodeState.get() != state.get() ||
      node.rootOrdinal_ != rootOrdinal_ || node.peOrdinal_ != peOrdinal_ ||
      node.fuOrdinal_ != fuOrdinal_ || !node.operation_)
    return invalid("foreign FuNode cannot cross FU owners");
  return node.operation_;
}

llvm::Expected<FuValue> FuNode::output(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (!operation_ || operation_->getParentOp() != (*fu)->operation ||
      ordinal >= operation_->getNumResults())
    return invalid("FU node output ordinal is out of range");
  return FuValue(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                 operation_->getResult(ordinal));
}

llvm::Expected<FuValue> FuBuilder::input(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  mlir::Block &body = (*fu)->operation.getBody().front();
  if (ordinal >= body.getNumArguments())
    return invalid("FU input ordinal is out of range");
  return FuValue(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                 body.getArgument(ordinal));
}

llvm::Expected<FuBackedge> FuBuilder::createBackedge(const PortType &type) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (type.kind() != PortType::Kind::Bits)
    return invalid("FU backedge must carry untagged Fabric bits");

  mlir::Type resultType = materializePortType((*state)->context, type);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*fu)->operation.getBody().front());
  auto placeholder = mlir::UnrealizedConversionCastOp::create(
      builder, (*fu)->operation.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{});
  (*fu)->unresolvedBackedges.push_back(placeholder.getOperation());
  FuValue value(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                placeholder.getResult(0));
  return FuBackedge(value, placeholder.getOperation());
}

llvm::Error FuBuilder::resolveBackedge(FuBackedge &&backedge, FuValue source) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (!backedge.placeholder_)
    return invalid("FU backedge is already resolved or moved");

  auto placeholder = resolveValue(*state, backedge.value_);
  if (!placeholder)
    return placeholder.takeError();
  auto resolvedSource = resolveValue(*state, source);
  if (!resolvedSource)
    return resolvedSource.takeError();
  auto found = llvm::find((*fu)->unresolvedBackedges, backedge.placeholder_);
  if (found == (*fu)->unresolvedBackedges.end() ||
      placeholder->getDefiningOp() != backedge.placeholder_)
    return invalid("FU backedge does not belong to this unresolved graph");
  if (*placeholder == *resolvedSource)
    return invalid("FU backedge cannot resolve to its own placeholder");
  if (placeholder->getType() != resolvedSource->getType())
    return invalid("FU backedge source type does not match its declaration");

  mlir::Operation *placeholderOperation = backedge.placeholder_;
  placeholder->replaceAllUsesWith(*resolvedSource);
  backedge.value_ = FuValue();
  backedge.placeholder_ = nullptr;
  (*fu)->unresolvedBackedges.erase(found);
  placeholderOperation->erase();
  return llvm::Error::success();
}

llvm::Expected<FuNode>
FuBuilder::addOperation(llvm::ArrayRef<FuValue> inputs,
                        const OperationCapabilitySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (spec.enabledOperations.empty())
    return invalid("fabric.op capability requires an enabled operation");
  if (static_cast<std::uint32_t>(spec.implementationFamily) >=
      ::fabric::implementationFamilyCount())
    return invalid("fabric.op implementation family is not registered");

  llvm::SmallVector<mlir::Value, 8> values;
  for (const FuValue &input : inputs) {
    auto resolved = resolveValue(*state, input);
    if (!resolved)
      return resolved.takeError();
    if (!mlir::isa<::fabric::BitsType>(resolved->getType()))
      return invalid("fabric.op inputs must be untagged Fabric bits");
    values.push_back(*resolved);
  }

  llvm::SmallVector<mlir::Type, 4> outputTypes;
  for (const PortType &type : spec.outputTypes) {
    if (type.kind() != PortType::Kind::Bits)
      return invalid("fabric.op outputs must be untagged Fabric bits");
    outputTypes.push_back(materializePortType((*state)->context, type));
  }

  std::vector<::dataflow::OperationSchemaId> schemas = spec.enabledOperations;
  llvm::sort(schemas, [](auto left, auto right) {
    return static_cast<std::uint32_t>(left) < static_cast<std::uint32_t>(right);
  });
  if (std::adjacent_find(schemas.begin(), schemas.end()) != schemas.end())
    return invalid("fabric.op capability contains a duplicate operation");

  llvm::SmallVector<mlir::Attribute, 8> operationAttrs;
  for (::dataflow::OperationSchemaId schema : schemas) {
    if (static_cast<std::uint32_t>(schema) >=
        ::dataflow::operationSchemaCount())
      return invalid("fabric.op capability names an unregistered operation");
    operationAttrs.push_back(mlir::FlatSymbolRefAttr::get(
        &(*state)->context, ::dataflow::operationSchemaSpelling(schema)));
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*fu)->operation.getBody().front());
  auto operation = ::fabric::OpOp::create(
      builder, (*fu)->operation.getLoc(), outputTypes, values,
      ::fabric::ImplementationFamilyIdAttr::get(&(*state)->context,
                                                spec.implementationFamily),
      mlir::ArrayAttr::get(&(*state)->context, operationAttrs),
      ::fabric::getFamilyCapabilityParamsAttr(&(*state)->context,
                                              spec.hardwareParameters));
  auto contractBytes =
      ::fabric::encodeResourceContractRecord(spec.resourceContract);
  if (!contractBytes) {
    operation.erase();
    return contractBytes.takeError();
  }
  llvm::SmallVector<std::int8_t, 64> signedContractBytes;
  signedContractBytes.reserve(contractBytes->size());
  for (std::uint8_t byte : *contractBytes)
    signedContractBytes.push_back(static_cast<std::int8_t>(byte));
  operation->setAttr(
      ::fabric::kResourceContractRecordAttrName,
      mlir::DenseI8ArrayAttr::get(&(*state)->context, signedContractBytes));
  if (llvm::Error error = verifyNewOperation(operation, "operation capability"))
    return std::move(error);
  if (llvm::Error error = (*state)->spatialRoots[rootOrdinal_]
                              .domainRelation.noteInternalMember(
                                  operation.getOperation(),
                                  DomainMemberRole::FuNode, 0))
    return std::move(error);

  return FuNode(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                operation.getOperation());
}

llvm::Expected<FuNode> FuBuilder::addMux(llvm::ArrayRef<FuValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (inputs.size() < 2)
    return invalid("fabric.mux requires at least two inputs");

  llvm::SmallVector<mlir::Value, 4> values;
  mlir::Type type;
  for (const FuValue &input : inputs) {
    auto resolved = resolveValue(*state, input);
    if (!resolved)
      return resolved.takeError();
    if (!type)
      type = resolved->getType();
    if (resolved->getType() != type || !mlir::isa<::fabric::BitsType>(type))
      return invalid("fabric.mux inputs must have one Fabric bits type");
    values.push_back(*resolved);
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*fu)->operation.getBody().front());
  auto mux = ::fabric::MuxOp::create(builder, (*fu)->operation.getLoc(), type,
                                     values, mlir::IntegerAttr(),
                                     mlir::BoolAttr(), mlir::BoolAttr());
  if (llvm::Error error = verifyNewOperation(mux, "FU mux"))
    return std::move(error);
  if (llvm::Error error = (*state)->spatialRoots[rootOrdinal_]
                              .domainRelation.noteInternalMember(
                                  mux.getOperation(), DomainMemberRole::FuNode,
                                  0))
    return std::move(error);
  return FuNode(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                mux.getOperation());
}

llvm::Expected<FuNode> FuBuilder::addDemux(FuValue input,
                                           std::uint32_t outputCount) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (outputCount < 2)
    return invalid("fabric.demux requires at least two outputs");
  auto resolved = resolveValue(*state, input);
  if (!resolved)
    return resolved.takeError();
  if (!mlir::isa<::fabric::BitsType>(resolved->getType()))
    return invalid("fabric.demux input must be untagged Fabric bits");

  llvm::SmallVector<mlir::Type, 4> outputTypes(outputCount,
                                               resolved->getType());
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*fu)->operation.getBody().front());
  auto demux = ::fabric::DemuxOp::create(
      builder, (*fu)->operation.getLoc(), outputTypes, *resolved,
      mlir::IntegerAttr(), mlir::BoolAttr(), mlir::BoolAttr());
  if (llvm::Error error = verifyNewOperation(demux, "FU demux"))
    return std::move(error);
  if (llvm::Error error = (*state)->spatialRoots[rootOrdinal_]
                              .domainRelation.noteInternalMember(
                                  demux.getOperation(), DomainMemberRole::FuNode,
                                  0))
    return std::move(error);

  return FuNode(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                demux.getOperation());
}

llvm::Error
FuBuilder::addCapabilityTemplate(const FuCapabilityTemplateSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (spec.activeOperations.empty())
    return invalid("FU capability template requires an active operation");

  detail::FuCapabilityTemplateDraft draft;
  draft.activeOperations.reserve(spec.activeOperations.size());
  for (const FuNode &node : spec.activeOperations) {
    auto operation = resolveNode(*state, node);
    if (!operation)
      return operation.takeError();
    if (!mlir::isa<::fabric::OpOp>(*operation))
      return invalid("FU capability activation must name fabric.op");
    draft.activeOperations.push_back(*operation);
  }

  draft.routes.reserve(spec.routes.size());
  for (const FuRouteSelection &selection : spec.routes) {
    auto operation = resolveNode(*state, selection.selector);
    if (!operation)
      return operation.takeError();
    if (auto mux = mlir::dyn_cast<::fabric::MuxOp>(*operation)) {
      if (selection.selectedPort >= mux.getInputs().size())
        return invalid("FU capability mux input ordinal is out of range");
    } else if (auto demux = mlir::dyn_cast<::fabric::DemuxOp>(*operation)) {
      if (selection.selectedPort >= demux.getOutputs().size())
        return invalid("FU capability demux output ordinal is out of range");
    } else {
      return invalid(
          "FU capability route must name fabric.mux or fabric.demux");
    }
    draft.routes.emplace_back(*operation, selection.selectedPort);
  }
  (*fu)->capabilityTemplates.push_back(std::move(draft));
  return llvm::Error::success();
}

llvm::Error FuBuilder::close(llvm::ArrayRef<FuValue> outputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (!(*fu)->unresolvedBackedges.empty())
    return invalid("FU contains an unresolved backedge");
  if (outputs.size() != (*fu)->operation.getNumResults())
    return invalid("FU output count does not match its declaration");

  if (!(*fu)->capabilityTemplates.empty()) {
    llvm::DenseMap<mlir::Operation *, std::uint64_t> nodeOrdinals;
    std::uint64_t nextNode = 0;
    for (mlir::Operation &operation : (*fu)->operation.getBody().front())
      if (mlir::isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
              operation))
        nodeOrdinals[&operation] = nextNode++;

    std::vector<::fabric::FuCapabilityTemplateSelection> selections;
    selections.reserve((*fu)->capabilityTemplates.size());
    for (const detail::FuCapabilityTemplateDraft &draft :
         (*fu)->capabilityTemplates) {
      ::fabric::FuCapabilityTemplateSelection selection;
      for (mlir::Operation *operation : draft.activeOperations) {
        auto found = nodeOrdinals.find(operation);
        if (found == nodeOrdinals.end())
          return invalid("FU capability operation is absent from its body");
        selection.activeOperationNodeOrdinals.push_back(found->second);
      }
      for (const auto &[operation, selectedPort] : draft.routes) {
        auto found = nodeOrdinals.find(operation);
        if (found == nodeOrdinals.end())
          return invalid("FU capability selector is absent from its body");
        selection.routes.push_back({found->second, selectedPort});
      }
      selections.push_back(std::move(selection));
    }
    auto domain =
        ::fabric::FuCapabilityDomainRecord::create(std::move(selections));
    if (!domain)
      return domain.takeError();
    auto bytes = ::fabric::encodeFuCapabilityDomainRecord(*domain);
    if (!bytes)
      return bytes.takeError();
    std::vector<std::int8_t> signedBytes;
    signedBytes.reserve(bytes->size());
    for (std::uint8_t byte : *bytes)
      signedBytes.push_back(static_cast<std::int8_t>(byte));
    (*fu)->operation.setCapabilityTemplatesAttr(
        ::fabric::FuCapabilityDomainAttr::get(
            &(*state)->context,
            mlir::DenseI8ArrayAttr::get(&(*state)->context, signedBytes)));
  }

  llvm::SmallVector<mlir::Value, 4> values;
  llvm::SmallVector<mlir::Attribute, 4> declaredTypes;
  bool hasWidening = false;
  for (auto [ordinal, output] : llvm::enumerate(outputs)) {
    auto resolved = resolveValue(*state, output);
    if (!resolved)
      return resolved.takeError();
    mlir::Type outerType = (*fu)->operation.getResult(ordinal).getType();
    auto innerWidth = ::fabric::getFabricBitsWidth(resolved->getType());
    auto outerWidth = ::fabric::getFabricBitsWidth(outerType);
    if (!innerWidth || !outerWidth || *innerWidth > *outerWidth)
      return invalid(
          "FU output boundary requires bits inner width <= outer width");
    hasWidening |= resolved->getType() != outerType;
    values.push_back(*resolved);
    declaredTypes.push_back(mlir::TypeAttr::get(outerType));
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*fu)->operation.getBody().front());
  auto yield =
      ::fabric::YieldOp::create(builder, (*fu)->operation.getLoc(), values);
  if (hasWidening)
    yield->setAttr("declared_types",
                   mlir::ArrayAttr::get(&(*state)->context, declaredTypes));
  if (mlir::failed(mlir::verify((*fu)->operation))) {
    yield.erase();
    return invalid("Fabric rejected the completed typed FU graph");
  }
  (*fu)->closed = true;
  return llvm::Error::success();
}

llvm::Expected<mlir::Value>
PeBuilder::resolveValue(const std::shared_ptr<detail::DesignState> &state,
                        const PeValue &value) const {
  std::shared_ptr<detail::DesignState> valueState = value.state_.lock();
  if (!valueState || valueState.get() != state.get() ||
      value.rootOrdinal_ != rootOrdinal_ || value.peOrdinal_ != peOrdinal_ ||
      !value.value_)
    return invalid("foreign PeValue cannot cross PE owners");
  return value.value_;
}

llvm::Expected<PeValue> PeBuilder::input(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto pe = activePe(*state, rootOrdinal_, peOrdinal_);
  if (!pe)
    return pe.takeError();
  if ((*pe)->closed)
    return invalid("PE is already closed");
  mlir::Block &body = (*pe)->operation.getBody().front();
  if (ordinal >= body.getNumArguments())
    return invalid("PE input ordinal is out of range");
  return PeValue(*state, rootOrdinal_, peOrdinal_, body.getArgument(ordinal));
}

llvm::Expected<SpatialValue> PeBuilder::output(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto pe = activePe(*state, rootOrdinal_, peOrdinal_);
  if (!pe)
    return pe.takeError();
  if (ordinal >= (*pe)->operation.getNumResults())
    return invalid("PE output ordinal is out of range");
  return SpatialValue(*state, rootOrdinal_,
                      (*pe)->operation.getResult(ordinal));
}

llvm::Expected<FuBuilder> PeBuilder::addFu(llvm::ArrayRef<PeValue> inputs,
                                           const FuSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto pe = activePe(*state, rootOrdinal_, peOrdinal_);
  if (!pe)
    return pe.takeError();
  if ((*pe)->closed)
    return invalid("PE is already closed");
  if (inputs.size() != spec.inputTypes.size())
    return invalid("FU input count does not match its typed contract");

  llvm::SmallVector<mlir::Value, 4> values;
  llvm::SmallVector<mlir::Type, 4> innerInputTypes;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    mlir::Type innerType = materializePortType((*state)->context, type);
    auto outerWidth = ::fabric::getFabricBitsWidth(resolved->getType());
    auto innerWidth = ::fabric::getFabricBitsWidth(innerType);
    if (!outerWidth || !innerWidth || *outerWidth < *innerWidth)
      return invalid(
          "FU input boundary requires bits outer width >= inner width");
    values.push_back(*resolved);
    innerInputTypes.push_back(innerType);
  }
  for (const PortType &type : spec.outputTypes) {
    if (type.kind() != PortType::Kind::Bits)
      return invalid("FU outputs must be untagged Fabric bits");
    outputTypes.push_back(materializePortType((*state)->context, type));
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&(*pe)->operation.getBody().front());
  auto operation = ::fabric::FuOp::create(
      builder, (*pe)->operation.getLoc(), outputTypes, mlir::StringAttr(),
      mlir::TypeAttr(), ::fabric::FuCapabilityDomainAttr(), values);
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : innerInputTypes)
    body->addArgument(type, operation.getLoc());

  if (llvm::Error error = (*state)->spatialRoots[rootOrdinal_]
                              .domainRelation.noteInternalMember(
                                  operation.getOperation(),
                                  DomainMemberRole::Occurrence, 0))
    return std::move(error);
  const std::size_t ordinal = (*state)->fus.size();
  (*state)->fus.push_back(
      detail::FuState{operation, rootOrdinal_, peOrdinal_, false, {}, {}});
  return FuBuilder(*state, rootOrdinal_, peOrdinal_, ordinal,
                   operation.getOperation());
}

llvm::Error PeBuilder::close() {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto pe = activePe(*state, rootOrdinal_, peOrdinal_);
  if (!pe)
    return pe.takeError();
  if ((*pe)->closed)
    return invalid("PE is already closed");

  bool hasFu = false;
  for (const detail::FuState &fu : (*state)->fus) {
    if (fu.peOrdinal != peOrdinal_)
      continue;
    hasFu = true;
    if (!fu.closed)
      return invalid("PE contains an FU that is not closed");
  }
  if (!hasFu)
    return invalid("PE requires at least one FU");
  if (mlir::failed(mlir::verify((*pe)->operation)))
    return invalid("Fabric rejected the completed typed PE graph");
  (*pe)->closed = true;
  return llvm::Error::success();
}

llvm::Expected<mlir::Value> SpatialCoreBuilder::resolveValue(
    const std::shared_ptr<detail::DesignState> &state,
    const SpatialValue &value) const {
  std::shared_ptr<detail::DesignState> valueState = value.state_.lock();
  if (!valueState || valueState.get() != state.get() ||
      value.rootOrdinal_ != rootOrdinal_ || !value.value_)
    return invalid("foreign SpatialValue cannot cross SpatialCore owners");
  return value.value_;
}

llvm::Expected<SpatialValue>
SpatialCoreBuilder::input(std::size_t ordinal) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  mlir::Block &body = root.operation.getBody().front();
  if (ordinal >= body.getNumArguments())
    return invalid("SpatialCore input ordinal is out of range");
  return SpatialValue(*state, rootOrdinal_, body.getArgument(ordinal));
}

llvm::Expected<SpatialBackedge>
SpatialCoreBuilder::createBackedge(const PortType &type) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");

  mlir::Type resultType = materializePortType((*state)->context, type);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto placeholder = mlir::UnrealizedConversionCastOp::create(
      builder, root.operation.getLoc(), mlir::TypeRange{resultType},
      mlir::ValueRange{});
  root.unresolvedBackedges.push_back(placeholder.getOperation());
  SpatialValue value(*state, rootOrdinal_, placeholder.getResult(0));
  return SpatialBackedge(value, placeholder.getOperation());
}

llvm::Error SpatialCoreBuilder::resolveBackedge(SpatialBackedge &&backedge,
                                                SpatialValue source) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (!backedge.placeholder_)
    return invalid("SpatialCore backedge is already resolved or moved");

  auto placeholder = resolveValue(*state, backedge.value_);
  if (!placeholder)
    return placeholder.takeError();
  auto resolvedSource = resolveValue(*state, source);
  if (!resolvedSource)
    return resolvedSource.takeError();
  auto found = llvm::find(root.unresolvedBackedges, backedge.placeholder_);
  if (found == root.unresolvedBackedges.end() ||
      placeholder->getDefiningOp() != backedge.placeholder_)
    return invalid(
        "SpatialCore backedge does not belong to this unresolved graph");
  if (*placeholder == *resolvedSource)
    return invalid(
        "SpatialCore backedge cannot resolve to its own placeholder");
  if (placeholder->getType() != resolvedSource->getType())
    return invalid(
        "SpatialCore backedge source type does not match its declaration");

  mlir::Operation *placeholderOperation = backedge.placeholder_;
  placeholder->replaceAllUsesWith(*resolvedSource);
  backedge.value_ = SpatialValue();
  backedge.placeholder_ = nullptr;
  root.unresolvedBackedges.erase(found);
  placeholderOperation->erase();
  return llvm::Error::success();
}

llvm::Expected<std::vector<SpatialValue>>
SpatialCoreBuilder::instantiate(
    const SpatialCoreBuilder &target, llvm::ArrayRef<SpatialValue> inputs,
    llvm::ArrayRef<ModuleInstanceDomainSlotBinding> domainBindings) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto targetState = activeState(target.state_);
  if (!targetState)
    return targetState.takeError();
  if (state->get() != targetState->get())
    return invalid("SpatialCore template belongs to a different design");
  if (rootOrdinal_ >= (*state)->spatialRoots.size() ||
      target.rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  if (rootOrdinal_ == target.rootOrdinal_)
    return invalid("SpatialCore cannot instantiate itself");

  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  detail::SpatialRootState &targetRoot =
      (*state)->spatialRoots[target.rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (!targetRoot.closed)
    return invalid("SpatialCore template must be closed before instantiation");

  mlir::FunctionType signature = targetRoot.operation.getFunctionType();
  if (inputs.size() != signature.getNumInputs())
    return invalid("SpatialCore template input count does not match its "
                   "declared signature");

  llvm::SmallVector<mlir::Value, 8> resolvedInputs;
  llvm::SmallVector<mlir::Type, 8> innerInputTypes;
  bool hasNormalizedInput = false;
  for (auto [input, innerType] : llvm::zip(inputs, signature.getInputs())) {
    auto resolved = resolveValue(*state, input);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    if (!sameFabricKind(resolved->getType(), innerType))
      return invalid("SpatialCore template source and input port have "
                     "different kinds");
    if (mlir::isa<mlir::MemRefType>(innerType) &&
        resolved->getType() != innerType)
      return invalid("SpatialCore template memory ports require exact types");
    hasNormalizedInput |= resolved->getType() != innerType;
    resolvedInputs.push_back(*resolved);
    innerInputTypes.push_back(innerType);
  }

  // Build the complete relation on a copy so every rejected call preserves
  // both the input values and the parent's domain-authoring state.
  ::fabric::ModuleDomainAuthoringRelation pendingDomain = root.domainRelation;
  std::vector<::fabric::ModuleInstanceDomainSlotBinding> domainRows;
  domainRows.reserve(domainBindings.size());
  for (const ModuleInstanceDomainSlotBinding &binding : domainBindings) {
    if (llvm::Error error =
            checkDomainHandleOwner(*state, target.rootOrdinal_,
                                   binding.childSlot.state_,
                                   binding.childSlot.rootOrdinal_,
                                   "domain slot"))
      return error;
    if (llvm::Error error = checkDomainHandleOwner(
            *state, rootOrdinal_, binding.parentSlot.state_,
            binding.parentSlot.rootOrdinal_, "domain slot"))
      return error;
    if (binding.childSlot.kind_ != binding.parentSlot.kind_)
      return invalid("module instance domain slot binding mixes different "
                     "slot kinds");
    domainRows.push_back({binding.childSlot.kind_, binding.childSlot.ordinal_,
                          binding.parentSlot.ordinal_});
  }
  llvm::sort(domainRows,
             [](const ::fabric::ModuleInstanceDomainSlotBinding &left,
                const ::fabric::ModuleInstanceDomainSlotBinding &right) {
               if (left.kind != right.kind)
                 return static_cast<std::uint32_t>(left.kind) <
                        static_cast<std::uint32_t>(right.kind);
               return left.childSlotOrdinal < right.childSlotOrdinal;
             });
  const ::fabric::ModuleDomainSlotCounts childCounts{
      targetRoot.domainRelation.declaredSlotCount(
          loom::fabric::FabricClockResetKind::Clock),
      targetRoot.domainRelation.declaredSlotCount(
          loom::fabric::FabricClockResetKind::Reset)};
  const ::fabric::ModuleDomainSlotCounts parentCounts{
      pendingDomain.declaredSlotCount(
          loom::fabric::FabricClockResetKind::Clock),
      pendingDomain.declaredSlotCount(
          loom::fabric::FabricClockResetKind::Reset)};
  if (llvm::Error error = ::fabric::validateModuleInstanceDomainSlotBindings(
          childCounts, parentCounts, domainRows))
    return invalid("module instance domain slot binding rejected: " +
                   llvm::toString(std::move(error)));

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto instance = ::fabric::InstantiateOp::create(
      builder, root.operation.getLoc(), signature.getResults(),
      targetRoot.operation.getSymName(), resolvedInputs,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(innerInputTypes)
                         : llvm::ArrayRef<mlir::Type>{},
      ::fabric::encodeModuleInstanceDomainSlotBindings(&(*state)->context,
                                                       domainRows));
  if (llvm::Error error = verifyNewOperation(instance, "module instance")) {
    return std::move(error);
  }
  if (llvm::Error error = pendingDomain.noteInstanceBindings(
          instance.getOperation(), targetRoot.domainRelation)) {
    instance.erase();
    return std::move(error);
  }
  root.domainRelation = std::move(pendingDomain);

  std::vector<SpatialValue> outputs;
  outputs.reserve(instance.getNumResults());
  for (mlir::Value output : instance.getResults())
    outputs.push_back(SpatialValue(*state, rootOrdinal_, output));
  return outputs;
}

llvm::Expected<FifoResult>
SpatialCoreBuilder::addFifo(SpatialValue input, const FifoSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  auto source = resolveValue(*state, input);
  if (!source)
    return source.takeError();
  if (!source->use_empty())
    return invalid("SpatialCore transport source already has a consumer");
  mlir::Type outputType =
      materializePortType((*state)->context, spec.outputType);
  if (!sameFabricKind(source->getType(), outputType) ||
      mlir::isa<mlir::MemRefType>(outputType))
    return invalid("FIFO input and output must have one transport kind");
  if (spec.maxDepth == 0 ||
      spec.maxDepth >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    return invalid("FIFO maxDepth must fit positive i32");

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto fifo = ::fabric::FifoOp::create(builder, root.operation.getLoc(),
                                       outputType, *source, spec.maxDepth,
                                       spec.bypassable, mlir::BoolAttr());
  if (llvm::Error error = verifyNewOperation(fifo, "FIFO"))
    return std::move(error);
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          fifo.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);
  return FifoResult(SpatialValue(*state, rootOrdinal_, fifo.getOutput()),
                    ModuleDomainMemberHandle::internal(
                        state_, rootOrdinal_, fifo.getOperation(),
                        DomainMemberRole::Occurrence, 0));
}

llvm::Expected<BoundaryResult>
SpatialCoreBuilder::addBoundary(llvm::ArrayRef<SpatialValue> inputs,
                                const BoundarySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (inputs.size() != spec.inputTypes.size())
    return invalid("Boundary input count does not match its typed contract");

  llvm::SmallVector<mlir::Value, 2> values;
  llvm::SmallVector<mlir::Type, 2> inputTypes;
  llvm::SmallVector<mlir::Type, 2> outputTypes;
  bool hasNormalizedInput = false;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    mlir::Type inputType = materializePortType((*state)->context, type);
    if (!sameFabricKind(resolved->getType(), inputType))
      return invalid("Boundary source and input port have different kinds");
    hasNormalizedInput |= resolved->getType() != inputType;
    values.push_back(*resolved);
    inputTypes.push_back(inputType);
  }
  for (const PortType &type : spec.outputTypes)
    outputTypes.push_back(materializePortType((*state)->context, type));

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto boundary = ::fabric::BoundaryOp::create(
      builder, root.operation.getLoc(), outputTypes, values, spec.direction,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      mlir::ArrayAttr(), mlir::DictionaryAttr());
  if (llvm::Error error = verifyNewOperation(boundary, "Boundary"))
    return std::move(error);
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          boundary.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);

  std::vector<SpatialValue> results;
  results.reserve(boundary.getNumResults());
  for (mlir::Value result : boundary.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return BoundaryResult(std::move(results),
                        ModuleDomainMemberHandle::internal(
                            state_, rootOrdinal_, boundary.getOperation(),
                            DomainMemberRole::Occurrence, 0));
}

llvm::Expected<SwitchResult>
SpatialCoreBuilder::addSwitch(llvm::ArrayRef<SpatialValue> inputs,
                              const SwitchSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (inputs.empty() || spec.outputTypes.empty())
    return invalid("Switch requires non-empty input and output sets");
  if (inputs.size() != spec.inputTypes.size())
    return invalid("Switch input count does not match its typed contract");
  if (spec.sourcesByOutput.size() != spec.outputTypes.size())
    return invalid("Switch connectivity row count does not match its outputs");
  if (spec.schedule == ::fabric::Schedule::Spatial && spec.routeTableSize)
    return invalid("Spatial switch cannot declare a route-table capacity");
  if (spec.schedule == ::fabric::Schedule::Spatial && spec.grantPolicy)
    return invalid("Spatial switch cannot declare temporal arbitration");
  if (spec.schedule == ::fabric::Schedule::Temporal &&
      (!spec.routeTableSize || *spec.routeTableSize == 0))
    return invalid("Temporal switch requires a positive route-table capacity");

  if (spec.schedule == ::fabric::Schedule::Temporal) {
    if (spec.inputTypes.size() > std::numeric_limits<std::uint32_t>::max() ||
        spec.outputTypes.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("Temporal switch port domain exceeds u32");
    auto resources = ::fabric::TemporalSwitchResourceContract::create(
        {static_cast<std::uint32_t>(spec.inputTypes.size()),
         static_cast<std::uint32_t>(spec.outputTypes.size()),
         spec.sourcesByOutput, spec.grantPolicy});
    if (!resources)
      return resources.takeError();
  }

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  bool hasNormalizedInput = false;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    mlir::Type inputType = materializePortType((*state)->context, type);
    if (!sameFabricKind(resolved->getType(), inputType))
      return invalid("Switch source and input port have different kinds");
    hasNormalizedInput |= resolved->getType() != inputType;
    values.push_back(*resolved);
    inputTypes.push_back(inputType);
  }
  for (const PortType &type : spec.outputTypes)
    outputTypes.push_back(materializePortType((*state)->context, type));

  std::vector<bool> inputCovered(inputs.size(), false);
  llvm::SmallVector<mlir::Attribute, 8> rows;
  for (llvm::ArrayRef<std::uint32_t> sources : spec.sourcesByOutput) {
    if (sources.empty())
      return invalid("Switch output has no physical input source");
    std::string row(inputs.size(), '0');
    for (std::uint32_t inputOrdinal : sources) {
      if (inputOrdinal >= inputs.size())
        return invalid("Switch connectivity input ordinal is out of range");
      const std::size_t position = inputs.size() - 1 - inputOrdinal;
      if (row[position] == '1')
        return invalid("Switch connectivity row contains a duplicate input");
      row[position] = '1';
      inputCovered[inputOrdinal] = true;
    }
    rows.push_back(mlir::StringAttr::get(&(*state)->context, row));
  }
  for (bool covered : inputCovered)
    if (!covered)
      return invalid("Switch input has no physical destination");

  mlir::NamedAttrList hardware;
  hardware.set("connectivity_table",
               mlir::ArrayAttr::get(&(*state)->context, rows));
  if (spec.routeTableSize)
    hardware.set(
        "route_table_size",
        mlir::IntegerAttr::get(mlir::IntegerType::get(&(*state)->context, 32),
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
                &(*state)->context,
                mlir::DenseI64ArrayAttr::get(&(*state)->context, requesters));
          } else {
            requesters.assign(selected.requesterCycle.begin(),
                              selected.requesterCycle.end());
            return ::fabric::SwitchRoundRobinAttr::get(
                &(*state)->context,
                mlir::DenseI64ArrayAttr::get(&(*state)->context, requesters),
                selected.resetRequester);
          }
        },
        *spec.grantPolicy);
    hardware.set(::fabric::kSwitchGrantPolicyParameterName, policy);
  }
  mlir::ArrayAttr hardwareParameters = mlir::ArrayAttr::get(
      &(*state)->context, {hardware.getDictionary(&(*state)->context)});

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto sw = ::fabric::SwitchOp::create(
      builder, root.operation.getLoc(), outputTypes, values, mlir::StringAttr(),
      mlir::TypeAttr(), spec.schedule,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      hardwareParameters, mlir::DictionaryAttr());
  if (llvm::Error error = verifyNewOperation(sw, "Switch"))
    return std::move(error);
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          sw.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);

  std::vector<SpatialValue> results;
  results.reserve(sw.getNumResults());
  for (mlir::Value result : sw.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return SwitchResult(std::move(results),
                      ModuleDomainMemberHandle::internal(
                          state_, rootOrdinal_, sw.getOperation(),
                          DomainMemberRole::Occurrence, 0));
}

llvm::Expected<MemoryResult>
SpatialCoreBuilder::addMemory(llvm::ArrayRef<SpatialValue> inputs,
                              const MemorySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (inputs.size() != spec.inputTypes_.size())
    return invalid("memory input count does not match its typed contract");

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  bool hasNormalizedInput = false;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes_)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!mlir::isa<mlir::MemRefType>(resolved->getType()) &&
        !resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    mlir::Type inputType = materializePortType((*state)->context, type);
    if (!sameFabricKind(resolved->getType(), inputType) ||
        (mlir::isa<mlir::MemRefType>(inputType) &&
         resolved->getType() != inputType))
      return invalid("memory source and input port have incompatible types");
    hasNormalizedInput |= resolved->getType() != inputType;
    values.push_back(*resolved);
    inputTypes.push_back(inputType);
  }
  for (const PortType &type : spec.outputTypes_)
    outputTypes.push_back(materializePortType((*state)->context, type));

  auto encodeOrdinals =
      [&](llvm::ArrayRef<std::uint32_t> ordinals, std::size_t endpointCount,
          llvm::StringRef role) -> llvm::Expected<mlir::DenseI32ArrayAttr> {
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
    return mlir::DenseI32ArrayAttr::get(&(*state)->context, encoded);
  };

  auto managers =
      encodeOrdinals(spec.managerInputOrdinals_, inputTypes.size(), "manager");
  if (!managers)
    return managers.takeError();
  auto subordinates = encodeOrdinals(spec.subordinateOutputOrdinals_,
                                     outputTypes.size(), "subordinate");
  if (!subordinates)
    return subordinates.takeError();

  mlir::FunctionType functionType =
      mlir::FunctionType::get(&(*state)->context, inputTypes, outputTypes);
  auto endpoints =
      ::fabric::deriveMemoryTransportEndpointInventory(functionType);
  if (!endpoints)
    return endpoints.takeError();

  ::fabric::MemoryEngineAttr engineAttr;
  mlir::ArrayAttr operationPortsAttr;
  if (spec.engine_) {
    ::fabric::MemoryResidentContextsAttr residentContexts;
    if (spec.engine_->residentContextCount_)
      residentContexts = ::fabric::MemoryResidentContextsAttr::get(
          &(*state)->context, *spec.engine_->residentContextCount_);
    engineAttr = ::fabric::MemoryEngineAttr::get(
        &(*state)->context, spec.engine_->schedule_, residentContexts);
    llvm::SmallVector<mlir::Attribute, 4> encodedPorts;
    encodedPorts.reserve(spec.engine_->operationPorts_.size());
    for (const ::fabric::MemoryOperationPortDeclaration &declaration :
         spec.engine_->operationPorts_) {
      auto record = ::fabric::MemoryOperationPortRecord::fromCanonical(
          &(*state)->context, spec.engine_->schedule_, *endpoints, declaration);
      if (!record)
        return record.takeError();
      auto bytes = ::fabric::encodeMemoryOperationPortRecord(*record);
      if (!bytes)
        return bytes.takeError();
      llvm::SmallVector<std::int8_t, 64> signedBytes;
      signedBytes.reserve(bytes->size());
      for (std::uint8_t byte : *bytes)
        signedBytes.push_back(static_cast<std::int8_t>(byte));
      encodedPorts.push_back(
          mlir::DenseI8ArrayAttr::get(&(*state)->context, signedBytes));
    }
    operationPortsAttr = mlir::ArrayAttr::get(&(*state)->context, encodedPorts);
  }

  ::fabric::LocalMemoryServiceAttr localServiceAttr;
  if (spec.localService_) {
    llvm::SmallVector<std::int8_t, 64> signedBytes;
    signedBytes.reserve(spec.localService_->contractBytes_.size());
    for (std::uint8_t byte : spec.localService_->contractBytes_)
      signedBytes.push_back(static_cast<std::int8_t>(byte));
    auto serviceContract = ::fabric::MemoryServiceContractAttr::get(
        &(*state)->context,
        mlir::DenseI8ArrayAttr::get(&(*state)->context, signedBytes));
    localServiceAttr = ::fabric::LocalMemoryServiceAttr::get(
        &(*state)->context, spec.localService_->capacityBytes_,
        serviceContract);
  }

  llvm::SmallVector<std::int8_t, 64> signedConnectivity;
  signedConnectivity.reserve(spec.connectivity_.canonicalBytes_.size());
  for (std::uint8_t byte : spec.connectivity_.canonicalBytes_)
    signedConnectivity.push_back(static_cast<std::int8_t>(byte));
  auto connectivityAttr = ::fabric::MemoryConnectivityContractAttr::get(
      &(*state)->context,
      mlir::DenseI8ArrayAttr::get(&(*state)->context, signedConnectivity));

  auto contract = ::fabric::MemoryContractAttr::get(
      &(*state)->context, engineAttr, localServiceAttr, connectivityAttr,
      *managers, *subordinates);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto memory = ::fabric::MemOp::create(
      builder, root.operation.getLoc(), outputTypes, values, mlir::StringAttr(),
      mlir::TypeAttr(), contract,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      mlir::ArrayAttr(), operationPortsAttr);
  if (llvm::Error error = verifyNewOperation(memory, "memory"))
    return std::move(error);
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          memory.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);
  std::vector<ModuleDomainMemberHandle> operationPorts;
  if (spec.engine_) {
    operationPorts.reserve(spec.engine_->operationPorts_.size());
    for (std::size_t port = 0; port < spec.engine_->operationPorts_.size();
         ++port) {
      if (llvm::Error error = root.domainRelation.noteInternalMember(
              memory.getOperation(), DomainMemberRole::MemoryOperationPort,
              port))
        return std::move(error);
      operationPorts.push_back(ModuleDomainMemberHandle::internal(
          state_, rootOrdinal_, memory.getOperation(),
          DomainMemberRole::MemoryOperationPort, port));
    }
  }
  std::optional<ModuleDomainMemberHandle> localService;
  if (spec.localService_) {
    if (llvm::Error error = root.domainRelation.noteInternalMember(
            memory.getOperation(), DomainMemberRole::LocalMemoryService, 0))
      return std::move(error);
    localService = ModuleDomainMemberHandle::internal(
        state_, rootOrdinal_, memory.getOperation(),
        DomainMemberRole::LocalMemoryService, 0);
  }

  std::vector<SpatialValue> results;
  results.reserve(memory.getNumResults());
  for (mlir::Value result : memory.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return MemoryResult(std::move(results),
                      ModuleDomainMemberHandle::internal(
                          state_, rootOrdinal_, memory.getOperation(),
                          DomainMemberRole::Occurrence, 0),
                      std::move(operationPorts), localService);
}

llvm::Expected<PeBuilder>
SpatialCoreBuilder::addPe(llvm::ArrayRef<SpatialValue> inputs,
                          const PeSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (inputs.empty() || spec.outputTypes_.empty())
    return invalid("PE requires non-empty input and output port sets");
  if (inputs.size() != spec.inputTypes_.size())
    return invalid("PE input count does not match its typed contract");

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Type, 8> innerInputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes_)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    values.push_back(*resolved);
    innerInputTypes.push_back(materializePortType((*state)->context, type));
  }
  for (const PortType &type : spec.outputTypes_)
    outputTypes.push_back(materializePortType((*state)->context, type));

  mlir::IntegerAttr tagWidth;
  mlir::IntegerAttr instructionCapacity;
  mlir::IntegerAttr registerFifoCount;
  mlir::IntegerAttr registerFifoDepth;
  mlir::IntegerAttr registerFifoPorts;
  mlir::StringAttr fuConfigurationMode;
  ::fabric::OperandBufferModeAttr operandBufferMode;
  mlir::IntegerAttr operandBufferSize;

  if (spec.schedule_ == ::fabric::Schedule::Spatial) {
    if (spec.temporal_)
      return invalid("spatial PE cannot carry temporal hardware parameters");
    auto width = ::fabric::getFabricBitsWidth(innerInputTypes.front());
    if (!width)
      return invalid("spatial PE inner ports must be untagged Fabric bits");
    for (auto [source, inner] : llvm::zip(values, innerInputTypes)) {
      if (!mlir::isa<::fabric::BitsType>(source.getType()) ||
          ::fabric::getFabricBitsWidth(inner) != width)
        return invalid(
            "spatial PE inputs require one uniform Fabric bits width");
    }
    for (mlir::Type output : outputTypes)
      if (::fabric::getFabricBitsWidth(output) != width)
        return invalid("spatial PE outputs require the uniform input width");
  } else {
    if (!spec.temporal_)
      return invalid("temporal PE requires temporal hardware parameters");
    auto firstBoundary =
        mlir::dyn_cast<::fabric::BitsTagType>(values.front().getType());
    if (!firstBoundary)
      return invalid("temporal PE inputs must be tagged Fabric bits");
    const std::uint32_t dataWidth = firstBoundary.getWidth();
    const std::uint32_t tagBits = firstBoundary.getTagWidth();
    for (auto [source, inner] : llvm::zip(values, innerInputTypes)) {
      auto boundary = mlir::dyn_cast<::fabric::BitsTagType>(source.getType());
      auto innerBits = mlir::dyn_cast<::fabric::BitsType>(inner);
      if (!boundary || boundary.getWidth() != dataWidth ||
          boundary.getTagWidth() != tagBits || !innerBits ||
          innerBits.getWidth() > dataWidth)
        return invalid(
            "temporal PE inputs violate its uniform tagged boundary");
    }
    for (mlir::Type output : outputTypes) {
      auto boundary = mlir::dyn_cast<::fabric::BitsTagType>(output);
      if (!boundary || boundary.getWidth() != dataWidth ||
          boundary.getTagWidth() != tagBits)
        return invalid(
            "temporal PE outputs violate its uniform tagged boundary");
    }

    const TemporalPeParameters &parameters = *spec.temporal_;
    auto tag = positiveI32Attr((*state)->context, tagBits, "PE tag width");
    if (!tag)
      return tag.takeError();
    tagWidth = *tag;
    auto instructions =
        positiveI32Attr((*state)->context, parameters.instructionCapacity,
                        "PE instruction capacity");
    if (!instructions)
      return instructions.takeError();
    instructionCapacity = *instructions;
    auto bufferSize =
        positiveI32Attr((*state)->context, parameters.operandBufferSize,
                        "PE operand-buffer size");
    if (!bufferSize)
      return bufferSize.takeError();
    operandBufferSize = *bufferSize;
    fuConfigurationMode = mlir::StringAttr::get(
        &(*state)->context,
        fuConfigurationModeSpelling(parameters.fuConfigurationMode));
    operandBufferMode = ::fabric::OperandBufferModeAttr::get(
        &(*state)->context, parameters.operandBufferMode);

    if (parameters.registerFifos) {
      const TemporalRegisterFifoParameters &fifos = *parameters.registerFifos;
      auto count = positiveI32Attr((*state)->context, fifos.count,
                                   "PE register-FIFO count");
      if (!count)
        return count.takeError();
      auto depth = positiveI32Attr((*state)->context, fifos.depth,
                                   "PE register-FIFO depth");
      if (!depth)
        return depth.takeError();
      if (fifos.ports != 1 && fifos.ports != 2)
        return invalid("PE register-FIFO ports must be one or two");
      auto ports = nonNegativeI32Attr((*state)->context, fifos.ports,
                                      "PE register-FIFO ports");
      if (!ports)
        return ports.takeError();
      registerFifoCount = *count;
      registerFifoDepth = *depth;
      registerFifoPorts = *ports;
    }
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto operation = ::fabric::PeOp::create(
      builder, root.operation.getLoc(), outputTypes, mlir::StringAttr(),
      mlir::TypeAttr(), spec.schedule_, values, tagWidth, instructionCapacity,
      registerFifoCount, registerFifoDepth, registerFifoPorts,
      fuConfigurationMode, operandBufferMode, operandBufferSize,
      mlir::BoolAttr(), mlir::ArrayAttr(), mlir::ArrayAttr());
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : innerInputTypes)
    body->addArgument(type, operation.getLoc());

  const std::size_t instructionContexts =
      spec.temporal_ ? spec.temporal_->instructionCapacity : 1;
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          operation.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);
  for (std::size_t context = 0; context < instructionContexts; ++context)
    if (llvm::Error error = root.domainRelation.noteInternalMember(
            operation.getOperation(), DomainMemberRole::InstructionContext,
            context))
      return std::move(error);
  const std::size_t ordinal = (*state)->pes.size();
  (*state)->pes.push_back(detail::PeState{operation, rootOrdinal_, false});
  return PeBuilder(*state, rootOrdinal_, ordinal, operation.getOperation(),
                   instructionContexts);
}

llvm::Error SpatialCoreBuilder::close(llvm::ArrayRef<SpatialValue> outputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (!root.unresolvedBackedges.empty())
    return invalid("SpatialCore contains an unresolved backedge");
  for (const detail::PeState &pe : (*state)->pes)
    if (pe.rootOrdinal == rootOrdinal_ && !pe.closed)
      return invalid("SpatialCore contains a PE that is not closed");
  if (outputs.size() != root.resultTypes.size())
    return invalid("SpatialCore output count does not match its declaration");

  llvm::SmallVector<mlir::Value, 4> values;
  for (auto [ordinal, pair] :
       llvm::enumerate(llvm::zip(outputs, llvm::ArrayRef(root.resultTypes)))) {
    const SpatialValue &output = std::get<0>(pair);
    mlir::Type declared = std::get<1>(pair);
    auto resolved = resolveValue(*state, output);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore output source already has a consumer");
    if (!sameFabricKind(resolved->getType(), declared) ||
        (mlir::isa<mlir::MemRefType>(declared) &&
         resolved->getType() != declared))
      return invalid(llvm::formatv(
          "SpatialCore output #{0} is incompatible with its declared type",
          ordinal));
    values.push_back(*resolved);
  }

  if (llvm::Error error = root.domainRelation.ensureDefaultAssignments(
          root.operation.getFunctionType().getNumInputs(),
          root.operation.getFunctionType().getNumResults()))
    return error;
  llvm::Error totality = root.domainRelation.validateTotality(
      root.operation.getFunctionType().getNumInputs(),
      root.operation.getFunctionType().getNumResults());
  if (totality)
    return totality;

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto yield =
      ::fabric::YieldOp::create(builder, root.operation.getLoc(), values);
  llvm::SmallVector<mlir::Attribute, 4> declaredTypes;
  for (mlir::Type type : root.resultTypes)
    declaredTypes.push_back(mlir::TypeAttr::get(type));
  yield->setAttr("declared_types",
                 mlir::ArrayAttr::get(&(*state)->context, declaredTypes));
  root.closed = true;
  return llvm::Error::success();
}

DesignBuilder::DesignBuilder(const loom::ArtifactStore &store)
    : state_(std::make_shared<detail::DesignState>(store)) {}

DesignBuilder::~DesignBuilder() = default;
DesignBuilder::DesignBuilder(DesignBuilder &&) noexcept = default;
DesignBuilder &DesignBuilder::operator=(DesignBuilder &&) noexcept = default;

llvm::Expected<SpatialCoreBuilder>
DesignBuilder::createSpatialCore(llvm::StringRef label,
                                 llvm::ArrayRef<PortType> inputs,
                                 llvm::ArrayRef<PortType> outputs) {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  if (label.empty())
    return invalid("SpatialCore diagnostic label cannot be empty");
  if (!state_->labels.insert(label).second)
    return invalid("duplicate SpatialCore diagnostic label '" + label + "'");

  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> resultTypes;
  for (const PortType &type : inputs)
    inputTypes.push_back(materializePortType(state_->context, type));
  for (const PortType &type : outputs)
    resultTypes.push_back(materializePortType(state_->context, type));
  mlir::FunctionType signature =
      mlir::FunctionType::get(&state_->context, inputTypes, resultTypes);

  mlir::OpBuilder builder(&state_->context);
  builder.setInsertionPointToEnd(state_->draft->getBody());
  ::fabric::ModuleOp root = ::fabric::ModuleOp::create(
      builder, state_->draft->getLoc(), label, signature, mlir::IntegerAttr(),
      mlir::IntegerAttr(), mlir::ArrayAttr(), mlir::ArrayAttr());
  mlir::Block *body = new mlir::Block();
  root.getBody().push_back(body);
  for (mlir::Type type : inputTypes)
    body->addArgument(type, root.getLoc());

  const std::size_t ordinal = state_->spatialRoots.size();
  state_->spatialRoots.push_back(detail::SpatialRootState{
      root,
      label.str(),
      std::vector<mlir::Type>(resultTypes.begin(), resultTypes.end()),
      {},
      {},
      false});
  return SpatialCoreBuilder(state_, ordinal);
}

llvm::Expected<FinalizedFabricDesign> DesignBuilder::finalize() && {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  for (const detail::SpatialRootState &root : state_->spatialRoots) {
    if (!root.closed)
      return invalid("SpatialCore '" + root.label + "' is not closed");
  }
  for (const detail::SystemRootState &root : state_->systemRoots)
    if (!root.closed)
      return invalid("System '" + root.label + "' is not closed");

  state_->consumed = true;
  std::vector<loom::fabric::FinalizedFabricRoot> finalized;
  finalized.reserve(state_->spatialRoots.size() + state_->systemRoots.size());
  for (const detail::SpatialRootState &root : state_->spatialRoots) {
    auto result = loom::fabric::finalizeFabricRoot(
        root.operation, root.domainRelation, state_->store);
    if (!result)
      return result.takeError();
    finalized.push_back(std::move(*result));
  }
  for (const detail::SystemRootState &root : state_->systemRoots) {
    llvm::SmallVector<ArtifactRootReference, 4> importedModules;
    importedModules.reserve(root.importedModules.size());
    for (const detail::ImportedModuleState &module : root.importedModules)
      importedModules.push_back(module.reference);
    auto result = loom::fabric::finalizeFabricRoot(
        root.operation, importedModules, state_->store);
    if (!result)
      return result.takeError();
    finalized.push_back(std::move(*result));
  }
  return FinalizedFabricDesign(std::move(finalized));
}

} // namespace loom::adg
