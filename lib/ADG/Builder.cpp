#include "ADG/Builder.h"

#include "Fabric/IR/ResourceContractRecord.h"

#include "BuilderInternal.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/FuCapabilityDomain.h"

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

llvm::Error checkDomainHandleOwner(const std::shared_ptr<DesignState> &state,
                                   std::size_t rootOrdinal,
                                   const std::weak_ptr<DesignState> &owner,
                                   std::size_t handleRootOrdinal,
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

std::optional<loom::fabric::FabricEntityKind>
moduleOccurrenceKind(mlir::Operation *operation) {
  using loom::fabric::FabricEntityKind;
  if (mlir::isa<::fabric::ModuleOp>(operation))
    return FabricEntityKind::FabricModuleTemplate;
  if (mlir::isa<::fabric::PeOp>(operation))
    return FabricEntityKind::FabricPeOccurrence;
  if (mlir::isa<::fabric::FuOp>(operation))
    return FabricEntityKind::FabricFuOccurrence;
  if (mlir::isa<::fabric::MemOp>(operation))
    return FabricEntityKind::FabricMemoryOccurrence;
  if (mlir::isa<::fabric::SwitchOp>(operation))
    return FabricEntityKind::FabricSwitchOccurrence;
  if (mlir::isa<::fabric::FifoOp>(operation))
    return FabricEntityKind::FabricFifoOccurrence;
  if (mlir::isa<::fabric::BoundaryOp>(operation))
    return FabricEntityKind::FabricBoundaryOccurrence;
  return std::nullopt;
}

void assignAuthoringEntityIds(::fabric::ModuleOp root) {
  loom::fabric::FabricEntityId next = 0;
  root->walk([&](mlir::Operation *operation) {
    if (!moduleOccurrenceKind(operation))
      return;
    operation->setAttr(
        ::fabric::kEntityIdAttrName,
        ::fabric::EntityIdAttr::get(operation->getContext(), next++));
  });
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
  if (!(*fu)->named && !(*state)->pes[peOrdinal_].named)
    if (llvm::Error error =
            (*state)
                ->spatialRoots[rootOrdinal_]
                .domainRelation.noteInternalMember(operation.getOperation(),
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
  if (!(*fu)->named && !(*state)->pes[peOrdinal_].named)
    if (llvm::Error error =
            (*state)
                ->spatialRoots[rootOrdinal_]
                .domainRelation.noteInternalMember(mux.getOperation(),
                                                   DomainMemberRole::FuNode, 0))
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
  if (!(*fu)->named && !(*state)->pes[peOrdinal_].named)
    if (llvm::Error error =
            (*state)
                ->spatialRoots[rootOrdinal_]
                .domainRelation.noteInternalMember(demux.getOperation(),
                                                   DomainMemberRole::FuNode, 0))
      return std::move(error);

  return FuNode(*state, rootOrdinal_, peOrdinal_, fuOrdinal_,
                demux.getOperation());
}

llvm::Error
FuBuilder::addCapabilityTemplate(const FuCapabilityTemplateSpec &spec) {
  auto handle = addCapabilityTemplateImpl(spec, false);
  if (!handle)
    return handle.takeError();
  return llvm::Error::success();
}

llvm::Expected<FuCapabilityTemplateHandle>
FuBuilder::addCapabilityTemplateWithHandle(
    const FuCapabilityTemplateSpec &spec) {
  return addCapabilityTemplateImpl(spec, true);
}

llvm::Expected<FuCapabilityTemplateHandle>
FuBuilder::addCapabilityTemplateImpl(const FuCapabilityTemplateSpec &spec,
                                     bool exposeHandle) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if (exposeHandle && (*fu)->named)
    return invalid("named FU capability rows do not have a unique finalized "
                   "occurrence handle");
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
  const std::size_t draftOrdinal = (*fu)->capabilityTemplates.size();
  (*fu)->capabilityTemplates.push_back(std::move(draft));
  (*fu)->capabilityTemplates.back().handleExposed = exposeHandle;
  return FuCapabilityTemplateHandle((*state)->identity, rootOrdinal_,
                                    fuOrdinal_, draftOrdinal);
}

llvm::Error FuBuilder::close(llvm::ArrayRef<FuValue> outputs) {
  return closeImpl(outputs, false);
}

llvm::Error FuBuilder::closeImpl(llvm::ArrayRef<FuValue> outputs,
                                 bool templateClose) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto fu = activeFu(*state, rootOrdinal_, peOrdinal_, fuOrdinal_);
  if (!fu)
    return fu.takeError();
  if ((*fu)->closed)
    return invalid("FU is already closed");
  if ((*fu)->named != templateClose)
    return invalid((*fu)->named
                       ? "named FU must be closed with closeTemplate"
                       : "anonymous FU cannot be closed with closeTemplate");
  if (!(*fu)->unresolvedBackedges.empty())
    return invalid("FU contains an unresolved backedge");
  llvm::SmallVector<mlir::Type, 4> outerOutputTypes;
  if ((*fu)->named) {
    auto functionTypeAttr = (*fu)->operation.getFunctionTypeAttr();
    auto functionType =
        functionTypeAttr
            ? mlir::dyn_cast<mlir::FunctionType>(functionTypeAttr.getValue())
            : mlir::FunctionType();
    if (!functionType)
      return invalid("named FU has no function signature");
    outerOutputTypes.append(functionType.getResults().begin(),
                            functionType.getResults().end());
  } else {
    outerOutputTypes.append((*fu)->operation.getResultTypes().begin(),
                            (*fu)->operation.getResultTypes().end());
  }
  if (outputs.size() != outerOutputTypes.size())
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
    const std::vector<::fabric::FuCapabilityTemplateSelection>
        sourceSelections = selections;
    auto domain =
        ::fabric::FuCapabilityDomainRecord::create(std::move(selections));
    if (!domain)
      return domain.takeError();
    for (auto [draft, source] :
         llvm::zip_equal((*fu)->capabilityTemplates, sourceSelections)) {
      auto normalized =
          ::fabric::FuCapabilityDomainRecord::create({std::move(source)});
      if (!normalized)
        return normalized.takeError();
      auto found =
          llvm::find(domain->templates(), normalized->templates().front());
      if (found == domain->templates().end())
        return invalid("FU capability row was lost during normalization");
      draft.canonicalOrdinal = static_cast<loom::fabric::FabricOrdinal>(
          found - domain->templates().begin());
    }
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
    mlir::Type outerType = outerOutputTypes[ordinal];
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

  if (!(*pe)->named)
    if (llvm::Error error =
            (*state)
                ->spatialRoots[rootOrdinal_]
                .domainRelation.noteInternalMember(
                    operation.getOperation(), DomainMemberRole::Occurrence, 0))
      return std::move(error);
  const std::size_t ordinal = (*state)->fus.size();
  (*state)->fus.push_back(detail::FuState{
      operation, rootOrdinal_, peOrdinal_, false, false, {}, {}});
  return FuBuilder(*state, rootOrdinal_, peOrdinal_, ordinal,
                   operation.getOperation());
}

llvm::Error PeBuilder::close() { return closeImpl(false); }

llvm::Error PeBuilder::closeImpl(bool templateClose) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto pe = activePe(*state, rootOrdinal_, peOrdinal_);
  if (!pe)
    return pe.takeError();
  if ((*pe)->closed)
    return invalid("PE is already closed");
  if ((*pe)->named != templateClose)
    return invalid((*pe)->named
                       ? "named PE must be closed with closeTemplate"
                       : "anonymous PE cannot be closed with closeTemplate");

  bool hasFu = false;
  for (const detail::FuState &fu : (*state)->fus) {
    if (fu.peOrdinal != peOrdinal_)
      continue;
    hasFu |= !fu.named;
    if (!fu.closed)
      return invalid("PE contains an FU that is not closed");
  }
  for ([[maybe_unused]] ::fabric::InstantiateOp instance :
       (*pe)->operation.getBody().front().getOps<::fabric::InstantiateOp>())
    hasFu = true;
  if (!hasFu)
    return invalid("PE requires at least one FU");
  ::fabric::YieldOp namedYield;
  if ((*pe)->named) {
    mlir::OpBuilder builder(&(*state)->context);
    builder.setInsertionPointToEnd(&(*pe)->operation.getBody().front());
    namedYield = ::fabric::YieldOp::create(builder, (*pe)->operation.getLoc(),
                                           mlir::ValueRange{});
  }
  if (mlir::failed(mlir::verify((*pe)->operation))) {
    if (namedYield)
      namedYield.erase();
    return invalid("Fabric rejected the completed typed PE graph");
  }
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

llvm::Expected<std::vector<SpatialValue>> SpatialCoreBuilder::instantiate(
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
    if (!detail::BuilderSpecMaterializer::samePortKind(resolved->getType(),
                                                       innerType))
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
    if (llvm::Error error = checkDomainHandleOwner(
            *state, target.rootOrdinal_, binding.childSlot.state_,
            binding.childSlot.rootOrdinal_, "domain slot"))
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

llvm::Expected<FifoResult> SpatialCoreBuilder::addFifo(SpatialValue input,
                                                       const FifoSpec &spec) {
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
  if (!detail::BuilderSpecMaterializer::samePortKind(source->getType(),
                                                     outputType) ||
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
    if (!detail::BuilderSpecMaterializer::samePortKind(resolved->getType(),
                                                       inputType))
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
  if (inputs.size() != spec.inputTypes.size())
    return invalid("Switch input count does not match its typed contract");
  auto materialized =
      detail::BuilderSpecMaterializer::switchSpec((*state)->context, spec);
  if (!materialized)
    return materialized.takeError();

  llvm::SmallVector<mlir::Value, 8> values;
  bool hasNormalizedInput = false;
  for (auto [value, inputType] : llvm::zip(inputs, materialized->inputTypes)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    if (!detail::BuilderSpecMaterializer::samePortKind(resolved->getType(),
                                                       inputType))
      return invalid("Switch source and input port have different kinds");
    hasNormalizedInput |= resolved->getType() != inputType;
    values.push_back(*resolved);
  }

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto sw = ::fabric::SwitchOp::create(
      builder, root.operation.getLoc(), materialized->outputTypes, values,
      mlir::StringAttr(), mlir::TypeAttr(), spec.schedule,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(materialized->inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      materialized->hardwareParameters, mlir::DictionaryAttr());
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
  auto materialized =
      detail::BuilderSpecMaterializer::memory((*state)->context, spec);
  if (!materialized)
    return materialized.takeError();

  llvm::SmallVector<mlir::Value, 8> values;
  bool hasNormalizedInput = false;
  for (auto [value, inputType] : llvm::zip(inputs, materialized->inputTypes)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!mlir::isa<mlir::MemRefType>(resolved->getType()) &&
        !resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    if (!detail::BuilderSpecMaterializer::samePortKind(resolved->getType(),
                                                       inputType) ||
        (mlir::isa<mlir::MemRefType>(inputType) &&
         resolved->getType() != inputType))
      return invalid("memory source and input port have incompatible types");
    hasNormalizedInput |= resolved->getType() != inputType;
    values.push_back(*resolved);
  }
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto memory = ::fabric::MemOp::create(
      builder, root.operation.getLoc(), materialized->outputTypes, values,
      mlir::StringAttr(), mlir::TypeAttr(), materialized->contract,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(materialized->inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      mlir::ArrayAttr(), materialized->operationPorts);
  if (llvm::Error error = verifyNewOperation(memory, "memory"))
    return std::move(error);
  if (llvm::Error error = root.domainRelation.noteInternalMember(
          memory.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);
  std::vector<ModuleDomainMemberHandle> operationPorts;
  if (materialized->operationPortCount != 0) {
    operationPorts.reserve(materialized->operationPortCount);
    for (std::size_t port = 0; port < materialized->operationPortCount;
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
  if (materialized->hasLocalService) {
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
  if (inputs.size() != spec.inputTypes_.size())
    return invalid("PE input count does not match its typed contract");

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Type, 8> boundaryInputTypes;
  for (const SpatialValue &value : inputs) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    values.push_back(*resolved);
    boundaryInputTypes.push_back(resolved->getType());
  }
  auto materialized = detail::BuilderSpecMaterializer::pe(
      (*state)->context, boundaryInputTypes, spec, false);
  if (!materialized)
    return materialized.takeError();

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto operation = ::fabric::PeOp::create(
      builder, root.operation.getLoc(), materialized->outputTypes,
      mlir::StringAttr(), mlir::TypeAttr(), spec.schedule_, values,
      materialized->tagWidth, materialized->instructionCapacity,
      materialized->registerFifoCount, materialized->registerFifoDepth,
      materialized->registerFifoPorts, materialized->fuConfigurationMode,
      materialized->operandBufferMode, materialized->operandBufferSize);
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : materialized->bodyInputTypes)
    body->addArgument(type, operation.getLoc());

  if (llvm::Error error = root.domainRelation.noteInternalMember(
          operation.getOperation(), DomainMemberRole::Occurrence, 0))
    return std::move(error);
  for (std::size_t context = 0; context < materialized->instructionContexts;
       ++context)
    if (llvm::Error error = root.domainRelation.noteInternalMember(
            operation.getOperation(), DomainMemberRole::InstructionContext,
            context))
      return std::move(error);
  const std::size_t ordinal = (*state)->pes.size();
  (*state)->pes.push_back(
      detail::PeState{operation, rootOrdinal_, false, false});
  return PeBuilder(*state, rootOrdinal_, ordinal, operation.getOperation(),
                   materialized->instructionContexts);
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
    if (!detail::BuilderSpecMaterializer::samePortKind(resolved->getType(),
                                                       declared) ||
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
  bool hasRelaxation = false;
  for (auto [value, type] : llvm::zip(values, root.resultTypes)) {
    hasRelaxation |= value.getType() != type;
    declaredTypes.push_back(mlir::TypeAttr::get(type));
  }
  if (hasRelaxation)
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
      false,
      std::nullopt,
      {}});
  return SpatialCoreBuilder(state_, ordinal);
}

namespace {

llvm::Error
consumeClosedDesign(const std::shared_ptr<detail::DesignState> &state) {
  if (!state || state->consumed)
    return invalid("DesignBuilder is already consumed");
  for (const detail::SpatialRootState &root : state->spatialRoots) {
    if (!root.closed)
      return invalid("SpatialCore '" + root.label + "' is not closed");
  }
  for (const detail::SystemRootState &root : state->systemRoots)
    if (!root.closed)
      return invalid("System '" + root.label + "' is not closed");
  state->consumed = true;
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FinalizedFabricDesign> DesignBuilder::finalize() && {
  if (llvm::Error error = consumeClosedDesign(state_))
    return std::move(error);
  std::vector<loom::fabric::FinalizedFabricRoot> finalized;
  std::vector<FinalizedFabricDesign::FuCapabilityResolution>
      capabilityResolutions;
  finalized.reserve(state_->spatialRoots.size() + state_->systemRoots.size());
  for (auto [rootOrdinal, root] : llvm::enumerate(state_->spatialRoots)) {
    bool captureCapabilities = false;
    for (const detail::FuState &fu : state_->fus)
      if (fu.rootOrdinal == rootOrdinal)
        captureCapabilities |=
            llvm::any_of(fu.capabilityTemplates,
                         [](const auto &draft) { return draft.handleExposed; });
    if (!captureCapabilities) {
      auto result = loom::fabric::finalizeFabricRoot(
          root.operation, root.domainRelation, state_->store);
      if (!result)
        return result.takeError();
      finalized.push_back(std::move(*result));
      continue;
    }
    assignAuthoringEntityIds(root.operation);
    auto result =
        loom::fabric::finalizeFabricModuleWithCapabilityCorrespondence(
            root.operation, root.domainRelation, state_->store);
    if (!result)
      return result.takeError();
    for (auto [fuOrdinal, fu] : llvm::enumerate(state_->fus)) {
      if (fu.rootOrdinal != rootOrdinal)
        continue;
      auto sourceId = fu.operation->getAttrOfType<::fabric::EntityIdAttr>(
          ::fabric::kEntityIdAttrName);
      if (!sourceId)
        return invalid("FU authoring correspondence has no source identity");
      for (auto [draftOrdinal, draft] :
           llvm::enumerate(fu.capabilityTemplates)) {
        if (!draft.handleExposed)
          continue;
        if (!draft.canonicalOrdinal)
          return invalid("FU capability handle was not closed");
        const loom::fabric::FabricFuCapabilityTemplateCorrespondence *match =
            nullptr;
        for (const auto &candidate : result->capabilities) {
          if (candidate.source.fu.kind !=
                  loom::fabric::FabricEntityKind::FabricFuOccurrence ||
              candidate.source.fu.id != sourceId.getId() ||
              candidate.source.ordinal != *draft.canonicalOrdinal)
            continue;
          if (match)
            return invalid("FU capability handle has multiple canonical rows");
          match = &candidate;
        }
        if (!match)
          return invalid("FU capability handle has no canonical row");
        capabilityResolutions.push_back(
            {rootOrdinal,
             fuOrdinal,
             draftOrdinal,
             {result->root.reference().artifact, match->target}});
      }
    }
    finalized.push_back(std::move(result->root));
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
  return FinalizedFabricDesign(state_->identity, std::move(finalized),
                               std::move(capabilityResolutions));
}

llvm::Expected<ArtifactReference<loom::fabric::FabricFuCapabilityTemplateRef>>
FinalizedFabricDesign::resolve(const FuCapabilityTemplateHandle &handle) const {
  std::shared_ptr<detail::DesignIdentity> identity = handle.identity_.lock();
  if (!identity || identity.get() != identity_.get())
    return invalid("FU capability handle belongs to a foreign design");
  const FuCapabilityResolution *match = nullptr;
  for (const FuCapabilityResolution &candidate : capabilities_) {
    if (candidate.rootOrdinal != handle.rootOrdinal_ ||
        candidate.fuOrdinal != handle.fuOrdinal_ ||
        candidate.draftOrdinal != handle.draftOrdinal_)
      continue;
    if (match)
      return invalid("FU capability handle has multiple finalized targets");
    match = &candidate;
  }
  if (!match)
    return invalid("FU capability handle has no finalized target");
  return match->target;
}

llvm::Expected<std::vector<ArtifactIdentity>>
DesignBuilder::deriveRootIdentities() && {
  if (llvm::Error error = consumeClosedDesign(state_))
    return std::move(error);
  std::vector<ArtifactIdentity> identities;
  identities.reserve(state_->spatialRoots.size() + state_->systemRoots.size());
  for (const detail::SpatialRootState &root : state_->spatialRoots) {
    auto identity = loom::fabric::deriveFabricRootIdentity(
        root.operation, root.domainRelation, state_->store);
    if (!identity)
      return identity.takeError();
    identities.push_back(std::move(*identity));
  }
  for (const detail::SystemRootState &root : state_->systemRoots) {
    llvm::SmallVector<ArtifactRootReference, 4> importedModules;
    importedModules.reserve(root.importedModules.size());
    for (const detail::ImportedModuleState &module : root.importedModules)
      importedModules.push_back(module.reference);
    auto identity = loom::fabric::deriveFabricRootIdentity(
        root.operation, importedModules, state_->store);
    if (!identity)
      return identity.takeError();
    identities.push_back(std::move(*identity));
  }
  return identities;
}

llvm::Expected<loom::fabric::FinalizedFabricModuleProjection>
DesignBuilder::finalizeDerivedSpatialCoreWithCorrespondence() && {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  if (state_->spatialRoots.size() != 1 || !state_->systemRoots.empty())
    return invalid(
        "correspondence finalization requires one sole derived Module");
  const detail::SpatialRootState &root = state_->spatialRoots.front();
  if (!root.closed)
    return invalid("derived Module is not closed");
  if (!root.derivedParent)
    return invalid("correspondence finalization requires a derived Module");
  state_->consumed = true;
  return loom::fabric::finalizeFabricModuleWithCorrespondence(
      root.operation, root.domainRelation, state_->store);
}

llvm::Expected<loom::fabric::FinalizedFabricSystemProjection>
DesignBuilder::finalizeDerivedSystemWithCorrespondence() && {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  if (!state_->spatialRoots.empty() || state_->systemRoots.size() != 1)
    return invalid(
        "correspondence finalization requires one sole derived System");
  const detail::SystemRootState &root = state_->systemRoots.front();
  if (!root.closed)
    return invalid("derived System is not closed");
  if (!root.derivedParent)
    return invalid("correspondence finalization requires a derived System");

  llvm::SmallVector<ArtifactRootReference, 4> importedModules;
  importedModules.reserve(root.importedModules.size());
  for (const detail::ImportedModuleState &module : root.importedModules)
    importedModules.push_back(module.reference);
  state_->consumed = true;
  return loom::fabric::finalizeFabricSystemWithCorrespondence(
      root.operation, importedModules, state_->store);
}

} // namespace loom::adg
