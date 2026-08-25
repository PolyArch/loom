#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <string>
#include <utility>

namespace loom::adg {
namespace {

using detail::activeState;
using detail::invalid;
using detail::materializePortType;
using DomainMemberRole =
    ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;

struct TemplateOwnerDescriptor final {
  std::vector<mlir::Operation *> instancePath;
  mlir::Operation *owner = nullptr;
  DomainMemberRole role = DomainMemberRole::Occurrence;
  loom::fabric::FabricOrdinal ordinal = 0;
};

llvm::Error requireAvailableSymbol(mlir::Operation *scope,
                                   llvm::StringRef label) {
  if (label.empty())
    return invalid("Fabric template label must not be empty");
  if (mlir::SymbolTable::lookupSymbolIn(scope, label))
    return invalid("Fabric template label is already defined in this scope");
  return llvm::Error::success();
}

llvm::Expected<mlir::FunctionType> templateSignature(mlir::Operation *target) {
  auto typeAttr = target
                      ? target->getAttrOfType<mlir::TypeAttr>("function_type")
                      : mlir::TypeAttr();
  auto functionType =
      typeAttr ? mlir::dyn_cast<mlir::FunctionType>(typeAttr.getValue())
               : mlir::FunctionType();
  if (!functionType)
    return invalid("Fabric template has no function signature");
  return functionType;
}

llvm::Expected<::fabric::InstantiateOp>
createInstance(mlir::MLIRContext &context, mlir::Location location,
               mlir::Block &parentBlock, mlir::Operation *target,
               llvm::ArrayRef<mlir::Value> inputs) {
  auto signature = templateSignature(target);
  if (!signature)
    return signature.takeError();
  if (inputs.size() != signature->getNumInputs())
    return invalid(
        "template instance input count does not match its signature");

  llvm::SmallVector<mlir::Type, 8> innerInputTypes;
  bool hasNormalizedInput = false;
  for (auto [input, inner] : llvm::zip(inputs, signature->getInputs())) {
    if (!detail::BuilderSpecMaterializer::samePortKind(input.getType(), inner))
      return invalid("template instance source and input have different kinds");
    if (mlir::isa<mlir::MemRefType>(inner) && input.getType() != inner)
      return invalid("template instance memory ports require exact types");
    hasNormalizedInput |= input.getType() != inner;
    innerInputTypes.push_back(inner);
  }

  auto symbol = target->getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (!symbol)
    return invalid("template instance target is not a named Fabric symbol");
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&parentBlock);
  auto instance = ::fabric::InstantiateOp::create(
      builder, location, signature->getResults(),
      mlir::FlatSymbolRefAttr::get(&context, symbol.getValue()), inputs,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(innerInputTypes)
                         : llvm::ArrayRef<mlir::Type>{},
      mlir::DenseI64ArrayAttr::get(&context, {}));
  if (mlir::failed(mlir::verify(instance))) {
    instance.erase();
    return invalid("Fabric rejected the typed non-Module instance");
  }
  return instance;
}

llvm::Error collectFuOwners(::fabric::FuOp target,
                            llvm::ArrayRef<mlir::Operation *> path,
                            std::vector<TemplateOwnerDescriptor> &owners) {
  owners.push_back({std::vector<mlir::Operation *>(path.begin(), path.end()),
                    target.getOperation(), DomainMemberRole::Occurrence, 0});
  for (mlir::Operation &operation : target.getBody().front())
    if (mlir::isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
            operation))
      owners.push_back(
          {std::vector<mlir::Operation *>(path.begin(), path.end()), &operation,
           DomainMemberRole::FuNode, 0});
  return llvm::Error::success();
}

llvm::Expected<std::vector<TemplateOwnerDescriptor>>
collectTemplateOwners(mlir::Operation *target) {
  std::vector<TemplateOwnerDescriptor> owners;
  if (auto pe = mlir::dyn_cast_or_null<::fabric::PeOp>(target)) {
    owners.push_back({{}, target, DomainMemberRole::Occurrence, 0});
    std::uint64_t contexts = 1;
    if (pe.getSchedule() == ::fabric::Schedule::Temporal) {
      auto count = pe.getNumInstruction();
      if (!count || *count <= 0)
        return invalid("temporal PE template has no instruction contexts");
      contexts = static_cast<std::uint64_t>(*count);
    }
    for (std::uint64_t ordinal = 0; ordinal < contexts; ++ordinal)
      owners.push_back(
          {{}, target, DomainMemberRole::InstructionContext, ordinal});

    mlir::SymbolTableCollection symbols;
    for (mlir::Operation &operation : pe.getBody().front()) {
      if (auto fu = mlir::dyn_cast<::fabric::FuOp>(operation)) {
        if (!fu.getSymNameAttr())
          if (llvm::Error error = collectFuOwners(fu, {}, owners))
            return std::move(error);
        continue;
      }
      auto instance = mlir::dyn_cast<::fabric::InstantiateOp>(operation);
      if (!instance)
        continue;
      auto fu = mlir::dyn_cast_or_null<::fabric::FuOp>(
          ::fabric::resolveInstantiateTarget(instance, symbols));
      if (!fu)
        return invalid("PE template contains a non-FU instance");
      mlir::Operation *path[] = {instance.getOperation()};
      if (llvm::Error error = collectFuOwners(fu, path, owners))
        return std::move(error);
    }
    return owners;
  }
  if (auto fu = mlir::dyn_cast_or_null<::fabric::FuOp>(target)) {
    if (llvm::Error error = collectFuOwners(fu, {}, owners))
      return std::move(error);
    return owners;
  }
  if (mlir::isa<::fabric::SwitchOp>(target)) {
    owners.push_back({{}, target, DomainMemberRole::Occurrence, 0});
    return owners;
  }
  if (auto memory = mlir::dyn_cast_or_null<::fabric::MemOp>(target)) {
    owners.push_back({{}, target, DomainMemberRole::Occurrence, 0});
    if (auto ports = memory.getMemoryOperationPortsAttr())
      for (std::size_t ordinal = 0; ordinal < ports.size(); ++ordinal)
        owners.push_back(
            {{}, target, DomainMemberRole::MemoryOperationPort, ordinal});
    if (memory.getMemoryContract().getLocalService())
      owners.push_back({{}, target, DomainMemberRole::LocalMemoryService, 0});
    return owners;
  }
  return invalid("operation is not a verifier-legal non-Module target");
}

llvm::Error
registerInstanceOwners(::fabric::ModuleDomainAuthoringRelation &relation,
                       mlir::Operation *instance, mlir::Operation *target) {
  auto owners = collectTemplateOwners(target);
  if (!owners)
    return owners.takeError();
  for (const TemplateOwnerDescriptor &owner : *owners) {
    llvm::SmallVector<mlir::Operation *, 4> path;
    path.push_back(instance);
    path.append(owner.instancePath.begin(), owner.instancePath.end());
    if (llvm::Error error = relation.noteInstantiatedMember(
            path, owner.owner, owner.role, owner.ordinal))
      return error;
  }
  return llvm::Error::success();
}

mlir::Operation *templateScope(detail::DesignState &state,
                               std::size_t rootOrdinal, detail::PeState &pe,
                               detail::FuState *fu = nullptr) {
  if (fu && fu->named)
    return fu->operation.getOperation();
  if (pe.named)
    return pe.operation.getOperation();
  return state.spatialRoots[rootOrdinal].operation.getOperation();
}

} // namespace

TemplatePhysicalOwnerHandle PeTemplateHandle::occurrenceOwner() const {
  return TemplatePhysicalOwnerHandle(identity_.lock(), rootOrdinal_, operation_,
                                     {}, operation_,
                                     DomainMemberRole::Occurrence, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle>
PeTemplateHandle::instructionContextOwner(std::size_t ordinal) const {
  if (ordinal >= instructionContexts_)
    return invalid("PE template instruction context ordinal is out of range");
  return TemplatePhysicalOwnerHandle(
      identity_.lock(), rootOrdinal_, operation_, {}, operation_,
      DomainMemberRole::InstructionContext, ordinal);
}

TemplatePhysicalOwnerHandle FuTemplateHandle::occurrenceOwner() const {
  return TemplatePhysicalOwnerHandle(identity_.lock(), rootOrdinal_, operation_,
                                     {}, operation_,
                                     DomainMemberRole::Occurrence, 0);
}

TemplatePhysicalOwnerHandle SwitchTemplateHandle::occurrenceOwner() const {
  return TemplatePhysicalOwnerHandle(identity_.lock(), rootOrdinal_, operation_,
                                     {}, operation_,
                                     DomainMemberRole::Occurrence, 0);
}

TemplatePhysicalOwnerHandle MemoryTemplateHandle::occurrenceOwner() const {
  return TemplatePhysicalOwnerHandle(identity_.lock(), rootOrdinal_, operation_,
                                     {}, operation_,
                                     DomainMemberRole::Occurrence, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle>
MemoryTemplateHandle::operationPortOwner(std::size_t ordinal) const {
  if (ordinal >= operationPorts_)
    return invalid("memory template operation port ordinal is out of range");
  return TemplatePhysicalOwnerHandle(
      identity_.lock(), rootOrdinal_, operation_, {}, operation_,
      DomainMemberRole::MemoryOperationPort, ordinal);
}

std::optional<TemplatePhysicalOwnerHandle>
MemoryTemplateHandle::localServiceOwner() const {
  if (!hasLocalService_)
    return std::nullopt;
  return TemplatePhysicalOwnerHandle(identity_.lock(), rootOrdinal_, operation_,
                                     {}, operation_,
                                     DomainMemberRole::LocalMemoryService, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle> FuNode::templateOwner() const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (fuOrdinal_ >= (*state)->fus.size() || peOrdinal_ >= (*state)->pes.size())
    return invalid("FU node has an invalid template owner");
  detail::FuState &fu = (*state)->fus[fuOrdinal_];
  detail::PeState &pe = (*state)->pes[peOrdinal_];
  if (!operation_ || operation_->getParentOp() != fu.operation)
    return invalid("FU node is outside its template owner");
  return TemplatePhysicalOwnerHandle(
      (*state)->identity, rootOrdinal_,
      templateScope(**state, rootOrdinal_, pe, &fu), {}, operation_,
      DomainMemberRole::FuNode, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle> FuBuilder::templateOwner() const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (fuOrdinal_ >= (*state)->fus.size() || peOrdinal_ >= (*state)->pes.size())
    return invalid("FU has an invalid template owner");
  detail::FuState &fu = (*state)->fus[fuOrdinal_];
  detail::PeState &pe = (*state)->pes[peOrdinal_];
  return TemplatePhysicalOwnerHandle(
      (*state)->identity, rootOrdinal_,
      templateScope(**state, rootOrdinal_, pe, &fu), {}, fu.operation,
      DomainMemberRole::Occurrence, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle> PeBuilder::templateOwner() const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (peOrdinal_ >= (*state)->pes.size())
    return invalid("PE has an invalid template owner");
  detail::PeState &pe = (*state)->pes[peOrdinal_];
  return TemplatePhysicalOwnerHandle((*state)->identity, rootOrdinal_,
                                     templateScope(**state, rootOrdinal_, pe),
                                     {}, pe.operation,
                                     DomainMemberRole::Occurrence, 0);
}

llvm::Expected<TemplatePhysicalOwnerHandle>
SpatialTemplateInstanceResult::project(
    const TemplatePhysicalOwnerHandle &owner) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = owner.identity_.lock();
  if (!identity || identity.get() != identity_.get() ||
      owner.rootOrdinal_ != rootOrdinal_ || owner.scope_ != target_ ||
      !owner.owner_)
    return invalid("physical owner belongs to a foreign template instance");
  std::vector<mlir::Operation *> path;
  path.reserve(owner.instancePath_.size() + 1);
  path.push_back(instance_);
  path.insert(path.end(), owner.instancePath_.begin(),
              owner.instancePath_.end());
  return TemplatePhysicalOwnerHandle(identity_, rootOrdinal_, parentScope_,
                                     std::move(path), owner.owner_, owner.role_,
                                     owner.ordinal_);
}

llvm::Expected<TemplatePhysicalOwnerHandle> PeTemplateInstanceResult::project(
    const TemplatePhysicalOwnerHandle &owner) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = owner.identity_.lock();
  if (!identity || identity.get() != identity_.get() ||
      owner.rootOrdinal_ != rootOrdinal_ || owner.scope_ != target_ ||
      !owner.owner_)
    return invalid("physical owner belongs to a foreign template instance");
  std::vector<mlir::Operation *> path;
  path.reserve(owner.instancePath_.size() + 1);
  path.push_back(instance_);
  path.insert(path.end(), owner.instancePath_.begin(),
              owner.instancePath_.end());
  return TemplatePhysicalOwnerHandle(identity_, rootOrdinal_, parentScope_,
                                     std::move(path), owner.owner_, owner.role_,
                                     owner.ordinal_);
}

llvm::Expected<FuBuilder> PeBuilder::createFuTemplate(llvm::StringRef label,
                                                      const FuSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (peOrdinal_ >= (*state)->pes.size())
    return invalid("PE handle has an invalid owner ordinal");
  detail::PeState &pe = (*state)->pes[peOrdinal_];
  if (pe.closed)
    return invalid("PE is already closed");
  if (llvm::Error error = requireAvailableSymbol(pe.operation, label))
    return std::move(error);

  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  for (const PortType &type : spec.inputTypes) {
    if (type.kind() != PortType::Kind::Bits)
      return invalid("FU template inputs must be untagged Fabric bits");
    inputTypes.push_back(materializePortType((*state)->context, type));
  }
  for (const PortType &type : spec.outputTypes) {
    if (type.kind() != PortType::Kind::Bits)
      return invalid("FU template outputs must be untagged Fabric bits");
    outputTypes.push_back(materializePortType((*state)->context, type));
  }
  mlir::FunctionType signature =
      mlir::FunctionType::get(&(*state)->context, inputTypes, outputTypes);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&pe.operation.getBody().front());
  auto operation = ::fabric::FuOp::create(
      builder, pe.operation.getLoc(), mlir::TypeRange{},
      mlir::StringAttr::get(&(*state)->context, label),
      mlir::TypeAttr::get(signature), ::fabric::FuCapabilityDomainAttr(),
      mlir::ValueRange{});
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : inputTypes)
    body->addArgument(type, operation.getLoc());

  const std::size_t ordinal = (*state)->fus.size();
  (*state)->fus.push_back(detail::FuState{
      operation, rootOrdinal_, peOrdinal_, true, false, {}, {}});
  return FuBuilder(*state, rootOrdinal_, peOrdinal_, ordinal,
                   operation.getOperation());
}

llvm::Expected<FuTemplateHandle>
FuBuilder::closeTemplate(llvm::ArrayRef<FuValue> outputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (fuOrdinal_ >= (*state)->fus.size() || !(*state)->fus[fuOrdinal_].named)
    return invalid("closeTemplate requires a named FU declaration");
  if (llvm::Error error = closeImpl(outputs, true))
    return std::move(error);
  return FuTemplateHandle((*state)->identity, rootOrdinal_, peOrdinal_,
                          (*state)->fus[fuOrdinal_].operation);
}

llvm::Expected<PeTemplateInstanceResult>
PeBuilder::instantiate(const FuTemplateHandle &target,
                       llvm::ArrayRef<PeValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = target.identity_.lock();
  if (!identity || identity.get() != (*state)->identity.get() ||
      target.rootOrdinal_ != rootOrdinal_ || target.peOrdinal_ != peOrdinal_ ||
      !target.operation_)
    return invalid("FU template belongs to a foreign PE");
  if (peOrdinal_ >= (*state)->pes.size())
    return invalid("PE handle has an invalid owner ordinal");
  detail::PeState &pe = (*state)->pes[peOrdinal_];
  if (pe.closed)
    return invalid("PE is already closed");

  llvm::SmallVector<mlir::Value, 8> values;
  values.reserve(inputs.size());
  for (const PeValue &input : inputs) {
    auto resolved = resolveValue(*state, input);
    if (!resolved)
      return resolved.takeError();
    values.push_back(*resolved);
  }
  auto instance =
      createInstance((*state)->context, pe.operation.getLoc(),
                     pe.operation.getBody().front(), target.operation_, values);
  if (!instance)
    return instance.takeError();
  if (!pe.named)
    if (llvm::Error error = registerInstanceOwners(
            (*state)->spatialRoots[rootOrdinal_].domainRelation,
            instance->getOperation(), target.operation_)) {
      instance->erase();
      return std::move(error);
    }

  std::vector<PeValue> results;
  results.reserve(instance->getNumResults());
  for (mlir::Value value : instance->getResults())
    results.push_back(PeValue(*state, rootOrdinal_, peOrdinal_, value));
  mlir::Operation *scope = templateScope(**state, rootOrdinal_, pe);
  TemplatePhysicalOwnerHandle occurrence(
      (*state)->identity, rootOrdinal_, scope, {instance->getOperation()},
      target.operation_, DomainMemberRole::Occurrence, 0);
  return PeTemplateInstanceResult(
      std::move(results), std::move(occurrence), *state, (*state)->identity,
      rootOrdinal_, scope, target.operation_, instance->getOperation());
}

llvm::Expected<PeBuilder> SpatialCoreBuilder::createPeTemplate(
    llvm::StringRef label, llvm::ArrayRef<PortType> boundaryInputTypes,
    const PeSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (llvm::Error error = requireAvailableSymbol(root.operation, label))
    return std::move(error);
  llvm::SmallVector<mlir::Type, 8> boundaryInputs;
  for (const PortType &type : boundaryInputTypes)
    boundaryInputs.push_back(materializePortType((*state)->context, type));
  auto materialized = detail::BuilderSpecMaterializer::pe(
      (*state)->context, boundaryInputs, spec, true);
  if (!materialized)
    return materialized.takeError();

  mlir::FunctionType signature = mlir::FunctionType::get(
      &(*state)->context, materialized->boundaryInputTypes,
      materialized->outputTypes);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto operation = ::fabric::PeOp::create(
      builder, root.operation.getLoc(), mlir::TypeRange{},
      mlir::StringAttr::get(&(*state)->context, label),
      mlir::TypeAttr::get(signature), spec.schedule_, mlir::ValueRange{},
      materialized->tagWidth, materialized->instructionCapacity,
      materialized->registerFifoCount, materialized->registerFifoDepth,
      materialized->registerFifoPorts, materialized->fuConfigurationMode,
      materialized->operandBufferMode, materialized->operandBufferSize);
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : materialized->bodyInputTypes)
    body->addArgument(type, operation.getLoc());

  const std::size_t ordinal = (*state)->pes.size();
  (*state)->pes.push_back(
      detail::PeState{operation, rootOrdinal_, true, false});
  return PeBuilder(*state, rootOrdinal_, ordinal, operation.getOperation(),
                   materialized->instructionContexts);
}

llvm::Expected<PeTemplateHandle> PeBuilder::closeTemplate() {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (peOrdinal_ >= (*state)->pes.size() || !(*state)->pes[peOrdinal_].named)
    return invalid("closeTemplate requires a named PE declaration");
  if (llvm::Error error = closeImpl(true))
    return std::move(error);
  return PeTemplateHandle((*state)->identity, rootOrdinal_,
                          (*state)->pes[peOrdinal_].operation,
                          instructionContexts_);
}

llvm::Expected<SwitchTemplateHandle>
SpatialCoreBuilder::createSwitchTemplate(llvm::StringRef label,
                                         const SwitchSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (llvm::Error error = requireAvailableSymbol(root.operation, label))
    return std::move(error);
  auto materialized =
      detail::BuilderSpecMaterializer::switchSpec((*state)->context, spec);
  if (!materialized)
    return materialized.takeError();
  mlir::FunctionType signature = mlir::FunctionType::get(
      &(*state)->context, materialized->inputTypes, materialized->outputTypes);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto sw = ::fabric::SwitchOp::create(
      builder, root.operation.getLoc(), mlir::TypeRange{}, mlir::ValueRange{},
      mlir::StringAttr::get(&(*state)->context, label),
      mlir::TypeAttr::get(signature), spec.schedule,
      llvm::ArrayRef<mlir::Type>{}, materialized->hardwareParameters,
      mlir::DictionaryAttr());
  if (mlir::failed(mlir::verify(sw))) {
    sw.erase();
    return invalid("Fabric rejected the typed Switch template");
  }
  return SwitchTemplateHandle((*state)->identity, rootOrdinal_, sw);
}

llvm::Expected<MemoryTemplateHandle>
SpatialCoreBuilder::createMemoryTemplate(llvm::StringRef label,
                                         const MemorySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (llvm::Error error = requireAvailableSymbol(root.operation, label))
    return std::move(error);

  auto materialized =
      detail::BuilderSpecMaterializer::memory((*state)->context, spec);
  if (!materialized)
    return materialized.takeError();
  mlir::FunctionType signature = mlir::FunctionType::get(
      &(*state)->context, materialized->inputTypes, materialized->outputTypes);

  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto memory = ::fabric::MemOp::create(
      builder, root.operation.getLoc(), mlir::TypeRange{}, mlir::ValueRange{},
      mlir::StringAttr::get(&(*state)->context, label),
      mlir::TypeAttr::get(signature), materialized->contract,
      llvm::ArrayRef<mlir::Type>{}, mlir::ArrayAttr(),
      materialized->operationPorts);
  if (mlir::failed(mlir::verify(memory))) {
    memory.erase();
    return invalid("Fabric rejected the typed memory template");
  }
  return MemoryTemplateHandle((*state)->identity, rootOrdinal_, memory,
                              materialized->operationPortCount,
                              materialized->hasLocalService);
}

llvm::Expected<SpatialTemplateInstanceResult>
SpatialCoreBuilder::instantiate(const PeTemplateHandle &target,
                                llvm::ArrayRef<SpatialValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = target.identity_.lock();
  if (!identity || identity.get() != (*state)->identity.get() ||
      target.rootOrdinal_ != rootOrdinal_ || !target.operation_)
    return invalid("PE template belongs to a foreign SpatialCore");
  return instantiateTemplate(target.operation_, inputs);
}

llvm::Expected<SpatialTemplateInstanceResult>
SpatialCoreBuilder::instantiate(const SwitchTemplateHandle &target,
                                llvm::ArrayRef<SpatialValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = target.identity_.lock();
  if (!identity || identity.get() != (*state)->identity.get() ||
      target.rootOrdinal_ != rootOrdinal_ || !target.operation_)
    return invalid("Switch template belongs to a foreign SpatialCore");
  return instantiateTemplate(target.operation_, inputs);
}

llvm::Expected<SpatialTemplateInstanceResult>
SpatialCoreBuilder::instantiate(const MemoryTemplateHandle &target,
                                llvm::ArrayRef<SpatialValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = target.identity_.lock();
  if (!identity || identity.get() != (*state)->identity.get() ||
      target.rootOrdinal_ != rootOrdinal_ || !target.operation_)
    return invalid("memory template belongs to a foreign SpatialCore");
  return instantiateTemplate(target.operation_, inputs);
}

llvm::Expected<SpatialTemplateInstanceResult>
SpatialCoreBuilder::instantiateTemplate(mlir::Operation *target,
                                        llvm::ArrayRef<SpatialValue> inputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->spatialRoots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->spatialRoots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");

  llvm::SmallVector<mlir::Value, 8> values;
  for (const SpatialValue &input : inputs) {
    auto resolved = resolveValue(*state, input);
    if (!resolved)
      return resolved.takeError();
    if (!mlir::isa<mlir::MemRefType>(resolved->getType()) &&
        !resolved->use_empty())
      return invalid("SpatialCore transport source already has a consumer");
    values.push_back(*resolved);
  }
  auto instance =
      createInstance((*state)->context, root.operation.getLoc(),
                     root.operation.getBody().front(), target, values);
  if (!instance)
    return instance.takeError();
  if (llvm::Error error = registerInstanceOwners(
          root.domainRelation, instance->getOperation(), target)) {
    instance->erase();
    return std::move(error);
  }

  std::vector<SpatialValue> results;
  for (mlir::Value value : instance->getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, value));
  TemplatePhysicalOwnerHandle occurrence(
      (*state)->identity, rootOrdinal_, root.operation,
      {instance->getOperation()}, target, DomainMemberRole::Occurrence, 0);
  return SpatialTemplateInstanceResult(
      std::move(results), std::move(occurrence), *state, (*state)->identity,
      rootOrdinal_, root.operation, target, instance->getOperation());
}

llvm::Expected<ModuleDomainMemberHandle> SpatialCoreBuilder::moduleMember(
    const TemplatePhysicalOwnerHandle &owner) const {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  auto identity = owner.identity_.lock();
  if (!identity || identity.get() != (*state)->identity.get() ||
      owner.rootOrdinal_ != rootOrdinal_ ||
      rootOrdinal_ >= (*state)->spatialRoots.size() ||
      owner.scope_ != (*state)->spatialRoots[rootOrdinal_].operation ||
      !owner.owner_)
    return invalid("physical owner is not relative to this SpatialCore");
  return ModuleDomainMemberHandle::instantiated(
      state_, rootOrdinal_, owner.instancePath_, owner.owner_, owner.role_,
      owner.ordinal_);
}

} // namespace loom::adg
