#include "ADG/Builder.h"

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace loom::adg {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_builder_invalid: " + message);
}

bool sameFabricKind(mlir::Type left, mlir::Type right) {
  return (mlir::isa<::fabric::BitsType>(left) &&
          mlir::isa<::fabric::BitsType>(right)) ||
         (mlir::isa<::fabric::BitsTagType>(left) &&
          mlir::isa<::fabric::BitsTagType>(right)) ||
         (mlir::isa<mlir::MemRefType>(left) &&
          mlir::isa<mlir::MemRefType>(right));
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

} // namespace

namespace detail {

struct SpatialRootState final {
  ::fabric::ModuleOp operation;
  std::string label;
  std::vector<mlir::Type> resultTypes;
  bool closed = false;
};

struct PeState final {
  ::fabric::PeOp operation;
  std::size_t rootOrdinal;
  bool closed = false;
};

struct FuState final {
  ::fabric::FuOp operation;
  std::size_t rootOrdinal;
  std::size_t peOrdinal;
  bool closed = false;
};

class DesignState final {
public:
  explicit DesignState(const loom::ArtifactStore &store) : store(store) {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    draft = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> draft;
  const loom::ArtifactStore &store;
  std::vector<SpatialRootState> roots;
  std::vector<PeState> pes;
  std::vector<FuState> fus;
  llvm::StringSet<> labels;
  bool consumed = false;
};

} // namespace detail

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
  return {::fabric::Schedule::Spatial, std::move(inputTypes),
          std::move(outputTypes), std::move(sourcesByOutput), std::nullopt};
}

SwitchSpec
SwitchSpec::temporal(std::vector<PortType> inputTypes,
                     std::vector<PortType> outputTypes,
                     std::vector<std::vector<std::uint32_t>> sourcesByOutput,
                     std::uint32_t routeTableSize) {
  return {::fabric::Schedule::Temporal, std::move(inputTypes),
          std::move(outputTypes), std::move(sourcesByOutput), routeTableSize};
}

MemorySpec MemorySpec::spatial(
    std::vector<PortType> inputTypes, std::vector<PortType> outputTypes,
    std::vector<std::uint32_t> managerInputOrdinals,
    std::vector<std::uint32_t> subordinateOutputOrdinals,
    std::vector<::fabric::MemoryOperationPortDeclaration> operationPorts) {
  return MemorySpec(std::move(inputTypes), std::move(outputTypes),
                    std::move(managerInputOrdinals),
                    std::move(subordinateOutputOrdinals),
                    std::move(operationPorts));
}

namespace {

llvm::Expected<std::shared_ptr<detail::DesignState>>
activeState(const std::weak_ptr<detail::DesignState> &weak) {
  std::shared_ptr<detail::DesignState> state = weak.lock();
  if (!state || state->consumed)
    return invalid("ADG Builder view is stale");
  return state;
}

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

llvm::Expected<std::vector<FuValue>>
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
  if (llvm::Error error = verifyNewOperation(operation, "operation capability"))
    return std::move(error);

  std::vector<FuValue> results;
  results.reserve(operation.getNumResults());
  for (mlir::Value result : operation.getResults())
    results.push_back(
        FuValue(*state, rootOrdinal_, peOrdinal_, fuOrdinal_, result));
  return results;
}

llvm::Expected<FuValue> FuBuilder::addMux(llvm::ArrayRef<FuValue> inputs) {
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
  return FuValue(*state, rootOrdinal_, peOrdinal_, fuOrdinal_, mux.getOutput());
}

llvm::Expected<std::vector<FuValue>>
FuBuilder::addDemux(FuValue input, std::uint32_t outputCount) {
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

  std::vector<FuValue> results;
  results.reserve(demux.getNumResults());
  for (mlir::Value result : demux.getResults())
    results.push_back(
        FuValue(*state, rootOrdinal_, peOrdinal_, fuOrdinal_, result));
  return results;
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
  if (outputs.size() != (*fu)->operation.getNumResults())
    return invalid("FU output count does not match its declaration");

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
  auto operation =
      ::fabric::FuOp::create(builder, (*pe)->operation.getLoc(), outputTypes,
                             mlir::StringAttr(), mlir::TypeAttr(), values);
  mlir::Block *body = new mlir::Block();
  operation.getBody().push_back(body);
  for (mlir::Type type : innerInputTypes)
    body->addArgument(type, operation.getLoc());

  const std::size_t ordinal = (*state)->fus.size();
  (*state)->fus.push_back(
      detail::FuState{operation, rootOrdinal_, peOrdinal_, false});
  return FuBuilder(*state, rootOrdinal_, peOrdinal_, ordinal);
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
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  mlir::Block &body = root.operation.getBody().front();
  if (ordinal >= body.getNumArguments())
    return invalid("SpatialCore input ordinal is out of range");
  return SpatialValue(*state, rootOrdinal_, body.getArgument(ordinal));
}

llvm::Expected<SpatialValue> SpatialCoreBuilder::addFifo(SpatialValue input,
                                                         const FifoSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
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
  return SpatialValue(*state, rootOrdinal_, fifo.getOutput());
}

llvm::Expected<std::vector<SpatialValue>>
SpatialCoreBuilder::addBoundary(llvm::ArrayRef<SpatialValue> inputs,
                                const BoundarySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
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

  std::vector<SpatialValue> results;
  results.reserve(boundary.getNumResults());
  for (mlir::Value result : boundary.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return results;
}

llvm::Expected<std::vector<SpatialValue>>
SpatialCoreBuilder::addSwitch(llvm::ArrayRef<SpatialValue> inputs,
                              const SwitchSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
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
  if (spec.schedule == ::fabric::Schedule::Temporal &&
      (!spec.routeTableSize || *spec.routeTableSize == 0))
    return invalid("Temporal switch requires a positive route-table capacity");

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

  std::vector<SpatialValue> results;
  results.reserve(sw.getNumResults());
  for (mlir::Value result : sw.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return results;
}

llvm::Expected<std::vector<SpatialValue>>
SpatialCoreBuilder::addMemory(llvm::ArrayRef<SpatialValue> inputs,
                              const MemorySpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
  if (inputs.size() != spec.inputTypes_.size())
    return invalid("memory input count does not match its typed contract");
  if (spec.operationPorts_.empty())
    return invalid("memory Operation Engine requires an operation port");

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;
  bool hasNormalizedInput = false;
  for (auto [value, type] : llvm::zip(inputs, spec.inputTypes_)) {
    auto resolved = resolveValue(*state, value);
    if (!resolved)
      return resolved.takeError();
    if (!resolved->use_empty())
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

  llvm::SmallVector<mlir::Attribute, 4> encodedPorts;
  encodedPorts.reserve(spec.operationPorts_.size());
  for (const ::fabric::MemoryOperationPortDeclaration &declaration :
       spec.operationPorts_) {
    auto record = ::fabric::MemoryOperationPortRecord::create(
        &(*state)->context, ::fabric::Schedule::Spatial, *endpoints,
        declaration);
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

  auto contract = ::fabric::MemoryContractAttr::get(
      &(*state)->context,
      ::fabric::MemoryEngineAttr::get(&(*state)->context,
                                      ::fabric::Schedule::Spatial),
      ::fabric::LocalMemoryServiceAttr(), *managers, *subordinates);
  mlir::OpBuilder builder(&(*state)->context);
  builder.setInsertionPointToEnd(&root.operation.getBody().front());
  auto memory = ::fabric::MemOp::create(
      builder, root.operation.getLoc(), outputTypes, values, mlir::StringAttr(),
      mlir::TypeAttr(), contract,
      hasNormalizedInput ? llvm::ArrayRef<mlir::Type>(inputTypes)
                         : llvm::ArrayRef<mlir::Type>(),
      mlir::ArrayAttr(),
      mlir::ArrayAttr::get(&(*state)->context, encodedPorts));
  if (llvm::Error error = verifyNewOperation(memory, "memory"))
    return std::move(error);

  std::vector<SpatialValue> results;
  results.reserve(memory.getNumResults());
  for (mlir::Value result : memory.getResults())
    results.push_back(SpatialValue(*state, rootOrdinal_, result));
  return results;
}

llvm::Expected<PeBuilder>
SpatialCoreBuilder::addPe(llvm::ArrayRef<SpatialValue> inputs,
                          const PeSpec &spec) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
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

  const std::size_t ordinal = (*state)->pes.size();
  (*state)->pes.push_back(detail::PeState{operation, rootOrdinal_, false});
  return PeBuilder(*state, rootOrdinal_, ordinal);
}

llvm::Error SpatialCoreBuilder::close(llvm::ArrayRef<SpatialValue> outputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
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
      mlir::IntegerAttr());
  mlir::Block *body = new mlir::Block();
  root.getBody().push_back(body);
  for (mlir::Type type : inputTypes)
    body->addArgument(type, root.getLoc());

  const std::size_t ordinal = state_->roots.size();
  state_->roots.push_back(detail::SpatialRootState{
      root, label.str(),
      std::vector<mlir::Type>(resultTypes.begin(), resultTypes.end()), false});
  return SpatialCoreBuilder(state_, ordinal);
}

llvm::Expected<FinalizedFabricDesign> DesignBuilder::finalize() && {
  if (!state_ || state_->consumed)
    return invalid("DesignBuilder is already consumed");
  for (const detail::SpatialRootState &root : state_->roots)
    if (!root.closed)
      return invalid("SpatialCore '" + root.label + "' is not closed");

  state_->consumed = true;
  std::vector<loom::fabric::FinalizedFabricRoot> finalized;
  finalized.reserve(state_->roots.size());
  for (const detail::SpatialRootState &root : state_->roots) {
    auto result =
        loom::fabric::finalizeFabricRoot(root.operation, state_->store);
    if (!result)
      return result.takeError();
    finalized.push_back(std::move(*result));
  }
  return FinalizedFabricDesign(std::move(finalized));
}

} // namespace loom::adg
