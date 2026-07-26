#include "ADG/Builder.h"

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
  for (std::int64_t extent : shape)
    if (extent == 0 || extent < mlir::ShapedType::kDynamic)
      return invalid("Fabric memory shape contains an invalid extent");
  return PortType(Kind::Memory, elementType.width(), elementType.tagWidth(),
                  std::vector<std::int64_t>(shape.begin(), shape.end()));
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

} // namespace

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

llvm::Error SpatialCoreBuilder::close(llvm::ArrayRef<SpatialValue> outputs) {
  auto state = activeState(state_);
  if (!state)
    return state.takeError();
  if (rootOrdinal_ >= (*state)->roots.size())
    return invalid("SpatialCore handle has an invalid owner ordinal");
  detail::SpatialRootState &root = (*state)->roots[rootOrdinal_];
  if (root.closed)
    return invalid("SpatialCore is already closed");
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
