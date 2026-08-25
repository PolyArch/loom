#include "Fabric/IR/Elaboration.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace mlir;

namespace {

template <typename OpTy>
static SmallVector<Type> getEffectiveInnerTypes(OpTy op) {
  ArrayRef<Type> stored = op.getInnerInputTypes();
  if (!stored.empty() && stored.size() == op->getNumOperands())
    return SmallVector<Type>(stored);
  return SmallVector<Type>(op->getOperandTypes());
}

static SmallVector<Type> getDeclaredYieldTypes(fabric::YieldOp yield) {
  auto declared = yield->getAttrOfType<ArrayAttr>("declared_types");
  if (declared && declared.size() == yield->getNumOperands()) {
    SmallVector<Type> types;
    types.reserve(declared.size());
    for (Attribute attr : declared) {
      auto typeAttr = dyn_cast<TypeAttr>(attr);
      if (!typeAttr)
        return SmallVector<Type>(yield->getOperandTypes());
      types.push_back(typeAttr.getValue());
    }
    return types;
  }
  return SmallVector<Type>(yield->getOperandTypes());
}

static SmallVector<Type> getEndpointTypes(Operation *op) {
  if (auto instantiate = dyn_cast<fabric::InstantiateOp>(op))
    return getEffectiveInnerTypes(instantiate);
  if (auto boundary = dyn_cast<fabric::BoundaryOp>(op))
    return getEffectiveInnerTypes(boundary);
  if (auto mem = dyn_cast<fabric::MemOp>(op))
    return getEffectiveInnerTypes(mem);
  if (auto switchOp = dyn_cast<fabric::SwitchOp>(op))
    return getEffectiveInnerTypes(switchOp);
  if (auto yield = dyn_cast<fabric::YieldOp>(op))
    return getDeclaredYieldTypes(yield);
  return {};
}

static Type getOperandEndpointType(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  SmallVector<Type> endpointTypes = getEndpointTypes(owner);
  if (endpointTypes.size() == owner->getNumOperands())
    return endpointTypes[operand.getOperandNumber()];
  if (auto pe = dyn_cast<fabric::PeOp>(owner))
    return pe.getBody()
        .front()
        .getArgument(operand.getOperandNumber())
        .getType();
  if (auto fu = dyn_cast<fabric::FuOp>(owner))
    return fu.getBody()
        .front()
        .getArgument(operand.getOperandNumber())
        .getType();
  if (auto fifo = dyn_cast<fabric::FifoOp>(owner))
    return fifo.getOutput().getType();
  return {};
}

struct SemanticConfig {
  std::optional<unsigned> addrBits;
  std::optional<unsigned> memBusWidth;

  bool operator==(const SemanticConfig &other) const {
    return addrBits == other.addrBits && memBusWidth == other.memBusWidth;
  }
};

static fabric::ModuleOp getEnclosingFabricModule(Operation *op) {
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (auto module = dyn_cast<fabric::ModuleOp>(current))
      return module;
  }
  return {};
}

static SemanticConfig getSemanticConfig(Operation *op) {
  fabric::ModuleOp module = getEnclosingFabricModule(op);
  if (!module)
    return {};

  std::optional<unsigned> addrBits;
  if (auto attr = module.getLoomAddrBitsAttr())
    addrBits = static_cast<unsigned>(attr.getInt());

  std::optional<unsigned> memBusWidth;
  if (auto attr = module.getLoomMemBusWidthAttr())
    memBusWidth = static_cast<unsigned>(attr.getInt());

  return {addrBits, memBusWidth};
}

static std::string configValue(std::optional<unsigned> value) {
  return value ? std::to_string(*value) : "unset";
}

static LogicalResult
checkWidthComponent(fabric::InstantiateOp instantiate, StringRef direction,
                    unsigned index, StringRef component, unsigned sourceWidth,
                    unsigned intermediateWidth, unsigned destinationWidth) {
  if (intermediateWidth >= std::min(sourceWidth, destinationWidth))
    return success();
  return instantiate.emitError()
         << "cannot inline fabric.module @" << instantiate.getCallee()
         << " at fabric.instantiate " << direction << " #" << index
         << ": intermediate " << component << " width " << intermediateWidth
         << " is narrower than source width " << sourceWidth
         << " and destination width " << destinationWidth
         << "; the removed low-bit normalization is not representable by "
            "the existing direct endpoint connection";
}

static LogicalResult
checkNormalizationComposition(fabric::InstantiateOp instantiate,
                              StringRef direction, unsigned index, Type source,
                              Type intermediate, Type destination) {
  if (source == intermediate || intermediate == destination)
    return success();

  if (auto sourceBits = dyn_cast<fabric::BitsType>(source)) {
    auto intermediateBits = dyn_cast<fabric::BitsType>(intermediate);
    auto destinationBits = dyn_cast<fabric::BitsType>(destination);
    if (intermediateBits && destinationBits)
      return checkWidthComponent(
          instantiate, direction, index, "payload", sourceBits.getWidth(),
          intermediateBits.getWidth(), destinationBits.getWidth());
  }

  if (auto sourceTag = dyn_cast<fabric::BitsTagType>(source)) {
    auto intermediateTag = dyn_cast<fabric::BitsTagType>(intermediate);
    auto destinationTag = dyn_cast<fabric::BitsTagType>(destination);
    if (intermediateTag && destinationTag) {
      if (failed(checkWidthComponent(
              instantiate, direction, index, "payload", sourceTag.getWidth(),
              intermediateTag.getWidth(), destinationTag.getWidth())))
        return failure();
      return checkWidthComponent(
          instantiate, direction, index, "tag", sourceTag.getTagWidth(),
          intermediateTag.getTagWidth(), destinationTag.getTagWidth());
    }
  }

  return instantiate.emitError()
         << "cannot inline fabric.module @" << instantiate.getCallee()
         << " at fabric.instantiate " << direction << " #" << index
         << ": removed normalization " << source << " -> " << intermediate
         << " followed by destination endpoint " << destination
         << " cannot be represented by the existing direct endpoint typing";
}

template <typename OpTy>
static void setCanonicalInnerTypes(OpTy op, ArrayRef<Type> innerTypes) {
  bool anyDifference = false;
  for (auto [operand, innerType] : llvm::zip(op->getOperands(), innerTypes)) {
    if (operand.getType() != innerType) {
      anyDifference = true;
      break;
    }
  }
  op.setInnerInputTypes(anyDifference ? innerTypes : ArrayRef<Type>{});
}

static void setEndpointTypes(Operation *op, ArrayRef<Type> endpointTypes) {
  if (endpointTypes.size() != op->getNumOperands())
    return;
  if (auto instantiate = dyn_cast<fabric::InstantiateOp>(op)) {
    setCanonicalInnerTypes(instantiate, endpointTypes);
    return;
  }
  if (auto boundary = dyn_cast<fabric::BoundaryOp>(op)) {
    setCanonicalInnerTypes(boundary, endpointTypes);
    return;
  }
  if (auto mem = dyn_cast<fabric::MemOp>(op)) {
    setCanonicalInnerTypes(mem, endpointTypes);
    return;
  }
  if (auto switchOp = dyn_cast<fabric::SwitchOp>(op)) {
    setCanonicalInnerTypes(switchOp, endpointTypes);
    return;
  }
  if (auto yield = dyn_cast<fabric::YieldOp>(op)) {
    bool anyDifference = false;
    SmallVector<Attribute> attrs;
    attrs.reserve(endpointTypes.size());
    for (auto [operand, endpointType] :
         llvm::zip(yield->getOperands(), endpointTypes)) {
      anyDifference |= operand.getType() != endpointType;
      attrs.push_back(TypeAttr::get(endpointType));
    }
    if (anyDifference)
      yield->setAttr("declared_types",
                     ArrayAttr::get(yield.getContext(), attrs));
    else
      yield->removeAttr("declared_types");
  }
}

static void replaceUsePreservingEndpoint(OpOperand &use, Value newValue) {
  Operation *owner = use.getOwner();
  SmallVector<Type> endpointTypes = getEndpointTypes(owner);
  use.set(newValue);
  if (!endpointTypes.empty())
    setEndpointTypes(owner, endpointTypes);
}

static bool isNamedPhysicalDeclaration(Operation *op) {
  if (isa<fabric::ModuleOp>(op))
    return true;
  if (auto pe = dyn_cast<fabric::PeOp>(op))
    return static_cast<bool>(pe.getSymNameAttr());
  if (auto switchOp = dyn_cast<fabric::SwitchOp>(op))
    return static_cast<bool>(switchOp.getSymNameAttr());
  if (auto mem = dyn_cast<fabric::MemOp>(op))
    return static_cast<bool>(mem.getSymNameAttr());
  if (auto fu = dyn_cast<fabric::FuOp>(op))
    return static_cast<bool>(fu.getSymNameAttr());
  return false;
}

struct WorkItem {
  fabric::InstantiateOp instantiate;
  Operation *target;
};

struct ModuleBoundary {
  fabric::InstantiateOp instantiate;
  fabric::ModuleOp target;
  SmallVector<Value> inputs;
  SmallVector<SmallVector<Type>> inputDestinations;
  SmallVector<Value> replacements;
  SmallVector<SmallVector<Type>> outputDestinations;
};

static Operation *getDirectChild(Operation *op, Operation *ancestor) {
  while (op && op->getParentOp() != ancestor)
    op = op->getParentOp();
  return op;
}

class InstanceElaborator {
public:
  explicit InstanceElaborator(
      fabric::ModuleDomainAuthoringRelation *domainRelation = nullptr)
      : domainRelation(domainRelation) {}

  LogicalResult rewrite(fabric::ModuleOp root) {
    if (failed(elaborateModule(root)) || failed(verify(root)))
      return failure();

    WalkResult remaining = root.walk([&](fabric::InstantiateOp instantiate) {
      instantiate.emitError(
          "internal: Fabric elaboration left a concrete fabric.instantiate");
      return WalkResult::interrupt();
    });
    return failure(remaining.wasInterrupted());
  }

private:
  LogicalResult elaborateModule(fabric::ModuleOp module) {
    for (Operation &op : llvm::make_early_inc_range(module.getBody().front())) {
      if (auto nested = dyn_cast<fabric::ModuleOp>(op)) {
        if (failed(elaborateModule(nested)))
          return failure();
      } else if (auto pe = dyn_cast<fabric::PeOp>(op)) {
        if (failed(elaboratePe(pe)))
          return failure();
      }
    }
    return elaborateBlock(module.getBody().front());
  }

  LogicalResult elaboratePe(fabric::PeOp pe) {
    return elaborateBlock(pe.getBody().front());
  }

  LogicalResult elaborateBlock(Block &block) {
    resetPlan();
    for (Operation &op : block)
      if (auto instantiate = dyn_cast<fabric::InstantiateOp>(op))
        if (failed(enqueue(instantiate)))
          return failure();

    for (size_t index = 0; index < worklist.size(); ++index)
      if (failed(expand(worklist[index])))
        return failure();

    if (failed(validatePlan()))
      return failure();
    return publishPlan();
  }

  void resetPlan() {
    worklist.clear();
    queued.clear();
    replacements.clear();
    replacementOrder.clear();
    replacementOwners.clear();
    boundaries.clear();
    finalReplacements.clear();
  }

  LogicalResult enqueue(fabric::InstantiateOp instantiate,
                        Operation *knownTarget = nullptr) {
    if (!queued.insert(instantiate.getOperation()).second)
      return success();
    Operation *target = knownTarget;
    if (!target)
      target = fabric::resolveInstantiateTarget(instantiate, symbolTables);
    if (!target)
      return instantiate.emitOpError("references undefined symbol '@")
             << instantiate.getCallee() << "'";
    worklist.push_back({instantiate, target});
    return success();
  }

  LogicalResult expand(WorkItem item) {
    if (failed(checkSemanticConfig(item.instantiate, item.target)))
      return failure();
    if (auto module = dyn_cast<fabric::ModuleOp>(item.target))
      return expandModule(item.instantiate, module);
    if (!isa<fabric::PeOp, fabric::SwitchOp, fabric::MemOp, fabric::FuOp>(
            item.target))
      return item.instantiate.emitOpError(
          "resolved target is not a materializable Fabric declaration");
    return expandPhysical(item.instantiate, item.target);
  }

  LogicalResult checkSemanticConfig(fabric::InstantiateOp instantiate,
                                    Operation *target) {
    SemanticConfig definition = getSemanticConfig(target);
    SemanticConfig destination = getSemanticConfig(instantiate);
    if (definition == destination)
      return success();

    if (isa<fabric::ModuleOp>(target))
      return instantiate.emitError()
             << "cannot inline fabric.module @" << instantiate.getCallee()
             << " because module-scoped semantic configuration differs: "
                "loom_addr_bits callee="
             << configValue(definition.addrBits)
             << " caller=" << configValue(destination.addrBits)
             << ", loom_mem_bus_width callee="
             << configValue(definition.memBusWidth)
             << " caller=" << configValue(destination.memBusWidth);

    return instantiate.emitError()
           << "cannot materialize " << target->getName() << " @"
           << instantiate.getCallee()
           << " because module-scoped semantic configuration differs: "
              "loom_addr_bits definition="
           << configValue(definition.addrBits)
           << " destination=" << configValue(destination.addrBits)
           << ", loom_mem_bus_width definition="
           << configValue(definition.memBusWidth)
           << " destination=" << configValue(destination.memBusWidth);
  }

  LogicalResult collectMappedInstances(Operation *source, IRMapping &mapping,
                                       SmallVectorImpl<WorkItem> &mapped) {
    WalkResult result = source->walk([&](fabric::InstantiateOp instantiate) {
      auto cloned = dyn_cast_or_null<fabric::InstantiateOp>(
          mapping.lookupOrNull(instantiate.getOperation()));
      if (!cloned) {
        instantiate.emitError(
            "internal: cloned instance is missing from the graph mapping");
        return WalkResult::interrupt();
      }
      Operation *target =
          fabric::resolveInstantiateTarget(instantiate, symbolTables);
      if (!target) {
        instantiate.emitOpError("references undefined symbol '@")
            << instantiate.getCallee() << "'";
        return WalkResult::interrupt();
      }
      mapped.push_back({cloned, target});
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult expandModule(fabric::InstantiateOp instantiate,
                             fabric::ModuleOp target) {
    ModuleBoundary boundary;
    boundary.instantiate = instantiate;
    boundary.target = target;
    boundary.inputs.assign(instantiate.getInputs().begin(),
                           instantiate.getInputs().end());
    boundary.inputDestinations.resize(instantiate.getNumOperands());
    boundary.outputDestinations.resize(instantiate.getNumResults());

    Block &sourceBody = target.getBody().front();
    for (auto [index, argument] : llvm::enumerate(sourceBody.getArguments())) {
      for (OpOperand &use : argument.getUses()) {
        Operation *top = getDirectChild(use.getOwner(), target);
        if (top && isNamedPhysicalDeclaration(top))
          continue;
        Type endpoint = getOperandEndpointType(use);
        if (!endpoint)
          return instantiate.emitError()
                 << "cannot inline fabric.module @" << instantiate.getCallee()
                 << " at fabric.instantiate input #" << index
                 << ": adjacent destination endpoint type is unavailable for "
                 << use.getOwner()->getName();
        boundary.inputDestinations[index].push_back(endpoint);
      }
    }
    for (auto [index, result] : llvm::enumerate(instantiate.getResults())) {
      for (OpOperand &use : result.getUses()) {
        Type endpoint = getOperandEndpointType(use);
        if (!endpoint)
          return instantiate.emitError()
                 << "cannot inline fabric.module @" << instantiate.getCallee()
                 << " at fabric.instantiate output #" << index
                 << ": adjacent destination endpoint type is unavailable for "
                 << use.getOwner()->getName();
        boundary.outputDestinations[index].push_back(endpoint);
      }
    }

    Region clonedRegion;
    IRMapping mapping;
    mapping.map(sourceBody.getArguments(), instantiate.getInputs());
    target.getBody().cloneInto(&clonedRegion, mapping);
    if (!llvm::hasSingleElement(clonedRegion))
      return target.emitOpError("must contain exactly one body block");
    if (domainRelation)
      if (llvm::Error error = domainRelation->composeInstance(
              instantiate.getOperation(), mapping)) {
        instantiate.emitError(llvm::toString(std::move(error)));
        return failure();
      }

    auto sourceYield = dyn_cast<fabric::YieldOp>(sourceBody.getTerminator());
    if (!sourceYield)
      return target.emitOpError("has no fabric.yield terminator");
    boundary.replacements.reserve(sourceYield.getNumOperands());
    for (Value value : sourceYield.getValues()) {
      Value replacement = mapping.lookupOrNull(value);
      if (!replacement)
        return sourceYield.emitOpError(
            "yield value is missing from the graph clone mapping");
      boundary.replacements.push_back(replacement);
    }

    SmallVector<WorkItem> mappedInstances;
    for (Operation &source : sourceBody) {
      if (isa<fabric::YieldOp>(source) || isNamedPhysicalDeclaration(&source))
        continue;
      if (failed(collectMappedInstances(&source, mapping, mappedInstances)))
        return failure();
    }

    auto clonedYield =
        cast<fabric::YieldOp>(mapping.lookup(sourceYield.getOperation()));
    clonedYield.erase();
    for (Operation &source : sourceBody) {
      if (!isNamedPhysicalDeclaration(&source))
        continue;
      Operation *clone = mapping.lookup(&source);
      clone->erase();
    }

    Block &clonedBody = clonedRegion.front();
    while (!clonedBody.empty())
      clonedBody.front().moveBefore(instantiate);

    recordReplacements(instantiate, boundary.replacements);
    boundaries.push_back(std::move(boundary));
    for (const WorkItem &nested : mappedInstances)
      if (failed(enqueue(nested.instantiate, nested.target)))
        return failure();
    return success();
  }

  LogicalResult expandPhysical(fabric::InstantiateOp instantiate,
                               Operation *target) {
    OpBuilder builder(instantiate);
    OperationState state(instantiate.getLoc(), target->getName());
    state.addOperands(instantiate.getInputs());
    state.addTypes(instantiate.getResultTypes());

    NamedAttrList attrs(target->getAttrs());
    attrs.erase(SymbolTable::getSymbolAttrName());
    attrs.erase(SymbolTable::getVisibilityAttrName());
    attrs.erase("function_type");
    state.addAttributes(attrs);

    IRMapping bodyMapping;
    for (Region &region : target->getRegions()) {
      Region *cloned = state.addRegion();
      region.cloneInto(cloned, bodyMapping);
    }

    Operation *occurrence = builder.create(state);
    bodyMapping.map(instantiate.getOperation(), occurrence);
    if (domainRelation)
      if (llvm::Error error = domainRelation->materializePhysicalInstance(
              instantiate.getOperation(), target, occurrence, bodyMapping)) {
        instantiate.emitError(llvm::toString(std::move(error)));
        occurrence->erase();
        return failure();
      }
    if (isa<fabric::SwitchOp, fabric::MemOp>(occurrence))
      setEndpointTypes(occurrence, getEffectiveInnerTypes(instantiate));
    if (auto pe = dyn_cast<fabric::PeOp>(occurrence)) {
      Block &body = pe.getBody().front();
      if (auto yield = dyn_cast<fabric::YieldOp>(body.getTerminator()))
        yield.erase();
    }

    SmallVector<WorkItem> mappedInstances;
    if (failed(collectMappedInstances(target, bodyMapping, mappedInstances)))
      return failure();
    recordReplacements(instantiate, occurrence->getResults());
    for (const WorkItem &nested : mappedInstances)
      if (failed(enqueue(nested.instantiate, nested.target)))
        return failure();
    return success();
  }

  void recordReplacements(fabric::InstantiateOp instantiate,
                          ValueRange values) {
    for (auto [result, replacement] :
         llvm::zip(instantiate.getResults(), values)) {
      replacements[result] = replacement;
      replacementOrder.push_back(result);
      replacementOwners[result] = instantiate.getOperation();
    }
  }

  FailureOr<Value> resolveFinal(Value value) {
    if (Value cached = finalReplacements.lookup(value))
      return cached;

    auto advance = [&](Value current) -> std::pair<Value, bool> {
      if (Value cached = finalReplacements.lookup(current))
        return {cached, true};
      auto replacement = replacements.find(current);
      if (replacement == replacements.end())
        return {current, true};
      return {replacement->second, false};
    };

    Value slow = value;
    Value fast = value;
    Value terminal;
    while (true) {
      auto [nextSlow, slowIsTerminal] = advance(slow);
      if (slowIsTerminal) {
        terminal = nextSlow;
        break;
      }
      slow = nextSlow;

      auto [nextFast, fastIsTerminal] = advance(fast);
      if (fastIsTerminal) {
        terminal = nextFast;
        break;
      }
      fast = nextFast;
      std::tie(nextFast, fastIsTerminal) = advance(fast);
      if (fastIsTerminal) {
        terminal = nextFast;
        break;
      }
      fast = nextFast;

      if (slow == fast) {
        auto instantiate =
            cast<fabric::InstantiateOp>(replacementOwners.lookup(slow));
        return instantiate.emitError(
            "cannot eliminate fabric.module instance feedback cycle with no "
            "physical producer");
      }
    }

    for (Value current = value;;) {
      if (finalReplacements.count(current))
        break;
      auto replacement = replacements.find(current);
      if (replacement == replacements.end())
        break;
      Value next = replacement->second;
      finalReplacements[current] = terminal;
      current = next;
    }
    return terminal;
  }

  LogicalResult validatePlan() {
    for (Value result : replacementOrder)
      if (failed(resolveFinal(result)))
        return failure();

    for (ModuleBoundary &boundary : boundaries) {
      Block &targetBody = boundary.target.getBody().front();
      for (auto [index, input] : llvm::enumerate(boundary.inputs)) {
        FailureOr<Value> source = resolveFinal(input);
        if (failed(source))
          return failure();
        Type intermediate = targetBody.getArgument(index).getType();
        for (Type destination : boundary.inputDestinations[index])
          if (failed(checkNormalizationComposition(
                  boundary.instantiate, "input", index, source->getType(),
                  intermediate, destination)))
            return failure();
      }

      for (auto [index, replacement] : llvm::enumerate(boundary.replacements)) {
        FailureOr<Value> source = resolveFinal(replacement);
        if (failed(source))
          return failure();
        Type intermediate = boundary.instantiate.getResult(index).getType();
        for (Type destination : boundary.outputDestinations[index])
          if (failed(checkNormalizationComposition(
                  boundary.instantiate, "output", index, source->getType(),
                  intermediate, destination)))
            return failure();
      }
    }
    return success();
  }

  LogicalResult publishPlan() {
    for (Value result : replacementOrder) {
      FailureOr<Value> replacement = resolveFinal(result);
      if (failed(replacement))
        return failure();
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : result.getUses())
        uses.push_back(&use);
      for (OpOperand *use : uses) {
        if (queued.contains(use->getOwner()))
          continue;
        replaceUsePreservingEndpoint(*use, *replacement);
      }
    }

    for (WorkItem &item : worklist)
      item.instantiate->dropAllReferences();
    for (WorkItem &item : llvm::reverse(worklist)) {
      for (Value result : item.instantiate.getResults()) {
        if (!result.use_empty())
          return item.instantiate.emitError(
              "internal: global Fabric instance replacement left a live use");
      }
      item.instantiate.erase();
    }
    return success();
  }

  SymbolTableCollection symbolTables;
  SmallVector<WorkItem> worklist;
  SmallPtrSet<Operation *, 16> queued;
  DenseMap<Value, Value> replacements;
  SmallVector<Value> replacementOrder;
  DenseMap<Value, Operation *> replacementOwners;
  SmallVector<ModuleBoundary, 0> boundaries;
  DenseMap<Value, Value> finalReplacements;
  fabric::ModuleDomainAuthoringRelation *domainRelation = nullptr;
};

static OwningOpRef<ModuleOp> cloneBuiltinModule(ModuleOp module) {
  return OwningOpRef<ModuleOp>(cast<ModuleOp>(module->clone()));
}

static OwningOpRef<ModuleOp> cloneBuiltinModule(ModuleOp module,
                                                IRMapping &mapping) {
  return OwningOpRef<ModuleOp>(cast<ModuleOp>(module->clone(mapping)));
}

struct ElaborateInstancesPass
    : public PassWrapper<ElaborateInstancesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ElaborateInstancesPass)

  StringRef getArgument() const final {
    return "loom-elaborate-fabric-instances";
  }
  StringRef getDescription() const final {
    return "Canonicalize concrete Fabric instances under top-level "
           "fabric.module roots";
  }

  void runOnOperation() final {
    ModuleOp builtinModule = getOperation();
    for (Operation &op : builtinModule.getBody()->getOperations()) {
      auto instantiate = dyn_cast<fabric::InstantiateOp>(op);
      if (!instantiate)
        continue;
      instantiate.emitError(
          "root-local Fabric instance elaboration does not support "
          "fabric.instantiate @")
          << instantiate.getCallee()
          << " directly under builtin.module: no fabric.module occurrence "
             "owner exists";
      signalPassFailure();
      return;
    }

    if (failed(verify(builtinModule))) {
      signalPassFailure();
      return;
    }

    OwningOpRef<ModuleOp> scratch = cloneBuiltinModule(builtinModule);

    for (fabric::ModuleOp root : scratch->getOps<fabric::ModuleOp>()) {
      InstanceElaborator elaborator;
      if (failed(elaborator.rewrite(root))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(verify(*scratch))) {
      signalPassFailure();
      return;
    }

    builtinModule.getBodyRegion().takeBody(scratch->getBodyRegion());
  }
};

} // namespace

namespace fabric {

static LogicalResult
elaborateInstancesImpl(ModuleOp root,
                       ModuleDomainAuthoringRelation *domainRelation) {
  auto builtinModule = dyn_cast_or_null<::mlir::ModuleOp>(root->getParentOp());
  if (!builtinModule)
    return root.emitError(
        "fabric::elaborateInstances supports only top-level fabric.module "
        "roots directly under builtin.module");
  if (failed(verify(builtinModule)))
    return failure();

  IRMapping mapping;
  OwningOpRef<::mlir::ModuleOp> scratch =
      cloneBuiltinModule(builtinModule, mapping);
  std::optional<ModuleDomainAuthoringRelation> remappedDomain;
  if (domainRelation) {
    auto remapped = domainRelation->remap(mapping);
    if (!remapped) {
      root.emitError(llvm::toString(remapped.takeError()));
      return failure();
    }
    remappedDomain = std::move(*remapped);
  }
  Operation *clonedSymbol =
      SymbolTable::lookupSymbolIn(*scratch, root.getSymNameAttr());
  auto clonedRoot = dyn_cast_or_null<fabric::ModuleOp>(clonedSymbol);
  if (!clonedRoot || clonedRoot->getParentOp() != scratch->getOperation())
    return root.emitError(
        "failed to locate the selected top-level fabric.module root in the "
        "transactional scratch module");

  if (failed(InstanceElaborator(remappedDomain ? &*remappedDomain : nullptr)
                 .rewrite(clonedRoot)) ||
      failed(verify(*scratch)))
    return failure();

  root.getBody().takeBody(clonedRoot.getBody());
  if (domainRelation)
    *domainRelation = std::move(*remappedDomain);
  return success();
}

LogicalResult elaborateInstances(ModuleOp root) {
  return elaborateInstancesImpl(root, nullptr);
}

LogicalResult
elaborateInstances(ModuleOp root,
                   ModuleDomainAuthoringRelation &domainRelation) {
  return elaborateInstancesImpl(root, &domainRelation);
}

std::unique_ptr<Pass> createElaborateInstancesPass() {
  return std::make_unique<ElaborateInstancesPass>();
}

void registerFabricIRPasses() { PassRegistration<ElaborateInstancesPass>(); }

} // namespace fabric
