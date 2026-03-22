#include "loom/ADG/ADGVerifier.h"

#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Mapper/TypeCompat.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"

namespace loom {

namespace {

using SymbolTargetMap = llvm::StringMap<mlir::Operation *>;
using ScopeDefinitionMap =
    llvm::DenseMap<mlir::Block *, llvm::StringMap<mlir::Operation *>>;

std::optional<llvm::StringRef> getDefinitionName(mlir::Operation &op) {
  if (auto fuOp = mlir::dyn_cast<loom::fabric::FunctionUnitOp>(op))
    return fuOp.getSymName();
  if (auto peOp = mlir::dyn_cast<loom::fabric::SpatialPEOp>(op))
    return peOp.getSymName();
  if (auto peOp = mlir::dyn_cast<loom::fabric::TemporalPEOp>(op))
    return peOp.getSymName();
  if (auto swOp = mlir::dyn_cast<loom::fabric::SpatialSwOp>(op))
    return swOp.getSymName();
  if (auto swOp = mlir::dyn_cast<loom::fabric::TemporalSwOp>(op))
    return swOp.getSymName();
  if (auto extOp = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(op))
    return extOp.getSymName();
  if (auto memOp = mlir::dyn_cast<loom::fabric::MemoryOp>(op))
    return memOp.getSymName();
  if (auto fifoOp = mlir::dyn_cast<loom::fabric::FifoOp>(op))
    return fifoOp.getSymName();
  return std::nullopt;
}

bool isNamedDefinitionOp(mlir::Operation &op) {
  return getDefinitionName(op).has_value();
}

bool isModuleLevelComponentDef(mlir::Operation &op) {
  return mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp,
                   loom::fabric::SpatialSwOp, loom::fabric::TemporalSwOp,
                   loom::fabric::ExtMemoryOp, loom::fabric::MemoryOp,
                   loom::fabric::FifoOp>(op);
}

bool isFunctionUnitDef(mlir::Operation &op) {
  return mlir::isa<loom::fabric::FunctionUnitOp>(op);
}

bool isInlineInstantiationOp(mlir::Operation &op) {
  return op.hasAttr("inline_instantiation");
}

bool isActualDefinitionOp(mlir::Operation &op) {
  if (isFunctionUnitDef(op))
    return true;
  if (!isModuleLevelComponentDef(op))
    return false;
  return !isInlineInstantiationOp(op);
}

bool isDefinitionScopeOwner(mlir::Operation *op) {
  return mlir::isa_and_nonnull<mlir::ModuleOp, loom::fabric::ModuleOp,
                               loom::fabric::SpatialPEOp,
                               loom::fabric::TemporalPEOp>(op);
}

llvm::StringRef getDefinitionKind(mlir::Operation &op) {
  if (mlir::isa<loom::fabric::FunctionUnitOp>(op))
    return "fabric.function_unit";
  if (mlir::isa<loom::fabric::SpatialPEOp>(op))
    return "fabric.spatial_pe";
  if (mlir::isa<loom::fabric::TemporalPEOp>(op))
    return "fabric.temporal_pe";
  if (mlir::isa<loom::fabric::SpatialSwOp>(op))
    return "fabric.spatial_sw";
  if (mlir::isa<loom::fabric::TemporalSwOp>(op))
    return "fabric.temporal_sw";
  if (mlir::isa<loom::fabric::ExtMemoryOp>(op))
    return "fabric.extmemory";
  if (mlir::isa<loom::fabric::MemoryOp>(op))
    return "fabric.memory";
  if (mlir::isa<loom::fabric::FifoOp>(op))
    return "fabric.fifo";
  return op.getName().getStringRef();
}

void collectDefinitionsInBlock(mlir::Block &block, ScopeDefinitionMap &scopes,
                               bool &ok) {
  auto &scopeMap = scopes[&block];
  for (mlir::Operation &op : block.getOperations()) {
    if (!isActualDefinitionOp(op))
      continue;
    auto defName = getDefinitionName(op);
    if (!defName)
      continue;
    auto it = scopeMap.find(*defName);
    if (it != scopeMap.end()) {
      llvm::errs() << "fabric verify: duplicate definition name '" << *defName
                   << "' in the same host scope between "
                   << getDefinitionKind(*it->second) << " and "
                   << getDefinitionKind(op) << "\n";
      ok = false;
      continue;
    }
    scopeMap[*defName] = &op;
  }
}

void collectDefinitionScopes(mlir::ModuleOp topModule, ScopeDefinitionMap &scopes,
                             bool &ok) {
  topModule.walk([&](mlir::Operation *op) {
    if (!isDefinitionScopeOwner(op))
      return;
    if (auto *region = op->getNumRegions() > 0 ? &op->getRegion(0) : nullptr) {
      if (!region->empty())
        collectDefinitionsInBlock(region->front(), scopes, ok);
    }
  });
}

llvm::SmallVector<mlir::Block *, 4>
getDefinitionSearchBlocks(mlir::Operation *anchor) {
  llvm::SmallVector<mlir::Block *, 4> blocks;
  mlir::Block *block = anchor ? anchor->getBlock() : nullptr;
  while (block) {
    blocks.push_back(block);
    mlir::Operation *parent = block->getParentOp();
    if (!parent)
      break;
    block = parent->getBlock();
  }
  return blocks;
}

mlir::Operation *resolveDefinition(mlir::Operation *anchor, llvm::StringRef name,
                                   const ScopeDefinitionMap &scopes) {
  for (mlir::Block *block : getDefinitionSearchBlocks(anchor)) {
    auto it = scopes.find(block);
    if (it == scopes.end())
      continue;
    auto defIt = it->second.find(name);
    if (defIt != it->second.end())
      return defIt->second;
  }
  return nullptr;
}

bool isDefinitionOnlyOp(mlir::Operation &op) {
  if (mlir::isa<loom::fabric::FunctionUnitOp>(op))
    return true;

  if (isModuleLevelComponentDef(op))
    return !isInlineInstantiationOp(op);
  return false;
}

bool isGraphNodeOp(mlir::Operation &op) {
  if (mlir::isa<loom::fabric::InstanceOp, loom::fabric::AddTagOp,
                loom::fabric::DelTagOp, loom::fabric::MapTagOp>(op)) {
    return true;
  }
  if (mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp,
                loom::fabric::SpatialSwOp, loom::fabric::TemporalSwOp,
                loom::fabric::ExtMemoryOp, loom::fabric::MemoryOp,
                loom::fabric::FifoOp>(op))
    return !isDefinitionOnlyOp(op);
  return false;
}

std::optional<mlir::FunctionType> getDeclaredFunctionType(mlir::Operation &op,
                                                          const ScopeDefinitionMap &scopes) {
  if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
    mlir::Operation *target =
        resolveDefinition(instOp.getOperation(), instOp.getModule(), scopes);
    if (!target)
      return std::nullopt;
    if (auto pe = mlir::dyn_cast<loom::fabric::SpatialPEOp>(target))
      return pe.getFunctionType();
    if (auto pe = mlir::dyn_cast<loom::fabric::TemporalPEOp>(target))
      return pe.getFunctionType();
    if (auto sw = mlir::dyn_cast<loom::fabric::SpatialSwOp>(target))
      return sw.getFunctionType();
    if (auto sw = mlir::dyn_cast<loom::fabric::TemporalSwOp>(target))
      return sw.getFunctionType();
    if (auto ext = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(target))
      return ext.getFunctionType();
    if (auto mem = mlir::dyn_cast<loom::fabric::MemoryOp>(target))
      return mem.getFunctionType();
    if (auto fifo = mlir::dyn_cast<loom::fabric::FifoOp>(target))
      return fifo.getFunctionType();
    if (auto fu = mlir::dyn_cast<loom::fabric::FunctionUnitOp>(target))
      return fu.getFunctionType();
    return std::nullopt;
  }
  if (auto extOp = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(op))
    return extOp.getFunctionType();
  if (auto memOp = mlir::dyn_cast<loom::fabric::MemoryOp>(op))
    return memOp.getFunctionType();
  if (auto fifoOp = mlir::dyn_cast<loom::fabric::FifoOp>(op))
    return fifoOp.getFunctionType();
  if (auto peOp = mlir::dyn_cast<loom::fabric::SpatialPEOp>(op))
    return peOp.getFunctionType();
  if (auto peOp = mlir::dyn_cast<loom::fabric::TemporalPEOp>(op))
    return peOp.getFunctionType();
  if (auto swOp = mlir::dyn_cast<loom::fabric::SpatialSwOp>(op))
    return swOp.getFunctionType();
  if (auto swOp = mlir::dyn_cast<loom::fabric::TemporalSwOp>(op))
    return swOp.getFunctionType();
  return std::nullopt;
}

std::string describeOp(mlir::Operation &op) {
  if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
    auto symName = instOp.getSymName();
    if (symName)
      return ("instance '" + symName->str() + "'");
    return "fabric.instance";
  }
  auto printNamed = [&](auto namedOp) -> std::string {
    auto symName = namedOp.getSymName();
    if (symName)
      return (op.getName().getStringRef().str() + " '" + symName->str() + "'");
    return op.getName().getStringRef().str();
  };
  if (auto swOp = mlir::dyn_cast<loom::fabric::SpatialSwOp>(op))
    return printNamed(swOp);
  if (auto swOp = mlir::dyn_cast<loom::fabric::TemporalSwOp>(op))
    return printNamed(swOp);
  if (auto peOp = mlir::dyn_cast<loom::fabric::SpatialPEOp>(op))
    return printNamed(peOp);
  if (auto peOp = mlir::dyn_cast<loom::fabric::TemporalPEOp>(op))
    return printNamed(peOp);
  if (auto extOp = mlir::dyn_cast<loom::fabric::ExtMemoryOp>(op))
    return printNamed(extOp);
  if (auto memOp = mlir::dyn_cast<loom::fabric::MemoryOp>(op))
    return printNamed(memOp);
  if (auto fifoOp = mlir::dyn_cast<loom::fabric::FifoOp>(op))
    return printNamed(fifoOp);
  return op.getName().getStringRef().str();
}

} // namespace

mlir::LogicalResult verifyFabricModule(mlir::ModuleOp topModule) {
  if (failed(mlir::verify(topModule)))
    return mlir::failure();

  bool ok = true;

  // Find fabric.module
  loom::fabric::ModuleOp fabricMod;
  topModule->walk([&](loom::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod)
    return mlir::success(); // No fabric.module, nothing to check

  auto &body = fabricMod.getBody().front();

  ScopeDefinitionMap definitionScopes;
  collectDefinitionScopes(topModule, definitionScopes, ok);

  topModule.walk([&](loom::fabric::InstanceOp instOp) {
    mlir::Operation *parent = instOp->getBlock()->getParentOp();
    mlir::Operation *target =
        resolveDefinition(instOp.getOperation(), instOp.getModule(), definitionScopes);
    if (!target) {
      llvm::errs() << "fabric verify: instance target '" << instOp.getModule()
                   << "' cannot be resolved from the current host scope\n";
      ok = false;
      return;
    }

    if (mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp>(parent)) {
      if (!mlir::isa<loom::fabric::FunctionUnitOp>(target)) {
        llvm::errs() << "fabric verify: instances inside "
                     << parent->getName().getStringRef()
                     << " must target fabric.function_unit, but '"
                     << instOp.getModule() << "' resolves to "
                     << getDefinitionKind(*target) << "\n";
        ok = false;
      }
      if (instOp.getNumOperands() != 0 || instOp.getNumResults() != 0) {
        llvm::errs() << "fabric verify: function_unit instance '"
                     << instOp.getModule()
                     << "' inside a PE must not carry SSA operands or results\n";
        ok = false;
      }
      return;
    }

    if (mlir::isa<loom::fabric::ModuleOp>(parent)) {
      if (!mlir::isa<loom::fabric::SpatialPEOp, loom::fabric::TemporalPEOp,
                     loom::fabric::SpatialSwOp, loom::fabric::TemporalSwOp,
                     loom::fabric::ExtMemoryOp, loom::fabric::MemoryOp,
                     loom::fabric::FifoOp>(target)) {
        llvm::errs() << "fabric verify: instances inside fabric.module must target "
                     << "fabric.{spatial_pe, temporal_pe, spatial_sw, temporal_sw, "
                        "memory, extmemory, fifo}, but '"
                     << instOp.getModule() << "' resolves to "
                     << getDefinitionKind(*target) << "\n";
        ok = false;
      }
      return;
    }
  });

  llvm::DenseSet<mlir::Value> consumed;

  for (auto &op : body.getOperations()) {
    if (!isGraphNodeOp(op) &&
        !mlir::isa<loom::fabric::YieldOp>(op)) {
      continue;
    }

    for (auto operand : op.getOperands())
      consumed.insert(operand);
  }

  for (auto &op : body.getOperations()) {
    if (!isGraphNodeOp(op))
      continue;

    std::string opDesc = describeOp(op);

    if (auto fnType = getDeclaredFunctionType(op, definitionScopes)) {
      if (fnType->getNumInputs() != op.getNumOperands()) {
        llvm::errs() << "fabric verify: dangling input ports on " << opDesc
                     << " — expected " << fnType->getNumInputs()
                     << " connected input(s), got " << op.getNumOperands()
                     << "\n";
        ok = false;
      }
      if (fnType->getNumResults() != op.getNumResults()) {
        llvm::errs() << "fabric verify: output port count mismatch on "
                     << opDesc << " — expected " << fnType->getNumResults()
                     << " output(s), got " << op.getNumResults() << "\n";
        ok = false;
      }

      unsigned operandPairs =
          std::min<unsigned>(fnType->getNumInputs(), op.getNumOperands());
      for (unsigned i = 0; i < operandPairs; ++i) {
        if (!isHardwarePortCompatible(op.getOperand(i).getType(),
                                      fnType->getInput(i))) {
          llvm::errs() << "fabric verify: tag-kind mismatch on " << opDesc
                       << " input I" << i << " — producer type "
                       << op.getOperand(i).getType() << " is incompatible with "
                       << fnType->getInput(i) << "\n";
          ok = false;
        }
      }

      unsigned resultPairs =
          std::min<unsigned>(fnType->getNumResults(), op.getNumResults());
      for (unsigned i = 0; i < resultPairs; ++i) {
        if (!isHardwarePortCompatible(fnType->getResult(i),
                                      op.getResult(i).getType())) {
          llvm::errs() << "fabric verify: tag-kind mismatch on " << opDesc
                       << " output O" << i << " — declared type "
                       << fnType->getResult(i) << " is incompatible with "
                       << op.getResult(i).getType() << "\n";
          ok = false;
        }
      }
    } else if (auto instOp = mlir::dyn_cast<loom::fabric::InstanceOp>(op)) {
      llvm::errs() << "fabric verify: " << opDesc << " references unknown or invalid target '"
                   << instOp.getModule() << "'\n";
      ok = false;
    }

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mlir::Value result = op.getResult(i);
      if (!consumed.count(result)) {
        llvm::errs() << "fabric verify: dangling output port O" << i
                     << " of " << opDesc << " — result is not connected\n";
        ok = false;
      }
    }
  }

  for (auto arg : body.getArguments()) {
    if (!consumed.count(arg)) {
      llvm::errs() << "fabric verify: dangling module input port I"
                   << arg.getArgNumber() << " — not connected to any instance\n";
      ok = false;
    }
  }

  auto yieldOp = mlir::dyn_cast<loom::fabric::YieldOp>(body.getTerminator());
  if (yieldOp) {
    auto fnType = fabricMod.getFunctionType();
    unsigned resultPairs =
        std::min<unsigned>(yieldOp.getNumOperands(), fnType.getNumResults());
    for (unsigned i = 0; i < resultPairs; ++i) {
      if (!isHardwarePortCompatible(yieldOp.getOperand(i).getType(),
                                    fnType.getResult(i))) {
        llvm::errs() << "fabric verify: tag-kind mismatch on module output O"
                     << i << " — source type " << yieldOp.getOperand(i).getType()
                     << " is incompatible with " << fnType.getResult(i)
                     << "\n";
        ok = false;
      }
    }
  }

  return ok ? mlir::success() : mlir::failure();
}

} // namespace loom
