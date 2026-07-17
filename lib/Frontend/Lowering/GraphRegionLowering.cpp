#include "GraphRegionLowering.h"
#include "GraphIndexLowering.h"

#include "Common/IndexWidth.h"
#include "Frontend/Lowering/StreamLoopAttrs.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <optional>

namespace {

struct MemoryFrontier {
  ::mlir::Value write;
  ::mlir::Value read;
};

using MemoryState = ::llvm::SmallVector<MemoryFrontier, 4>;

struct RegionResult {
  ::mlir::Value execution;
  MemoryState memory;
};

bool isCompilerOwnedControlUse(::mlir::OpOperand &use) {
  ::mlir::Operation *owner = use.getOwner();
  if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(owner))
    return &use == &load.getCtrlMutable();
  if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(owner))
    return &use == &store.getCtrlMutable();
  if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(owner))
    return &use == &constant.getCtrlMutable();
  return false;
}

bool isUseInside(::mlir::OpOperand &use, ::mlir::Region &region) {
  return region.isAncestor(use.getOwner()->getParentRegion());
}

void replaceUsesInside(::mlir::Value from, ::mlir::Value to,
                       ::mlir::Region &region) {
  from.replaceUsesWithIf(
      to, [&](::mlir::OpOperand &use) { return isUseInside(use, region); });
}

bool hasSelectedParallelProvenance(::mlir::Operation *op) {
  return op->hasAttr("loom.parallel_group") ||
         op->hasAttr("loom.parallel_schedule") || op->hasAttr("mapping");
}

bool isGraphMemoryCapabilityType(::mlir::Type type) {
  return ::llvm::isa<::mlir::MemRefType, ::mlir::UnrankedMemRefType,
                     ::mlir::LLVM::LLVMPointerType>(type);
}

bool isKnownLeafComputation(::mlir::Operation *op) {
  ::llvm::StringRef dialect = op->getName().getDialectNamespace();
  if (dialect == "arith" || dialect == "math" || dialect == "ub")
    return true;
  if (dialect != "llvm")
    return false;
  return !::llvm::isa<::mlir::LLVM::CallOp, ::mlir::LLVM::StoreOp,
                      ::mlir::LLVM::MemcpyOp>(op);
}

::mlir::LogicalResult checkOneGraph(::dataflow::GraphFuncOp graph) {
  ::mlir::Block &entry = graph.getBody().front();
  if (entry.getNumArguments() == 0 ||
      !::llvm::isa<::mlir::NoneType>(entry.getArgument(0).getType()))
    return graph.emitError(
        "loom-lower-graph-memory: graph entry must start with none");

  ::mlir::Operation *parallel = nullptr;
  graph.getBody().walk([&](::mlir::Operation *op) {
    if (!::llvm::isa<::mlir::scf::ParallelOp, ::mlir::scf::ForallOp>(op))
      return ::mlir::WalkResult::advance();
    parallel = op;
    return ::mlir::WalkResult::interrupt();
  });
  if (parallel) {
    if (hasSelectedParallelProvenance(parallel))
      parallel->emitError("loom-lower-graph-memory: scheduled parallel SCF "
                          "must be normalized before graph-region lowering");
    else
      parallel->emitError()
          << "loom-lower-graph-memory: raw "
          << parallel->getName().getStringRef()
          << " requires a selected schedule and provenance before "
             "graph-region lowering";
    return ::mlir::failure();
  }

  ::mlir::WalkResult result =
      graph.getBody().walk([&](::mlir::Operation *op) -> ::mlir::WalkResult {
        if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op)) {
          if (load.getMemRefType().getRank() != 1 ||
              load.getIndices().size() != 1) {
            load.emitError("loom-lower-graph-memory: only rank-one memref.load "
                           "is supported by dataflow.load");
            return ::mlir::WalkResult::interrupt();
          }
        } else if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op)) {
          if (store.getMemRefType().getRank() != 1 ||
              store.getIndices().size() != 1) {
            store.emitError(
                "loom-lower-graph-memory: only rank-one memref.store is "
                "supported by dataflow.store");
            return ::mlir::WalkResult::interrupt();
          }
        }

        auto findMemoryCapability = [&](::mlir::TypeRange types) {
          for (::mlir::Type type : types)
            if (isGraphMemoryCapabilityType(type))
              return type;
          return ::mlir::Type{};
        };
        if (::llvm::isa<::dataflow::CarryOp, ::dataflow::MuxOp,
                        ::dataflow::DemuxOp, ::dataflow::GateOp,
                        ::dataflow::InvariantOp>(op)) {
          ::mlir::Type memory = findMemoryCapability(op->getOperandTypes());
          if (!memory)
            memory = findMemoryCapability(op->getResultTypes());
          if (memory) {
            op->emitError() << "cannot lower memory capability " << memory
                            << " through " << op->getName().getStringRef();
            return ::mlir::WalkResult::interrupt();
          }
        } else if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
          ::mlir::Type memory = findMemoryCapability(ifOp.getResultTypes());
          if (memory) {
            ifOp.emitError()
                << "cannot lower selected memory capability " << memory
                << " through dataflow.mux/demux";
            return ::mlir::WalkResult::interrupt();
          }
        } else if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
          ::mlir::Type memory =
              findMemoryCapability(forOp.getInitArgs().getTypes());
          if (memory) {
            forOp.emitError()
                << "cannot lower loop-carried memory capability " << memory
                << " through dataflow.carry";
            return ::mlir::WalkResult::interrupt();
          }
          if (::mlir::failed(::loom::lowering::inferStreamStepKind(forOp))) {
            forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                            "'loom.stream_step_kind'");
            return ::mlir::WalkResult::interrupt();
          }
          if (::mlir::failed(::loom::lowering::inferStreamPredicate(forOp))) {
            forOp.emitError("loom-lower-graph-memory: scf.for has invalid "
                            "'loom.stream_predicate'");
            return ::mlir::WalkResult::interrupt();
          }
        } else if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
          ::mlir::Type memory =
              findMemoryCapability(whileOp.getInits().getTypes());
          if (memory) {
            whileOp.emitError()
                << "cannot lower loop-carried memory capability " << memory
                << " through dataflow.carry";
            return ::mlir::WalkResult::interrupt();
          }
        }
        if (op->getName().getDialectNamespace() == "scf" &&
            !::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                         ::mlir::scf::WhileOp, ::mlir::scf::YieldOp,
                         ::mlir::scf::ConditionOp>(op)) {
          op->emitError("loom-lower-graph-memory: unsupported residual SCF "
                        "must be normalized before graph-region lowering");
          return ::mlir::WalkResult::interrupt();
        }
        bool modeled =
            ::llvm::isa<::mlir::scf::IfOp, ::mlir::scf::ForOp,
                        ::mlir::scf::WhileOp, ::mlir::scf::YieldOp,
                        ::mlir::scf::ConditionOp, ::mlir::memref::LoadOp,
                        ::mlir::memref::StoreOp, ::dataflow::LoadOp,
                        ::dataflow::StoreOp, ::dataflow::GraphReturnOp,
                        ::mlir::LLVM::LoadOp, ::mlir::LLVM::StoreOp,
                        ::mlir::LLVM::MemcpyOp>(op) ||
            op->getName().getStringRef() == "llvm.intr.memset";
        bool nested = op->getBlock() != &entry;
        bool admissibleLeaf = ::mlir::isPure(op) ||
                              (!nested && isKnownLeafComputation(op));
        if (!modeled &&
            (op->getNumRegions() != 0 || !admissibleLeaf ||
             op->getName().getStringRef() == "llvm.call")) {
          op->emitError()
              << "loom-lower-graph-memory: effectful or unmodeled graph "
                 "operation '"
              << op->getName().getStringRef() << "' is unsupported";
          return ::mlir::WalkResult::interrupt();
        }
        return ::mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

class GraphRegionLowerer {
public:
  explicit GraphRegionLowerer(::dataflow::GraphFuncOp graph)
      : graph(graph), builder(graph.getContext()),
        entry(graph.getBody().front()), anchor(entry.getTerminator()) {}

  ::mlir::LogicalResult run() {
    collectPartitions();
    MemoryState initial(partitionCount);
    for (MemoryFrontier &frontier : initial)
      frontier = {graph.getStart(), graph.getStart()};

    RegionResult result = lowerBlock(entry, graph.getStart(), std::move(initial));
    ::loom::lowering::lowerGraphIndexDomains(graph);
    finalizeReturn(::llvm::cast<::dataflow::GraphReturnOp>(anchor), result);
    return ::mlir::success();
  }

private:
  ::dataflow::GraphFuncOp graph;
  ::mlir::OpBuilder builder;
  ::mlir::Block &entry;
  ::mlir::Operation *anchor;
  unsigned partitionCount = 0;
  ::llvm::DenseMap<::mlir::Value, unsigned> partitionByRoot;
  ::mlir::Value sharedBoundaryRoot;
  ::llvm::DenseMap<::mlir::FlatSymbolRefAttr, ::mlir::Value> globalRoots;
  ::llvm::DenseMap<::mlir::Operation *, ::llvm::SmallVector<unsigned, 4>>
      partitionsByAccess;

  void setInsertionPoint(::mlir::Location) {
    builder.setInsertionPoint(anchor);
  }

  std::optional<::mlir::Value> findKnownRoot(::mlir::Value value) const {
    ::llvm::DenseSet<::mlir::Value> visited;
    while (value && visited.insert(value).second) {
      if (::llvm::isa<::mlir::BlockArgument>(value))
        return value;
      ::mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return value;
      if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                      ::mlir::memref::GetGlobalOp>(def))
        return value;
      if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
        value = view.getViewSource();
        continue;
      }
      if (auto cast =
              ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
        if (cast.getInputs().size() != 1)
          return std::nullopt;
        value = cast.getInputs().front();
        continue;
      }
      ::llvm::StringRef name = def->getName().getStringRef();
      if ((name == "dataflow.partition_layout" ||
           name == "dataflow.map_info") &&
          def->getNumOperands() == 1) {
        value = def->getOperand(0);
        continue;
      }
      return std::nullopt;
    }
    return std::nullopt;
  }

  bool isMemoryCapabilityCapture(::mlir::Value value) {
    if (!isGraphMemoryCapabilityType(value.getType()))
      return false;

    ::llvm::DenseSet<::mlir::Value> visited;
    while (value && visited.insert(value).second) {
      if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(value)) {
        if (argument.getOwner() != &entry || argument.getArgNumber() == 0)
          return true;
        return graph.getInputPortKind(argument.getArgNumber() - 1) ==
               ::dataflow::GraphPortKind::Memory;
      }

      ::mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return true;
      if (::llvm::isa<::mlir::memref::AllocOp, ::mlir::memref::AllocaOp,
                      ::mlir::memref::GetGlobalOp,
                      ::mlir::LLVM::AddressOfOp>(def))
        return true;
      if (auto view = ::llvm::dyn_cast<::mlir::ViewLikeOpInterface>(def)) {
        value = view.getViewSource();
        continue;
      }
      if (auto cast =
              ::llvm::dyn_cast<::mlir::UnrealizedConversionCastOp>(def)) {
        if (cast.getInputs().size() != 1)
          return true;
        value = cast.getInputs().front();
        continue;
      }
      if (auto gep = ::llvm::dyn_cast<::mlir::LLVM::GEPOp>(def)) {
        value = gep.getBase();
        continue;
      }
      return true;
    }
    return true;
  }

  ::llvm::SmallVector<::mlir::Value, 8>
  collectProjectedCaptures(::mlir::Region &region) {
    ::llvm::SetVector<::mlir::Value> candidates;
    ::mlir::getUsedValuesDefinedAbove(region, region, candidates);

    ::llvm::SmallVector<::mlir::Value, 8> captures;
    for (::mlir::Value value : candidates) {
      if (isMemoryCapabilityCapture(value))
        continue;
      bool hasSemanticUse = false;
      for (::mlir::OpOperand &use : value.getUses()) {
        if (!isUseInside(use, region))
          continue;
        if (!isCompilerOwnedControlUse(use)) {
          hasSemanticUse = true;
          break;
        }
      }
      if (hasSemanticUse)
        captures.push_back(value);
    }
    return captures;
  }

  bool hasExplicitNoAlias(::mlir::BlockArgument argument) const {
    if (argument.getOwner() != &entry || argument.getArgNumber() == 0)
      return false;
    ::mlir::DictionaryAttr attrs =
        ::mlir::function_interface_impl::getArgAttrDict(
            graph, argument.getArgNumber() - 1);
    return attrs && attrs.contains("llvm.noalias");
  }

  ::mlir::Value canonicalizeRoot(::mlir::Value root) {
    if (auto argument = ::llvm::dyn_cast<::mlir::BlockArgument>(root)) {
      if (argument.getOwner() == &entry) {
        assert(argument.getArgNumber() > 0 &&
               "start cannot be a memory capability root");
        assert(graph.getInputPortKind(argument.getArgNumber() - 1) ==
                   ::dataflow::GraphPortKind::Memory &&
               "boundary memory root must come from the memory segment");
      }
      if (argument.getOwner() == &entry && !hasExplicitNoAlias(argument)) {
        if (!sharedBoundaryRoot)
          sharedBoundaryRoot = root;
        return sharedBoundaryRoot;
      }
      return root;
    }
    if (auto global = root.getDefiningOp<::mlir::memref::GetGlobalOp>()) {
      return globalRoots.try_emplace(global.getNameAttr(), root).first->second;
    }
    return root;
  }

  ::mlir::Value getMemoryOperand(::mlir::Operation *op) const {
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op))
      return load.getMem();
    if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op))
      return store.getMem();
    if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op))
      return load.getMemref();
    if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op))
      return store.getMemref();
    return {};
  }

  bool isMemoryLeaf(::mlir::Operation *op) const {
    return static_cast<bool>(getMemoryOperand(op));
  }

  void collectPartitions() {
    struct AccessRoot {
      ::mlir::Operation *op;
      std::optional<::mlir::Value> root;
    };
    ::llvm::SmallVector<AccessRoot, 8> accesses;
    bool hasUnknown = false;
    graph.getBody().walk([&](::mlir::Operation *op) {
      ::mlir::Value mem = getMemoryOperand(op);
      if (!mem)
        return ::mlir::WalkResult::advance();
      std::optional<::mlir::Value> root = findKnownRoot(mem);
      if (root)
        root = canonicalizeRoot(*root);
      accesses.push_back({op, root});
      if (!root) {
        hasUnknown = true;
        return ::mlir::WalkResult::advance();
      }
      if (!partitionByRoot.contains(*root)) {
        unsigned index = partitionCount++;
        partitionByRoot.try_emplace(*root, index);
      }
      return ::mlir::WalkResult::advance();
    });
    if (hasUnknown)
      ++partitionCount;
    for (const AccessRoot &access : accesses) {
      ::llvm::SmallVector<unsigned, 4> membership;
      if (access.root) {
        membership.push_back(partitionByRoot.find(*access.root)->second);
      } else {
        membership.reserve(partitionCount);
        for (unsigned i = 0; i < partitionCount; ++i)
          membership.push_back(i);
      }
      partitionsByAccess.try_emplace(access.op, std::move(membership));
    }
  }

  ::llvm::SmallVector<unsigned, 4> partitionsFor(::mlir::Operation *op) const {
    auto it = partitionsByAccess.find(op);
    assert(it != partitionsByAccess.end() && "memory leaf was not analyzed");
    return it->second;
  }

  ::llvm::SmallBitVector touchedPartitions(::mlir::Region &region) const {
    ::llvm::SmallBitVector touched(partitionCount);
    region.walk([&](::mlir::Operation *op) {
      if (!isMemoryLeaf(op))
        return ::mlir::WalkResult::advance();
      for (unsigned partition : partitionsFor(op))
        touched.set(partition);
      return ::mlir::WalkResult::advance();
    });
    return touched;
  }

  bool causallyDependsOn(::mlir::Value event, ::mlir::Value prerequisite,
                         ::llvm::DenseSet<::mlir::Value> &visited) const {
    if (event == prerequisite)
      return true;
    if (!event || !visited.insert(event).second)
      return false;
    ::mlir::Operation *def = event.getDefiningOp();
    if (!def)
      return false;

    if (auto sync = ::llvm::dyn_cast<::dataflow::SyncOp>(def)) {
      return ::llvm::any_of(sync.getInputs(), [&](::mlir::Value input) {
        ::llvm::DenseSet<::mlir::Value> inputVisited = visited;
        return causallyDependsOn(input, prerequisite, inputVisited);
      });
    }
    if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(def))
      return event == load.getDone() &&
             causallyDependsOn(load.getCtrl(), prerequisite, visited);
    if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(def))
      return event == store.getDone() &&
             causallyDependsOn(store.getCtrl(), prerequisite, visited);
    if (auto demux = ::llvm::dyn_cast<::dataflow::DemuxOp>(def))
      return causallyDependsOn(demux.getInput(), prerequisite, visited);
    if (auto mux = ::llvm::dyn_cast<::dataflow::MuxOp>(def)) {
      return ::llvm::all_of(mux.getInputs(), [&](::mlir::Value input) {
        ::llvm::DenseSet<::mlir::Value> laneVisited = visited;
        return causallyDependsOn(input, prerequisite, laneVisited);
      });
    }
    if (auto carry = ::llvm::dyn_cast<::dataflow::CarryOp>(def)) {
      ::llvm::DenseSet<::mlir::Value> initVisited = visited;
      ::llvm::DenseSet<::mlir::Value> carryVisited = visited;
      return causallyDependsOn(carry.getInit(), prerequisite, initVisited) &&
             causallyDependsOn(carry.getCarry(), prerequisite, carryVisited);
    }
    if (auto invariant = ::llvm::dyn_cast<::dataflow::InvariantOp>(def))
      return causallyDependsOn(invariant.getInit(), prerequisite, visited);
    if (auto gate = ::llvm::dyn_cast<::dataflow::GateOp>(def))
      return causallyDependsOn(gate.getBeforeValue(), prerequisite, visited);
    if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(def))
      return causallyDependsOn(constant.getCtrl(), prerequisite, visited);
    return false;
  }

  bool causallyDependsOn(::mlir::Value event,
                         ::mlir::Value prerequisite) const {
    ::llvm::DenseSet<::mlir::Value> visited;
    return causallyDependsOn(event, prerequisite, visited);
  }

  ::llvm::SmallVector<::mlir::Value, 4>
  reduceEvents(::mlir::ValueRange inputs) const {
    ::llvm::SmallVector<::mlir::Value, 4> unique;
    for (::mlir::Value input : inputs)
      if (input && !::llvm::is_contained(unique, input))
        unique.push_back(input);

    ::llvm::SmallVector<::mlir::Value, 4> reduced;
    for (unsigned i = 0; i < unique.size(); ++i) {
      bool covered = false;
      for (unsigned j = 0; j < unique.size(); ++j) {
        if (i != j && causallyDependsOn(unique[j], unique[i])) {
          covered = true;
          break;
        }
      }
      if (!covered)
        reduced.push_back(unique[i]);
    }
    return reduced;
  }

  ::mlir::Value joinEvents(::mlir::ValueRange inputs, ::mlir::Location loc) {
    ::llvm::SmallVector<::mlir::Value, 4> reduced = reduceEvents(inputs);
    if (reduced.size() == 1)
      return reduced.front();

    setInsertionPoint(loc);
    ::llvm::SmallVector<::mlir::Type, 4> types(reduced.size(),
                                               builder.getNoneType());
    auto sync = ::dataflow::SyncOp::create(builder, loc, types, reduced);
    return sync.getOutputs().front();
  }

  void finalizeReturn(::dataflow::GraphReturnOp returnOp,
                      const RegionResult &result) {
    ::llvm::SmallVector<::mlir::Value, 8> candidates{result.execution};
    for (const MemoryFrontier &frontier : result.memory)
      candidates.push_back(frontier.read);
    ::mlir::Value start = graph.getStart();
    bool hasDerivedFrontier = ::llvm::any_of(
        candidates, [&](::mlir::Value candidate) { return candidate != start; });
    for (::mlir::Value witness : returnOp.getComplete())
      if (witness != start || !hasDerivedFrontier)
        candidates.push_back(witness);
    ::llvm::SmallVector<::mlir::Value, 4> reduced =
        reduceEvents(candidates);
    assert(!reduced.empty() && "graph retirement must have a witness");

    ::llvm::SmallVector<::mlir::Value, 4> values(returnOp.getValues().begin(),
                                                 returnOp.getValues().end());
    auto eraseUnusedWriteFrontierMuxes = [&]() {
      for (const MemoryFrontier &frontier : result.memory) {
        auto mux = frontier.write.getDefiningOp<::dataflow::MuxOp>();
        if (mux && mux.getOutput().use_empty())
          mux.erase();
      }
    };
    if (values.empty()) {
      returnOp.getCompleteMutable().assign(reduced);
      eraseUnusedWriteFrontierMuxes();
      return;
    }

    ::mlir::Value publicationBase = joinEvents(reduced, returnOp.getLoc());
    ::llvm::SmallVector<::mlir::Value, 4> publicationFrontier;
    for (auto [index, value] : ::llvm::enumerate(values)) {
      setInsertionPoint(returnOp.getLoc());
      auto sync = ::dataflow::SyncOp::create(
          builder, returnOp.getLoc(),
          ::mlir::TypeRange{builder.getNoneType(), value.getType()},
          ::mlir::ValueRange{publicationBase, value});
      publicationFrontier.push_back(sync.getOutputs().front());
      values[index] = sync.getOutputs().back();
    }
    returnOp.getValuesMutable().assign(values);
    returnOp.getCompleteMutable().assign(publicationFrontier);
    eraseUnusedWriteFrontierMuxes();
  }

  std::pair<::mlir::Value, ::mlir::Value>
  demux(::mlir::Value selector, ::mlir::Value input, ::mlir::Location loc) {
    setInsertionPoint(loc);
    auto op = ::dataflow::DemuxOp::create(
        builder, loc, ::mlir::TypeRange{input.getType(), input.getType()},
        selector, input);
    return {op.getOutputs()[0], op.getOutputs()[1]};
  }

  ::mlir::Value mux(::mlir::Value selector, ::mlir::Value falseValue,
                    ::mlir::Value trueValue, ::mlir::Location loc) {
    setInsertionPoint(loc);
    return ::dataflow::MuxOp::create(builder, loc, falseValue.getType(),
                                     selector,
                                     ::mlir::ValueRange{falseValue, trueValue})
        .getOutput();
  }

  ::mlir::Value gateTrueLane(::mlir::Value phase, ::mlir::Value value,
                             ::mlir::Location loc) {
    setInsertionPoint(loc);
    return ::dataflow::GateOp::create(builder, loc, builder.getI1Type(),
                                      value.getType(), phase, value)
        .getAfterValue();
  }

  void projectForCaptures(::mlir::Region &region, ::mlir::ValueRange captures,
                          ::mlir::Value phase, ::mlir::Location loc) {
    for (::mlir::Value capture : captures) {
      setInsertionPoint(loc);
      ::mlir::Value raw = ::dataflow::InvariantOp::create(
                              builder, loc, capture.getType(), phase, capture)
                              .getOutput();
      replaceUsesInside(capture, gateTrueLane(phase, raw, loc), region);
    }
  }

  ::llvm::SmallVector<::dataflow::InvariantOp, 4>
  projectWhileBeforeCaptures(::mlir::Region &region,
                             ::mlir::ValueRange captures,
                             ::mlir::Value condition, ::mlir::Location loc) {
    ::llvm::SmallVector<::dataflow::InvariantOp, 4> invariants;
    for (::mlir::Value capture : captures) {
      setInsertionPoint(loc);
      auto invariant = ::dataflow::InvariantOp::create(
          builder, loc, capture.getType(), condition, capture);
      replaceUsesInside(capture, invariant.getOutput(), region);
      invariants.push_back(invariant);
    }
    return invariants;
  }

  RegionResult lowerBlock(::mlir::Block &block, ::mlir::Value execution,
                          MemoryState memory) {
    ::llvm::SmallVector<::mlir::Operation *, 16> operations;
    for (::mlir::Operation &op : block.without_terminator())
      operations.push_back(&op);

    for (::mlir::Operation *op : operations) {
      if (auto ifOp = ::llvm::dyn_cast<::mlir::scf::IfOp>(op)) {
        RegionResult result = lowerIf(ifOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto forOp = ::llvm::dyn_cast<::mlir::scf::ForOp>(op)) {
        RegionResult result = lowerFor(forOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto whileOp = ::llvm::dyn_cast<::mlir::scf::WhileOp>(op)) {
        RegionResult result = lowerWhile(whileOp, execution, std::move(memory));
        execution = result.execution;
        memory = std::move(result.memory);
        continue;
      }
      if (auto load = ::llvm::dyn_cast<::mlir::memref::LoadOp>(op)) {
        lowerMemrefLoad(load, execution, memory);
        continue;
      }
      if (auto store = ::llvm::dyn_cast<::mlir::memref::StoreOp>(op)) {
        lowerMemrefStore(store, execution, memory);
        continue;
      }
      if (auto load = ::llvm::dyn_cast<::dataflow::LoadOp>(op)) {
        lowerDataflowLoad(load, execution, memory);
        continue;
      }
      if (auto store = ::llvm::dyn_cast<::dataflow::StoreOp>(op)) {
        lowerDataflowStore(store, execution, memory);
        continue;
      }
      if (auto constant = ::llvm::dyn_cast<::mlir::arith::ConstantOp>(op)) {
        if (constant->getBlock() == &entry)
          continue;
        setInsertionPoint(constant.getLoc());
        ::mlir::Value value =
            ::dataflow::ConstantOp::create(builder, constant.getLoc(),
                                           constant.getType(), execution,
                                           constant.getValue())
                .getValue();
        constant.getResult().replaceAllUsesWith(value);
        constant.erase();
        continue;
      }
      if (auto constant = ::llvm::dyn_cast<::dataflow::ConstantOp>(op))
        constant.getCtrlMutable().assign(execution);

      if (op->getBlock() != &entry) {
        assert(op->getNumRegions() == 0 && ::mlir::isPure(op) &&
               "graph preflight admitted an unmovable operation");
        op->moveBefore(anchor);
      }
    }
    return {execution, std::move(memory)};
  }

  void updateReadFrontiers(::mlir::Operation *op, ::mlir::Value done,
                           MemoryState &memory) {
    for (unsigned partition : partitionsFor(op)) {
      MemoryFrontier &frontier = memory[partition];
      frontier.read =
          joinEvents(::mlir::ValueRange{frontier.read, done}, op->getLoc());
    }
  }

  ::mlir::Value readControl(::mlir::Operation *op, ::mlir::Value execution,
                            MemoryState &memory) {
    ::llvm::SmallVector<::mlir::Value, 8> inputs{execution};
    for (unsigned partition : partitionsFor(op))
      inputs.push_back(memory[partition].write);
    return joinEvents(inputs, op->getLoc());
  }

  ::mlir::Value writeControl(::mlir::Operation *op, ::mlir::Value execution,
                             MemoryState &memory) {
    ::llvm::SmallVector<::mlir::Value, 8> inputs{execution};
    for (unsigned partition : partitionsFor(op))
      inputs.push_back(memory[partition].read);
    return joinEvents(inputs, op->getLoc());
  }

  void updateWriteFrontiers(::mlir::Operation *op, ::mlir::Value done,
                            MemoryState &memory) {
    for (unsigned partition : partitionsFor(op))
      memory[partition] = {done, done};
  }

  void lowerMemrefLoad(::mlir::memref::LoadOp load, ::mlir::Value execution,
                       MemoryState &memory) {
    ::llvm::SmallVector<unsigned, 4> membership = partitionsFor(load);
    ::mlir::Value ctrl = readControl(load, execution, memory);
    setInsertionPoint(load.getLoc());
    auto lowered = ::dataflow::LoadOp::create(
        builder, load.getLoc(), load.getType(), builder.getNoneType(),
        load.getMemref(), load.getIndices().front(), ctrl);
    partitionsByAccess.try_emplace(lowered, std::move(membership));
    load.getResult().replaceAllUsesWith(lowered.getData());
    updateReadFrontiers(lowered, lowered.getDone(), memory);
    load.erase();
  }

  void lowerMemrefStore(::mlir::memref::StoreOp store, ::mlir::Value execution,
                        MemoryState &memory) {
    ::llvm::SmallVector<unsigned, 4> membership = partitionsFor(store);
    ::mlir::Value ctrl = writeControl(store, execution, memory);
    setInsertionPoint(store.getLoc());
    auto lowered = ::dataflow::StoreOp::create(
        builder, store.getLoc(), builder.getNoneType(), store.getMemref(),
        store.getIndices().front(), store.getValue(), ctrl);
    partitionsByAccess.try_emplace(lowered, std::move(membership));
    updateWriteFrontiers(lowered, lowered.getDone(), memory);
    store.erase();
  }

  void lowerDataflowLoad(::dataflow::LoadOp load, ::mlir::Value execution,
                         MemoryState &memory) {
    load.getCtrlMutable().assign(readControl(load, execution, memory));
    updateReadFrontiers(load, load.getDone(), memory);
    if (load->getBlock() != &entry)
      load->moveBefore(anchor);
  }

  void lowerDataflowStore(::dataflow::StoreOp store, ::mlir::Value execution,
                          MemoryState &memory) {
    store.getCtrlMutable().assign(writeControl(store, execution, memory));
    updateWriteFrontiers(store, store.getDone(), memory);
    if (store->getBlock() != &entry)
      store->moveBefore(anchor);
  }

  RegionResult lowerIf(::mlir::scf::IfOp ifOp, ::mlir::Value execution,
                       MemoryState memory) {
    ::mlir::Location loc = ifOp.getLoc();
    ::mlir::Value selector = ifOp.getCondition();
    auto [falseExecution, trueExecution] = demux(selector, execution, loc);

    ::llvm::SmallBitVector touched = touchedPartitions(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      touched |= touchedPartitions(ifOp.getElseRegion());

    MemoryState falseMemory = memory;
    MemoryState trueMemory = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      auto [falseWrite, trueWrite] =
          demux(selector, memory[partition].write, loc);
      auto [falseRead, trueRead] = demux(selector, memory[partition].read, loc);
      falseMemory[partition] = {falseWrite, falseRead};
      trueMemory[partition] = {trueWrite, trueRead};
    }

    ::llvm::SetVector<::mlir::Value> captures;
    for (::mlir::Value value :
         collectProjectedCaptures(ifOp.getThenRegion()))
      captures.insert(value);
    if (!ifOp.getElseRegion().empty())
      for (::mlir::Value value :
           collectProjectedCaptures(ifOp.getElseRegion()))
        captures.insert(value);
    for (::mlir::Value capture : captures) {
      auto [falseValue, trueValue] = demux(selector, capture, loc);
      replaceUsesInside(capture, trueValue, ifOp.getThenRegion());
      if (!ifOp.getElseRegion().empty())
        replaceUsesInside(capture, falseValue, ifOp.getElseRegion());
    }

    RegionResult trueResult = lowerBlock(ifOp.getThenRegion().front(),
                                         trueExecution, std::move(trueMemory));
    RegionResult falseResult{falseExecution, std::move(falseMemory)};
    if (!ifOp.getElseRegion().empty())
      falseResult = lowerBlock(ifOp.getElseRegion().front(), falseExecution,
                               std::move(falseResult.memory));

    auto thenYield = ::llvm::cast<::mlir::scf::YieldOp>(
        ifOp.getThenRegion().front().getTerminator());
    ::mlir::scf::YieldOp elseYield;
    if (!ifOp.getElseRegion().empty())
      elseYield = ::llvm::cast<::mlir::scf::YieldOp>(
          ifOp.getElseRegion().front().getTerminator());
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      ::mlir::Value result =
          mux(selector, elseYield.getOperand(i), thenYield.getOperand(i), loc);
      ifOp.getResult(i).replaceAllUsesWith(result);
    }

    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      output[partition].write =
          mux(selector, falseResult.memory[partition].write,
              trueResult.memory[partition].write, loc);
      output[partition].read = mux(selector, falseResult.memory[partition].read,
                                   trueResult.memory[partition].read, loc);
    }
    ::mlir::Value outputExecution =
        mux(selector, falseResult.execution, trueResult.execution, loc);
    ifOp.erase();
    return {outputExecution, std::move(output)};
  }

  RegionResult lowerFor(::mlir::scf::ForOp forOp, ::mlir::Value execution,
                        MemoryState memory) {
    ::mlir::Location loc = forOp.getLoc();
    setInsertionPoint(loc);

    ::mlir::Value lower = forOp.getLowerBound();
    ::mlir::Value upper = forOp.getUpperBound();
    ::mlir::Value step = forOp.getStep();
    ::mlir::Type streamType = lower.getType();
    bool indexLoop = ::llvm::isa<::mlir::IndexType>(streamType);
    if (indexLoop) {
      streamType =
          ::mlir::IntegerType::get(graph.getContext(), ::loom::getIndexWidth());
      lower =
          ::mlir::arith::IndexCastOp::create(builder, loc, streamType, lower)
              .getResult();
      upper =
          ::mlir::arith::IndexCastOp::create(builder, loc, streamType, upper)
              .getResult();
      step = ::mlir::arith::IndexCastOp::create(builder, loc, streamType, step)
                 .getResult();
    }
    auto stepKind = ::loom::lowering::inferStreamStepKind(forOp);
    auto predicate = ::loom::lowering::inferStreamPredicate(forOp);
    auto stream = ::dataflow::StreamOp::create(
        builder, loc, streamType, builder.getI1Type(), lower, upper, step,
        *stepKind, *predicate);
    ::mlir::Value phase = stream.getPhase();
    ::mlir::Value bodyIv = stream.getIv();
    if (indexLoop)
      bodyIv = ::mlir::arith::IndexCastOp::create(
                   builder, loc, builder.getIndexType(), bodyIv)
                   .getResult();

    auto executionCarry = ::dataflow::CarryOp::create(
        builder, loc, builder.getNoneType(), phase, execution, execution);
    auto [executionExit, executionBody] =
        demux(phase, executionCarry.getOutput(), loc);

    ::llvm::SmallVector<::mlir::Value, 8> captures =
        collectProjectedCaptures(forOp.getRegion());
    ::llvm::SmallVector<::dataflow::CarryOp, 4> valueCarries;
    ::llvm::SmallVector<::mlir::Value, 4> valueExits;
    for (::mlir::Value init : forOp.getInitArgs()) {
      setInsertionPoint(loc);
      auto carry = ::dataflow::CarryOp::create(builder, loc, init.getType(),
                                               phase, init, init);
      auto [exit, body] = demux(phase, carry.getOutput(), loc);
      valueCarries.push_back(carry);
      valueExits.push_back(exit);
      replaceUsesInside(forOp.getRegionIterArgs()[valueCarries.size() - 1],
                        body, forOp.getRegion());
    }
    replaceUsesInside(forOp.getInductionVar(), bodyIv, forOp.getRegion());
    projectForCaptures(forOp.getRegion(), captures, phase, loc);

    ::llvm::SmallBitVector touched = touchedPartitions(forOp.getRegion());
    MemoryState bodyMemory = memory;
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> writeCarries(
        partitionCount);
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> readCarries(
        partitionCount);
    ::llvm::SmallVector<::mlir::Value, 4> writeExits(partitionCount);
    ::llvm::SmallVector<::mlir::Value, 4> readExits(partitionCount);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      setInsertionPoint(loc);
      auto writeCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), phase, memory[partition].write,
          memory[partition].write);
      auto readCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), phase, memory[partition].read,
          memory[partition].read);
      auto [writeExit, writeBody] = demux(phase, writeCarry.getOutput(), loc);
      auto [readExit, readBody] = demux(phase, readCarry.getOutput(), loc);
      writeCarries[partition] = writeCarry;
      readCarries[partition] = readCarry;
      writeExits[partition] = writeExit;
      readExits[partition] = readExit;
      bodyMemory[partition] = {writeBody, readBody};
    }

    RegionResult bodyResult = lowerBlock(forOp.getRegion().front(),
                                         executionBody, std::move(bodyMemory));
    auto yield = ::llvm::cast<::mlir::scf::YieldOp>(
        forOp.getRegion().front().getTerminator());
    executionCarry.getCarryMutable().assign(bodyResult.execution);
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      valueCarries[i].getCarryMutable().assign(yield.getOperand(i));

    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCarryMutable().assign(
          bodyResult.memory[partition].write);
      readCarries[partition]->getCarryMutable().assign(
          bodyResult.memory[partition].read);
      output[partition] = {writeExits[partition], readExits[partition]};
    }
    for (unsigned i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(valueExits[i]);
    forOp.erase();
    return {executionExit, std::move(output)};
  }

  RegionResult lowerWhile(::mlir::scf::WhileOp whileOp, ::mlir::Value execution,
                          MemoryState memory) {
    ::mlir::Location loc = whileOp.getLoc();
    auto condition = ::llvm::cast<::mlir::scf::ConditionOp>(
        whileOp.getBefore().front().getTerminator());

    ::llvm::SmallVector<::mlir::Value, 8> beforeCaptures =
        collectProjectedCaptures(whileOp.getBefore());
    ::llvm::SmallVector<::mlir::Value, 8> afterCaptures =
        collectProjectedCaptures(whileOp.getAfter());

    setInsertionPoint(loc);
    ::mlir::Value pendingSelector =
        ::mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                          builder.getBoolAttr(false))
            .getResult();
    auto executionCarry =
        ::dataflow::CarryOp::create(builder, loc, builder.getNoneType(),
                                    pendingSelector, execution, execution);

    ::llvm::SmallVector<::dataflow::CarryOp, 4> valueCarries;
    for (::mlir::Value init : whileOp.getInits()) {
      auto carry = ::dataflow::CarryOp::create(builder, loc, init.getType(),
                                               pendingSelector, init, init);
      valueCarries.push_back(carry);
    }
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      replaceUsesInside(whileOp.getBeforeArguments()[i],
                        valueCarries[i].getOutput(), whileOp.getBefore());
    ::llvm::SmallVector<::dataflow::InvariantOp, 4> beforeInvariants =
        projectWhileBeforeCaptures(whileOp.getBefore(), beforeCaptures,
                                   pendingSelector, loc);

    ::llvm::SmallBitVector touched = touchedPartitions(whileOp.getBefore());
    touched |= touchedPartitions(whileOp.getAfter());
    MemoryState beforeMemory = memory;
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> writeCarries(
        partitionCount);
    ::llvm::SmallVector<std::optional<::dataflow::CarryOp>, 4> readCarries(
        partitionCount);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      setInsertionPoint(loc);
      auto writeCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), pendingSelector,
          memory[partition].write, memory[partition].write);
      auto readCarry = ::dataflow::CarryOp::create(
          builder, loc, builder.getNoneType(), pendingSelector,
          memory[partition].read, memory[partition].read);
      writeCarries[partition] = writeCarry;
      readCarries[partition] = readCarry;
      beforeMemory[partition] = {writeCarry.getOutput(), readCarry.getOutput()};
    }

    RegionResult beforeResult =
        lowerBlock(whileOp.getBefore().front(), executionCarry.getOutput(),
                   std::move(beforeMemory));
    ::mlir::Value selector = condition.getCondition();
    executionCarry.getCondMutable().assign(selector);
    for (::dataflow::CarryOp carry : valueCarries)
      carry.getCondMutable().assign(selector);
    for (::dataflow::InvariantOp invariant : beforeInvariants)
      invariant.getCondMutable().assign(selector);
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCondMutable().assign(selector);
      readCarries[partition]->getCondMutable().assign(selector);
    }
    pendingSelector.getDefiningOp()->erase();

    auto [executionExit, unusedExecution] =
        demux(selector, beforeResult.execution, loc);
    (void)unusedExecution;
    ::mlir::Value executionAfter =
        gateTrueLane(selector, beforeResult.execution, loc);

    MemoryState afterMemory = beforeResult.memory;
    MemoryState output = memory;
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      auto [writeExit, writeAfter] =
          demux(selector, beforeResult.memory[partition].write, loc);
      auto [readExit, readAfter] =
          demux(selector, beforeResult.memory[partition].read, loc);
      output[partition] = {writeExit, readExit};
      afterMemory[partition] = {writeAfter, readAfter};
    }

    ::llvm::SmallVector<::mlir::Value, 4> resultValues;
    for (::mlir::Value value : condition.getArgs()) {
      auto [exit, after] = demux(selector, value, loc);
      resultValues.push_back(exit);
      replaceUsesInside(whileOp.getAfterArguments()[resultValues.size() - 1],
                        after, whileOp.getAfter());
    }
    projectForCaptures(whileOp.getAfter(), afterCaptures, selector, loc);

    RegionResult afterResult = lowerBlock(
        whileOp.getAfter().front(), executionAfter, std::move(afterMemory));
    auto yield = ::llvm::cast<::mlir::scf::YieldOp>(
        whileOp.getAfter().front().getTerminator());
    executionCarry.getCarryMutable().assign(afterResult.execution);
    for (unsigned i = 0; i < valueCarries.size(); ++i)
      valueCarries[i].getCarryMutable().assign(yield.getOperand(i));
    for (int partition = touched.find_first(); partition >= 0;
         partition = touched.find_next(partition)) {
      writeCarries[partition]->getCarryMutable().assign(
          afterResult.memory[partition].write);
      readCarries[partition]->getCarryMutable().assign(
          afterResult.memory[partition].read);
    }
    for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
      whileOp.getResult(i).replaceAllUsesWith(resultValues[i]);
    whileOp.erase();
    return {executionExit, std::move(output)};
  }
};

} // namespace

namespace loom {
namespace lowering {

::mlir::LogicalResult
checkGraphRegionLoweringPreconditions(::mlir::ModuleOp module) {
  ::mlir::WalkResult result =
      module.walk([&](::dataflow::GraphFuncOp graph) -> ::mlir::WalkResult {
        if (!graph.isExternal() && ::mlir::failed(checkOneGraph(graph)))
          return ::mlir::WalkResult::interrupt();
        return ::mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

::mlir::LogicalResult lowerGraphRegions(::dataflow::GraphFuncOp graph) {
  return GraphRegionLowerer(graph).run();
}

} // namespace lowering
} // namespace loom
