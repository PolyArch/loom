#include "Frontend/Analysis/StoredMemoryProvenance.h"
#include "Common/PointerLayout.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/Analysis/ByteAddressDomain.h"
#include "Frontend/Analysis/CountedLoopProjection.h"
#include "Frontend/Analysis/MemoryAddressProjection.h"
#include "Frontend/Analysis/MemoryProvenance.h"
#include "Frontend/Analysis/ScalarGuardDomain.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::frontend::analysis {
namespace {

/// An inline assembly with an empty template executes no instruction, so it
/// cannot write memory; its clobbers constrain the compiler only. The
/// computation-interval markers are such barriers.
bool isInstructionFreeInlineAsm(mlir::Operation *operation) {
  auto assembly = llvm::dyn_cast<mlir::LLVM::InlineAsmOp>(operation);
  return assembly && assembly.getAsmString().empty();
}

} // namespace

class StoredMemoryProvenance::Impl final {
  static constexpr std::size_t maximumStaticValues = 4096;
  static constexpr std::size_t maximumControlAlternatives = 16384;
  static constexpr unsigned maximumProjectionDepth = 128;
  struct Path final {
    llvm::DenseMap<mlir::Value, mlir::Value> aliases;
    llvm::DenseMap<mlir::Value, llvm::APInt> constants;
    llvm::SmallVector<std::pair<mlir::Value, llvm::APInt>> excluded;
  };

  struct Frame final {
    mlir::Operation *callable;
    Frame *caller = nullptr;
    mlir::Operation *start = nullptr;
    mlir::Operation *completion = nullptr;
    bool hasUnretiredAsyncEffect = false;

    mlir::Region &body() const { return callable->getRegion(0); }
    mlir::Block *entry() const { return &body().front(); }
    mlir::ValueRange actuals() const {
      if (auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(start))
        return call.getArgOperands();
      if (auto launch = llvm::dyn_cast_or_null<dataflow::ThreadLaunchOp>(start))
        return launch.getBodyOperands();
      return {};
    }
  };
  struct Address final {
    mlir::Value root;
    Frame *rootFrame = nullptr;
    int64_t bias = 0;
    uint64_t bytes = 0;
    llvm::SmallVector<loom::frontend::analysis::LinearByteTerm, 4> terms;
    /// Every index step from the root object to this address is in bounds, so
    /// a defined access lies inside the root's fixed extent.
    bool inBoundsOfRoot = false;
  };

  struct PointerRoots final {
    llvm::SmallVector<std::pair<mlir::Value, Frame *>> roots;
    bool mayBeNull = false;
  };

public:
  explicit Impl(mlir::LLVM::LLVMFuncOp rootCallable) {
    if (!rootCallable || !rootCallable.getBody().hasOneBlock()) {
      refusal_ = StoredPointerRefusal::OpenInvocationDomain;
      return;
    }
    frames_.push_back(std::make_unique<Frame>(Frame{rootCallable}));
    for (std::size_t index = 0; index < frames_.size(); ++index) {
      Frame &frame = *frames_[index];
      frame.callable->walk([&](mlir::Operation *operation) {
        if (operation == frame.callable)
          return;
        mlir::Operation *target = nullptr;
        mlir::Operation *completion = operation;
        bool invocation = false;
        if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(operation)) {
          invocation = true;
          if (call.getCalleeAttr())
            target = mlir::SymbolTable::lookupNearestSymbolFrom<
                mlir::LLVM::LLVMFuncOp>(call, call.getCalleeAttr());
        } else if (auto launch =
                       llvm::dyn_cast<dataflow::ThreadLaunchOp>(operation)) {
          invocation = true;
          // A stored-program interval is serial only when its explicit wait
          // retires this launch before the caller can perform another effect.
          // Grid invocations require a separate race/iteration proof.
          if (launch.getGridUpperBounds().empty() &&
              launch.getAsyncDependencies().empty() &&
              completedLaunch(launch)) {
            target =
                mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
                    launch, launch.getCalleeAttr());
            completion = launch->getNextNode();
          } else {
            frame.hasUnretiredAsyncEffect = true;
          }
        } else if (llvm::isa<dataflow::GraphLaunchOp>(operation)) {
          frame.hasUnretiredAsyncEffect = true;
        }
        if (invocation) {
          bool recursive = false;
          for (Frame *parent = &frame; parent; parent = parent->caller)
            recursive |= target && parent->callable == target;
          if (!target || target->getNumRegions() != 1 ||
              !target->getRegion(0).hasOneBlock() || recursive ||
              frames_.size() == maximumStaticValues) {
            if (!openWriteDomain_)
              openWriteDomain_ = StoredPointerRefusal::OpenInvocationDomain;
            return;
          }
          frames_.push_back(std::make_unique<Frame>(
              Frame{target, &frame, operation, completion}));
          return;
        }
        if (auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(operation)) {
          auto launch = llvm::dyn_cast_or_null<dataflow::ThreadLaunchOp>(
              wait->getPrevNode());
          if (launch && completedLaunch(launch))
            return;
        }
        if (llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                      mlir::LLVM::AllocaOp, mlir::LLVM::LifetimeStartOp,
                      mlir::LLVM::LifetimeEndOp>(operation) ||
            operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>() ||
            mlir::isMemoryEffectFree(operation) ||
            isInstructionFreeInlineAsm(operation))
          return;
        if (!openWriteDomain_)
          openWriteDomain_ = StoredPointerRefusal::UnsupportedMemoryEffect;
      });
    }
    if (!openWriteDomain_)
      deriveReloadEquivalences();
    for (const auto &frame : frames_)
      frame->callable->walk([&](mlir::LLVM::StoreOp write) {
        effects_.push_back(writeEffect(write, *frame));
      });
    writePositions_.resize(effects_.size());
    if (!openWriteDomain_)
      deriveReloadEquivalences(true);
  }

  ReachingPointerValueOutcome
  reaching(mlir::LLVM::LoadOp read,
           llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
    if (frames_.empty())
      return StoredPointerRefusal::OpenInvocationDomain;
    if (read.getVolatile_() ||
        read.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return StoredPointerRefusal::UnsupportedMemoryEffect;
    auto value = forwarded(read.getResult(), Path());
    if (value != read.getResult() &&
        llvm::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
      return value;
    mlir::Value reachingValue;
    for (const auto &candidate : frames_) {
      Frame &frame = *candidate;
      if (frame.callable != enclosingCallable(read.getResult()))
        continue;
      if (invocationPath.empty() && openWriteDomain_ && frame.caller)
        continue;
      for (Frame *ancestor = &frame; ancestor; ancestor = ancestor->caller)
        if (ancestor->hasUnretiredAsyncEffect)
          return StoredPointerRefusal::OpenInvocationDomain;
      llvm::SmallVector<mlir::Operation *> path;
      for (Frame *current = &frame; current->caller; current = current->caller)
        path.push_back(current->start);
      std::reverse(path.begin(), path.end());
      if (!invocationPath.empty() &&
          (path.size() > invocationPath.size() ||
           !llvm::all_of(llvm::enumerate(path), [&](auto entry) {
             return entry.value() ==
                    mlir::LLVM::CallOp(invocationPath[entry.index()])
                        .getOperation();
           })))
        continue;
      auto location = address(read.getAddr(), read.getType(), frame);
      auto offset = location ? constantByteOffset(*location) : std::nullopt;
      auto extent = location ? allocationExtent(location->root) : std::nullopt;
      if (!location || !offset || !extent)
        return StoredPointerRefusal::UnknownByteAddress;
      if (!withinAllocation(*location, *offset))
        return StoredPointerRefusal::OutOfBoundsAccess;
      mlir::Value local;
      for (const auto &possible : effects_) {
        if (!possible)
          continue;
        const WriteEffect &effect = *possible;
        auto write = effect.operation;
        if (effect.kind != WriteKind::Scalar ||
            write.getValue().getType() != read.getType() ||
            !sameFrameRoot(*location, effect.destination) ||
            location->bytes != effect.destination.bytes)
          continue;
        auto destination = constantByteOffset(effect.destination);
        if (!destination || *offset != *destination ||
            !invocationOrdered(write, effect.frame, read, &frame))
          continue;
        Frame *current = &frame;
        mlir::Operation *point = read;
        bool preserves = true;
        while (current != effect.frame && current->caller) {
          preserves &=
              noInterveningWrite(nullptr, point, *current, *location, true);
          point = current->start;
          current = current->caller;
        }
        if (!preserves || current != effect.frame ||
            !noInterveningWrite(write, point, *current, *location, true))
          continue;
        if (local && local != write.getValue())
          return StoredPointerRefusal::DistinctPointerOrigins;
        local = write.getValue();
      }
      if (!local)
        return StoredPointerRefusal::IncompleteInitialization;
      if (reachingValue && reachingValue != local)
        return StoredPointerRefusal::DistinctPointerOrigins;
      reachingValue = local;
    }
    return reachingValue ? ReachingPointerValueOutcome(reachingValue)
                         : ReachingPointerValueOutcome(
                               StoredPointerRefusal::OpenInvocationDomain);
  }

  StoredPointerTargetOutcome
  project(mlir::Value pointer, llvm::ArrayRef<mlir::Value> boundary = {}) {
    if (frames_.empty())
      return StoredPointerRefusal::OpenInvocationDomain;
    if (openWriteDomain_ == StoredPointerRefusal::OpenInvocationDomain)
      return *openWriteDomain_;
    refusal_ = StoredPointerRefusal::UnsupportedPointerOrigin;
    mlir::Operation *callable = enclosingCallable(pointer);
    std::optional<StoredPointerTarget> result;
    for (const auto &frame : frames_) {
      if (frame->callable != callable)
        continue;
      auto possible =
          roots(pointer, *frame,
                pointer.getDefiningOp() ? enclosingPath(pointer.getDefiningOp())
                                        : Path());
      if (!possible)
        return refusal_;
      if (possible->roots.empty())
        return StoredPointerRefusal::NullPointerOrigin;
      if (possible->roots.size() != 1)
        return StoredPointerRefusal::DistinctPointerOrigins;
      auto [root, owner] = possible->roots.front();
      if (!boundary.empty()) {
        mlir::Value represented;
        for (mlir::Value input : boundary) {
          if (!llvm::isa<mlir::LLVM::LLVMPointerType>(input.getType()))
            continue;
          auto projected = address(
              input, mlir::IntegerType::get(pointer.getContext(), 8), *frame);
          auto offset =
              projected ? constantByteOffset(*projected) : std::nullopt;
          if (projected && projected->root == root &&
              projected->rootFrame == owner && offset && *offset == 0) {
            represented = input;
            break;
          }
        }
        if (!represented)
          return StoredPointerRefusal::OriginNotInBoundary;
        root = represented;
      }
      if (result && result->root != root)
        return StoredPointerRefusal::DistinctPointerOrigins;
      if (!result)
        result = StoredPointerTarget{root, possible->mayBeNull};
      else
        result->mayBeNull |= possible->mayBeNull;
    }
    return result ? StoredPointerTargetOutcome(*result)
                  : StoredPointerTargetOutcome(
                        StoredPointerRefusal::OpenInvocationDomain);
  }

private:
  std::nullopt_t refuse(StoredPointerRefusal reason) {
    refusal_ = reason;
    return std::nullopt;
  }

  static bool completedLaunch(dataflow::ThreadLaunchOp launch) {
    auto wait =
        llvm::dyn_cast_or_null<dataflow::ThreadWaitOp>(launch->getNextNode());
    return wait && wait.getAsyncDependencies().size() == 1 &&
           wait.getAsyncDependencies().front() == launch.getAsyncToken();
  }

  static mlir::Operation *enclosingCallable(mlir::Value value) {
    for (mlir::Operation *scope = value.getParentBlock()->getParentOp(); scope;
         scope = scope->getParentOp())
      if (llvm::isa<mlir::LLVM::LLVMFuncOp, dataflow::ThreadOp>(scope))
        return scope;
    return nullptr;
  }

  mlir::Value forwarded(mlir::Value value, const Path &path) {
    llvm::SmallDenseSet<mlir::Value, 8> seen;
    while (seen.insert(value).second) {
      auto found = path.aliases.find(value);
      if (found != path.aliases.end()) {
        value = found->second;
        continue;
      }
      auto reload = reloadEquivalences_.find(value);
      if (reload == reloadEquivalences_.end())
        break;
      value = reload->second;
    }
    return value;
  }

  std::optional<llvm::APInt> evaluate(mlir::Value value, const Path &path,
                                      unsigned depth = 0) {
    if (depth > maximumProjectionDepth)
      return std::nullopt;
    auto fixed = path.constants.find(value);
    if (fixed != path.constants.end())
      return fixed->second;
    value = forwarded(value, path);
    fixed = path.constants.find(value);
    if (fixed != path.constants.end())
      return fixed->second;
    llvm::APInt literal;
    if (mlir::matchPattern(value, mlir::m_ConstantInt(&literal)))
      return literal;
    auto operation = value.getDefiningOp();
    if (!operation)
      return std::nullopt;
    if (llvm::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(
            operation)) {
      auto input = evaluate(operation->getOperand(0), path, depth + 1);
      if (!input)
        return std::nullopt;
      auto size = mlir::DataLayout::closest(operation).getTypeSizeInBits(
          value.getType());
      if (size.isScalable() || !size.getFixedValue())
        return std::nullopt;
      return llvm::isa<mlir::arith::IndexCastOp>(operation)
                 ? input->sextOrTrunc(size.getFixedValue())
                 : input->zextOrTrunc(size.getFixedValue());
    }
    auto infer = llvm::dyn_cast<mlir::InferIntRangeInterface>(operation);
    if (!infer || !llvm::isa<mlir::IntegerType>(value.getType()))
      return std::nullopt;
    llvm::SmallVector<mlir::IntegerValueRange> operands;
    for (mlir::Value input : operation->getOperands()) {
      if (auto constant = evaluate(input, path, depth + 1))
        operands.emplace_back(mlir::ConstantIntRanges::constant(*constant));
      else
        operands.push_back(mlir::IntegerValueRange::getMaxRange(input));
    }
    std::optional<llvm::APInt> result;
    infer.inferResultRangesFromOptional(
        operands,
        [&](mlir::Value output, const mlir::IntegerValueRange &range) {
          if (output == value && !range.isUninitialized())
            result = range.getValue().getConstantValue();
        });
    return result;
  }

  bool consistent(const Path &path) {
    for (const auto &[value, expected] : path.constants) {
      Path without = path;
      without.constants.erase(value);
      auto actual = evaluate(value, without);
      if (actual && *actual != expected)
        return false;
    }
    for (const auto &[value, excluded] : path.excluded) {
      auto actual = evaluate(value, path);
      if (actual && *actual == excluded)
        return false;
    }
    return true;
  }

  mlir::Operation *firstControl(mlir::Value value, const Path &path,
                                llvm::DenseSet<mlir::Value> &seen) {
    value = forwarded(value, path);
    if (!seen.insert(value).second || evaluate(value, path))
      return nullptr;
    auto operation = value.getDefiningOp();
    if (!operation)
      return nullptr;
    if (llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp,
                  mlir::scf::WhileOp>(operation))
      return operation;
    // A while result is projected only through its false ConditionOp edge.
    // Its before-region arguments remain unknown, preserving every possible
    // exit iteration rather than treating a sample or initializer as a proof.
    if (llvm::isa<mlir::scf::ForOp, mlir::LLVM::LoadOp, mlir::LLVM::CallOp>(
            operation))
      return nullptr;
    for (mlir::Value input : operation->getOperands())
      if (auto *control = firstControl(input, path, seen))
        return control;
    return nullptr;
  }

  bool bind(Path &path, mlir::Value value, llvm::APInt constant) {
    auto old = path.constants.find(value);
    if (old != path.constants.end() && old->second != constant)
      return false;
    path.constants[value] = constant;
    return consistent(path);
  }

  std::vector<Path> expandControl(mlir::Operation *operation,
                                  const Path &path) {
    std::vector<Path> expanded;
    auto append = [&](mlir::Region &region, Path next) {
      if (!region.hasOneBlock())
        return;
      auto yield =
          llvm::dyn_cast<mlir::scf::YieldOp>(region.front().getTerminator());
      if (!yield || yield.getResults().size() != operation->getNumResults())
        return;
      for (auto [result, source] :
           llvm::zip(operation->getResults(), yield.getResults()))
        next.aliases[result] = source;
      if (consistent(next))
        expanded.push_back(std::move(next));
    };
    if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(operation)) {
      if (!loop.getBefore().hasOneBlock())
        return expanded;
      auto condition = loop.getConditionOp();
      if (condition.getArgs().size() != loop.getNumResults())
        return expanded;
      Path next = path;
      for (auto [result, source] :
           llvm::zip(loop.getResults(), condition.getArgs()))
        next.aliases[result] = source;
      if (bind(next, condition.getCondition(), llvm::APInt(1, false)))
        expanded.push_back(std::move(next));
    } else if (auto branch = llvm::dyn_cast<mlir::scf::IfOp>(operation)) {
      for (unsigned choice = 0; choice < 2; ++choice) {
        Path next = path;
        if (bind(next, branch.getCondition(), llvm::APInt(1, choice == 0)))
          append(choice == 0 ? branch.getThenRegion() : branch.getElseRegion(),
                 std::move(next));
      }
    } else if (auto branch =
                   llvm::dyn_cast<mlir::scf::IndexSwitchOp>(operation)) {
      unsigned width = mlir::DataLayout::closest(operation)
                           .getTypeSizeInBits(branch.getArg().getType())
                           .getFixedValue();
      for (auto [ordinal, constant] : llvm::enumerate(branch.getCases())) {
        Path next = path;
        if (bind(next, branch.getArg(), llvm::APInt(width, constant)))
          append(branch.getCaseRegions()[ordinal], std::move(next));
      }
      Path next = path;
      for (int64_t constant : branch.getCases())
        next.excluded.emplace_back(branch.getArg(),
                                   llvm::APInt(width, constant));
      append(branch.getDefaultRegion(), std::move(next));
    }
    return expanded;
  }

  std::optional<std::vector<Path>>
  controlAlternatives(llvm::ArrayRef<mlir::Value> values, Path initial) {
    std::vector<Path> pending{std::move(initial)}, results;
    std::size_t expansions = 0;
    while (!pending.empty()) {
      Path path = std::move(pending.back());
      pending.pop_back();
      if (!consistent(path))
        continue;
      mlir::Operation *operation = nullptr;
      for (mlir::Value value : values) {
        llvm::DenseSet<mlir::Value> seen;
        operation = firstControl(value, path, seen);
        if (operation)
          break;
      }
      // Constraints may project a different result of an expanded tuple.
      if (!operation)
        for (const auto &[value, constant] : path.constants) {
          llvm::DenseSet<mlir::Value> seen;
          Path without = path;
          without.constants.erase(value);
          operation = firstControl(value, without, seen);
          if (operation)
            break;
        }
      if (!operation) {
        results.push_back(std::move(path));
        continue;
      }
      if (++expansions > maximumControlAlternatives) {

        return std::nullopt;
      }
      for (Path &next : expandControl(operation, path))
        pending.push_back(std::move(next));
    }
    return results;
  }

  Path enclosingPath(mlir::Operation *operation) {
    Path path;
    while (mlir::Operation *parent = operation->getParentOp()) {
      if (auto branch = llvm::dyn_cast<mlir::scf::IfOp>(parent))
        path.constants[branch.getCondition()] = llvm::APInt(
            1, operation->getParentRegion() == &branch.getThenRegion());
      if (auto branch = llvm::dyn_cast<mlir::scf::IndexSwitchOp>(parent)) {
        unsigned width = mlir::DataLayout::closest(parent)
                             .getTypeSizeInBits(branch.getArg().getType())
                             .getFixedValue();
        bool found = false;
        for (auto [ordinal, region] : llvm::enumerate(branch.getCaseRegions()))
          if (operation->getParentRegion() == &region) {
            path.constants[branch.getArg()] =
                llvm::APInt(width, branch.getCases()[ordinal]);
            found = true;
          }
        if (!found)
          for (int64_t constant : branch.getCases())
            path.excluded.emplace_back(branch.getArg(),
                                       llvm::APInt(width, constant));
      }
      operation = parent;
    }
    return path;
  }

  std::vector<GuardedValueDomain> enclosingGuardDomains(
      mlir::Operation *operation,
      llvm::function_ref<std::optional<std::set<int64_t>>(mlir::Value)> range =
          {},
      mlir::Value interested = {}) {
    Path path = enclosingPath(operation);
    llvm::SmallVector<std::pair<mlir::Value, bool>> taken;
    for (const auto &[value, constant] : path.constants)
      if (constant.getBitWidth() == 1)
        taken.emplace_back(value, !constant.isZero());
    // The context holds function references, so both callables must outlive
    // the projection call.
    const auto constantValue = [&](mlir::Value value) {
      return evaluate(value, Path());
    };
    const auto canonicalValue = [&](mlir::Value value) {
      return forwarded(value, Path());
    };
    ScalarGuardContext context{constantValue, canonicalValue, range,
                               maximumStaticValues};
    return projectGuardedValueDomains(taken, context, interested);
  }

  bool dependsOn(mlir::Value value, mlir::Value target,
                 llvm::DenseSet<mlir::Value> &seen) {
    value = forwarded(value, Path());
    target = forwarded(target, Path());
    if (value == target)
      return true;
    if (!seen.insert(value).second)
      return false;
    auto operation = value.getDefiningOp();
    if (!operation ||
        llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::CallOp, mlir::scf::WhileOp,
                  mlir::scf::ForOp>(operation))
      return false;
    for (mlir::Value input : operation->getOperands())
      if (dependsOn(input, target, seen))
        return true;
    if (llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(operation)) {
      unsigned ordinal = llvm::cast<mlir::OpResult>(value).getResultNumber();
      for (mlir::Region &region : operation->getRegions())
        if (region.hasOneBlock())
          if (auto yield = llvm::dyn_cast<mlir::scf::YieldOp>(
                  region.front().getTerminator()))
            if (ordinal < yield.getResults().size() &&
                dependsOn(yield.getResults()[ordinal], target, seen))
              return true;
    }
    return false;
  }

  std::optional<std::set<int64_t>> scalarDomainAt(mlir::Value value,
                                                  mlir::Operation *anchor) {
    if (anchor) {
      auto guarded = enclosingGuardDomains(
          anchor, [&](mlir::Value bound) { return scalarDomain(bound); },
          value);
      for (const GuardedValueDomain &domain : guarded)
        if (domain.value == forwarded(value, Path()))
          return domain.values;
    }
    return scalarDomain(value);
  }

  std::optional<std::set<int64_t>> scalarDomain(mlir::Value value) {
    if (auto found = scalarDomains_.find(value); found != scalarDomains_.end())
      return found->second;
    if (!activeScalarDomains_.insert(value).second)
      return std::nullopt;
    auto result = deriveScalarDomain(value);
    activeScalarDomains_.erase(value);
    if (result)
      scalarDomains_[value] = result;
    return result;
  }

  /// Value domain of one boundary argument. A selected region and an owned
  /// invocation denote their entry arguments by explicit actuals, and the
  /// address projection crosses both, so an integer index crosses the same
  /// boundaries and a materialized scope keeps the domains its source proved.
  std::optional<std::set<int64_t>>
  boundaryScalarDomain(mlir::BlockArgument argument) {
    mlir::Operation *owner = argument.getOwner()->getParentOp();
    if (auto spatial = llvm::dyn_cast_or_null<::loom::SpatialRegionOp>(owner)) {
      if (argument.getArgNumber() >= spatial->getNumOperands())
        return std::nullopt;
      return scalarDomain(spatial->getOperand(argument.getArgNumber()));
    }
    std::optional<std::set<int64_t>> crossed;
    for (const auto &frame : frames_) {
      if (!frame->caller || frame->entry() != argument.getOwner())
        continue;
      if (argument.getArgNumber() >= frame->actuals().size())
        return std::nullopt;
      auto incoming = scalarDomain(frame->actuals()[argument.getArgNumber()]);
      if (!incoming)
        return std::nullopt;
      if (!crossed)
        crossed.emplace();
      crossed->insert(incoming->begin(), incoming->end());
      if (crossed->size() > maximumStaticValues)
        return std::nullopt;
    }
    return crossed;
  }

  std::optional<std::set<int64_t>> deriveScalarDomain(mlir::Value value) {
    auto scalar = evaluate(value, Path());
    if (scalar && scalar->isSignedIntN(64))
      return std::set<int64_t>{scalar->getSExtValue()};
    if (auto definition = value.getDefiningOp()) {
      auto guarded = enclosingGuardDomains(
          definition, [&](mlir::Value bound) { return scalarDomain(bound); },
          value);
      for (const GuardedValueDomain &domain : guarded)
        if (domain.value == forwarded(value, Path()))
          return domain.values;
    }
    mlir::Value original = value;
    value = forwarded(value, Path());
    if (original != value)
      return scalarDomain(value);
    if (auto read = value.getDefiningOp<mlir::LLVM::LoadOp>())
      return loadedScalarValues(read);
    if (auto induction = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      if (auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
              induction.getOwner()->getParentOp())) {
        if (induction != loop.getInductionVar())
          return std::nullopt;
        auto start = evaluate(loop.getLowerBound(), Path()),
             end = evaluate(loop.getUpperBound(), Path()),
             step = evaluate(loop.getStep(), Path());
        if (!start || !end || !step || !start->isSignedIntN(64) ||
            !end->isSignedIntN(64) || !step->isSignedIntN(64) ||
            step->getSExtValue() <= 0)
          return std::nullopt;
        std::set<int64_t> result;
        int64_t current = start->getSExtValue();
        while (current < end->getSExtValue()) {
          if (result.size() == maximumStaticValues)
            return std::nullopt;
          result.insert(current);
          if (llvm::AddOverflow(current, step->getSExtValue(), current))
            return std::nullopt;
        }
        return result;
      }
      if (auto loop = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(
              induction.getOwner()->getParentOp()))
        return whileDomain(loop, induction);
      return boundaryScalarDomain(induction);
    }
    auto operation = value.getDefiningOp();
    if (!operation || operation->getNumRegions() ||
        operation->getNumOperands() == 0)
      return std::nullopt;
    if (!llvm::isa<mlir::arith::ExtUIOp, mlir::arith::ExtSIOp,
                   mlir::arith::TruncIOp, mlir::arith::IndexCastOp,
                   mlir::arith::IndexCastUIOp, mlir::arith::AddIOp,
                   mlir::arith::SubIOp, mlir::arith::MulIOp,
                   mlir::arith::ShLIOp>(operation))
      return std::nullopt;
    std::vector<Path> paths(1);
    for (mlir::Value input : operation->getOperands()) {
      auto domain = scalarDomain(input);
      if (!domain || domain->empty() ||
          paths.size() * domain->size() > maximumStaticValues)
        return std::nullopt;
      unsigned width =
          llvm::isa<mlir::IndexType>(input.getType())
              ? mlir::DataLayout::closest(operation)
                    .getTypeSizeInBits(input.getType())
                    .getFixedValue()
              : llvm::cast<mlir::IntegerType>(input.getType()).getWidth();
      std::vector<Path> expanded;
      for (const Path &path : paths)
        for (int64_t integer : *domain) {
          Path next = path;
          next.constants[input] = llvm::APInt(width, integer);
          expanded.push_back(std::move(next));
        }
      paths = std::move(expanded);
    }
    std::set<int64_t> result;
    for (const Path &path : paths) {
      auto scalar = evaluate(value, path);
      if (!scalar || !scalar->isSignedIntN(64))
        return std::nullopt;
      result.insert(scalar->getSExtValue());
    }
    return result;
  }

  std::optional<std::set<int64_t>> whileDomain(mlir::scf::WhileOp loop,
                                               mlir::BlockArgument induction) {
    if (induction.getOwner() != loop.getBeforeBody() ||
        !loop.getAfter().hasOneBlock() ||
        !loop.getAfterBody()->without_terminator().empty() ||
        loop.getInits().size() != loop.getBeforeBody()->getNumArguments() ||
        loop.getYieldOp().getResults().size() !=
            loop.getAfterBody()->getNumArguments())
      return std::nullopt;
    for (auto [argument, yielded] :
         llvm::zip(loop.getAfterBody()->getArguments(),
                   loop.getYieldOp().getResults()))
      if (argument != yielded)
        return std::nullopt;
    unsigned lane = induction.getArgNumber();
    auto initial = evaluate(loop.getInits()[lane], Path());
    auto integer = llvm::dyn_cast<mlir::IntegerType>(induction.getType());
    if (!initial || !integer || !initial->isSignedIntN(64))
      return std::nullopt;
    auto condition = loop.getConditionOp();
    if (lane >= condition.getArgs().size())
      return std::nullopt;
    std::vector<Path> invariantPaths(1);
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<mlir::Value> pending{condition.getCondition(),
                                           condition.getArgs()[lane]};
    llvm::SmallVector<GuardedValueDomain> memoryInvariants;
    while (!pending.empty()) {
      mlir::Value value = forwarded(pending.pop_back_val(), Path());
      if (!visited.insert(value).second || evaluate(value, Path()))
        continue;
      auto definition = value.getDefiningOp();
      if (!definition)
        continue;
      if (llvm::isa<mlir::LLVM::LoadOp>(definition) &&
          !loop->isProperAncestor(definition)) {
        auto finite = scalarDomain(value);
        if (finite)
          memoryInvariants.push_back({value, *finite});
        continue;
      }
      if (llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::LoadOp>(definition))
        continue;
      pending.append(definition->getOperands().begin(),
                     definition->getOperands().end());
      if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
        unsigned ordinal = result.getResultNumber();
        if (auto nested = llvm::dyn_cast<mlir::scf::WhileOp>(definition)) {
          auto edge = nested.getConditionOp();
          pending.push_back(edge.getCondition());
          if (ordinal < edge.getArgs().size())
            pending.push_back(edge.getArgs()[ordinal]);
        } else if (llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(
                       definition)) {
          for (mlir::Region &region : definition->getRegions())
            if (region.hasOneBlock())
              if (auto yielded = llvm::dyn_cast<mlir::scf::YieldOp>(
                      region.front().getTerminator()))
                if (ordinal < yielded.getResults().size())
                  pending.push_back(yielded.getResults()[ordinal]);
        }
      }
    }
    Path guardedPath = enclosingPath(loop);
    for (const GuardedValueDomain &domain : memoryInvariants) {
      if (domain.values.empty())
        return std::set<int64_t>{};
      if (invariantPaths.size() * domain.values.size() > maximumStaticValues)
        return std::nullopt;
      std::vector<Path> expanded;
      for (const Path &path : invariantPaths)
        for (int64_t item : domain.values) {
          Path next = path;
          next.constants[domain.value] =
              llvm::APInt(domain.value.getType().getIntOrFloatBitWidth(), item);
          for (const auto &[guard, constant] : guardedPath.constants) {
            llvm::DenseSet<mlir::Value> dependencies;
            if (dependsOn(guard, domain.value, dependencies))
              next.constants[guard] = constant;
          }
          if (consistent(next))
            expanded.push_back(std::move(next));
        }
      invariantPaths = std::move(expanded);
    }
    for (const GuardedValueDomain &domain : enclosingGuardDomains(loop)) {
      llvm::DenseSet<mlir::Value> dependencies;
      if (!dependsOn(condition.getCondition(), domain.value, dependencies) &&
          !dependsOn(condition.getArgs()[lane], domain.value, dependencies))
        continue;
      if (domain.values.empty())
        return std::set<int64_t>{};
      if (invariantPaths.size() * domain.values.size() > maximumStaticValues)
        return std::nullopt;
      auto type = llvm::dyn_cast<mlir::IntegerType>(domain.value.getType());
      if (!type)
        continue;
      std::vector<Path> expanded;
      for (const Path &path : invariantPaths)
        for (int64_t value : domain.values) {
          Path next = path;
          next.constants[domain.value] = llvm::APInt(type.getWidth(), value);
          expanded.push_back(std::move(next));
        }
      invariantPaths = std::move(expanded);
    }
    std::set<int64_t> allValues;
    for (const Path &invariants : invariantPaths) {
      std::set<int64_t> seen;
      std::vector<int64_t> pending{initial->getSExtValue()};
      while (!pending.empty()) {
        int64_t current = pending.back();
        pending.pop_back();
        if (seen.count(current))
          continue;
        if (seen.size() == maximumStaticValues) {
          return std::nullopt;
        }
        seen.insert(current);
        Path path = invariants;
        path.constants[induction] = llvm::APInt(integer.getWidth(), current);
        auto alternatives = controlAlternatives(
            {condition.getCondition(), condition.getArgs()[lane]},
            std::move(path));
        if (!alternatives)
          return std::nullopt;
        for (const Path &alternative : *alternatives) {
          auto proceeds = evaluate(condition.getCondition(), alternative);
          if (proceeds && proceeds->isZero())
            continue;
          auto next = evaluate(condition.getArgs()[lane], alternative);
          if (!next || !next->isSignedIntN(64)) {
            return std::nullopt;
          }
          // Cyclic feedback can still have a finite value domain. This proves
          // the value set only, and makes no claim of loop termination.
          pending.push_back(next->getSExtValue());
        }
      }
      allValues.insert(seen.begin(), seen.end());
      if (allValues.size() > maximumStaticValues)
        return std::nullopt;
    }
    return allValues;
  }
  mlir::Value invariantRegionPointer(mlir::Value pointer) {
    mlir::Operation *owner = pointer.getDefiningOp();
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer))
      owner = argument.getOwner()->getParentOp();
    if (!llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(owner))
      return pointer;
    llvm::SmallVector<mlir::Value> pending{pointer};
    llvm::DenseSet<mlir::Value> seen;
    mlir::Value seed;
    while (!pending.empty()) {
      mlir::Value value = pending.pop_back_val();
      if (!seen.insert(value).second)
        continue;
      if (seen.size() > maximumStaticValues)
        return {};
      auto local = resolveLinearPointerAddress(
          value, mlir::IntegerType::get(value.getContext(), 8));
      if (!local)
        return {};
      Address location{local->root, nullptr, local->byteBias,
                       local->accessByteCount, local->terms};
      auto offset = constantByteOffset(location);
      if (offset && *offset == 0 && local->root != value) {
        pending.push_back(local->root);
        continue;
      }

      mlir::RegionBranchOpInterface branch;
      std::optional<mlir::RegionSuccessor> successor;
      if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
        branch = llvm::dyn_cast<mlir::RegionBranchOpInterface>(result.getOwner());
        if (branch)
          successor.emplace(result.getOwner());
      } else if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(value)) {
        branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(
            argument.getOwner()->getParentOp());
        if (branch)
          successor.emplace(argument.getOwner()->getParent());
      }
      if (!branch) {
        if (seed && seed != value)
          return {};
        seed = value;
        continue;
      }
      mlir::ValueRange inputs = branch.getSuccessorInputs(*successor);
      auto input = llvm::find(inputs, value);
      if (input == inputs.end())
        return {};
      llvm::SmallVector<mlir::RegionBranchPoint> predecessors;
      llvm::SmallVector<mlir::Value> incoming;
      branch.getPredecessors(*successor, predecessors);
      branch.getPredecessorValues(
          *successor, static_cast<unsigned>(std::distance(inputs.begin(), input)),
          incoming);
      if (incoming.empty() || incoming.size() != predecessors.size())
        return {};
      pending.append(incoming.begin(), incoming.end());
    }
    // Every cycle contributes only identity forwarding, and every entry edge
    // contributes this one seed. This proves equality for every iteration.
    return seed;
  }

  std::optional<Address> address(mlir::Value pointer, mlir::Type type,
                                 Frame &frame, unsigned depth = 0) {
    if (depth > maximumProjectionDepth)
      return refuse(StoredPointerRefusal::UnknownByteAddress);
    auto local =
        loom::frontend::analysis::resolveLinearPointerAddress(pointer, type);
    if (!local)
      return refuse(StoredPointerRefusal::UnknownByteAddress);
    Address result{local->root, &frame, local->byteBias, local->accessByteCount,
                   local->terms};
    result.inBoundsOfRoot = inBoundsIndexing(*local);
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(local->root);
    mlir::Value actual;
    Frame *parent = &frame;
    if (auto invariant = invariantRegionPointer(local->root);
        invariant && invariant != local->root)
      actual = invariant;
    if (argument) {
      if (auto spatial = llvm::dyn_cast<::loom::SpatialRegionOp>(
              argument.getOwner()->getParentOp())) {
        if (argument.getArgNumber() >= spatial->getNumOperands())
          return refuse(StoredPointerRefusal::UnknownByteAddress);
        actual = spatial->getOperand(argument.getArgNumber());
      } else if (frame.caller && argument.getOwner() == frame.entry()) {
        if (argument.getArgNumber() >= frame.actuals().size())
          return refuse(StoredPointerRefusal::UnknownByteAddress);
        actual = frame.actuals()[argument.getArgNumber()];
        parent = frame.caller;
      }
    }
    if (actual) {
      auto outer = address(actual, type, *parent, depth + 1);
      if (!outer || llvm::AddOverflow(outer->bias, result.bias, result.bias))
        return refuse(StoredPointerRefusal::UnknownByteAddress);
      result.root = outer->root;
      result.rootFrame = outer->rootFrame;
      result.terms.append(outer->terms.begin(), outer->terms.end());
      result.inBoundsOfRoot &= outer->inBoundsOfRoot;
    }
    return result;
  }

  /// Every index step of this chain stays inside the object its base points
  /// into. A defined access through the chain therefore stays inside the root
  /// object, which bounds a runtime index that has no bound of its own.
  static bool
  inBoundsIndexing(const loom::frontend::analysis::ResolvedLinearMemoryAddress
                       &address) {
    for (mlir::Operation *step : address.gepsLeafToRoot)
      if (!mlir::LLVM::bitEnumContainsAny(
              llvm::cast<mlir::LLVM::GEPOp>(step).getNoWrapFlags(),
              mlir::LLVM::GEPNoWrapFlags::inboundsFlag))
        return false;
    return true;
  }

  std::optional<int64_t> constantByteOffset(const Address &location) {
    int64_t offset = location.bias;
    for (const auto &term : location.terms) {
      auto index = evaluate(term.index, Path());
      if (!index || !index->isSignedIntN(64))
        return std::nullopt;
      int64_t delta;
      if (llvm::MulOverflow(index->getSExtValue(), term.byteStride, delta) ||
          llvm::AddOverflow(offset, delta, offset))
        return std::nullopt;
    }
    return offset;
  }

  bool sameFrameRoot(const Address &lhs, const Address &rhs) {
    return lhs.root == rhs.root && lhs.rootFrame == rhs.rootFrame;
  }

  bool distinctFrameRoots(const Address &lhs, const Address &rhs) {
    if (sameFrameRoot(lhs, rhs))
      return false;
    if (loom::frontend::analysis::haveProvenDistinctMemoryRoots(lhs.root,
                                                                rhs.root))
      return true;
    auto freshAgainstInput = [](const Address &allocation,
                                const Address &input) {
      if (!allocation.root.getDefiningOp<mlir::LLVM::AllocaOp>())
        return false;
      auto argument = llvm::dyn_cast<mlir::BlockArgument>(input.root);
      if (!argument || argument.getOwner() != input.rootFrame->entry())
        return false;
      for (Frame *parent = allocation.rootFrame->caller; parent;
           parent = parent->caller)
        if (parent == input.rootFrame)
          return true;
      return false;
    };
    return freshAgainstInput(lhs, rhs) || freshAgainstInput(rhs, lhs);
  }

  bool preservesLocation(mlir::Operation *operation, Frame &frame,
                         const Address &read, bool finiteWrites = false) {
    if (auto write = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation)) {
      auto target = address(write.getAddr(), write.getValue().getType(), frame);
      if (!target)
        return false;
      if (distinctFrameRoots(read, *target))
        return true;
      if (!sameFrameRoot(read, *target))
        return false;
      auto readOffset = constantByteOffset(read),
           writeOffset = constantByteOffset(*target);
      if (!readOffset)
        return false;
      int64_t readEnd;
      if (llvm::AddOverflow(*readOffset, int64_t(read.bytes), readEnd))
        return false;
      llvm::SmallVector<int64_t> writes;
      if (writeOffset) {
        writes.push_back(*writeOffset);
      } else if (finiteWrites) {
        auto domain = addressDomain(*target);
        if (!domain)
          return false;
        // A write that no defined execution performs in bounds preserves the
        // location, and so does one whose whole domain lies beside the read.
        if (domain->offsets.isEmpty())
          return true;
        int64_t domainEnd;
        if (llvm::AddOverflow(domain->offsets.highest(),
                              int64_t(target->bytes), domainEnd))
          return false;
        if (readEnd <= domain->offsets.lowest() || domainEnd <= *readOffset)
          return true;
        auto offsets = domain->offsets.enumerate(maximumStaticValues);
        if (!offsets)
          return false;
        writes = std::move(*offsets);
      } else {
        return false;
      }
      for (int64_t offset : writes) {
        int64_t writeEnd;
        if (llvm::AddOverflow(offset, int64_t(target->bytes), writeEnd) ||
            (readEnd > offset && writeEnd > *readOffset))
          return false;
      }
      return true;
    }
    if (llvm::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp>(
            operation)) {
      auto target =
          address(operation->getOperand(0),
                  mlir::IntegerType::get(operation->getContext(), 8), frame);
      return target && loom::frontend::analysis::haveProvenDistinctMemoryRoots(
                           read.root, target->root);
    }
    if (llvm::isa<mlir::LLVM::CallOp, dataflow::ThreadLaunchOp>(operation)) {
      Frame *callee = nullptr;
      for (const auto &candidate : frames_)
        if (candidate->caller == &frame && candidate->start == operation) {
          callee = candidate.get();
          break;
        }
      if (!callee || !callee->body().hasOneBlock())
        return false;
      for (mlir::Operation &nested : *callee->entry())
        if (!preservesLocation(&nested, *callee, read, finiteWrites))
          return false;
      return true;
    }
    if (auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(operation)) {
      auto launch =
          llvm::dyn_cast_or_null<dataflow::ThreadLaunchOp>(wait->getPrevNode());
      return launch && completedLaunch(launch);
    }
    if (llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::AllocaOp>(operation) ||
        mlir::isMemoryEffectFree(operation))
      return true;
    if (!operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
      return false;
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::Operation &nested : block)
          if (!preservesLocation(&nested, frame, read, finiteWrites))
            return false;
    return true;
  }

  bool noInterveningWrite(mlir::Operation *before, mlir::Operation *after,
                          Frame &frame, const Address &read,
                          bool finiteWrites = false) {
    mlir::Operation *current = after;
    while (current) {
      for (mlir::Operation *previous = current->getPrevNode(); previous;
           previous = previous->getPrevNode()) {
        if (previous == before)
          return true;
        if (!preservesLocation(previous, frame, read, finiteWrites))
          return false;
      }
      mlir::Operation *parent = current->getParentOp();
      if (!parent || parent == frame.callable)
        return before == nullptr;
      // Crossing a repeated region conservatively includes every iteration's
      // effects. A read is reused only when even this larger effect set is
      // safe.
      if (llvm::isa<mlir::scf::WhileOp, mlir::scf::ForOp>(parent)) {
        if (!preservesLocation(parent, frame, read, finiteWrites))
          return false;
      } else if (!llvm::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp,
                            ::loom::SpatialRegionOp>(parent)) {
        return false;
      }
      current = parent;
    }
    return false;
  }

  void deriveReloadEquivalences(bool finiteWrites = false) {
    llvm::DenseSet<mlir::Operation *> done;
    for (const auto &representative : frames_) {
      auto callable = representative->callable;
      if (!done.insert(callable).second)
        continue;
      std::vector<mlir::LLVM::LoadOp> reads;
      std::vector<mlir::LLVM::StoreOp> writes;
      callable->walk([&](mlir::LLVM::LoadOp read) {
        if (llvm::isa<mlir::IntegerType, mlir::LLVM::LLVMPointerType>(
                read.getResult().getType()) &&
            !read.getVolatile_() &&
            read.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic)
          reads.push_back(read);
      });
      callable->walk([&](mlir::LLVM::StoreOp write) {
        if (llvm::isa<mlir::IntegerType, mlir::LLVM::LLVMPointerType>(
                write.getValue().getType()) &&
            !write.getVolatile_() &&
            write.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic)
          writes.push_back(write);
      });
      mlir::DominanceInfo dominance(callable);
      for (mlir::LLVM::LoadOp after : reads) {
        bool stored = false;
        for (mlir::LLVM::StoreOp before : writes) {
          if (before.getValue().getType() != after.getResult().getType() ||
              !dominance.properlyDominates(before.getOperation(),
                                           after.getOperation()))
            continue;
          bool proved = true;
          for (const auto &frame : frames_) {
            if (frame->callable != callable)
              continue;
            auto lhs = address(before.getAddr(), before.getValue().getType(),
                               *frame),
                 rhs = address(after.getAddr(), after.getResult().getType(),
                               *frame);
            auto lhsOffset = lhs ? constantByteOffset(*lhs) : std::nullopt;
            auto rhsOffset = rhs ? constantByteOffset(*rhs) : std::nullopt;
            if (!lhs || !rhs || !sameFrameRoot(*lhs, *rhs) || !lhsOffset ||
                !rhsOffset || *lhsOffset != *rhsOffset ||
                lhs->bytes != rhs->bytes ||
                (lhs->root.getDefiningOp<mlir::LLVM::AllocaOp>() &&
                 !withinAllocation(*lhs, *lhsOffset)) ||
                !noInterveningWrite(before, after, *frame, *lhs,
                                    finiteWrites)) {
              proved = false;
              break;
            }
          }
          if (!proved)
            continue;
          reloadEquivalences_[after.getResult()] = before.getValue();

          stored = true;
          break;
        }
        if (stored)
          continue;
        for (mlir::LLVM::LoadOp before : reads) {
          if (before == after ||
              before.getResult().getType() != after.getResult().getType() ||
              !dominance.properlyDominates(before.getOperation(),
                                           after.getOperation()))
            continue;
          bool proved = true;
          for (const auto &frame : frames_) {
            if (frame->callable != callable)
              continue;
            auto lhs = address(before.getAddr(), before.getResult().getType(),
                               *frame),
                 rhs = address(after.getAddr(), after.getResult().getType(),
                               *frame);
            auto lhsOffset = lhs ? constantByteOffset(*lhs) : std::nullopt;
            auto rhsOffset = rhs ? constantByteOffset(*rhs) : std::nullopt;
            if (!lhs || !rhs || !sameFrameRoot(*lhs, *rhs) || !lhsOffset ||
                !rhsOffset || *lhsOffset != *rhsOffset ||
                lhs->bytes != rhs->bytes ||
                (lhs->root.getDefiningOp<mlir::LLVM::AllocaOp>() &&
                 !withinAllocation(*lhs, *lhsOffset)) ||
                !noInterveningWrite(before, after, *frame, *lhs,
                                    finiteWrites)) {
              proved = false;
              break;
            }
          }
          if (!proved)
            continue;
          reloadEquivalences_[after.getResult()] = before.getResult();

          break;
        }
      }
    }
  }

  std::optional<AddressByteDomain>
  addressDomain(const Address &location,
                llvm::ArrayRef<mlir::Value> completedIndices = {}) {
    AddressByteDomainRequest request;
    request.root = location.root;
    request.terms = location.terms;
    request.byteBias = location.bias;
    request.accessByteCount = location.bytes;
    request.allocationByteCount = allocationExtent(location.root);
    request.inBoundsOfAllocation = location.inBoundsOfRoot;
    request.completedIndices = completedIndices;
    return projectAddressByteDomain(
        request, [&](mlir::Value index) { return scalarDomain(index); });
  }

  enum class WriteKind { Scalar, Zero, Copy };

  struct WriteEffect final {
    mlir::LLVM::StoreOp operation;
    Frame *frame;
    mlir::Operation *completion;
    Address destination;
    WriteKind kind = WriteKind::Scalar;
    std::optional<Address> source = std::nullopt;
    /// Induction variables of the statically counted loops that this write
    /// completes, whose every value it therefore writes.
    llvm::SmallVector<mlir::Value, 2> completedIndices;
  };

  /// Enumerated write positions of one effect, memoized because a query joins
  /// every effect at every queried byte offset.
  struct WritePositions final {
    llvm::SmallVector<int64_t> starts;
    bool exhaustive = false;
    int64_t lowest = 0;
    int64_t highestEnd = 0;
    bool projected = false;
    bool computed = false;
  };

  const WritePositions *writePositions(std::size_t ordinal,
                                       const WriteEffect &effect) {
    if (!writePositions_[ordinal].computed) {
      WritePositions positions;
      auto domain = addressDomain(effect.destination, effect.completedIndices);
      auto offsets = domain ? domain->offsets.enumerate(maximumStaticValues)
                            : std::nullopt;
      if (offsets) {
        positions.starts = std::move(*offsets);
        positions.exhaustive = domain->exhaustive;
        positions.projected = true;
        if (!positions.starts.empty()) {
          positions.lowest = positions.starts.front();
          if (llvm::AddOverflow(positions.starts.back(),
                                int64_t(effect.destination.bytes),
                                positions.highestEnd))
            positions.projected = false;
        }
      }
      positions.computed = true;
      writePositions_[ordinal] = std::move(positions);
    }
    const WritePositions &cached = writePositions_[ordinal];
    return cached.projected ? &cached : nullptr;
  }

  std::optional<WriteEffect> writeEffect(mlir::LLVM::StoreOp write,
                                         Frame &frame) {
    auto destination =
        address(write.getAddr(), write.getValue().getType(), frame);
    if (!destination || write.getVolatile_() ||
        write.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
      return std::nullopt;
    WriteEffect effect{write, &frame, write, *destination};
    auto scalar = evaluate(write.getValue(), Path());
    if (scalar && scalar->isZero())
      effect.kind = WriteKind::Zero;
    auto read = write.getValue().getDefiningOp<mlir::LLVM::LoadOp>();
    mlir::Operation *loop = write->getParentOp();
    auto exact = projectUnitStrideCountedLoop(
        loop, [&](mlir::Value value) { return evaluate(value, Path()); });
    // A constant fill tiles the range at its own access width; a transfer
    // remains byte granular because a wider element copy would additionally
    // have to prove element alignment of both views.
    if (exact &&
        (effect.kind == WriteKind::Zero ||
         (destination->bytes == 1 && read && read->getParentOp() == loop &&
          read->isBeforeInBlock(write) && !read.getVolatile_() &&
          read.getOrdering() == mlir::LLVM::AtomicOrdering::not_atomic))) {
      bool sole = true;
      loop->walk([&](mlir::Operation *operation) {
        if (operation != write && operation != read && operation != loop &&
            !mlir::isMemoryEffectFree(operation) &&
            !operation->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
          sole = false;
      });
      llvm::SmallVector<loom::frontend::analysis::LinearByteTerm, 4> tiled =
          destination->terms;
      int64_t filled = 0;
      if (sole &&
          removeContiguousIndex(tiled, exact->induction, destination->bytes) &&
          !llvm::MulOverflow(int64_t(exact->iterationCount),
                             int64_t(destination->bytes), filled)) {
        if (read) {
          auto from =
              address(read.getAddr(), read.getResult().getType(), frame);
          if (!from ||
              !removeContiguousIndex(from->terms, exact->induction,
                                     from->bytes) ||
              !distinctFrameRoots(*from, *destination))
            return std::nullopt;
          effect.source = *from;
          effect.kind = WriteKind::Copy;
        }
        effect.destination.terms = std::move(tiled);
        effect.destination.bytes = uint64_t(filled);
        effect.completion = loop;
      }
    }
    // A statically non-empty for executes each body effect at every enumerated
    // induction value. Its completion therefore owns initialization of the
    // union of those byte locations, including nested exact byte-fill loops.
    while (auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
               effect.completion->getParentOp())) {
      auto lower = evaluate(loop.getLowerBound(), Path()),
           upper = evaluate(loop.getUpperBound(), Path()),
           step = evaluate(loop.getStep(), Path());
      if (!lower || !upper || !step || !step->isStrictlyPositive() ||
          !lower->slt(*upper))
        break;
      effect.completion = loop;
    }
    // Every loop between the write and its completion is a statically
    // non-empty counted loop, so one completed execution of the completion
    // performs this write at every value of those induction variables.
    for (mlir::Operation *scope = write.getOperation(); scope;
         scope = scope->getParentOp()) {
      if (auto counted = llvm::dyn_cast<mlir::scf::ForOp>(scope))
        effect.completedIndices.push_back(counted.getInductionVar());
      if (scope == effect.completion)
        break;
    }
    return effect;
  }

  bool invocationOrdered(mlir::Operation *before, Frame *beforeFrame,
                         mlir::Operation *after, Frame *afterFrame,
                         bool requireDominance = true) {
    std::vector<Frame *> beforePath, afterPath;
    for (Frame *frame = beforeFrame; frame; frame = frame->caller)
      beforePath.push_back(frame);
    for (Frame *frame = afterFrame; frame; frame = frame->caller)
      afterPath.push_back(frame);
    while (beforePath.size() > afterPath.size()) {
      before = beforeFrame->completion;
      beforeFrame = beforeFrame->caller;
      beforePath.pop_back();
    }
    while (afterPath.size() > beforePath.size()) {
      after = afterFrame->start;
      afterFrame = afterFrame->caller;
      afterPath.pop_back();
    }
    while (beforeFrame != afterFrame) {
      if (!beforeFrame || !afterFrame)
        return false;
      before = beforeFrame->completion;
      after = afterFrame->start;
      beforeFrame = beforeFrame->caller;
      afterFrame = afterFrame->caller;
    }
    if (!beforeFrame || before == after)
      return false;
    if (!requireDominance) {
      llvm::SmallVector<mlir::Operation *> beforeAncestors, afterAncestors;
      for (mlir::Operation *ancestor = before; ancestor;
           ancestor = ancestor->getParentOp())
        beforeAncestors.push_back(ancestor);
      for (mlir::Operation *ancestor = after; ancestor;
           ancestor = ancestor->getParentOp())
        afterAncestors.push_back(ancestor);
      while (!beforeAncestors.empty() && !afterAncestors.empty() &&
             beforeAncestors.back() == afterAncestors.back()) {
        beforeAncestors.pop_back();
        afterAncestors.pop_back();
      }
      if (beforeAncestors.empty() || afterAncestors.empty())
        return false;
      before = beforeAncestors.back();
      after = afterAncestors.back();
      return before->getBlock() == after->getBlock() &&
             before->isBeforeInBlock(after);
    }
    mlir::DominanceInfo dominance(beforeFrame->callable);
    return dominance.properlyDominates(before, after);
  }

  std::optional<std::set<int64_t>> loadedScalarValues(mlir::LLVM::LoadOp read) {
    if (read.getVolatile_() ||
        read.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic ||
        !llvm::isa<mlir::IntegerType>(read.getResult().getType()))
      return std::nullopt;
    std::set<int64_t> all;
    bool visited = false;
    auto callable = enclosingCallable(read.getResult());
    for (const auto &frame : frames_) {
      if (frame->callable != callable)
        continue;
      visited = true;
      auto location =
          address(read.getAddr(), read.getResult().getType(), *frame);
      auto local = location ? scalar(*location, read, *frame) : std::nullopt;
      if (!local)
        return std::nullopt;
      all.insert(local->begin(), local->end());
      if (all.size() > maximumStaticValues)
        return std::nullopt;
    }
    if (!visited)
      return std::nullopt;

    return all;
  }

  enum class PayloadKind { Zero, Value };
  struct Payload final {
    PayloadKind kind;
    mlir::Value value;
    Frame *frame;
    mlir::Operation *point;
    bool operator==(const Payload &other) const {
      return kind == other.kind && value == other.value &&
             frame == other.frame && point == other.point;
    }
  };

  static void appendPayloads(std::vector<Payload> &destination,
                             const std::vector<Payload> &source) {
    for (const Payload &payload : source)
      if (!llvm::is_contained(destination, payload))
        destination.push_back(payload);
  }

  bool initialized(const WriteEffect &effect, const Address &query,
                   mlir::Operation *point, Frame &frame) {
    // Descendant effects may be conditional inside a call. They contribute
    // possible values, but cannot be a definite initialization of its caller.
    bool ancestor = false;
    for (Frame *current = &frame; current; current = current->caller)
      ancestor |= current == effect.frame;
    if (!ancestor ||
        !invocationOrdered(effect.completion, effect.frame, point, &frame))
      return false;
    bool live = true;
    for (const auto &candidate : frames_)
      candidate->callable->walk([&](mlir::Operation *operation) {
        const bool start = llvm::isa<mlir::LLVM::LifetimeStartOp>(operation);
        if (!start && !llvm::isa<mlir::LLVM::LifetimeEndOp>(operation))
          return;
        auto location = address(
            operation->getOperand(0),
            mlir::IntegerType::get(operation->getContext(), 8), *candidate);
        if (!location) {
          live = false;
          return;
        }
        if (distinctFrameRoots(query, *location))
          return;
        if (!sameFrameRoot(query, *location)) {
          live = false;
          return;
        }
        if (start)
          live &= invocationOrdered(operation, candidate.get(),
                                    effect.completion, effect.frame);
        else
          live &= invocationOrdered(point, &frame, operation, candidate.get(),
                                    false);
      });
    return live;
  }

  std::optional<uint64_t> allocationExtent(mlir::Value root) {
    auto [entry, inserted] = allocationExtents_.try_emplace(root, std::nullopt);
    if (inserted)
      if (auto allocation = root.getDefiningOp<mlir::LLVM::AllocaOp>())
        entry->second = projectFixedAllocationByteCount(allocation);
    return entry->second;
  }

  bool withinAllocation(const Address &location, int64_t offset) {
    auto extent = allocationExtent(location.root);
    return extent && offset >= 0 && uint64_t(offset) <= *extent &&
           location.bytes <= *extent - uint64_t(offset);
  }

  std::optional<std::vector<Payload>>
  sources(const Address &query, mlir::Operation *point, Frame &frame) {
    // General target joins require the complete write domain. A fixed-view
    // read can instead use the stronger dominating-store interval in reaching;
    // an opaque operation after that read cannot invalidate the SSA equality.
    if (openWriteDomain_)
      return refuse(*openWriteDomain_);
    auto extent = allocationExtent(query.root);
    if (!extent)
      return refuse(StoredPointerRefusal::UnknownByteAddress);
    auto domain = addressDomain(query);
    auto offsets = domain ? domain->offsets.enumerate(maximumStaticValues)
                          : std::nullopt;
    if (!offsets)
      return refuse(StoredPointerRefusal::UnknownIntegerDomain);
    if (offsets->empty())
      return refuse(StoredPointerRefusal::OutOfBoundsAccess);
    std::vector<Payload> result;
    for (int64_t offset : *offsets) {
      if (!withinAllocation(query, offset))
        return refuse(StoredPointerRefusal::OutOfBoundsAccess);
      using Query = std::tuple<mlir::Value, Frame *, int64_t, uint64_t,
                               mlir::Operation *, Frame *>;
      Query key{query.root,  query.rootFrame, offset,
                query.bytes, point,           &frame};
      auto cached = memorySources_.find(key);
      if (cached != memorySources_.end()) {
        appendPayloads(result, cached->second);
        continue;
      }
      if (!activeMemorySources_.insert(key).second)
        return refuse(StoredPointerRefusal::IncompleteInitialization);
      auto local = slotSources(query, offset, point, frame);
      activeMemorySources_.erase(key);
      if (!local)
        return std::nullopt;
      memorySources_[key] = *local;
      appendPayloads(result, *local);
      if (result.size() > maximumStaticValues)
        return refuse(StoredPointerRefusal::UnknownIntegerDomain);
    }
    return result;
  }

  /// Refuses one queried byte range and records which write refused it. The
  /// record is the only evidence that separates a genuinely partial stored
  /// representation from an over-approximated write position, so it stays
  /// available behind the detail verbosity rather than being removed with the
  /// investigation that needed it.
  std::nullopt_t refuseSlot(StoredPointerRefusal reason, const Address &query,
                            int64_t offset, const WriteEffect *effect = nullptr,
                            const WritePositions *positions = nullptr,
                            int64_t writeOffset = 0) {
    ByteRangeRefusal record;
    record.reason = storedPointerRefusalSpelling(reason);
    record.root = query.root;
    record.queryByteOffset = offset;
    record.queryByteCount = query.bytes;
    if (effect) {
      mlir::LLVM::StoreOp write = effect->operation;
      record.writeValue = write.getValue();
      record.writeKind = effect->kind == WriteKind::Zero   ? "zero"
                         : effect->kind == WriteKind::Copy ? "copy"
                                                           : "scalar";
      record.writeByteCount = effect->destination.bytes;
    }
    if (positions) {
      record.writeByteOffset = writeOffset;
      record.writePositionCount = positions->starts.size();
      record.writePositionLowest = positions->lowest;
      record.writePositionEnd = positions->highestEnd;
      record.writeExhaustive = positions->exhaustive;
    }
    reportByteRangeRefusal(record);
    return refuse(reason);
  }

  /// The payload of the one write proven to be the last writer of a queried
  /// byte range: it covers the range at a position one completed execution
  /// always performs, and no effect between it and the read may touch that
  /// range. Every earlier write of the range is then dead, so a write whose
  /// position is only bounded cannot make the representation partial.
  std::optional<std::vector<Payload>> lastWritePayload(const Address &query,
                                                       int64_t offset,
                                                       mlir::Operation *point,
                                                       Frame &frame) {
    Address slot{query.root, query.rootFrame, offset, query.bytes, {},
                 query.inBoundsOfRoot};
    int64_t slotEnd;
    if (llvm::AddOverflow(offset, int64_t(slot.bytes), slotEnd))
      return std::nullopt;
    for (std::size_t ordinal = 0; ordinal != effects_.size(); ++ordinal) {
      const auto &possible = effects_[ordinal];
      if (!possible)
        return std::nullopt;
      const WriteEffect &effect = *possible;
      if (effect.kind == WriteKind::Copy ||
          !sameFrameRoot(slot, effect.destination))
        continue;
      const WritePositions *positions = writePositions(ordinal, effect);
      if (!positions || !positions->exhaustive)
        continue;
      const bool covers = llvm::any_of(positions->starts, [&](int64_t begin) {
        int64_t end = 0;
        if (llvm::AddOverflow(begin, int64_t(effect.destination.bytes), end))
          return false;
        return effect.kind == WriteKind::Zero
                   ? begin <= offset && end >= slotEnd
                   : begin == offset &&
                         effect.destination.bytes == slot.bytes;
      });
      if (!covers || !initialized(effect, slot, point, frame))
        continue;
      Frame *current = &frame;
      mlir::Operation *reached = point;
      bool preserves = true;
      while (current != effect.frame && current->caller) {
        preserves &= noInterveningWrite(nullptr, reached, *current, slot, true);
        reached = current->start;
        current = current->caller;
      }
      if (!preserves || current != effect.frame ||
          !noInterveningWrite(effect.operation, reached, *current, slot, true))
        continue;
      if (effect.kind == WriteKind::Zero)
        return std::vector<Payload>{
            {PayloadKind::Zero, {}, effect.frame, effect.operation}};
      mlir::LLVM::StoreOp lastWrite = effect.operation;
      return std::vector<Payload>{{PayloadKind::Value, lastWrite.getValue(),
                                   effect.frame, effect.operation}};
    }
    return std::nullopt;
  }

  std::optional<std::vector<Payload>> slotSources(const Address &query,
                                                  int64_t offset,
                                                  mlir::Operation *point,
                                                  Frame &frame) {
    std::vector<Payload> result;
    bool mustInitialized = false;
    int64_t queryEnd;
    if (llvm::AddOverflow(offset, int64_t(query.bytes), queryEnd))
      return refuse(StoredPointerRefusal::UnknownByteAddress);
    for (std::size_t ordinal = 0; ordinal != effects_.size(); ++ordinal) {
      const auto &possible = effects_[ordinal];
      if (!possible)
        return refuseSlot(StoredPointerRefusal::UnsupportedMemoryEffect, query,
                          offset);
      const WriteEffect &effect = *possible;
      auto write = effect.operation;
      if (distinctFrameRoots(query, effect.destination))
        continue;
      if (!sameFrameRoot(query, effect.destination))
        return refuseSlot(StoredPointerRefusal::UnknownAlias, query, offset,
                          &effect);
      const WritePositions *positions = writePositions(ordinal, effect);
      if (!positions)
        return refuseSlot(StoredPointerRefusal::UnknownIntegerDomain, query,
                          offset, &effect);
      if (positions->starts.empty())
        return refuseSlot(StoredPointerRefusal::OutOfBoundsAccess, query,
                          offset, &effect, positions);
      if (queryEnd <= positions->lowest || positions->highestEnd <= offset)
        continue;
      for (int64_t begin : positions->starts) {
        if (!withinAllocation(effect.destination, begin))
          return refuseSlot(StoredPointerRefusal::OutOfBoundsAccess, query,
                            offset, &effect, positions, begin);
        int64_t end;
        if (llvm::AddOverflow(begin, int64_t(effect.destination.bytes), end))
          return refuse(StoredPointerRefusal::UnknownByteAddress);
        if (queryEnd <= begin || end <= offset)
          continue;
        if (begin > offset || end < queryEnd) {
          if (auto last = lastWritePayload(query, offset, point, frame))
            return last;
          return refuseSlot(StoredPointerRefusal::PartialPointerWrite, query,
                            offset, &effect, positions, begin);
        }
        if (effect.kind == WriteKind::Zero) {
          result.push_back({PayloadKind::Zero, {}, effect.frame, write});
        } else if (effect.kind == WriteKind::Copy) {
          Address from = *effect.source;
          if (llvm::AddOverflow(from.bias, offset - begin, from.bias))
            return refuse(StoredPointerRefusal::UnknownByteAddress);
          from.bytes = query.bytes;
          auto incoming = sources(from, write, *effect.frame);
          if (!incoming)
            return std::nullopt;
          appendPayloads(result, *incoming);
        } else if (begin == offset && effect.destination.bytes == query.bytes) {
          result.push_back(
              {PayloadKind::Value, write.getValue(), effect.frame, write});
        } else {
          if (auto last = lastWritePayload(query, offset, point, frame))
            return last;
          return refuseSlot(StoredPointerRefusal::PartialPointerWrite, query,
                            offset, &effect, positions, begin);
        }
        // A write whose position is chosen at runtime contributes its payload
        // but covers no byte, because no execution is known to perform it at
        // this offset.
        mustInitialized |=
            positions->exhaustive && initialized(effect, query, point, frame);
      }
    }
    if (!mustInitialized)
      return refuseSlot(StoredPointerRefusal::IncompleteInitialization, query,
                        offset);
    return result;
  }

  std::optional<std::set<int64_t>>
  scalar(const Address &query, mlir::Operation *point, Frame &frame) {
    auto incoming = sources(query, point, frame);
    if (!incoming)
      return std::nullopt;
    std::set<int64_t> result;
    for (const Payload &payload : *incoming) {
      if (payload.kind == PayloadKind::Zero) {
        result.insert(0);
        continue;
      }
      if (!llvm::isa<mlir::IntegerType>(payload.value.getType()))
        return std::nullopt;
      auto finite = scalarDomainAt(payload.value, payload.point);
      if (!finite)
        return std::nullopt;
      result.insert(finite->begin(), finite->end());
      if (result.size() > maximumStaticValues)
        return std::nullopt;
    }
    return result;
  }

  std::optional<PointerRoots> roots(mlir::Value value, Frame &frame, Path path,
                                    unsigned depth = 0) {
    if (depth > maximumProjectionDepth ||
        !llvm::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
      return refuse(StoredPointerRefusal::UnsupportedPointerOrigin);
    auto pointerType = llvm::cast<mlir::LLVM::LLVMPointerType>(value.getType());
    auto layout = ::loom::resolvePointerLayout(frame.callable,
                                               pointerType.getAddressSpace());
    if (!layout) {
      llvm::consumeError(layout.takeError());
      return refuse(StoredPointerRefusal::UnsupportedPointerRepresentation);
    }
    if (layout->kind != ::loom::PointerLayoutKind::StableIntegral)
      return refuse(StoredPointerRefusal::UnsupportedPointerRepresentation);
    auto alternatives = controlAlternatives({value}, std::move(path));
    if (!alternatives)
      return refuse(StoredPointerRefusal::UnknownIntegerDomain);
    PointerRoots result;
    auto append = [&](const PointerRoots &incoming) {
      result.mayBeNull |= incoming.mayBeNull;
      for (const auto &root : incoming.roots)
        if (!llvm::is_contained(result.roots, root))
          result.roots.push_back(root);
    };
    for (const Path &alternative : *alternatives) {
      mlir::Value pointer = forwarded(value, alternative);
      if (mlir::Operation *definition = pointer.getDefiningOp();
          definition &&
          llvm::isa<mlir::arith::SelectOp, mlir::LLVM::SelectOp>(definition)) {
        mlir::Value condition = definition->getOperand(0);
        auto selected = evaluate(condition, alternative);
        for (bool truth : {false, true}) {
          if (selected && !selected->isZero() != truth)
            continue;
          Path arm = alternative;
          if (!bind(arm, condition, llvm::APInt(1, truth)))
            continue;
          auto resolved = roots(definition->getOperand(truth ? 1 : 2), frame,
                                std::move(arm), depth + 1);
          if (!resolved)
            return std::nullopt;
          append(*resolved);
        }
        continue;
      }
      if (pointer.getDefiningOp<mlir::LLVM::ZeroOp>()) {
        result.mayBeNull = true;
        continue;
      }
      if (pointer.getDefiningOp<mlir::LLVM::UndefOp>())
        return refuse(StoredPointerRefusal::UnsupportedPointerOrigin);
      if (auto read = pointer.getDefiningOp<mlir::LLVM::LoadOp>()) {
        if (read.getVolatile_() ||
            read.getOrdering() != mlir::LLVM::AtomicOrdering::not_atomic)
          return refuse(StoredPointerRefusal::UnsupportedMemoryEffect);
        auto location =
            address(read.getAddr(), read.getResult().getType(), frame);
        auto incoming =
            location ? sources(*location, read, frame) : std::nullopt;
        if (!incoming)
          return std::nullopt;
        for (const Payload &payload : *incoming) {
          if (payload.kind == PayloadKind::Zero) {
            result.mayBeNull = true;
            continue;
          }
          auto key =
              std::make_tuple(payload.value, payload.frame, payload.point);
          auto cached = pointerPayloads_.find(key);
          if (cached != pointerPayloads_.end()) {
            append(cached->second);
            continue;
          }
          auto resolved = roots(payload.value, *payload.frame,
                                enclosingPath(payload.point), depth + 1);
          if (!resolved)
            return std::nullopt;
          pointerPayloads_[key] = *resolved;
          append(*resolved);
        }
        continue;
      }
      auto location = address(
          pointer, mlir::IntegerType::get(pointer.getContext(), 8), frame);
      if (!location)
        return std::nullopt;
      if (location->root != pointer || location->rootFrame != &frame) {
        auto resolved =
            roots(location->root, *location->rootFrame, alternative, depth + 1);
        if (!resolved)
          return std::nullopt;
        append(*resolved);
        continue;
      }
      if (pointer.getDefiningOp<mlir::LLVM::AllocaOp>() ||
          pointer.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
        append(PointerRoots{{{pointer, &frame}}, false});
        continue;
      }
      auto input = llvm::dyn_cast<mlir::BlockArgument>(pointer);
      if (input && !frame.caller && input.getOwner() == frame.entry()) {
        append(PointerRoots{{{pointer, &frame}}, false});
        continue;
      }
      return refuse(StoredPointerRefusal::UnsupportedPointerOrigin);
    }
    return result;
  }

  std::optional<StoredPointerRefusal> openWriteDomain_;
  std::vector<std::unique_ptr<Frame>> frames_;
  StoredPointerRefusal refusal_ =
      StoredPointerRefusal::UnsupportedPointerOrigin;
  llvm::DenseMap<mlir::Value, mlir::Value> reloadEquivalences_;
  llvm::DenseMap<mlir::Value, std::optional<std::set<int64_t>>> scalarDomains_;
  llvm::DenseSet<mlir::Value> activeScalarDomains_;
  std::vector<std::optional<WriteEffect>> effects_;
  std::vector<WritePositions> writePositions_;
  llvm::DenseMap<mlir::Value, std::optional<uint64_t>> allocationExtents_;
  llvm::DenseSet<std::tuple<mlir::Value, Frame *, int64_t, uint64_t,
                            mlir::Operation *, Frame *>>
      activeMemorySources_;
  llvm::DenseMap<std::tuple<mlir::Value, Frame *, int64_t, uint64_t,
                            mlir::Operation *, Frame *>,
                 std::vector<Payload>>
      memorySources_;
  llvm::DenseMap<std::tuple<mlir::Value, Frame *, mlir::Operation *>,
                 PointerRoots>
      pointerPayloads_;
};

StoredMemoryProvenance::StoredMemoryProvenance(
    mlir::LLVM::LLVMFuncOp rootCallable)
    : impl_(std::make_unique<Impl>(rootCallable)) {}
StoredMemoryProvenance::~StoredMemoryProvenance() = default;
StoredMemoryProvenance::StoredMemoryProvenance(
    StoredMemoryProvenance &&) noexcept = default;
StoredMemoryProvenance &
StoredMemoryProvenance::operator=(StoredMemoryProvenance &&) noexcept = default;

StoredPointerTargetOutcome
StoredMemoryProvenance::projectPointerTarget(mlir::Value pointer) {
  return impl_->project(pointer);
}
StoredPointerTargetOutcome StoredMemoryProvenance::projectPointerTarget(
    mlir::Value pointer, llvm::ArrayRef<mlir::Value> boundaryValues) {
  return impl_->project(pointer, boundaryValues);
}
ReachingPointerValueOutcome StoredMemoryProvenance::projectReachingPointerValue(
    mlir::LLVM::LoadOp read,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  return impl_->reaching(read, invocationPath);
}

llvm::StringRef storedPointerRefusalSpelling(StoredPointerRefusal refusal) {
  switch (refusal) {
  case StoredPointerRefusal::OpenInvocationDomain:
    return "open invocation domain";
  case StoredPointerRefusal::UnsupportedMemoryEffect:
    return "unsupported memory effect";
  case StoredPointerRefusal::UnknownByteAddress:
    return "unknown byte address";
  case StoredPointerRefusal::OutOfBoundsAccess:
    return "access exceeds the finite allocation";
  case StoredPointerRefusal::UnknownAlias:
    return "unknown memory alias";
  case StoredPointerRefusal::UnknownIntegerDomain:
    return "unknown integer domain";
  case StoredPointerRefusal::IncompleteInitialization:
    return "incomplete initialization";
  case StoredPointerRefusal::PartialPointerWrite:
    return "partial pointer representation write";
  case StoredPointerRefusal::UnsupportedPointerOrigin:
    return "unsupported pointer origin";
  case StoredPointerRefusal::UnsupportedPointerRepresentation:
    return "unsupported pointer representation";
  case StoredPointerRefusal::DistinctPointerOrigins:
    return "distinct pointer origins";
  case StoredPointerRefusal::NullPointerOrigin:
    return "pointer has only a null origin";
  case StoredPointerRefusal::OriginNotInBoundary:
    return "pointer origin is absent from the input boundary";
  }
  llvm_unreachable("unknown stored pointer refusal");
}

} // namespace loom::frontend::analysis
