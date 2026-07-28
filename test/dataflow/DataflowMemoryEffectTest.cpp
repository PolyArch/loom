//===- DataflowMemoryEffectTest.cpp - Memory actor effect projection ------===//
//
// Proves the exact standard MLIR memory-effect projection of the canonical
// Dataflow memory actors. The addressed base projection names the actor's
// memory operand: a load reads it, a store writes it, an atomic
// read-modify-write and a compare-exchange do both. An actor whose aggregate
// contract is atomic or volatile also publishes conservative unbound effects so
// it can never be treated as trivially dead, and a fence publishes only those.
//
// The assertions are exact: every effect's kind, multiplicity, bound value or
// absence of one, and resource are checked, so a duplicated or extra effect
// fails.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

// Each function holds exactly one canonical memory actor. The plain load and
// store fix the addressed base projection; the volatile and atomic loads are
// the two contract arms that add the conservative unbound effects.
constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @plain_load(%mem: memref<10xi32>, %addr: index, %ctrl: none)
      -> (i32, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xi32>
    return %data, %done : i32, none
  }

  func.func @plain_store(%mem: memref<10xi32>, %addr: index, %value: i32,
                         %ctrl: none) -> none {
    %done = dataflow.store %mem[%addr] %value %ctrl : memref<10xi32>
    return %done : none
  }

  func.func @volatile_load(%mem: memref<10xi32>, %addr: index, %ctrl: none)
      -> (i32, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl
        {contract = #dataflow.plain_access<is_volatile = true>}
        : memref<10xi32>
    return %data, %done : i32, none
  }

  func.func @atomic_load(%mem: memref<10xi32>, %addr: index, %ctrl: none)
      -> (i32, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl
        {contract = #dataflow.atomic_access<ordering = acquire,
                                            sync_scope = <system>,
                                            source_alignment_bytes = 4>}
        : memref<10xi32>
    return %data, %done : i32, none
  }

  func.func @rmw(%mem: memref<10xi32>, %addr: index, %value: i32, %ctrl: none)
      -> (i32, none) {
    %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<10xi32>
    return %old, %done : i32, none
  }

  func.func @cmpxchg(%mem: memref<10xi32>, %addr: index, %value: i32,
                     %ctrl: none) -> (i32, i1, none) {
    %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %value %value %ctrl
        {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                               failure_ordering = monotonic,
                                               sync_scope = <system>,
                                               source_alignment_bytes = 4>}
        : memref<10xi32> -> i1
    return %old, %ok, %done : i32, i1, none
  }

  func.func @fence(%ctrl: none) -> none {
    %done = dataflow.fence %ctrl
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    return %done : none
  }
}
)mlir";

/// How many effects of each kind an actor publishes, split by whether the
/// effect names the actor's memory operand.
struct EffectCounts {
  unsigned boundRead = 0;
  unsigned boundWrite = 0;
  unsigned unboundRead = 0;
  unsigned unboundWrite = 0;
};

void print(llvm::raw_ostream &stream, const EffectCounts &counts) {
  stream << "bound read=" << counts.boundRead
         << " bound write=" << counts.boundWrite
         << " unbound read=" << counts.unboundRead
         << " unbound write=" << counts.unboundWrite;
}

struct Expectation {
  llvm::StringRef function;
  EffectCounts effects;
};

constexpr Expectation expectations[] = {
    {"plain_load", {1, 0, 0, 0}},    {"plain_store", {0, 1, 0, 0}},
    {"volatile_load", {1, 0, 1, 1}}, {"atomic_load", {1, 0, 1, 1}},
    {"rmw", {1, 1, 1, 1}},           {"cmpxchg", {1, 1, 1, 1}},
    {"fence", {0, 0, 1, 1}},
};

/// The single Dataflow actor in one fixture function.
mlir::Operation *findActor(mlir::func::FuncOp function) {
  mlir::Operation *actor = nullptr;
  unsigned found = 0;
  function.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "dataflow")
      return;
    actor = op;
    ++found;
  });
  return found == 1 ? actor : nullptr;
}

/// Sorts every published effect into its bucket, rejecting anything the
/// projection must not publish: an effect that is neither a read nor a write,
/// an effect on a resource other than the default one, and a bound effect that
/// names a value other than the actor's memory operand.
bool countEffects(mlir::Operation *actor, llvm::StringRef name,
                  EffectCounts &counts) {
  auto interface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(actor);
  if (!interface) {
    llvm::errs() << name << " does not project memory effects\n";
    return false;
  }
  mlir::Value memory;
  if (actor->getNumOperands() != 0 &&
      llvm::isa<mlir::MemRefType>(actor->getOperand(0).getType()))
    memory = actor->getOperand(0);

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> effects;
  interface.getEffects(effects);
  for (const mlir::MemoryEffects::EffectInstance &effect : effects) {
    if (effect.getResource() != mlir::SideEffects::DefaultResource::get()) {
      llvm::errs() << name
                   << " publishes an effect on a non-default resource\n";
      return false;
    }
    const bool isRead =
        llvm::isa<mlir::MemoryEffects::Read>(effect.getEffect());
    if (!isRead && !llvm::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
      llvm::errs() << name
                   << " publishes an effect that is neither a read nor "
                      "a write\n";
      return false;
    }
    mlir::Value value = effect.getValue();
    if (!value) {
      ++(isRead ? counts.unboundRead : counts.unboundWrite);
      continue;
    }
    if (value != memory) {
      llvm::errs() << name
                   << " binds an effect to a value that is not its "
                      "memory operand\n";
      return false;
    }
    ++(isRead ? counts.boundRead : counts.boundWrite);
  }
  return true;
}

bool checkActor(mlir::Operation *actor, const Expectation &expectation) {
  EffectCounts counts;
  if (!countEffects(actor, expectation.function, counts))
    return false;
  const EffectCounts &expected = expectation.effects;
  if (counts.boundRead == expected.boundRead &&
      counts.boundWrite == expected.boundWrite &&
      counts.unboundRead == expected.unboundRead &&
      counts.unboundWrite == expected.unboundWrite)
    return true;
  llvm::errs() << expectation.function << " projects ";
  print(llvm::errs(), counts);
  llvm::errs() << ", expected ";
  print(llvm::errs(), expected);
  llvm::errs() << '\n';
  return false;
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry,
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module) {
    llvm::errs() << "failed to parse the memory actor fixture\n";
    return EXIT_FAILURE;
  }

  bool ok = true;
  for (const Expectation &expectation : expectations) {
    auto function =
        module->lookupSymbol<mlir::func::FuncOp>(expectation.function);
    mlir::Operation *actor = function ? findActor(function) : nullptr;
    if (!actor) {
      llvm::errs() << "fixture does not hold exactly one actor in "
                   << expectation.function << '\n';
      ok = false;
      continue;
    }
    ok &= checkActor(actor, expectation);
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
