#include "Fabric/IR/MemoryRoleBindings.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <utility>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace fabric;
using namespace loom::fabric;

namespace {

constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @masked_vector(%mem: memref<8xf32>, %address: index,
                           %mask: vector<4xi1>, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl mask %mask
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }
}
)mlir";

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted invalid active role bindings");
  llvm::consumeError(value.takeError());
}

CanonicalService serviceFor(mlir::ModuleOp module) {
  mlir::Operation *actor = nullptr;
  module.walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "dataflow.load")
      actor = operation;
  });
  if (!actor)
    fail("fixture", "load actor was not found");
  return take("service", CanonicalService::forActor(actor));
}

FabricTransportEndpointRef endpoint(std::uint64_t ordinal) {
  FabricMemoryOccurrenceRef memory(7);
  return FabricTransportEndpointRef{FabricTransportEndpointOwnerRef::of(memory),
                                    ordinal};
}

MemoryRoleBinding binding(ServiceValueRole role, std::uint64_t ordinal) {
  return MemoryRoleBinding{role, endpoint(ordinal)};
}

void temporalInputsOwnIndependentMatcherQueues(
    const CanonicalService &service) {
  const MemoryRoleBinding bindings[] = {
      binding(ServiceValueRole::Data, 1),
      binding(ServiceValueRole::Control, 0),
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Completion, 2),
      binding(ServiceValueRole::Mask, 0),
  };
  MemoryRoleBindingView view =
      take(__func__, MemoryRoleBindingView::create(Schedule::Temporal, service,
                                                   bindings));
  require(__func__, view.temporalInputMatcherQueues().size() == 3,
          "one matcher queue was not derived for every input role");
  require(__func__,
          view.activeBindings()[0].role == ServiceValueRole::Address &&
              view.activeBindings()[1].role == ServiceValueRole::Mask &&
              view.activeBindings()[2].role == ServiceValueRole::Control &&
              view.activeBindings()[3].role == ServiceValueRole::Data &&
              view.activeBindings()[4].role == ServiceValueRole::Completion,
          "active bindings did not normalize to Canonical Service order");
  for (const TemporalMemoryInputMatcherQueue &queue :
       view.temporalInputMatcherQueues())
    require(__func__, queue.endpoint() == endpoint(0),
            "shared ingress identity changed");
  require(__func__,
          view.temporalInputMatcherQueues()[0].role() ==
                  ServiceValueRole::Address &&
              view.temporalInputMatcherQueues()[1].role() ==
                  ServiceValueRole::Mask &&
              view.temporalInputMatcherQueues()[2].role() ==
                  ServiceValueRole::Control,
          "matcher queues are not independently role-owned");
}

void spatialBindingsStayInjective(const CanonicalService &service) {
  const MemoryRoleBinding bindings[] = {
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Mask, 0),
      binding(ServiceValueRole::Control, 2),
      binding(ServiceValueRole::Data, 3),
      binding(ServiceValueRole::Completion, 4),
  };
  expectRejected<MemoryRoleBindingView>(
      __func__,
      MemoryRoleBindingView::create(Schedule::Spatial, service, bindings));
}

void outputsStayInjective(const CanonicalService &service) {
  const MemoryRoleBinding bindings[] = {
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Mask, 1),
      binding(ServiceValueRole::Control, 2),
      binding(ServiceValueRole::Data, 3),
      binding(ServiceValueRole::Completion, 3),
  };
  expectRejected<MemoryRoleBindingView>(
      __func__,
      MemoryRoleBindingView::create(Schedule::Temporal, service, bindings));

  const MemoryRoleBinding outputSharesInput[] = {
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Mask, 1),
      binding(ServiceValueRole::Control, 2),
      binding(ServiceValueRole::Data, 0),
      binding(ServiceValueRole::Completion, 3),
  };
  expectRejected<MemoryRoleBindingView>(
      __func__, MemoryRoleBindingView::create(Schedule::Temporal, service,
                                              outputSharesInput));
}

void roleRelationIsTotal(const CanonicalService &service) {
  const MemoryRoleBinding missingMask[] = {
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Control, 2),
      binding(ServiceValueRole::Data, 3),
      binding(ServiceValueRole::Completion, 4),
  };
  expectRejected<MemoryRoleBindingView>(
      __func__,
      MemoryRoleBindingView::create(Schedule::Temporal, service, missingMask));

  const MemoryRoleBinding duplicateRole[] = {
      binding(ServiceValueRole::Address, 0),
      binding(ServiceValueRole::Address, 1),
      binding(ServiceValueRole::Mask, 2),
      binding(ServiceValueRole::Control, 3),
      binding(ServiceValueRole::Data, 4),
      binding(ServiceValueRole::Completion, 5),
  };
  expectRejected<MemoryRoleBindingView>(
      __func__, MemoryRoleBindingView::create(Schedule::Temporal, service,
                                              duplicateRole));
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry
      .insert<DataflowDialect, mlir::func::FuncDialect, mlir::DLTIDialect>();
  mlir::MLIRContext context(registry,
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module)
    fail("fixture", "failed to parse");

  CanonicalService service = serviceFor(*module);
  temporalInputsOwnIndependentMatcherQueues(service);
  spatialBindingsStayInjective(service);
  outputsStayInjective(service);
  roleRelationIsTotal(service);
  return EXIT_SUCCESS;
}
