#include "SimulationPointerCapture.h"

#include "Dataflow/IR/DataflowOps.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <system_error>

namespace loom::sim::capture_detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("simulation_input_capture_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("simulation_input_capture_unsupported: ") + message);
}

} // namespace

llvm::Expected<mlir::Value>
threadMemorySourceForRoot(const dataflow::CanonicalLogicalMemoryRootView &root,
                          detail::ResolvedLaunchContext &context) {
  if (root.formalArgIndex) {
    if (root.op != context.thread.getOperation() ||
        *root.formalArgIndex >= context.thread.getFunctionType().getNumInputs())
      return invalid("imported memory formal has the wrong thread owner");
    return context.thread.getBody().front().getArgument(*root.formalArgIndex);
  }

  auto service = llvm::dyn_cast_or_null<dataflow::MemoryServiceOp>(root.op);
  if (!service ||
      service->getParentOfType<dataflow::ThreadOp>() != context.thread)
    return invalid("imported memory root is neither a thread formal nor an "
                   "object-scoped memory service");
  if (auto argument =
          llvm::dyn_cast<mlir::BlockArgument>(service.getPointer())) {
    if (argument.getOwner() != &context.thread.getBody().front() ||
        argument.getArgNumber() >=
            context.thread.getFunctionType().getNumInputs())
      return invalid("memory service pointer has the wrong thread owner");
  }
  return service.getPointer();
}

llvm::Expected<std::optional<PointerValueTargetProjection>>
pointerValueTargetForInput(
    const dataflow::CanonicalDataflowProgramView &program,
    detail::ResolvedLaunchContext &context, std::uint64_t valueInputOrdinal) {
  if (valueInputOrdinal >= context.numValueInputs ||
      valueInputOrdinal >= context.valueInputShapes.size())
    return invalid("pointer value-input ordinal is outside the graph ABI");
  const detail::LaneShape &shape = context.valueInputShapes[valueInputOrdinal];
  if (!shape.pointerLayout)
    return std::optional<PointerValueTargetProjection>{};

  mlir::Value source =
      context.graphLaunchOp.getValueInputs()[valueInputOrdinal];
  llvm::SmallVector<std::uint64_t, 2> matchingRoots;
  for (auto [rootOrdinal, rootRef] : llvm::enumerate(context.importedRoots)) {
    llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> root =
        program.resolve(rootRef);
    if (!root)
      return root.takeError();
    auto service = llvm::dyn_cast_or_null<dataflow::MemoryServiceOp>(root->op);
    if (service && service.getPointer() == source)
      matchingRoots.push_back(rootOrdinal);
  }
  if (matchingRoots.empty())
    return unsupported(
        "first-class pointer input has no object-scoped memory service");
  return std::optional<PointerValueTargetProjection>(
      PointerValueTargetProjection{
          {matchingRoots.front(), shape.pointerLayout->addressBits},
          std::move(matchingRoots)});
}

llvm::Error
attachPointerValueTargets(const dataflow::CanonicalDataflowProgramView &program,
                          detail::ResolvedLaunchContext &context,
                          SimulationInputCapturePlan &plan) {
  if (plan.memoryRootBindings.size() != context.importedRoots.size())
    return invalid("memory-root capture is not dense in imported-root order");

  for (std::uint64_t valueOrdinal = 0; valueOrdinal < context.numValueInputs;
       ++valueOrdinal) {
    auto projection =
        pointerValueTargetForInput(program, context, valueOrdinal);
    if (!projection)
      return projection.takeError();
    if (!*projection)
      continue;
    if (valueOrdinal >= plan.valueInputs.size() ||
        plan.valueInputs[valueOrdinal].valueInputOrdinal != valueOrdinal)
      return invalid("pointer value-input capture is not dense");
    SimulationValueInputCapture &input = plan.valueInputs[valueOrdinal];
    if (input.fixedValue)
      return unsupported(
          "fixed first-class pointer inputs have no runtime object binding");

    const SimulationMemoryRootCapture &selected =
        plan.memoryRootBindings[(*projection)->target.memoryRootBindingOrdinal];
    for (std::uint64_t rootOrdinal :
         (*projection)->equivalentMemoryRootOrdinals) {
      const SimulationMemoryRootCapture &candidate =
          plan.memoryRootBindings[rootOrdinal];
      if (selected.objectIndex != candidate.objectIndex ||
          selected.byteOffset != candidate.byteOffset)
        return invalid("one pointer input resolves to inconsistent memory "
                       "service bindings");
    }
    input.pointerTarget = (*projection)->target;
  }
  return llvm::Error::success();
}

} // namespace loom::sim::capture_detail
