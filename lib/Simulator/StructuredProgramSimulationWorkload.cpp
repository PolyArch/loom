//===- StructuredProgramSimulationWorkload.cpp ---------------------------===//

#include "SimulationWireInternal.h"

#include "Common/ArtifactFinalizer.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <algorithm>
#include <utility>

namespace loom::sim {
namespace {

int compareStructuredRefs(const frontend::StructuredEntityRef &lhs,
                          const frontend::StructuredEntityRef &rhs) {
  if (int result = detail::compareIdentities(lhs.parent, rhs.parent))
    return result;
  if (lhs.kind != rhs.kind)
    return static_cast<std::uint32_t>(lhs.kind) <
                   static_cast<std::uint32_t>(rhs.kind)
               ? -1
               : 1;
  if (lhs.ordinal == rhs.ordinal)
    return 0;
  return lhs.ordinal < rhs.ordinal ? -1 : 1;
}

void encodeStructuredRef(detail::WireWriter &writer,
                         const frontend::StructuredEntityRef &reference) {
  writer.bytes(frontend::encodeStructuredEntityRef(reference));
}

llvm::Expected<frontend::StructuredEntityRef>
decodeStructuredRef(detail::WireReader &reader) {
  llvm::Expected<llvm::ArrayRef<std::uint8_t>> bytes =
      reader.bytes(frontend::structuredEntityRefWireSize);
  if (!bytes)
    return bytes.takeError();
  return frontend::decodeStructuredEntityRef(*bytes);
}

void encodeMemoryTarget(detail::WireWriter &writer,
                        const StructuredProgramMemoryTarget &target) {
  if (const auto *argument = std::get_if<EntryPointerArgumentTarget>(&target)) {
    writer.u32(0);
    writer.u64(argument->argumentOrdinal);
    return;
  }
  writer.u32(1);
  encodeStructuredRef(writer, std::get<GlobalObjectTarget>(target).global);
}

llvm::Expected<StructuredProgramMemoryTarget>
decodeMemoryTarget(detail::WireReader &reader) {
  llvm::Expected<std::uint32_t> tag = reader.u32();
  if (!tag)
    return tag.takeError();
  if (*tag == 0) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    return StructuredProgramMemoryTarget{EntryPointerArgumentTarget{*ordinal}};
  }
  if (*tag == 1) {
    llvm::Expected<frontend::StructuredEntityRef> global =
        decodeStructuredRef(reader);
    if (!global)
      return global.takeError();
    return StructuredProgramMemoryTarget{GlobalObjectTarget{*global}};
  }
  return detail::invalid(
      "simulation workload: unknown Structured memory target");
}

std::vector<std::uint8_t> encodeStructuredProgramWorkload(
    const StructuredProgramSimulationWorkload &workload,
    const detail::ResolvedStructuredProgramContext &context) {
  detail::WireWriter writer;
  writer.u32(
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram));
  encodeStructuredRef(writer, workload.entryRef);
  writer.u64(workload.argumentPlan.size());
  for (std::uint64_t ordinal = 0; ordinal < workload.argumentPlan.size();
       ++ordinal) {
    writer.u64(ordinal);
    const StructuredProgramArgumentSource &source =
        workload.argumentPlan[ordinal];
    if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source)) {
      writer.u32(0);
      assert(context.argumentShapes[ordinal] &&
             "validated fixed argument has no lane shape");
      detail::encodeValueSequence(writer, *fixed,
                                  *context.argumentShapes[ordinal]);
    } else if (std::holds_alternative<StructuredRuntimeValueInput>(source)) {
      writer.u32(1);
    } else {
      writer.u32(2);
    }
  }
  writer.u32(workload.observableContract.returnValue ? 1 : 0);
  writer.u64(workload.observableContract.memories.size());
  for (const StructuredProgramMemoryObservable &observable :
       workload.observableContract.memories) {
    encodeMemoryTarget(writer, observable.target);
    writer.u32(static_cast<std::uint32_t>(observable.form));
  }
  return writer.take();
}

struct DecodedStructuredProgramWorkload {
  StructuredProgramSimulationWorkload workload;
  detail::ResolvedStructuredProgramContext context;
};

llvm::Expected<DecodedStructuredProgramWorkload>
decodeStructuredProgramWorkload(
    llvm::ArrayRef<std::uint8_t> bytes,
    const frontend::StructuredProgramCandidateView &view) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root == static_cast<std::uint32_t>(SimulationWorkloadKind::Spatial))
    return detail::invalid("simulation workload: Structured import received "
                           "a Spatial root");
  if (*root == static_cast<std::uint32_t>(SimulationWorkloadKind::System))
    return detail::invalid(
        "simulation workload: the System root is fail-closed");
  if (*root !=
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram))
    return detail::invalid("simulation workload: unknown root discriminant");

  llvm::Expected<frontend::StructuredEntityRef> entry =
      decodeStructuredRef(reader);
  if (!entry)
    return entry.takeError();
  StructuredProgramSimulationWorkload workload{*entry};
  llvm::Expected<detail::ResolvedStructuredProgramContext> context =
      detail::resolveStructuredProgramContext(view, workload.entryRef);
  if (!context)
    return context.takeError();

  llvm::Expected<std::uint64_t> argumentCount = reader.u64();
  if (!argumentCount)
    return argumentCount.takeError();
  if (llvm::Error error = reader.guardCount(*argumentCount, 12))
    return std::move(error);
  workload.argumentPlan.reserve(*argumentCount);
  for (std::uint64_t index = 0; index < *argumentCount; ++index) {
    llvm::Expected<std::uint64_t> ordinal = reader.u64();
    if (!ordinal)
      return ordinal.takeError();
    if (*ordinal != index || *ordinal >= context->argumentTypes.size())
      return detail::invalid("simulation workload: Structured argument keys "
                             "are not the dense ABI ordinals");
    llvm::Expected<std::uint32_t> tag = reader.u32();
    if (!tag)
      return tag.takeError();
    if (*tag == 0) {
      if (!context->argumentShapes[*ordinal])
        return detail::invalid("simulation workload: a pointer argument "
                               "cannot carry fixed value bits");
      llvm::Expected<CanonicalValueSequence> fixed =
          detail::decodeValueSequence(reader,
                                      *context->argumentShapes[*ordinal]);
      if (!fixed)
        return fixed.takeError();
      workload.argumentPlan.emplace_back(std::move(*fixed));
    } else if (*tag == 1) {
      workload.argumentPlan.emplace_back(StructuredRuntimeValueInput{});
    } else if (*tag == 2) {
      workload.argumentPlan.emplace_back(StructuredRuntimeMemoryInput{});
    } else {
      return detail::invalid("simulation workload: unknown Structured "
                             "argument-source discriminant");
    }
  }

  llvm::Expected<std::uint32_t> returnValue = reader.u32();
  if (!returnValue)
    return returnValue.takeError();
  if (*returnValue > 1)
    return detail::invalid(
        "simulation workload: return selector is not canonical bool");
  workload.observableContract.returnValue = *returnValue != 0;

  llvm::Expected<std::uint64_t> memoryCount = reader.u64();
  if (!memoryCount)
    return memoryCount.takeError();
  if (llvm::Error error = reader.guardCount(*memoryCount, 8))
    return std::move(error);
  workload.observableContract.memories.reserve(*memoryCount);
  for (std::uint64_t index = 0; index < *memoryCount; ++index) {
    llvm::Expected<StructuredProgramMemoryTarget> target =
        decodeMemoryTarget(reader);
    if (!target)
      return target.takeError();
    llvm::Expected<std::uint32_t> form = reader.u32();
    if (!form)
      return form.takeError();
    if (*form >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return detail::invalid(
          "simulation workload: unknown memory observation form");
    if (index > 0 &&
        detail::compareStructuredMemoryTargets(
            *target, workload.observableContract.memories.back().target) <= 0)
      return detail::invalid("simulation workload: Structured memory targets "
                             "are not sorted or contain a duplicate");
    workload.observableContract.memories.push_back(
        {*target, static_cast<MemoryObservationForm>(*form)});
  }
  if (!reader.atEnd())
    return detail::invalid("simulation workload: trailing bytes");
  return DecodedStructuredProgramWorkload{std::move(workload),
                                          std::move(*context)};
}

} // namespace

namespace detail {

llvm::Expected<::loom::ArtifactIdentity>
structuredProgramWorkloadOwnerIdentity(llvm::ArrayRef<std::uint8_t> bytes) {
  detail::WireReader reader(bytes);
  llvm::Expected<std::uint32_t> root = reader.u32();
  if (!root)
    return root.takeError();
  if (*root !=
      static_cast<std::uint32_t>(SimulationWorkloadKind::StructuredProgram))
    return detail::invalid(
        "simulation workload: stored import requires a Structured root");
  llvm::Expected<frontend::StructuredEntityRef> entry =
      decodeStructuredRef(reader);
  if (!entry)
    return entry.takeError();
  return entry->parent;
}

int compareStructuredMemoryTargets(const StructuredProgramMemoryTarget &lhs,
                                   const StructuredProgramMemoryTarget &rhs) {
  if (lhs.index() != rhs.index())
    return lhs.index() < rhs.index() ? -1 : 1;
  if (const auto *lhsArgument = std::get_if<EntryPointerArgumentTarget>(&lhs)) {
    const std::uint64_t rhsOrdinal =
        std::get<EntryPointerArgumentTarget>(rhs).argumentOrdinal;
    if (lhsArgument->argumentOrdinal == rhsOrdinal)
      return 0;
    return lhsArgument->argumentOrdinal < rhsOrdinal ? -1 : 1;
  }
  return compareStructuredRefs(std::get<GlobalObjectTarget>(lhs).global,
                               std::get<GlobalObjectTarget>(rhs).global);
}

llvm::Expected<ResolvedStructuredProgramContext>
resolveStructuredProgramContext(
    const frontend::StructuredProgramCandidateView &view,
    const frontend::StructuredEntityRef &entry) {
  llvm::Expected<frontend::StructuredEntity> entity = view.resolve(entry);
  if (!entity)
    return entity.takeError();
  if (entry.kind != frontend::StructuredEntityKind::Operation ||
      !entity->operation)
    return invalid(
        "simulation workload: entry_ref is not a Structured operation");
  auto function = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(entity->operation);
  if (!function || function.isExternal())
    return invalid("simulation workload: entry_ref is not a defined llvm.func");

  ResolvedStructuredProgramContext context;
  context.entryOp = function.getOperation();
  mlir::LLVM::LLVMFunctionType functionType = function.getFunctionType();
  context.argumentTypes.assign(functionType.getParams().begin(),
                               functionType.getParams().end());
  context.argumentShapes.reserve(context.argumentTypes.size());
  for (mlir::Type type : context.argumentTypes) {
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
      context.argumentShapes.push_back(std::nullopt);
      continue;
    }
    llvm::Expected<LaneShape> shape =
        laneShapeOf(type, function.getOperation());
    if (!shape)
      return shape.takeError();
    context.argumentShapes.push_back(*shape);
  }
  context.returnType = functionType.getReturnType();
  if (!mlir::isa<mlir::LLVM::LLVMVoidType>(context.returnType)) {
    llvm::Expected<LaneShape> shape =
        laneShapeOf(context.returnType, function.getOperation());
    if (!shape)
      return shape.takeError();
    context.returnShape = *shape;
  }
  return context;
}

llvm::Error validateStructuredProgramWorkload(
    const StructuredProgramSimulationWorkload &workload,
    const ResolvedStructuredProgramContext &context,
    const frontend::StructuredProgramCandidateView &view) {
  if (workload.argumentPlan.size() != context.argumentTypes.size())
    return invalid("simulation workload: Structured argument plan is not "
                   "total over the entry ABI");
  for (std::uint64_t ordinal = 0; ordinal < workload.argumentPlan.size();
       ++ordinal) {
    const StructuredProgramArgumentSource &source =
        workload.argumentPlan[ordinal];
    const bool pointer = !context.argumentShapes[ordinal].has_value();
    if (pointer) {
      if (!std::holds_alternative<StructuredRuntimeMemoryInput>(source))
        return invalid("simulation workload: pointer arguments require "
                       "RuntimeMemory classification");
      continue;
    }
    if (std::holds_alternative<StructuredRuntimeMemoryInput>(source))
      return invalid("simulation workload: non-pointer arguments cannot use "
                     "RuntimeMemory classification");
    if (const auto *fixed = std::get_if<CanonicalValueSequence>(&source)) {
      if (fixed->tokenCount != 1)
        return invalid("simulation workload: fixed Structured arguments hold "
                       "exactly one token");
      if (llvm::Error error = validateValueSequence(
              *fixed, *context.argumentShapes[ordinal],
              "simulation workload: fixed Structured argument"))
        return error;
    }
  }

  if (workload.observableContract.returnValue && !context.returnShape)
    return invalid("simulation workload: a void entry has no return value to "
                   "observe");
  for (std::size_t index = 0;
       index < workload.observableContract.memories.size(); ++index) {
    const StructuredProgramMemoryObservable &observable =
        workload.observableContract.memories[index];
    if (static_cast<std::uint32_t>(observable.form) >
        static_cast<std::uint32_t>(MemoryObservationForm::DiffFromRuntimeInput))
      return invalid(
          "simulation workload: memory observation form is out of domain");
    if (index > 0 &&
        compareStructuredMemoryTargets(
            observable.target,
            workload.observableContract.memories[index - 1].target) <= 0)
      return invalid("simulation workload: Structured memory targets are not "
                     "sorted or contain a duplicate");
    if (const auto *argument =
            std::get_if<EntryPointerArgumentTarget>(&observable.target)) {
      if (argument->argumentOrdinal >= workload.argumentPlan.size() ||
          !std::holds_alternative<StructuredRuntimeMemoryInput>(
              workload.argumentPlan[argument->argumentOrdinal]))
        return invalid("simulation workload: memory observable does not name "
                       "a RuntimeMemory entry argument");
      continue;
    }
    const frontend::StructuredEntityRef &global =
        std::get<GlobalObjectTarget>(observable.target).global;
    llvm::Expected<frontend::StructuredEntity> entity = view.resolve(global);
    if (!entity)
      return entity.takeError();
    if (global.kind != frontend::StructuredEntityKind::Operation ||
        !llvm::isa_and_nonnull<mlir::LLVM::GlobalOp>(entity->operation))
      return invalid("simulation workload: GlobalObject does not resolve to "
                     "an LLVM global");
    if (observable.form == MemoryObservationForm::DiffFromRuntimeInput)
      return invalid("simulation workload: a Structured global without a "
                     "runtime binding requires FullState observation");
  }
  return llvm::Error::success();
}

} // namespace detail

llvm::Expected<CanonicalSimulationWorkload> finalizeSimulationWorkload(
    const StructuredProgramSimulationWorkload &workload,
    const frontend::StructuredProgramCandidateView &view) {
  llvm::Expected<detail::ResolvedStructuredProgramContext> context =
      detail::resolveStructuredProgramContext(view, workload.entryRef);
  if (!context)
    return context.takeError();
  if (llvm::Error error =
          detail::validateStructuredProgramWorkload(workload, *context, view))
    return std::move(error);
  ::loom::CanonicalSemanticBytes bytes(
      encodeStructuredProgramWorkload(workload, *context));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  return CanonicalSimulationWorkload(
      identity, SimulationWorkloadModel{workload}, std::move(bytes));
}

llvm::Expected<CanonicalSimulationWorkload>
importSimulationWorkload(llvm::ArrayRef<std::uint8_t> canonicalBytes,
                         const frontend::StructuredProgramCandidateView &view,
                         const ::loom::ArtifactIdentity &expectedIdentity) {
  llvm::Expected<DecodedStructuredProgramWorkload> decoded =
      decodeStructuredProgramWorkload(canonicalBytes, view);
  if (!decoded)
    return decoded.takeError();
  if (llvm::Error error = detail::validateStructuredProgramWorkload(
          decoded->workload, decoded->context, view))
    return std::move(error);
  const std::vector<std::uint8_t> reencoded =
      encodeStructuredProgramWorkload(decoded->workload, decoded->context);
  if (!llvm::ArrayRef<std::uint8_t>(reencoded).equals(canonicalBytes))
    return detail::invalid(
        "simulation workload: noncanonical bytes do not re-encode exactly");
  ::loom::CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(canonicalBytes.begin(), canonicalBytes.end()));
  ::loom::ArtifactIdentity identity =
      ::loom::finalizeArtifactIdentity(simulationWorkloadSchema, bytes);
  if (identity != expectedIdentity)
    return detail::invalid(
        "simulation workload: identity does not match the expected artifact");
  return CanonicalSimulationWorkload(
      identity, SimulationWorkloadModel{std::move(decoded->workload)},
      std::move(bytes));
}

} // namespace loom::sim
