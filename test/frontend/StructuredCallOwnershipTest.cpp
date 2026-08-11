#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "structured call ownership anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::frontend::StructuredProgramCandidate makeProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func internal @helper(%value: i32) -> i32 {
    %one = arith.constant 1 : i32
    %sum = arith.addi %value, %one : i32
    llvm.return %sum : i32
  }

  llvm.func @kernel(%value: i32) -> i32 {
    %from_helper = llvm.call @helper(%value) : (i32) -> i32
    %two = arith.constant 2 : i32
    %result = arith.muli %from_helper, %two : i32
    llvm.return %result : i32
  }

  llvm.func @other_caller(%value: i32) -> i32 {
    %result = llvm.call @helper(%value) : (i32) -> i32
    llvm.return %result : i32
  }

  llvm.func @dead_result_caller(%value: i32) -> i32 {
    %unused = llvm.call @helper(%value) : (i32) -> i32
    llvm.return %value : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the direct-call fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeNestedProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func internal @nested_helper(%value: i32, %condition: i1) -> i32 {
    %selected = scf.if %condition -> (i32) {
      %one = arith.constant 1 : i32
      %sum = arith.addi %value, %one : i32
      scf.yield %sum : i32
    } else {
      %two = arith.constant 2 : i32
      %sum = arith.addi %value, %two : i32
      scf.yield %sum : i32
    }
    llvm.return %selected : i32
  }

  llvm.func @nested_kernel(%value: i32, %condition: i1) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%current = %value) -> (i32) {
      %next = llvm.call @nested_helper(%current, %condition)
          : (i32, i1) -> i32
      scf.yield %next : i32
    }
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the nested direct-call fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeNoInlineProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func internal @noinline_helper(%value: i32) -> i32 attributes {no_inline} {
    %one = arith.constant 1 : i32
    %sum = arith.addi %value, %one : i32
    llvm.return %sum : i32
  }

  llvm.func @noinline_kernel(%value: i32) -> i32 {
    %result = llvm.call @noinline_helper(%value) : (i32) -> i32
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the no-inline direct-call fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeUnregisteredIntrinsicProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func @read_cycle() -> i32 {
    %cycles = llvm.call_intrinsic "llvm.readcyclecounter"() : () -> i64
    %result = llvm.trunc %cycles : i64 to i32
    llvm.return %result : i32
  }

  llvm.func internal @read_cycle_helper() -> i32 {
    %cycles = llvm.call_intrinsic "llvm.readcyclecounter"() : () -> i64
    %result = llvm.trunc %cycles : i64 to i32
    llvm.return %result : i32
  }

  llvm.func @read_cycle_caller() -> i32 {
    %result = llvm.call @read_cycle_helper() : () -> i32
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the unregistered-intrinsic fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeAllocatingCalleeProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func internal @allocating_helper(%value: i32) -> i32 {
    %slot = memref.alloc() : memref<1xi32>
    %c0 = arith.constant 0 : index
    memref.store %value, %slot[%c0] : memref<1xi32>
    %result = memref.load %slot[%c0] : memref<1xi32>
    llvm.return %result : i32
  }

  llvm.func @allocating_caller(%value: i32) -> i32 {
    %result = llvm.call @allocating_helper(%value) : (i32) -> i32
    llvm.return %result : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the allocating-callee fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeBrokenFinalLoweringProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<dataflow::DataflowDialect, loom::LoomDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %descriptor: !llvm.ptr, %value: i32) ctrl (%start: none) {
    "loom.spatial_region"(%descriptor, %value)
        <{graph_name = "broken_dynamic_service",
          operandSegmentSizes = array<i32: 2, 0, 0, 0>,
          resultSegmentSizes = array<i32: 0, 0>, source_maps = []}> ({
      ^bb0(%descriptor_input: !llvm.ptr, %value_input: i32):
        %target = llvm.load %descriptor_input : !llvm.ptr -> !llvm.ptr
        llvm.store %value_input, %target : i32, !llvm.ptr
        "loom.spatial_yield"()
            <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
    }) : (!llvm.ptr, i32) -> ()
    dataflow.thread.yield
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the final-lowering boundary fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate makeDescriptorPointerLoopProgram() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128",
  llvm.target_triple = "riscv64-unknown-unknown-elf"
} {
  llvm.func @descriptor_pointer_loop(%descriptor: !llvm.ptr, %value: i32) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %iv = %zero to %one step %one {
      %target = llvm.load %descriptor : !llvm.ptr -> !llvm.ptr
      llvm.store %value, %target : i32, !llvm.ptr
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the descriptor-pointer loop fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredEntityRef
findFunction(const loom::frontend::StructuredProgramCandidate &program,
             llvm::StringRef name) {
  auto view = take(program.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("missing function " + name);
}

loom::frontend::StructuredEntityRef
findCall(const loom::frontend::StructuredProgramCandidate &program,
         llvm::StringRef caller, llvm::StringRef callee) {
  auto view = take(program.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(entity.operation);
    if (!call || !call.getCalleeAttr() ||
        call.getCalleeAttr().getValue() != callee)
      continue;
    auto function = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == caller)
      return entity.reference;
  }
  fail("missing exact direct call site");
}

loom::frontend::StructuredEntityRef
findFor(const loom::frontend::StructuredProgramCandidate &program,
        llvm::StringRef functionName) {
  auto view = take(program.view());
  for (const loom::frontend::StructuredEntity &entity :
       view.entities(loom::frontend::StructuredEntityKind::Operation)) {
    if (!llvm::isa_and_nonnull<mlir::scf::ForOp>(entity.operation))
      continue;
    auto function = entity.operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (function && function.getSymName() == functionName)
      return entity.reference;
  }
  fail("missing selected scf.for");
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void decisionCodecHasStableFraming() {
  constexpr llvm::StringLiteral expectedSchema =
      "loom.spatial_ownership.decision.1.1";
  llvm::ArrayRef<std::uint8_t> schema =
      loom::frontend::spatialOwnershipDecisionSchemaBytes();
  if (llvm::StringRef(reinterpret_cast<const char *>(schema.data()),
                      schema.size()) != expectedSchema)
    fail("spatial ownership decision schema spelling drifted");

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> identityBytes{};
  for (std::size_t index = 0; index < identityBytes.size(); ++index)
    identityBytes[index] = static_cast<std::uint8_t>(index);
  loom::ArtifactIdentity parent =
      take(loom::ArtifactIdentity::fromBytes(identityBytes));
  const loom::frontend::StructuredEntityRef scope{
      parent, loom::frontend::StructuredEntityKind::Operation,
      UINT64_C(0x0102030405060708)};
  const loom::frontend::StructuredEntityRef call{
      parent, loom::frontend::StructuredEntityKind::Operation,
      UINT64_C(0x1112131415161718)};
  const loom::frontend::SpatialOwnershipDecision decision{
      {scope},
      {loom::frontend::RootRelativeAddressProjection{64},
       loom::frontend::ForallOwnershipShape::LogicalThreadDomain,
       loom::frontend::DirectCallSpecializationShape::UniformExactConstants,
       loom::frontend::DirectCallInliningDecision{call}}};
  std::vector<std::uint8_t> expected(identityBytes.begin(),
                                     identityBytes.end());
  appendU32(expected, 0);
  appendU64(expected, UINT64_C(0x0102030405060708));
  expected.push_back(1);
  appendU32(expected, 64);
  expected.push_back(2);
  expected.push_back(1);
  expected.push_back(1);
  expected.insert(expected.end(), identityBytes.begin(), identityBytes.end());
  appendU32(expected, 0);
  appendU64(expected, UINT64_C(0x1112131415161718));
  auto encoded = take(loom::frontend::encodeSpatialOwnershipDecision(decision));
  if (encoded != expected)
    fail("spatial ownership decision schema 1.1 bytes drifted");

  const loom::frontend::SpatialOwnershipDecision absent{{scope}, {}};
  auto absentBytes =
      take(loom::frontend::encodeSpatialOwnershipDecision(absent));
  constexpr std::size_t fixedPrefix =
      loom::frontend::structuredEntityRefWireSize + 8;
  if (absentBytes.size() !=
          fixedPrefix + loom::frontend::structuredEntityRefWireSize ||
      llvm::any_of(
          llvm::ArrayRef<std::uint8_t>(absentBytes).drop_front(fixedPrefix),
          [](std::uint8_t byte) { return byte != 0; }))
    fail("absent inline coordinate did not use a fixed all-zero slot");
  absentBytes.back() = 1;
  auto malformed = loom::frontend::adoptSpatialOwnershipDecision(absentBytes);
  if (malformed)
    fail("nonzero bytes in an absent inline slot were accepted");
  llvm::consumeError(malformed.takeError());
}

void exactDirectCallSiteProducesAnInlineCandidate() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-call-ownership", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto program = makeProgram();
  const loom::frontend::StructuredEntityRef kernel =
      findFunction(program, "kernel");
  const loom::frontend::StructuredEntityRef helper =
      findFunction(program, "helper");
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "kernel", "helper");

  auto scopes =
      take(loom::frontend::enumerateSpatialOwnershipScopeDomain(program));
  bool acceptedKernel = false;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : scopes)
    if (const auto *scope =
            std::get_if<loom::frontend::SpatialOwnershipScope>(&entry))
      acceptedKernel |= scope->selection == kernel;
  if (!acceptedKernel)
    fail("an exact inlineable direct call rejected the callable scope");

  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, kernel));
  if (decisions.size() != 2 || decisions.front().directCallInlining ||
      !decisions.back().directCallInlining ||
      decisions.back().directCallInlining->callSite != call)
    fail("direct-call decision domain order is not canonical");
  const auto noInlineDecision = llvm::find_if(
      decisions,
      [](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return !point.directCallInlining;
      });
  if (noInlineDecision == decisions.end())
    fail("decision domain omitted the exact no-inline coordinate");
  auto unresolved =
      loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {kernel}, *noInlineDecision);
  if (unresolved)
    fail("a general call entered a materialized Spatial region");
  bool classifiedNonFinalizable = false;
  llvm::Error unhandled = llvm::handleErrors(
      unresolved.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        classifiedNonFinalizable =
            error.kind() ==
                loom::frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable &&
            error.message().find("unresolved general call") !=
                std::string::npos;
      });
  if (unhandled)
    fail("unresolved call escaped candidate rejection: " +
         llvm::toString(std::move(unhandled)));
  if (!classifiedNonFinalizable)
    fail("unresolved call lacked a typed NonFinalizable disposition");

  const auto inlineDecision = llvm::find_if(
      decisions,
      [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining &&
               point.directCallInlining->callSite == call;
      });
  if (inlineDecision == decisions.end())
    fail("decision domain omitted the exact direct-call inline choice");

  const loom::frontend::SpatialOwnershipDecision decision{
      loom::frontend::SpatialOwnershipScope{kernel}, *inlineDecision};
  auto encoded = take(loom::frontend::encodeSpatialOwnershipDecision(decision));
  if (!(take(loom::frontend::adoptSpatialOwnershipDecision(encoded)) ==
        decision))
    fail("direct-call inline decision did not roundtrip exactly");

  loom::frontend::SpatialOwnershipDecisionPoint invalid = *inlineDecision;
  invalid.directCallInlining =
      loom::frontend::DirectCallInliningDecision{helper};
  auto invalidCandidate =
      loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {kernel}, invalid);
  if (invalidCandidate)
    fail("a non-call operation was accepted as an inline call site");
  bool invalidBecameCandidateRejection = false;
  llvm::Error invalidUnhandled = llvm::handleErrors(
      invalidCandidate.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &) {
        invalidBecameCandidateRejection = true;
      });
  llvm::consumeError(std::move(invalidUnhandled));
  if (invalidBecameCandidateRejection)
    fail("an invalid inline coordinate became a candidate disposition");

  auto materialized =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {kernel}, *inlineDecision));
  if (!materialized.blockActivityLineage.empty())
    fail("direct-call inlining published a non-exact activity projection");
  bool retainedCallInSpatialRegion = false;
  materialized.structuredProgram.module().walk(
      [&](loom::SpatialRegionOp spatial) {
        spatial.walk([&](mlir::LLVM::CallOp) {
          retainedCallInSpatialRegion = true;
          return mlir::WalkResult::interrupt();
        });
      });
  if (retainedCallInSpatialRegion)
    fail("materialized Spatial region retained a general call");

  auto finalized = take(loom::frontend::finalizeSpatialOwnershipCandidate(
      std::move(materialized), design.roots().front()));
  auto dataflow = take(finalized.canonicalDataflow.view());
  bool sawAdd = false;
  bool sawMultiply = false;
  for (const dataflow::CanonicalActorView &actor : dataflow.actors()) {
    auto projection =
        take(dataflow::projectRegisteredActorSchemaProjection(actor.op));
    sawAdd |= projection.schema == dataflow::OperationSchemaId::ArithAddI;
    sawMultiply |= projection.schema == dataflow::OperationSchemaId::ArithMulI;
  }
  if (!sawAdd || !sawMultiply)
    fail("inline candidate lost caller or callee computation");

  auto parentView = take(program.view());
  auto parentCall = take(parentView.resolve(call));
  if (!llvm::isa_and_nonnull<mlir::LLVM::CallOp>(parentCall.operation))
    fail("candidate-local inlining mutated the exact parent");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void exactDirectCallLeafProducesAnAtomicCandidate() {
  auto program = makeProgram();
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "kernel", "helper");

  auto scopes =
      take(loom::frontend::enumerateSpatialOwnershipScopeDomain(program));
  bool acceptedCall = false;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : scopes)
    if (const auto *scope =
            std::get_if<loom::frontend::SpatialOwnershipScope>(&entry))
      acceptedCall |= scope->selection == call;
  if (!acceptedCall)
    fail("exact direct leaf call was not an ownership scope");

  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, call));
  if (decisions.size() != 2 || decisions.front().directCallInlining ||
      !decisions.back().directCallInlining ||
      decisions.back().directCallInlining->callSite != call)
    fail("direct-call leaf decision domain is not canonical");

  auto unresolved =
      loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {call}, decisions.front());
  if (unresolved)
    fail("a direct-call leaf entered a Spatial region without inlining");
  bool classifiedNonFinalizable = false;
  llvm::Error unhandled = llvm::handleErrors(
      unresolved.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        classifiedNonFinalizable =
            error.kind() ==
                loom::frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable &&
            error.message().find("requires its exact inline coordinate") !=
                std::string::npos;
      });
  if (unhandled)
    fail("direct-call leaf refusal escaped candidate classification: " +
         llvm::toString(std::move(unhandled)));
  if (!classifiedNonFinalizable)
    fail("direct-call leaf refusal lacked typed NonFinalizable");

  auto materialized =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {call}, decisions.back()));
  bool sawThreadLaunch = false;
  bool sawCallerMultiply = false;
  bool sawCallInSpatialRegion = false;
  materialized.structuredProgram.module().walk([&](mlir::Operation *operation) {
    sawThreadLaunch |= llvm::isa<dataflow::ThreadLaunchOp>(operation);
    if (auto multiply = llvm::dyn_cast<mlir::arith::MulIOp>(operation))
      sawCallerMultiply |= !multiply->getParentOfType<loom::SpatialRegionOp>();
    if (auto spatial = llvm::dyn_cast<loom::SpatialRegionOp>(operation))
      spatial.walk([&](mlir::LLVM::CallOp) { sawCallInSpatialRegion = true; });
  });
  if (!sawThreadLaunch || !sawCallerMultiply || sawCallInSpatialRegion)
    fail("direct-call leaf did not preserve the exact caller/callee boundary");
}

void exactDirectCallLeafDropsDeadResultsAtTheOwnershipBoundary() {
  auto program = makeProgram();
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "dead_result_caller", "helper");
  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, call));
  const auto inlineDecision = llvm::find_if(
      decisions,
      [](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining.has_value();
      });
  if (inlineDecision == decisions.end())
    fail("dead direct-call result omitted the exact inline coordinate");

  auto materialized =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {call}, *inlineDecision));
  bool sawZeroResultSpatial = false;
  bool sawCallInSpatialRegion = false;
  materialized.structuredProgram.module().walk(
      [&](loom::SpatialRegionOp spatial) {
        sawZeroResultSpatial |= spatial.getValueResults().empty();
        spatial.walk(
            [&](mlir::LLVM::CallOp) { sawCallInSpatialRegion = true; });
      });
  if (!sawZeroResultSpatial || sawCallInSpatialRegion)
    fail("dead direct-call result crossed the ownership boundary");
}

void nestedStructuredScopeRequiresExactActivityObservations() {
  auto program = makeNestedProgram();
  const loom::frontend::StructuredEntityRef loop =
      findFor(program, "nested_kernel");
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "nested_kernel", "nested_helper");

  auto scopes =
      take(loom::frontend::enumerateSpatialOwnershipScopeDomain(program));
  bool acceptedLoop = false;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : scopes)
    if (const auto *scope =
            std::get_if<loom::frontend::SpatialOwnershipScope>(&entry))
      acceptedLoop |= scope->selection == loop;
  if (!acceptedLoop)
    fail("nested inlineable call rejected its structured scope preflight");

  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, loop));
  const auto inlineDecision = llvm::find_if(
      decisions,
      [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining &&
               point.directCallInlining->callSite == call;
      });
  if (inlineDecision == decisions.end())
    fail("nested structured scope omitted its exact inline coordinate");
  auto materialized =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {loop}, *inlineDecision));
  if (!materialized.blockActivityLineage.empty())
    fail("nested direct-call inlining published aggregate callee activity as "
         "an exact candidate projection");
}

void pinnedInlinerRefusalIsCandidateLocal() {
  auto program = makeNoInlineProgram();
  const loom::frontend::StructuredEntityRef kernel =
      findFunction(program, "noinline_kernel");
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "noinline_kernel", "noinline_helper");
  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, kernel));
  const auto inlineDecision = llvm::find_if(
      decisions,
      [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining &&
               point.directCallInlining->callSite == call;
      });
  if (inlineDecision == decisions.end())
    fail("pinned-inliner refusal was hidden from the decision domain");
  auto rejected = loom::frontend::materializeStructuredSpatialOwnershipDecision(
      program, {kernel}, *inlineDecision);
  if (rejected)
    fail("a no-inline callable was inlined");
  bool candidateLocal = false;
  llvm::Error unhandled = llvm::handleErrors(
      rejected.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        candidateLocal =
            error.kind() ==
                loom::frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable &&
            error.message().find("pinned MLIR inliner rejected") !=
                std::string::npos;
      });
  if (unhandled)
    fail("expected pinned-inliner refusal escaped candidate classification: " +
         llvm::toString(std::move(unhandled)));
  if (!candidateLocal)
    fail("pinned-inliner refusal lacked a typed candidate disposition");
}

void callablePreflightRejectsAnUnregisteredIntrinsic() {
  auto program = makeUnregisteredIntrinsicProgram();
  const loom::frontend::StructuredEntityRef directCallable =
      findFunction(program, "read_cycle");
  const loom::frontend::StructuredEntityRef caller =
      findFunction(program, "read_cycle_caller");
  auto scopes =
      take(loom::frontend::enumerateSpatialOwnershipScopeDomain(program));

  bool rejectedDirectCallable = false;
  bool acceptedCaller = false;
  for (const loom::frontend::SpatialOwnershipScopeDomainEntry &entry : scopes) {
    if (const auto *accepted =
            std::get_if<loom::frontend::SpatialOwnershipScope>(&entry)) {
      if (accepted->selection == directCallable)
        fail("an unregistered intrinsic entered the ownership domain");
      acceptedCaller |= accepted->selection == caller;
      continue;
    }
    const auto &rejected =
        std::get<loom::frontend::RejectedSpatialOwnershipScope>(entry);
    if (rejected.scope.selection == directCallable &&
        rejected.message.find("llvm.call_intrinsic") != std::string::npos)
      rejectedDirectCallable = true;
  }
  if (!rejectedDirectCallable)
    fail("callable preflight did not classify the unregistered intrinsic");
  if (!acceptedCaller)
    fail("callee structure removed the caller before its inline decision");

  const loom::frontend::StructuredEntityRef call =
      findCall(program, "read_cycle_caller", "read_cycle_helper");
  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, caller));
  auto inlineDecision = llvm::find_if(
      decisions,
      [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining &&
               point.directCallInlining->callSite == call;
      });
  if (inlineDecision == decisions.end())
    fail("caller omitted the exact inline decision");
  auto rejected = loom::frontend::materializeStructuredSpatialOwnershipDecision(
      program, {caller}, *inlineDecision);
  if (rejected)
    fail("an unregistered intrinsic entered the inlined Spatial candidate");
  bool candidateLocal = false;
  llvm::Error unhandled = llvm::handleErrors(
      rejected.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &error) {
        candidateLocal =
            error.kind() ==
                loom::frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable &&
            error.message().find("llvm.call_intrinsic") != std::string::npos;
      });
  if (unhandled)
    fail("inlined structural rejection escaped candidate classification: " +
         llvm::toString(std::move(unhandled)));
  if (!candidateLocal)
    fail("inlined callee structure lacked a typed candidate disposition");
}

void topLevelCalleeAllocationUsesTheMaterializedGraphFrontier() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-call-allocation", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto program = makeAllocatingCalleeProgram();
  const loom::frontend::StructuredEntityRef caller =
      findFunction(program, "allocating_caller");
  const loom::frontend::StructuredEntityRef call =
      findCall(program, "allocating_caller", "allocating_helper");

  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, caller));
  auto inlineDecision = llvm::find_if(
      decisions,
      [&](const loom::frontend::SpatialOwnershipDecisionPoint &point) {
        return point.directCallInlining &&
               point.directCallInlining->callSite == call;
      });
  if (inlineDecision == decisions.end())
    fail("allocating callee omitted the exact inline decision");
  auto materialized =
      take(loom::frontend::materializeStructuredSpatialOwnershipDecision(
          program, {caller}, *inlineDecision));
  auto finalized = take(loom::frontend::finalizeSpatialOwnershipCandidate(
      std::move(materialized), design.roots().front()));
  auto view = take(finalized.canonicalDataflow.view());
  if (view.graphs().size() != 1 || view.actors().empty())
    fail("top-level callee allocation did not reach a canonical graph");

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void finalLoweringInvariantIsNotCandidatePruning() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-structured-call-final-lowering", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  loom::frontend::MaterializedStructuredOwnershipCandidate bypassedPreflight{
      makeBrokenFinalLoweringProgram(), std::nullopt, {}, {}};
  auto rejected = loom::frontend::finalizeSpatialOwnershipCandidate(
      std::move(bypassedPreflight), design.roots().front());
  if (rejected)
    fail("an unbound dynamic memory service reached Canonical Dataflow");
  bool becameCandidatePruning = false;
  std::string message;
  llvm::Error unhandled = llvm::handleErrors(
      rejected.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &failure) {
        becameCandidatePruning = true;
        message = failure.message();
      });
  if (becameCandidatePruning)
    fail("final lowering invariant became candidate pruning: " + message);
  message = llvm::toString(std::move(unhandled));
  if (message.find("canonical_dataflow_lowering_invalid") == std::string::npos)
    fail("final lowering did not preserve its owner error: " + message);

  error = llvm::sys::fs::remove_directories(directory);
  if (error)
    fail("cannot remove ArtifactStore directory: " + error.message());
}

void selectedDynamicPointerServiceCutIsNonFinalizable() {
  auto program = makeDescriptorPointerLoopProgram();
  const loom::frontend::StructuredEntityRef loop =
      findFor(program, "descriptor_pointer_loop");
  auto decisions = take(
      loom::frontend::enumerateSpatialOwnershipDecisionDomain(program, loop));
  if (decisions.size() != 1)
    fail("descriptor-pointer loop did not have one exact decision");

  auto rejected = loom::frontend::materializeStructuredSpatialOwnershipDecision(
      program, {loop}, decisions.front());
  if (rejected)
    fail("an unbound dynamic pointer service entered a Spatial candidate");
  bool classifiedNonFinalizable = false;
  llvm::Error unhandled = llvm::handleErrors(
      rejected.takeError(),
      [&](const loom::frontend::SpatialOwnershipCandidateRejection &failure) {
        classifiedNonFinalizable =
            failure.kind() ==
                loom::frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable &&
            failure.message().find(
                "no pointer service at the selected Spatial boundary") !=
                std::string::npos;
      });
  if (unhandled)
    fail(
        "dynamic pointer-service rejection escaped candidate classification: " +
        llvm::toString(std::move(unhandled)));
  if (!classifiedNonFinalizable)
    fail("dynamic pointer-service cut lacked typed NonFinalizable");
}

} // namespace

int main() {
  decisionCodecHasStableFraming();
  exactDirectCallSiteProducesAnInlineCandidate();
  exactDirectCallLeafProducesAnAtomicCandidate();
  exactDirectCallLeafDropsDeadResultsAtTheOwnershipBoundary();
  nestedStructuredScopeRequiresExactActivityObservations();
  pinnedInlinerRefusalIsCandidateLocal();
  callablePreflightRejectsAnUnregisteredIntrinsic();
  topLevelCalleeAllocationUsesTheMaterializedGraphFrontier();
  selectedDynamicPointerServiceCutIsNonFinalizable();
  finalLoweringInvariantIsNotCandidatePruning();
  llvm::outs() << "structured call ownership anchor passed\n";
  return EXIT_SUCCESS;
}
