#include "Frontend/Compilation/OwnershipCandidateGenerator.h"

#include "Dataflow/IR/DataflowAttrs.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Frontend/Compilation/FabricCapabilityIndex.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "ownership_candidate_invalid: " + message);
}

std::string typeSpelling(mlir::FunctionType type) {
  std::string spelling;
  llvm::raw_string_ostream stream(spelling);
  stream << type;
  return spelling;
}

std::string uniqueSymbol(mlir::ModuleOp module, llvm::StringRef prefix,
                         llvm::StringRef sourceName) {
  std::string base = (llvm::Twine(prefix) + sourceName).str();
  std::string candidate = base;
  for (std::uint64_t suffix = 0; module.lookupSymbol(candidate); ++suffix)
    candidate = (llvm::Twine(base) + "_" + llvm::Twine(suffix)).str();
  return candidate;
}

llvm::Error verifyEligibleCallable(mlir::LLVM::LLVMFuncOp function) {
  if (function.isExternal())
    return invalid("selected callable has no definition");
  if (function.isVarArg())
    return invalid("variadic callable ownership is not materialized");
  if (!llvm::isa<mlir::LLVM::LLVMVoidType>(
          function.getFunctionType().getReturnType()))
    return invalid("whole-callable ownership currently requires void return");
  if (!function.getBody().hasOneBlock())
    return invalid("whole-callable ownership requires one structured block");

  mlir::Block &body = function.getBody().front();
  auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 0)
    return invalid("selected callable must return void directly");

  mlir::Operation *nestedCall = nullptr;
  function.getBody().walk([&](mlir::Operation *operation) {
    if (llvm::isa<mlir::LLVM::CallOp, mlir::LLVM::InvokeOp>(operation)) {
      nestedCall = operation;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (nestedCall)
    return invalid("selected callable contains an unresolved nested call");
  return llvm::Error::success();
}

llvm::Expected<dataflow::ThreadOp>
materializeThread(mlir::ModuleOp module, mlir::LLVM::LLVMFuncOp function) {
  if (llvm::Error error = verifyEligibleCallable(function))
    return std::move(error);
  mlir::MLIRContext *context = module.getContext();
  mlir::OpBuilder builder(context);
  mlir::Location location = function.getLoc();
  const std::string threadName =
      uniqueSymbol(module, "__loom_thread_", function.getSymName());
  const std::string graphName =
      uniqueSymbol(module, "__loom_graph_", function.getSymName());

  mlir::Block &source = function.getBody().front();
  llvm::SmallVector<mlir::Type, 8> inputTypes(source.getArgumentTypes());
  builder.setInsertionPointToEnd(module.getBody());
  auto thread = dataflow::ThreadOp::create(
      builder, location, threadName,
      builder.getFunctionType(inputTypes, mlir::TypeRange{}),
      dataflow::ThreadDomainAttr::get(context));
  thread.setSymVisibilityAttr(builder.getStringAttr("private"));

  llvm::SmallVector<mlir::DictionaryAttr, 8> argumentAttrs;
  argumentAttrs.reserve(inputTypes.size());
  for (std::size_t index = 0; index < inputTypes.size(); ++index)
    argumentAttrs.push_back(
        mlir::function_interface_impl::getArgAttrDict(function, index));
  mlir::function_interface_impl::setAllArgAttrDicts(thread, argumentAttrs);

  mlir::Block *threadEntry = builder.createBlock(&thread.getBody());
  for (mlir::Type type : inputTypes)
    threadEntry->addArgument(type, location);
  threadEntry->addArgument(builder.getNoneType(), location);

  llvm::SmallVector<mlir::Value, 8> values;
  llvm::SmallVector<mlir::Value, 8> memories;
  llvm::SmallVector<std::size_t, 8> spatialArgument(inputTypes.size());
  for (auto [index, argument] : llvm::enumerate(
           threadEntry->getArguments().take_front(inputTypes.size()))) {
    if (dataflow::DataflowDialect::isMemoryCapabilityType(argument.getType())) {
      spatialArgument[index] = memories.size();
      memories.push_back(argument);
      continue;
    }
    if (dataflow::DataflowDialect::containsMemoryCapability(argument.getType()))
      return invalid(
          "callable input embeds an unmaterialized memory capability");
    spatialArgument[index] = values.size();
    values.push_back(argument);
  }
  for (std::size_t index = 0; index < inputTypes.size(); ++index)
    if (dataflow::DataflowDialect::isMemoryCapabilityType(inputTypes[index]))
      spatialArgument[index] += values.size();

  builder.setInsertionPointToStart(threadEntry);
  auto spatial = loom::SpatialRegionOp::create(
      builder, location, values, mlir::ValueRange{}, memories,
      mlir::ValueRange{}, mlir::TypeRange{}, mlir::TypeRange{},
      builder.getArrayAttr({}), builder.getStringAttr(graphName));
  mlir::Block *spatialEntry = builder.createBlock(&spatial.getBody());
  for (mlir::Type type : spatial.getOperandTypes())
    spatialEntry->addArgument(type, location);

  mlir::IRMapping mapping;
  for (auto [index, argument] : llvm::enumerate(source.getArguments()))
    mapping.map(argument, spatialEntry->getArgument(spatialArgument[index]));
  builder.setInsertionPointToEnd(spatialEntry);
  for (mlir::Operation &operation : source.without_terminator())
    builder.clone(operation, mapping);
  loom::SpatialYieldOp::create(builder, location, mlir::ValueRange{},
                               mlir::ValueRange{});

  builder.setInsertionPointToEnd(threadEntry);
  dataflow::ThreadYieldOp::create(builder, location, mlir::ValueRange{});

  while (!source.empty())
    source.back().erase();
  builder.setInsertionPointToStart(&source);
  mlir::FlatSymbolRefAttr callee =
      mlir::FlatSymbolRefAttr::get(context, threadName);
  auto launch = dataflow::ThreadLaunchOp::create(
      builder, location, callee, source.getArguments(), mlir::ValueRange{},
      mlir::ValueRange{});
  dataflow::ThreadWaitOp::create(builder, location,
                                 mlir::ValueRange{launch.getAsyncToken()});
  mlir::LLVM::ReturnOp::create(builder, location, mlir::ValueRange{});
  return thread;
}

llvm::Error requireExactFabricCapabilities(
    const dataflow::CanonicalDataflowArtifact &program,
    const fabric::FinalizedFabricRoot &fabric) {
  auto view = program.view();
  if (!view)
    return view.takeError();
  if (view->graphs().empty() || view->actors().empty())
    return invalid("materialized candidate has no SpatialCore workload");

  FabricCapabilityIndex capabilities(fabric.view());
  for (const dataflow::CanonicalActorView &actor : view->actors()) {
    auto projection =
        dataflow::projectRegisteredActorSchemaProjection(actor.op);
    if (!projection)
      return projection.takeError();
    if (actor.kind == dataflow::CanonicalDataflowActorKind::Memory) {
      auto resources = capabilities.admittingMemoryResources(actor.op);
      if (!resources)
        return resources.takeError();
      if (resources->empty())
        return invalid("exact Fabric admits no memory resource for actor " +
                       dataflow::operationSchemaSpelling(projection->schema));
      continue;
    }
    auto resources = capabilities.admittingOperationResources(actor.op);
    if (!resources)
      return resources.takeError();
    if (resources->empty())
      return invalid("exact Fabric admits no operation resource for actor " +
                     dataflow::operationSchemaSpelling(projection->schema) +
                     " with type " + typeSpelling(projection->type));
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<MaterializedOwnershipCandidate>
materializeWholeCallableSpatialOwnership(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &callable,
    const fabric::FinalizedFabricRoot &fabric,
    const WholeCallableSpatialOwnershipOptions &options) {
  auto parentView = parent.view();
  if (!parentView)
    return parentView.takeError();
  auto parentEntity = parentView->resolve(callable);
  if (!parentEntity)
    return parentEntity.takeError();
  if (!llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(parentEntity->operation))
    return invalid("selected StructuredEntityRef is not an LLVM callable");

  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone()));
  auto cloneView =
      buildStructuredProgramCandidateView(clone.get(), parent.identity());
  if (!cloneView)
    return cloneView.takeError();
  auto clonedEntity = cloneView->resolve(callable);
  if (!clonedEntity)
    return clonedEntity.takeError();
  auto function =
      llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(clonedEntity->operation);
  if (!function)
    return invalid("selected callable changed kind in the private clone");

  if (options.fmuladdExecutionShape) {
    mlir::PassManager materialization(
        function.getContext(), mlir::LLVM::LLVMFuncOp::getOperationName());
    materialization.enableVerifier(options.lowering.verifyEach);
    materialization.addPass(
        raising::createMaterializeFMulAddPass(*options.fmuladdExecutionShape));
    if (mlir::failed(materialization.run(function.getOperation())))
      return invalid("selected fmuladd execution shape is not materializable");
  }

  if (auto thread = materializeThread(clone.get(), function); !thread)
    return thread.takeError();
  if (mlir::failed(mlir::verify(clone.get())))
    return invalid("materialized Structured Program does not verify");

  auto structured = finalizeStructuredProgram(clone.get());
  if (!structured)
    return structured.takeError();
  auto canonical =
      lowering::lowerStructuredProgramToCanonicalDataflow(*structured,
                                                           options.lowering);
  if (!canonical)
    return canonical.takeError();
  if (llvm::Error error = requireExactFabricCapabilities(*canonical, fabric))
    return std::move(error);
  return MaterializedOwnershipCandidate{std::move(*structured),
                                        std::move(*canonical)};
}

} // namespace loom::frontend
