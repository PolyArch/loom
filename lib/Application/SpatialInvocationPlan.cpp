#include "ExecutionGlue.h"

#include "Dataflow/IR/DataflowCanonicalEntity.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Simulator/SimulationArtifacts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::application::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_execution_glue_invalid: " + message);
}

llvm::Expected<std::uint32_t>
transportBitCount(sim::SpatialSimulationValueShape shape) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint32_t>::max() / shape.laneBitWidth)
    return invalid("Spatial value shape exceeds the invocation ABI");
  return static_cast<std::uint32_t>(shape.lanesPerToken) * shape.laneBitWidth;
}

llvm::Expected<std::vector<std::uint32_t>>
transportBitCounts(llvm::ArrayRef<sim::SpatialSimulationValueShape> shapes) {
  std::vector<std::uint32_t> result;
  result.reserve(shapes.size());
  for (sim::SpatialSimulationValueShape shape : shapes) {
    auto count = transportBitCount(shape);
    if (!count)
      return count.takeError();
    result.push_back(*count);
  }
  return result;
}

llvm::Error verifySelectedCallableBoundary(
    const dataflow::CanonicalRootThreadLaunchView &rootView,
    dataflow::GraphLaunchOp graphLaunch,
    llvm::ArrayRef<std::uint32_t> valueBitCounts,
    llvm::ArrayRef<std::uint32_t> resultBitCounts,
    std::vector<std::uint64_t> &resultRootOperandOrdinals,
    std::string &sourceCallableSymbol) {
  auto rootLaunch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootView.op);
  auto thread = llvm::dyn_cast<dataflow::ThreadOp>(rootView.callee);
  auto callable = rootView.op
                      ? rootView.op->getParentOfType<mlir::LLVM::LLVMFuncOp>()
                      : mlir::LLVM::LLVMFuncOp{};
  if (!rootLaunch || !thread || !callable || callable.isExternal() ||
      !callable.getBody().hasOneBlock())
    return invalid("root launch is not owned by one defined LLVM callable");
  if (!rootLaunch.getAsyncDependencies().empty())
    return invalid("initial dispatch requires a dependency-free root");
  if (callable.getFunctionType().isVarArg())
    return invalid("initial dispatch callable boundary is variadic");
  if (graphLaunch.getValueInputs().size() != valueBitCounts.size() ||
      graphLaunch.getValueResults().size() != resultBitCounts.size())
    return invalid("root and graph value boundaries are not exact");

  mlir::Block &threadBlock = thread.getBody().front();
  if (!rootLaunch.getAsyncToken().hasOneUse())
    return invalid("source callable has no unique root retirement wait");
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
      rootLaunch.getAsyncToken().use_begin()->getOwner());
  if (!wait || wait->getBlock() != rootLaunch->getBlock() ||
      !rootLaunch->isBeforeInBlock(wait))
    return invalid("source callable does not retire the selected root");

  resultRootOperandOrdinals.clear();
  resultRootOperandOrdinals.reserve(resultBitCounts.size());
  for (mlir::Value graphResult : graphLaunch.getValueResults()) {
    if (!graphResult.hasOneUse())
      return invalid("graph value result has no unique publication");
    auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(
        graphResult.use_begin()->getOwner());
    auto threadResultSlot =
        store ? llvm::dyn_cast<mlir::BlockArgument>(store.getAddr())
              : mlir::BlockArgument{};
    if (!store || store.getValue() != graphResult || !threadResultSlot ||
        threadResultSlot.getOwner() != &threadBlock ||
        threadResultSlot.getArgNumber() >= rootLaunch.getBodyOperands().size())
      return invalid(
          "graph value result is not stored through its thread slot");
    mlir::Value callerResultSlot =
        rootLaunch.getBodyOperands()[threadResultSlot.getArgNumber()];
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(callerResultSlot.getType()))
      return invalid("thread result slot is not a pointer");
    resultRootOperandOrdinals.push_back(threadResultSlot.getArgNumber());
  }

  sourceCallableSymbol = callable.getSymName().str();
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::uint64_t>
ApplicationSpatialInvocationPlan::Launch::dispatchOperandOrdinal(
    std::uint64_t rootOperandOrdinal) const {
  auto found = llvm::find(dispatchRootOperandOrdinals, rootOperandOrdinal);
  if (found == dispatchRootOperandOrdinals.end())
    return invalid("root operand is absent from the dispatch ABI");
  return static_cast<std::uint64_t>(
      std::distance(dispatchRootOperandOrdinals.begin(), found));
}

llvm::Expected<ApplicationSpatialInvocationPlan>
deriveApplicationSpatialInvocationPlan(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::StringRef entrySymbol) {
  auto roots =
      dataflow.projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!roots)
    return roots.takeError();
  if (roots->empty())
    return invalid("initial dispatch requires at least one reachable root");

  struct LaunchBoundary final {
    ApplicationSpatialInvocationPlan::Launch launch;
    dataflow::ThreadLaunchOp rootLaunch;
    dataflow::GraphLaunchOp graphLaunch;
    std::string callableSymbol;
  };
  std::vector<LaunchBoundary> boundaries;
  boundaries.reserve(roots->size());
  for (dataflow::RootThreadLaunchRef root : *roots) {
    std::vector<dataflow::RootedGraphLaunchRef> graphs;
    dataflow.forEachRootedGraphLaunch(
        [&](dataflow::RootedGraphLaunchRef graph) {
          if (graph.rootThreadLaunch == root)
            graphs.push_back(graph);
        });
    if (graphs.size() != 1)
      return invalid("each dynamic root must own exactly one graph launch");
    auto rootView = dataflow.resolve(root);
    if (!rootView)
      return rootView.takeError();
    auto graphView = dataflow.resolve(graphs.front().staticGraphLaunch);
    if (!graphView)
      return graphView.takeError();
    auto rootLaunch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(rootView->op);
    auto graphLaunch = llvm::dyn_cast<dataflow::GraphLaunchOp>(graphView->op);
    if (!rootLaunch || !graphLaunch || !graphLaunch.getMemoryResults().empty())
      return invalid(
          "dynamic invocation requires an imported-memory graph boundary");
    auto shapes =
        sim::projectSpatialSimulationBoundaryShapes(dataflow, graphs.front());
    if (!shapes)
      return shapes.takeError();
    auto coordinates = dataflow.enumerateStaticDenseCoordinates(
        graphs.front(), runtime::gem5MaximumDynamicSpatialInvocations,
        entrySymbol);
    if (!coordinates)
      return coordinates.takeError();
    if (!*coordinates || (*coordinates)->empty())
      return invalid(
          "dynamic invocation requires a finite nonempty dense domain");
    auto valueBitCounts = transportBitCounts(shapes->valueInputs);
    auto resultBitCounts = transportBitCounts(shapes->valueResults);
    if (!valueBitCounts || !resultBitCounts)
      return llvm::joinErrors(valueBitCounts ? llvm::Error::success()
                                             : valueBitCounts.takeError(),
                              resultBitCounts ? llvm::Error::success()
                                              : resultBitCounts.takeError());
    std::vector<std::uint64_t> resultRootOperandOrdinals;
    std::string callableSymbol;
    if (llvm::Error error = verifySelectedCallableBoundary(
            *rootView, graphLaunch, *valueBitCounts, *resultBitCounts,
            resultRootOperandOrdinals, callableSymbol))
      return std::move(error);
    std::vector<std::uint64_t> dispatchRootOperandOrdinals;
    for (const auto operand : llvm::enumerate(rootLaunch.getBodyOperands()))
      if (!llvm::isa<dataflow::ChannelType>(operand.value().getType()))
        dispatchRootOperandOrdinals.push_back(operand.index());
    std::vector<ApplicationSpatialInvocationPlan::Launch::Point> points;
    points.reserve((*coordinates)->size());
    for (std::vector<std::uint64_t> &point : **coordinates)
      points.push_back({0, std::move(point)});
    boundaries.push_back({{root,
                           graphs.front(),
                           std::move(points),
                           std::move(dispatchRootOperandOrdinals),
                           std::move(*valueBitCounts),
                           std::move(*resultBitCounts),
                           std::move(resultRootOperandOrdinals),
                           {}},
                          rootLaunch,
                          graphLaunch,
                          std::move(callableSymbol)});
  }
  llvm::sort(
      boundaries, [](const LaunchBoundary &lhs, const LaunchBoundary &rhs) {
        return std::tuple(lhs.launch.root.entity.value(),
                          lhs.launch.graph.staticGraphLaunch.entity.value()) <
               std::tuple(rhs.launch.root.entity.value(),
                          rhs.launch.graph.staticGraphLaunch.entity.value());
      });
  std::uint64_t nextDispatchTarget = 0;
  for (LaunchBoundary &boundary : boundaries)
    for (ApplicationSpatialInvocationPlan::Launch::Point &point :
         boundary.launch.points)
      point.dispatchTargetOrdinal = nextDispatchTarget++;

  std::map<std::string, std::vector<std::uint64_t>> launchesByCallable;
  for (const auto indexed : llvm::enumerate(boundaries))
    launchesByCallable[indexed.value().callableSymbol].push_back(
        indexed.index());
  std::vector<ApplicationSpatialInvocationPlan::Callable> callables;
  callables.reserve(launchesByCallable.size());
  for (auto &[symbol, launchOrdinals] : launchesByCallable)
    callables.push_back({std::move(symbol), std::move(launchOrdinals)});

  for (const ApplicationSpatialInvocationPlan::Callable &callable : callables) {
    auto paths = dataflow.projectRootThreadInvocationPathsFromAbiEntry(
        entrySymbol, boundaries[callable.launchOrdinals.front()].launch.root);
    if (!paths)
      return paths.takeError();
    std::vector<std::pair<std::string, std::uint64_t>> leafLocators;
    for (const auto callableLaunchIndexed :
         llvm::enumerate(callable.launchOrdinals)) {
      LaunchBoundary &boundary = boundaries[callableLaunchIndexed.value()];
      ApplicationSpatialInvocationPlan::Launch &launch = boundary.launch;
      launch.sites.reserve(paths->size());
      for (const auto pathIndexed : llvm::enumerate(*paths)) {
        llvm::SmallVector<mlir::LLVM::CallOp, 4> path;
        path.reserve(pathIndexed.value().calls.size());
        for (mlir::Operation *operation : pathIndexed.value().calls) {
          auto call = llvm::dyn_cast_or_null<mlir::LLVM::CallOp>(operation);
          if (!call)
            return invalid("canonical invocation path contains a non-call");
          path.push_back(call);
        }
        auto capture = sim::deriveOperationSimulationInputCapturePlan(
            dataflow, launch.graph, boundary.rootLaunch.getBodyOperands(),
            boundary.graphLaunch.getValueResults(), path);
        if (!capture)
          return capture.takeError();
        if (capture->invocationPath.empty())
          return invalid("dynamic invocation capture has no call locator");
        const sim::DirectCallCaptureSite &leaf = capture->invocationPath.back();
        const std::pair<std::string, std::uint64_t> leafLocator{
            leaf.hostCallerSymbol, leaf.hostCallOrdinal};
        if (callableLaunchIndexed.index() == 0) {
          if (llvm::is_contained(leafLocators, leafLocator))
            return invalid(
                "one dynamic invocation call is reachable through multiple "
                "paths");
          leafLocators.push_back(leafLocator);
        } else if (pathIndexed.index() >= leafLocators.size() ||
                   leafLocators[pathIndexed.index()] != leafLocator) {
          return invalid("dynamic roots disagree on their invocation sites");
        }
        if (capture->input.valueInputs.size() != launch.valueBitCounts.size() ||
            capture->input.valueResults.size() != launch.resultBitCounts.size())
          return invalid(
              "dynamic invocation capture differs from graph boundary");
        for (const sim::SimulationValueInputCapture &input :
             capture->input.valueInputs) {
          if (input.unusedByGraph &&
              (!input.fixedValue ||
               llvm::any_of(input.fixedValue->lanes, [](const auto &lane) {
                 return lane.state != sim::SemanticState::Defined ||
                        lane.pointerTarget.has_value();
               })))
            return invalid(
                "graph-unobserved capture does not carry a defined scalar "
                "wire value");
          if (!input.fixedValue)
            continue;
          if (llvm::any_of(input.fixedValue->lanes, [](const auto &lane) {
                return lane.state != sim::SemanticState::Defined ||
                       lane.pointerTarget.has_value();
              }))
            return invalid(
                "fixed invocation value is not representable on the wire: "
                "undef, poison, or pointer lane");
        }

        std::vector<mlir::Value> boundObjectBases(
            capture->input.objects.size());
        std::vector<bool> objectHasRoot(capture->input.objects.size(), false);
        std::vector<ApplicationSpatialInvocationPlan::MemoryRootSource>
            memoryRootSources;
        memoryRootSources.reserve(capture->input.memoryRootBindings.size());
        for (const sim::SimulationMemoryRootCapture &binding :
             capture->input.memoryRootBindings) {
          if (binding.objectIndex >= boundObjectBases.size())
            return invalid("dynamic invocation memory root exceeds objects");
          objectHasRoot[binding.objectIndex] = true;
          const sim::SimulationMemoryCaptureObject &object =
              capture->input.objects[binding.objectIndex];
          auto operand = llvm::find(boundary.rootLaunch.getBodyOperands(),
                                    binding.boundaryPointer);
          if (operand == boundary.rootLaunch.getBodyOperands().end())
            return invalid(
                "dynamic invocation memory root is not a root operand");
          const std::uint64_t rootOperandOrdinal =
              static_cast<std::uint64_t>(std::distance(
                  boundary.rootLaunch.getBodyOperands().begin(), operand));
          auto dispatchOrdinal =
              launch.dispatchOperandOrdinal(rootOperandOrdinal);
          if (!dispatchOrdinal)
            return dispatchOrdinal.takeError();
          memoryRootSources.push_back({*dispatchOrdinal, binding.objectIndex});
          if (!object.baseBindingCallOrdinal)
            continue;
          mlir::Value base = binding.boundaryPointer;
          while (base) {
            if (auto gep = base.getDefiningOp<mlir::LLVM::GEPOp>()) {
              base = gep.getBase();
              continue;
            }
            if (auto cast = base.getDefiningOp<mlir::LLVM::BitcastOp>()) {
              base = cast.getArg();
              continue;
            }
            if (auto cast = base.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>()) {
              base = cast.getArg();
              continue;
            }
            if (auto cast =
                    base.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
              if (cast.getInputs().size() != 1)
                return invalid(
                    "bound invocation memory base has a non-unary cast");
              base = cast.getInputs().front();
              continue;
            }
            break;
          }
          auto argument = llvm::dyn_cast<mlir::BlockArgument>(base);
          auto owner = argument
                           ? llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
                                 argument.getOwner()->getParentOp())
                           : mlir::LLVM::LLVMFuncOp{};
          if (!argument || !owner ||
              owner.getSymName() != callable.sourceCallableSymbol)
            return invalid(
                "bound invocation memory base does not reach a selected "
                "callable argument");
          if (boundObjectBases[binding.objectIndex] &&
              boundObjectBases[binding.objectIndex] != base)
            return invalid(
                "bound invocation memory object has conflicting callable "
                "bases");
          boundObjectBases[binding.objectIndex] = base;
        }
        std::vector<ApplicationSpatialInvocationPlan::MemoryObjectSource>
            memoryObjectSources;
        memoryObjectSources.reserve(capture->input.objects.size());
        std::uint64_t nextLocalBaseArgument =
            launch.dispatchRootOperandOrdinals.size();
        for (const auto objectIndexed :
             llvm::enumerate(capture->input.objects)) {
          if (!objectHasRoot[objectIndexed.index()])
            return invalid(
                "dynamic invocation memory object has no logical root");
          const sim::SimulationMemoryCaptureObject &object =
              objectIndexed.value();
          mlir::Value base = object.baseBindingCallOrdinal
                                 ? boundObjectBases[objectIndexed.index()]
                                 : object.base;
          if (!base)
            return invalid(
                "dynamic invocation memory object has no callable base");
          memoryObjectSources.push_back(
              {nextLocalBaseArgument++, object.operandByteOffset, base});
        }
        std::vector<runtime::SpatialInvocationValueLayout> valueLayouts;
        valueLayouts.reserve(launch.valueBitCounts.size());
        for (const auto indexed : llvm::enumerate(capture->input.valueInputs)) {
          const sim::SimulationValueInputCapture &input = indexed.value();
          const std::uint64_t expectedByteCount =
              (launch.valueBitCounts[indexed.index()] + 7) / 8;
          const bool byteCountMatches =
              input.fixedValue ? input.byteCount == 0
                               : input.byteCount == expectedByteCount;
          if (input.valueInputOrdinal != indexed.index() || !byteCountMatches)
            return invalid(
                llvm::Twine("dynamic invocation value capture is not exact: ") +
                "root=" + llvm::Twine(boundary.launch.root.entity.value()) +
                " graph=" +
                llvm::Twine(launch.graph.staticGraphLaunch.entity.value()) +
                " path=" + llvm::Twine(pathIndexed.index()) +
                " input=" + llvm::Twine(indexed.index()) +
                " captured_ordinal=" + llvm::Twine(input.valueInputOrdinal) +
                " captured_bytes=" + llvm::Twine(input.byteCount) +
                " expected_bytes=" +
                llvm::Twine(input.fixedValue ? 0 : expectedByteCount));
          if (!input.fixedValue) {
            const bool rootOperand = input.boundaryOperandOrdinal.has_value();
            const bool coordinate = input.denseCoordinateDimension.has_value();
            if (rootOperand == coordinate)
              return invalid(
                  "dynamic invocation value does not have one source");
            if (rootOperand && *input.boundaryOperandOrdinal >=
                                   boundary.rootLaunch.getBodyOperands().size())
              return invalid("dynamic invocation value has no root operand");
            if (coordinate && *input.denseCoordinateDimension >=
                                  launch.points.front().denseCoordinates.size())
              return invalid("dynamic invocation coordinate is out of range");
          }
          std::optional<runtime::SpatialInvocationPointerTarget> pointerTarget;
          if (input.pointerTarget) {
            const std::uint64_t rootOrdinal =
                input.pointerTarget->memoryRootBindingOrdinal;
            if (rootOrdinal >= capture->input.memoryRootBindings.size())
              return invalid("dynamic invocation pointer target exceeds roots");
            const sim::SimulationMemoryRootCapture &binding =
                capture->input.memoryRootBindings[rootOrdinal];
            if (binding.objectIndex > std::numeric_limits<std::uint32_t>::max())
              return invalid("dynamic invocation object ordinal exceeds ABI");
            pointerTarget = runtime::SpatialInvocationPointerTarget{
                static_cast<std::uint32_t>(binding.objectIndex),
                binding.byteOffset};
          }
          valueLayouts.push_back(
              {launch.valueBitCounts[indexed.index()], pointerTarget});
        }
        std::vector<runtime::SpatialInvocationMemoryObjectLayout> objectLayouts;
        objectLayouts.reserve(capture->input.objects.size());
        for (const sim::SimulationMemoryCaptureObject &object :
             capture->input.objects)
          objectLayouts.push_back({object.byteCount});
        std::vector<runtime::SpatialInvocationMemoryRootBinding> rootBindings;
        rootBindings.reserve(capture->input.memoryRootBindings.size());
        for (const sim::SimulationMemoryRootCapture &binding :
             capture->input.memoryRootBindings) {
          if (binding.objectIndex > std::numeric_limits<std::uint32_t>::max())
            return invalid("dynamic invocation object ordinal exceeds ABI");
          rootBindings.push_back(
              {binding.root.entity.value(),
               static_cast<std::uint32_t>(binding.objectIndex),
               binding.byteOffset});
        }
        std::vector<runtime::SpatialInvocationWireLayout> pointWireLayouts;
        pointWireLayouts.reserve(launch.points.size());
        for (const ApplicationSpatialInvocationPlan::Launch::Point &point :
             launch.points) {
          runtime::SpatialInvocationWireLayout wireLayout;
          std::string diagnostic;
          if (!runtime::projectSpatialInvocationWireLayout(
                  dataflow.identity().bytes(), launch.root.entity.value(),
                  launch.graph.staticGraphLaunch.entity.value(),
                  point.denseCoordinates, valueLayouts, objectLayouts,
                  rootBindings, launch.resultBitCounts, wireLayout, diagnostic))
            return invalid(diagnostic);
          if (wireLayout.templateBytes.size() >
              std::numeric_limits<std::uint32_t>::max())
            return invalid(
                "invocation wire exceeds the dispatch size register");
          pointWireLayouts.push_back(std::move(wireLayout));
        }
        launch.sites.push_back(
            {std::move(*capture), std::move(memoryObjectSources),
             std::move(memoryRootSources), std::move(pointWireLayouts)});
      }
    }
  }
  std::vector<ApplicationSpatialInvocationPlan::Launch> launches;
  launches.reserve(boundaries.size());
  for (LaunchBoundary &boundary : boundaries)
    launches.push_back(std::move(boundary.launch));
  return ApplicationSpatialInvocationPlan{std::move(launches),
                                          std::move(callables)};
}

} // namespace loom::application::detail
