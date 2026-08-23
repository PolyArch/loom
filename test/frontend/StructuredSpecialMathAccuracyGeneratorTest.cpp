#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/SpecialMathAccuracy.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "structuredSpecialMathAccuracy: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireFailure(
    llvm::Expected<loom::frontend::MaterializedStructuredSpecialMathCandidate>
        value,
    llvm::StringRef message) {
  if (value)
    fail("invalid special-math decision unexpectedly succeeded");
  std::string error = llvm::toString(value.takeError());
  if (!llvm::StringRef(error).contains(message))
    fail("unexpected rejection: " + error);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *result = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, loom::LoomDialect,
                    mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                    mlir::math::MathDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *result;
}

loom::frontend::StructuredProgramCandidate parseProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @host_math(%x: f32) -> f32 {
    %token = dataflow.thread.launch @selected(%x) :
        (f32) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    %result = math.tan %x fastmath<afn> : f32
    llvm.return %result : f32
  }

  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %input: f32) ctrl (%start: none) {
    %result = "loom.spatial_region"(%input)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%value: f32):
        %strict = math.sin %value : f32
        %approximate = math.cos %strict fastmath<afn> : f32
        "loom.spatial_yield"(%approximate)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "special_math", source_maps = []} : (f32) -> f32
    dataflow.thread.yield
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse special-math fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate parseApproximatePairProgram() {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func @host_passthrough(%x: f32) -> f32 {
    %token = dataflow.thread.launch @selected(%x) :
        (f32) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return %x : f32
  }

  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %input: f32) ctrl (%start: none) {
    %result = "loom.spatial_region"(%input)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%value: f32):
        %sin = math.sin %value fastmath<afn> : f32
        %cos = math.cos %sin fastmath<afn> : f32
        "loom.spatial_yield"(%cos)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "approximate_pair", source_maps = []} : (f32) -> f32
    dataflow.thread.yield
  }
}
)mlir",
                                                        &context());
  if (!module)
    fail("cannot parse approximate special-math pair fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::frontend::StructuredProgramCandidate
parseSingleSpecialMathProgram(bool approximationPermitted) {
  const llvm::StringRef fastMath =
      approximationPermitted ? " fastmath<afn>" : "";
  std::string source = R"mlir(
module {
  llvm.func @host_passthrough(%x: f32) -> f32 {
    %token = dataflow.thread.launch @selected(%x) :
        (f32) -> !dataflow.thread_token
    dataflow.thread.wait %token : !dataflow.thread_token
    llvm.return %x : f32
  }

  dataflow.thread private @selected domain(#dataflow.thread_domain<dense>)(
      %input: f32) ctrl (%start: none) {
    %result = "loom.spatial_region"(%input)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,
          resultSegmentSizes = array<i32: 1, 0>}> ({
      ^bb0(%value: f32):
        %sin = math.sin %value)mlir";
  source += fastMath;
  source += R"mlir( : f32
        "loom.spatial_yield"(%sin)
            <{operandSegmentSizes = array<i32: 1, 0>}> : (f32) -> ()
    }) {graph_name = "single_special_math", source_maps = []} :
        (f32) -> f32
    dataflow.thread.yield
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail("cannot parse single special-math fixture");
  return take(loom::frontend::finalizeStructuredProgram(module.get()));
}

loom::fabric::FinalizedFabricRoot
makeSinFabric(loom::ArtifactStore &store, fabric::FloatFormat format,
              loom::SpecialMathAccuracyTier guarantee) {
  loom::adg::DesignBuilder design(store);
  const loom::adg::PortType bits128 = take(loom::adg::PortType::bits(128));
  const std::vector<loom::adg::PortType> boundaryTypes = {
      bits128, bits128, bits128, bits128, bits128};
  auto core =
      take(design.createSpatialCore("sin", boundaryTypes, boundaryTypes));
  std::vector<loom::adg::SpatialValue> coreInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryTypes.size(); ++ordinal)
    coreInputs.push_back(take(core.input(ordinal)));
  auto pe = take(core.addPe(
      coreInputs, loom::adg::PeSpec::spatial(boundaryTypes, boundaryTypes)));
  auto fu = take(
      pe.addFu({take(pe.input(0))}, loom::adg::FuSpec{{bits128}, {bits128}}));
  fabric::FloatBehaviorProfile behavior =
      fabric::FloatBehaviorProfile::strictIEEE();
  if (guarantee != loom::SpecialMathAccuracyTier::CorrectlyRounded)
    behavior.requiredFastMath = mlir::arith::FastMathFlags::afn;
  auto operation = take(fu.addOperation(
      {take(fu.input(0))},
      loom::adg::OperationCapabilitySpec{
          fabric::ImplementationFamilyId::ScalarMathSin,
          fabric::ScalarSpecialMathParams{fabric::FloatFormatSet::get({format}),
                                          behavior, guarantee},
          {dataflow::OperationSchemaId::MathSin},
          {bits128},
          fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error = fu.addCapabilityTemplate({{operation}, {}}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(operation.output(0))}))
    fail(llvm::toString(std::move(error)));

  std::vector<loom::adg::PeValue> syncInputs;
  const std::vector<loom::adg::PortType> syncTypes(4, bits128);
  for (std::size_t ordinal = 1; ordinal != boundaryTypes.size(); ++ordinal)
    syncInputs.push_back(take(pe.input(ordinal)));
  auto syncFu =
      take(pe.addFu(syncInputs, loom::adg::FuSpec{syncTypes, syncTypes}));
  std::vector<loom::adg::FuValue> syncOperationInputs;
  for (std::size_t ordinal = 0; ordinal != syncTypes.size(); ++ordinal)
    syncOperationInputs.push_back(take(syncFu.input(ordinal)));
  auto sync = take(syncFu.addOperation(
      syncOperationInputs,
      loom::adg::OperationCapabilitySpec{
          fabric::ImplementationFamilyId::TokenSync,
          fabric::RoutedTokenParams{128, 4},
          {dataflow::OperationSchemaId::DataflowSync},
          syncTypes,
          fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error = syncFu.addCapabilityTemplate({{sync}, {}}))
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> syncOutputs;
  for (std::size_t ordinal = 0; ordinal != syncTypes.size(); ++ordinal)
    syncOutputs.push_back(take(sync.output(ordinal)));
  if (llvm::Error error = syncFu.close(syncOutputs))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> coreOutputs;
  for (std::size_t ordinal = 0; ordinal != boundaryTypes.size(); ++ordinal)
    coreOutputs.push_back(take(pe.output(ordinal)));
  if (llvm::Error error = core.close(coreOutputs))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  return take(loom::fabric::importEntireFabricRoot(
      finalized.roots().front().reference(), store));
}

template <typename Op>
Op findOperation(const loom::frontend::StructuredProgramCandidate &candidate,
                 bool selectedSpatial) {
  Op result;
  candidate.module().walk([&](Op operation) {
    const bool selected = static_cast<bool>(
        operation->template getParentOfType<loom::SpatialRegionOp>());
    if (selected == selectedSpatial)
      result = operation;
  });
  if (!result)
    fail("fixture operation does not resolve");
  return result;
}

void canonicalDomainAndScopedMaterialization() {
  using loom::SpecialMathAccuracyTier;
  using loom::frontend::StructuredSpecialMathAccuracyDecision;

  auto current = parseProgram();
  auto sourceStrict = findOperation<mlir::math::SinOp>(current, true);
  auto sourceSchema = dataflow::operationSchemaOf(sourceStrict);
  if (!sourceSchema ||
      dataflow::semanticsCase(*sourceSchema) !=
          dataflow::OperationSemanticsCase::SpecialMathAccuracy)
    fail("math.sin is not registered with special-math semantics");
  bool sawStrict = false;
  bool sawApproximate = false;
  std::optional<StructuredSpecialMathAccuracyDecision> previous;
  for (unsigned unresolved = 0; unresolved != 2; ++unresolved) {
    auto domain =
        take(loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(
            current));
    if (domain.empty())
      fail("special-math domain closed before every selected operation");
    if (domain.front().operation.parent != current.identity())
      fail("special-math decision is not parent-local");
    auto view = take(current.view());
    auto entity = take(view.resolve(domain.front().operation));

    StructuredSpecialMathAccuracyDecision selected = domain.front();
    if (llvm::isa<mlir::math::SinOp>(entity.operation)) {
      sawStrict = true;
      if (domain.size() != 1 ||
          domain.front().accuracy != SpecialMathAccuracyTier::CorrectlyRounded)
        fail("strict operation did not have one canonical decision");
      requireFailure(
          loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
              current,
              {domain.front().operation, SpecialMathAccuracyTier::Max1Ulp}),
          "exact parent decision domain");
    } else if (llvm::isa<mlir::math::CosOp>(entity.operation)) {
      sawApproximate = true;
      if (domain.size() != 4)
        fail("afn operation did not expose the complete four-tier domain");
      for (auto item : llvm::enumerate(loom::specialMathAccuracyTiers()))
        if (domain[item.index()].accuracy != item.value() ||
            domain[item.index()].operation != domain.front().operation)
          fail("afn tier domain is not canonical");
      selected = domain[2];
      auto oneUlp =
          take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
              current, domain[1]));
      auto twoUlp =
          take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
              current, domain[2]));
      if (oneUlp.structuredProgram.identity() ==
          twoUlp.structuredProgram.identity())
        fail("distinct selected tiers produced one Structured identity");
    } else {
      fail("decision did not identify the canonical special-math operation");
    }

    auto encoded = take(
        loom::frontend::encodeStructuredSpecialMathAccuracyDecision(selected));
    auto adopted = take(
        loom::frontend::adoptStructuredSpecialMathAccuracyDecision(encoded));
    if (!(adopted == selected))
      fail("special-math decision did not round-trip");

    auto child =
        take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
            current, selected));
    previous = selected;
    current = std::move(child.structuredProgram);
    requireFailure(
        loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
            current, *previous),
        "exact parent decision domain");
  }

  if (!sawStrict || !sawApproximate)
    fail("canonical traversal did not close both special-math operations");
  if (!take(loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(
                current))
           .empty())
    fail("fully selected candidate retained an unresolved accuracy decision");

  auto strict = findOperation<mlir::math::SinOp>(current, true);
  auto approximate = findOperation<mlir::math::CosOp>(current, true);
  auto host = findOperation<mlir::math::TanOp>(current, false);
  auto strictAttr = llvm::dyn_cast_or_null<mlir::StringAttr>(
      strict->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
  auto approximateAttr = llvm::dyn_cast_or_null<mlir::StringAttr>(
      approximate->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
  if (!strictAttr || strictAttr.getValue() != "CorrectlyRounded" ||
      !approximateAttr || approximateAttr.getValue() != "Max2Ulp" ||
      host->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName))
    fail("accuracy materialization escaped its selected operation");
}

loom::frontend::StructuredProgramCandidate
closeProgram(loom::frontend::StructuredProgramCandidate candidate) {
  while (true) {
    auto domain =
        take(loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(
            candidate));
    if (domain.empty())
      return candidate;
    auto child =
        take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
            candidate, domain.front()));
    candidate = std::move(child.structuredProgram);
  }
}

void recursiveGeneratorPublishesOnlyCompleteLeaves() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-special-math-accuracy-generator", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code create = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + create.message());
  const loom::BlobStore blobs(blobPath);
  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));

  auto parent = parseProgram();
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto config = take(
      loom::dse::
          projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView());
  auto inputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {parentReference}, design.roots().front().reference()));
  auto binding = take(
      loom::dse::resolveStructuredSpecialMathAccuracyCandidateGeneratorBinding(
          config));
  auto result =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &result.outcome);
  if (!completed || completed->outputBindings.size() != 1)
    fail("special-math generator returned malformed output bindings");
  if (completed->outputBindings.front().artifacts.size() != 4 ||
      completed->lineageEdges.size() != 4)
    fail(llvm::Twine("special-math generator returned ") +
         llvm::Twine(completed->outputBindings.front().artifacts.size()) +
         " outputs and " + llvm::Twine(completed->lineageEdges.size()) +
         " lineage edges");
  if (result.workSummary.size() != 2 ||
      result.workSummary[0].consumed != 4 ||
      result.workSummary[0].planned != 4 ||
      result.workSummary[1].consumed != 4 ||
      result.workSummary[1].planned != 4)
    fail(llvm::Twine("special-math lineage and work accounting differ: ") +
         llvm::Twine(result.workSummary[0].planned) + "/" +
         llvm::Twine(result.workSummary[0].consumed) + ", " +
         llvm::Twine(result.workSummary[1].planned) + "/" +
         llvm::Twine(result.workSummary[1].consumed));

  std::map<loom::ArtifactRootReference, loom::ArtifactRootReference,
           decltype(&loom::artifactRootReferenceLess)>
      parentByChild(&loom::artifactRootReferenceLess);
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed->lineageEdges) {
    if (edge.kind !=
            loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        edge.parents.size() != 1)
      fail("special-math generator emitted non-atomic lineage");
    take(loom::frontend::adoptStructuredSpecialMathAccuracyDecision(
        edge.ownerPayload));
    auto [entry, inserted] =
        parentByChild.try_emplace(edge.output, edge.parents.front());
    if (!inserted && entry->second != edge.parents.front())
      fail("special-math child has conflicting lineage parents");
  }

  std::map<std::string, unsigned> tiers;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    auto child =
        take(loom::frontend::importStructuredProgram(reference, store));
    if (!take(loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(
                  child))
             .empty())
      fail("generator output retains unresolved special-math accuracy");
    auto strict = findOperation<mlir::math::SinOp>(child, true);
    auto approximate = findOperation<mlir::math::CosOp>(child, true);
    auto strictAttr = llvm::dyn_cast_or_null<mlir::StringAttr>(
        strict->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
    auto approximateAttr = llvm::dyn_cast_or_null<mlir::StringAttr>(
        approximate->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
    if (!strictAttr || strictAttr.getValue() != "CorrectlyRounded" ||
        !approximateAttr)
      fail("generator output lost a selected tier");
    ++tiers[approximateAttr.getValue().str()];

    auto found = parentByChild.find(reference);
    if (found == parentByChild.end() || found->second != parentReference)
      fail("complete leaf does not descend from its exact input");
  }
  for (loom::SpecialMathAccuracyTier tier : loom::specialMathAccuracyTiers())
    if (tiers[stringifySpecialMathAccuracyTier(tier).str()] != 1)
      fail("generator duplicated or omitted an afn tier");

  auto pairParent = parseApproximatePairProgram();
  auto pairParentReference =
      take(loom::frontend::publishStructuredProgram(pairParent, store));
  auto pairInputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {pairParentReference}, design.roots().front().reference()));
  auto pairResult = take(
      loom::dse::invokeCandidateGenerator(pairInputs, binding, store, blobs));
  auto *pairCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &pairResult.outcome);
  if (!pairCompleted || pairCompleted->outputBindings.size() != 1 ||
      pairCompleted->outputBindings.front().artifacts.size() != 16 ||
      pairCompleted->lineageEdges.size() != 20 ||
      pairResult.workSummary.size() != 2 ||
      pairResult.workSummary.front().planned != 20 ||
      pairResult.workSummary.front().consumed != 20 ||
      pairResult.workSummary[1].planned != 0 ||
      pairResult.workSummary[1].consumed != 0)
    fail("two afn operations did not produce the complete finite domain");
  std::map<std::pair<std::string, std::string>, unsigned> combinations;
  for (const loom::ArtifactRootReference &reference :
       pairCompleted->outputBindings.front().artifacts) {
    auto child =
        take(loom::frontend::importStructuredProgram(reference, store));
    auto sin = findOperation<mlir::math::SinOp>(child, true);
    auto cos = findOperation<mlir::math::CosOp>(child, true);
    auto sinAccuracy = llvm::dyn_cast_or_null<mlir::StringAttr>(
        sin->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
    auto cosAccuracy = llvm::dyn_cast_or_null<mlir::StringAttr>(
        cos->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
    if (!sinAccuracy || !cosAccuracy)
      fail("complete two-operation leaf lost an accuracy tier");
    ++combinations[{sinAccuracy.getValue().str(),
                    cosAccuracy.getValue().str()}];
  }
  if (combinations.size() != 16 ||
      llvm::any_of(combinations,
                   [](const auto &entry) { return entry.second != 1; }))
    fail("two afn operations did not publish every tier pair exactly once");

  auto closed = closeProgram(parseProgram());
  auto closedReference =
      take(loom::frontend::publishStructuredProgram(closed, store));
  auto closedInputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {closedReference}, design.roots().front().reference()));
  auto passThrough = take(
      loom::dse::invokeCandidateGenerator(closedInputs, binding, store, blobs));
  auto *passed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &passThrough.outcome);
  if (!passed ||
      passed->outputBindings.front().artifacts !=
          std::vector<loom::ArtifactRootReference>{closedReference} ||
      !passed->lineageEdges.empty())
    fail("closed candidate did not pass through the exact closure gate");

  if (std::error_code cleanup = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void exactFabricPruningRetainsOnlyReachableLineage() {
  using loom::SpecialMathAccuracyTier;
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-special-math-fabric-pruning", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code create = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + create.message());
  const loom::BlobStore blobs(blobPath);
  auto config = take(
      loom::dse::
          projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView());
  auto binding = take(
      loom::dse::resolveStructuredSpecialMathAccuracyCandidateGeneratorBinding(
          config));

  auto parent = parseSingleSpecialMathProgram(true);
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto partialFabric = makeSinFabric(store, fabric::FloatFormat::F32,
                                     SpecialMathAccuracyTier::Max2Ulp);
  auto domain = take(
      loom::frontend::enumerateStructuredSpecialMathAccuracyDecisions(parent));
  auto max2 = llvm::find_if(domain, [](const auto &decision) {
    return decision.accuracy == SpecialMathAccuracyTier::Max2Ulp;
  });
  if (max2 == domain.end())
    fail("single afn operation omitted Max2Ulp");
  auto selected =
      take(loom::frontend::materializeStructuredSpecialMathAccuracyDecision(
          parent, *max2));
  auto admitted = loom::frontend::finalizeSpatialOwnershipCandidate(
      {std::move(selected.structuredProgram),
       std::nullopt,
       {},
       std::move(selected.sourceProvenance)},
      partialFabric);
  if (!admitted)
    fail("Max2Ulp Fabric rejected its exact actor: " +
         llvm::toString(admitted.takeError()));
  auto partialInputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {parentReference}, partialFabric.reference()));
  auto partial = take(loom::dse::invokeCandidateGenerator(
      partialInputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &partial.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      partial.workSummary.size() != 2)
    fail("partial Fabric admission returned a malformed result");
  if (completed->outputBindings.front().artifacts.size() != 2 ||
      completed->lineageEdges.size() != 2 ||
      partial.workSummary.front().planned != 4 ||
      partial.workSummary.front().consumed != 4 ||
      partial.workSummary[1].planned != 0 ||
      partial.workSummary[1].consumed != 0)
    fail(llvm::Twine("partial Fabric admission returned ") +
         llvm::Twine(completed->outputBindings.front().artifacts.size()) +
         " outputs, " + llvm::Twine(completed->lineageEdges.size()) +
         " edges, and " + llvm::Twine(partial.workSummary.front().planned) +
         " work items");

  std::set<std::string> admittedTiers;
  for (const loom::ArtifactRootReference &reference :
       completed->outputBindings.front().artifacts) {
    auto child =
        take(loom::frontend::importStructuredProgram(reference, store));
    auto operation = findOperation<mlir::math::SinOp>(child, true);
    auto accuracy = llvm::dyn_cast_or_null<mlir::StringAttr>(
        operation->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
    if (!accuracy)
      fail("admitted leaf lost its selected accuracy");
    admittedTiers.insert(accuracy.getValue().str());
  }
  if (admittedTiers != std::set<std::string>{"Max2Ulp", "Max4Ulp"})
    fail("partial Fabric admission retained the wrong accuracy tiers");
  for (const loom::dse::CandidateGeneratorLineageEdge &edge :
       completed->lineageEdges) {
    if (edge.kind !=
            loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        edge.parents !=
            std::vector<loom::ArtifactRootReference>{parentReference} ||
        !llvm::is_contained(completed->outputBindings.front().artifacts,
                            edge.output))
      fail("partial Fabric admission retained orphan lineage");
  }

  auto rejectingFabric =
      makeSinFabric(store, fabric::FloatFormat::F64,
                    SpecialMathAccuracyTier::CorrectlyRounded);
  auto rejectingInputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {parentReference}, rejectingFabric.reference()));
  auto rejected = take(loom::dse::invokeCandidateGenerator(
      rejectingInputs, binding, store, blobs));
  auto *empty = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &rejected.outcome);
  if (!empty || empty->outputBindings.size() != 1 ||
      !empty->outputBindings.front().artifacts.empty() ||
      !empty->lineageEdges.empty() || rejected.workSummary.size() != 2 ||
      rejected.workSummary.front().planned != 4 ||
      rejected.workSummary.front().consumed != 4 ||
      rejected.workSummary[1].planned != 0 ||
      rejected.workSummary[1].consumed != 0)
    fail("complete Fabric rejection did not produce an empty finite set");

  if (std::error_code cleanup = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

void strictClosureIsMechanical() {
  llvm::SmallString<128> directory;
  std::error_code error = llvm::sys::fs::createUniqueDirectory(
      "loom-special-math-strict-closure", directory);
  if (error)
    fail("cannot create ArtifactStore directory: " + error.message());
  loom::ArtifactStore store(directory);
  llvm::SmallString<128> blobPath(directory);
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code create = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + create.message());
  const loom::BlobStore blobs(blobPath);
  auto fabric = makeSinFabric(store, fabric::FloatFormat::F32,
                              loom::SpecialMathAccuracyTier::CorrectlyRounded);
  auto parent = parseSingleSpecialMathProgram(false);
  auto parentReference =
      take(loom::frontend::publishStructuredProgram(parent, store));
  auto config = take(
      loom::dse::
          projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView());
  auto inputs =
      take(loom::dse::bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
          {parentReference}, fabric.reference()));
  auto binding = take(
      loom::dse::resolveStructuredSpecialMathAccuracyCandidateGeneratorBinding(
          config));
  auto result =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  auto *completed = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &result.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1 ||
      completed->lineageEdges.size() != 1 || result.workSummary.size() != 2 ||
      result.workSummary.front().planned != 0 ||
      result.workSummary.front().consumed != 0 ||
      result.workSummary[1].planned != 1 ||
      result.workSummary[1].consumed != 1)
    fail("strict special math created an accuracy choice");
  const auto &edge = completed->lineageEdges.front();
  if (edge.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation ||
      !edge.parents.empty() || !edge.ownerPayload.empty() ||
      edge.output != completed->outputBindings.front().artifacts.front())
    fail("strict closure did not use mechanical lineage");

  auto child = take(loom::frontend::importStructuredProgram(
      completed->outputBindings.front().artifacts.front(), store));
  auto operation = findOperation<mlir::math::SinOp>(child, true);
  auto accuracy = llvm::dyn_cast_or_null<mlir::StringAttr>(
      operation->getDiscardableAttr(loom::kSpecialMathAccuracyAttrName));
  if (!accuracy || accuracy.getValue() != "CorrectlyRounded")
    fail("strict closure did not materialize correctly rounded accuracy");

  if (std::error_code cleanup = llvm::sys::fs::remove_directories(directory))
    fail("cannot remove ArtifactStore directory: " + cleanup.message());
}

} // namespace

int main() {
  canonicalDomainAndScopedMaterialization();
  recursiveGeneratorPublishesOnlyCompleteLeaves();
  exactFabricPruningRetainsOnlyReachableLineage();
  strictClosureIsMechanical();
  return EXIT_SUCCESS;
}
