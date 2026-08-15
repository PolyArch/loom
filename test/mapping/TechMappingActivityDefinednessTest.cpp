#include "TechMappingCandidateDomain.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping activity definedness test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  llvm::StringRef body) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(body, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
parallelizeDataflow(mlir::MLIRContext &context, bool provedPhase) {
  constexpr llvm::StringLiteral proved = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @parallelize(%start: none, %data: i8)
      -> (vector<4xi8>, vector<4xi1>, i1)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 3, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %phase = dataflow.stream %zero, %one, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %units = dataflow.invariant %group_phase, %start : none
    %close:2 = dataflow.demux %group_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%vector, %mask, %group_phase
          : vector<4xi8>, vector<4xi1>, i1)
        memories() complete(%close#0 : none)
  }
}
)mlir";
  constexpr llvm::StringLiteral unproved = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @parallelize(
      %start: none, %data: i8)
      -> (vector<4xi8>, vector<4xi1>, i1)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 3, 0>} {
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %stream_phase = dataflow.stream %data, %one, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %stream_phase
        : (i8, i1) -> (vector<4xi8>, vector<4xi1>, i1)
    %units = dataflow.invariant %group_phase, %start : none
    %close:2 = dataflow.demux %group_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%vector, %mask, %group_phase
          : vector<4xi8>, vector<4xi1>, i1)
        memories() complete(%close#0 : none)
  }
}
)mlir";
  return buildDataflow(context, provedPhase ? proved : unproved);
}

dataflow::CanonicalDataflowArtifact
serializeDataflow(mlir::MLIRContext &context, bool provedMask,
                  bool provedPhase) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @)mlir"
     << (provedMask
             ? (provedPhase ? "serialize_defined" : "serialize_unproved_phase")
             : "serialize_unproved_mask")
     << R"mlir((%start: none)mlir";
  if (!provedPhase)
    os << ", %phase_seed: i8";
  os << ") -> (i8, i1) attributes {input_segments = array<i32: " << !provedPhase
     << R"mlir(, 0, 0>, result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %ordinal, %phase_defined = dataflow.stream )mlir"
     << (provedPhase ? "%zero" : "%phase_seed") << R"mlir(, %one, %one
        step add while ult : i8
    %group_units = dataflow.invariant %phase_defined, %start : none
    %group_events:2 = dataflow.demux %phase_defined, %group_units
        : (i1, none) -> (none, none)
    %vector_bits = dataflow.constant %group_events#1
        {const_value = 197121 : i32} : i32
    %vectors = dataflow.unpack %vector_bits : i32 -> vector<4xi8>
    %mask_bits = dataflow.constant %group_events#1
        {const_value = 15 : i4} : i4
    %mask_defined = dataflow.unpack %mask_bits : i4 -> vector<4xi1>
)mlir";
  if (provedMask) {
    os << R"mlir(    %masks = dataflow.sync %mask_defined
        : (vector<4xi1>) -> vector<4xi1>
)mlir";
  } else {
    os << R"mlir(    %masks = arith.divui %mask_defined, %mask_defined
        : vector<4xi1>
)mlir";
  }
  os << R"mlir(
    %data, %scalar_phase =
      dataflow.serialize %vectors, %masks, %phase_defined
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%data, %scalar_phase : i8, i1)
        memories() complete(%close#0 : none)
  }
}
)mlir";
  return buildDataflow(context, os.str());
}

loom::fabric::FinalizedFabricRoot buildFabric(loom::ArtifactStore &store) {
  loom::adg::DesignBuilder builder(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, loom::adg::BuiltinTargetPreset::Small));
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("builtin Fabric did not publish one root");
  bool sawParallelize = false;
  bool sawSerialize = false;
  const auto view = design.roots().front().view();
  for (std::uint64_t entity = 0;; ++entity) {
    const auto kind = view.entityKind(entity);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    const loom::fabric::FabricFuTemplateRef fu(entity);
    for (const auto &capability : view.resolvedFabricOpCapabilities(fu)) {
      dataflow::OperationSchemaId schema;
      bool *observed = nullptr;
      if (capability.implementationFamily ==
          fabric::ImplementationFamilyId::FixedVectorParallelize) {
        schema = dataflow::OperationSchemaId::DataflowParallelize;
        observed = &sawParallelize;
      } else if (capability.implementationFamily ==
                 fabric::ImplementationFamilyId::FixedVectorSerialize) {
        schema = dataflow::OperationSchemaId::DataflowSerialize;
        observed = &sawSerialize;
      } else {
        continue;
      }
      const auto *parameters = std::get_if<fabric::FixedVectorAdapterParams>(
          &capability.parameterizedCapability);
      if (!parameters)
        fail("builtin adapter lost its typed capability parameters");
      const auto maximumLaneCount =
          take(fabric::maximumFixedVectorAdapterLaneCount(*parameters));
      const auto exact =
          take(fabric::isOrderedCardinalityOperationResourceContract(
              capability.resourceStateAndTimingContract, schema,
              maximumLaneCount));
      if (!exact)
        fail("builtin adapter does not carry its exact ordered contract");
      *observed = true;
    }
  }
  if (!sawParallelize || !sawSerialize)
    fail("builtin Fabric lost an ordered-cardinality adapter family");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
buildLegacyAdapterFabric(loom::ArtifactStore &store,
                         fabric::ImplementationFamilyId family) {
  const bool parallelize =
      family == fabric::ImplementationFamilyId::FixedVectorParallelize;
  if (!parallelize &&
      family != fabric::ImplementationFamilyId::FixedVectorSerialize)
    fail("legacy adapter fixture received a non-adapter family");
  const dataflow::OperationSchemaId schema =
      parallelize ? dataflow::OperationSchemaId::DataflowParallelize
                  : dataflow::OperationSchemaId::DataflowSerialize;
  const std::size_t inputCount = parallelize ? 2 : 3;
  const std::size_t resultCount = parallelize ? 3 : 2;
  const auto bits128 = take(loom::adg::PortType::bits(128));
  const std::vector<loom::adg::PortType> inputTypes(inputCount, bits128);
  const std::vector<loom::adg::PortType> resultTypes(resultCount, bits128);

  loom::adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore(
      parallelize ? "legacy-parallelize" : "legacy-serialize", inputTypes,
      resultTypes));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != inputCount; ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(
      spatialInputs, loom::adg::PeSpec::spatial(inputTypes, resultTypes)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != inputCount; ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  auto fu =
      take(pe.addFu(peInputs, loom::adg::FuSpec{inputTypes, resultTypes}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != inputCount; ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  const fabric::FixedVectorAdapterParams parameters{
      fabric::IntegerWidthSet::get({fabric::IntegerWidth::I8}),
      fabric::FloatFormatSet{}, 128};
  auto operation = take(fu.addOperation(
      fuInputs, loom::adg::OperationCapabilitySpec{
                    family,
                    parameters,
                    {schema},
                    resultTypes,
                    fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error = fu.addCapabilityTemplate(
          loom::adg::FuCapabilityTemplateSpec{{operation}, {}}))
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> fuOutputs;
  for (std::size_t ordinal = 0; ordinal != resultCount; ++ordinal)
    fuOutputs.push_back(take(operation.output(ordinal)));
  if (llvm::Error error = fu.close(fuOutputs))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal != resultCount; ++ordinal)
    spatialOutputs.push_back(take(pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("legacy adapter fixture did not publish one root");
  return design.roots().front();
}

struct DerivedRows final {
  std::size_t rows = 0;
  std::uint64_t capabilityRejections = 0;
};

struct CollectedRows final {
  dataflow::CanonicalDataflowProgramView dataflow;
  dataflow::CanonicalActorView actor;
  std::vector<loom::mapping::detail::TechMatchRow> rows;
  std::uint64_t capabilityRejections = 0;
};

CollectedRows collectRows(dataflow::CanonicalDataflowArtifact &artifact,
                          dataflow::OperationSchemaId schema,
                          const loom::fabric::FinalizedFabricRoot &fabric,
                          loom::ArtifactStore &store) {
  const auto view = take(artifact.view());
  const dataflow::CanonicalActorView *selected = nullptr;
  for (const auto &actor : view.actors())
    if (dataflow::requireOperationSchema(actor.op) == schema) {
      selected = &actor;
      break;
    }
  if (!selected)
    fail("Dataflow fixture has no selected adapter");

  const auto fabricView = fabric.view();
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {selected->graph};
  const std::array<dataflow::CanonicalActorView, 1> actors = {*selected};
  const std::array<dataflow::ActorRef, 1> actorRefs = {selected->ref};
  loom::mapping::TechMappingGenerationAccounting accounting;
  loom::mapping::detail::TechMatchRowCollector collector(actorRefs, 64,
                                                         accounting, {});
  const loom::mapping::TechMappingGenerationInputs inputs{
      view, covers, fabricView, config, store};
  const auto families =
      loom::mapping::detail::deriveComputeRowFamilies(inputs, actors);
  for (const auto family : families)
    if (llvm::Error error = loom::mapping::detail::deriveComputeRows(
            inputs, actors, family, collector))
      fail(llvm::toString(std::move(error)));
  auto rows = take(collector.takeRows());
  return {view, *selected, std::move(rows),
          collector.rejectionCount(
              loom::mapping::detail::TechMatchSeedRejectionReason::
                  CapabilityInadmissible)};
}

DerivedRows deriveRows(dataflow::CanonicalDataflowArtifact &artifact,
                       dataflow::OperationSchemaId schema,
                       const loom::fabric::FinalizedFabricRoot &fabric,
                       loom::ArtifactStore &store) {
  CollectedRows collected = collectRows(artifact, schema, fabric, store);
  return {collected.rows.size(), collected.capabilityRejections};
}

void activityProofGatesProspectiveSeeds(llvm::StringRef storeRoot) {
  if (std::error_code error = llvm::sys::fs::create_directories(storeRoot))
    fail("cannot create ArtifactStore root: " + error.message());
  loom::ArtifactStore store(storeRoot);
  const auto fabric = buildFabric(store);

  auto check = [&](dataflow::CanonicalDataflowArtifact artifact,
                   dataflow::OperationSchemaId schema, bool admitted,
                   llvm::StringRef message) {
    const DerivedRows result = deriveRows(artifact, schema, fabric, store);
    if (admitted ? result.rows == 0
                 : result.rows != 0 || result.capabilityRejections == 0)
      fail(message);
  };

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mapping::MappingDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  check(parallelizeDataflow(context, true),
        dataflow::OperationSchemaId::DataflowParallelize, true,
        "proved parallelize phase rejected its prospective seed");
  check(parallelizeDataflow(context, false),
        dataflow::OperationSchemaId::DataflowParallelize, false,
        "unproved parallelize phase admitted a match row");
  check(serializeDataflow(context, true, true),
        dataflow::OperationSchemaId::DataflowSerialize, true,
        "proved serialize activity rejected its prospective seed");
  check(serializeDataflow(context, false, true),
        dataflow::OperationSchemaId::DataflowSerialize, false,
        "unproved serialize mask admitted a match row");
  check(serializeDataflow(context, true, false),
        dataflow::OperationSchemaId::DataflowSerialize, false,
        "unproved serialize phase admitted a match row");

  const auto legacyParallelize = buildLegacyAdapterFabric(
      store, fabric::ImplementationFamilyId::FixedVectorParallelize);
  auto definedParallelize = parallelizeDataflow(context, true);
  const DerivedRows rejectedParallelize = deriveRows(
      definedParallelize, dataflow::OperationSchemaId::DataflowParallelize,
      legacyParallelize, store);
  if (rejectedParallelize.rows != 0 ||
      rejectedParallelize.capabilityRejections == 0)
    fail("legacy one-cycle parallelize contract was admitted");

  const auto legacySerialize = buildLegacyAdapterFabric(
      store, fabric::ImplementationFamilyId::FixedVectorSerialize);
  auto definedSerialize = serializeDataflow(context, true, true);
  const DerivedRows rejectedSerialize = deriveRows(
      definedSerialize, dataflow::OperationSchemaId::DataflowSerialize,
      legacySerialize, store);
  if (rejectedSerialize.rows != 0 ||
      rejectedSerialize.capabilityRejections == 0)
    fail("legacy one-cycle serialize contract was admitted");

  auto defined = parallelizeDataflow(context, true);
  auto unproved = parallelizeDataflow(context, false);
  CollectedRows definedRows = collectRows(
      defined, dataflow::OperationSchemaId::DataflowParallelize, fabric, store);
  if (definedRows.rows.empty())
    fail("strict-import fixture has no valid physical correspondence");
  take(dataflow::publishCanonicalDataflow(unproved, store));
  auto unprovedView = take(unproved.view());
  const dataflow::CanonicalActorView *unprovedActor = nullptr;
  for (const auto &actor : unprovedView.actors())
    if (dataflow::requireOperationSchema(actor.op) ==
        dataflow::OperationSchemaId::DataflowParallelize) {
      unprovedActor = &actor;
      break;
    }
  if (!unprovedActor)
    fail("strict-import fixture lost its unproved adapter");

  auto forged = definedRows.rows.front();
  auto *realization = std::get_if<loom::mapping::TechComputeRealizationView>(
      &forged.realization);
  if (!realization || realization->actors.size() != 1)
    fail("strict-import fixture did not select one compute actor");
  const dataflow::ActorRef definedActor = realization->actors.front().actor;
  realization->actors.front().actor = unprovedActor->ref;
  for (auto &boundary : realization->boundaries) {
    if (boundary.actor != definedActor)
      fail("strict-import fixture has a foreign boundary actor");
    boundary.actor = unprovedActor->ref;
  }

  const auto strictConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(
          loom::defaultResolvedConfig()));
  const std::array<dataflow::GraphRef, 1> strictCovers = {unprovedActor->graph};
  const std::array<const loom::mapping::detail::TechMatchRow *, 1> forgedRows =
      {&forged};
  auto imported = loom::mapping::detail::materializeTechMappingCandidate(
      {unprovedView, strictCovers, fabric.view(), strictConfig, store},
      forgedRows);
  if (imported)
    fail("strict import trusted a forged unproved adapter row");
  const std::string diagnostic = llvm::toString(imported.takeError());
  if (diagnostic.find("activity definedness") == std::string::npos)
    fail("strict import rejected the forged row after the activity joint: " +
         diagnostic);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one ArtifactStore root");
  activityProofGatesProspectiveSeeds(argv[1]);
  llvm::outs() << "tech mapping activity definedness tests passed\n";
  return 0;
}
