#include "TechMappingCandidateDomain.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
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
  return design.roots().front();
}

struct DerivedRows final {
  std::size_t rows = 0;
  std::uint64_t capabilityRejections = 0;
};

DerivedRows deriveRows(dataflow::CanonicalDataflowArtifact &artifact,
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
                                                         accounting);
  if (llvm::Error error = loom::mapping::detail::deriveComputeRows(
          {view, covers, fabricView, config, store}, actors, collector))
    fail(llvm::toString(std::move(error)));
  const auto rows = take(collector.takeRows());
  return {rows.size(), collector.rejectionCount(
                           loom::mapping::detail::TechMatchSeedRejectionReason::
                               CapabilityInadmissible)};
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
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one ArtifactStore root");
  activityProofGatesProspectiveSeeds(argv[1]);
  llvm::outs() << "tech mapping activity definedness tests passed\n";
  return 0;
}
