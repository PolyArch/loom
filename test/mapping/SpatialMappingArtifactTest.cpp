#include "ADG/Builder.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial mapping artifact test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-spatial-mapping-artifact", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

void addTokenSyncFu(loom::adg::PeBuilder &pe,
                    llvm::ArrayRef<loom::adg::PeValue> inputs,
                    const loom::adg::PortType &type) {
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;

  const std::vector<loom::adg::PortType> types(4, type);
  auto fu = take(pe.addFu(inputs, FuSpec{types, types}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{128, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(
      fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<loom::adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(outputs));
}

loom::fabric::FinalizedFabricRoot buildFabric(loom::ArtifactStore &store) {
  using loom::adg::DesignBuilder;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> types(4, bits128);
  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(spatialInputs, PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  addTokenSyncFu(pe, peInputs, bits128);
  requireSuccess(pe.close());
  std::vector<loom::adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("Fabric fixture did not publish exactly one root");
  return design.roots().front();
}

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string text = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      text += ", ";
    text += std::to_string(static_cast<std::int8_t>(byte));
  }
  return text + "]";
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

loom::mapping::FinalizedSpatialMappingConstraintSet
buildConstraints(mlir::MLIRContext &context,
                 const dataflow::CanonicalDataflowProgramView &dataflow,
                 const loom::mapping::TechMappingView &tech,
                 const loom::fabric::FabricArtifactView &fabric,
                 const loom::ArtifactStore &store) {
  const std::string text = "module {\n  mapping.constraints.spatial dataflow(" +
                           identityAttr(dataflow.identity()) +
                           ") tech_mapping(" + identityAttr(tech.identity()) +
                           ") fabric(" + identityAttr(fabric.identity()) +
                           ") {\n  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse empty MappingConstraintSet fixture");
  auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
  return take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *roots.begin(), dataflow, tech, fabric, store));
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSpatial(mlir::MLIRContext &context,
             const loom::CanonicalSemanticBytes &bytes) {
  std::string text = "module {\n";
  text.append(reinterpret_cast<const char *>(bytes.bytes().data()),
              bytes.bytes().size());
  text += "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

void completeCandidateRoundTrip() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generated = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  auto *candidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&generated);
  if (!candidates || candidates->candidates.size() != 1)
    fail("TechMapping fixture did not produce one candidate");
  const auto tech = take(
      loom::mapping::importTechMapping(candidates->candidates.front(), store));
  const auto constraints =
      buildConstraints(context, dataflow, tech.view(), fabric.view(), store);
  const auto pnrConfig = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::defaultResolvedConfig()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  auto candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
  loom::pnr::SpatialCandidateScratch candidateScratch;
  requireSuccess(candidateScratch.prepare(*problem));
  auto costs = take(loom::pnr::SpatialRouteCostState::create(*candidate));
  loom::pnr::SpatialPathFinderRouterScratch router;
  requireSuccess(router.prepare(*problem));
  take(router.routeToClosure(
      *candidate, candidateScratch, costs,
      {pnrConfig.policy().search.routing.endpointExpansionLimit,
       pnrConfig.policy().search.routing.negotiationIterationLimit},
      {}));
  requireSuccess(candidate->verify());

  auto finalized = take(loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(), store));
  auto imported =
      take(loom::mapping::importSpatialMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      imported.view().computeBindings().size() != 1 ||
      imported.view().routeTrees().empty() ||
      imported.view().resourceUses().empty())
    fail("strict SpatialMapping round trip lost selected closure");

  auto missingRoute = parseSpatial(context, finalized.canonicalBytes());
  if (!missingRoute)
    fail("cannot parse finalized SpatialMapping fixture");
  auto root = *missingRoute->getOps<::mapping::SpatialOp>().begin();
  auto routes = root.getBody().front().getOps<::mapping::RouteTreeOp>();
  (*routes.begin()).erase();
  if (!rejected(loom::mapping::finalizeSpatialMapping(root, store)))
    fail("SpatialMapping finalized without a required RouteTree");

  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse finalized SpatialMapping fixture");
  auto useRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto uses = useRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  (*uses.begin()).erase();
  if (!rejected(loom::mapping::finalizeSpatialMapping(useRoot, store)))
    fail("SpatialMapping finalized without a required ResourceUse");
}

} // namespace

int main() {
  completeCandidateRoundTrip();
  llvm::outs() << "spatial mapping artifact tests passed\n";
  return 0;
}
