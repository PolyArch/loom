#include "SpatialMemoryConstraintTestSupport.h"

#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial memory constraint test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
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

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &owner, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(take(dataflow::encodeDataflowReference(owner, ref))) + ">";
}

} // namespace

void loom::test::exerciseSpatialMemoryOperationPortRelations(
    mlir::MLIRContext &context,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::TechMappingView &techMapping,
    const fabric::FabricArtifactView &fabric, const ArtifactStore &store) {
  if (techMapping.memoryRealizations().size() != 1 ||
      techMapping.memoryRealizations().front().actors.size() != 2)
    fail("fixture is not one grouped two-actor memory realization");
  const auto &actors = techMapping.memoryRealizations().front().actors;
  const bool samePort =
      actors[0].operationPort.ordinal == actors[1].operationPort.ordinal;
  const auto buildConstraints = [&](llvm::StringRef relation) {
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n    mapping.constraint." +
        relation.str() + " projection(memory_operation_port) subjects([" +
        dataflowAttr("actor_ref", dataflow.identity(), actors[0].actor) + ", " +
        dataflowAttr("actor_ref", dataflow.identity(), actors[1].actor) +
        "])\n  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse memory operation-port relation fixture");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };

  const auto pnrConfig = take(
      pnr::projectResolvedSpatialPnrConfigView(loom::defaultResolvedConfig()));
  const auto feasibleConstraints =
      buildConstraints(samePort ? "equal" : "disjoint");
  auto feasibleProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, feasibleConstraints.view()));
  auto feasible = take(pnr::createCanonicalSpatialCandidate(feasibleProblem));
  if (llvm::Error error = feasible->verify())
    fail("memory operation-port relation failed cold verification: " +
         llvm::toString(std::move(error)));

  const auto impossibleConstraints =
      buildConstraints(samePort ? "disjoint" : "equal");
  auto impossibleProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, impossibleConstraints.view()));
  auto impossible = pnr::createCanonicalSpatialCandidate(impossibleProblem);
  if (impossible)
    fail("contradictory memory operation-port relation produced a candidate");
  llvm::consumeError(impossible.takeError());
}
