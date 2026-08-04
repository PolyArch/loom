#include "SpatialMemoryConstraintTestSupport.h"

#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCandidateState.h"

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

template <typename Ref>
std::string fabricAttr(llvm::StringRef spelling, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(loom::fabric::canonicalFabricBytes(ref)) + ">";
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

  if (dataflow.logicalMemoryRoots().size() != 2)
    fail("fixture does not expose two independent logical memory roots");
  const auto memories = fabric.memoryOccurrences();
  if (memories.empty())
    fail("fixture Fabric has no local memory occurrence");
  const auto *service = fabric.localMemoryService(memories.front());
  if (!service || service->regions().empty())
    fail("fixture Fabric has no local memory service region");
  const fabric::FabricMemoryServiceRef serviceRef =
      fabric::FabricMemoryServiceRef::local(memories.front());
  const auto &region = service->regions().front();
  const std::uint64_t regionEnd = region.addressBaseBytes + region.sizeBytes;
  const std::string roots =
      dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                   dataflow.logicalMemoryRoots()[0].ref) +
      ", " +
      dataflowAttr("logical_memory_root_ref", dataflow.identity(),
                   dataflow.logicalMemoryRoots()[1].ref);
  const std::string serviceValue =
      fabricAttr("fabric_memory_service_ref", serviceRef);
  const std::string addressValue =
      "#mapping.constraint_address_region<service = " + serviceValue +
      ", intervals = [#mapping.constraint_unsigned_interval<lower = " +
      std::to_string(region.addressBaseBytes) +
      " : ui64, upper = " + std::to_string(regionEnd) + " : ui64>]>";
  const auto buildMemoryRootConstraints = [&](llvm::StringRef serviceRelation,
                                              llvm::StringRef addressRelation) {
    std::string clauses;
    for (const auto &root : dataflow.logicalMemoryRoots()) {
      const std::string subject = dataflowAttr("logical_memory_root_ref",
                                               dataflow.identity(), root.ref);
      clauses += "    mapping.constraint.domain_restriction "
                 "projection(memory_bound_services) subject(" +
                 subject + ") admissible_domain([" + serviceValue + "])\n";
      clauses += "    mapping.constraint.domain_restriction "
                 "projection(memory_address_region) subject(" +
                 subject + ") admissible_domain([" + addressValue + "])\n";
    }
    clauses += "    mapping.constraint." + serviceRelation.str() +
               " projection(memory_bound_services) subjects([" + roots + "])\n";
    clauses += "    mapping.constraint." + addressRelation.str() +
               " projection(memory_address_region) subjects([" + roots + "])\n";
    const std::string text =
        "module {\n  mapping.constraints.spatial dataflow(" +
        identityAttr(dataflow.identity()) + ") tech_mapping(" +
        identityAttr(techMapping.identity()) + ") fabric(" +
        identityAttr(fabric.identity()) + ") {\n" + clauses + "  }\n}\n";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      fail("cannot parse memory-root relation fixture");
    auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
    return take(mapping::finalizeSpatialMappingConstraintSet(
        *roots.begin(), dataflow, techMapping, fabric, store));
  };

  const auto feasibleMemoryRoots =
      buildMemoryRootConstraints("equal", "disjoint");
  auto feasibleMemoryProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, feasibleMemoryRoots.view()));
  auto feasibleMemoryCandidate =
      take(pnr::createCanonicalSpatialCandidate(feasibleMemoryProblem));
  if (llvm::Error error = feasibleMemoryCandidate->verify())
    fail("memory service/address relations failed cold verification: " +
         llvm::toString(std::move(error)));
  for (pnr::PnrIndex binding = 0; binding < 2; ++binding) {
    const auto target =
        feasibleMemoryCandidate->logicalMemoryBinding(binding).target;
    if (target >= feasibleMemoryProblem->memory().bindingTargets().size() ||
        !std::holds_alternative<fabric::FabricMemoryServiceRegionRef>(
            feasibleMemoryProblem->memory().bindingTargets()[target].target))
      fail("feasible memory-root relation escaped through BoundaryProxy");
  }

  const auto impossibleMemoryRoots =
      buildMemoryRootConstraints("disjoint", "equal");
  auto impossibleMemoryProblem = take(pnr::freezeSpatialPnrProblem(
      dataflow, techMapping, fabric, pnrConfig, impossibleMemoryRoots.view()));
  auto emptyMemoryCandidate =
      take(pnr::createCanonicalSpatialCandidate(impossibleMemoryProblem));
  if (llvm::Error error = emptyMemoryCandidate->verify())
    fail("empty zero-or-more memory projections failed verification: " +
         llvm::toString(std::move(error)));
  pnr::PnrIndex localTarget = pnr::getInvalidPnrIndex();
  for (auto [ordinal, target] :
       llvm::enumerate(impossibleMemoryProblem->memory().bindingTargets()))
    if (std::holds_alternative<fabric::FabricMemoryServiceRegionRef>(
            target.target)) {
      localTarget = static_cast<pnr::PnrIndex>(ordinal);
      break;
    }
  if (localTarget == pnr::getInvalidPnrIndex())
    fail("memory relation fixture has no local target");
  pnr::SpatialCandidateScratch scratch;
  if (llvm::Error error = scratch.prepare(*impossibleMemoryProblem))
    fail(llvm::toString(std::move(error)));
  auto move = take(emptyMemoryCandidate->beginMove(scratch));
  if (llvm::Error error = move.setLogicalMemoryBinding(0, localTarget, 0))
    fail(llvm::toString(std::move(error)));
  auto closed = move.close();
  if (closed)
    fail("memory service/address relation accepted an unequal local move");
  llvm::consumeError(closed.takeError());
}
