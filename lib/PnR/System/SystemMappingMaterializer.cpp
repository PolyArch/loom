#include "PnR/System/SystemMappingMaterializer.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace loom::pnr {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_materialization_invalid: " +
                                     message);
}

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Attr, typename Ref>
llvm::Expected<Attr> dataflowRefAttr(mlir::MLIRContext *context,
                                     const ArtifactIdentity &owner,
                                     const Ref &reference) {
  auto bytes = ::dataflow::encodeDataflowReference(owner, reference);
  if (!bytes)
    return bytes.takeError();
  return Attr::get(context, bytesAttr(context, *bytes));
}

::mapping::ArtifactIdentityAttr identityAttr(mlir::MLIRContext *context,
                                             const ArtifactIdentity &identity) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, bytesAttr(context, identity.bytes()));
}

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const ArtifactRootReference &reference) {
  return ::mapping::ArtifactRootReferenceAttr::get(
      context, bytesAttr(context, encodeArtifactRootReference(reference)));
}

::mapping::FabricAccCoreOccurrenceRefAttr
accCoreAttr(mlir::MLIRContext *context,
            ::loom::fabric::AccCoreOccurrenceRef core) {
  return ::mapping::FabricAccCoreOccurrenceRefAttr::get(
      context, bytesAttr(context, ::loom::fabric::canonicalFabricBytes(core)));
}

::mapping::SystemPresburgerCellAttr
cellAttr(mlir::MLIRContext *context,
         const ::loom::mapping::SystemPresburgerCell &cell) {
  llvm::SmallVector<mlir::Attribute> equalities;
  llvm::SmallVector<mlir::Attribute> inequalities;
  for (const auto &row : cell.equalities)
    equalities.push_back(mlir::DenseI64ArrayAttr::get(context, row));
  for (const auto &row : cell.inequalities)
    inequalities.push_back(mlir::DenseI64ArrayAttr::get(context, row));
  return ::mapping::SystemPresburgerCellAttr::get(
      context, cell.dimensionCount, cell.symbolCount,
      mlir::ArrayAttr::get(context, equalities),
      mlir::ArrayAttr::get(context, inequalities));
}

template <typename BindingOp, typename KeyAttr, typename ClauseOp,
          typename TargetAttr>
llvm::Expected<BindingOp> createBinding(
    mlir::OpBuilder &builder, mlir::Location location, mlir::Block &parent,
    KeyAttr key,
    llvm::ArrayRef<std::pair<::loom::mapping::SystemPresburgerCell, TargetAttr>>
        clauses) {
  mlir::OperationState bindingState(location, BindingOp::getOperationName());
  bindingState.addAttribute("key", key);
  bindingState.addAttribute(
      "relation_kind",
      ::mapping::SystemBindingRelationKindAttr::get(
          builder.getContext(),
          ::mapping::SystemBindingRelationKind::PresburgerPartition));
  bindingState.addRegion();
  builder.setInsertionPointToEnd(&parent);
  auto binding = mlir::cast<BindingOp>(builder.create(bindingState));
  binding.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&binding.getBody().front());
  for (const auto &[cell, target] : clauses) {
    mlir::OperationState clauseState(location, ClauseOp::getOperationName());
    clauseState.addAttribute(
        "cells", mlir::ArrayAttr::get(builder.getContext(),
                                      {cellAttr(builder.getContext(), cell)}));
    clauseState.addAttribute("target", target);
    builder.create(clauseState);
  }
  return binding;
}

} // namespace

llvm::Expected<mlir::OwningOpRef<mlir::Operation *>>
materializeSystemExecutionBindings(const SystemCandidateState &candidate,
                                   mlir::MLIRContext &context) {
  if (llvm::Error error = candidate.verify())
    return std::move(error);
  context.getOrLoadDialect<::mapping::MappingDialect>();
  const FrozenSystemPnrProblem &problem = candidate.problem();

  std::vector<ArtifactRootReference> imports;
  imports.reserve(problem.graphDecisions().size());
  for (PnrIndex decision = 0; decision != problem.graphDecisions().size();
       ++decision)
    imports.push_back(candidate.selectedSpatialMapping(decision));
  llvm::sort(imports, artifactRootReferenceLess);
  imports.erase(std::unique(imports.begin(), imports.end()), imports.end());
  std::map<ArtifactRootReference, std::uint64_t,
           decltype(&artifactRootReferenceLess)>
      importOrdinals(&artifactRootReferenceLess);
  for (const auto &[ordinal, reference] : llvm::enumerate(imports))
    importOrdinals.emplace(reference, ordinal);

  mlir::OpBuilder builder(&context);
  mlir::Location location = builder.getUnknownLoc();
  mlir::OperationState rootState(location,
                                 ::mapping::SystemOp::getOperationName());
  rootState.addAttribute("dataflow",
                         identityAttr(&context, problem.dataflowIdentity()));
  rootState.addAttribute("fabric",
                         identityAttr(&context, problem.fabricIdentity()));
  llvm::SmallVector<mlir::Attribute> importAttrs;
  for (const ArtifactRootReference &reference : imports)
    importAttrs.push_back(rootReferenceAttr(&context, reference));
  rootState.addAttribute("spatial_mapping_imports",
                         builder.getArrayAttr(importAttrs));
  llvm::SmallVector<mlir::Attribute> rootAttrs;
  for (const auto &root : problem.rootThreadLaunches()) {
    auto attribute = dataflowRefAttr<::mapping::RootThreadLaunchRefAttr>(
        &context, problem.dataflowIdentity(), root);
    if (!attribute)
      return attribute.takeError();
    rootAttrs.push_back(*attribute);
  }
  rootState.addAttribute("root_thread_launches",
                         builder.getArrayAttr(rootAttrs));
  rootState.addRegion();
  mlir::OwningOpRef<mlir::Operation *> result(builder.create(rootState));
  auto root = mlir::cast<::mapping::SystemOp>(result.get());
  root.getBody().emplaceBlock();

  std::map<std::vector<std::uint8_t>,
           std::vector<std::pair<::loom::mapping::SystemPresburgerCell,
                                 ::mapping::FabricAccCoreOccurrenceRefAttr>>>
      threadClauses;
  std::map<std::vector<std::uint8_t>, ::dataflow::RootThreadLaunchRef>
      threadKeys;
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.threadDecisions())) {
    auto key = ::dataflow::encodeDataflowReference(problem.dataflowIdentity(),
                                                   frozen.root);
    if (!key)
      return key.takeError();
    threadKeys.emplace(*key, frozen.root);
    threadClauses[*key].push_back(
        {frozen.cell,
         accCoreAttr(&context, candidate.selectedAccCore(
                                   static_cast<PnrIndex>(decision)))});
  }
  for (const auto &[keyBytes, clauses] : threadClauses) {
    auto reference = threadKeys.find(keyBytes);
    if (reference == threadKeys.end())
      return invalid("thread binding lost its Dataflow-owned key");
    auto key = dataflowRefAttr<::mapping::RootThreadLaunchRefAttr>(
        &context, problem.dataflowIdentity(), reference->second);
    if (!key)
      return key.takeError();
    auto binding = createBinding<::mapping::ThreadExecutionBindingOp,
                                 ::mapping::RootThreadLaunchRefAttr,
                                 ::mapping::ThreadPresburgerClauseOp,
                                 ::mapping::FabricAccCoreOccurrenceRefAttr>(
        builder, location, root.getBody().front(), *key, clauses);
    if (!binding)
      return binding.takeError();
  }

  std::map<std::vector<std::uint8_t>,
           std::vector<std::pair<::loom::mapping::SystemPresburgerCell,
                                 ::mapping::SpatialMappingImportRefAttr>>>
      graphClauses;
  std::map<std::vector<std::uint8_t>, ::dataflow::RootedGraphLaunchRef>
      graphKeys;
  for (const auto &[decision, frozen] :
       llvm::enumerate(problem.graphDecisions())) {
    auto key = ::dataflow::encodeDataflowReference(problem.dataflowIdentity(),
                                                   frozen.launch);
    if (!key)
      return key.takeError();
    graphKeys.emplace(*key, frozen.launch);
    const ArtifactRootReference &selected =
        candidate.selectedSpatialMapping(static_cast<PnrIndex>(decision));
    auto imported = importOrdinals.find(selected);
    if (imported == importOrdinals.end())
      return invalid("selected SpatialMapping is absent from the import set");
    graphClauses[*key].push_back(
        {frozen.cell, ::mapping::SpatialMappingImportRefAttr::get(
                          &context, imported->second)});
  }
  for (const auto &[keyBytes, clauses] : graphClauses) {
    auto reference = graphKeys.find(keyBytes);
    if (reference == graphKeys.end())
      return invalid("graph binding lost its Dataflow-owned key");
    auto key = dataflowRefAttr<::mapping::RootedGraphLaunchRefAttr>(
        &context, problem.dataflowIdentity(), reference->second);
    if (!key)
      return key.takeError();
    auto binding = createBinding<::mapping::GraphExecutionBindingOp,
                                 ::mapping::RootedGraphLaunchRefAttr,
                                 ::mapping::GraphPresburgerClauseOp,
                                 ::mapping::SpatialMappingImportRefAttr>(
        builder, location, root.getBody().front(), *key, clauses);
    if (!binding)
      return binding.takeError();
  }
  return result;
}

} // namespace loom::pnr
