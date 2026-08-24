#include "Application/ProductVisualization.h"

#include "ADG/Export.h"
#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

llvm::Error visualizationError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "loom_visualization_export_failed: " +
                                     message);
}

void canonicalize(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

void writeReferenceArray(llvm::json::OStream &json, llvm::StringRef name,
                         llvm::ArrayRef<ArtifactRootReference> references) {
  json.attributeArray(name, [&] {
    for (const ArtifactRootReference &reference : references)
      json.object(
          [&] { writeArtifactRootReferenceJsonFields(json, reference); });
  });
}

llvm::Error writeBundle(llvm::StringRef destination,
                        const fabric::FinalizedFabricRoot &system,
                        const PreparedApplicationBuild &prepared,
                        const ApplicationMappingExecution &execution,
                        const ApplicationDeploymentArtifacts &deployment,
                        const ArtifactStore &artifacts) {
  if (!execution.provenance.pairDecision)
    return visualizationError("completed Mapping has no pair decision");

  std::vector<ArtifactRootReference> structuredPrograms;
  std::vector<ArtifactRootReference> dataflows;
  for (const PreparedApplicationSoftware &software : prepared.software) {
    auto structured = frontend::importStructuredProgram(
        software.compilation.structuredProgram, artifacts);
    if (!structured)
      return visualizationError(
          "cannot strictly import a Structured Program: " +
          llvm::toString(structured.takeError()));
    auto importedDataflow = dataflow::importCanonicalDataflow(
        software.compilation.canonicalDataflow, artifacts);
    if (!importedDataflow)
      return visualizationError(
          "cannot strictly import a Canonical Dataflow: " +
          llvm::toString(importedDataflow.takeError()));
    structuredPrograms.push_back(software.compilation.structuredProgram);
    dataflows.push_back(software.compilation.canonicalDataflow);
  }
  canonicalize(structuredPrograms);
  canonicalize(dataflows);

  std::vector<ArtifactRootReference> systemMappings;
  std::vector<ArtifactRootReference> spatialMappings;
  std::vector<ArtifactRootReference> techMappings;
  std::vector<ArtifactRootReference> runtimeEvidence;
  for (const ApplicationMappingCandidateOutcome &outcome :
       execution.candidateOutcomes) {
    systemMappings.insert(systemMappings.end(), outcome.systemMappings.begin(),
                          outcome.systemMappings.end());
    runtimeEvidence.insert(runtimeEvidence.end(),
                           outcome.runtimeEvidence.begin(),
                           outcome.runtimeEvidence.end());
  }
  canonicalize(systemMappings);
  canonicalize(runtimeEvidence);
  for (const ArtifactRootReference &reference : systemMappings) {
    auto imported = mapping::importSystemMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a SystemMapping: " +
                                llvm::toString(imported.takeError()));
    if (imported->view().fabricIdentity() != system.reference().artifact)
      return visualizationError("SystemMapping names a foreign Fabric");
    spatialMappings.insert(
        spatialMappings.end(),
        imported->view().executionBindings().spatialMappingImports().begin(),
        imported->view().executionBindings().spatialMappingImports().end());
  }
  canonicalize(spatialMappings);
  for (const ArtifactRootReference &reference : spatialMappings) {
    auto imported = mapping::importSpatialMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a SpatialMapping: " +
                                llvm::toString(imported.takeError()));
    techMappings.push_back({mapping::mappingArtifactSchema.identity.str(),
                            mapping::mappingArtifactSchema.version,
                            imported->view().techMappingIdentity()});
  }
  canonicalize(techMappings);
  for (const ArtifactRootReference &reference : techMappings) {
    auto imported = mapping::importTechMapping(reference, artifacts);
    if (!imported)
      return visualizationError("cannot strictly import a TechMapping: " +
                                llvm::toString(imported.takeError()));
  }
  for (const ArtifactRootReference &reference : runtimeEvidence) {
    auto evidence = artifacts.get(reference);
    if (!evidence)
      return visualizationError("runtime Evidence reference is unavailable: " +
                                llvm::toString(evidence.takeError()));
  }

  llvm::SmallString<256> bundlePath(destination);
  llvm::sys::path::append(bundlePath, "viz.bundle.json");
  llvm::SmallString<256> temporaryModel(bundlePath);
  temporaryModel.append(".tmp-%%%%%%");
  auto temporary = llvm::sys::fs::TempFile::create(
      temporaryModel, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporary)
    return visualizationError("cannot create bundle output: " +
                              llvm::toString(temporary.takeError()));
  llvm::scope_exit discard([&] { llvm::consumeError(temporary->discard()); });
  {
    llvm::raw_fd_ostream output(temporary->FD, false);
    llvm::json::OStream json(output, 2);
    json.object([&] {
      json.attribute("schema", "loom.visualization_bundle");
      json.attribute("version", "1.0");
      json.attributeObject("fabric", [&] {
        writeArtifactRootReferenceJsonFields(json, system.reference());
      });
      writeReferenceArray(json, "structured_programs", structuredPrograms);
      writeReferenceArray(json, "canonical_dataflows", dataflows);
      writeReferenceArray(json, "tech_mappings", techMappings);
      writeReferenceArray(json, "spatial_mappings", spatialMappings);
      writeReferenceArray(json, "system_mappings", systemMappings);
      writeReferenceArray(json, "runtime_evidence", runtimeEvidence);
      json.attribute("pair_decision", projectApplicationPairDecisionJson(
                                          *execution.provenance.pairDecision));
      json.attributeObject("configuration_abi", [&] {
        writeArtifactRootReferenceJsonFields(json, deployment.configurationAbi);
      });
      writeReferenceArray(
          json, "configuration_images",
          deployment.deployment.deployment().configurationImages());
      json.attributeObject("deployment", [&] {
        writeArtifactRootReferenceJsonFields(json,
                                             deployment.deployment.reference());
      });
    });
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      return visualizationError("cannot write bundle output: " +
                                error.message());
    }
  }
  if (llvm::Error error = temporary->keep(bundlePath))
    return visualizationError("cannot publish bundle output: " +
                              llvm::toString(std::move(error)));
  discard.release();
  return llvm::Error::success();
}

} // namespace

llvm::Error
exportProductVisualization(llvm::StringRef destination,
                           const fabric::FinalizedFabricRoot &system,
                           const PreparedApplicationBuild &prepared,
                           const ApplicationMappingExecution &mapping,
                           const ApplicationDeploymentArtifacts &deployment,
                           const ArtifactStore &artifacts, const BlobStore &) {
  if (destination.empty())
    return visualizationError("destination is empty");
  if (std::error_code error = llvm::sys::fs::create_directories(destination))
    return visualizationError("cannot create destination: " + error.message());
  if (!llvm::sys::fs::is_directory(destination))
    return visualizationError("destination is not a directory");

  llvm::SmallString<256> fabricBase(destination);
  llvm::sys::path::append(fabricBase, "fabric");
  if (llvm::Error error =
          adg::exportFabricDesign(system, artifacts, fabricBase))
    return visualizationError(llvm::toString(std::move(error)));
  return writeBundle(destination, system, prepared, mapping, deployment,
                     artifacts);
}

} // namespace loom::application
