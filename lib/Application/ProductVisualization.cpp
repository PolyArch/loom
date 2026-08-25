#include "Application/ProductVisualization.h"

#include "ADG/Export.h"
#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Deployment/Deployment.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
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

void writeReference(llvm::json::OStream &json, llvm::StringRef name,
                    const ArtifactRootReference &reference) {
  json.attributeObject(
      name, [&] { writeArtifactRootReferenceJsonFields(json, reference); });
}

void writeAllocations(
    llvm::json::OStream &json, llvm::StringRef name,
    llvm::ArrayRef<pnr::ResourceTimeRegionAllocation> allocations) {
  json.attributeArray(name, [&] {
    for (const pnr::ResourceTimeRegionAllocation &allocation : allocations) {
      json.object([&] {
        json.attribute("region_artifact",
                       formatArtifactIdentityHex(allocation.region.artifact));
        json.attribute("region_entity", allocation.region.entity.value());
        json.attributeArray("resources", [&] {
          for (const auto &resource : allocation.resources)
            json.value(llvm::toHex(fabric::canonicalFabricBytes(resource),
                                   /*LowerCase=*/true));
        });
      });
    }
  });
}

void writeSpectrumSummary(llvm::json::OStream &json, llvm::StringRef name,
                          const dse::ResourceTimeSpectrumFunnelResult &result) {
  const auto &spectrum =
      std::get<dse::VerifiedResourceTimeSpectrum>(result.verification);
  json.attributeObject(name, [&] {
    writeReference(json, "dataflow", spectrum.dataflow);
    writeReference(json, "fabric", spectrum.fabric);
    json.attributeArray("scenarios", [&] {
      for (const dse::VerifiedResourceTimeSpectrumScenario &scenario :
           spectrum.scenarios) {
        json.object([&] {
          json.attribute("ordinal", scenario.scenarioOrdinal);
          json.attribute("spectrum_class",
                         dse::toString(scenario.spectrumClass));
          json.attribute("peak_concurrent_regions",
                         scenario.peakConcurrentRegions);
          json.attribute("makespan_picoseconds", scenario.makespanPicoseconds);
          writeReferenceArray(json, "system_mappings", scenario.systemMappings);
          json.attribute("state_count", scenario.states.size());
          json.attribute("active_state_count",
                         llvm::count_if(scenario.states, [](const auto &state) {
                           return !state.active.empty();
                         }));
        });
      }
    });
  });
}

llvm::Expected<std::vector<std::string>>
verifyResourceTimeEvidence(const ApplicationDeploymentArtifacts &application,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs) {
  if (!application.resourceTimeTransitionGraph) {
    if (!application.resourceTimeTransitions.empty())
      return visualizationError("resource-time edge evidence has no finite "
                                "transition graph");
    return std::vector<std::string>{};
  }
  const pnr::ResourceTimeTransitionGraph &graph =
      *application.resourceTimeTransitionGraph;
  if (llvm::Error error =
          pnr::verifyResourceTimeTransitionGraph(graph, artifacts, blobs))
    return visualizationError("resource-time transition graph failed "
                              "independent closure: " +
                              llvm::toString(std::move(error)));
  if (graph.transitions.size() != application.resourceTimeTransitions.size())
    return visualizationError("resource-time graph and edge evidence counts "
                              "disagree");

  std::vector<std::string> triggers;
  triggers.reserve(application.resourceTimeTransitions.size());
  for (const ApplicationResourceTimeTransitionEvidence &evidence :
       application.resourceTimeTransitions) {
    if (llvm::Error error = pnr::verifyResourceTimeTransitionClosure(
            evidence.transition, artifacts, blobs))
      return visualizationError("resource-time transition evidence failed "
                                "independent closure: " +
                                llvm::toString(std::move(error)));
    for (const pnr::ResourceTimeTransitionEndpointReference *endpoint :
         {&evidence.transition.parent, &evidence.transition.child}) {
      if (!llvm::is_contained(graph.endpoints, *endpoint))
        return visualizationError("resource-time transition lost an exact "
                                  "endpoint catalog member");
    }
    if (llvm::count_if(graph.transitions, [&](const auto &candidate) {
          return candidate.parent == evidence.transition.parent &&
                 candidate.child == evidence.transition.child &&
                 candidate.trigger == evidence.transition.trigger;
        }) != 1)
      return visualizationError("resource-time edge evidence is not an exact "
                                "finite-graph member");
    const auto *parent = std::get_if<dse::VerifiedResourceTimeSpectrum>(
        &evidence.parentSpectrum.verification);
    const auto *child = std::get_if<dse::VerifiedResourceTimeSpectrum>(
        &evidence.childSpectrum.verification);
    if (!parent || !child)
      return visualizationError("resource-time transition retained an "
                                "unverified endpoint spectrum");
    const bool parentContainsEdge =
        llvm::any_of(parent->scenarios, [&](const auto &scenario) {
          return llvm::any_of(
              scenario.transitions.transitions, [&](const auto &candidate) {
                return candidate.parent == evidence.transition.parent &&
                       candidate.child == evidence.transition.child &&
                       candidate.status == evidence.transition.status;
              });
        });
    if (!parentContainsEdge)
      return visualizationError("parent spectrum does not contain its exact "
                                "resource-time transition");
    const bool childCarriesActiveWork =
        llvm::any_of(child->scenarios, [](const auto &scenario) {
          return llvm::any_of(scenario.states, [](const auto &state) {
            return !state.active.empty();
          });
        });
    if (!childCarriesActiveWork)
      return visualizationError("child spectrum carries no active Mapping "
                                "evidence");
    if (!evidence.transition.safePoint)
      return visualizationError("verified resource-time transition lost its "
                                "safe point");
    auto trigger = dataflow::encodeDataflowReference(
        evidence.transition.safePoint->artifact.artifact,
        evidence.transition.trigger);
    if (!trigger)
      return visualizationError("cannot encode a resource-time trigger: " +
                                llvm::toString(trigger.takeError()));
    triggers.push_back(llvm::toHex(*trigger, /*LowerCase=*/true));
  }
  return triggers;
}

llvm::Error writeBundle(llvm::StringRef destination,
                        const fabric::FinalizedFabricRoot &system,
                        const PreparedApplicationBuild &prepared,
                        const ApplicationMappingExecution &execution,
                        const ApplicationDeploymentArtifacts &deployment,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  if (!execution.provenance.pairDecision)
    return visualizationError("completed Mapping has no pair decision");
  auto transitionTriggers =
      verifyResourceTimeEvidence(deployment, artifacts, blobs);
  if (!transitionTriggers)
    return transitionTriggers.takeError();

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
      json.attribute("version", "1.1");
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
      json.attributeArray("resource_time_endpoints", [&] {
        if (deployment.resourceTimeTransitionGraph)
          for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
               deployment.resourceTimeTransitionGraph->endpoints)
            json.object([&] {
              writeReference(json, "mapping", endpoint.mapping);
              writeReference(json, "deployment", *endpoint.deployment);
            });
      });
      json.attributeArray("resource_time_transitions", [&] {
        for (const auto indexed :
             llvm::enumerate(deployment.resourceTimeTransitions)) {
          const ApplicationResourceTimeTransitionEvidence &evidence =
              indexed.value();
          const pnr::ResourceTimeTransition &transition = evidence.transition;
          json.object([&] {
            json.attribute("trigger", (*transitionTriggers)[indexed.index()]);
            json.attributeObject("safe_point", [&] {
              writeReference(json, "artifact", transition.safePoint->artifact);
              json.attribute("kind", pnr::resourceTimeSafePointKindSpelling(
                                         transition.safePoint->kind));
            });
            json.attributeObject("parent", [&] {
              writeReference(json, "mapping", transition.parent.mapping);
              writeReference(json, "deployment", *transition.parent.deployment);
            });
            json.attributeObject("child", [&] {
              writeReference(json, "mapping", transition.child.mapping);
              writeReference(json, "deployment", *transition.child.deployment);
            });
            writeAllocations(json, "before_active", transition.beforeActive);
            writeAllocations(json, "after_active", transition.afterActive);
            json.attributeArray("completed_before", [&] {
              for (const auto root : transition.completedBefore) {
                json.object([&] {
                  json.attribute("artifact",
                                 formatArtifactIdentityHex(root.artifact));
                  json.attribute("entity", root.entity.value());
                });
              }
            });
            writeReferenceArray(json, "before_live_work",
                                transition.beforeLiveWork);
            writeReferenceArray(json, "after_live_work",
                                transition.afterLiveWork);
            if (transition.tokenLiveStateCorrespondence)
              writeReference(json, "token_live_state_correspondence",
                             *transition.tokenLiveStateCorrespondence);
            else
              json.attribute("token_live_state_correspondence", nullptr);
            json.attribute(
                "resource_delta",
                formatComponentViewDigestHex(*transition.resourceDeltaDigest));
            json.attribute("configuration_delta",
                           formatComponentViewDigestHex(
                               *transition.configurationDeltaDigest));
            json.attribute("route_delta", formatComponentViewDigestHex(
                                              *transition.routeDeltaDigest));
            json.attribute("reprogramming_time_picoseconds",
                           *transition.reprogrammingTimePicoseconds);
            json.attribute("migration_time_picoseconds",
                           *transition.migrationTimePicoseconds);
            json.attribute("status", pnr::resourceTimeTransitionStatusSpelling(
                                         transition.status));
            writeSpectrumSummary(json, "parent_spectrum",
                                 evidence.parentSpectrum);
            writeSpectrumSummary(json, "child_spectrum",
                                 evidence.childSpectrum);
          });
        }
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

llvm::Error exportProductVisualization(
    llvm::StringRef destination, const fabric::FinalizedFabricRoot &system,
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mapping,
    const ApplicationDeploymentArtifacts &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
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
                     artifacts, blobs);
}

} // namespace loom::application
