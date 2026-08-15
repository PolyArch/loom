#include "DeploymentInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/InstructionCoreBinary.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/IR/MappingSchema.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::deployment::detail {
namespace {

template <class... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "deployment_runtime_image_invalid: " +
                                     message);
}

void writeRootReference(llvm::json::OStream &json,
                        const ArtifactRootReference &reference) {
  json.attribute("schema", reference.schemaIdentity);
  json.attribute("schema_version",
                 formatSchemaVersion(reference.schemaVersion));
  json.attribute("artifact", formatArtifactIdentityHex(reference.artifact));
}

class ReferenceEncoder final {
public:
  explicit ReferenceEncoder(ArtifactIdentity dataflowIdentity)
      : dataflowIdentity_(std::move(dataflowIdentity)) {}

  template <typename Ref> std::string dataflow(const Ref &reference) {
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflowIdentity_, reference);
    if (!bytes) {
      record(bytes.takeError());
      return {};
    }
    return formatArtifactLocalPayloadHex(*bytes);
  }

  std::string context(const mapping::ExecutionContextKey &key) {
    auto bytes = mapping::encodeExecutionContextKey(key);
    if (!bytes) {
      record(bytes.takeError());
      return {};
    }
    return formatArtifactLocalPayloadHex(*bytes);
  }

  std::string
  serviceObligation(const mapping::SystemServiceObligationKey &key) {
    auto bytes =
        mapping::encodeSystemServiceObligationKey(dataflowIdentity_, key);
    if (!bytes) {
      record(bytes.takeError());
      return {};
    }
    return formatArtifactLocalPayloadHex(*bytes);
  }

  std::string serviceSelection(const mapping::ServicePlanSelectionKey &key) {
    auto bytes = mapping::encodeServicePlanSelectionKey(dataflowIdentity_, key);
    if (!bytes) {
      record(bytes.takeError());
      return {};
    }
    return formatArtifactLocalPayloadHex(*bytes);
  }

  bool failed() const { return diagnostic_.has_value(); }

  void recordError(llvm::Error error) { record(std::move(error)); }

  llvm::Error takeError() {
    if (!diagnostic_)
      return llvm::Error::success();
    return invalid(*diagnostic_);
  }

private:
  void record(llvm::Error error) {
    if (!diagnostic_)
      diagnostic_ = llvm::toString(std::move(error));
    else
      llvm::consumeError(std::move(error));
  }

  ArtifactIdentity dataflowIdentity_;
  std::optional<std::string> diagnostic_;
};

template <typename Ref>
void writeDataflowReference(llvm::json::OStream &json,
                            ReferenceEncoder &encoder, llvm::StringRef kind,
                            const Ref &reference) {
  json.object([&] {
    json.attribute("kind", kind);
    json.attribute("reference", encoder.dataflow(reference));
  });
}

void writeContext(llvm::json::OStream &json, ReferenceEncoder &encoder,
                  const mapping::ExecutionContextKey &context) {
  json.object([&] {
    if (const auto *instruction =
            std::get_if<mapping::InstructionExecutionContextKey>(&context)) {
      json.attribute("kind", "instruction");
      json.attribute("acc_core_ref",
                     formatArtifactLocalPayloadHex(
                         fabric::canonicalFabricBytes(instruction->accCore)));
    } else {
      const auto &spatial =
          std::get<mapping::SpatialExecutionContextKey>(context);
      json.attribute("kind", "spatial");
      json.attribute("acc_core_ref",
                     formatArtifactLocalPayloadHex(
                         fabric::canonicalFabricBytes(spatial.accCore)));
      json.attribute("spatial_mapping_artifact",
                     formatArtifactIdentityHex(spatial.spatialMapping));
    }
    json.attribute("canonical_key", encoder.context(context));
  });
}

void writeIntegerRows(llvm::json::OStream &json, llvm::StringRef name,
                      llvm::ArrayRef<std::vector<std::int64_t>> rows) {
  json.attributeArray(name, [&] {
    for (const std::vector<std::int64_t> &row : rows)
      json.array([&] {
        for (std::int64_t value : row)
          json.value(value);
      });
  });
}

void writeCell(llvm::json::OStream &json,
               const mapping::SystemPresburgerCell &cell) {
  json.object([&] {
    json.attribute("dimension_count", cell.dimensionCount);
    json.attribute("symbol_count", cell.symbolCount);
    json.attribute("local_count", cell.localCount);
    writeIntegerRows(json, "equalities", cell.equalities);
    writeIntegerRows(json, "inequalities", cell.inequalities);
  });
}

std::string cellKey(const mapping::SystemPresburgerCell &cell) {
  std::string key;
  llvm::raw_string_ostream output(key);
  output << cell.dimensionCount << ':' << cell.symbolCount << ':'
         << cell.localCount << ':';
  for (const auto &row : cell.equalities) {
    output << '=';
    for (std::int64_t value : row)
      output << value << ',';
  }
  for (const auto &row : cell.inequalities) {
    output << '>';
    for (std::int64_t value : row)
      output << value << ',';
  }
  return output.str();
}

void writeCells(llvm::json::OStream &json, llvm::StringRef name,
                llvm::ArrayRef<mapping::SystemPresburgerCell> cells) {
  json.attributeArray(name, [&] {
    for (const mapping::SystemPresburgerCell &cell : cells)
      writeCell(json, cell);
  });
}

void writeEvent(llvm::json::OStream &json, ReferenceEncoder &encoder,
                const dataflow::EventFamilyKey &event) {
  json.object([&] {
    if (const auto *transfer =
            std::get_if<dataflow::StaticTransferEventRef>(&event)) {
      if (const auto *produced =
              std::get_if<dataflow::ProducedTransferEventRef>(transfer)) {
        json.attribute("kind", "transfer_produced");
        json.attribute("terminal_ref", encoder.dataflow(produced->terminal));
      } else {
        const auto &consumed =
            std::get<dataflow::ConsumedTransferEventRef>(*transfer);
        json.attribute("kind", "transfer_consumed");
        json.attribute("terminal_ref", encoder.dataflow(consumed.terminal));
      }
    } else {
      const auto &transition =
          std::get<dataflow::ContextualActorTransitionEventRef>(event);
      json.attribute("kind", "actor_transition");
      json.attribute("transition_ref", encoder.dataflow(transition));
    }
    json.attribute("canonical_key", encoder.dataflow(event));
  });
}

std::string eventKey(ReferenceEncoder &encoder,
                     const dataflow::EventFamilyKey &event) {
  return encoder.dataflow(event);
}

dataflow::RootThreadLaunchRef
rootOf(const dataflow::CanonicalProducerTerminalRef &terminal) {
  return std::visit(
      Overloaded{
          [](const dataflow::RootThreadBoundarySourceRef &source) {
            return std::visit([](const auto &value) { return value.launch; },
                              source.transfer);
          },
          [](const dataflow::GraphLaunchBoundarySourceRef &source) {
            return std::visit(
                [](const auto &value) { return value.launch.rootThreadLaunch; },
                source.transfer);
          },
          [](const dataflow::ChannelProducerTerminalRef &source) {
            return std::visit(
                Overloaded{
                    [](const dataflow::GraphStreamOutputProducerRef &value) {
                      return value.launch.rootThreadLaunch;
                    },
                    [](const dataflow::ThreadChannelSendSiteRef &value) {
                      return value.launch;
                    }},
                source.producer);
          }},
      terminal);
}

dataflow::RootThreadLaunchRef
rootOf(const dataflow::CanonicalSinkTerminalRef &terminal) {
  return std::visit(
      Overloaded{
          [](const dataflow::RootThreadBoundarySinkRef &sink) {
            return std::visit([](const auto &value) { return value.launch; },
                              sink.transfer);
          },
          [](const dataflow::GraphLaunchBoundarySinkRef &sink) {
            return std::visit(
                [](const auto &value) { return value.launch.rootThreadLaunch; },
                sink.transfer);
          },
          [](const dataflow::ChannelConsumerTerminalRef &sink) {
            return std::visit(
                Overloaded{
                    [](const dataflow::GraphStreamInputConsumerRef &value) {
                      return value.launch.rootThreadLaunch;
                    },
                    [](const dataflow::ThreadChannelReceiveSiteRef &value) {
                      return value.launch;
                    }},
                sink.consumer);
          }},
      terminal);
}

dataflow::RootThreadLaunchRef rootOf(const dataflow::EventFamilyKey &event) {
  if (const auto *transition =
          std::get_if<dataflow::ContextualActorTransitionEventRef>(&event))
    return transition->actor.launch.rootThreadLaunch;
  const auto &transfer = std::get<dataflow::StaticTransferEventRef>(event);
  if (const auto *produced =
          std::get_if<dataflow::ProducedTransferEventRef>(&transfer))
    return rootOf(produced->terminal);
  return rootOf(
      std::get<dataflow::ConsumedTransferEventRef>(transfer).terminal);
}

dataflow::RootThreadLaunchRef
rootOf(const mapping::ServicePlanSelectionAnchor &anchor,
       const mapping::SystemServiceObligationKey &obligation) {
  if (const auto *member =
          std::get_if<mapping::ServiceMemberPlanSelectionAnchor>(&anchor)) {
    return std::visit(
        Overloaded{[&](const dataflow::MessageTransferMemberRef &) {
                     return rootOf(
                         std::get<mapping::TransferObligationFamilyKey>(
                             obligation));
                   },
                   [](const dataflow::AddressedMemoryActorMemberRef &value) {
                     return value.actor.launch.rootThreadLaunch;
                   },
                   [](const dataflow::FenceActorMemberRef &value) {
                     return value.actor.launch.rootThreadLaunch;
                   }},
        member->member);
  }
  return std::get<mapping::MemoryExposurePlanSelectionAnchor>(anchor)
      .exposure.launch.rootThreadLaunch;
}

void writeUsePatternValue(llvm::json::OStream &json,
                          const ::fabric::UsePatternValue &value) {
  json.object([&] {
    const auto &tag = std::get<::fabric::PhysicalTagPatternValue>(value);
    llvm::SmallString<64> spelling;
    tag.value.toStringUnsigned(spelling, 16);
    json.attribute("kind", "physical_tag");
    json.attribute("bit_width", tag.value.getBitWidth());
    json.attribute("value_hex", spelling);
  });
}

void writeUsePatternValues(llvm::json::OStream &json, llvm::StringRef name,
                           llvm::ArrayRef<::fabric::UsePatternValue> values) {
  json.attributeArray(name, [&] {
    for (const ::fabric::UsePatternValue &value : values)
      writeUsePatternValue(json, value);
  });
}

std::string
usePatternValuesKey(llvm::ArrayRef<::fabric::UsePatternValue> values) {
  std::string key;
  llvm::raw_string_ostream output(key);
  for (const ::fabric::UsePatternValue &value : values) {
    const auto &tag = std::get<::fabric::PhysicalTagPatternValue>(value);
    llvm::SmallString<64> spelling;
    tag.value.toStringUnsigned(spelling, 16);
    output << tag.value.getBitWidth() << ':' << spelling << ';';
  }
  return output.str();
}

struct ActivationGroup final {
  mapping::ExecutionContextKey context;
  std::vector<mapping::SystemPresburgerCell> relationDomain;
  std::vector<dataflow::EventFamilyKey> triggerAlternatives;
  std::vector<const mapping::SystemResourceActivationProjection *> members;
};

std::string
activationKey(ReferenceEncoder &encoder,
              const mapping::SystemResourceActivationProjection &activation) {
  std::string key = formatArtifactLocalPayloadHex(
      fabric::canonicalFabricBytes(activation.physicalOwner));
  key += ':' + std::to_string(activation.usePatternOrdinal) + ':';
  key += usePatternValuesKey(activation.parameters);
  key += ':' + usePatternValuesKey(activation.sharingAssignments);
  for (const auto &claim : activation.capacityClaims)
    key += ':' + std::to_string(claim.capacityCellOrdinal) + '=' +
           std::to_string(claim.amount);
  for (const auto &point : activation.causalRelease) {
    key += '|';
    for (const auto &event : point.alternatives)
      key += eventKey(encoder, event) + ',';
    if (point.guaranteedOffset)
      key += formatArtifactLocalPayloadHex(*point.guaranteedOffset);
  }
  return key;
}

llvm::Expected<std::vector<ActivationGroup>>
buildActivationGroups(const mapping::SystemMappingClosureProjection &closure,
                      ReferenceEncoder &encoder) {
  std::vector<ActivationGroup> groups;
  for (const auto &activation : closure.resourceActivations) {
    if (activation.relationDomain.empty())
      return invalid("resource activation has an empty relation domain");
    if (activation.triggerAlternatives.empty())
      return invalid("resource activation has no trigger alternative");
    auto found = llvm::find_if(groups, [&](const ActivationGroup &group) {
      return group.context == activation.context &&
             group.relationDomain == activation.relationDomain &&
             group.triggerAlternatives == activation.triggerAlternatives;
    });
    if (found == groups.end()) {
      groups.push_back({activation.context,
                        activation.relationDomain,
                        activation.triggerAlternatives,
                        {&activation}});
    } else {
      found->members.push_back(&activation);
    }
  }
  for (ActivationGroup &group : groups)
    llvm::sort(group.members, [&](const auto *lhs, const auto *rhs) {
      return activationKey(encoder, *lhs) < activationKey(encoder, *rhs);
    });
  llvm::sort(groups, [&](const ActivationGroup &lhs,
                         const ActivationGroup &rhs) {
    const std::string lhsContext = encoder.context(lhs.context);
    const std::string rhsContext = encoder.context(rhs.context);
    if (lhsContext != rhsContext)
      return lhsContext < rhsContext;
    const std::string lhsEvent = eventKey(encoder, lhs.triggerAlternatives[0]);
    const std::string rhsEvent = eventKey(encoder, rhs.triggerAlternatives[0]);
    if (lhsEvent != rhsEvent)
      return lhsEvent < rhsEvent;
    return cellKey(lhs.relationDomain[0]) < cellKey(rhs.relationDomain[0]);
  });
  if (encoder.failed())
    return encoder.takeError();
  return groups;
}

std::string caseKey(ReferenceEncoder &encoder, const ActivationGroup &group) {
  std::string key;
  for (const auto *member : group.members)
    key += activationKey(encoder, *member) + '#';
  return key;
}

void writeActivationMember(
    llvm::json::OStream &json, ReferenceEncoder &encoder,
    const mapping::SystemResourceActivationProjection &member,
    std::uint64_t ordinal) {
  json.object([&] {
    json.attribute("activation_member_ordinal", ordinal);
    json.attribute("physical_owner_kind",
                   static_cast<std::uint32_t>(member.physicalOwner.kind()));
    json.attribute("physical_owner_ref",
                   formatArtifactLocalPayloadHex(
                       fabric::canonicalFabricBytes(member.physicalOwner)));
    json.attribute("use_pattern_ordinal", member.usePatternOrdinal);
    writeUsePatternValues(json, "parameters", member.parameters);
    writeUsePatternValues(json, "sharing_assignments",
                          member.sharingAssignments);
    json.attributeArray("capacity_claims", [&] {
      for (const auto &claim : member.capacityClaims)
        json.object([&] {
          json.attribute("capacity_cell_ordinal", claim.capacityCellOrdinal);
          json.attribute("amount", claim.amount);
        });
    });
  });
}

void writeReleaseRule(llvm::json::OStream &json, ReferenceEncoder &encoder,
                      const mapping::SystemResourceActivationProjection &member,
                      std::uint64_t ordinal) {
  json.object([&] {
    json.attribute("activation_member_ordinal", ordinal);
    json.attribute("fabric_intrinsic_release", true);
    if (!member.causalRelease.empty())
      json.attributeObject("causal_release", [&] {
        json.attributeArray("all_of", [&] {
          for (const auto &point : member.causalRelease)
            json.object([&] {
              json.attributeArray("alternatives", [&] {
                for (const auto &event : point.alternatives)
                  writeEvent(json, encoder, event);
              });
              if (point.guaranteedOffset)
                json.attribute(
                    "guaranteed_offset",
                    formatArtifactLocalPayloadHex(*point.guaranteedOffset));
            });
        });
      });
  });
}

CanonicalSemanticBytes finishJson(llvm::SmallVectorImpl<char> &storage) {
  return CanonicalSemanticBytes(
      std::vector<std::uint8_t>(storage.begin(), storage.end()));
}

llvm::Expected<CanonicalSemanticBytes>
deriveAdmissionImage(const ArtifactRootReference &systemMapping,
                     const mapping::SystemMappingClosureProjection &closure,
                     ReferenceEncoder &encoder,
                     llvm::ArrayRef<ActivationGroup> groups) {
  struct ContextRow final {
    mapping::ExecutionContextKey context;
    std::vector<const ActivationGroup *> groups;
  };
  struct EventRow final {
    dataflow::EventFamilyKey event;
    std::vector<ContextRow> contexts;
  };

  std::vector<EventRow> rows;
  for (const ActivationGroup &group : groups) {
    for (const dataflow::EventFamilyKey &event : group.triggerAlternatives) {
      auto row = llvm::find_if(rows, [&](const EventRow &candidate) {
        return candidate.event == event;
      });
      if (row == rows.end()) {
        rows.push_back({event, {}});
        row = std::prev(rows.end());
      }
      auto context = llvm::find_if(row->contexts, [&](const ContextRow &entry) {
        return entry.context == group.context;
      });
      if (context == row->contexts.end()) {
        row->contexts.push_back({group.context, {}});
        context = std::prev(row->contexts.end());
      }
      context->groups.push_back(&group);
    }
  }
  llvm::sort(rows, [&](const EventRow &lhs, const EventRow &rhs) {
    return eventKey(encoder, lhs.event) < eventKey(encoder, rhs.event);
  });
  for (EventRow &row : rows) {
    llvm::sort(row.contexts, [&](const ContextRow &lhs, const ContextRow &rhs) {
      return encoder.context(lhs.context) < encoder.context(rhs.context);
    });
  }
  if (encoder.failed())
    return encoder.takeError();

  llvm::SmallString<16384> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", admissionImageSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(admissionImageSchema.version));
    json.attributeObject("source_system_mapping_ref",
                         [&] { writeRootReference(json, systemMapping); });
    json.attributeObject("payload", [&] {
      json.attributeArray("capacity_cells", [&] {
        for (const auto indexed : llvm::enumerate(closure.capacityCells)) {
          const std::uint64_t ordinal = indexed.index();
          const auto &cell = indexed.value();
          json.object([&] {
            json.attribute("capacity_cell_ordinal", ordinal);
            json.attribute(
                "physical_owner_kind",
                static_cast<std::uint32_t>(cell.physicalOwner.kind()));
            json.attribute(
                "physical_owner_ref",
                formatArtifactLocalPayloadHex(
                    fabric::canonicalFabricBytes(cell.physicalOwner)));
            json.attribute("state_ordinal", cell.state.ordinal());
            json.attribute("capacity_dimension_ordinal",
                           cell.dimension.ordinal());
            json.attribute("capacity", cell.capacity);
            json.attribute("baseline_occupancy", cell.baselineOccupancy);
          });
        }
      });
      json.attributeArray("rows", [&] {
        for (const EventRow &row : rows)
          json.object([&] {
            json.attributeBegin("event_family_key");
            writeEvent(json, encoder, row.event);
            json.attributeEnd();
            json.attributeArray("contexts", [&] {
              for (const ContextRow &context : row.contexts)
                json.object([&] {
                  json.attributeBegin("execution_context_key");
                  writeContext(json, encoder, context.context);
                  json.attributeEnd();

                  std::vector<std::pair<std::string, const ActivationGroup *>>
                      cases;
                  for (const ActivationGroup *group : context.groups)
                    cases.emplace_back(caseKey(encoder, *group), group);
                  llvm::sort(cases, [](const auto &lhs, const auto &rhs) {
                    return lhs.first < rhs.first;
                  });
                  std::vector<std::pair<std::string, const ActivationGroup *>>
                      uniqueCases;
                  for (const auto &entry : cases)
                    if (uniqueCases.empty() ||
                        uniqueCases.back().first != entry.first)
                      uniqueCases.push_back(entry);

                  struct RelationClause final {
                    std::string key;
                    const mapping::SystemPresburgerCell *cell = nullptr;
                    std::uint64_t caseOrdinal = 0;
                  };
                  std::vector<RelationClause> clauses;
                  for (const ActivationGroup *group : context.groups) {
                    const std::string key = caseKey(encoder, *group);
                    const auto found =
                        llvm::find_if(uniqueCases, [&](const auto &entry) {
                          return entry.first == key;
                        });
                    const std::uint64_t caseOrdinal =
                        static_cast<std::uint64_t>(found - uniqueCases.begin());
                    for (const mapping::SystemPresburgerCell &cell :
                         group->relationDomain)
                      clauses.push_back({cellKey(cell), &cell, caseOrdinal});
                  }
                  llvm::sort(clauses, [](const auto &lhs, const auto &rhs) {
                    return std::tie(lhs.key, lhs.caseOrdinal) <
                           std::tie(rhs.key, rhs.caseOrdinal);
                  });
                  json.attributeObject("parameter_relation", [&] {
                    json.attributeArray("clauses", [&] {
                      for (const RelationClause &clause : clauses)
                        json.object([&] {
                          json.attributeBegin("cell");
                          writeCell(json, *clause.cell);
                          json.attributeEnd();
                          json.attribute("admission_case_ordinal",
                                         clause.caseOrdinal);
                        });
                    });
                  });
                  json.attributeArray("cases", [&] {
                    for (const auto indexed : llvm::enumerate(uniqueCases)) {
                      const std::uint64_t caseOrdinal = indexed.index();
                      const ActivationGroup &group = *indexed.value().second;
                      json.object([&] {
                        json.attribute("admission_case_ordinal", caseOrdinal);
                        json.attributeArray("atomic_activation_set", [&] {
                          for (const auto &[memberOrdinal, member] :
                               llvm::enumerate(group.members))
                            writeActivationMember(
                                json, encoder, *member,
                                static_cast<std::uint64_t>(memberOrdinal));
                        });
                        json.attributeArray("release_rules", [&] {
                          for (const auto &[memberOrdinal, member] :
                               llvm::enumerate(group.members))
                            writeReleaseRule(
                                json, encoder, *member,
                                static_cast<std::uint64_t>(memberOrdinal));
                        });
                      });
                    }
                  });
                });
            });
          });
      });
    });
  });
  if (encoder.failed())
    return encoder.takeError();
  return finishJson(storage);
}

struct ImportedBinary final {
  FinalizedInstructionCoreBinary binary;
  FinalizedCompilerTargetBinding target;
  bool used = false;
};

llvm::Expected<std::vector<ImportedBinary>>
importBinaries(llvm::ArrayRef<ArtifactRootReference> references,
               const ArtifactRootReference &dataflowReference,
               const ArtifactStore &artifacts, const BlobStore &blobs) {
  std::vector<ImportedBinary> result;
  result.reserve(references.size());
  for (const ArtifactRootReference &reference : references) {
    auto binary = importInstructionCoreBinary(reference, artifacts, blobs);
    if (!binary)
      return binary.takeError();
    if (binary->binary().canonicalDataflow() != dataflowReference)
      return invalid("InstructionCoreBinary has a foreign Dataflow owner");
    auto target = importCompilerTargetBinding(
        binary->binary().compilerTargetBinding(), artifacts);
    if (!target)
      return target.takeError();
    result.push_back({std::move(*binary), std::move(*target), false});
  }
  return result;
}

llvm::Expected<std::pair<ImportedBinary *, std::uint64_t>> selectBinary(
    dataflow::RootThreadLaunchRef root, fabric::AccCoreOccurrenceRef core,
    const ArtifactIdentity &fabricIdentity,
    std::vector<ImportedBinary> &binaries, const ArtifactStore &artifacts) {
  ImportedBinary *selected = nullptr;
  std::uint64_t entryOrdinal = 0;
  for (ImportedBinary &candidate : binaries) {
    const auto entry =
        llvm::find_if(candidate.binary.binary().threadEntryTable(),
                      [&](const ThreadEntryBinding &binding) {
                        return binding.rootThreadLaunch == root;
                      });
    if (entry == candidate.binary.binary().threadEntryTable().end())
      continue;
    const CompilerProcessorArchitectureRef processor =
        CompilerProcessorArchitectureRef::instruction(
            {fabricIdentity, fabric::InstructionCoreContextRef{core}});
    if (llvm::Error error = requireCompilerTargetCompatibility(
            candidate.target.binding(), processor, artifacts)) {
      llvm::consumeError(std::move(error));
      continue;
    }
    if (selected)
      return invalid("more than one InstructionCoreBinary supports one exact "
                     "thread target case");
    selected = &candidate;
    entryOrdinal = entry->entryOrdinal;
  }
  if (!selected)
    return invalid("no InstructionCoreBinary supports one exact thread target "
                   "case");
  selected->used = true;
  return std::make_pair(selected, entryOrdinal);
}

void writeServiceRequirements(
    llvm::json::OStream &json, ReferenceEncoder &encoder,
    dataflow::RootThreadLaunchRef root,
    const mapping::ExecutionContextKey &context,
    llvm::ArrayRef<mapping::SystemServiceRealizationView> realizations) {
  json.array([&] {
    for (const mapping::SystemServiceRealizationView &realization :
         realizations) {
      if (std::holds_alternative<mapping::TransferObligationFamilyKey>(
              realization.key))
        continue;
      for (const mapping::SystemServicePlanSelectionView &selection :
           realization.selections) {
        if (!(selection.key.context == context) ||
            rootOf(selection.key.anchor, realization.key) != root)
          continue;
        json.object([&] {
          json.attribute("service_obligation_ref",
                         encoder.serviceObligation(realization.key));
          json.attribute("selection_ref",
                         encoder.serviceSelection(selection.key));
          json.attributeArray("clauses", [&] {
            for (const auto &clause : selection.clauses)
              json.object([&] {
                writeCells(json, "cells", clause.cells);
                json.attribute("plan_ordinal", clause.target);
              });
          });
          if (selection.defaultPlanOrdinal)
            json.attribute("default_plan_ordinal",
                           *selection.defaultPlanOrdinal);
        });
      }
    }
  });
}

void writeLongLivedActivationRefs(llvm::json::OStream &json,
                                  ReferenceEncoder &encoder,
                                  dataflow::RootThreadLaunchRef root,
                                  const mapping::ExecutionContextKey &context,
                                  llvm::ArrayRef<ActivationGroup> groups) {
  json.array([&] {
    for (const ActivationGroup &group : groups) {
      if (!(group.context == context) || group.triggerAlternatives.empty() ||
          rootOf(group.triggerAlternatives.front()) != root ||
          !llvm::any_of(group.members, [](const auto *member) {
            return !member->causalRelease.empty();
          }))
        continue;
      json.object([&] {
        json.attributeArray("trigger_alternatives", [&] {
          for (const auto &event : group.triggerAlternatives)
            writeEvent(json, encoder, event);
        });
        writeCells(json, "parameter_relation_domain", group.relationDomain);
      });
    }
  });
}

llvm::Expected<CanonicalSemanticBytes> deriveThreadDispatchImage(
    const ArtifactRootReference &systemMapping,
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::SystemMappingClosureProjection &closure,
    const ArtifactIdentity &fabricIdentity,
    std::vector<ImportedBinary> &binaries, ReferenceEncoder &encoder,
    const ArtifactStore &artifacts,
    llvm::ArrayRef<ActivationGroup> activationGroups) {
  llvm::SmallString<16384> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", threadDispatchImageSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(threadDispatchImageSchema.version));
    json.attributeObject("source_system_mapping_ref",
                         [&] { writeRootReference(json, systemMapping); });
    json.attributeObject("payload", [&] {
      json.attributeArray("rows", [&] {
        for (const auto &rootView : dataflow.rootThreadLaunches()) {
          const dataflow::RootThreadLaunchRef root = rootView.ref;
          auto logical = dataflow.projectRootThreadLogicalDomain(root);
          if (!logical) {
            encoder.recordError(logical.takeError());
            continue;
          }
          json.object([&] {
            json.attributeBegin("root_thread_launch_ref");
            writeDataflowReference(json, encoder, "root_thread_launch", root);
            json.attributeEnd();
            json.attributeObject("compiled_thread_execution_binding", [&] {
              json.attributeArray("target_domains", [&] {
                for (const auto &domain :
                     closure.executionContexts.instructionDomains) {
                  if (domain.root != root)
                    continue;
                  json.object([&] {
                    json.attributeBegin("execution_context_key");
                    writeContext(json, encoder, domain.context);
                    json.attributeEnd();
                    writeCells(json, "cells", domain.cells);
                  });
                }
              });
            });
            json.attributeObject("logical_parameter_schema", [&] {
              json.attribute("coordinate_rank", logical->coordinateRank);
              json.attributeArray("slots", [&] {
                for (const auto indexed :
                     llvm::enumerate(logical->launchParameters)) {
                  const std::uint64_t ordinal = indexed.index();
                  const mlir::Value value = indexed.value();
                  auto type = dataflow::encodeCanonicalType(value.getType());
                  if (!type) {
                    encoder.recordError(type.takeError());
                    json.value(nullptr);
                    continue;
                  }
                  json.object([&] {
                    json.attribute("kind", "launch_parameter");
                    json.attribute("ordinal", ordinal);
                    json.attribute(
                        "semantic_type",
                        formatArtifactLocalPayloadHex(type->bytes()));
                  });
                }
              });
            });
            json.attributeArray("explicit_dependencies", [&] {
              llvm::Error error = dataflow.forEachProducerTerminal(
                  root,
                  [&](const dataflow::CanonicalProducerTerminalView &terminal)
                      -> llvm::Error {
                    const auto *boundary =
                        std::get_if<dataflow::RootThreadBoundarySourceRef>(
                            &terminal.terminal);
                    if (!boundary)
                      return llvm::Error::success();
                    if (std::holds_alternative<
                            dataflow::RootThreadCompletionTransferRef>(
                            boundary->transfer))
                      return llvm::Error::success();
                    writeDataflowReference(json, encoder,
                                           "root_thread_dependency",
                                           terminal.terminal);
                    return llvm::Error::success();
                  });
              if (error)
                encoder.recordError(std::move(error));
            });
            const dataflow::CanonicalProducerTerminalRef completion =
                dataflow::RootThreadBoundarySourceRef{
                    dataflow::RootThreadBoundaryTransferRef{
                        dataflow::RootThreadCompletionTransferRef{root}}};
            json.attributeBegin("thread_completion_destination");
            writeDataflowReference(json, encoder, "root_thread_completion",
                                   completion);
            json.attributeEnd();
            json.attributeArray("target_cases", [&] {
              for (const auto &domain :
                   closure.executionContexts.instructionDomains) {
                if (domain.root != root)
                  continue;
                auto selected =
                    selectBinary(root, domain.context.accCore, fabricIdentity,
                                 binaries, artifacts);
                if (!selected) {
                  encoder.recordError(selected.takeError());
                  continue;
                }
                json.object([&] {
                  json.attributeBegin("execution_context_key");
                  writeContext(json, encoder, domain.context);
                  json.attributeEnd();
                  json.attributeObject("instruction_core_entry_ref", [&] {
                    json.attributeObject("instruction_core_binary_ref", [&] {
                      writeRootReference(json,
                                         selected->first->binary.reference());
                    });
                    json.attribute("thread_entry_ordinal", selected->second);
                  });
                  json.attributeBegin("memory_capability_requirements");
                  writeServiceRequirements(
                      json, encoder, root,
                      mapping::ExecutionContextKey{domain.context},
                      closure.serviceRealizations);
                  json.attributeEnd();
                  json.attributeBegin("long_lived_activation_uses");
                  writeLongLivedActivationRefs(
                      json, encoder, root,
                      mapping::ExecutionContextKey{domain.context},
                      activationGroups);
                  json.attributeEnd();
                });
              }
            });
          });
        }
      });
    });
  });
  if (encoder.failed())
    return encoder.takeError();
  for (const ImportedBinary &binary : binaries)
    if (!binary.used)
      return invalid("instruction_core_binary_refs contains an unused binary");
  return finishJson(storage);
}

enum class BoundaryClass : std::uint8_t { Value, Stream, Control };

std::optional<BoundaryClass>
producerBoundaryClass(const dataflow::CanonicalGraphProducerEndpointRef &ref,
                      dataflow::GraphRef graph) {
  const auto *ingress = std::get_if<dataflow::GraphIngressTokenRef>(&ref);
  if (!ingress)
    return std::nullopt;
  return std::visit(
      Overloaded{
          [&](const dataflow::GraphStartTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Control)
                       : std::nullopt;
          },
          [&](const dataflow::GraphValueInputTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Value)
                       : std::nullopt;
          },
          [&](const dataflow::GraphStreamInputTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Stream)
                       : std::nullopt;
          }},
      *ingress);
}

std::optional<BoundaryClass>
consumerBoundaryClass(const dataflow::CanonicalGraphConsumerEndpointRef &ref,
                      dataflow::GraphRef graph) {
  const auto *egress = std::get_if<dataflow::GraphEgressTokenRef>(&ref);
  if (!egress)
    return std::nullopt;
  return std::visit(
      Overloaded{
          [&](const dataflow::GraphValueOutputTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Value)
                       : std::nullopt;
          },
          [&](const dataflow::GraphStreamOutputTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Stream)
                       : std::nullopt;
          },
          [&](const dataflow::GraphCompletionFrontierTokenRef &value)
              -> std::optional<BoundaryClass> {
            return value.graph == graph
                       ? std::optional<BoundaryClass>(BoundaryClass::Control)
                       : std::nullopt;
          }},
      *egress);
}

const mapping::SpatialRouteNodeView *
routeNode(const mapping::SpatialRouteTreeView &route, std::uint64_t ordinal) {
  const auto found = llvm::find_if(
      route.nodes, [&](const auto &node) { return node.ordinal == ordinal; });
  return found == route.nodes.end() ? nullptr : &*found;
}

void writeBoundaryBindings(llvm::json::OStream &json, ReferenceEncoder &encoder,
                           const mapping::SpatialMappingView &mapping,
                           dataflow::GraphRef graph, BoundaryClass category) {
  json.array([&] {
    for (const auto indexedRoute : llvm::enumerate(mapping.routeTrees())) {
      const std::uint64_t routeOrdinal = indexedRoute.index();
      const auto &route = indexedRoute.value();
      if (producerBoundaryClass(route.logicalNet, graph) == category)
        json.object([&] {
          json.attribute("direction", "input");
          json.attribute("logical_endpoint_ref",
                         encoder.dataflow(route.logicalNet));
          json.attribute("physical_endpoint_ref",
                         formatArtifactLocalPayloadHex(
                             fabric::canonicalFabricBytes(route.rootEndpoint)));
          json.attribute("route_tree_ordinal", routeOrdinal);
        });
      for (const auto indexedSink : llvm::enumerate(route.sinks)) {
        const std::uint64_t sinkOrdinal = indexedSink.index();
        const auto &sink = indexedSink.value();
        if (consumerBoundaryClass(sink.sink, graph) != category)
          continue;
        const mapping::SpatialRouteNodeView *node =
            routeNode(route, sink.nodeOrdinal);
        if (!node)
          continue;
        json.object([&] {
          json.attribute("direction", "output");
          json.attribute("logical_endpoint_ref", encoder.dataflow(sink.sink));
          json.attribute("physical_endpoint_ref",
                         formatArtifactLocalPayloadHex(
                             fabric::canonicalFabricBytes(node->endpoint)));
          json.attribute("route_tree_ordinal", routeOrdinal);
          json.attribute("sink_ordinal", sinkOrdinal);
        });
      }
    }
  });
}

void writeMemoryBoundaryBindings(llvm::json::OStream &json,
                                 ReferenceEncoder &encoder,
                                 const mapping::SpatialMappingView &mapping,
                                 dataflow::RootedGraphLaunchRef launch) {
  json.array([&] {
    for (const mapping::SpatialMemoryBindingView &binding :
         mapping.memoryBindings()) {
      for (const mapping::SpatialExposureEntryView &exposure :
           binding.exposures) {
        if (exposure.exposure.launch != launch)
          continue;
        json.object([&] {
          json.attribute("logical_memory_ref",
                         encoder.dataflow(binding.logicalMemory));
          json.attribute("memory_exposure_ref",
                         encoder.dataflow(exposure.exposure));
          json.attribute("subordinate_endpoint_ref",
                         formatArtifactLocalPayloadHex(
                             fabric::canonicalFabricBytes(exposure.terminal)));
          json.attribute("binding_ordinal", binding.entityId);
        });
      }
    }
  });
}

struct ConfigurationImageCatalogEntry final {
  ArtifactRootReference reference;
  std::vector<fabric::AccCoreOccurrenceRef> accCores;
};

llvm::Expected<std::vector<ConfigurationImageCatalogEntry>>
buildConfigurationImageCatalog(
    llvm::ArrayRef<ArtifactRootReference> imageReferences,
    const ArtifactStore &artifacts) {
  std::vector<hardware::FinalizedConfigurationABI> importedAbis;
  std::vector<ConfigurationImageCatalogEntry> catalog;
  importedAbis.reserve(imageReferences.size());
  catalog.reserve(imageReferences.size());

  for (const ArtifactRootReference &reference : imageReferences) {
    auto image = importHardwareConfigurationImage(reference, artifacts);
    if (!image)
      return image.takeError();

    const ArtifactRootReference &abiReference =
        image->image().configurationAbi();
    auto abi = llvm::find_if(importedAbis, [&](const auto &candidate) {
      return candidate.reference() == abiReference;
    });
    if (abi == importedAbis.end()) {
      auto imported =
          hardware::importConfigurationABI(abiReference, artifacts);
      if (!imported)
        return imported.takeError();
      importedAbis.push_back(std::move(*imported));
      abi = std::prev(importedAbis.end());
    }

    const hardware::ProgrammingUnit *unit =
        abi->abi().findProgrammingUnit(image->image().programmingUnitId());
    if (!unit)
      return invalid("configuration image names a missing programming unit");

    const hardware::ProgrammingUnitOccurrenceScope scope =
        hardware::deriveProgrammingUnitOccurrenceScope(*unit);
    if (scope.includesDirectSystemResources || scope.spatialCores.size() != 1)
      return invalid("configuration image programming unit is not local to "
                     "one SpatialCore occurrence");
    ConfigurationImageCatalogEntry entry{
        reference, {scope.spatialCores.front().core}};
    catalog.push_back(std::move(entry));
  }
  return catalog;
}

std::vector<ArtifactRootReference> configurationImagesFor(
    fabric::AccCoreOccurrenceRef core,
    llvm::ArrayRef<ConfigurationImageCatalogEntry> catalog) {
  std::vector<ArtifactRootReference> result;
  for (const ConfigurationImageCatalogEntry &entry : catalog)
    if (llvm::is_contained(entry.accCores, core))
      result.push_back(entry.reference);
  return result;
}

std::vector<dataflow::CanonicalProducerTerminalRef>
collectGraphDestinations(ReferenceEncoder &encoder,
                         const dataflow::CanonicalDataflowProgramView &view,
                         dataflow::RootedGraphLaunchRef launch,
                         bool completion) {
  std::vector<dataflow::CanonicalProducerTerminalRef> destinations;
  llvm::Error error = view.forEachProducerTerminal(
      launch.rootThreadLaunch,
      [&](const dataflow::CanonicalProducerTerminalView &terminal)
          -> llvm::Error {
        const auto *boundary =
            std::get_if<dataflow::GraphLaunchBoundarySourceRef>(
                &terminal.terminal);
        if (!boundary)
          return llvm::Error::success();
        const bool matches = std::visit(
            [&](const auto &value) {
              using Value = std::decay_t<decltype(value)>;
              const bool sameLaunch = value.launch == launch;
              if constexpr (std::is_same_v<
                                Value, dataflow::GraphLaunchDoneTransferRef>)
                return sameLaunch && completion;
              if constexpr (std::is_same_v<
                                Value,
                                dataflow::GraphLaunchValueResultTransferRef>)
                return sameLaunch && !completion;
              return false;
            },
            boundary->transfer);
        if (matches)
          destinations.push_back(terminal.terminal);
        return llvm::Error::success();
      });
  if (error)
    encoder.recordError(std::move(error));
  return destinations;
}

llvm::Expected<CanonicalSemanticBytes>
deriveSpatialLaunchImage(const ArtifactRootReference &systemMapping,
                         const dataflow::CanonicalDataflowProgramView &dataflow,
                         const mapping::SystemMappingClosureProjection &closure,
                         llvm::ArrayRef<ArtifactRootReference> imageReferences,
                         const ArtifactStore &artifacts,
                         ReferenceEncoder &encoder) {
  if (closure.executionContexts.spatialDomains.empty())
    return invalid("cannot derive an empty SpatialLaunchImage");
  auto imageCatalog =
      buildConfigurationImageCatalog(imageReferences, artifacts);
  if (!imageCatalog)
    return imageCatalog.takeError();

  llvm::SmallString<32768> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("schema", spatialLaunchImageSchema.identity);
    json.attribute("schema_version",
                   formatSchemaVersion(spatialLaunchImageSchema.version));
    json.attributeObject("source_system_mapping_ref",
                         [&] { writeRootReference(json, systemMapping); });
    json.attributeObject("payload", [&] {
      json.attributeArray("rows", [&] {
        std::vector<dataflow::RootedGraphLaunchRef> launches;
        for (const auto &domain : closure.executionContexts.spatialDomains)
          if (!llvm::is_contained(launches, domain.graph))
            launches.push_back(domain.graph);
        llvm::sort(launches, [&](const auto &lhs, const auto &rhs) {
          return encoder.dataflow(lhs) < encoder.dataflow(rhs);
        });
        for (const dataflow::RootedGraphLaunchRef &launch : launches) {
          auto graph = dataflow.resolve(launch);
          if (!graph) {
            encoder.recordError(graph.takeError());
            continue;
          }
          const auto resultDestinations =
              collectGraphDestinations(encoder, dataflow, launch, false);
          const auto doneDestinations =
              collectGraphDestinations(encoder, dataflow, launch, true);
          if (doneDestinations.size() != 1)
            encoder.recordError(invalid(
                "a rooted graph launch must have one done destination"));
          json.object([&] {
            json.attributeBegin("graph_execution_binding_key");
            writeDataflowReference(json, encoder, "rooted_graph_launch",
                                   launch);
            json.attributeEnd();
            json.attributeObject("compiled_graph_execution_binding", [&] {
              json.attributeArray("target_domains", [&] {
                for (const auto &domain :
                     closure.executionContexts.spatialDomains) {
                  if (domain.graph != launch)
                    continue;
                  json.object([&] {
                    json.attributeBegin("execution_context_key");
                    writeContext(json, encoder, domain.context);
                    json.attributeEnd();
                    writeCells(json, "cells", domain.cells);
                  });
                }
              });
            });
            json.attributeArray("target_cases", [&] {
              for (const auto &domain :
                   closure.executionContexts.spatialDomains) {
                if (domain.graph != launch)
                  continue;
                auto spatial = mapping::importSpatialMapping(
                    domain.spatialMapping, artifacts);
                if (!spatial) {
                  encoder.recordError(spatial.takeError());
                  continue;
                }
                const auto images = configurationImagesFor(
                    domain.context.accCore, *imageCatalog);
                json.object([&] {
                  json.attributeBegin("execution_context_key");
                  writeContext(json, encoder, domain.context);
                  json.attributeEnd();
                  json.attributeArray("required_configuration_image_refs", [&] {
                    for (const ArtifactRootReference &reference : images)
                      json.object([&] { writeRootReference(json, reference); });
                  });
                  json.attributeBegin("value_boundary_bindings");
                  writeBoundaryBindings(json, encoder, spatial->view(), *graph,
                                        BoundaryClass::Value);
                  json.attributeEnd();
                  json.attributeBegin("stream_boundary_bindings");
                  writeBoundaryBindings(json, encoder, spatial->view(), *graph,
                                        BoundaryClass::Stream);
                  json.attributeEnd();
                  json.attributeBegin("control_boundary_bindings");
                  writeBoundaryBindings(json, encoder, spatial->view(), *graph,
                                        BoundaryClass::Control);
                  json.attributeEnd();
                  json.attributeBegin("memory_boundary_bindings");
                  writeMemoryBoundaryBindings(json, encoder, spatial->view(),
                                              launch);
                  json.attributeEnd();
                  json.attributeObject("graph_start_activation_set_ref", [&] {
                    json.attributeBegin("event_family_key");
                    writeEvent(json, encoder,
                               dataflow::graphLaunchStartEventFamily(launch));
                    json.attributeEnd();
                    json.attributeBegin("execution_context_key");
                    writeContext(json, encoder, domain.context);
                    json.attributeEnd();
                  });
                  json.attributeArray("result_destinations", [&] {
                    for (const auto &destination : resultDestinations)
                      writeDataflowReference(json, encoder,
                                             "graph_result_destination",
                                             destination);
                  });
                  json.attributeBegin("done_destination");
                  if (doneDestinations.size() == 1)
                    writeDataflowReference(json, encoder,
                                           "graph_done_destination",
                                           doneDestinations.front());
                  else
                    json.value(nullptr);
                  json.attributeEnd();
                });
              }
            });
          });
        }
      });
    });
  });
  if (encoder.failed())
    return encoder.takeError();
  return finishJson(storage);
}

} // namespace

llvm::Expected<DerivedRuntimeImages> deriveRuntimeImages(
    const ArtifactRootReference &systemMappingReference,
    llvm::ArrayRef<ArtifactRootReference> instructionCoreBinaries,
    llvm::ArrayRef<ArtifactRootReference> configurationImages,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto systemMapping =
      mapping::importSystemMapping(systemMappingReference, artifacts);
  if (!systemMapping)
    return systemMapping.takeError();
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      systemMapping->view().dataflowIdentity()};
  const ArtifactRootReference fabricReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      systemMapping->view().fabricIdentity()};
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  auto fabricArtifact =
      fabric::importEntireFabricRoot(fabricReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto system = fabric::requireSystemRoot(fabricArtifact->view());
  if (!system)
    return system.takeError();
  auto closure = mapping::projectSystemMappingClosure(
      *dataflowView, *system, systemMapping->view(), artifacts);
  if (!closure)
    return closure.takeError();

  ReferenceEncoder encoder(dataflowView->identity());
  auto groups = buildActivationGroups(*closure, encoder);
  if (!groups)
    return groups.takeError();
  auto binaries = importBinaries(instructionCoreBinaries, dataflowReference,
                                 artifacts, blobs);
  if (!binaries)
    return binaries.takeError();
  auto admission =
      deriveAdmissionImage(systemMappingReference, *closure, encoder, *groups);
  if (!admission)
    return admission.takeError();
  auto thread =
      deriveThreadDispatchImage(systemMappingReference, *dataflowView, *closure,
                                systemMapping->view().fabricIdentity(),
                                *binaries, encoder, artifacts, *groups);
  if (!thread)
    return thread.takeError();
  std::optional<CanonicalSemanticBytes> spatial;
  if (!closure->executionContexts.spatialDomains.empty()) {
    auto image = deriveSpatialLaunchImage(systemMappingReference, *dataflowView,
                                          *closure, configurationImages,
                                          artifacts, encoder);
    if (!image)
      return image.takeError();
    spatial = std::move(*image);
  }
  return DerivedRuntimeImages{std::move(*thread), std::move(spatial),
                              std::move(*admission)};
}

} // namespace loom::deployment::detail
