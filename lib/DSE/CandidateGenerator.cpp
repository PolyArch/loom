#include "DSE/CandidateGenerator.h"

#include "Common/ArtifactStore.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace loom::dse {
namespace {

std::vector<const CandidateGeneratorDescriptor *> &descriptors() {
  static std::vector<const CandidateGeneratorDescriptor *> records;
  return records;
}

std::shared_mutex &descriptorMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

std::vector<CandidateGeneratorProvider> &providers() {
  static std::vector<CandidateGeneratorProvider> records;
  return records;
}

std::shared_mutex &providerMutex() {
  static std::shared_mutex mutex;
  return mutex;
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "candidate_generator_invalid: " + message);
}

bool isCanonicalAscii(llvm::StringRef value) {
  return !value.empty() && llvm::all_of(value, [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool acceptsSchema(const CandidateGeneratorInputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return slot.schema && slot.schema->identity == artifact.schemaIdentity &&
         slot.schema->version == artifact.schemaVersion;
}

bool matchesSchema(const CandidateGeneratorOutputSlotDescriptor &slot,
                   const ArtifactRootReference &artifact) {
  return slot.schema && slot.schema->identity == artifact.schemaIdentity &&
         slot.schema->version == artifact.schemaVersion;
}

bool containsReference(llvm::ArrayRef<ArtifactRootReference> references,
                       const ArtifactRootReference &reference) {
  return std::binary_search(references.begin(), references.end(), reference,
                            artifactRootReferenceLess);
}

bool lineageEdgeLess(const CandidateGeneratorLineageEdge &lhs,
                     const CandidateGeneratorLineageEdge &rhs) {
  if (lhs.kind != rhs.kind)
    return lhs.kind < rhs.kind;
  if (lhs.outputSlot != rhs.outputSlot)
    return lhs.outputSlot.ordinal() < rhs.outputSlot.ordinal();
  if (lhs.output != rhs.output)
    return artifactRootReferenceLess(lhs.output, rhs.output);
  if (lhs.parents != rhs.parents)
    return std::lexicographical_compare(lhs.parents.begin(), lhs.parents.end(),
                                        rhs.parents.begin(), rhs.parents.end(),
                                        artifactRootReferenceLess);
  return lhs.ownerPayload < rhs.ownerPayload;
}

struct LineageTargetKey final {
  CandidateGeneratorOutputSlotRef slot;
  ArtifactRootReference output;
};

bool lineageTargetLess(const LineageTargetKey &lhs,
                       const LineageTargetKey &rhs) {
  if (lhs.slot != rhs.slot)
    return lhs.slot.ordinal() < rhs.slot.ordinal();
  return artifactRootReferenceLess(lhs.output, rhs.output);
}

llvm::Error canonicalizeLineageEdges(
    const CandidateGeneratorDescriptor &descriptor,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs,
    std::vector<CandidateGeneratorLineageEdge> &edges,
    const ArtifactStore &store) {
  std::vector<ArtifactRootReference> invocationInputs;
  for (const CandidateGeneratorInputBinding &binding : inputs)
    invocationInputs.insert(invocationInputs.end(), binding.artifacts.begin(),
                            binding.artifacts.end());
  llvm::sort(invocationInputs, artifactRootReferenceLess);
  invocationInputs.erase(
      std::unique(invocationInputs.begin(), invocationInputs.end()),
      invocationInputs.end());

  for (CandidateGeneratorLineageEdge &edge : edges) {
    if (static_cast<std::uint32_t>(edge.kind) >
        static_cast<std::uint32_t>(
            CandidateGeneratorLineageEdgeKind::CandidateDecision))
      return invalid("provider returned an invalid lineage edge kind");
    if (edge.outputSlot.ordinal() >= outputs.size())
      return invalid("lineage edge references an unknown output slot");
    if (!matchesSchema(descriptor.outputSlots[edge.outputSlot.ordinal()],
                       edge.output))
      return invalid("lineage edge target does not match its output slot");
    auto storedOutput = store.get(edge.output);
    if (!storedOutput)
      return storedOutput.takeError();

    llvm::sort(edge.parents, artifactRootReferenceLess);
    edge.parents.erase(std::unique(edge.parents.begin(), edge.parents.end()),
                       edge.parents.end());
    for (const ArtifactRootReference &parent : edge.parents) {
      if (parent == edge.output)
        return invalid("lineage edge cannot reference its output as a parent");
    }

    if (edge.kind == CandidateGeneratorLineageEdgeKind::MechanicalDerivation) {
      if (!edge.parents.empty() || !edge.ownerPayload.empty())
        return invalid("mechanical lineage edge has decision fields");
      continue;
    }
    if (edge.parents.empty())
      return invalid("candidate decision lineage requires a parent");
    if (!descriptor.ownerLineagePayload)
      return invalid("descriptor does not own a lineage payload contract");
    if (llvm::Error error = descriptor.ownerLineagePayload->validateCanonical(
            edge.ownerPayload, edge.parents, store))
      return error;
    for (const ArtifactRootReference &parent : edge.parents) {
      auto stored = store.get(parent);
      if (!stored)
        return stored.takeError();
    }
  }

  llvm::sort(edges, lineageEdgeLess);
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::vector<LineageTargetKey> lineageTargets;
  lineageTargets.reserve(edges.size());
  for (const CandidateGeneratorLineageEdge &edge : edges)
    lineageTargets.push_back({edge.outputSlot, edge.output});
  llvm::sort(lineageTargets, lineageTargetLess);
  lineageTargets.erase(
      std::unique(lineageTargets.begin(), lineageTargets.end(),
                  [](const LineageTargetKey &lhs, const LineageTargetKey &rhs) {
                    return lhs.slot == rhs.slot && lhs.output == rhs.output;
                  }),
      lineageTargets.end());

  std::vector<ArtifactRootReference> produced;
  produced.reserve(edges.size());
  for (const CandidateGeneratorLineageEdge &edge : edges)
    produced.push_back(edge.output);
  llvm::sort(produced, artifactRootReferenceLess);
  produced.erase(std::unique(produced.begin(), produced.end()), produced.end());

  std::vector<ArtifactRootReference> consumed;
  for (const CandidateGeneratorLineageEdge &edge : edges)
    consumed.insert(consumed.end(), edge.parents.begin(), edge.parents.end());
  llvm::sort(consumed, artifactRootReferenceLess);
  consumed.erase(std::unique(consumed.begin(), consumed.end()), consumed.end());

  for (const CandidateGeneratorLineageEdge &edge : edges) {
    const bool isReturned = containsReference(
        outputs[edge.outputSlot.ordinal()].artifacts, edge.output);
    const bool isConsumed = containsReference(consumed, edge.output);
    if (!isReturned && !isConsumed)
      return invalid("internal lineage target does not reach an output");
    for (const ArtifactRootReference &parent : edge.parents)
      if (!containsReference(invocationInputs, parent) &&
          !containsReference(produced, parent))
        return invalid("lineage parent is not an invocation input or produced "
                       "target");
  }

  std::vector<std::vector<std::size_t>> successors(produced.size());
  std::vector<std::vector<std::size_t>> producers(produced.size());
  std::vector<std::size_t> indegrees(produced.size());
  const auto producedOrdinal = [&](const ArtifactRootReference &reference) {
    auto found =
        llvm::lower_bound(produced, reference, artifactRootReferenceLess);
    return static_cast<std::size_t>(found - produced.begin());
  };
  for (std::size_t edgeOrdinal = 0; edgeOrdinal < edges.size(); ++edgeOrdinal) {
    const CandidateGeneratorLineageEdge &edge = edges[edgeOrdinal];
    const std::size_t child = producedOrdinal(edge.output);
    producers[child].push_back(edgeOrdinal);
    for (const ArtifactRootReference &parent : edge.parents) {
      if (!containsReference(produced, parent))
        continue;
      const std::size_t parentOrdinal = producedOrdinal(parent);
      successors[parentOrdinal].push_back(child);
    }
  }
  for (std::vector<std::size_t> &children : successors) {
    llvm::sort(children);
    children.erase(std::unique(children.begin(), children.end()),
                   children.end());
    for (const std::size_t child : children)
      ++indegrees[child];
  }

  std::vector<std::size_t> ready;
  ready.reserve(produced.size());
  for (std::size_t ordinal = 0; ordinal < indegrees.size(); ++ordinal)
    if (indegrees[ordinal] == 0)
      ready.push_back(ordinal);
  std::vector<std::size_t> topologicalOrder;
  topologicalOrder.reserve(produced.size());
  for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
    const std::size_t parent = ready[cursor];
    topologicalOrder.push_back(parent);
    for (const std::size_t child : successors[parent])
      if (--indegrees[child] == 0)
        ready.push_back(child);
  }
  if (topologicalOrder.size() != produced.size())
    return invalid("candidate lineage contains a cycle");

  std::vector<bool> rooted(produced.size());
  for (const std::size_t ordinal : topologicalOrder) {
    for (const std::size_t edgeOrdinal : producers[ordinal]) {
      const CandidateGeneratorLineageEdge &edge = edges[edgeOrdinal];
      bool edgeIsRooted =
          edge.kind == CandidateGeneratorLineageEdgeKind::MechanicalDerivation;
      if (edge.kind == CandidateGeneratorLineageEdgeKind::CandidateDecision)
        edgeIsRooted = llvm::all_of(
            edge.parents, [&](const ArtifactRootReference &parent) {
              if (containsReference(invocationInputs, parent))
                return true;
              return static_cast<bool>(rooted[producedOrdinal(parent)]);
            });
      rooted[ordinal] = rooted[ordinal] || edgeIsRooted;
    }
  }
  for (const CandidateGeneratorLineageEdge &edge : edges) {
    if (!llvm::all_of(edge.parents, [&](const ArtifactRootReference &parent) {
          return containsReference(invocationInputs, parent) ||
                 static_cast<bool>(rooted[producedOrdinal(parent)]);
        }))
      return invalid("candidate lineage is not rooted in invocation inputs");
  }

  for (const CandidateGeneratorOutputBinding &binding : outputs) {
    for (const ArtifactRootReference &output : binding.artifacts) {
      if (containsReference(invocationInputs, output))
        continue;
      const bool hasLineage = std::binary_search(
          lineageTargets.begin(), lineageTargets.end(),
          LineageTargetKey{binding.slot, output}, lineageTargetLess);
      if (!hasLineage)
        return invalid("generated output has no lineage edge");
    }
  }
  return llvm::Error::success();
}

llvm::Error canonicalizeOutputBindings(
    const CandidateGeneratorDescriptor &descriptor,
    std::vector<CandidateGeneratorOutputBinding> &bindings,
    bool requireFinalCardinality, const ArtifactStore &store) {
  if (bindings.size() != descriptor.outputSlots.size())
    return invalid("provider does not bind every output slot");
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    CandidateGeneratorOutputBinding &binding = bindings[index];
    const CandidateGeneratorOutputSlotDescriptor &slot =
        descriptor.outputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("provider output bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!matchesSchema(slot, artifact))
        return invalid("provider output artifact schema does not match slot '" +
                       slot.semanticRole + "'");
    llvm::sort(binding.artifacts, artifactRootReferenceLess);
    binding.artifacts.erase(
        std::unique(binding.artifacts.begin(), binding.artifacts.end()),
        binding.artifacts.end());
    for (const ArtifactRootReference &artifact : binding.artifacts) {
      auto stored = store.get(artifact);
      if (!stored)
        return stored.takeError();
    }
    const PlanCardinalityBounds bounds =
        planCardinalityBounds(slot.cardinality);
    if (binding.artifacts.size() > bounds.maximum ||
        (requireFinalCardinality && binding.artifacts.size() < bounds.minimum))
      return invalid("provider output violates slot cardinality");
  }
  return llvm::Error::success();
}

llvm::Error validateDescriptor(const CandidateGeneratorDescriptor &descriptor) {
  if (!isCanonicalAscii(descriptor.spelling))
    return invalid("descriptor spelling must be nonempty canonical ASCII");
  if (!isCanonicalAscii(descriptor.implementationSemanticIdentity))
    return invalid("implementation semantic identity must be nonempty "
                   "canonical ASCII");
  if (descriptor.inputSlots.empty())
    return invalid("descriptor requires at least one typed input slot");
  if (descriptor.outputSlots.empty())
    return invalid("descriptor requires at least one typed output slot");
  if (descriptor.resolvedConfigView.schemaDescriptorBytes.empty() ||
      !descriptor.resolvedConfigView.validateCanonical)
    return invalid("descriptor requires an exact resolved config contract");
  if (descriptor.ownerLineagePayload &&
      (descriptor.ownerLineagePayload->schemaDescriptorBytes.empty() ||
       !descriptor.ownerLineagePayload->validateCanonical))
    return invalid("descriptor has an incomplete owner lineage contract");
  if (static_cast<std::uint32_t>(descriptor.determinism) >
      static_cast<std::uint32_t>(
          CandidateGeneratorDeterminism::IndependentReplicates))
    return invalid("descriptor has an invalid determinism contract");

  for (std::size_t index = 0; index < descriptor.inputSlots.size(); ++index) {
    const CandidateGeneratorInputSlotDescriptor &slot =
        descriptor.inputSlots[index];
    if (slot.slot.ordinal() != index)
      return invalid("input slots must be dense and canonical");
    if (!isCanonicalAscii(slot.semanticRole))
      return invalid("input slot role must be nonempty canonical ASCII");
    if (!slot.schema)
      return invalid("input slot requires one exact schema");
    if (static_cast<std::uint32_t>(slot.role) >
            static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet) ||
        static_cast<std::uint32_t>(slot.cardinality) >
            static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet))
      return invalid("input slot has an invalid plan value contract");
  }

  for (std::size_t index = 0; index < descriptor.outputSlots.size(); ++index) {
    const CandidateGeneratorOutputSlotDescriptor &slot =
        descriptor.outputSlots[index];
    if (slot.slot.ordinal() != index)
      return invalid("output slots must be dense and canonical");
    if (!isCanonicalAscii(slot.semanticRole) || !slot.schema)
      return invalid("output slot requires a role and exact schema");
    if (static_cast<std::uint32_t>(slot.role) >
            static_cast<std::uint32_t>(PlanValueRole::SimulationExecutionSet) ||
        static_cast<std::uint32_t>(slot.cardinality) >
            static_cast<std::uint32_t>(PlanValueCardinality::FiniteSet))
      return invalid("output slot has an invalid plan value contract");
  }

  for (std::size_t index = 0; index < descriptor.workUnits.size(); ++index) {
    const CandidateGeneratorWorkUnitDescriptor &unit =
        descriptor.workUnits[index];
    if (unit.unit.ordinal() != index || !isCanonicalAscii(unit.spelling))
      return invalid("work units must be dense canonical records");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CandidateGeneratorDescriptorRef>
CandidateGeneratorDescriptorRef::get(
    const ArtifactSchemaDescriptor &descriptorSchema,
    CandidateGeneratorKind kind) {
  if (descriptorSchema != candidateGeneratorDescriptorSchema)
    return invalid("candidate generator descriptor schema is unsupported");
  return CandidateGeneratorDescriptorRef(descriptorSchema, kind);
}

const CandidateGeneratorDescriptor *
CandidateGeneratorDescriptorRef::descriptor() const {
  if (descriptorSchema_ != candidateGeneratorDescriptorSchema)
    return nullptr;
  return findCandidateGeneratorDescriptor(kind_);
}

CandidateGeneratorDescriptorRef
CandidateGeneratorDescriptor::reference() const {
  return llvm::cantFail(CandidateGeneratorDescriptorRef::get(
      candidateGeneratorDescriptorSchema, kind));
}

const CandidateGeneratorInputSlotDescriptor *
CandidateGeneratorDescriptor::findInputSlot(
    CandidateGeneratorInputSlotRef slot) const {
  if (slot.ordinal() >= inputSlots.size())
    return nullptr;
  return &inputSlots[slot.ordinal()];
}

const CandidateGeneratorOutputSlotDescriptor *
CandidateGeneratorDescriptor::findOutputSlot(
    CandidateGeneratorOutputSlotRef slot) const {
  if (slot.ordinal() >= outputSlots.size())
    return nullptr;
  return &outputSlots[slot.ordinal()];
}

llvm::Error registerCandidateGeneratorDescriptor(
    const CandidateGeneratorDescriptor &descriptor) {
  if (llvm::Error error = validateDescriptor(descriptor))
    return error;

  std::unique_lock<std::shared_mutex> lock(descriptorMutex());
  for (const CandidateGeneratorDescriptor *existing : descriptors()) {
    if (existing == &descriptor)
      return llvm::Error::success();
    if (existing->kind == descriptor.kind)
      return invalid("conflicting registration for candidate generator kind " +
                     std::to_string(descriptor.kind.ordinal()));
    if (existing->spelling == descriptor.spelling)
      return invalid("conflicting registration for candidate generator '" +
                     descriptor.spelling + "'");
  }
  descriptors().push_back(&descriptor);
  std::sort(descriptors().begin(), descriptors().end(),
            [](const CandidateGeneratorDescriptor *lhs,
               const CandidateGeneratorDescriptor *rhs) {
              return lhs->kind < rhs->kind;
            });
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor *
findCandidateGeneratorDescriptor(CandidateGeneratorKind kind) {
  std::shared_lock<std::shared_mutex> lock(descriptorMutex());
  auto found =
      std::lower_bound(descriptors().begin(), descriptors().end(), kind,
                       [](const CandidateGeneratorDescriptor *descriptor,
                          CandidateGeneratorKind requested) {
                         return descriptor->kind < requested;
                       });
  if (found == descriptors().end() || (*found)->kind != kind)
    return nullptr;
  return *found;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
ResolvedCandidateGeneratorBinding::get(
    CandidateGeneratorDescriptorRef descriptorRef,
    llvm::ArrayRef<std::uint8_t> canonicalConfigBytes,
    const ComponentViewDigest &configDigest) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");

  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          canonicalConfigBytes, configDigest))
    return std::move(error);

  return ResolvedCandidateGeneratorBinding(
      descriptorRef, canonicalConfigBytes.vec(), configDigest);
}

llvm::Error validateCandidateGeneratorInputBindings(
    CandidateGeneratorDescriptorRef descriptorRef,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("input bindings reference an unregistered descriptor");
  if (inputBindings.size() != descriptor->inputSlots.size())
    return invalid("binding does not provide every descriptor input slot");

  for (std::size_t index = 0; index < inputBindings.size(); ++index) {
    const CandidateGeneratorInputBinding &binding = inputBindings[index];
    const CandidateGeneratorInputSlotDescriptor &slot =
        descriptor->inputSlots[index];
    if (binding.slot.ordinal() != index)
      return invalid("input bindings must be dense and canonical");
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (!acceptsSchema(slot, artifact))
        return invalid("input slot '" + slot.semanticRole +
                       "' does not accept artifact schema '" +
                       artifact.schemaIdentity + "'");
    if (!llvm::is_sorted(binding.artifacts, artifactRootReferenceLess) ||
        std::adjacent_find(binding.artifacts.begin(),
                           binding.artifacts.end()) != binding.artifacts.end())
      return invalid("input artifact sets must be canonical");
    if (!planCardinalityContains(slot.cardinality, binding.artifacts.size()))
      return invalid("canonical input set violates descriptor cardinality");
  }
  return llvm::Error::success();
}

llvm::Error validateCandidateGeneratorWorkSummary(
    CandidateGeneratorDescriptorRef descriptorRef,
    llvm::ArrayRef<CandidateGeneratorWorkUnitSummary> summary) {
  const CandidateGeneratorDescriptor *descriptor = descriptorRef.descriptor();
  if (!descriptor)
    return invalid("work summary references an unregistered descriptor");
  if (summary.size() != descriptor->workUnits.size())
    return invalid("work summary does not cover every descriptor work unit");
  for (std::size_t ordinal = 0; ordinal != summary.size(); ++ordinal) {
    const CandidateGeneratorWorkUnitSummary &entry = summary[ordinal];
    if (entry.unit.ordinal() != ordinal)
      return invalid("work summary entries must be dense and canonical");
    if (entry.consumed > entry.planned)
      return invalid("consumed work exceeds planned work");
  }
  return llvm::Error::success();
}

llvm::Error
registerCandidateGeneratorProvider(const CandidateGeneratorProvider &provider) {
  if (!provider.invoke || !provider.descriptor.descriptor())
    return invalid("provider requires a registered descriptor and callback");
  std::unique_lock<std::shared_mutex> lock(providerMutex());
  for (const CandidateGeneratorProvider &existing : providers()) {
    if (existing.descriptor != provider.descriptor)
      continue;
    if (existing.invoke == provider.invoke)
      return llvm::Error::success();
    return invalid("conflicting provider registration for candidate generator");
  }
  providers().push_back(provider);
  llvm::sort(providers(), [](const CandidateGeneratorProvider &lhs,
                             const CandidateGeneratorProvider &rhs) {
    return lhs.descriptor.kind() < rhs.descriptor.kind();
  });
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorInvocationOutcome> invokeCandidateGenerator(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store) {
  const CandidateGeneratorDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return invalid("binding references an unregistered descriptor");
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          binding.descriptorRef(), inputBindings))
    return std::move(error);

  CandidateGeneratorProviderFunction invoke = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(providerMutex());
    auto found =
        llvm::lower_bound(providers(), binding.descriptorRef().kind(),
                          [](const CandidateGeneratorProvider &provider,
                             CandidateGeneratorKind kind) {
                            return provider.descriptor.kind() < kind;
                          });
    if (found != providers().end() &&
        found->descriptor == binding.descriptorRef())
      invoke = found->invoke;
  }
  if (!invoke) {
    std::vector<CandidateGeneratorOutputBinding> outputs;
    outputs.reserve(descriptor->outputSlots.size());
    for (const CandidateGeneratorOutputSlotDescriptor &slot :
         descriptor->outputSlots)
      outputs.push_back({slot.slot, {}});
    std::vector<CandidateGeneratorWorkUnitSummary> workSummary;
    workSummary.reserve(descriptor->workUnits.size());
    for (const CandidateGeneratorWorkUnitDescriptor &unit :
         descriptor->workUnits)
      workSummary.push_back({unit.unit, 0, 0});
    return CandidateGeneratorInvocationOutcome{
        IncompleteCandidateGeneratorInvocation{
            CandidateGeneratorIncompleteReason::ProviderUnavailable,
            std::move(outputs),
            {},
            std::move(workSummary)}};
  }

  auto outcome = invoke(inputBindings, binding, store);
  if (!outcome)
    return outcome.takeError();
  if (auto *completed =
          std::get_if<CompletedCandidateGeneratorInvocation>(&*outcome)) {
    if (llvm::Error error = validateCandidateGeneratorWorkSummary(
            binding.descriptorRef(), completed->workSummary))
      return std::move(error);
    if (llvm::Error error = canonicalizeOutputBindings(
            *descriptor, completed->outputBindings, true, store))
      return std::move(error);
    if (llvm::Error error = canonicalizeLineageEdges(
            *descriptor, inputBindings, completed->outputBindings,
            completed->lineageEdges, store))
      return std::move(error);
  } else {
    auto &incomplete =
        std::get<IncompleteCandidateGeneratorInvocation>(*outcome);
    if (static_cast<std::uint32_t>(incomplete.reason) >
        static_cast<std::uint32_t>(
            CandidateGeneratorIncompleteReason::Unsupported))
      return invalid("provider returned an invalid Incomplete reason");
    if (llvm::Error error = validateCandidateGeneratorWorkSummary(
            binding.descriptorRef(), incomplete.workSummary))
      return std::move(error);
    if (llvm::Error error = canonicalizeOutputBindings(
            *descriptor, incomplete.retainedOutputBindings, false, store))
      return std::move(error);
    if (llvm::Error error = canonicalizeLineageEdges(
            *descriptor, inputBindings, incomplete.retainedOutputBindings,
            incomplete.lineageEdges, store))
      return std::move(error);
  }
  return outcome;
}

llvm::Error validateCanonicalCandidateGeneratorInvocation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs,
    llvm::ArrayRef<CandidateGeneratorLineageEdge> lineageEdges, bool completed,
    const ArtifactStore &store) {
  const CandidateGeneratorDescriptor *descriptor =
      binding.descriptorRef().descriptor();
  if (!descriptor)
    return invalid("invocation record references an unregistered descriptor");
  if (llvm::Error error = descriptor->resolvedConfigView.validateCanonical(
          binding.canonicalConfigBytes(), binding.configDigest()))
    return error;
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          binding.descriptorRef(), inputs))
    return error;
  for (const CandidateGeneratorInputBinding &input : inputs)
    for (const ArtifactRootReference &artifact : input.artifacts) {
      auto stored = store.get(artifact);
      if (!stored)
        return stored.takeError();
    }

  std::vector<CandidateGeneratorOutputBinding> canonicalOutputs(outputs.begin(),
                                                                outputs.end());
  if (llvm::Error error = canonicalizeOutputBindings(
          *descriptor, canonicalOutputs, completed, store))
    return error;
  if (canonicalOutputs.size() != outputs.size())
    return invalid("invocation output binding cardinality changed");
  for (auto [canonical, supplied] : llvm::zip_equal(canonicalOutputs, outputs))
    if (canonical.slot != supplied.slot ||
        canonical.artifacts != supplied.artifacts)
      return invalid("invocation output bindings are not canonical");

  std::vector<CandidateGeneratorLineageEdge> canonicalEdges(
      lineageEdges.begin(), lineageEdges.end());
  if (llvm::Error error = canonicalizeLineageEdges(
          *descriptor, inputs, canonicalOutputs, canonicalEdges, store))
    return error;
  if (llvm::ArrayRef(canonicalEdges) != lineageEdges)
    return invalid("invocation lineage edges are not canonical");
  return llvm::Error::success();
}

} // namespace loom::dse
