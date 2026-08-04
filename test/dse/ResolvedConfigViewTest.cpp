#include "DSE/ResolvedConfigView.h"

#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "resolved DSE config view test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireRejected(llvm::Error error, llvm::StringRef needle) {
  if (!error)
    fail("expected rejection containing '" + needle.str() + "'");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(needle))
    fail("expected rejection containing '" + needle.str() +
         "', got: " + message);
}

template <typename T>
void requireRejected(llvm::Expected<T> value, llvm::StringRef needle) {
  if (value)
    fail("expected rejection containing '" + needle.str() + "'");
  requireRejected(value.takeError(), needle);
}

constexpr ArtifactSchemaDescriptor candidateSchema{
    "loom.test.resolved_dse_candidate", SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> acquisitionConfigSchema = {0x44, 0x53,
                                                                 0x45, 0x31};
constexpr std::array<PromotionAcquisitionInputSlotDescriptor, 1> inputs = {{{
    PromotionAcquisitionInputSlotRef(0),
    "candidate",
    PlanValueRole::CandidateSet,
    &candidateSchema,
    PlanValueCardinality::ExactlyOne,
}}};

llvm::Error validateAcquisitionConfig(llvm::ArrayRef<std::uint8_t> bytes,
                                      const ComponentViewDigest &digest) {
  if (!bytes.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test acquisition config is not empty");
  return validateComponentViewDigest(acquisitionConfigSchema, bytes, digest);
}

const PromotionAcquisitionDescriptor acquisition{
    PromotionAcquisitionKind(0x7fff3000),
    "test.resolved_dse",
    "loom.test.resolved_dse.v1",
    inputs,
    PromotionAcquisitionInputSlotRef(0),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{acquisitionConfigSchema,
                                  validateAcquisitionConfig},
};

ArtifactRootReference candidate() {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> identity{};
  identity.fill(0x5a);
  return ArtifactRootReference{candidateSchema.identity.str(),
                               candidateSchema.version,
                               take(ArtifactIdentity::fromBytes(identity))};
}

PromotePlanNodeDefinition promoteNode(QualityGatePolicyRef gate) {
  const ComponentViewDigest digest = take(computeComponentViewDigest(
      acquisitionConfigSchema, llvm::ArrayRef<std::uint8_t>()));
  return PromotePlanNodeDefinition{
      acquisition.reference(),
      {ExactPlanArtifacts{{candidate()}}},
      {},
      digest,
      gate,
      AllPassingSelection{},
      PromotePurpose::CandidateSelection,
  };
}

void canonicalRoundTripOwnsThePlanPolicy() {
  if (llvm::Error error = registerPromotionAcquisitionDescriptor(acquisition))
    fail(llvm::toString(std::move(error)));

  std::vector<QualityGatePolicy> gates;
  gates.push_back(take(QualityGatePolicy::get({})));
  ResolvedDseConfigView view = take(ResolvedDseConfigView::get(
      {}, {}, resolvedBuiltinObjectiveCatalogs(), std::move(gates),
      {promoteNode(QualityGatePolicyRef(0))}));
  ResolvedDseConfigView adopted = take(adoptResolvedDseConfigView(
      view.schemaDescriptorBytes(), view.canonicalViewBytes(), view.digest()));

  if (adopted.canonicalViewBytes() != view.canonicalViewBytes() ||
      adopted.digest() != view.digest() || adopted.plan().nodes().size() != 1)
    fail("round-trip changed the sealed DSE policy view");
  const auto &promote =
      std::get<ResolvedPromotePlanNode>(adopted.plan().nodes().front());
  if (promote.qualityGateRef() != QualityGatePolicyRef(0) ||
      promote.purpose() != PromotePurpose::CandidateSelection ||
      !adopted.plan().resolve(promote.qualityGateRef()))
    fail("Promote did not resolve its view-owned quality gate");
}

void malformedAndForeignReferencesFailClosed() {
  std::vector<QualityGatePolicy> gates;
  gates.push_back(take(QualityGatePolicy::get({})));
  requireRejected(ResolvedDseConfigView::get(
                      {}, {}, resolvedBuiltinObjectiveCatalogs(),
                      std::move(gates), {promoteNode(QualityGatePolicyRef(1))}),
                  "quality gate reference is out of range");

  std::vector<QualityGatePolicy> duplicateGates;
  duplicateGates.push_back(take(QualityGatePolicy::get({})));
  duplicateGates.push_back(take(QualityGatePolicy::get({})));
  requireRejected(ResolvedDseConfigView::get({}, {},
                                             resolvedBuiltinObjectiveCatalogs(),
                                             std::move(duplicateGates), {}),
                  "quality gate policies are not canonical and unique");

  ResolvedDseConfigView view = take(
      ResolvedDseConfigView::get({}, {}, resolvedBuiltinObjectiveCatalogs(),
                                 {take(QualityGatePolicy::get({}))},
                                 {promoteNode(QualityGatePolicyRef(0))}));
  std::vector<std::uint8_t> trailing(view.canonicalViewBytes().begin(),
                                     view.canonicalViewBytes().end());
  trailing.push_back(0);
  const ComponentViewDigest trailingDigest =
      take(computeComponentViewDigest(view.schemaDescriptorBytes(), trailing));
  requireRejected(adoptResolvedDseConfigView(view.schemaDescriptorBytes(),
                                             trailing, trailingDigest),
                  "trailing bytes");

  std::array<std::uint8_t, ComponentViewDigest::byteSize> stale =
      view.digest().bytes();
  stale.back() ^= 1;
  requireRejected(adoptResolvedDseConfigView(
                      view.schemaDescriptorBytes(), view.canonicalViewBytes(),
                      take(ComponentViewDigest::fromBytes(stale))),
                  "component_view_digest_mismatch");
}

} // namespace

int main() {
  canonicalRoundTripOwnsThePlanPolicy();
  malformedAndForeignReferencesFailClosed();
  return 0;
}
