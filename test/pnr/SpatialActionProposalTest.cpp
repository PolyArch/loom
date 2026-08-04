#include "PnR/SpatialAction.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial action proposal test: " << message << '\n';
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::pnr::DeterministicPnrRandomStream actionStream() {
  return loom::pnr::DeterministicPnrRandomStream::create(
      UINT64_C(0x0123456789abcdef), 7,
      loom::pnr::PnrRandomStreamPurpose::ActionProposal);
}

void liveKindsAreReducedAndConsumeThreeDraws() {
  using namespace loom::pnr;
  const std::array<SpatialActionChoiceRange, 2> realizationAnchors{
      {{0, 1}, {1, 1}}};
  const std::array<SpatialRealizationBindingAction, 2> realizationChoices{
      SpatialComputeBindingAction{0, 1, 2}, SpatialMemoryBindingAction{0, 3}};
  const std::array<SpatialActionChoiceRange, 1> resourceAnchors{{{0, 2}}};
  const std::array<SpatialResourceAllocationAction, 2> resourceChoices{
      SpatialPortAttachmentAction{3, 10}, SpatialPortAttachmentAction{3, 11}};

  const SpatialActionProposalDomain domain{
      realizationAnchors, realizationChoices, {}, {},
      resourceAnchors,    resourceChoices};
  DeterministicPnrRandomStream stream = actionStream();
  const auto proposal = take(proposeSpatialAction(
      loom::ResolvedPnrActionProposalPolicy{2, 3, 4}, domain, stream));
  require(proposal.has_value(), "live domain produced no Action");
  const auto *resource =
      std::get_if<SpatialResourceAllocationAction>(&*proposal);
  require(resource, "live-weight selection chose the wrong Action kind");
  const auto *port = std::get_if<SpatialPortAttachmentAction>(resource);
  require(port && port->demand == 3 && port->attachmentOption == 10,
          "canonical anchor or choice selection changed");
  require(stream.nextU64() == UINT64_C(0x5c448abe844d0951),
          "Action proposal did not consume exactly three bounded draws");
}

void emptyAndMalformedDomainsDoNotConsumeEntropy() {
  using namespace loom::pnr;
  DeterministicPnrRandomStream emptyStream = actionStream();
  const auto empty = take(proposeSpatialAction(
      loom::ResolvedPnrActionProposalPolicy{1, 1, 1}, {}, emptyStream));
  require(!empty, "empty Action domain produced a proposal");
  DeterministicPnrRandomStream reference = actionStream();
  require(emptyStream.nextU64() == reference.nextU64(),
          "empty Action domain consumed entropy");

  const std::array<SpatialActionChoiceRange, 1> anchors{{{0, 2}}};
  const std::array<SpatialResourceAllocationAction, 2> reversed{
      SpatialPortAttachmentAction{3, 11}, SpatialPortAttachmentAction{3, 10}};
  DeterministicPnrRandomStream malformedStream = actionStream();
  llvm::Expected<std::optional<SpatialMappingAction>> malformed =
      proposeSpatialAction(loom::ResolvedPnrActionProposalPolicy{1, 1, 1},
                           {{}, {}, {}, {}, anchors, reversed},
                           malformedStream);
  require(!malformed &&
              llvm::toString(malformed.takeError()).find("canonical") !=
                  std::string::npos,
          "noncanonical Action choices were accepted");
  reference = actionStream();
  require(malformedStream.nextU64() == reference.nextU64(),
          "malformed Action domain consumed entropy before validation");
}

void policySumMustFitTheBoundedProtocol() {
  llvm::Error error = loom::validateResolvedPnrActionProposalPolicy(
      {UINT64_MAX, UINT64_MAX - 1, UINT64_MAX - 2});
  require(error &&
              llvm::toString(std::move(error)).find("sum") != std::string::npos,
          "unrepresentable Action weight sum was accepted");
}

void actionBatchesRequireCanonicalUniqueAnchors() {
  using namespace loom::pnr;
  const std::array<SpatialMappingAction, 3> canonical{
      SpatialRealizationBindingAction{SpatialComputeBindingAction{0, 1, 2}},
      SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{4}},
      SpatialResourceAllocationAction{SpatialPortAttachmentAction{3, 10}},
  };
  if (llvm::Error error = validateCanonicalSpatialActionBatch(canonical))
    fail(llvm::toString(std::move(error)));

  const std::array<SpatialMappingAction, 2> reversed{canonical[2],
                                                     canonical[0]};
  llvm::Error reversedError = validateCanonicalSpatialActionBatch(reversed);
  require(reversedError &&
              llvm::toString(std::move(reversedError)).find("canonical") !=
                  std::string::npos,
          "noncanonical ActionBatch was accepted");

  const std::array<SpatialMappingAction, 2> duplicate{
      SpatialRealizationBindingAction{SpatialComputeBindingAction{0, 1, 2}},
      SpatialRealizationBindingAction{SpatialComputeBindingAction{0, 3, 4}},
  };
  llvm::Error duplicateError = validateCanonicalSpatialActionBatch(duplicate);
  require(duplicateError &&
              llvm::toString(std::move(duplicateError)).find("unique") !=
                  std::string::npos,
          "duplicate ActionBatch anchor was accepted");

  const std::array<SpatialMappingAction, 2> duplicateNet{
      SpatialTransportRoutingAction{SpatialWholeNetRoutingAction{4}},
      SpatialTransportRoutingAction{SpatialSingleSinkRoutingAction{4, 1}},
  };
  llvm::Error duplicateNetError =
      validateCanonicalSpatialActionBatch(duplicateNet);
  require(duplicateNetError &&
              llvm::toString(std::move(duplicateNetError)).find("unique") !=
                  std::string::npos,
          "two routing neighborhoods changed the same net decision");
}

void routingNeighborhoodsShareOneChoiceRange() {
  using namespace loom::pnr;
  const std::array<SpatialActionChoiceRange, 1> anchors{{{0, 3}}};
  const std::array<SpatialTransportRoutingAction, 3> choices{
      SpatialWholeNetRoutingAction{2}, SpatialSingleSinkRoutingAction{2, 0},
      SpatialRootedSubtreeRoutingAction{2, 7}};
  DeterministicPnrRandomStream stream = actionStream();
  const auto proposal =
      take(proposeSpatialAction(loom::ResolvedPnrActionProposalPolicy{0, 1, 0},
                                {{}, {}, anchors, choices, {}, {}}, stream));
  require(proposal.has_value(),
          "canonical local routing choice range produced no Action");
}

} // namespace

int main() {
  liveKindsAreReducedAndConsumeThreeDraws();
  emptyAndMalformedDomainsDoNotConsumeEntropy();
  policySumMustFitTheBoundedProtocol();
  actionBatchesRequireCanonicalUniqueAnchors();
  routingNeighborhoodsShareOneChoiceRange();
  llvm::outs() << "spatial action proposal tests passed\n";
  return 0;
}
