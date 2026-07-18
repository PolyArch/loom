#include "EndpointMatching.h"

#include <algorithm>
#include <cassert>
#include <limits>

using namespace loom::pnr::detail;

void EndpointMatchingScratch::reset(std::size_t endpointCount) {
  endpoints_.clear();
  domains_.clear();
  activeEndpointCount_ = endpointCount;
  if (matchedDemand_.size() < endpointCount) {
    matchedDemand_.resize(endpointCount);
    matchedGeneration_.resize(endpointCount);
    visitedGeneration_.resize(endpointCount);
  }
}

std::size_t EndpointMatchingScratch::beginDomain() const {
  return endpoints_.size();
}

void EndpointMatchingScratch::addEndpoint(std::size_t endpoint) {
  endpoints_.push_back(endpoint);
}

EndpointDomainRange EndpointMatchingScratch::endDomain(std::size_t offset) {
  EndpointDomainRange range{offset, endpoints_.size() - offset};
  domains_.push_back(range);
  return range;
}

llvm::ArrayRef<std::size_t>
EndpointMatchingScratch::endpoints(EndpointDomainRange range) const {
  return llvm::ArrayRef<std::size_t>(endpoints_)
      .slice(range.offset, range.count);
}

bool EndpointMatchingScratch::allDomainsNonEmpty() const {
  return std::all_of(
      domains_.begin(), domains_.end(),
      [](EndpointDomainRange range) { return range.count != 0; });
}

bool EndpointMatchingScratch::hasInjectiveBinding() {
  if (!allDomainsNonEmpty())
    return false;
  advanceGeneration(matchingGeneration_, matchedGeneration_);
  for (std::size_t demand = 0; demand < domains_.size(); ++demand) {
    advanceGeneration(probeGeneration_, visitedGeneration_);
    if (!augment(demand))
      return false;
  }
  return true;
}

void EndpointMatchingScratch::advanceGeneration(
    std::uint64_t &generation, std::vector<std::uint64_t> &marks) {
  if (generation == std::numeric_limits<std::uint64_t>::max()) {
    std::fill(marks.begin(), marks.end(), 0);
    generation = 1;
    return;
  }
  ++generation;
}

bool EndpointMatchingScratch::augment(std::size_t demand) {
  const EndpointDomainRange range = domains_[demand];
  for (std::size_t endpoint : endpoints(range)) {
    assert(endpoint < activeEndpointCount_);
    if (visitedGeneration_[endpoint] == probeGeneration_)
      continue;
    visitedGeneration_[endpoint] = probeGeneration_;
    if (matchedGeneration_[endpoint] != matchingGeneration_ ||
        augment(matchedDemand_[endpoint])) {
      matchedDemand_[endpoint] = demand;
      matchedGeneration_[endpoint] = matchingGeneration_;
      return true;
    }
  }
  return false;
}
