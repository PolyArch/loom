#ifndef LOOM_LIB_PNR_ENDPOINTMATCHING_H
#define LOOM_LIB_PNR_ENDPOINTMATCHING_H

#include "llvm/ADT/ArrayRef.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::pnr::detail {

struct EndpointDomainRange {
  std::size_t offset;
  std::size_t count;
};

class EndpointMatchingScratch {
public:
  void reset(std::size_t endpointCount);
  std::size_t beginDomain() const;
  void addEndpoint(std::size_t endpoint);
  EndpointDomainRange endDomain(std::size_t offset);
  llvm::ArrayRef<std::size_t> endpoints(EndpointDomainRange range) const;
  bool allDomainsNonEmpty() const;
  bool hasInjectiveBinding();

private:
  static void advanceGeneration(std::uint64_t &generation,
                                std::vector<std::uint64_t> &marks);
  bool augment(std::size_t demand);

  std::vector<std::size_t> endpoints_;
  std::vector<EndpointDomainRange> domains_;
  std::vector<std::size_t> matchedDemand_;
  std::vector<std::uint64_t> matchedGeneration_;
  std::vector<std::uint64_t> visitedGeneration_;
  std::size_t activeEndpointCount_ = 0;
  std::uint64_t matchingGeneration_ = 0;
  std::uint64_t probeGeneration_ = 0;
};

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_ENDPOINTMATCHING_H
