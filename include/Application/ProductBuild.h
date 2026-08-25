#ifndef LOOM_APPLICATION_PRODUCTBUILD_H
#define LOOM_APPLICATION_PRODUCTBUILD_H

#include "DSE/JointDesignPolicy.h"
#include "DSE/PreMappingFrontier.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom::application {

inline constexpr std::uint64_t defaultProductTechCandidateLimit = 8;
inline constexpr std::uint64_t defaultProductMappingWallTimeLimitMilliseconds =
    120000;

struct ProductBuildOptions final {
  std::string deploymentOutput;
  std::string accelerationProfile;
  std::string externalHardwarePath;
  std::string visualizationPath;
  std::string localToolConfigPath;
  std::vector<std::string> operatorProtocolSymbols;
  std::uint64_t mappingTechCandidateLimit = defaultProductTechCandidateLimit;
  std::uint64_t mappingWallTimeLimitMilliseconds =
      defaultProductMappingWallTimeLimitMilliseconds;
  dse::JointDesignStoppingPolicy mappingStoppingPolicy =
      dse::JointDesignStoppingPolicy::FirstVerified;
  dse::PreMappingSpectrumEndpoint mappingSpectrumEndpoint =
      dse::PreMappingSpectrumEndpoint::Automatic;
  std::string portfolioManifestPath;
  std::string portfolioRepositoryRoot;
  std::string portfolioCacheRoot;
  std::string portfolioApplicationIdentity;
  std::string portfolioInputName;
};

llvm::Expected<dse::JointDesignStoppingPolicy>
parseProductMappingStoppingPolicy(llvm::StringRef spelling);

llvm::Expected<dse::PreMappingSpectrumEndpoint>
parseProductMappingSpectrumEndpoint(llvm::StringRef spelling);

/// One public product invocation owns target resolution, all invocation-only
/// bindings, the bounded Mapping workspace, and final Deployment publication.
/// The same object projects compiler arguments and consumes the resulting LLD
/// final-link output so no CLI protocol can become a second product owner.
class ProductBuildInvocation final {
public:
  static llvm::Expected<std::unique_ptr<ProductBuildInvocation>>
  create(ProductBuildOptions options);

  ProductBuildInvocation(ProductBuildInvocation &&) noexcept;
  ProductBuildInvocation &operator=(ProductBuildInvocation &&) noexcept;
  ~ProductBuildInvocation();

  ProductBuildInvocation(const ProductBuildInvocation &) = delete;
  ProductBuildInvocation &operator=(const ProductBuildInvocation &) = delete;

  std::vector<std::string> compilerArguments() const;
  llvm::Error buildFromFinalLink(llvm::StringRef finalLinkOutput);

private:
  class Impl;
  explicit ProductBuildInvocation(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace loom::application

#endif // LOOM_APPLICATION_PRODUCTBUILD_H
