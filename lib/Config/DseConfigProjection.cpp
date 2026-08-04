#include "DSE/ResolvedConfigView.h"

#include "Config/ResolvedConfig.h"

namespace loom::dse {

llvm::Expected<ResolvedDseConfigView>
projectResolvedDseConfigView(const ResolvedConfig &config) {
  return ResolvedDseConfigView::get(
      config.dse.modelAuthorizations, config.dse.evidenceObligationTemplates,
      config.dse.objectiveCatalogs, config.dse.qualityGatePolicies,
      config.dse.planNodes);
}

} // namespace loom::dse
