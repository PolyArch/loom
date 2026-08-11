#include "EDA/Adapters/AsicStandardCellContracts.h"

namespace loom::eda {

llvm::Error addAsicStandardCellContract(
    hardware::ExternalImplementationContractCatalog &catalog,
    llvm::StringRef contractRef) {
  using namespace hardware;
  return catalog.add(
      ExternalImplementationContract{contractRef.str(),
                                     {{asicStandardCellLibertyInputSlot.str(),
                                       {ExternalDependencyKind::ExplicitFile}}},
                                     {RepresentationRootVariant::GateNetlist,
                                      RepresentationRootVariant::AsicPhysical},
                                     true,
                                     false,
                                     nullptr});
}

llvm::Error addOpenRoadRoutedStandardCellContract(
    hardware::ExternalImplementationContractCatalog &catalog) {
  using namespace hardware;
  return catalog.add(ExternalImplementationContract{
      openRoadRoutedStandardCellContractRef.str(),
      {{openRoadTechnologyLefInputSlot.str(),
        {ExternalDependencyKind::ExplicitFile}},
       {openRoadCellLefInputSlot.str(), {ExternalDependencyKind::ExplicitFile}},
       {openRoadLibertyInputSlot.str(),
        {ExternalDependencyKind::ExplicitFile}}},
      {RepresentationRootVariant::AsicPhysical},
      true,
      false,
      nullptr});
}

llvm::Expected<hardware::ExternalImplementationContractCatalog>
makeKnownAsicStandardCellContractCatalog() {
  hardware::ExternalImplementationContractCatalog catalog;
  for (llvm::StringRef contractRef :
       {llvm::StringRef(cadenceGenusStandardCellContractRef),
        llvm::StringRef(synopsysDesignCompilerStandardCellContractRef),
        llvm::StringRef(openSourceYosysStandardCellContractRef)})
    if (llvm::Error error = addAsicStandardCellContract(catalog, contractRef))
      return std::move(error);
  if (llvm::Error error = addOpenRoadRoutedStandardCellContract(catalog))
    return std::move(error);
  return catalog;
}

} // namespace loom::eda
