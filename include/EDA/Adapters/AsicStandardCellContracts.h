#ifndef LOOM_EDA_ADAPTERS_ASICSTANDARDCELLCONTRACTS_H
#define LOOM_EDA_ADAPTERS_ASICSTANDARDCELLCONTRACTS_H

#include "Hardware/Implementation/HardwareImplementation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom::eda {

inline constexpr llvm::StringLiteral asicStandardCellLibertyInputSlot =
    "standard_cell_liberty";
inline constexpr llvm::StringLiteral cadenceGenusStandardCellContractRef =
    "cadence.genus.standard_cell_library";
inline constexpr llvm::StringLiteral
    synopsysDesignCompilerStandardCellContractRef =
        "synopsys.design_compiler.standard_cell_library";
inline constexpr llvm::StringLiteral openSourceYosysStandardCellContractRef =
    "open_source.yosys.standard_cell_library";
inline constexpr llvm::StringLiteral openRoadTechnologyLefInputSlot =
    "technology_lef";
inline constexpr llvm::StringLiteral openRoadCellLefInputSlot =
    "cell_lef.cells";
inline constexpr llvm::StringLiteral openRoadLibertyInputSlot =
    "liberty.timing";
inline constexpr llvm::StringLiteral openRoadRoutedStandardCellContractRef =
    "open_source.openroad.routed_standard_cell_library";

llvm::Error addAsicStandardCellContract(
    hardware::ExternalImplementationContractCatalog &catalog,
    llvm::StringRef contractRef);

llvm::Error addOpenRoadRoutedStandardCellContract(
    hardware::ExternalImplementationContractCatalog &catalog);

llvm::Expected<hardware::ExternalImplementationContractCatalog>
makeKnownAsicStandardCellContractCatalog();

} // namespace loom::eda

#endif // LOOM_EDA_ADAPTERS_ASICSTANDARDCELLCONTRACTS_H
