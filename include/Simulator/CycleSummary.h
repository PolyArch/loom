#ifndef LOOM_SIMULATOR_CYCLE_SUMMARY_H
#define LOOM_SIMULATOR_CYCLE_SUMMARY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace loom {
namespace sim {

struct CycleSummaryOptions {
  bool emitPrimitiveCountProxy = true;
};

struct CycleSummaryRow {
  std::string kernel;
  std::optional<std::uint64_t> dfgSimCycles;
  std::optional<std::uint64_t> cgraSimCycles;
  std::string status;
  std::string diagnostic;
};

llvm::SmallVector<CycleSummaryRow> scaffoldCycleSummaryRows();

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
summarizePrimitiveCoverage(llvm::StringRef csvPath,
                           const CycleSummaryOptions &options);

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
summarizeDFGReports(llvm::ArrayRef<std::string> reportPaths);

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
summarizeSimulationReports(llvm::ArrayRef<std::string> dfgReportPaths,
                           llvm::ArrayRef<std::string> cgraReportPaths);

llvm::Error writeCycleSummaryCsv(llvm::StringRef outputPath,
                                 llvm::ArrayRef<CycleSummaryRow> rows);

} // namespace sim
} // namespace loom

#endif // LOOM_SIMULATOR_CYCLE_SUMMARY_H
