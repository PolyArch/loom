#include "Simulator/CycleSummary.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <string>
#include <system_error>

using namespace loom::sim;

namespace {

struct PrimitiveStats {
  std::uint64_t totalOps = 0;
};

llvm::SmallVector<std::string> parseCsvLine(llvm::StringRef line) {
  llvm::SmallVector<std::string> cells;
  std::string cell;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (quoted) {
      if (ch == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          cell.push_back('"');
          ++i;
        } else {
          quoted = false;
        }
      } else {
        cell.push_back(ch);
      }
      continue;
    }
    if (ch == '"') {
      quoted = true;
    } else if (ch == ',') {
      cells.push_back(cell);
      cell.clear();
    } else {
      cell.push_back(ch);
    }
  }
  cells.push_back(cell);
  return cells;
}

std::string csvEscape(llvm::StringRef value) {
  if (value.find_first_of(",\"\n\r") == llvm::StringRef::npos)
    return value.str();
  std::string escaped = "\"";
  for (char ch : value) {
    if (ch == '"')
      escaped.push_back('"');
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

bool ignoredIdentity(llvm::StringRef value) {
  return value.empty() || value == "scaffold" || value == "none";
}

llvm::Error requireColumn(const llvm::StringMap<unsigned> &columns,
                          llvm::StringRef name, llvm::StringRef path) {
  if (columns.contains(name))
    return llvm::Error::success();
  return llvm::createStringError(std::errc::invalid_argument,
                                 "primitive coverage CSV %s is missing %s",
                                 path.str().c_str(), name.str().c_str());
}

llvm::StringMap<unsigned> indexHeader(llvm::ArrayRef<std::string> header) {
  llvm::StringMap<unsigned> columns;
  for (auto [index, name] : llvm::enumerate(header))
    columns.try_emplace(name, static_cast<unsigned>(index));
  return columns;
}

llvm::StringRef cellAt(llvm::ArrayRef<std::string> row, unsigned index) {
  if (index >= row.size())
    return "";
  return row[index];
}

CycleSummaryRow blockedRow(llvm::StringRef kernel, llvm::StringRef diagnostic) {
  return CycleSummaryRow{
      kernel.str(),
      std::nullopt,
      std::nullopt,
      "blocked",
      diagnostic.str(),
  };
}

CycleSummaryRow proxyOnlyRow(llvm::StringRef kernel,
                             const PrimitiveStats &stats) {
  std::string diagnostic =
      "primitive-count proxy only; DFG-sim report unavailable; "
      "CGRA-sim requires Fabric ADG and mapping artifact evidence; "
      "primitive_op_count=" +
      std::to_string(stats.totalOps);
  return CycleSummaryRow{
      kernel.str(),
      std::nullopt,
      std::nullopt,
      "blocked",
      std::move(diagnostic),
  };
}

} // namespace

llvm::SmallVector<CycleSummaryRow> loom::sim::scaffoldCycleSummaryRows() {
  return {blockedRow(
      "scaffold",
      "DFG-sim and CGRA-sim cycle evidence is not available yet")};
}

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
loom::sim::summarizePrimitiveCoverage(llvm::StringRef csvPath,
                                      const CycleSummaryOptions &options) {
  if (!options.emitPrimitiveCountProxy)
    return scaffoldCycleSummaryRows();

  auto bufferOrErr = llvm::MemoryBuffer::getFile(csvPath);
  if (std::error_code ec = bufferOrErr.getError())
    return llvm::createStringError(ec, "could not read %s",
                                   csvPath.str().c_str());

  llvm::SmallVector<llvm::StringRef> lines;
  (*bufferOrErr)->getBuffer().split(lines, '\n');
  if (lines.empty())
    return scaffoldCycleSummaryRows();

  llvm::StringRef headerLine = lines.front().rtrim("\r");
  auto header = parseCsvLine(headerLine);
  auto columns = indexHeader(header);
  if (auto err = requireColumn(columns, "workload", csvPath))
    return std::move(err);
  if (auto err = requireColumn(columns, "primitive", csvPath))
    return std::move(err);
  if (auto err = requireColumn(columns, "op_count", csvPath))
    return std::move(err);

  const unsigned workloadColumn = columns.lookup("workload");
  const unsigned opCountColumn = columns.lookup("op_count");

  std::map<std::string, PrimitiveStats> byWorkload;
  for (llvm::StringRef rawLine : llvm::drop_begin(lines)) {
    llvm::StringRef line = rawLine.rtrim("\r");
    if (line.trim().empty())
      continue;
    auto row = parseCsvLine(line);
    llvm::StringRef workload = cellAt(row, workloadColumn);
    if (ignoredIdentity(workload))
      continue;
    std::uint64_t opCount = 0;
    llvm::StringRef rawCount = cellAt(row, opCountColumn);
    if (rawCount.getAsInteger(10, opCount))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "invalid op_count %s for workload %s",
                                     rawCount.str().c_str(),
                                     workload.str().c_str());
    PrimitiveStats &stats = byWorkload[workload.str()];
    stats.totalOps += opCount;
  }

  if (byWorkload.empty())
    return scaffoldCycleSummaryRows();

  llvm::SmallVector<CycleSummaryRow> rows;
  for (const auto &[workload, stats] : byWorkload) {
    if (stats.totalOps == 0) {
      rows.push_back(blockedRow(
          workload, "primitive coverage has no positive operation count"));
      continue;
    }
    rows.push_back(proxyOnlyRow(workload, stats));
  }
  return rows;
}

llvm::Error loom::sim::writeCycleSummaryCsv(
    llvm::StringRef outputPath, llvm::ArrayRef<CycleSummaryRow> rows) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(ec, "could not create %s",
                                     parent.c_str());
  }

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());

  out << "kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic\n";
  for (const CycleSummaryRow &row : rows) {
    out << csvEscape(row.kernel) << ',';
    if (row.dfgSimCycles)
      out << *row.dfgSimCycles;
    out << ',';
    if (row.cgraSimCycles)
      out << *row.cgraSimCycles;
    out << ',' << csvEscape(row.status) << ','
        << csvEscape(row.diagnostic) << '\n';
  }
  return llvm::Error::success();
}
