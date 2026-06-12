#include "Simulator/CycleSummary.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
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

struct DFGSummary {
  CycleSummaryRow row;
  bool complete = true;
};

struct CGRASummary {
  std::string kernel;
  std::optional<std::uint64_t> cycles;
  std::string status;
  std::string diagnostic;
  bool complete = true;
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
      kernel.str(), std::nullopt, std::nullopt, "blocked", diagnostic.str(),
  };
}

void appendDiagnostic(std::string &target, llvm::StringRef diagnostic) {
  if (diagnostic.empty())
    return;
  if (!target.empty())
    target += "; ";
  target += diagnostic.str();
}

CycleSummaryRow proxyOnlyRow(llvm::StringRef kernel,
                             const PrimitiveStats &stats) {
  std::string diagnostic =
      "primitive-count proxy only; DFG-sim report unavailable; "
      "CGRA-sim requires Fabric ADG and mapping artifact evidence; "
      "primitive_op_count=" +
      std::to_string(stats.totalOps);
  return CycleSummaryRow{
      kernel.str(), std::nullopt,          std::nullopt,
      "blocked",    std::move(diagnostic),
  };
}

std::string diagnosticFromJsonArray(const llvm::json::Object &object) {
  const llvm::json::Array *array = object.getArray("diagnostics");
  if (!array || array->empty())
    return "";
  std::string diagnostic;
  for (const llvm::json::Value &value : *array) {
    std::optional<llvm::StringRef> text = value.getAsString();
    if (!text)
      continue;
    if (!diagnostic.empty())
      diagnostic += "; ";
    diagnostic += text->str();
  }
  return diagnostic;
}

llvm::Expected<CycleSummaryRow> summarizeOneDFGReport(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError())
    return llvm::createStringError(ec, "could not read %s", path.str().c_str());
  auto parsedOrErr = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsedOrErr)
    return parsedOrErr.takeError();
  const llvm::json::Object *object = parsedOrErr->getAsObject();
  if (!object)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG report %s is not a JSON object",
                                   path.str().c_str());

  std::optional<llvm::StringRef> kind = object->getString("kind");
  if (!kind || *kind != "dfg_sim_report")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG report %s has wrong kind",
                                   path.str().c_str());

  std::optional<llvm::StringRef> workload = object->getString("workload");
  std::optional<llvm::StringRef> graph = object->getString("graph");
  std::string kernel;
  if (workload && !workload->empty())
    kernel = workload->str();
  else if (graph && !graph->empty())
    kernel = graph->str();
  else
    return llvm::createStringError(std::errc::invalid_argument,
                                   "DFG report %s has no workload or graph",
                                   path.str().c_str());

  std::optional<llvm::StringRef> status = object->getString("status");
  std::string reportStatus = status ? status->str() : "blocked";
  std::string diagnostic = diagnosticFromJsonArray(*object);

  if (reportStatus != "pass") {
    if (diagnostic.empty())
      diagnostic = "DFG-sim report did not pass";
    return CycleSummaryRow{std::move(kernel), std::nullopt, std::nullopt,
                           reportStatus, std::move(diagnostic)};
  }

  std::optional<int64_t> cycles = object->getInteger("optimistic_cycles");
  if (!cycles || *cycles < 0)
    return CycleSummaryRow{
        std::move(kernel), std::nullopt, std::nullopt, "blocked",
        "DFG-sim report passed but lacks non-negative optimistic_cycles"};

  if (!diagnostic.empty())
    diagnostic += "; ";
  diagnostic +=
      "DFG-sim report available; CGRA-sim requires Fabric ADG and mapping "
      "artifact evidence";
  return CycleSummaryRow{std::move(kernel), static_cast<std::uint64_t>(*cycles),
                         std::nullopt, "blocked", std::move(diagnostic)};
}

llvm::Expected<CGRASummary> summarizeOneCGRAReport(llvm::StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError())
    return llvm::createStringError(ec, "could not read %s", path.str().c_str());
  auto parsedOrErr = llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!parsedOrErr)
    return parsedOrErr.takeError();
  const llvm::json::Object *object = parsedOrErr->getAsObject();
  if (!object)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "CGRA report %s is not a JSON object",
                                   path.str().c_str());

  std::optional<llvm::StringRef> kind = object->getString("kind");
  if (!kind || *kind != "cgra_sim_report")
    return llvm::createStringError(std::errc::invalid_argument,
                                   "CGRA report %s has wrong kind",
                                   path.str().c_str());

  std::optional<llvm::StringRef> workload = object->getString("workload");
  if (!workload || workload->empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "CGRA report %s has no workload",
                                   path.str().c_str());

  std::optional<llvm::StringRef> status = object->getString("status");
  std::string reportStatus = status ? status->str() : "blocked";
  std::string diagnostic = diagnosticFromJsonArray(*object);
  if (reportStatus != "pass") {
    if (diagnostic.empty())
      diagnostic = "CGRA-sim report did not pass";
    return CGRASummary{workload->str(), std::nullopt, reportStatus,
                       std::move(diagnostic), false};
  }

  std::optional<int64_t> cycles = object->getInteger("hardware_aware_cycles");
  if (!cycles || *cycles < 0)
    return CGRASummary{workload->str(), std::nullopt, "blocked",
                       "CGRA-sim report passed but lacks non-negative "
                       "hardware_aware_cycles",
                       false};

  if (diagnostic.empty())
    diagnostic = "CGRA-sim report available";
  return CGRASummary{workload->str(), static_cast<std::uint64_t>(*cycles),
                     "pass", std::move(diagnostic), true};
}

} // namespace

llvm::SmallVector<CycleSummaryRow> loom::sim::scaffoldCycleSummaryRows() {
  return {blockedRow(
      "scaffold", "DFG-sim and CGRA-sim cycle evidence is not available yet")};
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
      return llvm::createStringError(
          std::errc::invalid_argument, "invalid op_count %s for workload %s",
          rawCount.str().c_str(), workload.str().c_str());
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

void mergeDFGRow(DFGSummary &target, CycleSummaryRow source) {
  CycleSummaryRow &row = target.row;
  if (row.kernel.empty())
    row.kernel = std::move(source.kernel);
  if (!source.dfgSimCycles) {
    target.complete = false;
    row.dfgSimCycles.reset();
    row.status = source.status.empty() ? "blocked" : source.status;
    appendDiagnostic(row.diagnostic, source.diagnostic);
    return;
  }
  if (!target.complete) {
    if (row.status.empty())
      row.status = "blocked";
    appendDiagnostic(row.diagnostic, source.diagnostic);
    return;
  }
  if (row.dfgSimCycles)
    *row.dfgSimCycles += *source.dfgSimCycles;
  else if (row.status.empty() || row.status == "blocked")
    row.dfgSimCycles = *source.dfgSimCycles;
  if (row.status.empty())
    row.status = source.status;
  if (row.status == "pass" && source.status != "pass")
    row.status = source.status;
  appendDiagnostic(row.diagnostic, source.diagnostic);
}

void mergeCGRASummary(CGRASummary &target, CGRASummary source) {
  if (target.kernel.empty())
    target.kernel = std::move(source.kernel);
  if (!source.complete || !source.cycles) {
    target.complete = false;
    target.cycles.reset();
    target.status = source.status.empty() ? "blocked" : source.status;
    appendDiagnostic(target.diagnostic, source.diagnostic);
    return;
  }
  if (!target.complete) {
    if (target.status.empty())
      target.status = "blocked";
    appendDiagnostic(target.diagnostic, source.diagnostic);
    return;
  }
  if (target.cycles)
    *target.cycles += *source.cycles;
  else if (target.status.empty() || target.status == "blocked")
    target.cycles = *source.cycles;
  if (target.status.empty())
    target.status = source.status;
  if (target.status == "pass" && source.status != "pass")
    target.status = source.status;
  appendDiagnostic(target.diagnostic, source.diagnostic);
}

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
loom::sim::summarizeDFGReports(llvm::ArrayRef<std::string> reportPaths) {
  if (reportPaths.empty())
    return scaffoldCycleSummaryRows();

  std::map<std::string, DFGSummary> byKernel;
  for (const std::string &path : reportPaths) {
    auto rowOrErr = summarizeOneDFGReport(path);
    if (!rowOrErr)
      return rowOrErr.takeError();
    std::string kernel = rowOrErr->kernel;
    mergeDFGRow(byKernel[kernel], std::move(*rowOrErr));
  }
  llvm::SmallVector<CycleSummaryRow> rows;
  for (auto &[_, summary] : byKernel)
    rows.push_back(std::move(summary.row));
  return rows;
}

llvm::Expected<llvm::SmallVector<CycleSummaryRow>>
loom::sim::summarizeSimulationReports(
    llvm::ArrayRef<std::string> dfgReportPaths,
    llvm::ArrayRef<std::string> cgraReportPaths) {
  auto dfgRowsOrErr = summarizeDFGReports(dfgReportPaths);
  if (!dfgRowsOrErr)
    return dfgRowsOrErr.takeError();
  if (cgraReportPaths.empty())
    return *dfgRowsOrErr;

  std::map<std::string, CGRASummary> cgraByKernel;
  for (const std::string &path : cgraReportPaths) {
    auto cgraOrErr = summarizeOneCGRAReport(path);
    if (!cgraOrErr)
      return cgraOrErr.takeError();
    std::string key = cgraOrErr->kernel;
    mergeCGRASummary(cgraByKernel[key], std::move(*cgraOrErr));
  }

  for (CycleSummaryRow &row : *dfgRowsOrErr) {
    auto cgraIt = cgraByKernel.find(row.kernel);
    if (cgraIt == cgraByKernel.end())
      continue;
    CGRASummary &cgra = cgraIt->second;
    if (!row.dfgSimCycles) {
      row.status = "blocked";
      appendDiagnostic(
          row.diagnostic,
          "CGRA-sim report available but DFG-sim cycles are missing");
      continue;
    }
    if (!cgra.complete || !cgra.cycles) {
      row.status = cgra.status;
      appendDiagnostic(row.diagnostic, cgra.diagnostic);
      continue;
    }
    if (*cgra.cycles < *row.dfgSimCycles)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "CGRA-sim cycles for %s are more optimistic than DFG-sim cycles",
          row.kernel.c_str());
    row.cgraSimCycles = *cgra.cycles;
    row.status = "pass";
    row.diagnostic =
        "DFG-sim and CGRA-sim reports available; CGRA-sim includes mapping "
        "route, memory, and temporal penalties";
  }
  return *dfgRowsOrErr;
}

llvm::Error
loom::sim::writeCycleSummaryCsv(llvm::StringRef outputPath,
                                llvm::ArrayRef<CycleSummaryRow> rows) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(ec, "could not create %s", parent.c_str());
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
    out << ',' << csvEscape(row.status) << ',' << csvEscape(row.diagnostic)
        << '\n';
  }
  return llvm::Error::success();
}
