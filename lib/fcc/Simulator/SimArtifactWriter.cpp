#include "fcc/Simulator/SimArtifactWriter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace fcc {
namespace sim {

namespace {

void writeEscapedJsonString(llvm::raw_ostream &out, const std::string &value) {
  out << '"';
  for (char ch : value) {
    switch (ch) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      out << ch;
      break;
    }
  }
  out << '"';
}

} // namespace

bool SimArtifactWriter::writeTrace(const SimResult &result,
                                   const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "SimArtifactWriter: cannot open " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  out << "[\n";
  for (size_t idx = 0; idx < result.traceEvents.size(); ++idx) {
    const TraceEvent &event = result.traceEvents[idx];
    out << "  {\"cycle\": " << event.cycle;
    out << ", \"epoch_id\": " << event.epochId;
    out << ", \"invocation_id\": " << event.invocationId;
    out << ", \"core_id\": " << event.coreId;
    out << ", \"hw_node_id\": " << event.hwNodeId;
    out << ", \"event_kind\": ";
    writeEscapedJsonString(out, eventKindName(event.eventKind));
    out << ", \"lane\": " << static_cast<unsigned>(event.lane);
    out << ", \"flags\": " << event.flags;
    out << ", \"arg0\": " << event.arg0;
    out << ", \"arg1\": " << event.arg1 << "}";
    if (idx + 1 != result.traceEvents.size())
      out << ",";
    out << "\n";
  }
  out << "]\n";
  return true;
}

bool SimArtifactWriter::writeStat(const SimResult &result,
                                  const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "SimArtifactWriter: cannot open " << path << ": "
                 << ec.message() << "\n";
    return false;
  }

  out << "success: " << (result.success ? "true" : "false") << "\n";
  out << "termination: " << runTerminationName(result.termination) << "\n";
  out << "total_cycles: " << result.totalCycles << "\n";
  out << "config_cycles: " << result.configCycles << "\n";
  out << "total_config_writes: " << result.totalConfigWrites << "\n";
  out << "trace_event_count: " << result.traceEvents.size() << "\n";
  out << "perf_snapshot_count: " << result.nodePerf.size() << "\n";
  out << "error_message: ";
  if (result.errorMessage.empty())
    out << "<none>\n";
  else
    out << result.errorMessage << "\n";

  if (!result.nodePerf.empty()) {
    out << "\n[node_perf]\n";
    for (const PerfSnapshot &perf : result.nodePerf) {
      out << "node=" << perf.nodeIndex << " active=" << perf.activeCycles
          << " stall_in=" << perf.stallCyclesIn
          << " stall_out=" << perf.stallCyclesOut
          << " tokens_in=" << perf.tokensIn
          << " tokens_out=" << perf.tokensOut
          << " config_writes=" << perf.configWrites << "\n";
    }
  }

  return true;
}

} // namespace sim
} // namespace fcc
