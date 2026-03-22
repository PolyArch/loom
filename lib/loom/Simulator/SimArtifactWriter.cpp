#include "loom/Simulator/SimArtifactWriter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {
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

  TraceDocument doc = result.traceDocument;
  if (doc.events.empty())
    doc.events = result.traceEvents;
  if (doc.events.empty()) {
    doc.version = 1;
    doc.traceKind = "loom_cycle_trace";
    doc.producer = "loom";
  } else {
    doc.epochId = doc.events.front().epochId;
    doc.invocationId = doc.events.front().invocationId;
    doc.coreId = doc.events.front().coreId;
  }

  out << "{\n";
  out << "  \"version\": " << doc.version << ",\n";
  out << "  \"trace_kind\": ";
  writeEscapedJsonString(out, doc.traceKind);
  out << ",\n";
  out << "  \"producer\": ";
  writeEscapedJsonString(out, doc.producer);
  out << ",\n";
  out << "  \"epoch_id\": " << doc.epochId << ",\n";
  out << "  \"invocation_id\": " << doc.invocationId << ",\n";
  out << "  \"core_id\": " << doc.coreId << ",\n";
  out << "  \"modules\": [\n";
  for (size_t idx = 0; idx < doc.modules.size(); ++idx) {
    const TraceModuleInfo &module = doc.modules[idx];
    out << "  {\"hw_node_id\": " << module.hwNodeId
        << ", \"kind\": ";
    writeEscapedJsonString(out, module.kind);
    out << ", \"name\": ";
    writeEscapedJsonString(out, module.name);
    out << ", \"component_name\": ";
    writeEscapedJsonString(out, module.componentName);
    out << ", \"function_unit_name\": ";
    writeEscapedJsonString(out, module.functionUnitName);
    out << ", \"boundary_ordinal\": " << module.boundaryOrdinal << "}";
    if (idx + 1 != doc.modules.size())
      out << ",";
    out << "\n";
  }
  out << "  ],\n";
  out << "  \"events\": [\n";
  for (size_t idx = 0; idx < doc.events.size(); ++idx) {
    const TraceEvent &event = doc.events[idx];
    out << "  {\"cycle\": " << event.cycle;
    out << ", \"phase\": ";
    writeEscapedJsonString(out, simPhaseName(event.phase));
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
    if (idx + 1 != doc.events.size())
      out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
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
} // namespace loom
