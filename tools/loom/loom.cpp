#include "loom_args.h"
#include "loom_runtime.h"
#include "loom_pipeline.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"
#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "loom/Dialect/Fabric/FabricTypes.h"
#include "loom/ADG/ADGVerifier.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/DFGBuilder.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/TypeCompat.h"
#include "loom/Simulator/PortTraceExporter.h"
#include "loom/Simulator/SimArtifactWriter.h"
#include "loom/Simulator/SimBundle.h"
#include "loom/Simulator/SimInputSynthesis.h"
#include "loom/Simulator/RuntimeImageBuilder.h"
#include "loom/Simulator/SimSession.h"
#include "loom/SVGen/SVGen.h"
#include "loom/Viz/VizExporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <mutex>

using namespace mlir;
using namespace loom;

static OwningOpRef<ModuleOp>
loadMLIR(const std::string &path, MLIRContext &context) {
  llvm::SourceMgr srcMgr;
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    llvm::errs() << "loom: cannot open " << path << "\n";
    return {};
  }
  srcMgr.AddNewSourceBuffer(std::move(*buf), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(srcMgr, &context);
}

static std::string deriveAdgStem(const std::string &adgPath) {
  if (adgPath.empty())
    return "";
  llvm::StringRef stem = llvm::sys::path::stem(adgPath);
  if (stem.ends_with(".fabric"))
    stem = stem.drop_back(7);
  return stem.str();
}

static std::string joinOutputBase(const std::string &outputDir,
                                  const std::string &stem) {
  return outputDir + "/" + stem;
}

static std::string sanitizeSnapshotLabel(llvm::StringRef label) {
  std::string sanitized;
  sanitized.reserve(label.size());
  bool lastDash = false;
  for (char ch : label) {
    bool alphaNum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9');
    if (alphaNum) {
      sanitized.push_back(static_cast<char>(std::tolower(ch)));
      lastDash = false;
      continue;
    }
    if (!lastDash) {
      sanitized.push_back('-');
      lastDash = true;
    }
  }
  while (!sanitized.empty() && sanitized.back() == '-')
    sanitized.pop_back();
  if (sanitized.empty())
    return "snapshot";
  return sanitized;
}

static void writeEscapedJsonString(llvm::raw_ostream &out,
                                   const std::string &value) {
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

static void writeNamedCountersJson(
    llvm::raw_ostream &out,
    const std::vector<loom::sim::NamedCounter> &counters, unsigned indent) {
  std::string pad(indent, ' ');
  std::string childPad(indent + 2, ' ');
  out << "[\n";
  for (size_t idx = 0; idx < counters.size(); ++idx) {
    const auto &counter = counters[idx];
    out << childPad << "{\"name\": ";
    writeEscapedJsonString(out, counter.name);
    out << ", \"value\": " << counter.value << "}";
    if (idx + 1 != counters.size())
      out << ",";
    out << "\n";
  }
  out << pad << "]";
}

template <typename T>
static void writeJsonIntegerArray(llvm::raw_ostream &out,
                                  const std::vector<T> &values) {
  out << "[";
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx)
      out << ", ";
    out << static_cast<uint64_t>(values[idx]);
  }
  out << "]";
}

struct SynthesizedOutputInfo {
  unsigned portIdx = 0;
  int64_t resultIndex = -1;
};

static std::vector<unsigned> buildBoundaryOutputOrdinals(const Graph &adg) {
  std::vector<unsigned> ordinals(adg.nodes.size(), 0);
  unsigned nextOrdinal = 0;
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::ModuleOutputNode)
      continue;
    ordinals[nodeId] = nextOrdinal++;
  }
  return ordinals;
}

static std::vector<SynthesizedOutputInfo>
collectSynthesizedOutputs(const Graph &dfg, const Graph &adg,
                          const MappingState &mapping) {
  std::vector<SynthesizedOutputInfo> outputs;
  std::vector<unsigned> boundaryOutputOrdinals = buildBoundaryOutputOrdinals(adg);
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::ModuleOutputNode)
      continue;
    if (swNodeId >= static_cast<IdIndex>(mapping.swNodeToHwNode.size()))
      continue;
    IdIndex hwNodeId = mapping.swNodeToHwNode[swNodeId];
    if (hwNodeId == INVALID_ID ||
        hwNodeId >= static_cast<IdIndex>(boundaryOutputOrdinals.size()))
      continue;
    SynthesizedOutputInfo output;
    output.portIdx = boundaryOutputOrdinals[hwNodeId];
    output.resultIndex = getNodeAttrInt(swNode, "result_index", -1);
    outputs.push_back(output);
  }
  std::sort(outputs.begin(), outputs.end(),
            [](const SynthesizedOutputInfo &lhs,
               const SynthesizedOutputInfo &rhs) {
              return lhs.portIdx < rhs.portIdx;
            });
  return outputs;
}

static bool writeStandaloneSimulationResult(
    const std::string &path, const std::string &tracePath,
    const std::string &statPath, const loom::sim::SimResult &simResult,
    const loom::sim::SimSession &session,
    const std::vector<SynthesizedOutputInfo> &outputs,
    const loom::sim::SynthesizedSetup &synthSetup,
    const std::vector<std::vector<uint8_t>> &regionStorage) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "loom: cannot open standalone simulation result '" << path
                 << "': " << ec.message() << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"success\": " << (simResult.success ? "true" : "false") << ",\n";
  out << "  \"termination\": ";
  writeEscapedJsonString(out, loom::sim::runTerminationName(simResult.termination));
  out << ",\n";
  out << "  \"error_message\": ";
  writeEscapedJsonString(out, simResult.errorMessage);
  out << ",\n";
  out << "  \"cycle_count\": " << simResult.totalCycles << ",\n";
  out << "  \"config_cycles\": " << simResult.configCycles << ",\n";
  out << "  \"config_writes\": " << simResult.totalConfigWrites << ",\n";
  out << "  \"trace_path\": ";
  writeEscapedJsonString(out, tracePath);
  out << ",\n";
  out << "  \"stat_path\": ";
  writeEscapedJsonString(out, statPath);
  out << ",\n";

  out << "  \"outputs\": [\n";
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto &output = outputs[idx];
    out << "    {\"port\": " << output.portIdx
        << ", \"result_index\": " << output.resultIndex << ", \"data\": ";
    writeJsonIntegerArray(out, session.getOutput(output.portIdx));
    out << ", \"tags\": ";
    writeJsonIntegerArray(out, session.getOutputTags(output.portIdx));
    out << "}";
    if (idx + 1 != outputs.size())
      out << ",";
    out << "\n";
  }
  out << "  ],\n";

  out << "  \"final_state\": {\n";
  out << "    \"obligations_satisfied\": "
      << (simResult.finalState.obligationsSatisfied ? "true" : "false")
      << ",\n";
  out << "    \"hardware_empty\": "
      << (simResult.finalState.hardwareEmpty ? "true" : "false") << ",\n";
  out << "    \"quiescent\": "
      << (simResult.finalState.quiescent ? "true" : "false") << ",\n";
  out << "    \"done\": " << (simResult.finalState.done ? "true" : "false")
      << ",\n";
  out << "    \"deadlocked\": "
      << (simResult.finalState.deadlocked ? "true" : "false") << ",\n";
  out << "    \"idle_cycle_streak\": " << simResult.finalState.idleCycleStreak
      << ",\n";
  out << "    \"outstanding_memory_request_count\": "
      << simResult.finalState.outstandingMemoryRequestCount << ",\n";
  out << "    \"completed_memory_response_count\": "
      << simResult.finalState.completedMemoryResponseCount << ",\n";
  out << "    \"termination_audit_error\": ";
  writeEscapedJsonString(out, simResult.finalState.terminationAuditError);
  out << ",\n";

  out << "    \"live_ports\": [\n";
  for (size_t idx = 0; idx < simResult.finalState.livePorts.size(); ++idx) {
    const auto &port = simResult.finalState.livePorts[idx];
    out << "      {\"port_id\": " << port.portId
        << ", \"parent_node_id\": " << port.parentNodeId
        << ", \"direction\": ";
    writeEscapedJsonString(out, port.isInput ? "input" : "output");
    out << ", \"valid\": " << (port.valid ? "true" : "false")
        << ", \"ready\": " << (port.ready ? "true" : "false")
        << ", \"data\": " << port.data << ", \"tag\": " << port.tag
        << ", \"has_tag\": " << (port.hasTag ? "true" : "false")
        << ", \"generation\": " << port.generation << "}";
    if (idx + 1 != simResult.finalState.livePorts.size())
      out << ",";
    out << "\n";
  }
  out << "    ],\n";

  out << "    \"live_edges\": [\n";
  for (size_t idx = 0; idx < simResult.finalState.liveEdges.size(); ++idx) {
    const auto &edge = simResult.finalState.liveEdges[idx];
    out << "      {\"edge_index\": " << edge.edgeIndex
        << ", \"hw_edge_id\": " << edge.hwEdgeId
        << ", \"src_port\": " << edge.srcPort
        << ", \"dst_port\": " << edge.dstPort
        << ", \"valid\": " << (edge.valid ? "true" : "false")
        << ", \"ready\": " << (edge.ready ? "true" : "false")
        << ", \"data\": " << edge.data << ", \"tag\": " << edge.tag
        << ", \"has_tag\": " << (edge.hasTag ? "true" : "false")
        << ", \"generation\": " << edge.generation << "}";
    if (idx + 1 != simResult.finalState.liveEdges.size())
      out << ",";
    out << "\n";
  }
  out << "    ],\n";

  out << "    \"pending_modules\": [\n";
  for (size_t idx = 0; idx < simResult.finalState.pendingModules.size();
       ++idx) {
    const auto &module = simResult.finalState.pendingModules[idx];
    out << "      {\"hw_node_id\": " << module.hwNodeId << ", \"name\": ";
    writeEscapedJsonString(out, module.name);
    out << ", \"kind\": ";
    writeEscapedJsonString(out, module.kind);
    out << ", \"has_pending_work\": "
        << (module.hasPendingWork ? "true" : "false")
        << ", \"collected_token_count\": " << module.collectedTokenCount
        << ", \"logical_fire_count\": " << module.logicalFireCount
        << ", \"input_capture_count\": " << module.inputCaptureCount
        << ", \"output_transfer_count\": " << module.outputTransferCount
        << ", \"debug_state\": ";
    writeEscapedJsonString(out, module.debugState);
    out << ", \"counters\": ";
    writeNamedCountersJson(out, module.counters, 6);
    out << "}";
    if (idx + 1 != simResult.finalState.pendingModules.size())
      out << ",";
    out << "\n";
  }
  out << "    ],\n";

  out << "    \"module_summaries\": [\n";
  for (size_t idx = 0; idx < simResult.finalState.moduleSummaries.size();
       ++idx) {
    const auto &module = simResult.finalState.moduleSummaries[idx];
    out << "      {\"hw_node_id\": " << module.hwNodeId << ", \"name\": ";
    writeEscapedJsonString(out, module.name);
    out << ", \"kind\": ";
    writeEscapedJsonString(out, module.kind);
    out << ", \"has_pending_work\": "
        << (module.hasPendingWork ? "true" : "false")
        << ", \"collected_token_count\": " << module.collectedTokenCount
        << ", \"logical_fire_count\": " << module.logicalFireCount
        << ", \"input_capture_count\": " << module.inputCaptureCount
        << ", \"output_transfer_count\": " << module.outputTransferCount
        << ", \"debug_state\": ";
    writeEscapedJsonString(out, module.debugState);
    out << ", \"counters\": ";
    writeNamedCountersJson(out, module.counters, 6);
    out << "}";
    if (idx + 1 != simResult.finalState.moduleSummaries.size())
      out << ",";
    out << "\n";
  }
  out << "    ]\n";
  out << "  },\n";

  out << "  \"memory_regions\": [\n";
  for (size_t idx = 0; idx < synthSetup.memoryRegions.size(); ++idx) {
    const auto &region = synthSetup.memoryRegions[idx];
    out << "    {\"region\": " << region.regionId
        << ", \"memref_arg_index\": " << region.memrefArgIndex
        << ", \"elem_size_log2\": " << region.elemSizeLog2
        << ", \"byte_size\": " << regionStorage[idx].size()
        << ", \"bytes\": ";
    writeJsonIntegerArray(out, regionStorage[idx]);
    out << "}";
    if (idx + 1 != synthSetup.memoryRegions.size())
      out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return true;
}

static void attachADGCapacityAttrs(ModuleOp module, const std::string &adgPath,
                                   MLIRContext &context) {
  if (adgPath.empty())
    return;

  auto adgModule = loadMLIR(adgPath, context);
  if (!adgModule) {
    llvm::errs() << "loom: warning: cannot read ADG for capacity summary: "
                 << adgPath << "\n";
    return;
  }

  loom::ADGFlattener flattener;
  if (!flattener.flatten(*adgModule, &context)) {
    llvm::errs() << "loom: warning: cannot flatten ADG for capacity summary: "
                 << adgPath << "\n";
    return;
  }

  const Graph &adg = flattener.getADG();
  unsigned totalPEs = flattener.getPEContainment().size();
  unsigned totalFUs = 0;
  for (const auto &pe : flattener.getPEContainment())
    totalFUs += pe.fuNodeIds.size();

  unsigned totalMemModules = 0;
  unsigned totalMemRegions = 0;
  unsigned maxDataWidth = 0;
  unsigned maxJoinFanin = 0;
  for (const Node *node : adg.nodeRange()) {
    if (getNodeAttrStr(node, "resource_class") == "memory") {
      totalMemModules++;
      totalMemRegions +=
          static_cast<unsigned>(std::max<int64_t>(1, getNodeAttrInt(node, "numRegion", 1)));
    }
  }
  for (const Port *port : adg.portRange()) {
    if (auto info = loom::detail::getPortTypeInfo(port->type)) {
      maxDataWidth = std::max(maxDataWidth, info->valueWidth);
      continue;
    }
    if (auto memWidth = loom::detail::getMemRefElementWidth(port->type))
      maxDataWidth = std::max(maxDataWidth, *memWidth);
  }

  adgModule->walk([&](loom::fabric::FunctionUnitOp fuOp) {
    for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
      if (bodyOp.getName().getStringRef() == "handshake.join")
        maxJoinFanin =
            std::max<unsigned>(maxJoinFanin, bodyOp.getNumOperands());
    }
  });

  Builder builder(&context);
  module->setAttr("loom.adg_total_pes",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalPEs)));
  module->setAttr("loom.adg_total_fus",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalFUs)));
  module->setAttr(
      "loom.adg_total_mem_modules",
      builder.getI64IntegerAttr(static_cast<int64_t>(totalMemModules)));
  module->setAttr(
      "loom.adg_total_mem_regions",
      builder.getI64IntegerAttr(static_cast<int64_t>(totalMemRegions)));
  module->setAttr(
      "loom.adg_max_data_width",
      builder.getI64IntegerAttr(static_cast<int64_t>(maxDataWidth)));
  module->setAttr(
      "loom.adg_max_join_fanin",
      builder.getI64IntegerAttr(static_cast<int64_t>(maxJoinFanin)));

  llvm::outs() << "loom: ADG capacity summary: PEs=" << totalPEs
               << ", FUs=" << totalFUs
               << ", mem=" << totalMemModules
               << ", regions=" << totalMemRegions
               << ", maxWidth=" << maxDataWidth
               << ", maxJoin=" << maxJoinFanin << "\n";
}

static void warnMapperOwnedRuntimeConfig(ModuleOp adgModule) {
  unsigned muxCount = 0;
  adgModule->walk([&](loom::fabric::MuxOp muxOp) {
    (void)muxOp;
    ++muxCount;
  });
  if (muxCount == 0)
    return;
  llvm::errs()
      << "loom: warning: ADG contains " << muxCount
      << " fabric.mux runtime-config hints; mapper treats them as "
         "initial values and may overwrite sel/discard/disconnect\n";
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse arguments
  LoomArgs args;
  if (!parseArgs(argc, argv, args))
    return 1;

  // Set up MLIR context with all needed dialects
  DialectRegistry registry;
  registry.insert<DLTIDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<ub::UBDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<loom::fabric::FabricDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (!args.runtimeManifestPath.empty())
    return runRuntimeReplay(args, context);

  const std::string softwareStem = args.baseName;
  const std::string adgStem = deriveAdgStem(args.adgPath);
  const std::string softwareBase = joinOutputBase(args.outputDir, softwareStem);
  const std::string hardwareBase =
      joinOutputBase(args.outputDir, adgStem.empty() ? softwareStem : adgStem);
  const std::string mixedStem =
      !softwareStem.empty() && !adgStem.empty()
          ? softwareStem + "." + adgStem
          : (softwareStem.empty() ? adgStem : softwareStem);
  const std::string mixedBase = joinOutputBase(args.outputDir, mixedStem);
  const std::string selectedDfgPath =
      !args.dfgPath.empty() ? args.dfgPath : (softwareBase + ".dfg.mlir");

  // Helper: load ADG, build DFG, run mapper, generate viz
  auto runMappingPipeline = [&](OwningOpRef<ModuleOp> &dfgModule) -> int {
    llvm::outs() << "loom: loading ADG from " << args.adgPath << "...\n";

    llvm::SourceMgr adgSourceMgr;
    auto adgBuf = llvm::MemoryBuffer::getFile(args.adgPath);
    if (!adgBuf) {
      llvm::errs() << "loom: cannot open ADG file: " << args.adgPath << "\n";
      return 1;
    }
    adgSourceMgr.AddNewSourceBuffer(std::move(*adgBuf), llvm::SMLoc());
    auto adgModule = parseSourceFile<ModuleOp>(adgSourceMgr, &context);
    if (!adgModule) {
      llvm::errs() << "loom: failed to parse ADG MLIR\n";
      return 1;
    }

    // Verify fabric.module compliance (no dangling ports)
    if (failed(loom::verifyFabricModule(*adgModule))) {
      llvm::errs() << "loom: ADG fabric.module verification failed\n";
      return 1;
    }
    warnMapperOwnedRuntimeConfig(*adgModule);

    llvm::outs() << "loom: flattening ADG...\n";
    loom::ADGFlattener flattener;
    if (!flattener.flatten(*adgModule, &context)) {
      llvm::errs() << "loom: ADG flattening failed\n";
      return 1;
    }

    llvm::outs() << "loom: building DFG...\n";
    loom::DFGBuilder dfgBuilder;
    if (!dfgBuilder.build(*dfgModule, &context)) {
      llvm::errs() << "loom: DFG building failed\n";
      return 1;
    }

    llvm::outs() << "loom: running mapper...\n";
    loom::Mapper::Options mapOpts = args.mapperOptions;
    const bool snapshotEnabled =
        mapOpts.snapshotIntervalSeconds > 0.0 ||
        mapOpts.snapshotIntervalRounds > 0;
    const std::string snapshotDir = args.outputDir + "/mapper-snapshots";
    auto snapshotSerial = std::make_shared<std::atomic<unsigned>>(0);
    auto snapshotMutex = std::make_shared<std::mutex>();

    struct MappingAttemptView {
      unsigned laneIndex = 0;
      const MappingState *state = nullptr;
      llvm::ArrayRef<TechMappedEdgeKind> edgeKinds;
      llvm::ArrayRef<FUConfigSelection> fuConfigs;
      const MapperTimingSummary *timingSummary = nullptr;
      const MapperSearchSummary *searchSummary = nullptr;
    };

    struct AttemptOutcome {
      bool success = false;
      bool fatal = false;
      std::string error;
    };

    auto makeMapper = [&](int activeSeed) {
      loom::Mapper mapper;
      if (snapshotEnabled) {
        mapper.setSnapshotCallback(
            [snapshotDir, snapshotSerial, snapshotMutex, mixedStem,
             &args, &adgModule, &context, &dfgBuilder, &dfgModule,
             &flattener, activeSeed](const MappingState &snapshotState,
                                     llvm::ArrayRef<TechMappedEdgeKind>
                                         snapshotEdgeKinds,
                                     llvm::ArrayRef<FUConfigSelection>
                                         snapshotFuConfigs,
                                     llvm::StringRef trigger,
                                     unsigned mapperOrdinal) {
              std::lock_guard<std::mutex> lock(*snapshotMutex);
              if (llvm::sys::fs::create_directories(snapshotDir)) {
                llvm::errs()
                    << "loom: warning: cannot create mapper snapshot dir "
                    << snapshotDir << "\n";
                return;
              }

              unsigned serial = snapshotSerial->fetch_add(1) + 1;
              std::string triggerLabel = sanitizeSnapshotLabel(trigger);
              std::string serialStr = llvm::formatv("{0:0>4}", serial).str();
              std::string ordinalStr =
                  llvm::formatv("{0:0>4}", mapperOrdinal).str();
              std::string snapshotBase =
                  snapshotDir + "/" + mixedStem + ".snapshot-" + serialStr +
                  "." + triggerLabel + ".mapper-" + ordinalStr;

              loom::ConfigGen snapshotConfigGen;
              if (!snapshotConfigGen.generate(snapshotState, dfgBuilder.getDFG(),
                                              flattener.getADG(), flattener,
                                              snapshotEdgeKinds,
                                              snapshotFuConfigs, snapshotBase,
                                              activeSeed, nullptr, nullptr,
                                              nullptr, nullptr, "")) {
                llvm::errs() << "loom: warning: mapper snapshot config export "
                                "failed for "
                             << trigger << "\n";
                return;
              }

              std::string snapshotVizPath = snapshotBase + ".viz.html";
              std::string snapshotMapPath = snapshotBase + ".map.json";
              if (failed(loom::exportVizWithMapping(snapshotVizPath, *adgModule,
                                                  *dfgModule, snapshotMapPath,
                                                  args.adgPath,
                                                  args.vizLayout, &context))) {
                llvm::errs()
                    << "loom: warning: mapper snapshot visualization failed for "
                    << trigger << "\n";
                return;
              }

              llvm::outs() << "loom: mapper snapshot " << serial << " ("
                           << trigger << ")\n";
              llvm::outs() << "  " << snapshotVizPath << "\n";
            });
      }
      return mapper;
    };

    auto runMappingAttempt =
        [&](const loom::Mapper::Result &mapResult, int activeSeed,
            const MappingAttemptView &attempt) -> AttemptOutcome {
      llvm::outs() << "loom: generating config...\n";
      loom::ConfigGen configGen;
      if (!configGen.generate(*attempt.state, dfgBuilder.getDFG(),
                              flattener.getADG(), flattener, attempt.edgeKinds,
                              attempt.fuConfigs, mixedBase, activeSeed,
                              &mapResult.techMapPlan,
                              &mapResult.techMapMetrics, attempt.timingSummary,
                              attempt.searchSummary,
                              mapResult.techMapDiagnostics)) {
        return {false, true, "loom: config generation failed"};
      }
      llvm::outs() << "loom: mapping output:\n";
      llvm::outs() << "  " << mixedBase << ".config.bin\n";
      llvm::outs() << "  " << mixedBase << ".config.json\n";
      llvm::outs() << "  " << mixedBase << ".config.h\n";
      llvm::outs() << "  " << mixedBase << ".map.json\n";
      llvm::outs() << "  " << mixedBase << ".map.txt\n";
      if (!configGen.isConfigComplete()) {
        llvm::outs() << "loom: warning: config artifacts include all currently "
                        "serialized slice families, but some slice contents are "
                        "still incomplete\n";
      }

      std::string vizPath = mixedBase + ".viz.html";
      std::string mapJsonPath = mixedBase + ".map.json";
      llvm::outs() << "loom: generating visualization...\n";
      if (failed(loom::exportVizWithMapping(vizPath, *adgModule, *dfgModule,
                                          mapJsonPath, args.adgPath,
                                          args.vizLayout, &context))) {
        llvm::errs() << "loom: warning: visualization generation failed\n";
      } else {
        llvm::outs() << "  " << vizPath << "\n";
      }

      if (!mapResult.success)
        return {true, false, ""};

      std::string runtimeManifestPath = mixedBase + ".runtime.json";
      std::string runtimeImagePath = mixedBase + ".simimage.json";
      std::string runtimeImageBinPath = mixedBase + ".simimage.bin";
      loom::sim::RuntimeImage runtimeImage;
      // Capture port/module info before the static model is moved into session.
      std::vector<loom::sim::StaticModuleDesc> savedModuleDescs;
      std::vector<loom::sim::StaticPortDesc> savedPortDescs;
      {
        std::string runtimeImageError;
        if (!loom::sim::buildRuntimeImage(
                dfgBuilder.getDFG(), flattener.getADG(), *attempt.state,
                flattener.getPEContainment(), configGen.getConfigSlices(),
                configGen.getConfigWords(), runtimeImage, runtimeImageError) ||
            !loom::sim::writeRuntimeImageJson(runtimeImage, runtimeImagePath,
                                             runtimeImageError) ||
            !loom::sim::writeRuntimeImageBinary(runtimeImage,
                                               runtimeImageBinPath,
                                               runtimeImageError)) {
          return {false, true,
                  "loom: failed to write runtime image: " + runtimeImageError};
        }
        if (!args.tracePortDump.empty()) {
          savedModuleDescs = runtimeImage.staticModel.getModules();
          savedPortDescs = runtimeImage.staticModel.getPorts();
        }
      }
      llvm::outs() << "  " << runtimeImagePath << "\n";
      llvm::outs() << "  " << runtimeImageBinPath << "\n";
      if (!writeRuntimeManifest(runtimeManifestPath, mixedStem,
                                selectedDfgPath, args.adgPath,
                                mixedBase + ".config.bin", runtimeImagePath,
                                runtimeImageBinPath, dfgBuilder.getDFG(),
                                flattener.getADG(), *attempt.state)) {
        return {false, true, "loom: failed to write runtime manifest"};
      }
      llvm::outs() << "  " << runtimeManifestPath << "\n";

      if (!args.simulate)
        return {true, false, ""};

      llvm::outs() << "loom: running standalone simulation...\n";
      loom::sim::SimConfig simConfig;
      simConfig.maxCycles = args.simMaxCycles;
      loom::sim::SynthesizedSetup synthSetup;
      loom::sim::ResolvedSimulationBundle resolvedBundle;
      bool hasResolvedBundle = false;
      if (!args.simBundlePath.empty()) {
        loom::sim::SimulationBundle bundle;
        std::string bundleError;
        if (!loom::sim::loadSimulationBundle(args.simBundlePath, bundle,
                                            bundleError)) {
          return {false, true,
                  "loom: failed to load simulation bundle: " + bundleError};
        }
        if (!loom::sim::resolveSimulationBundle(
                bundle, dfgBuilder.getDFG(), flattener.getADG(), *attempt.state,
                resolvedBundle, bundleError)) {
          return {false, true,
                  "loom: failed to resolve simulation bundle: " + bundleError};
        }
        synthSetup = resolvedBundle.setup;
        hasResolvedBundle = true;
      } else {
        synthSetup = loom::sim::synthesizeSimulationSetup(
            dfgBuilder.getDFG(), flattener.getADG(), *attempt.state);
      }
      std::string tracePath = mixedBase + ".sim.trace";
      std::string statPath = mixedBase + ".sim.stat";
      std::string setupPath = mixedBase + ".sim.setup.json";
      std::string resultPath = mixedBase + ".sim.result.json";
      std::string reportPath = mixedBase + ".sim.report.json";
      loom::sim::SimSession session(nullptr, simConfig);
      loom::sim::SimArtifactWriter artifactWriter;

      if (std::string err = session.connect(); !err.empty())
        return {false, true, "loom: simulation setup failed: " + err};
      if (std::string err =
              session.buildFromStaticModel(std::move(runtimeImage.staticModel));
          !err.empty()) {
        return {false, true, "loom: simulation graph build failed: " + err};
      }
      if (std::string err =
              session.loadConfig(configGen.getConfigBlob(),
                                 configGen.getConfigSlices());
          !err.empty()) {
        return {false, true, "loom: simulation config load failed: " + err};
      }

      if (!loom::sim::writeSetupManifest(synthSetup, setupPath))
        return {false, true, "loom: failed to write simulation setup manifest"};

      for (const auto &input : synthSetup.inputs) {
        if (std::string err =
                session.setInput(input.portIdx, input.data, input.tags);
            !err.empty()) {
          return {false, true,
                  "loom: failed to bind synthetic input port " +
                      std::to_string(input.portIdx) + ": " + err};
        }
      }

      std::vector<std::vector<uint8_t>> regionStorage;
      regionStorage.reserve(synthSetup.memoryRegions.size());
      for (const auto &region : synthSetup.memoryRegions)
        regionStorage.push_back(region.data);
      for (size_t idx = 0; idx < synthSetup.memoryRegions.size(); ++idx) {
        const auto &region = synthSetup.memoryRegions[idx];
        auto &bytes = regionStorage[idx];
        if (std::string err = session.setExtMemoryBacking(
                region.regionId, bytes.data(), bytes.size());
            !err.empty()) {
          return {false, true,
                  "loom: failed to bind synthetic memory region " +
                      std::to_string(region.regionId) + ": " + err};
        }
      }

      // Set up port trace export if requested.
      std::unique_ptr<loom::sim::PortTraceExporter> portTraceExporter;
      if (!args.tracePortDump.empty()) {
        std::string traceOutputDir = args.outputDir + "/rtl-traces";
        if (std::error_code ec =
                llvm::sys::fs::create_directories(traceOutputDir)) {
          llvm::errs() << "loom: cannot create trace output dir '"
                       << traceOutputDir << "': " << ec.message() << "\n";
          return {false, true, "loom: cannot create trace output dir"};
        }
        portTraceExporter =
            std::make_unique<loom::sim::PortTraceExporter>(traceOutputDir);

        for (unsigned modIdx = 0; modIdx < savedModuleDescs.size(); ++modIdx) {
          const auto &mod = savedModuleDescs[modIdx];
          if (mod.name != args.tracePortDump)
            continue;

          std::vector<loom::sim::PortTraceExporter::TracedPort> tracedPorts;
          auto addPorts = [&](const std::vector<IdIndex> &portIds,
                              const std::string &prefix) {
            unsigned iter_var0 = 0;
            for (IdIndex portId : portIds) {
              if (portId < static_cast<IdIndex>(savedPortDescs.size())) {
                const auto &pd = savedPortDescs[portId];
                loom::sim::PortTraceExporter::TracedPort tp;
                tp.portIndex = static_cast<unsigned>(portId);
                tp.dir = pd.direction;
                tp.valueWidth = pd.valueWidth;
                tp.tagWidth = pd.tagWidth;
                tp.isTagged = pd.isTagged;
                tp.name = prefix + std::to_string(iter_var0);
                tracedPorts.push_back(tp);
              }
              ++iter_var0;
            }
          };
          addPorts(mod.inputPorts, "in");
          addPorts(mod.outputPorts, "out");

          portTraceExporter->addTracedModule(modIdx, mod.name, tracedPorts);
          llvm::outs() << "loom: tracing ports for module '" << mod.name
                       << "' (" << tracedPorts.size() << " ports)\n";
        }

        session.setCycleCallback(
            [&portTraceExporter](uint64_t cycle,
                                 const std::vector<loom::sim::SimChannel> &ps) {
              portTraceExporter->recordCycle(cycle, ps);
            });
      }

      auto [simResult, invokeErr] = session.invoke();
      std::vector<SynthesizedOutputInfo> synthesizedOutputs =
          collectSynthesizedOutputs(dfgBuilder.getDFG(), flattener.getADG(),
                                    *attempt.state);
      if (!artifactWriter.writeTrace(simResult, tracePath) ||
          !artifactWriter.writeStat(simResult, statPath) ||
          !writeStandaloneSimulationResult(resultPath, tracePath, statPath,
                                           simResult, session,
                                           synthesizedOutputs, synthSetup,
                                           regionStorage)) {
        return {false, true, "loom: failed to write simulation artifacts"};
      }

      llvm::outs() << "  " << tracePath << "\n";
      llvm::outs() << "  " << statPath << "\n";
      llvm::outs() << "  " << setupPath << "\n";
      llvm::outs() << "  " << resultPath << "\n";

      if (portTraceExporter) {
        if (!portTraceExporter->flush()) {
          llvm::errs() << "loom: warning: port trace export flush failed\n";
        } else {
          llvm::outs() << "loom: port traces written to "
                       << args.outputDir << "/rtl-traces/\n";
        }
      }

      if (!invokeErr.empty())
        return {false, true, "loom: simulation invocation failed: " + invokeErr};
      if (!simResult.success)
        return {false, false,
                "loom: simulation failed: " + simResult.errorMessage};
      if (hasResolvedBundle) {
        loom::sim::SimValidationReport report =
            loom::sim::validateSimulationBundle(session, resolvedBundle);
        if (!loom::sim::writeValidationReport(report, reportPath)) {
          return {false, true,
                  "loom: failed to write simulation validation report"};
        }
        llvm::outs() << "  " << reportPath << "\n";
        if (!report.pass) {
          std::string error = "loom: simulation validation failed";
          if (!report.diagnostics.empty())
            error += ": " + report.diagnostics.front();
          return {false, false, error};
        }
      }

      if (failed(loom::exportVizWithMapping(vizPath, *adgModule, *dfgModule,
                                          mapJsonPath, args.adgPath,
                                          args.vizLayout, &context))) {
        llvm::errs() << "loom: warning: visualization refresh after simulation "
                        "failed\n";
      }
      return {true, false, ""};
    };

    const unsigned remapAttemptCount = args.simulate ? 4u : 1u;
    const int remapSeedStride = static_cast<int>(
        std::max(1u, args.mapperOptions.lane.restartSeedStride));
    std::string lastSimulationError;
    for (unsigned remapAttempt = 0; remapAttempt < remapAttemptCount;
         ++remapAttempt) {
      int activeSeed =
          args.mapperOptions.seed + static_cast<int>(remapAttempt) *
                                        remapSeedStride;
      loom::Mapper mapper = makeMapper(activeSeed);
      loom::Mapper::Options activeMapOpts = mapOpts;
      activeMapOpts.seed = activeSeed;
      auto mapResult =
          mapper.run(dfgBuilder.getDFG(), flattener.getADG(), flattener,
                     *adgModule, activeMapOpts);

      if (!mapResult.success) {
        llvm::errs() << "loom: mapping failed: " << mapResult.diagnostics
                     << "\n";
      }

      MappingAttemptView selectedAttempt{mapResult.selectedLaneIndex,
                                         &mapResult.state,
                                         mapResult.edgeKinds,
                                         mapResult.fuConfigs,
                                         &mapResult.timingSummary,
                                         &mapResult.searchSummary};
      AttemptOutcome selectedOutcome =
          runMappingAttempt(mapResult, activeSeed, selectedAttempt);
      if (selectedOutcome.success)
        return mapResult.success ? 0 : 1;
      if (selectedOutcome.fatal) {
        llvm::errs() << selectedOutcome.error << "\n";
        return 1;
      }

      lastSimulationError = selectedOutcome.error;
      llvm::errs() << selectedOutcome.error << "\n";
      for (const auto &alternative : mapResult.routedAlternatives) {
        llvm::outs() << "loom: retrying standalone simulation with routed lane "
                     << alternative.laneIndex << "...\n";
        MappingAttemptView alternativeAttempt{alternative.laneIndex,
                                              &alternative.state,
                                              alternative.edgeKinds,
                                              alternative.fuConfigs,
                                              &alternative.timingSummary,
                                              &alternative.searchSummary};
        AttemptOutcome alternativeOutcome =
            runMappingAttempt(mapResult, activeSeed, alternativeAttempt);
        if (alternativeOutcome.success)
          return 0;
        if (alternativeOutcome.fatal) {
          llvm::errs() << alternativeOutcome.error << "\n";
          return 1;
        }
        lastSimulationError = alternativeOutcome.error;
        llvm::errs() << alternativeOutcome.error << "\n";
      }

      if (!args.simulate)
        return 1;
      if (remapAttempt + 1 < remapAttemptCount) {
        llvm::outs() << "loom: remapping after simulation failure with seed "
                     << (activeSeed + remapSeedStride) << "...\n";
      }
    }
    if (!lastSimulationError.empty())
      llvm::errs() << lastSimulationError << "\n";
    return 1;
  };

  // ===== SVGen mode: generate SystemVerilog from ADG =====
  if (args.genSV) {
    if (args.adgPath.empty()) {
      llvm::errs() << "loom: --gen-sv requires --adg\n";
      return 1;
    }
    auto adgMod = loadMLIR(args.adgPath, context);
    if (!adgMod)
      return 1;
    if (failed(loom::verifyFabricModule(*adgMod))) {
      llvm::errs() << "loom: ADG fabric.module verification failed\n";
      return 1;
    }

    loom::svgen::SVGenOptions svOpts;
    svOpts.rtlSourceDir = std::string(LOOM_SOURCE_DIR) + "/src/rtl";
    svOpts.outputDir = args.outputDir;
    svOpts.fpIpProfile = args.fpIpProfile;

    llvm::outs() << "loom: generating SystemVerilog...\n";
    if (!loom::svgen::generateSV(*adgMod, &context, svOpts)) {
      llvm::errs() << "loom: SystemVerilog generation failed\n";
      return 1;
    }
    // If --simulate is also requested, fall through to the simulation path
    // instead of returning early. This enables combined gen+simulate+trace
    // workflows used by the behaviour verification runner.
    if (!args.simulate)
      return 0;
    // Fall through to simulation below...
  }

  // ===== Viz-only mode: just visualize, no mapping =====
  if (args.vizOnly) {
    OwningOpRef<ModuleOp> adgMod, dfgMod;
    if (!args.adgPath.empty()) {
      adgMod = loadMLIR(args.adgPath, context);
      if (!adgMod) return 1;
      if (failed(loom::verifyFabricModule(*adgMod))) {
        llvm::errs() << "loom: ADG fabric.module verification failed\n";
        return 1;
      }
      llvm::outs() << "loom: loaded ADG from " << args.adgPath << "\n";
    }
    if (!args.dfgPath.empty()) {
      dfgMod = loadMLIR(args.dfgPath, context);
      if (!dfgMod) return 1;
      llvm::outs() << "loom: loaded DFG from " << args.dfgPath << "\n";
    }
    std::string vizPath =
        (!args.adgPath.empty() && !args.dfgPath.empty())
            ? mixedBase + ".viz.html"
            : (args.adgPath.empty() ? softwareBase + ".viz.html"
                                    : hardwareBase + ".viz.html");
    llvm::outs() << "loom: generating viz-only...\n";
    mlir::LogicalResult vizResult =
        args.mapJsonPath.empty()
            ? loom::exportVizOnly(vizPath, adgMod ? *adgMod : ModuleOp(),
                                 dfgMod ? *dfgMod : ModuleOp(), args.adgPath,
                                 args.vizLayout, &context)
            : loom::exportVizWithMapping(vizPath,
                                        adgMod ? *adgMod : ModuleOp(),
                                        dfgMod ? *dfgMod : ModuleOp(),
                                        args.mapJsonPath, args.adgPath,
                                        args.vizLayout, &context);
    if (failed(vizResult)) {
      llvm::errs() << "loom: viz generation failed\n";
      return 1;
    }
    llvm::outs() << "  " << vizPath << "\n";
    return 0;
  }

  // ===== DFG-direct mode: skip frontend, load pre-built DFG =====
  if (!args.dfgPath.empty()) {
    llvm::outs() << "loom: loading DFG from " << args.dfgPath << "...\n";
    llvm::SourceMgr dfgSourceMgr;
    auto dfgBuf = llvm::MemoryBuffer::getFile(args.dfgPath);
    if (!dfgBuf) {
      llvm::errs() << "loom: cannot open DFG file: " << args.dfgPath << "\n";
      return 1;
    }
    dfgSourceMgr.AddNewSourceBuffer(std::move(*dfgBuf), llvm::SMLoc());
    auto dfgModule = parseSourceFile<ModuleOp>(dfgSourceMgr, &context);
    if (!dfgModule) {
      llvm::errs() << "loom: failed to parse DFG MLIR\n";
      return 1;
    }

    return runMappingPipeline(dfgModule);
  }

  // ===== Full pipeline: C -> LLVM -> CF -> SCF -> DFG =====
  std::string llPath = softwareBase + ".ll";
  llvm::outs() << "loom: compiling and importing...\n";
  auto module = compileAndImport(args, context, llPath);
  if (!module)
    return 1;

  std::string llvmMlirPath = softwareBase + ".llvm.mlir";
  if (failed(writeMLIR(*module, llvmMlirPath)))
    return 1;

  llvm::outs() << "loom: converting LLVM to CF...\n";
  if (failed(runLLVMToCF(*module)))
    return 1;

  std::string cfPath = softwareBase + ".cf.mlir";
  if (failed(writeMLIR(*module, cfPath)))
    return 1;

  llvm::outs() << "loom: lifting CF to SCF...\n";
  if (failed(runCFToSCF(*module)))
    return 1;

  std::string scfPath = softwareBase + ".scf.mlir";
  if (failed(writeMLIR(*module, scfPath)))
    return 1;

  attachADGCapacityAttrs(*module, args.adgPath, context);

  llvm::outs() << "loom: converting SCF to DFG...\n";
  if (failed(runSCFToDFG(*module)))
    return 1;

  std::string dfgPath = softwareBase + ".dfg.mlir";
  if (failed(writeMLIR(*module, dfgPath)))
    return 1;

  std::string hostPath = softwareBase + "_host.c";
  std::string accelHeaderPath = args.outputDir + "/loom_accel.h";
  std::string accelRuntimePath = args.outputDir + "/loom_accel.c";
  std::string origSource =
      args.sources.empty() ? "" : args.sources[0];
  llvm::outs() << "loom: generating host code...\n";
  if (failed(runHostCodeGen(*module, hostPath, origSource)))
    return 1;

  llvm::outs() << "loom: compilation complete.\n";
  llvm::outs() << "  " << llPath << "\n";
  llvm::outs() << "  " << llvmMlirPath << "\n";
  llvm::outs() << "  " << cfPath << "\n";
  llvm::outs() << "  " << scfPath << "\n";
  llvm::outs() << "  " << dfgPath << "\n";
  llvm::outs() << "  " << accelHeaderPath << "\n";
  llvm::outs() << "  " << accelRuntimePath << "\n";
  llvm::outs() << "  " << hostPath << "\n";

  if (!args.adgPath.empty()) {
    int rc = runMappingPipeline(module);
    if (rc != 0)
      return rc;
  }

  return 0;
}
