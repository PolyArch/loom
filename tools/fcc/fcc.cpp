#include "fcc_args.h"
#include "fcc_runtime.h"
#include "fcc_pipeline.h"

#include "fcc/Dialect/Dataflow/DataflowDialect.h"
#include "fcc/Dialect/Dataflow/DataflowOps.h"
#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/ADG/ADGVerifier.h"
#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/DFGBuilder.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/TypeCompat.h"
#include "fcc/Simulator/SimArtifactWriter.h"
#include "fcc/Simulator/SimBundle.h"
#include "fcc/Simulator/SimInputSynthesis.h"
#include "fcc/Simulator/RuntimeImageBuilder.h"
#include "fcc/Simulator/SimSession.h"
#include "fcc/SVGen/SVGen.h"
#include "fcc/Viz/VizExporter.h"

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
using namespace fcc;

static OwningOpRef<ModuleOp>
loadMLIR(const std::string &path, MLIRContext &context) {
  llvm::SourceMgr srcMgr;
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf) {
    llvm::errs() << "fcc: cannot open " << path << "\n";
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
    const std::vector<fcc::sim::NamedCounter> &counters, unsigned indent) {
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
    const std::string &statPath, const fcc::sim::SimResult &simResult,
    const fcc::sim::SimSession &session,
    const std::vector<SynthesizedOutputInfo> &outputs,
    const fcc::sim::SynthesizedSetup &synthSetup,
    const std::vector<std::vector<uint8_t>> &regionStorage) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc: cannot open standalone simulation result '" << path
                 << "': " << ec.message() << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"success\": " << (simResult.success ? "true" : "false") << ",\n";
  out << "  \"termination\": ";
  writeEscapedJsonString(out, fcc::sim::runTerminationName(simResult.termination));
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
    llvm::errs() << "fcc: warning: cannot read ADG for capacity summary: "
                 << adgPath << "\n";
    return;
  }

  fcc::ADGFlattener flattener;
  if (!flattener.flatten(*adgModule, &context)) {
    llvm::errs() << "fcc: warning: cannot flatten ADG for capacity summary: "
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
    if (auto info = fcc::detail::getPortTypeInfo(port->type)) {
      maxDataWidth = std::max(maxDataWidth, info->valueWidth);
      continue;
    }
    if (auto memWidth = fcc::detail::getMemRefElementWidth(port->type))
      maxDataWidth = std::max(maxDataWidth, *memWidth);
  }

  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    for (mlir::Operation &bodyOp : fuOp.getBody().front().getOperations()) {
      if (bodyOp.getName().getStringRef() == "handshake.join")
        maxJoinFanin =
            std::max<unsigned>(maxJoinFanin, bodyOp.getNumOperands());
    }
  });

  Builder builder(&context);
  module->setAttr("fcc.adg_total_pes",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalPEs)));
  module->setAttr("fcc.adg_total_fus",
                  builder.getI64IntegerAttr(static_cast<int64_t>(totalFUs)));
  module->setAttr(
      "fcc.adg_total_mem_modules",
      builder.getI64IntegerAttr(static_cast<int64_t>(totalMemModules)));
  module->setAttr(
      "fcc.adg_total_mem_regions",
      builder.getI64IntegerAttr(static_cast<int64_t>(totalMemRegions)));
  module->setAttr(
      "fcc.adg_max_data_width",
      builder.getI64IntegerAttr(static_cast<int64_t>(maxDataWidth)));
  module->setAttr(
      "fcc.adg_max_join_fanin",
      builder.getI64IntegerAttr(static_cast<int64_t>(maxJoinFanin)));

  llvm::outs() << "fcc: ADG capacity summary: PEs=" << totalPEs
               << ", FUs=" << totalFUs
               << ", mem=" << totalMemModules
               << ", regions=" << totalMemRegions
               << ", maxWidth=" << maxDataWidth
               << ", maxJoin=" << maxJoinFanin << "\n";
}

static void warnMapperOwnedRuntimeConfig(ModuleOp adgModule) {
  unsigned muxCount = 0;
  adgModule->walk([&](fcc::fabric::MuxOp muxOp) {
    (void)muxOp;
    ++muxCount;
  });
  if (muxCount == 0)
    return;
  llvm::errs()
      << "fcc: warning: ADG contains " << muxCount
      << " fabric.mux runtime-config hints; mapper treats them as "
         "initial values and may overwrite sel/discard/disconnect\n";
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse arguments
  FccArgs args;
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
  registry.insert<fcc::dataflow::DataflowDialect>();
  registry.insert<fcc::fabric::FabricDialect>();
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
    llvm::outs() << "fcc: loading ADG from " << args.adgPath << "...\n";

    llvm::SourceMgr adgSourceMgr;
    auto adgBuf = llvm::MemoryBuffer::getFile(args.adgPath);
    if (!adgBuf) {
      llvm::errs() << "fcc: cannot open ADG file: " << args.adgPath << "\n";
      return 1;
    }
    adgSourceMgr.AddNewSourceBuffer(std::move(*adgBuf), llvm::SMLoc());
    auto adgModule = parseSourceFile<ModuleOp>(adgSourceMgr, &context);
    if (!adgModule) {
      llvm::errs() << "fcc: failed to parse ADG MLIR\n";
      return 1;
    }

    // Verify fabric.module compliance (no dangling ports)
    if (failed(fcc::verifyFabricModule(*adgModule))) {
      llvm::errs() << "fcc: ADG fabric.module verification failed\n";
      return 1;
    }
    warnMapperOwnedRuntimeConfig(*adgModule);

    llvm::outs() << "fcc: flattening ADG...\n";
    fcc::ADGFlattener flattener;
    if (!flattener.flatten(*adgModule, &context)) {
      llvm::errs() << "fcc: ADG flattening failed\n";
      return 1;
    }

    llvm::outs() << "fcc: building DFG...\n";
    fcc::DFGBuilder dfgBuilder;
    if (!dfgBuilder.build(*dfgModule, &context)) {
      llvm::errs() << "fcc: DFG building failed\n";
      return 1;
    }

    llvm::outs() << "fcc: running mapper...\n";
    fcc::Mapper mapper;
    fcc::Mapper::Options mapOpts = args.mapperOptions;
    const bool snapshotEnabled =
        mapOpts.snapshotIntervalSeconds > 0.0 ||
        mapOpts.snapshotIntervalRounds > 0;
    if (snapshotEnabled) {
      const std::string snapshotDir = args.outputDir + "/mapper-snapshots";
      auto snapshotSerial = std::make_shared<std::atomic<unsigned>>(0);
      auto snapshotMutex = std::make_shared<std::mutex>();
      mapper.setSnapshotCallback(
          [snapshotDir, snapshotSerial, snapshotMutex, mixedStem,
           &args, &adgModule, &context, &dfgBuilder, &dfgModule,
           &flattener](const MappingState &snapshotState,
                       llvm::ArrayRef<TechMappedEdgeKind> snapshotEdgeKinds,
                       llvm::ArrayRef<FUConfigSelection> snapshotFuConfigs,
                       llvm::StringRef trigger, unsigned mapperOrdinal) {
            std::lock_guard<std::mutex> lock(*snapshotMutex);
            if (llvm::sys::fs::create_directories(snapshotDir)) {
              llvm::errs()
                  << "fcc: warning: cannot create mapper snapshot dir "
                  << snapshotDir << "\n";
              return;
            }

            unsigned serial = snapshotSerial->fetch_add(1) + 1;
            std::string triggerLabel = sanitizeSnapshotLabel(trigger);
            std::string serialStr =
                llvm::formatv("{0:0>4}", serial).str();
            std::string ordinalStr =
                llvm::formatv("{0:0>4}", mapperOrdinal).str();
            std::string snapshotBase =
                snapshotDir + "/" + mixedStem + ".snapshot-" + serialStr +
                "." + triggerLabel + ".mapper-" + ordinalStr;

            fcc::ConfigGen snapshotConfigGen;
            if (!snapshotConfigGen.generate(snapshotState, dfgBuilder.getDFG(),
                                            flattener.getADG(), flattener,
                                            snapshotEdgeKinds,
                                            snapshotFuConfigs, snapshotBase,
                                            args.mapperOptions.seed)) {
              llvm::errs() << "fcc: warning: mapper snapshot config export "
                              "failed for "
                           << trigger << "\n";
              return;
            }

            std::string snapshotVizPath = snapshotBase + ".viz.html";
            std::string snapshotMapPath = snapshotBase + ".map.json";
            if (failed(fcc::exportVizWithMapping(snapshotVizPath, *adgModule,
                                                *dfgModule, snapshotMapPath,
                                                args.adgPath, args.vizLayout,
                                                &context))) {
              llvm::errs()
                  << "fcc: warning: mapper snapshot visualization failed for "
                  << trigger << "\n";
              return;
            }

            llvm::outs() << "fcc: mapper snapshot " << serial << " ("
                         << trigger << ")\n";
            llvm::outs() << "  " << snapshotVizPath << "\n";
          });
    }

    auto mapResult =
        mapper.run(dfgBuilder.getDFG(), flattener.getADG(), flattener,
                   *adgModule, mapOpts);

    if (!mapResult.success) {
      llvm::errs() << "fcc: mapping failed: " << mapResult.diagnostics << "\n";
    }

    llvm::outs() << "fcc: generating config...\n";
    fcc::ConfigGen configGen;
    if (!configGen.generate(mapResult.state, dfgBuilder.getDFG(),
                            flattener.getADG(), flattener,
                            mapResult.edgeKinds, mapResult.fuConfigs, mixedBase,
                            args.mapperOptions.seed)) {
      llvm::errs() << "fcc: config generation failed\n";
      return 1;
    }
    llvm::outs() << "fcc: mapping output:\n";
    llvm::outs() << "  " << mixedBase << ".config.bin\n";
    llvm::outs() << "  " << mixedBase << ".config.json\n";
    llvm::outs() << "  " << mixedBase << ".config.h\n";
    llvm::outs() << "  " << mixedBase << ".map.json\n";
    llvm::outs() << "  " << mixedBase << ".map.txt\n";
    if (!configGen.isConfigComplete()) {
      llvm::outs() << "fcc: warning: config artifacts include all currently "
                      "serialized slice families, but some slice contents are "
                      "still incomplete\n";
    }

    // Generate visualization with mapping data
    std::string vizPath = mixedBase + ".viz.html";
    std::string mapJsonPath = mixedBase + ".map.json";
    llvm::outs() << "fcc: generating visualization...\n";

    // We need the original MLIR modules for viz serialization.
    // adgModule is already loaded above. dfgModule is the parameter.
    if (failed(fcc::exportVizWithMapping(vizPath, *adgModule, *dfgModule,
                                        mapJsonPath, args.adgPath,
                                        args.vizLayout,
                                        &context))) {
      llvm::errs() << "fcc: warning: visualization generation failed\n";
    } else {
      llvm::outs() << "  " << vizPath << "\n";
    }

    if (!mapResult.success)
      return 1;

    std::string runtimeManifestPath = mixedBase + ".runtime.json";
    std::string runtimeImagePath = mixedBase + ".simimage.json";
    std::string runtimeImageBinPath = mixedBase + ".simimage.bin";
    fcc::sim::RuntimeImage runtimeImage;
    {
      std::string runtimeImageError;
      if (!fcc::sim::buildRuntimeImage(
              dfgBuilder.getDFG(), flattener.getADG(), mapResult.state,
              flattener.getPEContainment(), configGen.getConfigSlices(),
              configGen.getConfigWords(), runtimeImage, runtimeImageError) ||
          !fcc::sim::writeRuntimeImageJson(runtimeImage, runtimeImagePath,
                                           runtimeImageError) ||
          !fcc::sim::writeRuntimeImageBinary(runtimeImage, runtimeImageBinPath,
                                             runtimeImageError)) {
        llvm::errs() << "fcc: failed to write runtime image: "
                     << runtimeImageError << "\n";
        return 1;
      }
    }
    llvm::outs() << "  " << runtimeImagePath << "\n";
    llvm::outs() << "  " << runtimeImageBinPath << "\n";
    if (!writeRuntimeManifest(runtimeManifestPath, mixedStem,
                              selectedDfgPath, args.adgPath,
                              mixedBase + ".config.bin", runtimeImagePath,
                              runtimeImageBinPath,
                              dfgBuilder.getDFG(), flattener.getADG(),
                              mapResult.state)) {
      llvm::errs() << "fcc: failed to write runtime manifest\n";
      return 1;
    }
    llvm::outs() << "  " << runtimeManifestPath << "\n";

    if (args.simulate) {
      llvm::outs() << "fcc: running standalone simulation...\n";
      fcc::sim::SimConfig simConfig;
      simConfig.maxCycles = args.simMaxCycles;
      fcc::sim::SynthesizedSetup synthSetup;
      fcc::sim::ResolvedSimulationBundle resolvedBundle;
      bool hasResolvedBundle = false;
      if (!args.simBundlePath.empty()) {
        fcc::sim::SimulationBundle bundle;
        std::string bundleError;
        if (!fcc::sim::loadSimulationBundle(args.simBundlePath, bundle,
                                            bundleError)) {
          llvm::errs() << "fcc: failed to load simulation bundle: "
                       << bundleError << "\n";
          return 1;
        }
        if (!fcc::sim::resolveSimulationBundle(
                bundle, dfgBuilder.getDFG(), flattener.getADG(),
                mapResult.state, resolvedBundle, bundleError)) {
          llvm::errs() << "fcc: failed to resolve simulation bundle: "
                       << bundleError << "\n";
          return 1;
        }
        synthSetup = resolvedBundle.setup;
        hasResolvedBundle = true;
      } else {
        synthSetup = fcc::sim::synthesizeSimulationSetup(
            dfgBuilder.getDFG(), flattener.getADG(), mapResult.state);
      }
      std::string tracePath = mixedBase + ".sim.trace";
      std::string statPath = mixedBase + ".sim.stat";
      std::string setupPath = mixedBase + ".sim.setup.json";
      std::string resultPath = mixedBase + ".sim.result.json";
      std::string reportPath = mixedBase + ".sim.report.json";
      fcc::sim::SimSession session(nullptr, simConfig);
      fcc::sim::SimArtifactWriter artifactWriter;

      if (std::string err = session.connect(); !err.empty()) {
        llvm::errs() << "fcc: simulation setup failed: " << err << "\n";
        return 1;
      }
      if (std::string err =
              session.buildFromStaticModel(std::move(runtimeImage.staticModel));
          !err.empty()) {
        llvm::errs() << "fcc: simulation graph build failed: " << err << "\n";
        return 1;
      }
      if (std::string err = session.loadConfig(configGen.getConfigBlob(),
                                               configGen.getConfigSlices());
          !err.empty()) {
        llvm::errs() << "fcc: simulation config load failed: " << err << "\n";
        return 1;
      }

      if (!fcc::sim::writeSetupManifest(synthSetup, setupPath)) {
        llvm::errs() << "fcc: failed to write simulation setup manifest\n";
        return 1;
      }

      for (const auto &input : synthSetup.inputs) {
        if (std::string err =
                session.setInput(input.portIdx, input.data, input.tags);
            !err.empty()) {
          llvm::errs() << "fcc: failed to bind synthetic input port "
                       << input.portIdx << ": " << err << "\n";
          return 1;
        }
      }

      std::vector<std::vector<uint8_t>> regionStorage;
      regionStorage.reserve(synthSetup.memoryRegions.size());
      for (const auto &region : synthSetup.memoryRegions) {
        regionStorage.push_back(region.data);
      }
      for (size_t idx = 0; idx < synthSetup.memoryRegions.size(); ++idx) {
        const auto &region = synthSetup.memoryRegions[idx];
        auto &bytes = regionStorage[idx];
        if (std::string err = session.setExtMemoryBacking(
                region.regionId, bytes.data(), bytes.size());
            !err.empty()) {
          llvm::errs() << "fcc: failed to bind synthetic memory region "
                       << region.regionId << ": " << err << "\n";
          return 1;
        }
      }

      auto [simResult, invokeErr] = session.invoke();
      std::vector<SynthesizedOutputInfo> synthesizedOutputs =
          collectSynthesizedOutputs(dfgBuilder.getDFG(), flattener.getADG(),
                                    mapResult.state);
      if (!artifactWriter.writeTrace(simResult, tracePath) ||
          !artifactWriter.writeStat(simResult, statPath) ||
          !writeStandaloneSimulationResult(resultPath, tracePath, statPath,
                                           simResult, session,
                                           synthesizedOutputs, synthSetup,
                                           regionStorage)) {
        llvm::errs() << "fcc: failed to write simulation artifacts\n";
        return 1;
      }

      llvm::outs() << "  " << tracePath << "\n";
      llvm::outs() << "  " << statPath << "\n";
      llvm::outs() << "  " << setupPath << "\n";
      llvm::outs() << "  " << resultPath << "\n";

      if (!invokeErr.empty()) {
        llvm::errs() << "fcc: simulation invocation failed: " << invokeErr
                     << "\n";
        return 1;
      }
      if (!simResult.success) {
        llvm::errs() << "fcc: simulation failed: " << simResult.errorMessage
                     << "\n";
        return 1;
      }
      if (hasResolvedBundle) {
        fcc::sim::SimValidationReport report =
            fcc::sim::validateSimulationBundle(session, resolvedBundle);
        if (!fcc::sim::writeValidationReport(report, reportPath)) {
          llvm::errs() << "fcc: failed to write simulation validation report\n";
          return 1;
        }
        llvm::outs() << "  " << reportPath << "\n";
        if (!report.pass) {
          llvm::errs() << "fcc: simulation validation failed";
          if (!report.diagnostics.empty())
            llvm::errs() << ": " << report.diagnostics.front();
          llvm::errs() << "\n";
          return 1;
        }
      }

      if (failed(fcc::exportVizWithMapping(vizPath, *adgModule, *dfgModule,
                                          mapJsonPath, args.adgPath,
                                          args.vizLayout, &context))) {
        llvm::errs() << "fcc: warning: visualization refresh after simulation "
                        "failed\n";
      }
    }
    return 0;
  };

  // ===== SVGen mode: generate SystemVerilog from ADG =====
  if (args.genSV) {
    if (args.adgPath.empty()) {
      llvm::errs() << "fcc: --gen-sv requires --adg\n";
      return 1;
    }
    auto adgMod = loadMLIR(args.adgPath, context);
    if (!adgMod)
      return 1;
    if (failed(fcc::verifyFabricModule(*adgMod))) {
      llvm::errs() << "fcc: ADG fabric.module verification failed\n";
      return 1;
    }

    fcc::svgen::SVGenOptions svOpts;
    svOpts.rtlSourceDir = std::string(FCC_SOURCE_DIR) + "/src/rtl";
    svOpts.outputDir = args.outputDir;
    svOpts.fpIpProfile = args.fpIpProfile;

    llvm::outs() << "fcc: generating SystemVerilog...\n";
    if (!fcc::svgen::generateSV(*adgMod, &context, svOpts)) {
      llvm::errs() << "fcc: SystemVerilog generation failed\n";
      return 1;
    }
    return 0;
  }

  // ===== Viz-only mode: just visualize, no mapping =====
  if (args.vizOnly) {
    OwningOpRef<ModuleOp> adgMod, dfgMod;
    if (!args.adgPath.empty()) {
      adgMod = loadMLIR(args.adgPath, context);
      if (!adgMod) return 1;
      if (failed(fcc::verifyFabricModule(*adgMod))) {
        llvm::errs() << "fcc: ADG fabric.module verification failed\n";
        return 1;
      }
      llvm::outs() << "fcc: loaded ADG from " << args.adgPath << "\n";
    }
    if (!args.dfgPath.empty()) {
      dfgMod = loadMLIR(args.dfgPath, context);
      if (!dfgMod) return 1;
      llvm::outs() << "fcc: loaded DFG from " << args.dfgPath << "\n";
    }
    std::string vizPath =
        (!args.adgPath.empty() && !args.dfgPath.empty())
            ? mixedBase + ".viz.html"
            : (args.adgPath.empty() ? softwareBase + ".viz.html"
                                    : hardwareBase + ".viz.html");
    llvm::outs() << "fcc: generating viz-only...\n";
    mlir::LogicalResult vizResult =
        args.mapJsonPath.empty()
            ? fcc::exportVizOnly(vizPath, adgMod ? *adgMod : ModuleOp(),
                                 dfgMod ? *dfgMod : ModuleOp(), args.adgPath,
                                 args.vizLayout, &context)
            : fcc::exportVizWithMapping(vizPath,
                                        adgMod ? *adgMod : ModuleOp(),
                                        dfgMod ? *dfgMod : ModuleOp(),
                                        args.mapJsonPath, args.adgPath,
                                        args.vizLayout, &context);
    if (failed(vizResult)) {
      llvm::errs() << "fcc: viz generation failed\n";
      return 1;
    }
    llvm::outs() << "  " << vizPath << "\n";
    return 0;
  }

  // ===== DFG-direct mode: skip frontend, load pre-built DFG =====
  if (!args.dfgPath.empty()) {
    llvm::outs() << "fcc: loading DFG from " << args.dfgPath << "...\n";
    llvm::SourceMgr dfgSourceMgr;
    auto dfgBuf = llvm::MemoryBuffer::getFile(args.dfgPath);
    if (!dfgBuf) {
      llvm::errs() << "fcc: cannot open DFG file: " << args.dfgPath << "\n";
      return 1;
    }
    dfgSourceMgr.AddNewSourceBuffer(std::move(*dfgBuf), llvm::SMLoc());
    auto dfgModule = parseSourceFile<ModuleOp>(dfgSourceMgr, &context);
    if (!dfgModule) {
      llvm::errs() << "fcc: failed to parse DFG MLIR\n";
      return 1;
    }

    return runMappingPipeline(dfgModule);
  }

  // ===== Full pipeline: C -> LLVM -> CF -> SCF -> DFG =====
  std::string llPath = softwareBase + ".ll";
  llvm::outs() << "fcc: compiling and importing...\n";
  auto module = compileAndImport(args, context, llPath);
  if (!module)
    return 1;

  std::string llvmMlirPath = softwareBase + ".llvm.mlir";
  if (failed(writeMLIR(*module, llvmMlirPath)))
    return 1;

  llvm::outs() << "fcc: converting LLVM to CF...\n";
  if (failed(runLLVMToCF(*module)))
    return 1;

  std::string cfPath = softwareBase + ".cf.mlir";
  if (failed(writeMLIR(*module, cfPath)))
    return 1;

  llvm::outs() << "fcc: lifting CF to SCF...\n";
  if (failed(runCFToSCF(*module)))
    return 1;

  std::string scfPath = softwareBase + ".scf.mlir";
  if (failed(writeMLIR(*module, scfPath)))
    return 1;

  attachADGCapacityAttrs(*module, args.adgPath, context);

  llvm::outs() << "fcc: converting SCF to DFG...\n";
  if (failed(runSCFToDFG(*module)))
    return 1;

  std::string dfgPath = softwareBase + ".dfg.mlir";
  if (failed(writeMLIR(*module, dfgPath)))
    return 1;

  std::string hostPath = softwareBase + "_host.c";
  std::string accelHeaderPath = args.outputDir + "/fcc_accel.h";
  std::string accelRuntimePath = args.outputDir + "/fcc_accel.c";
  std::string origSource =
      args.sources.empty() ? "" : args.sources[0];
  llvm::outs() << "fcc: generating host code...\n";
  if (failed(runHostCodeGen(*module, hostPath, origSource)))
    return 1;

  llvm::outs() << "fcc: compilation complete.\n";
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
