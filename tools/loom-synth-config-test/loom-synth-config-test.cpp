// Tiny CLI used by lit tests to exercise the SynthConfig YAML/TOML loader.
//
// Usage: loom-synth-config-test [<path>]
//
// With a path, parses the file and prints one key=value pair per line in a
// stable order. Without a path, dumps the built-in defaults so tests can
// verify them without touching the parser.
//
// Errors are reported as `error: ...` to stderr and exit non-zero.

#include "Common/SynthConfig.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("[config-path]"),
              ::llvm::cl::Optional);

static void dump(const ::loom::SynthConfig &cfg) {
  ::llvm::outs() << "strategy=" << cfg.strategy << "\n";
  ::llvm::outs() << "parallelism.cross_group="
                 << (cfg.parallelismCrossGroup ? "true" : "false") << "\n";
  ::llvm::outs() << "parallelism.workers=" << cfg.parallelismWorkers << "\n";
  ::llvm::outs() << "coverage_verifier.parallel_match="
                 << (cfg.coverageVerifierParallelMatch ? "true" : "false")
                 << "\n";
  ::llvm::outs() << "fallback_chain.size=" << cfg.fallbackChain.size() << "\n";
  for (size_t i = 0; i < cfg.fallbackChain.size(); ++i)
    ::llvm::outs() << "fallback_chain[" << i << "]=" << cfg.fallbackChain[i]
                   << "\n";
  ::llvm::outs() << "cost.mux_penalty=" << cfg.costMuxPenalty << "\n";
  ::llvm::outs() << "cost.demux_penalty=" << cfg.costDemuxPenalty << "\n";
  ::llvm::outs() << "cost.carry_penalty=" << cfg.costCarryPenalty << "\n";
  ::llvm::outs() << "anchor.allow_intra_position_mux="
                 << (cfg.anchorAllowIntraPositionMux ? "true" : "false")
                 << "\n";
  ::llvm::outs() << "incremental.input_order_heuristic="
                 << cfg.incrementalInputOrderHeuristic << "\n";
  ::llvm::outs() << "incremental.coverage_verify_each_attempt="
                 << (cfg.incrementalCoverageVerifyEachAttempt ? "true"
                                                              : "false")
                 << "\n";
  ::llvm::outs() << "incremental_random.restarts=" << cfg.incrementalRandomRestarts
                 << "\n";
  ::llvm::outs() << "incremental_random.seed=" << cfg.incrementalRandomSeed
                 << "\n";
  ::llvm::outs() << "incremental_random.input_order_heuristic="
                 << cfg.incrementalRandomInputOrderHeuristic << "\n";
  ::llvm::outs() << "mcs.timeout_sec=" << cfg.mcsTimeoutSec << "\n";
  ::llvm::outs() << "mcs.branch_workers=" << cfg.mcsBranchWorkers << "\n";
  ::llvm::outs() << "mcs.candidate_cap=" << cfg.mcsCandidateCap << "\n";
  ::llvm::outs() << "scc_full_unroll="
                 << (cfg.sccFullUnroll ? "true" : "false") << "\n";
  ::llvm::outs() << "subgraph_share_recurse="
                 << (cfg.subgraphShareRecurse ? "true" : "false") << "\n";
}

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                       "loom-synth-config-test: parse and "
                                       "dump a SynthConfig file\n");
  if (inputPath.getValue().empty()) {
    ::loom::SynthConfig cfg;
    dump(cfg);
    return 0;
  }
  auto cfg = ::loom::loadSynthConfig(inputPath.getValue());
  if (!cfg) {
    ::llvm::errs() << "error: " << ::llvm::toString(cfg.takeError()) << "\n";
    return 1;
  }
  dump(*cfg);
  return 0;
}
