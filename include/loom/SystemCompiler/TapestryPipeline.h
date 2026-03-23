#ifndef LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
#define LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H

#include "loom/SystemCompiler/BendersDriver.h"
#include "loom/TDG/ContractLegalityChecker.h"
#include "loom/MultiCoreSim/MultiCoreSimSession.h"

#include <string>
#include <vector>

namespace loom {
namespace syscomp {

// End-to-end pipeline for the Tapestry system compiler.
//
// The pipeline orchestrates three stages:
//   1. Benders decomposition: partition tasks across cores.
//   2. Contract legality: validate that data-movement contracts are legal.
//   3. Multi-core simulation: estimate end-to-end latency.
//
// This class wires the three subsystems together.
class TapestryPipeline {
public:
  explicit TapestryPipeline(const BendersDriverOptions &options);

  // Add a task to the pipeline.
  void addTask(const BendersTask &task);

  // Add an edge (data dependency) to the pipeline.
  void addEdge(const BendersEdge &edge);

  // Run the full pipeline: partition, check legality, simulate.
  // Returns an error string on failure, empty string on success.
  std::string run();

  // Accessors for results after a successful run().
  const BendersResult &getBendersResult() const { return bendersResult_; }
  const mcsim::MultiCoreSimResult &getSimResult() const { return simResult_; }
  bool legalityPassed() const { return legalityPassed_; }

private:
  BendersDriverOptions options_;
  std::vector<BendersTask> tasks_;
  std::vector<BendersEdge> edges_;

  BendersResult bendersResult_;
  mcsim::MultiCoreSimResult simResult_;
  bool legalityPassed_ = false;
};

} // namespace syscomp
} // namespace loom

#endif // LOOM_SYSTEMCOMPILER_TAPESTRYPIPELINE_H
