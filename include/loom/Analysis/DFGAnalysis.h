//===-- DFGAnalysis.h - DFG Analysis Framework --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Analyzes handshake+dataflow MLIR graphs to classify operations by execution
// characteristics. Produces machine-readable annotations (loom.analysis dict
// attrs) that guide ADGGen (spatial/temporal partitioning) and the Mapper
// (placement scoring).
//
// Two-level architecture:
//   Level A (MLIR-level): loop structure detection, execution frequency
//   Level B (Graph-level): recurrence detection, critical path, temporal score
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_ANALYSIS_DFGANALYSIS_H
#define LOOM_ANALYSIS_DFGANALYSIS_H

#include "loom/Mapper/Graph.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringRef.h"

namespace loom {
namespace analysis {

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

/// Configuration for the DFG analysis framework.
struct DFGAnalysisConfig {
  /// Default trip count when loom.loop.tripcount annotation is absent.
  int64_t defaultTripCount = 100;

  /// Threshold for temporal candidacy score. Ops with score >= threshold
  /// are considered temporal candidates.
  double temporalThreshold = 0.5;

  /// Weights for temporal candidacy score computation.
  double w1 = 0.4; // exec_freq weight (higher freq => more spatial)
  double w2 = 0.3; // critical_path weight
  double w3 = 0.2; // loop_depth weight (outside loops => more temporal)
  double w4 = 0.1; // recurrence weight

  /// Print human-readable analysis summary to stdout.
  bool dumpAnalysis = false;
};

//===----------------------------------------------------------------------===//
// Level A: MLIR-level analysis
//===----------------------------------------------------------------------===//

/// Run Level A analysis on a handshake::FuncOp.
/// Detects loop structure via dataflow.stream tracing and computes execution
/// frequency from trip count annotations. Attaches "loom.analysis" DictionaryAttr
/// to each operation in the function body.
void analyzeMLIR(circt::handshake::FuncOp funcOp,
                 const DFGAnalysisConfig &config = {});

//===----------------------------------------------------------------------===//
// Level B: Graph-level analysis
//===----------------------------------------------------------------------===//

/// Run Level B analysis on an extracted DFG Graph.
/// Performs recurrence detection, critical path estimation, and computes
/// the composite temporal candidacy score. Updates node attributes with
/// "loom.*" prefixed analysis results.
void analyzeGraph(Graph &graph, const DFGAnalysisConfig &config = {});

//===----------------------------------------------------------------------===//
// Attribute accessors (for reading analysis results from Graph nodes)
//===----------------------------------------------------------------------===//

/// Get an integer analysis attribute from a Graph node.
/// Returns defaultVal if the attribute is not present.
inline int64_t getAnalysisIntAttr(const Node *node, llvm::StringRef key,
                                  int64_t defaultVal = 0) {
  if (!node)
    return defaultVal;
  for (auto &attr : node->attributes) {
    if (attr.getName() == key) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return defaultVal;
}

/// Get a float analysis attribute from a Graph node.
/// Returns defaultVal if the attribute is not present.
inline double getAnalysisFloatAttr(const Node *node, llvm::StringRef key,
                                   double defaultVal = 0.0) {
  if (!node)
    return defaultVal;
  for (auto &attr : node->attributes) {
    if (attr.getName() == key) {
      if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr.getValue()))
        return floatAttr.getValueAsDouble();
    }
  }
  return defaultVal;
}

/// Get a boolean analysis attribute from a Graph node.
/// Returns defaultVal if the attribute is not present.
inline bool getAnalysisBoolAttr(const Node *node, llvm::StringRef key,
                                bool defaultVal = false) {
  if (!node)
    return defaultVal;
  for (auto &attr : node->attributes) {
    if (attr.getName() == key) {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return defaultVal;
}

/// Check if a Graph node has an analysis attribute with the given key.
inline bool hasAnalysisAttr(const Node *node, llvm::StringRef key) {
  if (!node)
    return false;
  for (auto &attr : node->attributes) {
    if (attr.getName() == key)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// MLIR attribute helpers
//===----------------------------------------------------------------------===//

/// Read a "loom.analysis" DictionaryAttr from an MLIR operation.
/// Returns nullptr if not present.
mlir::DictionaryAttr getAnalysisDict(mlir::Operation *op);

/// Check if an operation's op_name indicates it is forced spatial-only
/// (dataflow.*, handshake.*, memory ops).
bool isForcedSpatialOp(llvm::StringRef opName);

//===----------------------------------------------------------------------===//
// Write-back utility
//===----------------------------------------------------------------------===//

/// Write Level B analysis results from Graph node attributes back to the
/// corresponding MLIR operations in a handshake::FuncOp.
/// This updates on_recurrence, recurrence_id, on_critical_path, and
/// temporal_score in each op's "loom.analysis" DictionaryAttr.
void writeBackToMLIR(const Graph &graph, circt::handshake::FuncOp funcOp);

//===----------------------------------------------------------------------===//
// Dump utility
//===----------------------------------------------------------------------===//

/// Print a human-readable analysis summary for a handshake::FuncOp.
void dumpAnalysisSummary(circt::handshake::FuncOp funcOp);

} // namespace analysis
} // namespace loom

#endif // LOOM_ANALYSIS_DFGANALYSIS_H
