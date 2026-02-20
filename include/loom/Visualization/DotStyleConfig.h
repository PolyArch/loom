//===- DotStyleConfig.h - Centralized DOT Visualization Styles --*- C++ -*-===//
//
// Centralized styling configuration for all DOT graph visualizations.
// This file defines the SINGLE source of truth for all operator type →
// (shape, color, border) mappings used by handshake CDFG visualizations.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_VISUALIZATION_DOTSTYLECONFIG_H
#define LOOM_VISUALIZATION_DOTSTYLECONFIG_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace loom {
namespace visualization {

//===----------------------------------------------------------------------===//
// Node Style Configuration
//===----------------------------------------------------------------------===//

/// Complete visual style for a DOT node
struct DotNodeStyle {
  std::string shape;       // DOT shape (box, circle, triangle, etc.)
  std::string fillColor;   // Fill color
  std::string fontColor;   // Font color (default: black)
  std::string borderStyle; // Border line style (solid, dashed, dotted)
  double borderWidth;      // Border width
  std::string extraAttrs;  // Extra DOT attributes (e.g., width, height)

  DotNodeStyle(llvm::StringRef s = "box", llvm::StringRef fc = "white",
               llvm::StringRef fnc = "black", llvm::StringRef bs = "solid",
               double bw = 1.0, llvm::StringRef extra = "")
      : shape(s), fillColor(fc), fontColor(fnc), borderStyle(bs),
        borderWidth(bw), extraAttrs(extra) {}
};

/// Complete visual style for a DOT edge
struct DotEdgeStyle {
  std::string color; // Edge color
  std::string style; // Edge style (solid, dashed, dotted)
  double width;      // Edge width
  bool showLabel;    // Whether to show label (port mapping)

  DotEdgeStyle(llvm::StringRef c = "black", llvm::StringRef s = "solid",
               double w = 1.0, bool label = false)
      : color(c), style(s), width(w), showLabel(label) {}
};

//===----------------------------------------------------------------------===//
// Centralized Style Provider
//===----------------------------------------------------------------------===//

class DotStyleConfig {
public:
  /// Get node style for a given operation
  /// \param opName Operation name (e.g., "arith.addi", "handshake.fork")
  /// \param isMapped Whether this node is mapped (for hardware graphs)
  /// \param isFuncArg Whether this is a function argument
  /// \param isReturn Whether this is a return operation
  static DotNodeStyle getNodeStyle(llvm::StringRef opName, bool isMapped = true,
                                   bool isFuncArg = false,
                                   bool isReturn = false);

  /// Get edge style
  /// \param isMapped Whether this edge is mapped
  /// \param showPortLabel Whether to show port mapping label
  static DotEdgeStyle getEdgeStyle(bool isMapped = true,
                                   bool showPortLabel = false);

  /// Get fallback style for unhandled operations
  static DotNodeStyle getFallbackStyle();

  /// Format port mapping label
  /// \param srcPort Source port index
  /// \param dstPort Destination port index
  static std::string formatPortLabel(int srcPort, int dstPort);

private:
  /// Check if operation is from arith dialect
  static bool isArithOp(llvm::StringRef opName);

  /// Check if operation is from math dialect
  static bool isMathOp(llvm::StringRef opName);

  /// Check if operation is an LLVM intrinsic
  static bool isLLVMIntrinsic(llvm::StringRef opName);

  /// Check if operation is UB dialect
  static bool isUBOp(llvm::StringRef opName);

  /// Check if operation is handshake memory-related
  static bool isHandshakeMemoryOp(llvm::StringRef opName);

  /// Check if operation is handshake.fork
  static bool isHandshakeFork(llvm::StringRef opName);

  /// Check if operation is handshake.mux
  static bool isHandshakeMux(llvm::StringRef opName);

  /// Get style for handshake memory operations
  static DotNodeStyle getHandshakeMemoryStyle(llvm::StringRef opName);
};

} // namespace visualization
} // namespace loom

#endif // LOOM_VISUALIZATION_DOTSTYLECONFIG_H
