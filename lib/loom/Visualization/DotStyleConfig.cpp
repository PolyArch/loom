//===- DotStyleConfig.cpp - Centralized DOT Visualization Styles ---------===//
//
// Implementation of centralized styling for all DOT visualizations.
//
//===----------------------------------------------------------------------===//

#include "loom/Visualization/DotStyleConfig.h"
#include "llvm/ADT/StringRef.h"
#include <sstream>

using namespace loom::visualization;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

bool DotStyleConfig::isArithOp(llvm::StringRef opName) {
  return opName.starts_with("arith.") && opName != "arith.constant";
}

bool DotStyleConfig::isMathOp(llvm::StringRef opName) {
  return opName.starts_with("math.");
}

bool DotStyleConfig::isLLVMIntrinsic(llvm::StringRef opName) {
  return opName.starts_with("llvm.intr.");
}

bool DotStyleConfig::isUBOp(llvm::StringRef opName) {
  return opName == "ub.poison";
}

bool DotStyleConfig::isHandshakeMemoryOp(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory" ||
         opName == "handshake.load" || opName == "handshake.store";
}

bool DotStyleConfig::isHandshakeFork(llvm::StringRef opName) {
  return opName == "handshake.fork" || opName == "fork";
}

bool DotStyleConfig::isHandshakeMux(llvm::StringRef opName) {
  return opName == "handshake.mux" || opName == "mux";
}

//===----------------------------------------------------------------------===//
// Style Implementation (Per Specification)
//===----------------------------------------------------------------------===//

DotNodeStyle DotStyleConfig::getHandshakeMemoryStyle(llvm::StringRef opName) {
  if (opName == "handshake.extmemory") {
    return DotNodeStyle("hexagon", "gold", "black", "solid", 1.0);
  } else if (opName == "handshake.memory") {
    return DotNodeStyle("doubleoctagon", "orange", "black", "solid", 1.0);
  } else if (opName == "handshake.load") {
    return DotNodeStyle("cylinder", "skyblue", "black", "solid", 1.0);
  } else if (opName == "handshake.store") {
    return DotNodeStyle("box3d", "salmon", "black", "solid", 1.0);
  }

  return getFallbackStyle();
}

DotNodeStyle DotStyleConfig::getNodeStyle(llvm::StringRef opName, bool isMapped,
                                          bool isFuncArg, bool isReturn) {
  // Special nodes: Function interface
  if (isFuncArg) {
    return DotNodeStyle("invhouse", "lightpink", "black", "solid", 1.0);
  }

  if (isReturn) {
    return DotNodeStyle("house", "lightcoral", "black", "solid", 1.0);
  }

  // Handshake.fork: small circle + light blue with label "f:<ID>"
  if (isHandshakeFork(opName)) {
    return DotNodeStyle("circle", "lightblue", "black", "solid", 1.0,
                        "width=0.7,height=0.7,fixedsize=true");
  }

  // Handshake.mux: inverted triangle + light cyan
  if (isHandshakeMux(opName)) {
    return DotNodeStyle("invtriangle", "lightcyan", "black", "solid", 1.0);
  }

  // Handshake memory operations: unique shapes and colors
  if (isHandshakeMemoryOp(opName)) {
    DotNodeStyle style = getHandshakeMemoryStyle(opName);
    if (!isMapped) {
      style.fillColor = "white";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // Other handshake operations (excluding fork, mux, memory ops)
  if (opName.starts_with("handshake.")) {
    DotNodeStyle style("ellipse", "lightblue", "black", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // All arith.* ops: Msquare + dark green
  if (isArithOp(opName)) {
    DotNodeStyle style("Msquare", "darkgreen", "white", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.fontColor = "black";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // All math.* ops: trapezoid + dark purple
  if (isMathOp(opName)) {
    DotNodeStyle style("trapezium", "purple4", "white", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.fontColor = "black";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // LLVM intrinsics: square + magenta
  if (isLLVMIntrinsic(opName)) {
    DotNodeStyle style("square", "magenta", "black", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // ub.poison: triangle + pure red
  if (isUBOp(opName)) {
    DotNodeStyle style("triangle", "red", "white", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.fontColor = "black";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // Constants: octagon + wheat
  if (opName == "arith.constant" || opName == "handshake.constant") {
    DotNodeStyle style("octagon", "wheat", "black", "solid", 1.0);
    if (!isMapped) {
      style.fillColor = "white";
      style.borderStyle = "dashed";
    }
    return style;
  }

  // Fallback for unhandled ops: five-pointed star + pure red
  DotNodeStyle style = getFallbackStyle();
  if (!isMapped) {
    style.fillColor = "white";
    style.borderStyle = "dashed";
  }
  return style;
}

DotNodeStyle DotStyleConfig::getFallbackStyle() {
  return DotNodeStyle("star", "red", "white", "solid", 2.0);
}

DotEdgeStyle DotStyleConfig::getEdgeStyle(bool isMapped, bool showPortLabel) {
  if (isMapped) {
    return DotEdgeStyle("black", "solid", 2.5, showPortLabel);
  } else {
    return DotEdgeStyle("gray", "dashed", 1.0, false);
  }
}

std::string DotStyleConfig::formatPortLabel(int srcPort, int dstPort) {
  std::ostringstream os;
  os << "port" << srcPort << " -> port" << dstPort;
  return os.str();
}
