#include "fcc/Viz/VizExporter.h"

#include "VizAssets.h"

#include "fcc/Dialect/Fabric/FabricDialect.h"
#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fcc {

// ---- helpers ----

static std::string jsonEsc(llvm::StringRef s) {
  std::string r;
  r.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"') r += "\\\"";
    else if (c == '\\') r += "\\\\";
    else if (c == '\n') r += "\\n";
    else r += c;
  }
  return r;
}

static std::string htmlEsc(llvm::StringRef s) {
  std::string r;
  for (char c : s) {
    if (c == '<') r += "&lt;";
    else if (c == '>') r += "&gt;";
    else if (c == '&') r += "&amp;";
    else if (c == '"') r += "&quot;";
    else r += c;
  }
  return r;
}

static std::string scriptSafe(const std::string &s) {
  std::string r;
  r.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '/') {
      r += "<\\/";
      ++i;
    } else {
      r += s[i];
    }
  }
  return r;
}

static std::string printType(mlir::Type type) {
  if (!type) return "";
  std::string s;
  llvm::raw_string_ostream os(s);
  type.print(os);
  return s;
}

static std::string dfgEdgeType(mlir::Type type) {
  if (!type) return "data";
  if (mlir::isa<mlir::MemRefType>(type)) return "memref";
  if (mlir::isa<mlir::NoneType>(type)) return "control";
  return "data";
}

static std::string dfgOperandName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getOperandName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getOperandName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getOperandName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getOperandName(idx);
  if (auto mux = mlir::dyn_cast<circt::handshake::MuxOp>(op))
    return mux.getOperandName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getOperandName(idx);
  if (auto constant = mlir::dyn_cast<circt::handshake::ConstantOp>(op))
    return constant.getOperandName(idx);
  return ("I" + std::to_string(idx));
}

static std::string dfgResultName(mlir::Operation *op, unsigned idx) {
  if (auto load = mlir::dyn_cast<circt::handshake::LoadOp>(op))
    return load.getResultName(idx);
  if (auto store = mlir::dyn_cast<circt::handshake::StoreOp>(op))
    return store.getResultName(idx);
  if (auto memory = mlir::dyn_cast<circt::handshake::MemoryOp>(op))
    return memory.getResultName(idx);
  if (auto ext = mlir::dyn_cast<circt::handshake::ExternalMemoryOp>(op))
    return ext.getResultName(idx);
  if (auto cbr = mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op))
    return cbr.getResultName(idx);
  if (auto ctrlMerge =
          mlir::dyn_cast<circt::handshake::ControlMergeOp>(op))
    return ctrlMerge.getResultName(idx);
  return ("O" + std::to_string(idx));
}

static std::string getADGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });
  if (!fabricMod) return "";
  return fabricMod.getSymName().str();
}

static std::string getDFGName(mlir::ModuleOp topModule) {
  if (!topModule) return "";
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });
  if (!funcOp) return "";
  return funcOp.getName().str();
}

static std::string makeVizTitle(mlir::ModuleOp adgModule,
                                mlir::ModuleOp dfgModule,
                                bool hasMapping) {
  std::string adgName = getADGName(adgModule);
  std::string dfgName = getDFGName(dfgModule);
  if (!dfgName.empty() && !adgName.empty())
    return dfgName + (hasMapping ? " on " : " and ") + adgName;
  if (!dfgName.empty()) return dfgName;
  if (!adgName.empty()) return adgName;
  return hasMapping ? "fcc viz (mapped)" : "fcc viz";
}

static std::string loadVizSidecarJson(mlir::ModuleOp adgModule,
                                      llvm::StringRef adgSourcePath) {
  if (!adgModule)
    return "null";

  fcc::fabric::ModuleOp fabricMod;
  adgModule->walk([&](fcc::fabric::ModuleOp mod) {
    if (!fabricMod)
      fabricMod = mod;
  });
  if (!fabricMod)
    return "null";

  auto pathAttr = fabricMod->getAttrOfType<mlir::StringAttr>("viz_file");
  if (!pathAttr)
    return "null";

  llvm::SmallString<256> resolved(pathAttr.getValue());
  if (!llvm::sys::path::is_absolute(resolved) && !adgSourcePath.empty()) {
    llvm::SmallString<256> baseDir(adgSourcePath);
    llvm::sys::path::remove_filename(baseDir);
    llvm::sys::path::append(baseDir, resolved);
    resolved = baseDir;
  }

  auto buffer = llvm::MemoryBuffer::getFile(resolved);
  if (!buffer) {
    llvm::errs() << "fcc viz: warning: cannot open ADG viz sidecar "
                 << resolved << "\n";
    return "null";
  }
  return (*buffer)->getBuffer().str();
}

struct AutoLayoutComponent {
  std::string name;
  std::string kind;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;
  unsigned fuCount = 0;
  double width = 160.0;
  double height = 80.0;
};

static unsigned getJsonUnsigned(const llvm::json::Object &obj,
                                llvm::StringRef key,
                                unsigned defaultValue = 0) {
  if (auto v = obj.getInteger(key); v && *v >= 0)
    return static_cast<unsigned>(*v);
  return defaultValue;
}

static AutoLayoutComponent
parseAutoLayoutComponent(const llvm::json::Object &obj) {
  AutoLayoutComponent comp;
  if (auto name = obj.getString("name"))
    comp.name = name->str();
  if (auto kind = obj.getString("kind"))
    comp.kind = kind->str();
  comp.numInputs = getJsonUnsigned(obj, "numInputs");
  comp.numOutputs = getJsonUnsigned(obj, "numOutputs");
  if (auto fus = obj.getArray("fus"))
    comp.fuCount = static_cast<unsigned>(fus->size());

  auto estimatePEBox = [&](unsigned fuCount) {
    unsigned safeCount = std::max(1U, fuCount);
    unsigned cols =
        std::max(1U, static_cast<unsigned>(std::ceil(std::sqrt(safeCount))));
    unsigned rows = (safeCount + cols - 1) / cols;
    constexpr double kFuW = 140.0;
    constexpr double kFuH = 108.0;
    constexpr double kGap = 12.0;
    constexpr double kPadX = 76.0;
    constexpr double kPadY = 84.0;
    comp.width = std::max(200.0,
                          static_cast<double>(cols) * kFuW +
                              static_cast<double>(std::max(0U, cols - 1)) * kGap +
                              kPadX);
    comp.height = std::max(200.0,
                           static_cast<double>(rows) * kFuH +
                               static_cast<double>(std::max(0U, rows - 1)) * kGap +
                               kPadY);
  };

  auto estimateSwitchBox = [&](unsigned numInputs, unsigned numOutputs) {
    constexpr double kSwitchPortPitch = 24.0;
    constexpr double kSwitchMinSide = 84.0;
    unsigned maxSideSlots =
        std::max((numInputs + 1) / 2, (numOutputs + 1) / 2);
    double side = std::max(
        kSwitchMinSide,
        32.0 + (static_cast<double>(std::max(1U, maxSideSlots)) + 1.0) *
                   kSwitchPortPitch);
    comp.width = side;
    comp.height = side;
  };

  if (comp.kind == "spatial_pe" || comp.kind == "temporal_pe") {
    estimatePEBox(comp.fuCount);
  } else if (comp.kind == "spatial_sw" || comp.kind == "temporal_sw") {
    estimateSwitchBox(comp.numInputs, comp.numOutputs);
  } else if (comp.kind == "memory") {
    comp.width = 170.0;
    comp.height = 80.0;
  } else if (comp.kind == "fifo") {
    comp.width = 100.0;
    comp.height = 56.0;
  } else if (comp.kind == "add_tag" || comp.kind == "map_tag" ||
             comp.kind == "del_tag") {
    comp.width = 92.0;
    comp.height = 52.0;
  } else {
    comp.width = 160.0;
    comp.height = 80.0;
  }

  return comp;
}

struct AutoLayoutConnection {
  std::string from;
  unsigned fromIdx = 0;
  std::string to;
  unsigned toIdx = 0;
};

struct AutoLayoutPlacement {
  double centerX = 0.0;
  double centerY = 0.0;
  bool valid = false;
};

struct AutoLayoutRoutePt {
  double x = 0.0;
  double y = 0.0;
};

struct AutoLayoutModuleBounds {
  double x = 0.0;
  double y = 0.0;
  double w = 0.0;
  double h = 0.0;
  bool valid = false;
};

static double computeModuleMarginForAreaRatio(double contentW, double contentH,
                                              double areaRatio) {
  if (contentW <= 0.0 || contentH <= 0.0)
    return 24.0;
  double perimeterSum = contentW + contentH;
  double targetExtraArea = std::max(0.0, areaRatio) * contentW * contentH;
  double disc = perimeterSum * perimeterSum + 4.0 * targetExtraArea;
  double margin = (std::sqrt(std::max(0.0, disc)) - perimeterSum) / 4.0;
  return std::max(24.0, std::round(margin));
}

static std::array<unsigned, 4> buildAutoLayoutPortSideCounts(unsigned count,
                                                             unsigned sideCount) {
  std::array<unsigned, 4> counts = {0, 0, 0, 0};
  for (unsigned idx = 0; idx < count; ++idx)
    counts[idx % sideCount] += 1;
  return counts;
}

static std::array<unsigned, 2>
buildAutoLayoutPEPortSideCounts(unsigned count) {
  std::array<unsigned, 2> counts = {0, 0};
  for (unsigned idx = 0; idx < count; ++idx)
    counts[idx % 2] += 1;
  return counts;
}

static AutoLayoutRoutePt
computeAutoLayoutInputPortPos(const AutoLayoutComponent &comp,
                              const AutoLayoutPlacement &placement,
                              unsigned portIdx) {
  AutoLayoutRoutePt pt;
  if (comp.kind == "spatial_pe" || comp.kind == "temporal_pe") {
    std::array<unsigned, 2> sideCounts =
        buildAutoLayoutPEPortSideCounts(comp.numInputs);
    unsigned sideIdx = portIdx % 2;
    unsigned localIdx = portIdx / 2;
    double ratio = static_cast<double>(localIdx + 1) /
                   static_cast<double>(sideCounts[sideIdx] + 1);
    if (sideIdx == 0) {
      pt.x = (placement.centerX - comp.width / 2.0) + comp.width * ratio;
      pt.y = placement.centerY - comp.height / 2.0;
    } else {
      pt.x = placement.centerX - comp.width / 2.0;
      pt.y = (placement.centerY - comp.height / 2.0) + comp.height * ratio;
    }
    return pt;
  }

  if (comp.kind == "spatial_sw" || comp.kind == "temporal_sw") {
    std::array<unsigned, 4> sideCounts =
        buildAutoLayoutPortSideCounts(comp.numInputs, 2);
    unsigned sideIdx = portIdx % 2;
    unsigned localIdx = portIdx / 2;
    unsigned slotCount = sideCounts[sideIdx];
    double ratio = static_cast<double>(localIdx + 1) /
                   static_cast<double>(slotCount + 1);
    if (sideIdx == 0) {
      pt.x = (placement.centerX - comp.width / 2.0) + comp.width * ratio;
      pt.y = placement.centerY - comp.height / 2.0;
    } else {
      pt.x = placement.centerX - comp.width / 2.0;
      pt.y = (placement.centerY - comp.height / 2.0) + comp.height * ratio;
    }
    return pt;
  }

  pt.x = placement.centerX - comp.width / 2.0;
  pt.y = placement.centerY - comp.height / 2.0 + 16.0 +
         (comp.height - 32.0) *
             (static_cast<double>(portIdx + 1) /
              static_cast<double>(comp.numInputs + 1));
  return pt;
}

static AutoLayoutRoutePt
computeAutoLayoutOutputPortPos(const AutoLayoutComponent &comp,
                               const AutoLayoutPlacement &placement,
                               unsigned portIdx) {
  AutoLayoutRoutePt pt;
  if (comp.kind == "spatial_pe" || comp.kind == "temporal_pe") {
    std::array<unsigned, 2> sideCounts =
        buildAutoLayoutPEPortSideCounts(comp.numOutputs);
    unsigned sideIdx = portIdx % 2;
    unsigned localIdx = portIdx / 2;
    double ratio = static_cast<double>(localIdx + 1) /
                   static_cast<double>(sideCounts[sideIdx] + 1);
    if (sideIdx == 0) {
      pt.x = placement.centerX + comp.width / 2.0;
      pt.y = (placement.centerY - comp.height / 2.0) + comp.height * ratio;
    } else {
      pt.x = (placement.centerX - comp.width / 2.0) + comp.width * ratio;
      pt.y = placement.centerY + comp.height / 2.0;
    }
    return pt;
  }

  if (comp.kind == "spatial_sw" || comp.kind == "temporal_sw") {
    std::array<unsigned, 4> sideCounts =
        buildAutoLayoutPortSideCounts(comp.numOutputs, 2);
    unsigned sideIdx = portIdx % 2;
    unsigned localIdx = portIdx / 2;
    unsigned slotCount = sideCounts[sideIdx];
    double ratio = static_cast<double>(localIdx + 1) /
                   static_cast<double>(slotCount + 1);
    if (sideIdx == 0) {
      pt.x = placement.centerX + comp.width / 2.0;
      pt.y = (placement.centerY - comp.height / 2.0) + comp.height * ratio;
    } else {
      pt.x = (placement.centerX - comp.width / 2.0) + comp.width * ratio;
      pt.y = placement.centerY + comp.height / 2.0;
    }
    return pt;
  }

  pt.x = placement.centerX + comp.width / 2.0;
  pt.y = placement.centerY - comp.height / 2.0 + 16.0 +
         (comp.height - 32.0) *
             (static_cast<double>(portIdx + 1) /
              static_cast<double>(comp.numOutputs + 1));
  return pt;
}

static AutoLayoutModuleBounds
computeAutoLayoutModuleBounds(const std::vector<AutoLayoutComponent> &components,
                              const std::vector<AutoLayoutPlacement> &placements) {
  AutoLayoutModuleBounds bounds;
  bool haveContent = false;
  double actualMinX = 0.0;
  double actualMinY = 0.0;
  double actualMaxX = 0.0;
  double actualMaxY = 0.0;
  for (size_t idx = 0; idx < components.size(); ++idx) {
    if (idx >= placements.size() || !placements[idx].valid)
      continue;
    double boxMinX = placements[idx].centerX - components[idx].width / 2.0;
    double boxMinY = placements[idx].centerY - components[idx].height / 2.0;
    double boxMaxX = placements[idx].centerX + components[idx].width / 2.0;
    double boxMaxY = placements[idx].centerY + components[idx].height / 2.0;
    if (!haveContent) {
      actualMinX = boxMinX;
      actualMinY = boxMinY;
      actualMaxX = boxMaxX;
      actualMaxY = boxMaxY;
      haveContent = true;
    } else {
      actualMinX = std::min(actualMinX, boxMinX);
      actualMinY = std::min(actualMinY, boxMinY);
      actualMaxX = std::max(actualMaxX, boxMaxX);
      actualMaxY = std::max(actualMaxY, boxMaxY);
    }
  }
  if (!haveContent)
    return bounds;

  double contentW = actualMaxX - actualMinX;
  double contentH = actualMaxY - actualMinY;
  double margin = computeModuleMarginForAreaRatio(contentW, contentH, 0.20);
  bounds.x = actualMinX - margin;
  bounds.y = actualMinY - margin;
  bounds.w = contentW + margin * 2.0;
  bounds.h = contentH + margin * 2.0;
  bounds.valid = true;
  return bounds;
}

static AutoLayoutRoutePt
computeAutoLayoutModuleInputPortPos(const AutoLayoutModuleBounds &bounds,
                                    unsigned numModuleInputs,
                                    unsigned portIdx) {
  AutoLayoutRoutePt pt;
  pt.x = bounds.x + bounds.w *
                    (static_cast<double>(portIdx + 1) /
                     static_cast<double>(numModuleInputs + 1));
  pt.y = bounds.y;
  return pt;
}

static AutoLayoutRoutePt
computeAutoLayoutModuleOutputPortPos(const AutoLayoutModuleBounds &bounds,
                                     unsigned numModuleOutputs,
                                     unsigned portIdx) {
  AutoLayoutRoutePt pt;
  pt.x = bounds.x + bounds.w *
                    (static_cast<double>(portIdx + 1) /
                     static_cast<double>(numModuleOutputs + 1));
  pt.y = bounds.y + bounds.h;
  return pt;
}

static std::vector<AutoLayoutRoutePt>
routeAutoLayoutModuleInputConnection(
    const AutoLayoutModuleBounds &bounds, unsigned numModuleInputs,
    unsigned scalarIdx, const AutoLayoutComponent &dstComp,
    const AutoLayoutPlacement &dstPlacement, unsigned dstPortIdx) {
  if (!bounds.valid || !dstPlacement.valid)
    return {};
  AutoLayoutRoutePt srcPort =
      computeAutoLayoutModuleInputPortPos(bounds, numModuleInputs, scalarIdx);
  AutoLayoutRoutePt dstPort =
      computeAutoLayoutInputPortPos(dstComp, dstPlacement, dstPortIdx);
  int signedLane = static_cast<int>(scalarIdx % 5) - 2;
  double laneOffset = static_cast<double>(signedLane) * 7.0;
  double entryY = bounds.y + 18.0 + std::abs(laneOffset);
  double dstApproachX = dstPort.x - (24.0 + std::abs(laneOffset));
  std::vector<AutoLayoutRoutePt> pts;
  pts.push_back({srcPort.x, entryY});
  if (std::abs(srcPort.x - dstApproachX) > 0.5)
    pts.push_back({dstApproachX, entryY});
  if (std::abs(entryY - dstPort.y) > 0.5)
    pts.push_back({dstApproachX, dstPort.y});
  return pts;
}

static std::vector<AutoLayoutRoutePt>
routeAutoLayoutModuleOutputConnection(
    const AutoLayoutModuleBounds &bounds, unsigned numModuleOutputs,
    const AutoLayoutComponent &srcComp, const AutoLayoutPlacement &srcPlacement,
    unsigned srcPortIdx, unsigned scalarOutIdx) {
  if (!bounds.valid || !srcPlacement.valid)
    return {};
  AutoLayoutRoutePt srcPort =
      computeAutoLayoutOutputPortPos(srcComp, srcPlacement, srcPortIdx);
  AutoLayoutRoutePt dstPort =
      computeAutoLayoutModuleOutputPortPos(bounds, numModuleOutputs, scalarOutIdx);
  int signedLane = static_cast<int>(scalarOutIdx % 5) - 2;
  double laneOffset = static_cast<double>(signedLane) * 7.0;
  double exitX = srcPort.x + (24.0 + std::abs(laneOffset));
  double corridorY =
      bounds.y + bounds.h - 18.0 - std::abs(laneOffset);
  std::vector<AutoLayoutRoutePt> pts;
  pts.push_back({exitX, srcPort.y});
  if (std::abs(srcPort.y - corridorY) > 0.5)
    pts.push_back({exitX, corridorY});
  if (std::abs(exitX - dstPort.x) > 0.5)
    pts.push_back({dstPort.x, corridorY});
  return pts;
}

static std::vector<AutoLayoutRoutePt>
routeAutoLayoutConnection(const AutoLayoutComponent &srcComp,
                          const AutoLayoutPlacement &srcPlacement,
                          unsigned srcPortIdx,
                          const AutoLayoutComponent &dstComp,
                          const AutoLayoutPlacement &dstPlacement,
                          unsigned dstPortIdx, unsigned routeOrdinal) {
  if (!srcPlacement.valid || !dstPlacement.valid)
    return {};

  AutoLayoutRoutePt srcPort =
      computeAutoLayoutOutputPortPos(srcComp, srcPlacement, srcPortIdx);
  AutoLayoutRoutePt dstPort =
      computeAutoLayoutInputPortPos(dstComp, dstPlacement, dstPortIdx);

  int signedLane = static_cast<int>(routeOrdinal % 5) - 2;
  double laneOffset = static_cast<double>(signedLane) * 7.0;
  double margin = 22.0 + std::abs(laneOffset) * 0.5;
  double srcRight = srcPlacement.centerX + srcComp.width / 2.0;
  double dstLeft = dstPlacement.centerX - dstComp.width / 2.0;
  double srcTop = srcPlacement.centerY - srcComp.height / 2.0;
  double srcBottom = srcPlacement.centerY + srcComp.height / 2.0;
  double dstTop = dstPlacement.centerY - dstComp.height / 2.0;
  double dstBottom = dstPlacement.centerY + dstComp.height / 2.0;

  std::vector<AutoLayoutRoutePt> pts;
  if (srcPlacement.centerX + 1.0 < dstPlacement.centerX) {
    double corridorX = (srcRight + dstLeft) / 2.0 + laneOffset;
    pts.push_back({corridorX, srcPort.y});
    if (std::abs(srcPort.y - dstPort.y) > 0.5)
      pts.push_back({corridorX, dstPort.y});
    return pts;
  }

  double srcExitX = srcRight + margin;
  double dstEntryX = dstLeft - margin;
  bool routeAbove = srcPlacement.centerY <= dstPlacement.centerY;
  double corridorY =
      routeAbove ? std::min(srcTop, dstTop) - margin - std::abs(laneOffset)
                 : std::max(srcBottom, dstBottom) + margin + std::abs(laneOffset);
  pts.push_back({srcExitX, srcPort.y});
  pts.push_back({srcExitX, corridorY});
  pts.push_back({dstEntryX, corridorY});
  pts.push_back({dstEntryX, dstPort.y});
  return pts;
}

static std::string buildAutoLayoutJson(
    const std::vector<AutoLayoutComponent> &components,
    const std::vector<AutoLayoutConnection> &connections,
    unsigned numModuleInputs, unsigned numModuleOutputs,
    const std::vector<AutoLayoutPlacement> &placements) {
  AutoLayoutModuleBounds moduleBounds =
      computeAutoLayoutModuleBounds(components, placements);
  std::map<std::string, unsigned> componentIndex;
  for (size_t idx = 0; idx < components.size(); ++idx)
    componentIndex[components[idx].name] = static_cast<unsigned>(idx);

  std::ostringstream os;
  os << "{\n  \"version\": 1,\n  \"components\": [\n";
  for (size_t idx = 0; idx < components.size(); ++idx) {
    if (idx > 0)
      os << ",\n";
    double centerX = placements[idx].valid
                         ? placements[idx].centerX
                         : 120.0 + components[idx].width / 2.0;
    double centerY = placements[idx].valid
                         ? placements[idx].centerY
                         : 120.0 + components[idx].height / 2.0;
    os << "    {\"name\": \"" << jsonEsc(components[idx].name)
       << "\", \"kind\": \"" << jsonEsc(components[idx].kind)
       << "\", \"center_x\": " << centerX
       << ", \"center_y\": " << centerY << "}";
  }
  os << "\n  ],\n  \"routes\": [\n";

  bool firstRoute = true;
  std::map<std::pair<unsigned, unsigned>, unsigned> nextPairOrdinal;
  auto emitRouteRecord =
      [&](llvm::StringRef fromName, unsigned fromPort, llvm::StringRef toName,
          unsigned toPort, const std::vector<AutoLayoutRoutePt> &pts) {
        if (!firstRoute)
          os << ",\n";
        firstRoute = false;
        os << "    {\"from\": \"" << jsonEsc(fromName) << "\""
           << ", \"from_port\": " << fromPort << ", \"to\": \""
           << jsonEsc(toName) << "\", \"to_port\": " << toPort
           << ", \"points\": [";
        for (size_t ptIdx = 0; ptIdx < pts.size(); ++ptIdx) {
          if (ptIdx > 0)
            os << ", ";
          os << "{\"x\": " << pts[ptIdx].x << ", \"y\": " << pts[ptIdx].y
             << "}";
        }
        os << "]}";
      };

  for (const auto &conn : connections) {
    if (conn.from == "module_in") {
      auto dstIt = componentIndex.find(conn.to);
      if (dstIt == componentIndex.end())
        continue;
      std::vector<AutoLayoutRoutePt> pts = routeAutoLayoutModuleInputConnection(
          moduleBounds, numModuleInputs, conn.fromIdx, components[dstIt->second],
          placements[dstIt->second], conn.toIdx);
      emitRouteRecord("module_in", conn.fromIdx, conn.to, conn.toIdx, pts);
      continue;
    }
    if (conn.to == "module_out") {
      auto srcIt = componentIndex.find(conn.from);
      if (srcIt == componentIndex.end())
        continue;
      std::vector<AutoLayoutRoutePt> pts = routeAutoLayoutModuleOutputConnection(
          moduleBounds, numModuleOutputs, components[srcIt->second],
          placements[srcIt->second], conn.fromIdx, conn.toIdx);
      emitRouteRecord(conn.from, conn.fromIdx, "module_out", conn.toIdx, pts);
      continue;
    }

    auto srcIt = componentIndex.find(conn.from);
    auto dstIt = componentIndex.find(conn.to);
    if (srcIt == componentIndex.end() || dstIt == componentIndex.end())
      continue;
    auto pairKey =
        std::make_pair(std::min(srcIt->second, dstIt->second),
                       std::max(srcIt->second, dstIt->second));
    unsigned pairOrdinal = nextPairOrdinal[pairKey]++;
    std::vector<AutoLayoutRoutePt> pts = routeAutoLayoutConnection(
        components[srcIt->second], placements[srcIt->second], conn.fromIdx,
        components[dstIt->second], placements[dstIt->second], conn.toIdx,
        pairOrdinal);
    emitRouteRecord(conn.from, conn.fromIdx, conn.to, conn.toIdx, pts);
  }

  os << "\n  ]\n}\n";
  return os.str();
}

static std::string
buildNeatoLayoutJsonFromADGJson(llvm::StringRef adgJsonText) {
  auto parsed = llvm::json::parse(adgJsonText);
  if (!parsed)
    return "null";
  auto *root = parsed->getAsObject();
  if (!root)
    return "null";
  auto *componentsJson = root->getArray("components");
  if (!componentsJson)
    return "null";
  unsigned numModuleInputs = getJsonUnsigned(*root, "numInputs");
  unsigned numModuleOutputs = getJsonUnsigned(*root, "numOutputs");

  std::vector<AutoLayoutComponent> components;
  components.reserve(componentsJson->size());
  std::map<std::string, unsigned> componentIndex;
  for (const auto &value : *componentsJson) {
    auto *obj = value.getAsObject();
    if (!obj)
      continue;
    AutoLayoutComponent comp = parseAutoLayoutComponent(*obj);
    if (comp.name.empty())
      continue;
    componentIndex[comp.name] = static_cast<unsigned>(components.size());
    components.push_back(std::move(comp));
  }
  if (components.empty())
    return "null";

  std::vector<AutoLayoutConnection> connections;
  std::map<std::pair<unsigned, unsigned>, unsigned> edgeMultiplicity;
  if (auto *connectionsJson = root->getArray("connections")) {
    for (const auto &value : *connectionsJson) {
      auto *obj = value.getAsObject();
      if (!obj)
        continue;
      auto from = obj->getString("from");
      auto to = obj->getString("to");
      if (!from || !to)
        continue;
      AutoLayoutConnection conn;
      conn.from = from->str();
      conn.to = to->str();
      conn.fromIdx = getJsonUnsigned(*obj, "fromIdx");
      conn.toIdx = getJsonUnsigned(*obj, "toIdx");
      connections.push_back(conn);
      if (*from == "module_in" || *to == "module_out")
        continue;
      auto fromIt = componentIndex.find(from->str());
      auto toIt = componentIndex.find(to->str());
      if (fromIt == componentIndex.end() || toIt == componentIndex.end() ||
          fromIt->second == toIt->second) {
        continue;
      }
      auto key =
          std::make_pair(std::min(fromIt->second, toIt->second),
                         std::max(fromIt->second, toIt->second));
      edgeMultiplicity[key] += 1;
    }
  }

  auto packRowMajor = [&]() {
    std::vector<AutoLayoutPlacement> placements(components.size());
    constexpr double kGapX = 88.0;
    constexpr double kGapY = 108.0;
    constexpr double kWrapWidth = 3600.0;
    double startX = 120.0;
    double cursorX = startX;
    double cursorY = 120.0;
    double rowHeight = 0.0;
    for (size_t idx = 0; idx < components.size(); ++idx) {
      const auto &comp = components[idx];
      if (cursorX > startX && cursorX + comp.width > startX + kWrapWidth) {
        cursorX = startX;
        cursorY += rowHeight + kGapY;
        rowHeight = 0.0;
      }
      placements[idx] = {cursorX + comp.width / 2.0,
                         cursorY + comp.height / 2.0, true};
      cursorX += comp.width + kGapX;
      rowHeight = std::max(rowHeight, comp.height);
    }
    return buildAutoLayoutJson(components, connections, numModuleInputs,
                               numModuleOutputs, placements);
  };

  if (components.size() == 1)
    return packRowMajor();

  auto neatoPath = llvm::sys::findProgramByName("neato");
  if (!neatoPath)
    return packRowMajor();

  llvm::SmallString<128> dotPath;
  llvm::SmallString<128> plainPath;
  int dotFd = -1;
  int plainFd = -1;
  if (llvm::sys::fs::createTemporaryFile("fcc_viz_layout", "dot", dotFd,
                                         dotPath))
    return packRowMajor();
  if (llvm::sys::fs::createTemporaryFile("fcc_viz_layout", "plain", plainFd,
                                         plainPath)) {
    llvm::sys::fs::remove(dotPath);
    return packRowMajor();
  }

  auto cleanupTempFiles = [&]() {
    llvm::sys::fs::remove(dotPath);
    llvm::sys::fs::remove(plainPath);
  };

  {
    llvm::raw_fd_ostream dotOS(dotFd, true);
    auto dotNum = [&](double value) {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(6) << value;
      return ss.str();
    };
    constexpr double kGraphvizUnit = 72.0;
    constexpr double kNodePad = 72.0;
    dotOS << "graph G {\n";
    dotOS << "  graph [layout=neato, overlap=false, sep=\"+48\", "
             "esep=\"+24\", splines=false, notranslate=true];\n";
    dotOS << "  node [shape=box, fixedsize=true, margin=0, label=\"\"];\n";
    dotOS << "  edge [len=2.4];\n";
    for (size_t idx = 0; idx < components.size(); ++idx) {
      double widthIn =
          std::max(0.25, (components[idx].width + kNodePad) / kGraphvizUnit);
      double heightIn =
          std::max(0.25, (components[idx].height + kNodePad) / kGraphvizUnit);
      dotOS << "  n" << idx << " [width=" << dotNum(widthIn)
            << ", height=" << dotNum(heightIn) << "];\n";
    }
    for (const auto &[key, multiplicity] : edgeMultiplicity) {
      unsigned weight = std::min(8U, multiplicity);
      dotOS << "  n" << key.first << " -- n" << key.second
            << " [weight=" << weight << "];\n";
    }
    dotOS << "}\n";
  }
  {
    llvm::raw_fd_ostream plainOS(plainFd, true);
    plainOS.flush();
  }

  std::string errMsg;
  bool execFailed = false;
  std::string neatoPathStr = *neatoPath;
  std::string dotPathStr = dotPath.str().str();
  std::string plainPathStr = plainPath.str().str();
  llvm::SmallVector<llvm::StringRef, 5> args = {
      llvm::StringRef(neatoPathStr), llvm::StringRef("-Tplain"),
      llvm::StringRef(dotPathStr), llvm::StringRef("-o"),
      llvm::StringRef(plainPathStr)};
  int rc = llvm::sys::ExecuteAndWait(neatoPathStr, args, std::nullopt, {}, 0,
                                     0, &errMsg, &execFailed);
  if (rc != 0 || execFailed) {
    cleanupTempFiles();
    return packRowMajor();
  }

  auto plainBuf = llvm::MemoryBuffer::getFile(plainPath);
  if (!plainBuf) {
    cleanupTempFiles();
    return packRowMajor();
  }

  std::vector<AutoLayoutPlacement> placements(components.size());
  std::istringstream plainStream((*plainBuf)->getBuffer().str());
  std::string lineStr;
  while (std::getline(plainStream, lineStr)) {
    llvm::StringRef line(lineStr);
    line = line.trim();
    if (line.empty())
      continue;
    std::istringstream lineStream(line.str());
    std::string kind;
    lineStream >> kind;
    if (kind != "node")
      continue;
    std::string nodeName;
    double xIn = 0.0;
    double yIn = 0.0;
    double widthIn = 0.0;
    double heightIn = 0.0;
    lineStream >> nodeName >> xIn >> yIn >> widthIn >> heightIn;
    if (nodeName.size() <= 1 || nodeName[0] != 'n')
      continue;
    unsigned idx = 0;
    try {
      idx = static_cast<unsigned>(std::stoul(nodeName.substr(1)));
    } catch (...) {
      continue;
    }
    if (idx >= placements.size())
      continue;
    constexpr double kGraphvizUnit = 72.0;
    placements[idx] = {xIn * kGraphvizUnit, yIn * kGraphvizUnit, true};
  }
  cleanupTempFiles();

  bool havePlacement = false;
  double minLeft = 0.0;
  double minTop = 0.0;
  for (size_t idx = 0; idx < components.size(); ++idx) {
    if (!placements[idx].valid)
      continue;
    double left = placements[idx].centerX - components[idx].width / 2.0;
    double top = placements[idx].centerY - components[idx].height / 2.0;
    if (!havePlacement) {
      minLeft = left;
      minTop = top;
      havePlacement = true;
    } else {
      minLeft = std::min(minLeft, left);
      minTop = std::min(minTop, top);
    }
  }
  if (!havePlacement)
    return packRowMajor();

  double shiftX = 120.0 - minLeft;
  double shiftY = 120.0 - minTop;

  for (size_t idx = 0; idx < components.size(); ++idx) {
    if (placements[idx].valid) {
      placements[idx].centerX += shiftX;
      placements[idx].centerY += shiftY;
    } else {
      placements[idx] = {120.0 + components[idx].width / 2.0,
                         120.0 + components[idx].height / 2.0, true};
    }
  }

  return buildAutoLayoutJson(components, connections, numModuleInputs,
                             numModuleOutputs, placements);
}

static std::string selectVizLayoutJson(mlir::ModuleOp adgModule,
                                       llvm::StringRef adgSourcePath,
                                       llvm::StringRef adgJson,
                                       VizLayoutMode layoutMode) {
  if (!adgModule)
    return "null";

  if (layoutMode == VizLayoutMode::Default) {
    std::string sidecar = loadVizSidecarJson(adgModule, adgSourcePath);
    if (sidecar != "null")
      return sidecar;
  }

  std::string autoLayout = buildNeatoLayoutJsonFromADGJson(adgJson);
  if (autoLayout != "null")
    return autoLayout;

  return loadVizSidecarJson(adgModule, adgSourcePath);
}

// ---- Serialize fabric.module to JSON ----

static void writeADGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule,
                          mlir::MLIRContext *ctx) {
  // Find fabric.module
  fcc::fabric::ModuleOp fabricMod;
  topModule->walk([&](fcc::fabric::ModuleOp mod) { fabricMod = mod; });

  os << "{\n";

  if (!fabricMod) {
    os << "  \"module\": null\n}";
    return;
  }

  // Module signature
  auto fnType = fabricMod.getFunctionType();
  os << "  \"name\": \"" << jsonEsc(fabricMod.getSymName().str())
     << "\",\n";
  os << "  \"numInputs\": " << fnType.getNumInputs()
     << ", \"numOutputs\": " << fnType.getNumResults() << ",\n";

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  topModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                   fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp,
                   fcc::fabric::ExtMemoryOp, fcc::fabric::MemoryOp>(op)) {
      return false;
    }
    return !op->hasAttr("inline_instantiation");
  };

  // Collect definition symbols from both top-level module and fabric.module.
  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefMap;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefMap;
  llvm::StringMap<fcc::fabric::SpatialSwOp> swDefMap;
  llvm::StringMap<fcc::fabric::TemporalSwOp> temporalSwDefMap;
  llvm::StringMap<fcc::fabric::ExtMemoryOp> extMemoryDefMap;
  llvm::StringMap<fcc::fabric::MemoryOp> memoryDefMap;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefMap;
  topModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto nameAttr = peOp.getSymNameAttr();
        nameAttr && isDefinitionOp(peOp.getOperation(), nameAttr.getValue()))
      peDefMap[nameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto nameAttr = peOp.getSymNameAttr();
        nameAttr && isDefinitionOp(peOp.getOperation(), nameAttr.getValue()))
      temporalPeDefMap[nameAttr.getValue()] = peOp;
  });
  topModule->walk([&](fcc::fabric::SpatialSwOp swOp) {
    if (auto nameAttr = swOp.getSymNameAttr();
        nameAttr && isDefinitionOp(swOp.getOperation(), nameAttr.getValue()))
      swDefMap[nameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::TemporalSwOp swOp) {
    if (auto nameAttr = swOp.getSymNameAttr();
        nameAttr && isDefinitionOp(swOp.getOperation(), nameAttr.getValue()))
      temporalSwDefMap[nameAttr.getValue()] = swOp;
  });
  topModule->walk([&](fcc::fabric::ExtMemoryOp extOp) {
    if (auto nameAttr = extOp.getSymNameAttr();
        nameAttr && isDefinitionOp(extOp.getOperation(), nameAttr.getValue()))
      extMemoryDefMap[nameAttr.getValue()] = extOp;
  });
  topModule->walk([&](fcc::fabric::MemoryOp memOp) {
    if (auto nameAttr = memOp.getSymNameAttr();
        nameAttr && isDefinitionOp(memOp.getOperation(), nameAttr.getValue()))
      memoryDefMap[nameAttr.getValue()] = memOp;
  });
  topModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefMap[symName] = fuOp;
  });

  llvm::DenseMap<mlir::Operation *, std::string> renderNameMap;
  unsigned extMemCount = 0;
  unsigned memoryCount = 0;
  unsigned temporalSwCount = 0;
  unsigned addTagCount = 0;
  unsigned delTagCount = 0;
  unsigned mapTagCount = 0;
  unsigned fifoCount = 0;

  auto getRenderName = [&](mlir::Operation *op) -> std::string {
    auto it = renderNameMap.find(op);
    if (it != renderNameMap.end())
      return it->second;

    std::string name;
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      name = instOp.getSymName().value_or("inst").str();
    } else if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      name = peOp.getSymName().value_or("pe").str();
    } else if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      name = swOp.getSymName().value_or("sw").str();
    } else if (auto tswOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (auto symAttr = tswOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "temporal_sw_" + std::to_string(temporalSwCount++);
    } else if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (auto symAttr = extOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "extmemory_" + std::to_string(extMemCount++);
    } else if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (auto symAttr = memOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "memory_" + std::to_string(memoryCount++);
    } else if (mlir::isa<fcc::fabric::AddTagOp>(op)) {
      name = "add_tag_" + std::to_string(addTagCount++);
    } else if (mlir::isa<fcc::fabric::DelTagOp>(op)) {
      name = "del_tag_" + std::to_string(delTagCount++);
    } else if (mlir::isa<fcc::fabric::MapTagOp>(op)) {
      name = "map_tag_" + std::to_string(mapTagCount++);
    } else if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (auto symAttr = fifoOp.getSymNameAttr())
        name = symAttr.getValue().str();
      else
        name = "fifo_" + std::to_string(fifoCount++);
    } else {
      name = op->getName().getStringRef().str();
    }

    renderNameMap[op] = name;
    return name;
  };

  // Helper: emit FU details for a PE definition.
  // Extracts full SSA connectivity with operand/result order:
  // inputEdges (arg->op), edges (op->op), outputEdges (op/arg->yield output).
  auto emitPEFUs = [&](auto peOp) {
    os << ", \"fus\": [";
    bool firstFU = true;
    auto &peBody = peOp.getBody().front();
    auto referencedIt = referencedTargetsByBlock.find(&peBody);
    const llvm::DenseSet<llvm::StringRef> *referencedTargets =
        referencedIt != referencedTargetsByBlock.end() ? &referencedIt->second
                                                       : nullptr;
    for (auto &innerOp : peBody.getOperations()) {
      fcc::fabric::FunctionUnitOp fuOp;
      std::string fuName;
      if (auto directFu = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(innerOp)) {
        llvm::StringRef symName = directFu.getSymNameAttr().getValue();
        if (!symName.empty() && referencedTargets &&
            referencedTargets->contains(symName))
          continue;
        fuOp = directFu;
        fuName = directFu.getSymName().str();
      } else if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(innerOp)) {
        auto fuIt = functionUnitDefMap.find(instOp.getModule());
        if (fuIt == functionUnitDefMap.end())
          continue;
        fuOp = fuIt->second;
        fuName = instOp.getSymName().value_or(instOp.getModule()).str();
      } else {
        continue;
      }

      if (!firstFU) os << ", ";
      firstFU = false;

      auto fuFnType = fuOp.getFunctionType();
      os << "{\"name\": \"" << jsonEsc(fuName) << "\"";
      os << ", \"numIn\": " << fuFnType.getNumInputs();
      os << ", \"numOut\": " << fuFnType.getNumResults();

      struct ValueRef {
        int owner = -1;      // block args are negative (-1 - argIdx), ops are 0+
        int resultIdx = 0;   // valid only when owner >= 0
      };
      struct InputEdgeRef {
        int argIdx = -1;
        int dstOp = -1;
        int dstOperand = -1;
      };
      struct DagEdgeRef {
        int srcOp = -1;
        int dstOp = -1;
        int srcResult = 0;
        int dstOperand = -1;
      };
      struct OutputEdgeRef {
        int srcOwner = -1;   // arg or op, same encoding as ValueRef::owner
        int yieldIdx = -1;
        int srcResult = 0;
      };

      llvm::DenseMap<mlir::Value, ValueRef> valToRef;
      for (auto arg : fuOp.getBody().front().getArguments())
        valToRef[arg] = {-1 - static_cast<int>(arg.getArgNumber()), 0};

      llvm::SmallVector<DagEdgeRef, 4> dagEdges;        // op -> op
      llvm::SmallVector<InputEdgeRef, 4> inputEdges;    // argIdx -> opIdx
      llvm::SmallVector<OutputEdgeRef, 4> outputEdges;  // valIdx -> yieldIdx

      os << ", \"ops\": [";
      bool firstOp = true;
      int opIdx = 0;

      for (auto &bodyOp : fuOp.getBody().front().getOperations()) {
        if (auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(bodyOp)) {
          // Track yield operands -> output connections
          for (unsigned yi = 0; yi < yieldOp.getNumOperands(); ++yi) {
            auto it = valToRef.find(yieldOp.getOperand(yi));
            if (it != valToRef.end()) {
              outputEdges.push_back(
                  {it->second.owner, static_cast<int>(yi), it->second.resultIdx});
            }
          }
          continue;
        }

        if (!firstOp) os << ", ";
        firstOp = false;
        os << "\"" << jsonEsc(bodyOp.getName().getStringRef().str()) << "\"";

        // Track operand sources
        for (unsigned operandIdx = 0; operandIdx < bodyOp.getNumOperands();
             ++operandIdx) {
          auto operand = bodyOp.getOperand(operandIdx);
          auto it = valToRef.find(operand);
          if (it != valToRef.end()) {
            if (it->second.owner >= 0) {
              dagEdges.push_back({it->second.owner, opIdx,
                                  it->second.resultIdx,
                                  static_cast<int>(operandIdx)});  // op -> op
            }
            else
              inputEdges.push_back({-(it->second.owner + 1), opIdx,
                                    static_cast<int>(operandIdx)}); // arg -> op
          }
        }
        for (unsigned resultIdx = 0; resultIdx < bodyOp.getNumResults();
             ++resultIdx)
          valToRef[bodyOp.getResult(resultIdx)] = {opIdx, static_cast<int>(resultIdx)};
        opIdx++;
      }
      os << "]";

      // op-to-op edges
      if (!dagEdges.empty()) {
        os << ", \"edges\": [";
        for (size_t k = 0; k < dagEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << dagEdges[k].srcOp << ", " << dagEdges[k].dstOp << ", "
             << dagEdges[k].srcResult << ", " << dagEdges[k].dstOperand << "]";
        }
        os << "]";
      }

      // input arg -> op edges
      if (!inputEdges.empty()) {
        os << ", \"inputEdges\": [";
        for (size_t k = 0; k < inputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << inputEdges[k].argIdx << ", " << inputEdges[k].dstOp
             << ", " << inputEdges[k].dstOperand << "]";
        }
        os << "]";
      }

      // op/arg -> yield output edges
      if (!outputEdges.empty()) {
        os << ", \"outputEdges\": [";
        for (size_t k = 0; k < outputEdges.size(); ++k) {
          if (k > 0) os << ", ";
          os << "[" << outputEdges[k].srcOwner << ", " << outputEdges[k].yieldIdx
             << ", " << outputEdges[k].srcResult << "]";
        }
        os << "]";
      }

      // Generate DOT: full FU internal DAG with I/O port nodes.
      // Graphviz handles ALL layout including port-to-op connections.
      if (opIdx > 0) {
        os << ", \"dot\": \"digraph FU {\\n";
        os << "  rankdir=TB;\\n";
        os << "  bgcolor=\\\"transparent\\\";\\n";
        os << "  node [style=filled, fontsize=9, "
           << "fontname=\\\"monospace\\\", fontcolor=\\\"#c8d6e5\\\"];\\n";
        os << "  edge [color=\\\"#6a8faf\\\", penwidth=1.2, "
           << "arrowsize=0.6];\\n";
        // Input port nodes
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai) {
          os << "  in" << ai
             << " [label=\\\"I" << ai << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#2a5f5f\\\", color=\\\"#4ecdc4\\\"];\\n";
        }
        // Op nodes
        {
          int oi2 = 0;
          for (auto &bodyOp2 : fuOp.getBody().front().getOperations()) {
            if (mlir::isa<fcc::fabric::YieldOp>(bodyOp2)) continue;
            std::string on = bodyOp2.getName().getStringRef().str();
            bool isMux = (on.find("mux") != std::string::npos);
            std::string displayName = on;
            auto dotPos = on.find('.');
            if (dotPos != std::string::npos)
              displayName = on.substr(0, dotPos) + "\\n" + on.substr(dotPos + 1);
            os << "  op" << oi2
               << " [label=\\\"" << jsonEsc(displayName) << "\\\", ";
            if (isMux)
              os << "shape=invtrapezium, fillcolor=\\\"#3a3520\\\", "
                 << "color=\\\"#ffd166\\\", fontcolor=\\\"#ffd166\\\"];\\n";
            else
              os << "shape=ellipse, fillcolor=\\\"#1a3050\\\", "
                 << "color=\\\"#5dade2\\\"];\\n";
            oi2++;
          }
        }
        // Output port nodes
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi) {
          os << "  out" << yi
             << " [label=\\\"O" << yi << "\\\", shape=square, "
             << "width=0.2, height=0.2, fontsize=7, "
             << "fillcolor=\\\"#5f2a1a\\\", color=\\\"#ff6b35\\\"];\\n";
        }
        // Rank constraints
        os << "  { rank=source; ";
        for (unsigned ai = 0; ai < fuFnType.getNumInputs(); ++ai)
          os << "in" << ai << "; ";
        os << "}\\n";
        os << "  { rank=sink; ";
        for (unsigned yi = 0; yi < fuFnType.getNumResults(); ++yi)
          os << "out" << yi << "; ";
        os << "}\\n";
        // Input -> op edges
        for (auto &ie : inputEdges)
          os << "  in" << ie.argIdx << " -> op" << ie.dstOp
             << " [color=\\\"#4ecdc4\\\"];\\n";
        // Op -> op edges
        for (auto &de : dagEdges)
          os << "  op" << de.srcOp << " -> op" << de.dstOp << ";\\n";
        // Op/arg -> output edges
        for (auto &oe : outputEdges) {
          if (oe.srcOwner >= 0)
            os << "  op" << oe.srcOwner << " -> out" << oe.yieldIdx
               << " [color=\\\"#ff6b35\\\"];\\n";
          else
            os << "  in" << (-(oe.srcOwner + 1)) << " -> out" << oe.yieldIdx
               << " [style=dashed, color=\\\"#888888\\\"];\\n";
        }
        os << "}\\n\"";
      }

      os << "}";
    }
    os << "]";
  };

  os << "  \"components\": [\n";
  bool first = true;
  auto &body = fabricMod.getBody().front();

  for (auto &op : body.getOperations()) {
    if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto peFnType = peOp.getFunctionType();
      os << "    {\"kind\": \"spatial_pe\", \"name\": \""
         << jsonEsc(getRenderName(peOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << peFnType.getNumInputs();
      os << ", \"numOutputs\": " << peFnType.getNumResults();
      emitPEFUs(peOp);
      os << "}";
      continue;
    }
    if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      if (!peOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = peOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto peFnType = peOp.getFunctionType();
      os << "    {\"kind\": \"temporal_pe\", \"name\": \""
         << jsonEsc(getRenderName(peOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << peFnType.getNumInputs();
      os << ", \"numOutputs\": " << peFnType.getNumResults();
      emitPEFUs(peOp);
      os << "}";
      continue;
    }
    if (auto swOp = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      if (!swOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = swOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto swFnType = swOp.getFunctionType();
      os << "    {\"kind\": \"spatial_sw\", \"name\": \""
         << jsonEsc(getRenderName(swOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << swFnType.getNumInputs();
      os << ", \"numOutputs\": " << swFnType.getNumResults();
      os << "}";
      continue;
    }
    if (auto swOp = mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      if (!swOp->hasAttr("inline_instantiation"))
        continue;
      llvm::StringRef symName;
      if (auto symNameAttr = swOp.getSymNameAttr())
        symName = symNameAttr.getValue();
      if (!first) os << ",\n";
      first = false;
      auto swFnType = swOp.getFunctionType();
      os << "    {\"kind\": \"temporal_sw\", \"name\": \""
         << jsonEsc(getRenderName(swOp.getOperation())) << "\"";
      if (!symName.empty())
        os << ", \"defName\": \"" << jsonEsc(symName.str()) << "\"";
      os << ", \"numInputs\": " << swFnType.getNumInputs();
      os << ", \"numOutputs\": " << swFnType.getNumResults();
      os << "}";
      continue;
    }

    if (auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      if (!extOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto memFnType = extOp.getFunctionType();
      os << "    {\"kind\": \"memory\", \"name\": \""
         << jsonEsc(getRenderName(extOp.getOperation())) << "\"";
      os << ", \"memoryKind\": \"extmemory\"";
      os << ", \"numInputs\": " << memFnType.getNumInputs();
      os << ", \"numOutputs\": " << memFnType.getNumResults();
      os << "}";
    }

    if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      if (!memOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto memFnType = memOp.getFunctionType();
      os << "    {\"kind\": \"memory\", \"name\": \""
         << jsonEsc(getRenderName(memOp.getOperation())) << "\"";
      os << ", \"memoryKind\": \"memory\"";
      os << ", \"numInputs\": " << memFnType.getNumInputs();
      os << ", \"numOutputs\": " << memFnType.getNumResults();
      os << "}";
    }

    if (auto addTagOp = mlir::dyn_cast<fcc::fabric::AddTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"add_tag\", \"name\": \""
         << jsonEsc(getRenderName(addTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto delTagOp = mlir::dyn_cast<fcc::fabric::DelTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"del_tag\", \"name\": \""
         << jsonEsc(getRenderName(delTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto mapTagOp = mlir::dyn_cast<fcc::fabric::MapTagOp>(op)) {
      if (!first) os << ",\n";
      first = false;
      os << "    {\"kind\": \"map_tag\", \"name\": \""
         << jsonEsc(getRenderName(mapTagOp.getOperation())) << "\"";
      os << ", \"numInputs\": 1, \"numOutputs\": 1}";
    }

    if (auto fifoOp = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      if (!fifoOp->hasAttr("inline_instantiation"))
        continue;
      if (!first) os << ",\n";
      first = false;
      auto fifoFnType = fifoOp.getFunctionType();
      os << "    {\"kind\": \"fifo\", \"name\": \""
         << jsonEsc(getRenderName(fifoOp.getOperation())) << "\"";
      os << ", \"numInputs\": " << fifoFnType.getNumInputs();
      os << ", \"numOutputs\": " << fifoFnType.getNumResults();
      os << "}";
    }

    // instances - resolve to PE/SW definitions for FU details
    if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
      if (!first) os << ",\n";
      first = false;

      auto moduleName = instOp.getModule();
      auto peIt = peDefMap.find(moduleName);
      auto temporalPeIt = temporalPeDefMap.find(moduleName);
      auto swIt = swDefMap.find(moduleName);

      if (peIt != peDefMap.end()) {
        // Instance of a spatial_pe - emit as PE with FU details
        auto peOp = peIt->second;
        auto peFnType = peOp.getFunctionType();
        os << "    {\"kind\": \"spatial_pe\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("pe").str()) << "\"";
        os << ", \"defName\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << peFnType.getNumInputs();
        os << ", \"numOutputs\": " << peFnType.getNumResults();
        emitPEFUs(peOp);
        os << "}";
      } else if (temporalPeIt != temporalPeDefMap.end()) {
        auto peOp = temporalPeIt->second;
        auto peFnType = peOp.getFunctionType();
        os << "    {\"kind\": \"temporal_pe\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("tpe").str()) << "\"";
        os << ", \"defName\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << peFnType.getNumInputs();
        os << ", \"numOutputs\": " << peFnType.getNumResults();
        emitPEFUs(peOp);
        os << "}";
      } else if (swIt != swDefMap.end()) {
        // Instance of a spatial_sw
        auto swOp = swIt->second;
        auto swFnType = swOp.getFunctionType();
        os << "    {\"kind\": \"spatial_sw\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("sw").str()) << "\"";
        os << ", \"numInputs\": " << swFnType.getNumInputs();
        os << ", \"numOutputs\": " << swFnType.getNumResults();
        os << "}";
      } else if (auto tswIt = temporalSwDefMap.find(moduleName);
                 tswIt != temporalSwDefMap.end()) {
        auto tswOp = tswIt->second;
        auto tswFnType = tswOp.getFunctionType();
        os << "    {\"kind\": \"temporal_sw\", \"name\": \""
           << jsonEsc(getRenderName(instOp.getOperation())) << "\"";
        os << ", \"numInputs\": " << tswFnType.getNumInputs();
        os << ", \"numOutputs\": " << tswFnType.getNumResults();
        os << "}";
      } else {
        // Generic instance
        os << "    {\"kind\": \"instance\", \"name\": \""
           << jsonEsc(instOp.getSymName().value_or("inst").str()) << "\"";
        os << ", \"module\": \"" << jsonEsc(moduleName.str()) << "\"";
        os << ", \"numInputs\": " << instOp.getNumOperands();
        os << ", \"numOutputs\": " << instOp.getNumResults();
        os << "}";
      }
    }
  }

  os << "\n  ],\n";

  // Connections: trace SSA (two-pass for graph-region circular references).
  os << "  \"connections\": [\n";
  bool firstConn = true;

  llvm::DenseMap<mlir::Value, int> blockArgIdx;
  for (auto arg : body.getArguments())
    blockArgIdx[arg] = static_cast<int>(arg.getArgNumber());

  // Pass 1: collect all producer results first (handles forward references).
  struct ResultProducer {
    std::string name;
    unsigned idx;
  };
  llvm::DenseMap<mlir::Value, ResultProducer> resultProducerMap;
  for (auto &op : body.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                  fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp>(op))
      continue;
    std::string renderName = getRenderName(&op);
    for (unsigned i = 0; i < op.getNumResults(); ++i)
      resultProducerMap[op.getResult(i)] = {renderName, i};
  }

  // Pass 2: trace all operand connections for renderable operations.
  for (auto &op : body.getOperations()) {
    if (mlir::isa<fcc::fabric::YieldOp>(op))
      continue;
    if (mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp,
                  fcc::fabric::SpatialSwOp, fcc::fabric::TemporalSwOp>(op))
      continue;
    std::string opName = getRenderName(&op);

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      // Module input -> instance
      auto argIt = blockArgIdx.find(operand);
      if (argIt != blockArgIdx.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"module_in\", \"fromIdx\": " << argIt->second
           << ", \"to\": \"" << jsonEsc(opName) << "\", \"toIdx\": " << i
           << "}";
      }
      // Instance -> instance (now works for circular refs too)
      auto irIt = resultProducerMap.find(operand);
      if (irIt != resultProducerMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(irIt->second.name)
           << "\", \"fromIdx\": " << irIt->second.idx
           << ", \"to\": \"" << jsonEsc(opName) << "\", \"toIdx\": " << i
           << "}";
      }
    }
  }

  // Module memref bindings and SW->ExtMem reverse links come from ExtMemory
  // metadata because the Fabric inline syntax does not spell them out as SSA
  // operands.
  for (auto &op : body.getOperations()) {
    auto extOp = mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op);
    if (!extOp)
      continue;
    if (extOp.getNumOperands() > 0)
      continue;

    std::string memName = getRenderName(extOp.getOperation());
    if (auto argIdxAttr =
            extOp->getAttrOfType<mlir::IntegerAttr>("memref_arg_index")) {
      if (!firstConn)
        os << ",\n";
      firstConn = false;
      os << "    {\"from\": \"module_in\", \"fromIdx\": "
         << argIdxAttr.getInt() << ", \"to\": \"" << jsonEsc(memName)
         << "\", \"toIdx\": 0}";
    }

    auto emitExtMemBackEdges = [&](mlir::ArrayAttr detailAttr,
                                   bool detailed) {
      for (auto elem : detailAttr) {
        llvm::StringRef swName;
        int64_t outputBase = detailed ? 0 : 4;
        if (detailed) {
          auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(elem);
          if (!dictAttr)
            continue;
          auto nameAttr = dictAttr.getAs<mlir::StringAttr>("name");
          if (!nameAttr)
            continue;
          swName = nameAttr.getValue();
          if (auto outBaseAttr =
                  dictAttr.getAs<mlir::IntegerAttr>("output_port_base")) {
            outputBase = outBaseAttr.getInt();
          }
        } else {
          auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
          if (!strAttr)
            continue;
          swName = strAttr.getValue();
        }

        unsigned numDataInputs =
            extOp.getFunctionType().getNumInputs() > 0
                ? extOp.getFunctionType().getNumInputs() - 1
                : 0;
        for (unsigned p = 0; p < numDataInputs; ++p) {
          if (!firstConn)
            os << ",\n";
          firstConn = false;
          os << "    {\"from\": \"" << jsonEsc(swName.str())
             << "\", \"fromIdx\": " << (outputBase + static_cast<int64_t>(p))
             << ", \"to\": \"" << jsonEsc(memName) << "\", \"toIdx\": "
             << (1 + p) << "}";
        }
      }
    };

    if (auto detailAttr =
            extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw_detail")) {
      emitExtMemBackEdges(detailAttr, /*detailed=*/true);
    } else if (auto connAttr =
                   extOp->getAttrOfType<mlir::ArrayAttr>("connected_sw")) {
      emitExtMemBackEdges(connAttr, /*detailed=*/false);
    }
  }

  // Yield: instance results -> module outputs
  auto yieldOp = mlir::dyn_cast<fcc::fabric::YieldOp>(body.getTerminator());
  if (yieldOp) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); ++i) {
      auto ir = resultProducerMap.find(yieldOp->getOperand(i));
      if (ir != resultProducerMap.end()) {
        if (!firstConn) os << ",\n";
        firstConn = false;
        os << "    {\"from\": \"" << jsonEsc(ir->second.name)
           << "\", \"fromIdx\": " << ir->second.idx
           << ", \"to\": \"module_out\", \"toIdx\": " << i << "}";
      }
    }
  }

  os << "\n  ]\n}";
}

// ---- Serialize handshake.func DFG to DOT + JSON ----

struct DFGJsonPortInfo {
  unsigned index = 0;
  std::string name;
  std::string type;
};

struct DFGJsonNodeInfo {
  int id = 0;
  std::string kind;
  std::string label;
  std::string display;
  std::string op;
  int argIndex = -1;
  int resultIndex = -1;
  std::string name;
  std::string type;
  std::vector<DFGJsonPortInfo> inputs;
  std::vector<DFGJsonPortInfo> outputs;
  double width = 0.0;
  double height = 0.0;
  double x = 0.0;
  double y = 0.0;
  bool hasLayout = false;
};

struct DFGJsonEdgeInfo {
  int id = 0;
  int from = 0;
  unsigned fromPort = 0;
  int to = 0;
  unsigned toPort = 0;
  std::string edgeType;
  std::string valueType;
  std::vector<std::pair<double, double>> points;
};

struct DFGJsonModel {
  std::string funcName;
  std::vector<DFGJsonNodeInfo> nodes;
  std::vector<DFGJsonEdgeInfo> edges;
};

static std::pair<double, double> getDFGPortAnchor(const DFGJsonNodeInfo &node,
                                                  bool isInput,
                                                  unsigned portIndex) {
  if (node.kind == "input")
    return {node.x, node.y + node.height / 2.0 - 2.0};
  if (node.kind == "output")
    return {node.x, node.y - node.height / 2.0 + 2.0};

  const auto &ports = isInput ? node.inputs : node.outputs;
  size_t count = std::max<size_t>(ports.size(), 1);
  double slot = static_cast<double>(portIndex + 1);
  double x = node.x - node.width / 2.0 + (node.width * slot) / (count + 1.0);
  double y =
      isInput ? (node.y - node.height / 2.0 + 2.0) : (node.y + node.height / 2.0 - 2.0);
  return {x, y};
}

static void simplifyDFGPolyline(
    std::vector<std::pair<double, double>> &points) {
  if (points.size() <= 2)
    return;
  std::vector<std::pair<double, double>> out;
  out.reserve(points.size());
  out.push_back(points.front());
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    const auto &a = out.back();
    const auto &b = points[i];
    const auto &c = points[i + 1];
    bool sameX = std::abs(a.first - b.first) < 0.01 &&
                 std::abs(b.first - c.first) < 0.01;
    bool sameY = std::abs(a.second - b.second) < 0.01 &&
                 std::abs(b.second - c.second) < 0.01;
    if (sameX || sameY)
      continue;
    out.push_back(b);
  }
  out.push_back(points.back());
  points.swap(out);
}

static std::vector<std::pair<double, double>>
buildFallbackDFGEdgePolyline(const DFGJsonModel &model,
                             const DFGJsonEdgeInfo &edge) {
  const auto &src = model.nodes[edge.from];
  const auto &dst = model.nodes[edge.to];
  auto srcPt = getDFGPortAnchor(src, /*isInput=*/false, edge.fromPort);
  auto dstPt = getDFGPortAnchor(dst, /*isInput=*/true, edge.toPort);
  bool isForward = srcPt.second < dstPt.second - 8.0;
  double laneSpread = static_cast<double>((edge.id % 5) - 2) * 18.0;
  std::vector<std::pair<double, double>> points;
  if (isForward) {
    double stub = 18.0;
    double midY = (srcPt.second + dstPt.second) / 2.0 + laneSpread;
    double minMid = srcPt.second + stub + 8.0;
    double maxMid = dstPt.second - stub - 8.0;
    if (minMid <= maxMid)
      midY = std::max(minMid, std::min(maxMid, midY));
    points = {
        srcPt,
        {srcPt.first, srcPt.second + stub},
        {srcPt.first, midY},
        {dstPt.first, midY},
        {dstPt.first, dstPt.second - stub},
        dstPt,
    };
  } else {
    double bendX =
        std::max(srcPt.first, dstPt.first) + 84.0 + static_cast<double>(edge.id % 4) * 18.0;
    points = {
        srcPt,
        {bendX, srcPt.second},
        {bendX, dstPt.second},
        dstPt,
    };
  }
  simplifyDFGPolyline(points);
  return points;
}

static bool routeDFGEdgesOffline(DFGJsonModel &model) {
  if (model.nodes.empty() || model.edges.empty())
    return false;
  for (const auto &node : model.nodes) {
    if (!node.hasLayout)
      return false;
  }

  constexpr int kGridStep = 10;
  constexpr double kNodePad = 10.0;
  constexpr double kCanvasPad = 90.0;
  constexpr double kStubLen = 22.0;

  double minX = std::numeric_limits<double>::infinity();
  double minY = std::numeric_limits<double>::infinity();
  double maxX = -std::numeric_limits<double>::infinity();
  double maxY = -std::numeric_limits<double>::infinity();
  for (const auto &node : model.nodes) {
    minX = std::min(minX, node.x - node.width / 2.0);
    minY = std::min(minY, node.y - node.height / 2.0);
    maxX = std::max(maxX, node.x + node.width / 2.0);
    maxY = std::max(maxY, node.y + node.height / 2.0);
  }
  if (!std::isfinite(minX) || !std::isfinite(minY) || !std::isfinite(maxX) ||
      !std::isfinite(maxY))
    return false;

  double gridMinX = minX - kCanvasPad;
  double gridMinY = minY - kCanvasPad;
  double gridMaxX = maxX + kCanvasPad;
  double gridMaxY = maxY + kCanvasPad;
  int gridCols =
      std::max(1, static_cast<int>(std::llround((gridMaxX - gridMinX) / kGridStep)));
  int gridRows =
      std::max(1, static_cast<int>(std::llround((gridMaxY - gridMinY) / kGridStep)));

  auto cellKey = [](int col, int row) -> int64_t {
    return (static_cast<int64_t>(col) << 32) ^
           static_cast<uint32_t>(row);
  };
  auto edgeKey = [&](int c0, int r0, int c1, int r1) -> std::tuple<int, int, int, int> {
    if (c0 < c1 || (c0 == c1 && r0 <= r1))
      return {c0, r0, c1, r1};
    return {c1, r1, c0, r0};
  };
  auto clampGrid = [&](int value, int upper) {
    return std::max(0, std::min(upper, value));
  };
  auto toGrid = [&](double x, double y) -> std::pair<int, int> {
    int col = clampGrid(static_cast<int>(std::llround((x - gridMinX) / kGridStep)),
                        gridCols);
    int row = clampGrid(static_cast<int>(std::llround((y - gridMinY) / kGridStep)),
                        gridRows);
    return {col, row};
  };
  auto toPixel = [&](int col, int row) -> std::pair<double, double> {
    return {gridMinX + static_cast<double>(col) * kGridStep,
            gridMinY + static_cast<double>(row) * kGridStep};
  };

  std::unordered_set<int64_t> blocked;
  for (const auto &node : model.nodes) {
    double left = node.x - node.width / 2.0 - kNodePad;
    double right = node.x + node.width / 2.0 + kNodePad;
    double top = node.y - node.height / 2.0 - kNodePad;
    double bottom = node.y + node.height / 2.0 + kNodePad;
    int c0 = static_cast<int>(std::floor((left - gridMinX) / kGridStep));
    int c1 = static_cast<int>(std::ceil((right - gridMinX) / kGridStep));
    int r0 = static_cast<int>(std::floor((top - gridMinY) / kGridStep));
    int r1 = static_cast<int>(std::ceil((bottom - gridMinY) / kGridStep));
    for (int col = c0; col <= c1; ++col) {
      for (int row = r0; row <= r1; ++row) {
        if (col < 0 || col > gridCols || row < 0 || row > gridRows)
          continue;
        blocked.insert(cellKey(col, row));
      }
    }
  }

  auto nearestUnblockedCell = [&](std::pair<int, int> seed) {
    if (!blocked.count(cellKey(seed.first, seed.second)))
      return seed;
    for (int radius = 1; radius < 24; ++radius) {
      for (int dc = -radius; dc <= radius; ++dc) {
        for (int dr = -radius; dr <= radius; ++dr) {
          if (std::abs(dc) != radius && std::abs(dr) != radius)
            continue;
          int col = seed.first + dc;
          int row = seed.second + dr;
          if (col < 0 || col > gridCols || row < 0 || row > gridRows)
            continue;
          if (!blocked.count(cellKey(col, row)))
            return std::pair<int, int>{col, row};
        }
      }
    }
    return seed;
  };

  struct RouteState {
    int col = 0;
    int row = 0;
    int dir = -1;
    bool operator<(const RouteState &other) const {
      return std::tie(col, row, dir) < std::tie(other.col, other.row, other.dir);
    }
  };
  struct QueueItem {
    RouteState state;
    double f = 0.0;
  };
  struct QueueItemGreater {
    bool operator()(const QueueItem &lhs, const QueueItem &rhs) const {
      return lhs.f > rhs.f;
    }
  };

  std::map<std::tuple<int, int, int, int>, unsigned> usedSegments;
  std::vector<size_t> edgeOrder(model.edges.size());
  std::iota(edgeOrder.begin(), edgeOrder.end(), 0);
  std::sort(edgeOrder.begin(), edgeOrder.end(),
            [&](size_t lhsIdx, size_t rhsIdx) {
              const auto &lhs = model.edges[lhsIdx];
              const auto &rhs = model.edges[rhsIdx];
              double lhsY = model.nodes[lhs.from].y + model.nodes[lhs.to].y;
              double rhsY = model.nodes[rhs.from].y + model.nodes[rhs.to].y;
              if (lhsY != rhsY)
                return lhsY < rhsY;
              return lhs.id < rhs.id;
            });

  auto routeGrid = [&](std::pair<int, int> start,
                       std::pair<int, int> goal) -> std::optional<std::vector<std::pair<int, int>>> {
    static constexpr std::array<std::pair<int, int>, 4> kDirs = {
        std::pair<int, int>{1, 0}, std::pair<int, int>{-1, 0},
        std::pair<int, int>{0, 1}, std::pair<int, int>{0, -1}};

    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueItemGreater> open;
    std::map<RouteState, double> cost;
    std::map<RouteState, RouteState> prev;
    std::set<RouteState> closed;

    RouteState startState{start.first, start.second, -1};
    cost[startState] = 0.0;
    open.push({startState, 0.0});

    std::optional<RouteState> goalState;
    while (!open.empty()) {
      RouteState cur = open.top().state;
      open.pop();
      if (closed.count(cur))
        continue;
      closed.insert(cur);
      if (cur.col == goal.first && cur.row == goal.second) {
        goalState = cur;
        break;
      }
      double baseCost = cost[cur];
      for (int dir = 0; dir < 4; ++dir) {
        int nextCol = cur.col + kDirs[dir].first;
        int nextRow = cur.row + kDirs[dir].second;
        if (nextCol < 0 || nextRow < 0 || nextCol > gridCols || nextRow > gridRows)
          continue;
        if (blocked.count(cellKey(nextCol, nextRow)))
          continue;

        double stepCost = 1.0;
        if (cur.dir >= 0 && dir != cur.dir)
          stepCost += 0.9;
        if (cur.dir >= 0 && dir == (cur.dir ^ 1))
          stepCost += 1.6;
        auto seg = edgeKey(cur.col, cur.row, nextCol, nextRow);
        auto segIt = usedSegments.find(seg);
        if (segIt != usedSegments.end())
          stepCost += 2.4 * static_cast<double>(segIt->second);

        RouteState nextState{nextCol, nextRow, dir};
        double newCost = baseCost + stepCost;
        auto costIt = cost.find(nextState);
        if (costIt != cost.end() && newCost >= costIt->second)
          continue;
        cost[nextState] = newCost;
        prev[nextState] = cur;
        double heuristic =
            static_cast<double>(std::abs(nextCol - goal.first) + std::abs(nextRow - goal.second));
        open.push({nextState, newCost + heuristic});
      }
    }

    if (!goalState)
      return std::nullopt;

    std::vector<std::pair<int, int>> path;
    for (RouteState cur = *goalState;;) {
      path.push_back({cur.col, cur.row});
      auto prevIt = prev.find(cur);
      if (prevIt == prev.end())
        break;
      cur = prevIt->second;
    }
    std::reverse(path.begin(), path.end());
    return path;
  };

  auto reservePath = [&](const std::vector<std::pair<int, int>> &path) {
    for (size_t i = 0; i + 1 < path.size(); ++i)
      usedSegments[edgeKey(path[i].first, path[i].second, path[i + 1].first,
                           path[i + 1].second)]++;
  };

  for (size_t edgeIdx : edgeOrder) {
    auto &edge = model.edges[edgeIdx];
    const auto &src = model.nodes[edge.from];
    const auto &dst = model.nodes[edge.to];
    auto srcPt = getDFGPortAnchor(src, /*isInput=*/false, edge.fromPort);
    auto dstPt = getDFGPortAnchor(dst, /*isInput=*/true, edge.toPort);
    std::pair<double, double> srcStub{srcPt.first, srcPt.second + kStubLen};
    std::pair<double, double> dstStub{dstPt.first, dstPt.second - kStubLen};
    auto startCell = nearestUnblockedCell(toGrid(srcStub.first, srcStub.second));
    auto goalCell = nearestUnblockedCell(toGrid(dstStub.first, dstStub.second));
    auto gridPath = routeGrid(startCell, goalCell);
    if (!gridPath) {
      edge.points = buildFallbackDFGEdgePolyline(model, edge);
      continue;
    }

    reservePath(*gridPath);
    edge.points.clear();
    edge.points.push_back(srcPt);
    edge.points.push_back(srcStub);
    for (const auto &cell : *gridPath)
      edge.points.push_back(toPixel(cell.first, cell.second));
    edge.points.push_back(dstStub);
    edge.points.push_back(dstPt);
    simplifyDFGPolyline(edge.points);
  }
  return true;
}

static std::pair<double, double> estimateDFGNodeSize(const DFGJsonNodeInfo &node) {
  if (node.kind == "input" || node.kind == "output")
    return {132.0, 62.0};

  auto maxLabelLen = [&](llvm::StringRef text) -> size_t {
    size_t maxLen = 0;
    size_t start = 0;
    while (start <= text.size()) {
      size_t end = text.find('\n', start);
      llvm::StringRef line =
          end == llvm::StringRef::npos ? text.substr(start) : text.slice(start, end);
      maxLen = std::max(maxLen, line.size());
      if (end == llvm::StringRef::npos)
        break;
      start = end + 1;
    }
    return maxLen;
  };

  size_t maxPortCount =
      std::max(node.inputs.size(), std::max(node.outputs.size(), size_t{1}));
  double width = std::max<double>(
      128.0,
      std::max<double>(static_cast<double>(maxLabelLen(node.display)) * 8.0 + 36.0,
                       static_cast<double>(maxPortCount) * 58.0));
  return {width, 82.0};
}

static void buildDFGJsonModel(DFGJsonModel &model, mlir::ModuleOp topModule) {
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });

  if (!funcOp) {
    model.funcName.clear();
    return;
  }

  auto &body = funcOp.getBody().front();
  auto argNames = funcOp->getAttrOfType<mlir::ArrayAttr>("argNames");
  auto resNames = funcOp->getAttrOfType<mlir::ArrayAttr>("resNames");
  model.funcName = funcOp.getName().str();

  llvm::SmallVector<mlir::Operation *, 16> ops;
  circt::handshake::ReturnOp returnOp;
  for (auto &op : body.getOperations()) {
    if (auto ret = mlir::dyn_cast<circt::handshake::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    ops.push_back(&op);
  }

  llvm::DenseMap<mlir::Value, std::pair<int, unsigned>> valueToNodePort;
  int nodeId = 0;

  for (auto arg : body.getArguments()) {
    std::string argName;
    if (argNames && arg.getArgNumber() < argNames.size()) {
      if (auto str =
              mlir::dyn_cast<mlir::StringAttr>(argNames[arg.getArgNumber()]))
        argName = str.getValue().str();
    }
    std::string typeStr = printType(arg.getType());
    DFGJsonNodeInfo node;
    node.id = nodeId;
    node.kind = "input";
    node.label = "arg" + std::to_string(arg.getArgNumber());
    node.argIndex = arg.getArgNumber();
    node.name = argName;
    node.type = typeStr;
    node.outputs.push_back({0, !argName.empty() ? argName : std::string("value"),
                            typeStr});
    auto [w, h] = estimateDFGNodeSize(node);
    node.width = w;
    node.height = h;
    model.nodes.push_back(std::move(node));
    valueToNodePort[arg] = {nodeId, 0};
    nodeId++;
  }

  for (auto *op : ops) {
    std::string opName = op->getName().getStringRef().str();
    std::string displayName = opName;
    size_t dotPos = displayName.find('.');
    if (dotPos != std::string::npos)
      displayName =
          displayName.substr(0, dotPos) + "\n" + displayName.substr(dotPos + 1);

    DFGJsonNodeInfo node;
    node.id = nodeId;
    node.kind = "op";
    node.label = opName;
    node.display = displayName;
    node.op = opName;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      node.inputs.push_back(
          {i, dfgOperandName(op, i), printType(op->getOperand(i).getType())});
    }
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      node.outputs.push_back(
          {i, dfgResultName(op, i), printType(op->getResult(i).getType())});
    }
    auto [w, h] = estimateDFGNodeSize(node);
    node.width = w;
    node.height = h;
    model.nodes.push_back(std::move(node));

    for (unsigned i = 0; i < op->getNumResults(); ++i)
      valueToNodePort[op->getResult(i)] = {nodeId, i};
    nodeId++;
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      std::string resName;
      if (resNames && i < resNames.size()) {
        if (auto str = mlir::dyn_cast<mlir::StringAttr>(resNames[i]))
          resName = str.getValue().str();
      }
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      DFGJsonNodeInfo node;
      node.id = nodeId;
      node.kind = "output";
      node.label = "return";
      node.resultIndex = static_cast<int>(i);
      node.name = resName;
      node.type = typeStr;
      node.inputs.push_back({0, !resName.empty() ? resName : std::string("value"),
                             typeStr});
      auto [w, h] = estimateDFGNodeSize(node);
      node.width = w;
      node.height = h;
      model.nodes.push_back(std::move(node));
      nodeId++;
    }
  }

  llvm::DenseMap<mlir::Operation *, int> opToNodeId;
  int firstOpNodeId = static_cast<int>(body.getNumArguments());
  for (unsigned i = 0; i < ops.size(); ++i)
    opToNodeId[ops[i]] = firstOpNodeId + static_cast<int>(i);

  int firstOutputNodeId = firstOpNodeId + static_cast<int>(ops.size());
  int edgeId = 0;

  for (auto *op : ops) {
    int dstNodeId = opToNodeId[op];
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = valueToNodePort.find(op->getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      std::string typeStr = printType(op->getOperand(i).getType());
      model.edges.push_back({edgeId, it->second.first, it->second.second,
                             dstNodeId, i,
                             dfgEdgeType(op->getOperand(i).getType()),
                             typeStr});
      edgeId++;
    }
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      auto it = valueToNodePort.find(returnOp.getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      model.edges.push_back({edgeId, it->second.first, it->second.second,
                             firstOutputNodeId + static_cast<int>(i), 0,
                             dfgEdgeType(returnOp.getOperand(i).getType()),
                             typeStr});
      edgeId++;
    }
  }
}

static bool applyDotLayoutToDFGModel(DFGJsonModel &model) {
  if (model.nodes.size() <= 1)
    return false;

  auto dotPath = llvm::sys::findProgramByName("dot");
  if (!dotPath)
    return false;

  llvm::SmallString<128> srcPath;
  llvm::SmallString<128> plainPath;
  int srcFd = -1;
  int plainFd = -1;
  if (llvm::sys::fs::createTemporaryFile("fcc_dfg_layout", "dot", srcFd, srcPath))
    return false;
  if (llvm::sys::fs::createTemporaryFile("fcc_dfg_layout", "plain", plainFd,
                                         plainPath)) {
    llvm::sys::fs::remove(srcPath);
    return false;
  }

  auto cleanup = [&]() {
    llvm::sys::fs::remove(srcPath);
    llvm::sys::fs::remove(plainPath);
  };

  {
    llvm::raw_fd_ostream dotOS(srcFd, true);
    auto dotNum = [&](double value) {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(6) << value;
      return ss.str();
    };
    constexpr double kPxPerIn = 72.0;
    dotOS << "digraph G {\n";
    dotOS << "  graph [rankdir=TB, overlap=false, splines=ortho, "
             "nodesep=1.25, ranksep=1.7, outputorder=edgesfirst, pad=0.2];\n";
    dotOS << "  node [fixedsize=true, margin=0, label=\"\"];\n";
    dotOS << "  edge [arrowsize=0.8, penwidth=1.2];\n";
    for (const auto &node : model.nodes) {
      llvm::StringRef shape =
          (node.kind == "op") ? llvm::StringRef("ellipse") : llvm::StringRef("box");
      dotOS << "  n" << node.id << " [shape=" << shape
            << ", width=" << dotNum(std::max(0.25, node.width / kPxPerIn))
            << ", height=" << dotNum(std::max(0.25, node.height / kPxPerIn))
            << "];\n";
    }
    for (const auto &edge : model.edges)
      dotOS << "  n" << edge.from << " -> n" << edge.to << ";\n";
    dotOS << "}\n";
  }
  {
    llvm::raw_fd_ostream plainOS(plainFd, true);
    plainOS.flush();
  }

  std::string errMsg;
  bool execFailed = false;
  std::string dotPathStr = *dotPath;
  std::string srcPathStr = srcPath.str().str();
  std::string plainPathStr = plainPath.str().str();
  llvm::SmallVector<llvm::StringRef, 5> args = {
      llvm::StringRef(dotPathStr), llvm::StringRef("-Tplain"),
      llvm::StringRef(srcPathStr), llvm::StringRef("-o"),
      llvm::StringRef(plainPathStr)};
  int rc = llvm::sys::ExecuteAndWait(dotPathStr, args, std::nullopt, {}, 0, 0,
                                     &errMsg, &execFailed);
  if (rc != 0 || execFailed) {
    cleanup();
    return false;
  }

  auto plainBuf = llvm::MemoryBuffer::getFile(plainPath);
  if (!plainBuf) {
    cleanup();
    return false;
  }

  double graphHeightIn = 0.0;
  std::map<int, std::pair<double, double>> nodePosIn;
  std::map<std::pair<int, int>, std::vector<size_t>> edgeBuckets;
  std::map<std::pair<int, int>, size_t> edgeBucketCursor;
  for (size_t idx = 0; idx < model.edges.size(); ++idx)
    edgeBuckets[{model.edges[idx].from, model.edges[idx].to}].push_back(idx);
  std::istringstream input((*plainBuf)->getBuffer().str());
  std::string lineStr;
  while (std::getline(input, lineStr)) {
    llvm::StringRef line(lineStr);
    line = line.trim();
    if (line.empty())
      continue;
    std::istringstream ls(line.str());
    std::string kind;
    ls >> kind;
    if (kind == "graph") {
      double scale = 0.0, width = 0.0;
      ls >> scale >> width >> graphHeightIn;
      (void)scale;
      (void)width;
      continue;
    }
    if (kind == "node") {
      std::string nodeName;
      double xIn = 0.0, yIn = 0.0, widthIn = 0.0, heightIn = 0.0;
      ls >> nodeName >> xIn >> yIn >> widthIn >> heightIn;
      if (nodeName.size() <= 1 || nodeName[0] != 'n')
        continue;
      try {
        int nodeId = std::stoi(nodeName.substr(1));
        nodePosIn[nodeId] = {xIn, yIn};
      } catch (...) {
      }
      continue;
    }
    if (kind == "edge") {
      std::string tailName, headName;
      int numPoints = 0;
      ls >> tailName >> headName >> numPoints;
      if (tailName.size() <= 1 || headName.size() <= 1 || tailName[0] != 'n' ||
          headName[0] != 'n')
        continue;
      int fromId = -1;
      int toId = -1;
      try {
        fromId = std::stoi(tailName.substr(1));
        toId = std::stoi(headName.substr(1));
      } catch (...) {
        continue;
      }
      auto key = std::make_pair(fromId, toId);
      auto bucketIt = edgeBuckets.find(key);
      if (bucketIt == edgeBuckets.end())
        continue;
      size_t bucketIdx = edgeBucketCursor[key];
      if (bucketIdx >= bucketIt->second.size())
        continue;
      auto &edge = model.edges[bucketIt->second[bucketIdx]];
      edgeBucketCursor[key] = bucketIdx + 1;
      for (int i = 0; i < numPoints; ++i) {
        double xIn = 0.0, yIn = 0.0;
        ls >> xIn >> yIn;
        edge.points.push_back({xIn, yIn});
      }
    }
  }
  cleanup();

  if (graphHeightIn <= 0.0 || nodePosIn.empty())
    return false;

  constexpr double kPxPerIn = 72.0;
  double minLeft = 0.0;
  double minTop = 0.0;
  bool haveBounds = false;
  for (auto &node : model.nodes) {
    auto it = nodePosIn.find(node.id);
    if (it == nodePosIn.end())
      continue;
    node.x = it->second.first * kPxPerIn;
    node.y = (graphHeightIn - it->second.second) * kPxPerIn;
    node.hasLayout = true;
    double left = node.x - node.width / 2.0;
    double top = node.y - node.height / 2.0;
    if (!haveBounds) {
      minLeft = left;
      minTop = top;
      haveBounds = true;
    } else {
      minLeft = std::min(minLeft, left);
      minTop = std::min(minTop, top);
    }
  }
  if (!haveBounds)
    return false;

  for (auto &edge : model.edges) {
    for (auto &pt : edge.points) {
      pt.first *= kPxPerIn;
      pt.second = (graphHeightIn - pt.second) * kPxPerIn;
      minLeft = std::min(minLeft, pt.first);
      minTop = std::min(minTop, pt.second);
    }
  }

  double shiftX = 80.0 - minLeft;
  double shiftY = 80.0 - minTop;
  for (auto &node : model.nodes) {
    if (node.hasLayout) {
      node.x += shiftX;
      node.y += shiftY;
    }
  }
  for (auto &edge : model.edges) {
    for (auto &pt : edge.points) {
      pt.first += shiftX;
      pt.second += shiftY;
    }
  }
  return true;
}

static void writeDFGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule) {
  DFGJsonModel model;
  buildDFGJsonModel(model, topModule);
  applyDotLayoutToDFGModel(model);
  routeDFGEdgesOffline(model);

  os << "{\n";
  if (model.nodes.empty()) {
    os << "  \"dot\": null, \"nodes\": [], \"edges\": []\n}";
    return;
  }

  os << "  \"func\": \"" << jsonEsc(model.funcName) << "\",\n";
  os << "  \"nodes\": [\n";
  for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
    const auto &node = model.nodes[ni];
    if (ni > 0)
      os << ",\n";
    os << "    {\"id\": " << node.id << ", \"kind\": \"" << node.kind << "\""
       << ", \"label\": \"" << jsonEsc(node.label) << "\"";
    if (!node.display.empty())
      os << ", \"display\": \"" << jsonEsc(node.display) << "\"";
    if (!node.op.empty())
      os << ", \"op\": \"" << jsonEsc(node.op) << "\"";
    if (node.argIndex >= 0)
      os << ", \"arg_index\": " << node.argIndex;
    if (node.resultIndex >= 0)
      os << ", \"result_index\": " << node.resultIndex;
    os << ", \"name\": \"" << jsonEsc(node.name) << "\""
       << ", \"type\": \"" << jsonEsc(node.type) << "\""
       << ", \"w\": " << node.width
       << ", \"h\": " << node.height;
    if (node.hasLayout)
      os << ", \"x\": " << node.x << ", \"y\": " << node.y;
    auto emitPorts = [&](llvm::StringRef key,
                         const std::vector<DFGJsonPortInfo> &ports) {
      os << ", \"" << key << "\": [";
      for (size_t pi = 0; pi < ports.size(); ++pi) {
        if (pi > 0)
          os << ", ";
        os << "{\"index\": " << ports[pi].index
           << ", \"name\": \"" << jsonEsc(ports[pi].name)
           << "\", \"type\": \"" << jsonEsc(ports[pi].type) << "\"}";
      }
      os << "]";
    };
    emitPorts("inputs", node.inputs);
    emitPorts("outputs", node.outputs);
    os << "}";
  }
  os << "\n  ],\n";

  os << "  \"edges\": [\n";
  for (size_t ei = 0; ei < model.edges.size(); ++ei) {
    const auto &edge = model.edges[ei];
    if (ei > 0)
      os << ",\n";
    os << "    {\"id\": " << edge.id
       << ", \"from\": " << edge.from
       << ", \"from_port\": " << edge.fromPort
       << ", \"to\": " << edge.to
       << ", \"to_port\": " << edge.toPort
       << ", \"edge_type\": \"" << jsonEsc(edge.edgeType)
       << "\", \"value_type\": \"" << jsonEsc(edge.valueType) << "\"";
    if (!edge.points.empty()) {
      os << ", \"points\": [";
      for (size_t pi = 0; pi < edge.points.size(); ++pi) {
        if (pi > 0)
          os << ", ";
        os << "{\"x\": " << edge.points[pi].first
           << ", \"y\": " << edge.points[pi].second << "}";
      }
      os << "]";
    }
    os << "}";
  }
  os << "\n  ],\n";
  os << "  \"dot\": " << (model.nodes.empty() ? "null" : "\"dot\"") << "\n}";
}

// ---- Public API ----

mlir::LogicalResult exportVizOnly(const std::string &outputPath,
                                  mlir::ModuleOp adgModule,
                                  mlir::ModuleOp dfgModule,
                                  const std::string &adgSourcePath,
                                  VizLayoutMode layoutMode,
                                  mlir::MLIRContext *ctx) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  std::string title = makeVizTitle(adgModule, dfgModule, false);

  // Serialize data
  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }
  std::string adgLayoutJson =
      selectVizLayoutJson(adgModule, adgSourcePath, adgJson, layoutMode);

  // Emit HTML
  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  // Embedded data
  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const ADG_LAYOUT_DATA = " << scriptSafe(adgLayoutJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n"
      << "const MAPPING_DATA = null;\n"
      << "</script>\n\n";

  // Bundled D3 for self-contained local viewing.
  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";

  // Renderer JS
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return mlir::success();
}

mlir::LogicalResult exportVizWithMapping(const std::string &outputPath,
                                         mlir::ModuleOp adgModule,
                                         mlir::ModuleOp dfgModule,
                                         const std::string &mapJsonPath,
                                         const std::string &adgSourcePath,
                                         VizLayoutMode layoutMode,
                                         mlir::MLIRContext *ctx) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << "\n";
    return mlir::failure();
  }

  // Read mapping JSON
  std::string mapJson = "null";
  auto mapBuf = llvm::MemoryBuffer::getFile(mapJsonPath);
  if (mapBuf)
    mapJson = (*mapBuf)->getBuffer().str();

  std::string adgJson;
  if (adgModule) {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJson(ss, adgModule, ctx);
  } else {
    adgJson = "null";
  }

  std::string dfgJson;
  if (dfgModule) {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJson(ss, dfgModule);
  } else {
    dfgJson = "null";
  }
  std::string adgLayoutJson =
      selectVizLayoutJson(adgModule, adgSourcePath, adgJson, layoutMode);

  std::string title = makeVizTitle(adgModule, dfgModule, true);

  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEsc(title) << "</title>\n"
      << "  <style>\n" << viz::RENDERER_CSS << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEsc(title) << "</span>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  out << "<script>\n"
      << "const ADG_DATA = " << scriptSafe(adgJson) << ";\n\n"
      << "const ADG_LAYOUT_DATA = " << scriptSafe(adgLayoutJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n\n"
      << "const MAPPING_DATA = " << scriptSafe(mapJson) << ";\n"
      << "</script>\n\n";

  out << "<script>\n" << viz::D3_MIN_JS << "\n</script>\n\n";
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";
  out << "</body>\n</html>\n";

  return mlir::success();
}

} // namespace fcc
