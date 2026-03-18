#include "VizExporterInternal.h"

namespace fcc {
namespace viz_detail {

AutoLayoutComponent
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

double computeModuleMarginForAreaRatio(double contentW, double contentH,
                                       double areaRatio) {
  if (contentW <= 0.0 || contentH <= 0.0)
    return 24.0;
  double perimeterSum = contentW + contentH;
  double targetExtraArea = std::max(0.0, areaRatio) * contentW * contentH;
  double disc = perimeterSum * perimeterSum + 4.0 * targetExtraArea;
  double margin = (std::sqrt(std::max(0.0, disc)) - perimeterSum) / 4.0;
  return std::max(24.0, std::round(margin));
}

std::array<unsigned, 4> buildAutoLayoutPortSideCounts(unsigned count,
                                                      unsigned sideCount) {
  std::array<unsigned, 4> counts = {0, 0, 0, 0};
  for (unsigned idx = 0; idx < count; ++idx)
    counts[idx % sideCount] += 1;
  return counts;
}

std::array<unsigned, 2>
buildAutoLayoutPEPortSideCounts(unsigned count) {
  std::array<unsigned, 2> counts = {0, 0};
  for (unsigned idx = 0; idx < count; ++idx)
    counts[idx % 2] += 1;
  return counts;
}

AutoLayoutRoutePt
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

AutoLayoutRoutePt
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

AutoLayoutModuleBounds
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

AutoLayoutRoutePt
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

AutoLayoutRoutePt
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

namespace {

std::vector<AutoLayoutRoutePt>
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

std::vector<AutoLayoutRoutePt>
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

std::vector<AutoLayoutRoutePt>
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

} // anonymous namespace

std::string buildAutoLayoutJson(
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

std::string
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

std::string selectVizLayoutJson(mlir::ModuleOp adgModule,
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

} // namespace viz_detail
} // namespace fcc
