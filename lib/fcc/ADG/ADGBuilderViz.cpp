//===-- ADGBuilderViz.cpp - ADG Builder visualization JSON --------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//
//
// Generates visualization sidecar JSON with explicit component positions and
// pre-routed module-level edges.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilderDetail.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>

namespace fcc {
namespace adg {

using namespace detail;

static double computeModuleMarginForAreaRatio(double contentW, double contentH,
                                              double areaRatio) {
  if (contentW <= 0.0 || contentH <= 0.0)
    return 24.0;
  double sum = contentW + contentH;
  double extraArea = std::max(0.0, areaRatio) * contentW * contentH;
  double disc = sum * sum + 4.0 * extraArea;
  double margin = (std::sqrt(std::max(0.0, disc)) - sum) / 4.0;
  return std::max(24.0, std::round(margin));
}

// Generate visualization sidecar JSON with explicit component positions and
// pre-routed module-level edges.
std::string ADGBuilder::Impl::generateVizJson() const {
  struct BoxInfo {
    double centerX = 0.0;
    double centerY = 0.0;
    double width = 0.0;
    double height = 0.0;
    unsigned numInputs = 0;
    unsigned numOutputs = 0;
    bool valid = false;
  };
  struct RoutePt {
    double x = 0.0;
    double y = 0.0;
  };

  auto estimateFUBox = [&](const FUDef &fu) -> std::pair<double, double> {
    unsigned opCount = static_cast<unsigned>(
        std::max<size_t>(1, fu.ops.empty() ? (fu.rawBody.empty() ? 1 : 4)
                                           : fu.ops.size()));
    unsigned cols =
        std::max(1U, static_cast<unsigned>(std::ceil(std::sqrt(opCount))));
    unsigned rows = (opCount + cols - 1) / cols;
    double opW = opCount > 1 ? 62.0 : 56.0;
    double opH = 34.0;
    double channelGap = 32.0;
    double colGap = 36.0;
    double innerPadX = 14.0;
    double innerTop = 26.0;
    double innerBottom = 22.0;
    double rowWidth =
        static_cast<double>(cols) * opW +
        static_cast<double>(std::max(0U, cols - 1)) * colGap;
    double portSpanW = std::max(
        {opW,
         (std::max(0.0, static_cast<double>(fu.inputTypes.size()) - 1.0) *
              26.0) +
             opW,
         (std::max(0.0, static_cast<double>(fu.outputTypes.size()) - 1.0) *
              26.0) +
             opW});
    double innerW = std::max(104.0, std::max(rowWidth, portSpanW));
    double innerH = static_cast<double>(rows) * opH +
                    static_cast<double>(rows + 1) * channelGap;
    return {innerW + innerPadX * 2.0, innerTop + innerH + innerBottom};
  };

  auto computePEBox = [&](const PEDef &peDef) -> std::pair<double, double> {
    constexpr double kFuBoxMargin = 12.0;
    constexpr double kPeInnerPadX = 20.0;
    constexpr double kPeInnerPadY = 30.0;
    constexpr double kRowGap = 14.0;
    constexpr double kPeMinW = 200.0;
    constexpr double kPeMinH = 90.0;

    if (peDef.fuIndices.empty())
      return {kPeMinW, kPeMinH};

    std::vector<double> fuWidths;
    std::vector<double> fuHeights;
    fuWidths.reserve(peDef.fuIndices.size());
    fuHeights.reserve(peDef.fuIndices.size());
    for (unsigned fuIdx : peDef.fuIndices) {
      auto [boxW, boxH] = estimateFUBox(fuDefs[fuIdx]);
      fuWidths.push_back(boxW);
      fuHeights.push_back(boxH);
    }

    double bestW = kPeMinW;
    double bestH = kPeMinH;
    double bestScore = std::numeric_limits<double>::infinity();
    for (unsigned cols = 1; cols <= fuWidths.size(); ++cols) {
      double contentW = 0.0;
      double contentH = 0.0;
      unsigned start = 0;
      while (start < fuWidths.size()) {
        unsigned end =
            std::min<unsigned>(static_cast<unsigned>(fuWidths.size()), start + cols);
        double rowWidth = 0.0;
        double rowHeight = 0.0;
        for (unsigned i = start; i < end; ++i) {
          rowWidth += fuWidths[i];
          if (i > start)
            rowWidth += kFuBoxMargin;
          rowHeight = std::max(rowHeight, fuHeights[i]);
        }
        contentW = std::max(contentW, rowWidth);
        contentH += rowHeight;
        if (end < fuWidths.size())
          contentH += kRowGap;
        start = end;
      }

      double candidateW =
          std::max(kPeMinW, contentW + kPeInnerPadX * 2.0);
      double candidateH =
          std::max(kPeMinH, contentH + kPeInnerPadY * 2.0);
      double longSide = std::max(candidateW, candidateH);
      double shortSide = std::max(1.0, std::min(candidateW, candidateH));
      double aspectPenalty = longSide / shortSide;
      double areaPenalty = candidateW * candidateH;
      double score = (aspectPenalty - 1.0) * 1000.0 + areaPenalty * 0.0001;
      if (score < bestScore) {
        bestScore = score;
        bestW = candidateW;
        bestH = candidateH;
      }
    }

    return {bestW, bestH};
  };

  auto computePEWidth = [&](const PEDef &peDef) {
    return computePEBox(peDef).first;
  };
  auto computePEHeight = [&](const PEDef &peDef) {
    return computePEBox(peDef).second;
  };
  constexpr double kSwitchPortPitch = 24.0;
  constexpr double kSwitchMinSide = 84.0;
  auto buildPortSideCounts = [&](unsigned count, unsigned sideCount) {
    std::array<unsigned, 4> counts = {0, 0, 0, 0};
    for (unsigned idx = 0; idx < count; ++idx)
      counts[idx % sideCount] += 1;
    return counts;
  };
  auto buildPEPortSideCounts = [&](unsigned count) {
    std::array<unsigned, 2> counts = {0, 0};
    for (unsigned idx = 0; idx < count; ++idx)
      counts[idx % 2] += 1;
    return counts;
  };
  auto computeSWSide = [&](const SWDef &swDef) {
    std::array<unsigned, 4> inCounts =
        buildPortSideCounts(swDef.inputTypes.size(), 2);
    std::array<unsigned, 4> outCounts =
        buildPortSideCounts(swDef.outputTypes.size(), 2);
    unsigned maxSideSlots = 0;
    maxSideSlots = std::max(maxSideSlots, inCounts[0]);
    maxSideSlots = std::max(maxSideSlots, inCounts[1]);
    maxSideSlots = std::max(maxSideSlots, outCounts[0]);
    maxSideSlots = std::max(maxSideSlots, outCounts[1]);
    return std::max(kSwitchMinSide,
                    32.0 +
                        (static_cast<double>(std::max(1U, maxSideSlots)) + 1.0) *
                            kSwitchPortPitch);
  };

  auto estimateBoxInfo = [&](unsigned instIdx) -> BoxInfo {
    BoxInfo info;
    const auto &inst = instances[instIdx];
    switch (inst.kind) {
    case InstanceKind::PE: {
      const auto &peDef = peDefs[inst.defIdx];
      info.width = computePEWidth(peDef);
      info.height = computePEHeight(peDef);
      info.numInputs = peDef.inputTypes.size();
      info.numOutputs = peDef.outputTypes.size();
      break;
    }
    case InstanceKind::SW: {
      const auto &swDef = swDefs[inst.defIdx];
      double side = computeSWSide(swDef);
      info.width = side;
      info.height = side;
      info.numInputs = swDef.inputTypes.size();
      info.numOutputs = swDef.outputTypes.size();
      break;
    }
    case InstanceKind::Memory: {
      const auto &memDef = memoryDefs[inst.defIdx];
      info.width = 170.0;
      info.height = 80.0;
      info.numInputs = memDef.ldPorts + memDef.stPorts * 2;
      info.numOutputs =
          memDef.ldPorts + memDef.stPorts + memDef.ldPorts + (memDef.isPrivate ? 0 : 1);
      break;
    }
    case InstanceKind::ExtMem: {
      const auto &memDef = extMemDefs[inst.defIdx];
      info.width = 170.0;
      info.height = 80.0;
      info.numInputs = 1 + memDef.ldPorts + memDef.stPorts * 2;
      info.numOutputs = memDef.ldPorts + memDef.stPorts + memDef.ldPorts;
      break;
    }
    case InstanceKind::FIFO:
      info.width = 100.0;
      info.height = 56.0;
      info.numInputs = 1;
      info.numOutputs = 1;
      break;
    case InstanceKind::AddTag:
    case InstanceKind::MapTag:
    case InstanceKind::DelTag:
      info.width = 92.0;
      info.height = 52.0;
      info.numInputs = 1;
      info.numOutputs = 1;
      break;
    }
    info.valid = true;
    return info;
  };

  auto computeEffectivePlacements = [&]() {
    std::map<unsigned, VizPlacement> placements = vizPlacements;

    auto packUnplacedInstances = [&](std::map<unsigned, VizPlacement> &packed) {
      double placedMinX = 0.0;
      double placedMaxY = 0.0;
      bool havePlaced = false;
      for (const auto &[instIdx, placement] : packed) {
        BoxInfo info = estimateBoxInfo(instIdx);
        if (!info.valid)
          continue;
        double boxMinX = placement.centerX - info.width / 2.0;
        double boxMaxY = placement.centerY + info.height / 2.0;
        if (!havePlaced) {
          placedMinX = boxMinX;
          placedMaxY = boxMaxY;
          havePlaced = true;
        } else {
          placedMinX = std::min(placedMinX, boxMinX);
          placedMaxY = std::max(placedMaxY, boxMaxY);
        }
      }

      constexpr double kAutoGapX = 88.0;
      constexpr double kAutoGapY = 108.0;
      constexpr double kAutoWrapWidth = 3600.0;
      double startX = havePlaced ? placedMinX : 120.0;
      double cursorX = startX;
      double cursorY = havePlaced ? placedMaxY + 160.0 : 120.0;
      double rowHeight = 0.0;
      int packedRow = havePlaced ? 1000 : 0;
      int packedCol = 0;
      for (unsigned instIdx = 0; instIdx < instances.size(); ++instIdx) {
        if (packed.count(instIdx))
          continue;
        BoxInfo info = estimateBoxInfo(instIdx);
        if (!info.valid)
          continue;
        if (cursorX > startX &&
            cursorX + info.width > startX + kAutoWrapWidth) {
          cursorX = startX;
          cursorY += rowHeight + kAutoGapY;
          rowHeight = 0.0;
          ++packedRow;
          packedCol = 0;
        }
        packed[instIdx] = {cursorX + info.width / 2.0,
                           cursorY + info.height / 2.0, packedRow,
                           packedCol++};
        cursorX += info.width + kAutoGapX;
        rowHeight = std::max(rowHeight, info.height);
      }
    };

    auto computeNeatoPlacements =
        [&]() -> std::optional<std::map<unsigned, VizPlacement>> {
      if (!vizPlacements.empty())
        return std::nullopt;

      std::vector<unsigned> autoInsts;
      for (unsigned instIdx = 0; instIdx < instances.size(); ++instIdx) {
        if (placements.count(instIdx))
          continue;
        BoxInfo info = estimateBoxInfo(instIdx);
        if (info.valid)
          autoInsts.push_back(instIdx);
      }
      if (autoInsts.size() < 2)
        return std::nullopt;

      auto neatoPath = llvm::sys::findProgramByName("neato");
      if (!neatoPath)
        return std::nullopt;

      std::map<std::pair<unsigned, unsigned>, unsigned> edgeMultiplicity;
      for (const auto &conn : connections) {
        if (conn.srcInst == conn.dstInst)
          continue;
        auto key = std::make_pair(std::min(conn.srcInst, conn.dstInst),
                                  std::max(conn.srcInst, conn.dstInst));
        edgeMultiplicity[key] += 1;
      }

      llvm::SmallString<128> dotPath;
      llvm::SmallString<128> plainPath;
      int dotFd = -1;
      int plainFd = -1;
      if (llvm::sys::fs::createTemporaryFile("fcc_adg_layout", "dot", dotFd,
                                             dotPath))
        return std::nullopt;
      if (llvm::sys::fs::createTemporaryFile("fcc_adg_layout", "plain",
                                             plainFd, plainPath)) {
        llvm::sys::fs::remove(dotPath);
        return std::nullopt;
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
        for (unsigned instIdx : autoInsts) {
          BoxInfo info = estimateBoxInfo(instIdx);
          if (!info.valid)
            continue;
          double widthIn = std::max(0.25, (info.width + kNodePad) / kGraphvizUnit);
          double heightIn =
              std::max(0.25, (info.height + kNodePad) / kGraphvizUnit);
          dotOS << "  n" << instIdx << " [width=" << dotNum(widthIn)
                << ", height=" << dotNum(heightIn) << "];\n";
        }
        for (const auto &[key, multiplicity] : edgeMultiplicity) {
          if (!std::binary_search(autoInsts.begin(), autoInsts.end(), key.first) ||
              !std::binary_search(autoInsts.begin(), autoInsts.end(), key.second))
            continue;
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
      int rc = llvm::sys::ExecuteAndWait(neatoPathStr, args, std::nullopt, {},
                                         0, 0, &errMsg, &execFailed);
      if (rc != 0 || execFailed) {
        cleanupTempFiles();
        return std::nullopt;
      }

      auto plainBuf = llvm::MemoryBuffer::getFile(plainPath);
      if (!plainBuf) {
        cleanupTempFiles();
        return std::nullopt;
      }

      std::map<unsigned, VizPlacement> neatoPlacements = placements;
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

        unsigned instIdx = 0;
        try {
          instIdx = static_cast<unsigned>(std::stoul(nodeName.substr(1)));
        } catch (...) {
          continue;
        }

        constexpr double kGraphvizUnit = 72.0;
        neatoPlacements[instIdx] = {xIn * kGraphvizUnit, yIn * kGraphvizUnit,
                                    -1, -1};
      }
      cleanupTempFiles();

      bool haveAutoPlacement = false;
      double minLeft = 0.0;
      double minTop = 0.0;
      for (unsigned instIdx : autoInsts) {
        auto it = neatoPlacements.find(instIdx);
        if (it == neatoPlacements.end())
          continue;
        BoxInfo info = estimateBoxInfo(instIdx);
        if (!info.valid)
          continue;
        double left = it->second.centerX - info.width / 2.0;
        double top = it->second.centerY - info.height / 2.0;
        if (!haveAutoPlacement) {
          minLeft = left;
          minTop = top;
          haveAutoPlacement = true;
        } else {
          minLeft = std::min(minLeft, left);
          minTop = std::min(minTop, top);
        }
      }
      if (!haveAutoPlacement)
        return std::nullopt;

      double shiftX = 120.0 - minLeft;
      double shiftY = 120.0 - minTop;
      for (unsigned instIdx : autoInsts) {
        auto it = neatoPlacements.find(instIdx);
        if (it == neatoPlacements.end())
          continue;
        it->second.centerX += shiftX;
        it->second.centerY += shiftY;
      }
      return neatoPlacements;
    };

    if (auto neatoPlacements = computeNeatoPlacements())
      return *neatoPlacements;

    auto resolvePlacementOverlaps =
        [&](std::map<unsigned, VizPlacement> &resolvedPlacements) {
          constexpr double kMinGap = 48.0;
          for (unsigned iter = 0; iter < 256; ++iter) {
            bool moved = false;
            for (auto itA = resolvedPlacements.begin();
                 itA != resolvedPlacements.end(); ++itA) {
              auto itB = itA;
              ++itB;
              for (; itB != resolvedPlacements.end(); ++itB) {
                BoxInfo boxA = estimateBoxInfo(itA->first);
                BoxInfo boxB = estimateBoxInfo(itB->first);
                if (!boxA.valid || !boxB.valid)
                  continue;
                double dx = itB->second.centerX - itA->second.centerX;
                double dy = itB->second.centerY - itA->second.centerY;
                double overlapX =
                    (boxA.width + boxB.width) / 2.0 + kMinGap - std::abs(dx);
                double overlapY =
                    (boxA.height + boxB.height) / 2.0 + kMinGap - std::abs(dy);
                if (overlapX <= 0.0 || overlapY <= 0.0)
                  continue;

                moved = true;
                if (std::abs(dx) < 1e-3)
                  dx = (itA->first < itB->first) ? -1.0 : 1.0;
                if (std::abs(dy) < 1e-3)
                  dy = (itA->first < itB->first) ? -1.0 : 1.0;

                if (overlapX < overlapY) {
                  double shift = overlapX / 2.0 + 1.0;
                  double sign = dx < 0.0 ? -1.0 : 1.0;
                  itA->second.centerX -= sign * shift;
                  itB->second.centerX += sign * shift;
                } else {
                  double shift = overlapY / 2.0 + 1.0;
                  double sign = dy < 0.0 ? -1.0 : 1.0;
                  itA->second.centerY -= sign * shift;
                  itB->second.centerY += sign * shift;
                }
              }
            }
            if (!moved)
              break;
          }
        };

    packUnplacedInstances(placements);
    resolvePlacementOverlaps(placements);
    return placements;
  };

  std::map<unsigned, VizPlacement> placements = computeEffectivePlacements();

  auto computeBoxInfo = [&](unsigned instIdx) -> BoxInfo {
    BoxInfo info = estimateBoxInfo(instIdx);
    if (!info.valid)
      return info;
    auto it = placements.find(instIdx);
    if (it == placements.end()) {
      info.valid = false;
      return info;
    }
    info.centerX = it->second.centerX;
    info.centerY = it->second.centerY;
    return info;
  };

  struct ModuleBounds {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    bool valid = false;
  };

  auto computeModuleBounds = [&]() -> ModuleBounds {
    ModuleBounds bounds;
    bool haveContent = false;
    double actualMinX = 0.0;
    double actualMinY = 0.0;
    double actualMaxX = 0.0;
    double actualMaxY = 0.0;
    for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
      BoxInfo box = computeBoxInfo(static_cast<unsigned>(instIdx));
      if (!box.valid)
        continue;
      double boxMinX = box.centerX - box.width / 2.0;
      double boxMinY = box.centerY - box.height / 2.0;
      double boxMaxX = box.centerX + box.width / 2.0;
      double boxMaxY = box.centerY + box.height / 2.0;
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
    bounds.y = actualMinY - margin - 28.0;
    bounds.w = contentW + margin * 2.0;
    bounds.h = contentH + margin * 2.0 + 28.0;
    bounds.valid = true;
    return bounds;
  };

  auto computeInputPortPos = [&](const BoxInfo &box, const InstanceDef &inst,
                                 unsigned portIdx) -> RoutePt {
    RoutePt pt;
    if (inst.kind == InstanceKind::PE) {
      const auto &peDef = peDefs[inst.defIdx];
      std::array<unsigned, 2> sideCounts =
          buildPEPortSideCounts(peDef.inputTypes.size());
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(sideCounts[sideIdx] + 1);
      if (sideIdx == 0) {
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY - box.height / 2.0;
      } else {
        pt.x = box.centerX - box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
      }
    } else if (inst.kind == InstanceKind::SW) {
      const auto &swDef = swDefs[inst.defIdx];
      std::array<unsigned, 4> inCounts =
          buildPortSideCounts(swDef.inputTypes.size(), 2);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      unsigned slotCount = inCounts[sideIdx];
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(slotCount + 1);
      switch (sideIdx) {
      case 0:
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY - box.height / 2.0;
        break;
      default:
        pt.x = box.centerX - box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
        break;
      }
    } else {
      pt.x = box.centerX - box.width / 2.0;
      pt.y = box.centerY - box.height / 2.0 + 16.0 +
             (box.height - 32.0) * (static_cast<double>(portIdx + 1) /
                                    static_cast<double>(box.numInputs + 1));
    }
    return pt;
  };

  auto computeOutputPortPos = [&](const BoxInfo &box, const InstanceDef &inst,
                                  unsigned portIdx) -> RoutePt {
    RoutePt pt;
    if (inst.kind == InstanceKind::PE) {
      const auto &peDef = peDefs[inst.defIdx];
      std::array<unsigned, 2> sideCounts =
          buildPEPortSideCounts(peDef.outputTypes.size());
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(sideCounts[sideIdx] + 1);
      if (sideIdx == 0) {
        pt.x = box.centerX + box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
      } else {
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY + box.height / 2.0;
      }
    } else if (inst.kind == InstanceKind::SW) {
      const auto &swDef = swDefs[inst.defIdx];
      std::array<unsigned, 4> outCounts =
          buildPortSideCounts(swDef.outputTypes.size(), 2);
      unsigned sideIdx = portIdx % 2;
      unsigned localIdx = portIdx / 2;
      unsigned slotCount = outCounts[sideIdx];
      double ratio = static_cast<double>(localIdx + 1) /
                     static_cast<double>(slotCount + 1);
      switch (sideIdx) {
      case 0:
        pt.x = box.centerX + box.width / 2.0;
        pt.y = (box.centerY - box.height / 2.0) + box.height * ratio;
        break;
      default:
        pt.x = (box.centerX - box.width / 2.0) + box.width * ratio;
        pt.y = box.centerY + box.height / 2.0;
        break;
      }
    } else {
      pt.x = box.centerX + box.width / 2.0;
      pt.y = box.centerY - box.height / 2.0 + 16.0 +
             (box.height - 32.0) * (static_cast<double>(portIdx + 1) /
                                    static_cast<double>(box.numOutputs + 1));
    }
    return pt;
  };

  ModuleBounds moduleBounds = computeModuleBounds();
  auto computeModuleInputPortPos = [&](unsigned portIdx) -> RoutePt {
    RoutePt pt;
    pt.x = moduleBounds.x + moduleBounds.w *
           (static_cast<double>(portIdx + 1) /
            static_cast<double>(scalarInputs.size() + 1));
    pt.y = moduleBounds.y;
    return pt;
  };
  auto computeModuleOutputPortPos = [&](unsigned portIdx) -> RoutePt {
    RoutePt pt;
    pt.x = moduleBounds.x + moduleBounds.w *
           (static_cast<double>(portIdx + 1) /
            static_cast<double>(scalarOutputs.size() + 1));
    pt.y = moduleBounds.y + moduleBounds.h;
    return pt;
  };

  auto routeModuleInputConnection = [&](unsigned scalarIdx, unsigned dstInstIdx,
                                        unsigned dstPortIdx) -> std::vector<RoutePt> {
    if (!moduleBounds.valid)
      return {};
    BoxInfo dstBox = computeBoxInfo(dstInstIdx);
    if (!dstBox.valid)
      return {};
    const auto &dstInst = instances[dstInstIdx];
    RoutePt srcPort = computeModuleInputPortPos(scalarIdx);
    RoutePt dstPort = computeInputPortPos(dstBox, dstInst, dstPortIdx);
    const int signedLane = static_cast<int>(scalarIdx % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double entryY = moduleBounds.y + 42.0 + std::abs(laneOffset);
    const double dstApproachX = dstPort.x - (24.0 + std::abs(laneOffset));
    std::vector<RoutePt> pts;
    pts.push_back({srcPort.x, entryY});
    if (std::abs(srcPort.x - dstApproachX) > 0.5)
      pts.push_back({dstApproachX, entryY});
    if (std::abs(entryY - dstPort.y) > 0.5)
      pts.push_back({dstApproachX, dstPort.y});
    return pts;
  };

  auto routeModuleOutputConnection = [&](unsigned srcInstIdx, unsigned srcPortIdx,
                                         unsigned scalarOutIdx) -> std::vector<RoutePt> {
    if (!moduleBounds.valid)
      return {};
    BoxInfo srcBox = computeBoxInfo(srcInstIdx);
    if (!srcBox.valid)
      return {};
    const auto &srcInst = instances[srcInstIdx];
    RoutePt srcPort = computeOutputPortPos(srcBox, srcInst, srcPortIdx);
    RoutePt dstPort = computeModuleOutputPortPos(scalarOutIdx);
    const int signedLane = static_cast<int>(scalarOutIdx % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double exitX = srcPort.x + (24.0 + std::abs(laneOffset));
    const double corridorY =
        moduleBounds.y + moduleBounds.h - 42.0 - std::abs(laneOffset);
    std::vector<RoutePt> pts;
    pts.push_back({exitX, srcPort.y});
    if (std::abs(srcPort.y - corridorY) > 0.5)
      pts.push_back({exitX, corridorY});
    if (std::abs(exitX - dstPort.x) > 0.5)
      pts.push_back({dstPort.x, corridorY});
    return pts;
  };

  auto routeConnection = [&](const Connection &conn, unsigned routeOrdinal)
      -> std::vector<RoutePt> {
    BoxInfo srcBox = computeBoxInfo(conn.srcInst);
    BoxInfo dstBox = computeBoxInfo(conn.dstInst);
    if (!srcBox.valid || !dstBox.valid)
      return {};

    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    RoutePt srcPort = computeOutputPortPos(srcBox, srcInst, conn.srcPort);
    RoutePt dstPort = computeInputPortPos(dstBox, dstInst, conn.dstPort);

    const int signedLane = static_cast<int>(routeOrdinal % 5) - 2;
    const double laneOffset = static_cast<double>(signedLane) * 7.0;
    const double margin = 22.0 + std::abs(laneOffset) * 0.5;
    const double srcRight = srcBox.centerX + srcBox.width / 2.0;
    const double dstLeft = dstBox.centerX - dstBox.width / 2.0;
    const double srcTop = srcBox.centerY - srcBox.height / 2.0;
    const double srcBottom = srcBox.centerY + srcBox.height / 2.0;
    const double dstTop = dstBox.centerY - dstBox.height / 2.0;
    const double dstBottom = dstBox.centerY + dstBox.height / 2.0;

    std::vector<RoutePt> pts;
    if (srcBox.centerX + 1.0 < dstBox.centerX) {
      double corridorX = (srcRight + dstLeft) / 2.0 + laneOffset;
      pts.push_back({corridorX, srcPort.y});
      if (std::abs(srcPort.y - dstPort.y) > 0.5)
        pts.push_back({corridorX, dstPort.y});
      return pts;
    }

    double srcExitX = srcRight + margin;
    double dstEntryX = dstLeft - margin;
    bool routeAbove = srcBox.centerY <= dstBox.centerY;
    double corridorY =
        routeAbove ? std::min(srcTop, dstTop) - margin - std::abs(laneOffset)
                   : std::max(srcBottom, dstBottom) + margin + std::abs(laneOffset);
    pts.push_back({srcExitX, srcPort.y});
    pts.push_back({srcExitX, corridorY});
    pts.push_back({dstEntryX, corridorY});
    pts.push_back({dstEntryX, dstPort.y});
    return pts;
  };

  std::ostringstream os;
  os << "{\n"
     << "  \"version\": 1,\n"
     << "  \"components\": [\n";

  bool first = true;
  for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
    auto it = placements.find(static_cast<unsigned>(instIdx));
    if (it == placements.end())
      continue;

    if (!first)
      os << ",\n";
    first = false;

    const auto &inst = instances[instIdx];
    const auto &placement = it->second;
    const char *kindName = "instance";
    switch (inst.kind) {
    case InstanceKind::PE:
      kindName = peDefs[inst.defIdx].temporal ? "temporal_pe" : "spatial_pe";
      break;
    case InstanceKind::SW:
      kindName = swDefs[inst.defIdx].temporal ? "temporal_sw" : "spatial_sw";
      break;
    case InstanceKind::Memory:
      kindName = "memory";
      break;
    case InstanceKind::ExtMem:
      kindName = "extmemory";
      break;
    case InstanceKind::FIFO:
      kindName = "fifo";
      break;
    case InstanceKind::AddTag:
      kindName = "add_tag";
      break;
    case InstanceKind::MapTag:
      kindName = "map_tag";
      break;
    case InstanceKind::DelTag:
      kindName = "del_tag";
      break;
    }

    os << "    {\"name\": \"" << inst.name << "\""
       << ", \"kind\": \"" << kindName << "\""
       << ", \"center_x\": " << placement.centerX
       << ", \"center_y\": " << placement.centerY;
    if (placement.gridRow >= 0)
      os << ", \"grid_row\": " << placement.gridRow;
    if (placement.gridCol >= 0)
      os << ", \"grid_col\": " << placement.gridCol;
    os << "}";
  }

  os << "\n  ],\n"
     << "  \"routes\": [\n";

  bool firstRoute = true;
  std::map<std::pair<unsigned, unsigned>, unsigned> nextPairOrdinal;
  auto emitRouteRecord = [&](llvm::StringRef fromName, unsigned fromPort,
                             llvm::StringRef toName, unsigned toPort,
                             const std::vector<RoutePt> &pts) {
    if (!firstRoute)
      os << ",\n";
    firstRoute = false;
    os << "    {\"from\": \"" << fromName.str() << "\""
       << ", \"from_port\": " << fromPort << ", \"to\": \"" << toName.str()
       << "\"" << ", \"to_port\": " << toPort << ", \"points\": [";
    for (size_t ptIdx = 0; ptIdx < pts.size(); ++ptIdx) {
      if (ptIdx > 0)
        os << ", ";
      os << "{\"x\": " << pts[ptIdx].x << ", \"y\": " << pts[ptIdx].y
         << "}";
    }
    os << "]}";
  };
  for (size_t connIdx = 0; connIdx < connections.size(); ++connIdx) {
    const auto &conn = connections[connIdx];
    if (placements.find(conn.srcInst) == placements.end() ||
        placements.find(conn.dstInst) == placements.end())
      continue;
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    auto pairKey =
        std::make_pair(std::min(conn.srcInst, conn.dstInst),
                       std::max(conn.srcInst, conn.dstInst));
    unsigned pairOrdinal = nextPairOrdinal[pairKey]++;
    std::vector<RoutePt> pts = routeConnection(conn, pairOrdinal);

    emitRouteRecord(srcInst.name, conn.srcPort, dstInst.name, conn.dstPort,
                    pts);
  }
  for (const auto &sc : scalarToInstConns) {
    if (placements.find(sc.dstInst) == placements.end())
      continue;
    std::vector<RoutePt> pts =
        routeModuleInputConnection(sc.scalarIdx, sc.dstInst, sc.dstPort);
    emitRouteRecord("module_in", sc.scalarIdx, instances[sc.dstInst].name,
                    sc.dstPort, pts);
  }
  for (const auto &ic : instToScalarConns) {
    if (placements.find(ic.srcInst) == placements.end())
      continue;
    std::vector<RoutePt> pts =
        routeModuleOutputConnection(ic.srcInst, ic.srcPort, ic.scalarOutputIdx);
    emitRouteRecord(instances[ic.srcInst].name, ic.srcPort, "module_out",
                    ic.scalarOutputIdx, pts);
  }
  os << "\n  ]\n"
     << "}\n";
  return os.str();
}

} // namespace adg
} // namespace fcc
