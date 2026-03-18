//===-- ADGBuilderTopology.cpp - ADG Builder topology helpers ----*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "fcc/ADG/ADGBuilderDetail.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace fcc {
namespace adg {

using namespace detail;

MeshResult ADGBuilder::buildMesh(unsigned rows, unsigned cols, PEHandle pe,
                                 SWHandle sw) {
  return buildMesh(rows, cols, [pe](unsigned, unsigned) { return pe; }, sw);
}

MeshResult ADGBuilder::buildMesh(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    SWHandle sw) {
  assert(!impl_->swDefs[sw.id].temporal &&
         "buildMesh expects a spatial_sw template");
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows, std::vector<InstanceHandle>(cols));

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string swName = "sw_" + std::to_string(r) + "_" + std::to_string(c);
      result.swGrid[r][c] = instantiateSW(sw, swName);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string peName = "pe_" + std::to_string(r) + "_" + std::to_string(c);
      assert(!impl_->peDefs[peSelector(r, c).id].temporal &&
             "buildMesh expects spatial_pe templates");
      result.peGrid[r][c] = instantiatePE(peSelector(r, c), peName);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      const auto &peDef = impl_->peDefs[peSelector(r, c).id];
      for (unsigned p = 0; p < peDef.outputTypes.size(); ++p)
        connect(result.peGrid[r][c], p, result.swGrid[r][c], p);
      for (unsigned p = 0; p < peDef.inputTypes.size(); ++p)
        connect(result.swGrid[r][c], p, result.peGrid[r][c], p);

      unsigned peOut = peDef.outputTypes.size();
      unsigned peIn = peDef.inputTypes.size();
      unsigned northR = (r > 0) ? r - 1 : rows - 1;
      connect(result.swGrid[r][c], peIn + 0, result.swGrid[northR][c],
              peOut + 1);
      unsigned southR = (r + 1 < rows) ? r + 1 : 0;
      connect(result.swGrid[r][c], peIn + 1, result.swGrid[southR][c],
              peOut + 0);
      unsigned eastC = (c + 1 < cols) ? c + 1 : 0;
      connect(result.swGrid[r][c], peIn + 2, result.swGrid[r][eastC],
              peOut + 3);
      unsigned westC = (c > 0) ? c - 1 : cols - 1;
      connect(result.swGrid[r][c], peIn + 3, result.swGrid[r][westC],
              peOut + 2);
    }
  }

  return result;
}

MeshResult ADGBuilder::buildLatticeMesh(unsigned rows, unsigned cols,
                                        PEHandle pe, int decomposableBits) {
  LatticeMeshOptions options;
  options.decomposableBits = decomposableBits;
  return buildLatticeMesh(rows, cols, [pe](unsigned, unsigned) { return pe; },
                          options);
}

MeshResult ADGBuilder::buildLatticeMesh(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    int decomposableBits) {
  LatticeMeshOptions options;
  options.decomposableBits = decomposableBits;
  return buildLatticeMesh(rows, cols, peSelector, options);
}

MeshResult ADGBuilder::buildLatticeMesh(unsigned rows, unsigned cols,
                                        PEHandle pe,
                                        const LatticeMeshOptions &options) {
  return buildLatticeMesh(rows, cols, [pe](unsigned, unsigned) { return pe; },
                          options);
}

MeshResult ADGBuilder::buildLatticeMesh(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    const LatticeMeshOptions &options) {
  assert(rows >= 1 && cols >= 1 &&
         "buildLatticeMesh expects at least one row and one column");
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows, std::vector<InstanceHandle>(cols));

  std::vector<std::vector<PEHandle>> selectedPEs(
      rows, std::vector<PEHandle>(cols));
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c)
      selectedPEs[r][c] = peSelector(r, c);
  }

  const auto &firstPEDef = impl_->peDefs[selectedPEs[0][0].id];
  assert(!firstPEDef.temporal &&
         "buildLatticeMesh expects spatial_pe templates");
  auto firstInWidth =
      inferUniformBitsWidth(firstPEDef.inputTypes, firstPEDef.inputTypes.size());
  auto firstOutWidth = inferUniformBitsWidth(firstPEDef.outputTypes,
                                             firstPEDef.outputTypes.size());
  assert(firstInWidth && firstOutWidth && *firstInWidth == *firstOutWidth &&
         "buildLatticeMesh expects uniform !fabric.bits<N> PE boundary ports");

  const unsigned peInputCount = firstPEDef.inputTypes.size();
  const unsigned peOutputCount = firstPEDef.outputTypes.size();

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      const auto &cellPEDef = impl_->peDefs[selectedPEs[r][c].id];
      assert(!cellPEDef.temporal &&
             "buildLatticeMesh expects spatial_pe templates");
      assert(cellPEDef.inputTypes.size() == peInputCount &&
             cellPEDef.outputTypes.size() == peOutputCount &&
             "buildLatticeMesh expects all PE templates to share boundary arity");
      auto cellInWidth =
          inferUniformBitsWidth(cellPEDef.inputTypes, cellPEDef.inputTypes.size());
      auto cellOutWidth = inferUniformBitsWidth(cellPEDef.outputTypes,
                                                cellPEDef.outputTypes.size());
      assert(cellInWidth && cellOutWidth && *cellInWidth == *cellOutWidth &&
             *cellInWidth == *firstInWidth &&
             "buildLatticeMesh expects all PE templates to share one uniform !fabric.bits<N> boundary type");
    }
  }

  std::map<std::pair<unsigned, unsigned>, SWHandle> switchTemplateCache;
  auto makeSwitchTemplate = [&](unsigned numInputs,
                                unsigned numOutputs) -> SWHandle {
    auto key = std::make_pair(numInputs, numOutputs);
    auto it = switchTemplateCache.find(key);
    if (it != switchTemplateCache.end())
      return it->second;

    std::vector<unsigned> inputWidths(numInputs, *firstInWidth);
    std::vector<unsigned> outputWidths(numOutputs, *firstInWidth);
    std::vector<std::vector<bool>> fullCrossbar(
        numOutputs, std::vector<bool>(numInputs, true));
    std::string name = "__lattice_sw_" + std::to_string(numInputs) + "x" +
                       std::to_string(numOutputs) + "_" +
                       std::to_string(impl_->swDefs.size());
    auto handle = defineSpatialSW(name, inputWidths, outputWidths, fullCrossbar,
                                  options.decomposableBits);
    switchTemplateCache[key] = handle;
    return handle;
  };

  auto switchDegree = [&](unsigned r, unsigned c) -> unsigned {
    unsigned degree = 0;
    if (r > 0)
      degree++;
    if (r + 1 < rows)
      degree++;
    if (c > 0)
      degree++;
    if (c + 1 < cols)
      degree++;
    return degree;
  };

  auto estimateSwitchSide = [&](unsigned numInputs, unsigned numOutputs) {
    unsigned maxSideSlots =
        std::max((numInputs + 1) / 2, (numOutputs + 1) / 2);
    return std::max(84.0, 32.0 + (static_cast<double>(maxSideSlots) + 1.0) *
                                     24.0);
  };
  const double approxFuBoxW = 140.0;
  const double approxFuGap = 12.0;
  const double approxPEPadX = 60.0;
  double approxPEBoxW = 200.0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      const auto &cellPEDef = impl_->peDefs[selectedPEs[r][c].id];
      approxPEBoxW = std::max(
          approxPEBoxW,
          std::max(200.0,
                   cellPEDef.fuIndices.size() * approxFuBoxW +
                       std::max(0.0,
                                static_cast<double>(cellPEDef.fuIndices.size()) -
                                    1.0) *
                           approxFuGap +
                       approxPEPadX));
    }
  }
  const double approxPEBoxH = 200.0;
  double maxSwitchBox = 84.0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      unsigned degree = switchDegree(r, c);
      unsigned extraInputs =
          (r == 0 && c == 0) ? options.topLeftExtraInputs : 0;
      unsigned extraOutputs =
          (r + 1 == rows && c + 1 == cols) ? options.bottomRightExtraOutputs : 0;
      maxSwitchBox =
          std::max(maxSwitchBox,
                   estimateSwitchSide(peOutputCount + degree + extraInputs,
                                      peInputCount + degree + extraOutputs));
    }
  }
  const double cellGapX = 56.0;
  const double cellGapY = 72.0;
  const double cellStepX = maxSwitchBox + approxPEBoxW + cellGapX;
  const double cellStepY = std::max(maxSwitchBox, approxPEBoxH) + cellGapY;
  const double originX = maxSwitchBox / 2.0 + 100.0;
  const double originY = std::max(maxSwitchBox, approxPEBoxH) / 2.0 + 100.0;
  const double peOffsetX =
      maxSwitchBox / 2.0 + cellGapX / 2.0 + approxPEBoxW / 2.0;

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      unsigned degree = switchDegree(r, c);
      unsigned extraInputs =
          (r == 0 && c == 0) ? options.topLeftExtraInputs : 0;
      unsigned extraOutputs =
          (r + 1 == rows && c + 1 == cols) ? options.bottomRightExtraOutputs : 0;
      auto swHandle = makeSwitchTemplate(peOutputCount + degree + extraInputs,
                                         peInputCount + degree + extraOutputs);
      std::string swName = "sw_" + std::to_string(r) + "_" + std::to_string(c);
      auto swInst = instantiateSW(swHandle, swName);
      result.swGrid[r][c] = swInst;
      setInstanceVizPosition(swInst, originX + c * cellStepX,
                             originY + r * cellStepY,
                             r, c);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string peName = "pe_" + std::to_string(r) + "_" + std::to_string(c);
      auto peInst = instantiatePE(selectedPEs[r][c], peName);
      result.peGrid[r][c] = peInst;
      setInstanceVizPosition(peInst, originX + c * cellStepX + peOffsetX,
                             originY + r * cellStepY, r, c);
    }
  }

  std::map<std::pair<unsigned, unsigned>, unsigned> nextSwitchInput;
  std::map<std::pair<unsigned, unsigned>, unsigned> nextSwitchOutput;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      auto key = std::make_pair(r, c);
      nextSwitchInput[key] = peOutputCount;
      nextSwitchOutput[key] = peInputCount;
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      auto peInst = result.peGrid[r][c];
      auto swInst = result.swGrid[r][c];
      for (unsigned p = 0; p < peOutputCount; ++p)
        connect(peInst, p, swInst, p);
      for (unsigned p = 0; p < peInputCount; ++p)
        connect(swInst, p, peInst, p);
    }
  }

  auto connectSwitchBidirectional = [&](unsigned r0, unsigned c0, unsigned r1,
                                        unsigned c1) {
    auto key0 = std::make_pair(r0, c0);
    auto key1 = std::make_pair(r1, c1);
    unsigned dstInput0 = nextSwitchInput[key1]++;
    unsigned dstInput1 = nextSwitchInput[key0]++;
    unsigned srcOutput0 = nextSwitchOutput[key0]++;
    unsigned srcOutput1 = nextSwitchOutput[key1]++;
    connect(result.swGrid[r0][c0], srcOutput0, result.swGrid[r1][c1], dstInput0);
    connect(result.swGrid[r1][c1], srcOutput1, result.swGrid[r0][c0], dstInput1);
  };

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c + 1 < cols; ++c)
      connectSwitchBidirectional(r, c, r, c + 1);
  }
  for (unsigned r = 0; r + 1 < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c)
      connectSwitchBidirectional(r, c, r + 1, c);
  }

  if (options.topLeftExtraInputs > 0) {
    unsigned topLeftDegree = switchDegree(0, 0);
    auto swInst = result.swGrid[0][0];
    for (unsigned idx = 0; idx < options.topLeftExtraInputs; ++idx)
      result.ingressPorts.push_back({swInst, peOutputCount + topLeftDegree + idx});
  }
  if (options.bottomRightExtraOutputs > 0) {
    unsigned bottomRightDegree = switchDegree(rows - 1, cols - 1);
    auto swInst = result.swGrid[rows - 1][cols - 1];
    for (unsigned idx = 0; idx < options.bottomRightExtraOutputs; ++idx)
      result.egressPorts.push_back(
          {swInst, peInputCount + bottomRightDegree + idx});
  }

  return result;
}

MeshResult ADGBuilder::buildChessMesh(unsigned rows, unsigned cols, PEHandle pe,
                                      int decomposableBits,
                                      unsigned topLeftExtraInputs,
                                      unsigned bottomRightExtraOutputs) {
  ChessMeshOptions options;
  options.decomposableBits = decomposableBits;
  options.topLeftExtraInputs = topLeftExtraInputs;
  options.bottomRightExtraOutputs = bottomRightExtraOutputs;
  return buildChessMesh(rows, cols, [pe](unsigned, unsigned) { return pe; },
                        options);
}

MeshResult ADGBuilder::buildChessMesh(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    const ChessMeshOptions &options) {
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows + 1, std::vector<InstanceHandle>(cols + 1));

  std::vector<std::vector<PEHandle>> selectedPEs(
      rows, std::vector<PEHandle>(cols));
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c)
      selectedPEs[r][c] = peSelector(r, c);
  }

  const auto &firstPEDef = impl_->peDefs[selectedPEs[0][0].id];
  assert(!firstPEDef.temporal &&
         "buildChessMesh expects spatial_pe templates");
  assert(firstPEDef.inputTypes.size() >= 4 &&
         "buildChessMesh expects at least four PE input ports");
  assert(firstPEDef.outputTypes.size() >= 4 &&
         "buildChessMesh expects at least four PE output ports");
  auto firstWidth = inferUniformBitsWidth(firstPEDef.inputTypes, 4);
  assert(firstWidth &&
         "buildChessMesh expects the first four PE input ports to be uniform !fabric.bits<N>");
  auto firstOutWidth = inferUniformBitsWidth(firstPEDef.outputTypes, 4);
  assert(firstOutWidth && *firstOutWidth == *firstWidth &&
         "buildChessMesh expects the first four PE output ports to share the same !fabric.bits<N> type");
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      const auto &cellPEDef = impl_->peDefs[selectedPEs[r][c].id];
      assert(!cellPEDef.temporal &&
             "buildChessMesh expects spatial_pe templates");
      assert(cellPEDef.inputTypes.size() >= 4 &&
             "buildChessMesh expects at least four PE input ports");
      assert(cellPEDef.outputTypes.size() >= 4 &&
             "buildChessMesh expects at least four PE output ports");
      auto cellInWidth = inferUniformBitsWidth(cellPEDef.inputTypes, 4);
      auto cellOutWidth = inferUniformBitsWidth(cellPEDef.outputTypes, 4);
      assert(cellInWidth && cellOutWidth && *cellInWidth == *cellOutWidth &&
             *cellInWidth == *firstWidth &&
             "buildChessMesh currently expects all PE templates to share one network-facing !fabric.bits<N> type");
    }
  }

  std::map<std::pair<unsigned, unsigned>, SWHandle> switchTemplateCache;
  auto makeSwitchTemplate = [&](unsigned numInputs,
                                unsigned numOutputs) -> SWHandle {
    auto key = std::make_pair(numInputs, numOutputs);
    auto it = switchTemplateCache.find(key);
    if (it != switchTemplateCache.end())
      return it->second;

    std::vector<unsigned> inputWidths(numInputs, *firstWidth);
    std::vector<unsigned> outputWidths(numOutputs, *firstWidth);
    std::vector<std::vector<bool>> fullCrossbar(
        numOutputs, std::vector<bool>(numInputs, true));
    std::string name = "__chess_sw_" + std::to_string(numInputs) + "x" +
                       std::to_string(numOutputs) + "_" +
                       std::to_string(impl_->swDefs.size());
    auto handle = defineSpatialSW(name, inputWidths, outputWidths, fullCrossbar,
                                  options.decomposableBits);
    switchTemplateCache[key] = handle;
    return handle;
  };

  auto switchDegree = [&](unsigned sr, unsigned sc) -> unsigned {
    unsigned degree = 0;
    if (sr > 0)
      degree++;
    if (sr + 1 < rows + 1)
      degree++;
    if (sc > 0)
      degree++;
    if (sc + 1 < cols + 1)
      degree++;
    if (sr > 0 && sc > 0)
      degree++;
    if (sr > 0 && sc < cols)
      degree++;
    if (sr < rows && sc > 0)
      degree++;
    if (sr < rows && sc < cols)
      degree++;
    return degree;
  };

  const unsigned maxSwitchDegree = 8;
  const double maxSwitchBox = std::max(80.0, maxSwitchDegree * 30.0 + 30.0);
  const double approxFuBoxW = 132.0;
  const double approxFuGap = 12.0;
  const double approxPEPadX = 40.0;
  double approxPEBoxW = 200.0;
  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      const auto &cellPEDef = impl_->peDefs[selectedPEs[r][c].id];
      approxPEBoxW = std::max(
          approxPEBoxW,
          std::max(200.0,
                   cellPEDef.fuIndices.size() * approxFuBoxW +
                       std::max(0.0,
                                static_cast<double>(cellPEDef.fuIndices.size()) -
                                    1.0) *
                           approxFuGap +
                       approxPEPadX));
    }
  }
  const double approxPEBoxH = 200.0;
  const double componentGap = 40.0;
  const double switchStepX =
      std::max(520.0, maxSwitchBox + approxPEBoxW + componentGap);
  const double switchStepY =
      std::max(520.0, maxSwitchBox + approxPEBoxH + componentGap);
  const double originX = maxSwitchBox / 2.0 + 80.0;
  const double originY = maxSwitchBox / 2.0 + 80.0;

  for (unsigned sr = 0; sr <= rows; ++sr) {
    for (unsigned sc = 0; sc <= cols; ++sc) {
      unsigned degree = switchDegree(sr, sc);
      unsigned numInputs = degree;
      unsigned numOutputs = degree;
      if (sr == 0 && sc == 0)
        numInputs += options.topLeftExtraInputs;
      if (sr == rows && sc == cols)
        numOutputs += options.bottomRightExtraOutputs;
      SWHandle swHandle = makeSwitchTemplate(numInputs, numOutputs);
      std::string swName = "sw_" + std::to_string(sr) + "_" + std::to_string(sc);
      auto inst = instantiateSW(swHandle, swName);
      result.swGrid[sr][sc] = inst;
      setInstanceVizPosition(inst, originX + sc * switchStepX,
                             originY + sr * switchStepY, sr, sc);
    }
  }

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      std::string peName = "pe_" + std::to_string(r) + "_" + std::to_string(c);
      auto inst = instantiatePE(selectedPEs[r][c], peName);
      result.peGrid[r][c] = inst;
      setInstanceVizPosition(inst, originX + (c + 0.5) * switchStepX,
                             originY + (r + 0.5) * switchStepY, r, c);
    }
  }

  std::map<std::pair<unsigned, unsigned>, unsigned> nextSwitchSlot;
  auto allocSwitchSlot = [&](unsigned sr, unsigned sc) -> unsigned {
    auto key = std::make_pair(sr, sc);
    unsigned slot = nextSwitchSlot[key];
    nextSwitchSlot[key] = slot + 1;
    return slot;
  };

  auto connectSwitchBidirectional = [&](unsigned sr0, unsigned sc0,
                                        unsigned sr1, unsigned sc1) {
    unsigned slot0 = allocSwitchSlot(sr0, sc0);
    unsigned slot1 = allocSwitchSlot(sr1, sc1);
    auto sw0 = result.swGrid[sr0][sc0];
    auto sw1 = result.swGrid[sr1][sc1];
    connect(sw0, slot0, sw1, slot1);
    connect(sw1, slot1, sw0, slot0);
  };

  for (unsigned sr = 0; sr <= rows; ++sr) {
    for (unsigned sc = 0; sc + 1 <= cols; ++sc) {
      if (sc + 1 <= cols)
        connectSwitchBidirectional(sr, sc, sr, sc + 1);
    }
  }
  for (unsigned sr = 0; sr + 1 <= rows; ++sr) {
    for (unsigned sc = 0; sc <= cols; ++sc) {
      if (sr + 1 <= rows)
        connectSwitchBidirectional(sr, sc, sr + 1, sc);
    }
  }

  auto connectPEToSwitch = [&](unsigned r, unsigned c, unsigned peSlot,
                               unsigned sr, unsigned sc) {
    unsigned swSlot = allocSwitchSlot(sr, sc);
    auto peInst = result.peGrid[r][c];
    auto swInst = result.swGrid[sr][sc];
    connect(peInst, peSlot, swInst, swSlot);
    connect(swInst, swSlot, peInst, peSlot);
  };

  for (unsigned r = 0; r < rows; ++r) {
    for (unsigned c = 0; c < cols; ++c) {
      connectPEToSwitch(r, c, 0, r, c);
      connectPEToSwitch(r, c, 1, r, c + 1);
      connectPEToSwitch(r, c, 2, r + 1, c);
      connectPEToSwitch(r, c, 3, r + 1, c + 1);
    }
  }

  if (options.topLeftExtraInputs > 0) {
    unsigned topLeftDegree = switchDegree(0, 0);
    auto swInst = result.swGrid[0][0];
    for (unsigned idx = 0; idx < options.topLeftExtraInputs; ++idx)
      result.ingressPorts.push_back({swInst, topLeftDegree + idx});
  }
  if (options.bottomRightExtraOutputs > 0) {
    unsigned bottomRightDegree = switchDegree(rows, cols);
    auto swInst = result.swGrid[rows][cols];
    for (unsigned idx = 0; idx < options.bottomRightExtraOutputs; ++idx)
      result.egressPorts.push_back({swInst, bottomRightDegree + idx});
  }

  return result;
}

CubeResult ADGBuilder::buildCube(unsigned depths, unsigned rows, unsigned cols,
                                 PEHandle pe, int decomposableBits,
                                 unsigned originExtraInputs,
                                 unsigned farCornerExtraOutputs) {
  CubeOptions options;
  options.decomposableBits = decomposableBits;
  options.originExtraInputs = originExtraInputs;
  options.farCornerExtraOutputs = farCornerExtraOutputs;
  return buildCube(
      depths, rows, cols,
      [pe](unsigned, unsigned, unsigned) { return pe; }, options);
}

CubeResult ADGBuilder::buildCube(unsigned depths, unsigned rows, unsigned cols,
                                 PEHandle pe, const CubeOptions &options) {
  return buildCube(
      depths, rows, cols,
      [pe](unsigned, unsigned, unsigned) { return pe; }, options);
}

CubeResult ADGBuilder::buildCube(
    unsigned depths, unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned, unsigned)> &peSelector,
    int decomposableBits) {
  CubeOptions options;
  options.decomposableBits = decomposableBits;
  return buildCube(depths, rows, cols, peSelector, options);
}

CubeResult ADGBuilder::buildCube(
    unsigned depths, unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned, unsigned)> &peSelector,
    const CubeOptions &options) {
  assert(depths >= 1 && rows >= 1 && cols >= 1 &&
         "buildCube expects positive depth, row, and column counts");
  CubeResult result;
  result.peGrid.resize(
      depths, std::vector<std::vector<InstanceHandle>>(
                  rows, std::vector<InstanceHandle>(cols)));
  result.swGrid.resize(
      depths + 1,
      std::vector<std::vector<InstanceHandle>>(
          rows + 1, std::vector<InstanceHandle>(cols + 1)));

  std::vector<std::vector<std::vector<PEHandle>>> selectedPEs(
      depths, std::vector<std::vector<PEHandle>>(
                  rows, std::vector<PEHandle>(cols)));
  for (unsigned d = 0; d < depths; ++d) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c)
        selectedPEs[d][r][c] = peSelector(d, r, c);
    }
  }

  const auto &firstPEDef = impl_->peDefs[selectedPEs[0][0][0].id];
  assert(!firstPEDef.temporal && "buildCube expects spatial_pe templates");
  assert(firstPEDef.inputTypes.size() >= 8 &&
         "buildCube expects at least eight PE input ports");
  assert(firstPEDef.outputTypes.size() >= 8 &&
         "buildCube expects at least eight PE output ports");
  auto firstWidth = inferUniformBitsWidth(firstPEDef.inputTypes, 8);
  auto firstOutWidth = inferUniformBitsWidth(firstPEDef.outputTypes, 8);
  assert(firstWidth && firstOutWidth && *firstWidth == *firstOutWidth &&
         "buildCube expects the first eight PE ports to share one !fabric.bits<N> type");

  for (unsigned d = 0; d < depths; ++d) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c) {
        const auto &cellPEDef = impl_->peDefs[selectedPEs[d][r][c].id];
        assert(!cellPEDef.temporal && "buildCube expects spatial_pe templates");
        assert(cellPEDef.inputTypes.size() >= 8 &&
               "buildCube expects at least eight PE input ports");
        assert(cellPEDef.outputTypes.size() >= 8 &&
               "buildCube expects at least eight PE output ports");
        auto cellInWidth = inferUniformBitsWidth(cellPEDef.inputTypes, 8);
        auto cellOutWidth = inferUniformBitsWidth(cellPEDef.outputTypes, 8);
        assert(cellInWidth && cellOutWidth && *cellInWidth == *cellOutWidth &&
               *cellInWidth == *firstWidth &&
               "buildCube expects all PE templates to share one network-facing !fabric.bits<N> type");
      }
    }
  }

  std::map<std::pair<unsigned, unsigned>, SWHandle> switchTemplateCache;
  auto makeSwitchTemplate = [&](unsigned numInputs,
                                unsigned numOutputs) -> SWHandle {
    auto key = std::make_pair(numInputs, numOutputs);
    auto it = switchTemplateCache.find(key);
    if (it != switchTemplateCache.end())
      return it->second;

    std::vector<unsigned> inputWidths(numInputs, *firstWidth);
    std::vector<unsigned> outputWidths(numOutputs, *firstWidth);
    std::vector<std::vector<bool>> fullCrossbar(
        numOutputs, std::vector<bool>(numInputs, true));
    std::string name = "__cube_sw_" + std::to_string(numInputs) + "x" +
                       std::to_string(numOutputs) + "_" +
                       std::to_string(impl_->swDefs.size());
    auto handle = defineSpatialSW(name, inputWidths, outputWidths, fullCrossbar,
                                  options.decomposableBits);
    switchTemplateCache[key] = handle;
    return handle;
  };

  auto switchDegree = [&](unsigned sd, unsigned sr, unsigned sc) -> unsigned {
    unsigned degree = 0;
    if (sd > 0)
      degree++;
    if (sd < depths)
      degree++;
    if (sr > 0)
      degree++;
    if (sr < rows)
      degree++;
    if (sc > 0)
      degree++;
    if (sc < cols)
      degree++;

    if (sd > 0 && sr > 0 && sc > 0)
      degree++;
    if (sd > 0 && sr > 0 && sc < cols)
      degree++;
    if (sd > 0 && sr < rows && sc > 0)
      degree++;
    if (sd > 0 && sr < rows && sc < cols)
      degree++;
    if (sd < depths && sr > 0 && sc > 0)
      degree++;
    if (sd < depths && sr > 0 && sc < cols)
      degree++;
    if (sd < depths && sr < rows && sc > 0)
      degree++;
    if (sd < depths && sr < rows && sc < cols)
      degree++;
    return degree;
  };

  const unsigned maxSwitchDegree = 14;
  const double maxSwitchBox = std::max(84.0, maxSwitchDegree * 22.0 + 36.0);
  const double approxFuBoxW = 132.0;
  const double approxFuGap = 12.0;
  const double approxPEPadX = 40.0;
  double approxPEBoxW = 200.0;
  for (unsigned d = 0; d < depths; ++d) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c) {
        const auto &cellPEDef = impl_->peDefs[selectedPEs[d][r][c].id];
        approxPEBoxW = std::max(
            approxPEBoxW,
            std::max(220.0,
                     cellPEDef.fuIndices.size() * approxFuBoxW +
                         std::max(0.0,
                                  static_cast<double>(cellPEDef.fuIndices.size()) -
                                      1.0) *
                             approxFuGap +
                         approxPEPadX));
      }
    }
  }
  const double approxPEBoxH = 200.0;
  const double componentGap = 52.0;
  const double switchStepX =
      std::max(620.0, maxSwitchBox + approxPEBoxW + componentGap);
  const double switchStepY =
      std::max(620.0, maxSwitchBox + approxPEBoxH + componentGap);
  const double sliceGapX = 220.0;
  const double sliceOffsetX =
      static_cast<double>(cols + 1) * switchStepX + sliceGapX;
  const double originX = maxSwitchBox / 2.0 + 120.0;
  const double originY = maxSwitchBox / 2.0 + 180.0;

  for (unsigned sd = 0; sd <= depths; ++sd) {
    for (unsigned sr = 0; sr <= rows; ++sr) {
      for (unsigned sc = 0; sc <= cols; ++sc) {
        unsigned degree = switchDegree(sd, sr, sc);
        unsigned numInputs = degree;
        unsigned numOutputs = degree;
        if (sd == 0 && sr == 0 && sc == 0)
          numInputs += options.originExtraInputs;
        if (sd == depths && sr == rows && sc == cols)
          numOutputs += options.farCornerExtraOutputs;
        auto swHandle = makeSwitchTemplate(numInputs, numOutputs);
        std::string swName = "sw_" + std::to_string(sd) + "_" +
                             std::to_string(sr) + "_" + std::to_string(sc);
        auto inst = instantiateSW(swHandle, swName);
        result.swGrid[sd][sr][sc] = inst;
        double sliceOriginX = originX + sd * sliceOffsetX;
        double x = sliceOriginX + sc * switchStepX;
        double y = originY + sr * switchStepY;
        setInstanceVizPosition(inst, x, y,
                               static_cast<int>(sd * (rows + 1) + sr),
                               static_cast<int>(sc));
      }
    }
  }

  for (unsigned d = 0; d < depths; ++d) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c) {
        std::string peName = "pe_" + std::to_string(d) + "_" +
                             std::to_string(r) + "_" + std::to_string(c);
        auto inst = instantiatePE(selectedPEs[d][r][c], peName);
        result.peGrid[d][r][c] = inst;
        double sliceOriginX = originX + d * sliceOffsetX;
        double x = sliceOriginX + (c + 0.5) * switchStepX;
        double y = originY + (r + 0.5) * switchStepY;
        setInstanceVizPosition(inst, x, y,
                               static_cast<int>(d * rows + r),
                               static_cast<int>(c));
      }
    }
  }

  std::map<std::tuple<unsigned, unsigned, unsigned>, unsigned> nextSwitchSlot;
  auto allocSwitchSlot = [&](unsigned sd, unsigned sr, unsigned sc) -> unsigned {
    auto key = std::make_tuple(sd, sr, sc);
    unsigned slot = nextSwitchSlot[key];
    nextSwitchSlot[key] = slot + 1;
    return slot;
  };

  auto connectSwitchBidirectional = [&](unsigned sd0, unsigned sr0, unsigned sc0,
                                        unsigned sd1, unsigned sr1, unsigned sc1) {
    unsigned slot0 = allocSwitchSlot(sd0, sr0, sc0);
    unsigned slot1 = allocSwitchSlot(sd1, sr1, sc1);
    auto sw0 = result.swGrid[sd0][sr0][sc0];
    auto sw1 = result.swGrid[sd1][sr1][sc1];
    connect(sw0, slot0, sw1, slot1);
    connect(sw1, slot1, sw0, slot0);
  };

  for (unsigned sd = 0; sd <= depths; ++sd) {
    for (unsigned sr = 0; sr <= rows; ++sr) {
      for (unsigned sc = 0; sc + 1 <= cols; ++sc) {
        if (sc + 1 <= cols)
          connectSwitchBidirectional(sd, sr, sc, sd, sr, sc + 1);
      }
    }
  }
  for (unsigned sd = 0; sd <= depths; ++sd) {
    for (unsigned sr = 0; sr + 1 <= rows; ++sr) {
      for (unsigned sc = 0; sc <= cols; ++sc) {
        if (sr + 1 <= rows)
          connectSwitchBidirectional(sd, sr, sc, sd, sr + 1, sc);
      }
    }
  }
  for (unsigned sd = 0; sd + 1 <= depths; ++sd) {
    for (unsigned sr = 0; sr <= rows; ++sr) {
      for (unsigned sc = 0; sc <= cols; ++sc) {
        if (sd + 1 <= depths)
          connectSwitchBidirectional(sd, sr, sc, sd + 1, sr, sc);
      }
    }
  }

  auto connectPEToSwitch = [&](unsigned d, unsigned r, unsigned c, unsigned peSlot,
                               unsigned sd, unsigned sr, unsigned sc) {
    unsigned swSlot = allocSwitchSlot(sd, sr, sc);
    auto peInst = result.peGrid[d][r][c];
    auto swInst = result.swGrid[sd][sr][sc];
    connect(peInst, peSlot, swInst, swSlot);
    connect(swInst, swSlot, peInst, peSlot);
  };

  for (unsigned d = 0; d < depths; ++d) {
    for (unsigned r = 0; r < rows; ++r) {
      for (unsigned c = 0; c < cols; ++c) {
        connectPEToSwitch(d, r, c, 0, d, r, c);
        connectPEToSwitch(d, r, c, 1, d, r, c + 1);
        connectPEToSwitch(d, r, c, 2, d, r + 1, c);
        connectPEToSwitch(d, r, c, 3, d, r + 1, c + 1);
        connectPEToSwitch(d, r, c, 4, d + 1, r, c);
        connectPEToSwitch(d, r, c, 5, d + 1, r, c + 1);
        connectPEToSwitch(d, r, c, 6, d + 1, r + 1, c);
        connectPEToSwitch(d, r, c, 7, d + 1, r + 1, c + 1);
      }
    }
  }

  if (options.originExtraInputs > 0) {
    unsigned originDegree = switchDegree(0, 0, 0);
    auto swInst = result.swGrid[0][0][0];
    for (unsigned idx = 0; idx < options.originExtraInputs; ++idx)
      result.ingressPorts.push_back({swInst, originDegree + idx});
  }
  if (options.farCornerExtraOutputs > 0) {
    unsigned farDegree = switchDegree(depths, rows, cols);
    auto swInst = result.swGrid[depths][rows][cols];
    for (unsigned idx = 0; idx < options.farCornerExtraOutputs; ++idx)
      result.egressPorts.push_back({swInst, farDegree + idx});
  }

  return result;
}

MeshResult ADGBuilder::buildTorusMesh(unsigned rows, unsigned cols, PEHandle pe,
                                      SWHandle sw) {
  return buildMesh(rows, cols, pe, sw);
}

MeshResult ADGBuilder::buildTorusMesh(
    unsigned rows, unsigned cols,
    const std::function<PEHandle(unsigned, unsigned)> &peSelector,
    SWHandle sw) {
  return buildMesh(rows, cols, peSelector, sw);
}

MeshResult ADGBuilder::buildRing(unsigned count, PEHandle pe, SWHandle sw) {
  assert(count >= 2 && "buildRing expects at least two positions");
  return buildMesh(1, count, pe, sw);
}

MeshResult ADGBuilder::buildRing(
    unsigned count, const std::function<PEHandle(unsigned)> &peSelector,
    SWHandle sw) {
  assert(count >= 2 && "buildRing expects at least two positions");
  return buildMesh(1, count,
                   [&](unsigned, unsigned col) { return peSelector(col); }, sw);
}

} // namespace adg
} // namespace fcc
