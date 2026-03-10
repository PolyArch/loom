//===-- ADGGenTemporal.cpp - Temporal domain for ADG generation --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates a dual-mesh temporal domain: a second lattice of temporal_sw
// with temporal_pe instances, bridged to the native mesh via add_tag and
// del_tag operations. Follows the pattern in mixed-temporal.fabric.mlir.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGGenHelpers.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace loom {
namespace adg {

void generateTemporal(ADGBuilder &builder, const MergedRequirements &reqs,
                      const GenConfig &config,
                      std::map<unsigned, WidthLattice> &lattices,
                      const std::map<unsigned, unsigned> &bridgeOutStartByWidth,
                      const std::map<unsigned, unsigned> &bridgeInStartByWidth) {
  if (reqs.peMaxCounts.empty())
    return;

  unsigned tagWidth = config.temporalTagWidth;
  unsigned numInstr = config.temporalNumInstruction;

  // When analysis-driven partitioning is available, use temporalPECounts
  // for temporal mesh sizing. Otherwise, fall back to duplicating all PEs.
  const auto &temporalCounts = config.temporalPECounts.empty()
                                   ? reqs.peMaxCounts
                                   : config.temporalPECounts;

  if (temporalCounts.empty())
    return;

  // Collect unique PE specs grouped by width plane.
  // Each width plane gets one temporal PE definition containing all
  // unique ops as FU sub-PEs.
  std::map<unsigned, std::vector<PESpec>> specsByWidth;
  for (const auto &[spec, count] : temporalCounts) {
    unsigned w = spec.primaryWidth();
    specsByWidth[w].push_back(spec);
  }

  for (auto &[width, specs] : specsByWidth) {
    // Determine temporal PE count from partitioned requirements.
    unsigned temporalPECount = 0;
    for (const auto &[spec, count] : temporalCounts) {
      if (spec.primaryWidth() == width)
        temporalPECount += count;
    }
    if (temporalPECount == 0)
      continue;
    unsigned nativePECount = temporalPECount;

    std::string wp = std::to_string(width);
    Type tagType = Type::iN(tagWidth);
    Type valueType = Type::bits(width);
    Type taggedType = Type::tagged(valueType, tagType);

    // Build temporal PE definition with FU sub-PEs from DFG ops.
    // Deduplicate FU definitions by PESpec.
    std::set<std::string> fuNames;
    auto tpeBuilder = builder.newTemporalPE("tpe_w" + wp)
                          .setNumRegisters(0)
                          .setNumInstructions(numInstr)
                          .setRegFifoDepth(0)
                          .setInterface(taggedType);

    for (const auto &spec : specs) {
      std::string fuName = "fu_" + spec.peName();
      if (fuNames.count(fuName))
        continue;
      fuNames.insert(fuName);
      PEHandle fu = createPEDef(builder, spec);
      tpeBuilder.addFU(fu);
    }
    TemporalPEHandle tpeDef = tpeBuilder;

    // Compute temporal mesh dimensions (same as native mesh).
    auto [tpeRows, tpeCols] = computeMesh2DDims(nativePECount);
    int tswRows = static_cast<int>(tpeRows) + 1;
    int tswCols = static_cast<int>(tpeCols) + 1;

    // Create temporal switch templates.
    // Temporal switches use the tagged type.
    // Port counts mirror native mesh: each cell has 2-in 1-out TPEs.
    // Use the same port assignment scheme as native mesh.

    // For simplicity, build the temporal mesh using the latticeMesh approach:
    // first create switches, then wire inter-switch connections, then place TPEs.

    // Compute per-switch port maps (reuse 2D scheme).
    // TPEs have 2 inputs and 1 output to match the mixed-temporal reference.
    unsigned tpeInCount = 2, tpeOutCount = 1;

    struct TCellInfo {
      unsigned numIn = 0, numOut = 0;
    };
    std::vector<std::vector<TCellInfo>> tCells(
        tpeRows, std::vector<TCellInfo>(tpeCols));
    {
      unsigned placed = 0;
      for (unsigned r = 0; r < tpeRows && placed < nativePECount; ++r) {
        for (unsigned c = 0; c < tpeCols && placed < nativePECount; ++c) {
          tCells[r][c] = {tpeInCount, tpeOutCount};
          ++placed;
        }
      }
    }

    // Compute temporal switch port maps.
    // Also reserve one cross-mesh input on leftmost column (from add_tag)
    // and one cross-mesh output on rightmost column (to del_tag).
    std::vector<std::vector<SwPortMap>> tswPorts(
        tswRows, std::vector<SwPortMap>(tswCols));

    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        pm.dirIn.assign(4, -1);
        pm.dirOut.assign(4, -1);
        unsigned inIdx = 0, outIdx = 0;

        bool hasDir[4] = {r > 0, c < tswCols - 1, r < tswRows - 1, c > 0};

        // Cross-mesh input: add_tag feeds leftmost column (c == 0).
        if (c == 0 && r < tswRows)
          pm.modIn.push_back(static_cast<int>(inIdx++));

        for (int dir = 0; dir < 4; ++dir) {
          if (hasDir[dir]) {
            pm.dirIn[dir] = static_cast<int>(inIdx++);
            pm.dirOut[dir] = static_cast<int>(outIdx++);
          }
        }

        // PE output ports (switch out -> TPE input).
        if (r < (int)tpeRows && c < (int)tpeCols && tCells[r][c].numIn > 0)
          pm.peOut[0] = static_cast<int>(outIdx++);
        if (r > 0 && c < (int)tpeCols && tCells[r - 1][c].numIn > 1)
          pm.peOut[1] = static_cast<int>(outIdx++);

        // PE input ports (TPE output -> switch in).
        if (r > 0 && c > 0 && tCells[r - 1][c - 1].numOut > 0)
          pm.peIn[3] = static_cast<int>(inIdx++);

        // Cross-mesh output: rightmost column (c == tswCols - 1) -> del_tag.
        if (c == tswCols - 1 && r < tswRows)
          pm.modOut.push_back(static_cast<int>(outIdx++));

        pm.numIn = inIdx;
        pm.numOut = outIdx;
      }
    }

    // Create temporal switch templates.
    std::map<std::pair<unsigned, unsigned>, TemporalSwitchHandle> tswTemplates;
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        if (tswTemplates.find(key) == tswTemplates.end()) {
          std::string name = "tsw_w" + wp + "_" + std::to_string(pm.numIn) +
                             "x" + std::to_string(pm.numOut);
          tswTemplates[key] = builder.newTemporalSwitch(name)
                                  .setPortCount(pm.numIn, pm.numOut)
                                  .setInterface(taggedType);
        }
      }
    }

    // Create temporal switch instances.
    std::vector<std::vector<InstanceHandle>> tswGrid(
        tswRows, std::vector<InstanceHandle>(tswCols));
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];
        auto key = std::make_pair(pm.numIn, pm.numOut);
        std::string instName =
            "tsw_" + std::to_string(r) + "_" + std::to_string(c);
        tswGrid[r][c] = builder.clone(tswTemplates[key], instName);
      }
    }

    // Create FIFOs for temporal mesh (reverse directions: W and N).
    FifoHandle tRevFifo{0};
    {
      auto fb = builder.newFifo("trev_fifo_w" + wp)
                    .setDepth(config.fifoDepth)
                    .setType(taggedType);
      if (config.fifoBypassable)
        fb.setBypassable(true);
      tRevFifo = fb;
    }

    // Wire temporal inter-switch connections.
    for (int r = 0; r < tswRows; ++r) {
      for (int c = 0; c < tswCols; ++c) {
        auto &pm = tswPorts[r][c];

        // East: direct
        if (c + 1 < tswCols) {
          auto &pmE = tswPorts[r][c + 1];
          int so = pm.dirOut[1], di = pmE.dirIn[3];
          if (so >= 0 && di >= 0)
            builder.connectPorts(tswGrid[r][c], so, tswGrid[r][c + 1], di);
          // West: via FIFO
          int rso = pmE.dirOut[3], rdi = pm.dirIn[1];
          if (rso >= 0 && rdi >= 0) {
            auto f = builder.clone(tRevFifo,
                "tfifo_w_" + std::to_string(r) + "_" + std::to_string(c));
            builder.connectPorts(tswGrid[r][c + 1], rso, f, 0);
            builder.connectPorts(f, 0, tswGrid[r][c], rdi);
          }
        }

        // South: direct
        if (r + 1 < tswRows) {
          auto &pmS = tswPorts[r + 1][c];
          int so = pm.dirOut[2], di = pmS.dirIn[0];
          if (so >= 0 && di >= 0)
            builder.connectPorts(tswGrid[r][c], so, tswGrid[r + 1][c], di);
          // North: via FIFO
          int rso = pmS.dirOut[0], rdi = pm.dirIn[2];
          if (rso >= 0 && rdi >= 0) {
            auto f = builder.clone(tRevFifo,
                "tfifo_n_" + std::to_string(r) + "_" + std::to_string(c));
            builder.connectPorts(tswGrid[r + 1][c], rso, f, 0);
            builder.connectPorts(f, 0, tswGrid[r][c], rdi);
          }
        }
      }
    }

    // Place temporal PE instances.
    {
      unsigned placed = 0;
      for (unsigned r = 0; r < tpeRows && placed < nativePECount; ++r) {
        for (unsigned c = 0; c < tpeCols && placed < nativePECount; ++c) {
          std::string instName =
              "tpe_r" + std::to_string(r) + "_c" + std::to_string(c);
          auto tpeInst = builder.clone(tpeDef, instName);

          // Input[0] <- UL switch (r, c)
          int swOut0 = tswPorts[r][c].peOut[0];
          if (swOut0 >= 0)
            builder.connectPorts(tswGrid[r][c], swOut0, tpeInst, 0);

          // Input[1] <- LL switch (r+1, c)
          int swOut1 = tswPorts[r + 1][c].peOut[1];
          if (swOut1 >= 0)
            builder.connectPorts(tswGrid[r + 1][c], swOut1, tpeInst, 1);

          // Output[0] -> LR switch (r+1, c+1)
          int swIn0 = tswPorts[r + 1][c + 1].peIn[3];
          if (swIn0 >= 0)
            builder.connectPorts(tpeInst, 0, tswGrid[r + 1][c + 1], swIn0);

          ++placed;
        }
      }
    }

    // Cross-mesh bridges: native mesh -> temporal mesh via add_tag.
    // add_tag on rightmost column of native mesh -> leftmost column of temporal.
    // Since we don't have access to the native mesh switch grid here, we create
    // add_tag instances connected to module I/O ports that the native mesh
    // can connect to.
    //
    // Bridge pattern (matching mixed-temporal reference):
    //   Native rightmost column -> add_tag -> temporal leftmost column
    //   Temporal rightmost column -> del_tag + FIFO -> native rightmost column

    // Create bridge FIFO for del_tag output.
    FifoHandle bridgeFifo{0};
    {
      auto fb = builder.newFifo("bridge_fifo_w" + wp)
                    .setDepth(config.fifoDepth)
                    .setType(valueType);
      if (config.fifoBypassable)
        fb.setBypassable(true);
      bridgeFifo = fb;
    }

    // add_tag bridges: feed temporal mesh leftmost column.
    for (int r = 0; r < tswRows; ++r) {
      auto &pm = tswPorts[r][0];
      if (pm.modIn.empty())
        continue;

      // Create add_tag: native bits -> tagged.
      std::string atName = "add_tag_r" + std::to_string(r) + "_w" + wp;
      InstanceHandle addTagInst =
          builder.newAddTag(atName)
              .setValueType(valueType)
              .setTagType(tagType);

      // Connect add_tag output to temporal switch leftmost column input.
      builder.connectPorts(addTagInst, 0, tswGrid[r][0], pm.modIn[0]);

      // Create a module input for the add_tag's native-side input.
      std::string portName = "bridge_in_r" + std::to_string(r) + "_w" + wp;
      auto bridgePort = builder.addModuleInput(portName, valueType);
      builder.connectToModuleInput(bridgePort, addTagInst, 0);
    }

    // del_tag bridges: temporal mesh rightmost column -> FIFO -> module output.
    for (int r = 0; r < tswRows; ++r) {
      auto &pm = tswPorts[r][tswCols - 1];
      if (pm.modOut.empty())
        continue;

      // Create del_tag: tagged -> native bits.
      std::string dtName = "del_tag_r" + std::to_string(r) + "_w" + wp;
      InstanceHandle delTagInst =
          builder.newDelTag(dtName)
              .setInputType(taggedType);

      // Connect temporal switch output to del_tag input.
      builder.connectPorts(tswGrid[r][tswCols - 1], pm.modOut[0],
                           delTagInst, 0);

      // del_tag output -> FIFO -> module output.
      std::string fifoName =
          "bridge_fifo_r" + std::to_string(r) + "_w" + wp;
      auto fifoInst = builder.clone(bridgeFifo, fifoName);
      builder.connectPorts(delTagInst, 0, fifoInst, 0);

      std::string portName = "bridge_out_r" + std::to_string(r) + "_w" + wp;
      auto bridgePort = builder.addModuleOutput(portName, valueType);
      builder.connectToModuleOutput(fifoInst, 0, bridgePort);
    }
  }
}

} // namespace adg
} // namespace loom
