//===-- gen_smoke.cpp - Smoke CGRA template generator ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates loom_cgra_smoke.fabric.mlir using width-based latticeMesh topology
// with three-phase construction (plan -> create -> connect).
// Target apps: vecsum, vecadd, vecmul.
//
//===----------------------------------------------------------------------===//

#include "cgra_pe_catalog.h"

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// PE instance table: max count across vecsum/vecadd/vecmul per PE type.
static const std::map<std::string, int> SMOKE_INSTANCES = {
    {"pe_join1", 1},
    {"pe_join3", 1},
    {"pe_const_i32", 1},
    {"pe_const_i64", 1},
    {"pe_const_index", 3},
    {"pe_cmpi", 1},
    {"pe_cond_br_none", 4},
    {"pe_cond_br_i32", 1},
    {"pe_sink_i1", 1},
    {"pe_sink_none", 1},
    {"pe_index_cast_i64", 1},
    {"pe_index_cast_i32", 1},
    {"pe_stream", 1},
    {"pe_gate_index", 1},
    {"pe_gate", 1},
    {"pe_carry", 1},
    {"pe_carry_none", 3},
    {"pe_load", 1},
    {"pe_load_f32", 2},
    {"pe_store_f32", 1},
    {"pe_addi", 1},
    {"pe_addf", 1},
    {"pe_mulf", 1},
    {"pe_select_index", 1},
    {"pe_mux_i32", 1},
    {"pe_mux_none", 3},
};

struct MemConfig {
  std::string elemType;
  int ldCount;
  int stCount;
};

// Max across apps: vecsum needs 1x i32 ld. vecadd/vecmul need 2x f32 ld + 1x f32 st.
static const MemConfig EXTMEM_SMOKE[] = {
    {"i32", 1, 0},
    {"f32", 1, 0},
    {"f32", 1, 0},
    {"f32", 0, 1},
};

//===----------------------------------------------------------------------===//
// Phase 1 data structures
//===----------------------------------------------------------------------===//

struct PESlot {
  std::string name;
  int instanceIdx;
  const PECatalogEntry *entry;
  std::set<WidthPlane> planes;
};

struct PlaneState {
  std::vector<int> peIndices;
  int peRows = 0, peCols = 0;
  int swRows = 0, swCols = 0;
  std::vector<int> swInNeeded;
  std::vector<int> swOutNeeded;
  int uniformPortCount = 0;
  std::vector<std::pair<int, int>> inBoundary;
  std::vector<std::pair<int, int>> outBoundary;
};

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char *argv[]) {
  std::string outputPath = "loom_cgra_smoke.fabric.mlir";
  if (argc > 1)
    outputPath = argv[1];

  ADGBuilder builder("loom_cgra_smoke");
  auto catalog = registerAllPEs(builder);

  //===------------------------------------------------------------------===//
  // PHASE 1: Parameter Negotiation
  //===------------------------------------------------------------------===//

  // 1a. Build PE slots and determine width plane membership.
  std::vector<PESlot> peSlots;
  for (auto &kv : SMOKE_INSTANCES) {
    auto it = catalog.find(kv.first);
    if (it == catalog.end()) {
      std::cerr << "PE not found: " << kv.first << "\n";
      return 1;
    }
    for (int i = 0; i < kv.second; i++) {
      PESlot slot;
      slot.name = kv.first;
      slot.instanceIdx = i;
      slot.entry = &it->second;
      for (auto &t : it->second.def.inputTypes)
        slot.planes.insert(typeNameToWidthPlane(t));
      for (auto &t : it->second.def.outputTypes)
        slot.planes.insert(typeNameToWidthPlane(t));
      peSlots.push_back(slot);
    }
  }

  // 1b. Assign PEs to width planes and size grids.
  PlaneState planes[WP_COUNT];
  for (size_t i = 0; i < peSlots.size(); i++)
    for (auto wp : peSlots[i].planes)
      planes[wp].peIndices.push_back(static_cast<int>(i));

  for (int wp = 0; wp < WP_COUNT; wp++) {
    auto &ps = planes[wp];
    int n = static_cast<int>(ps.peIndices.size());
    if (n == 0)
      continue;
    auto [rows, cols] = computePEGridDims(n);
    ps.peRows = rows;
    ps.peCols = cols;
    ps.swRows = rows + 1;
    ps.swCols = cols + 1;
    ps.swInNeeded.resize(ps.swRows * ps.swCols, 0);
    ps.swOutNeeded.resize(ps.swRows * ps.swCols, 0);
    ps.inBoundary = inputBoundaryOrder(ps.swRows, ps.swCols);
    ps.outBoundary = outputBoundaryOrder(ps.swRows, ps.swCols);
  }

  // 1c. Calculate switch port needs from PE-to-cell assignments.
  for (int wp = 0; wp < WP_COUNT; wp++) {
    auto &ps = planes[wp];
    if (ps.peRows == 0)
      continue;
    for (size_t peIdx = 0; peIdx < ps.peIndices.size(); peIdx++) {
      auto &slot = peSlots[ps.peIndices[peIdx]];
      int cellR = static_cast<int>(peIdx) / ps.peCols;
      int cellC = static_cast<int>(peIdx) % ps.peCols;
      int corners[4][2] = {{cellR, cellC},
                            {cellR, cellC + 1},
                            {cellR + 1, cellC},
                            {cellR + 1, cellC + 1}};
      // PE inputs need switch outputs
      int rrIn = 0;
      for (auto &t : slot.entry->def.inputTypes)
        if (typeNameToWidthPlane(t) == static_cast<WidthPlane>(wp)) {
          int ci = rrIn++ % 4;
          ps.swOutNeeded[corners[ci][0] * ps.swCols + corners[ci][1]]++;
        }
      // PE outputs need switch inputs
      int rrOut = 0;
      for (auto &t : slot.entry->def.outputTypes)
        if (typeNameToWidthPlane(t) == static_cast<WidthPlane>(wp)) {
          int ci = rrOut++ % 4;
          ps.swInNeeded[corners[ci][0] * ps.swCols + corners[ci][1]]++;
        }
    }
  }

  // 1d. Add boundary and memory port needs to switch counts.
  // Module inputs feed into switch inputs; module outputs drain switch outputs.
  // Memory native data ports are distributed round-robin.
  auto addToSwIn = [](PlaneState &ps, int r, int c) {
    ps.swInNeeded[r * ps.swCols + c]++;
  };
  auto addToSwOut = [](PlaneState &ps, int r, int c) {
    ps.swOutNeeded[r * ps.swCols + c]++;
  };

  // Module inputs: 2x i32 -> WP_32, 1x none -> WP_NONE
  // Use first few positions from input boundary.
  {
    int bi = 0;
    auto [r0, c0] = planes[WP_32].inBoundary[bi++];
    addToSwIn(planes[WP_32], r0, c0);
    auto [r1, c1] = planes[WP_32].inBoundary[bi++];
    addToSwIn(planes[WP_32], r1, c1);
  }
  {
    auto [r, c] = planes[WP_NONE].inBoundary[0];
    addToSwIn(planes[WP_NONE], r, c);
  }

  // Module outputs: 1x i32 -> WP_32, 1x none -> WP_NONE
  {
    auto [r, c] = planes[WP_32].outBoundary[0];
    addToSwOut(planes[WP_32], r, c);
  }
  {
    auto [r, c] = planes[WP_NONE].outBoundary[0];
    addToSwOut(planes[WP_NONE], r, c);
  }

  // Memory native data ports: distribute round-robin across all switches.
  // Total memory connections: index(4 in), wp32(1 in + 3 out), none(4 out).
  // Just add +1 to each plane's max as margin (very conservative for smoke).

  // 1e. Compute uniform port count per plane.
  for (int wp = 0; wp < WP_COUNT; wp++) {
    auto &ps = planes[wp];
    if (ps.peRows == 0)
      continue;
    int maxPorts = 0;
    for (int i = 0; i < ps.swRows * ps.swCols; i++) {
      int needed = std::max(4 + ps.swInNeeded[i], 4 + ps.swOutNeeded[i]);
      maxPorts = std::max(maxPorts, needed);
    }
    // Add margin for memory data ports (distributed round-robin).
    maxPorts += 2;
    if (maxPorts < 8)
      maxPorts = 8;
    if (maxPorts > 32)
      maxPorts = 32;
    ps.uniformPortCount = maxPorts;
  }

  //===------------------------------------------------------------------===//
  // PHASE 2: Module Creation
  //===------------------------------------------------------------------===//

  // 2a. Create lattice meshes.
  struct PlaneRuntime {
    WidthPlane wp;
    LatticeMeshResult lattice;
    std::vector<int> nextInPort;
    std::vector<int> nextOutPort;
  };
  std::map<WidthPlane, PlaneRuntime> runtime;

  static const char *wpNames[] = {"1", "none", "8", "16", "32", "64", "idx"};
  for (int wp = 0; wp < WP_COUNT; wp++) {
    auto &ps = planes[wp];
    if (ps.peRows == 0)
      continue;
    std::string swName = std::string("sw_wp") + wpNames[wp];
    auto sw = builder.newSwitch(swName)
                  .setPortCount(ps.uniformPortCount, ps.uniformPortCount)
                  .setType(widthPlaneToType(static_cast<WidthPlane>(wp)));
    auto lattice = builder.latticeMesh(ps.peRows, ps.peCols, sw);
    PlaneRuntime pr;
    pr.wp = static_cast<WidthPlane>(wp);
    pr.lattice = lattice;
    pr.nextInPort.resize(ps.swRows * ps.swCols, 4);
    pr.nextOutPort.resize(ps.swRows * ps.swCols, 4);
    runtime[static_cast<WidthPlane>(wp)] = pr;
  }

  // 2b. Clone PE instances.
  std::vector<InstanceHandle> peInstances;
  for (auto &slot : peSlots) {
    std::string name =
        slot.name.substr(3) + "_" + std::to_string(slot.instanceIdx);
    peInstances.push_back(builder.clone(slot.entry->tmpl.handle, name));
  }

  // 2c. Create extmemory instances with direct memref connections.
  struct ExtMemInst {
    InstanceHandle inst;
    std::string elemType;
    int ldCount, stCount;
  };
  std::vector<ExtMemInst> extMemInsts;
  std::map<std::string, int> extMemTypeIdx;

  for (auto &em : EXTMEM_SMOKE) {
    int idx = extMemTypeIdx[em.elemType]++;
    std::string defName =
        "extmem_" + em.elemType + "_def_" + std::to_string(idx);
    std::string instName =
        "extmem_" + em.elemType + "_" + std::to_string(idx);
    Type dataType = (em.elemType == "f32") ? Type::f32() : Type::i32();
    auto emBuilder = builder.newExtMemory(defName)
                         .setLoadPorts(em.ldCount)
                         .setStorePorts(em.stCount)
                         .setShape(MemrefType::dynamic1D(dataType));
    if (em.stCount > 0)
      emBuilder.setQueueDepth(4);
    auto inst = builder.clone(emBuilder, instName);

    std::string mrefName =
        "mem_" + em.elemType + "_" + std::to_string(idx);
    auto mref =
        builder.addModuleInput(mrefName, MemrefType::dynamic1D(dataType));
    builder.connectToModuleInput(mref, inst, 0);
    extMemInsts.push_back({inst, em.elemType, em.ldCount, em.stCount});
  }

  // 2d. Create streaming module I/O.
  auto ctrl_in = builder.addModuleInput("ctrl_in", Type::none());
  PortHandle in_i32[2];
  for (int i = 0; i < 2; i++)
    in_i32[i] = builder.addModuleInput("in" + std::to_string(i), Type::i32());
  auto out_i32 = builder.addModuleOutput("out_i32", Type::i32());
  auto out_none = builder.addModuleOutput("out_none", Type::none());

  //===------------------------------------------------------------------===//
  // PHASE 3: Port Connection
  //===------------------------------------------------------------------===//

  auto getSw = [&](WidthPlane wp, int r, int c) -> InstanceHandle {
    return runtime[wp].lattice.swGrid[r][c];
  };

  // 3a. Connect PE ports to corner switches.
  for (int wp = 0; wp < WP_COUNT; wp++) {
    auto &ps = planes[wp];
    if (ps.peRows == 0)
      continue;
    auto &pr = runtime[static_cast<WidthPlane>(wp)];

    for (size_t peIdx = 0; peIdx < ps.peIndices.size(); peIdx++) {
      int slotIdx = ps.peIndices[peIdx];
      auto &slot = peSlots[slotIdx];
      auto inst = peInstances[slotIdx];
      int cellR = static_cast<int>(peIdx) / ps.peCols;
      int cellC = static_cast<int>(peIdx) % ps.peCols;
      int corners[4][2] = {{cellR, cellC},
                            {cellR, cellC + 1},
                            {cellR + 1, cellC},
                            {cellR + 1, cellC + 1}};

      // Connect inputs: switch output -> PE input
      int rrIn = 0;
      for (size_t p = 0; p < slot.entry->def.inputTypes.size(); p++) {
        if (typeNameToWidthPlane(slot.entry->def.inputTypes[p]) !=
            static_cast<WidthPlane>(wp))
          continue;
        int ci = rrIn++ % 4;
        int sr = corners[ci][0], sc = corners[ci][1];
        int flat = sr * ps.swCols + sc;
        int port = pr.nextOutPort[flat]++;
        builder.connectPorts(getSw(static_cast<WidthPlane>(wp), sr, sc), port,
                             inst, static_cast<unsigned>(p));
      }

      // Connect outputs: PE output -> switch input
      int rrOut = 0;
      for (size_t p = 0; p < slot.entry->def.outputTypes.size(); p++) {
        if (typeNameToWidthPlane(slot.entry->def.outputTypes[p]) !=
            static_cast<WidthPlane>(wp))
          continue;
        int ci = rrOut++ % 4;
        int sr = corners[ci][0], sc = corners[ci][1];
        int flat = sr * ps.swCols + sc;
        int port = pr.nextInPort[flat]++;
        builder.connectPorts(inst, static_cast<unsigned>(p),
                             getSw(static_cast<WidthPlane>(wp), sr, sc), port);
      }
    }
  }

  // 3b. Connect boundary module I/O.
  {
    auto &ps32 = planes[WP_32];
    auto &pr32 = runtime[WP_32];
    // Module input in0 -> WP_32 boundary[0]
    {
      auto [r, c] = ps32.inBoundary[0];
      int port = pr32.nextInPort[r * ps32.swCols + c]++;
      builder.connectToModuleInput(in_i32[0], getSw(WP_32, r, c), port);
    }
    // Module input in1 -> WP_32 boundary[1]
    {
      auto [r, c] = ps32.inBoundary[1];
      int port = pr32.nextInPort[r * ps32.swCols + c]++;
      builder.connectToModuleInput(in_i32[1], getSw(WP_32, r, c), port);
    }
    // Module output out_i32 -> WP_32 outBoundary[0]
    {
      auto [r, c] = ps32.outBoundary[0];
      int port = pr32.nextOutPort[r * ps32.swCols + c]++;
      builder.connectToModuleOutput(getSw(WP_32, r, c), port, out_i32);
    }
  }
  {
    auto &psN = planes[WP_NONE];
    auto &prN = runtime[WP_NONE];
    // Module input ctrl_in -> WP_NONE boundary[0]
    {
      auto [r, c] = psN.inBoundary[0];
      int port = prN.nextInPort[r * psN.swCols + c]++;
      builder.connectToModuleInput(ctrl_in, getSw(WP_NONE, r, c), port);
    }
    // Module output out_none -> WP_NONE outBoundary[0]
    {
      auto [r, c] = psN.outBoundary[0];
      int port = prN.nextOutPort[r * psN.swCols + c]++;
      builder.connectToModuleOutput(getSw(WP_NONE, r, c), port, out_none);
    }
  }

  // 3c. Connect extmemory data ports via LatticeConnector wrappers.
  // Memory data ports use round-robin across all switches (like the original
  // generators), synced with the port cursors after PE and boundary connections.
  std::vector<std::unique_ptr<LatticeConnector>> connOwners;
  std::vector<LatticeConnector *> wpConn(WP_COUNT, nullptr);
  for (auto &[wp, pr] : runtime) {
    auto conn = std::make_unique<LatticeConnector>(
        builder, pr.lattice, planes[wp].uniformPortCount);
    // Sync port cursors to account for already-allocated ports.
    int total = planes[wp].swRows * planes[wp].swCols;
    for (int i = 0; i < total; i++) {
      conn->nextInPort[i] = pr.nextInPort[i];
      conn->nextOutPort[i] = pr.nextOutPort[i];
    }
    wpConn[wp] = conn.get();
    connOwners.push_back(std::move(conn));
  }

  for (auto &emi : extMemInsts) {
    auto layout = extMemPortLayout(emi.ldCount, emi.stCount);
    WidthPlane dataWp = typeNameToWidthPlane(emi.elemType);

    if (!layout.isTagged) {
      if (layout.ldAddrPort >= 0 && wpConn[WP_INDEX])
        wpConn[WP_INDEX]->feedPEInput(emi.inst, layout.ldAddrPort);
      if (layout.stAddrPort >= 0 && wpConn[WP_INDEX])
        wpConn[WP_INDEX]->feedPEInput(emi.inst, layout.stAddrPort);
      if (layout.stDataPort >= 0 && wpConn[dataWp])
        wpConn[dataWp]->feedPEInput(emi.inst, layout.stDataPort);
      if (layout.ldDataPort >= 0 && wpConn[dataWp])
        wpConn[dataWp]->drainPEOutput(emi.inst, layout.ldDataPort);
      if (layout.ldDonePort >= 0 && wpConn[WP_NONE])
        wpConn[WP_NONE]->drainPEOutput(emi.inst, layout.ldDonePort);
      if (layout.stDonePort >= 0 && wpConn[WP_NONE])
        wpConn[WP_NONE]->drainPEOutput(emi.inst, layout.stDonePort);
    }
  }

  // 3d. Finalize all lattices.
  for (auto &[wp, pr] : runtime)
    builder.finalizeLattice(pr.lattice);

  builder.exportMLIR(outputPath);
  std::cout << "Generated " << outputPath << "\n";
  return 0;
}
