//===-- gen_smoke.cpp - Smoke CGRA template generator ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates loom_cgra_smoke.fabric.mlir using width-based latticeMesh topology.
// Target apps: vecsum, vecadd, vecmul.
//
// Uses the proven LatticeConnector round-robin approach (same as gen_small)
// but with width-plane merging (i32+f32 share WP_32) for better per-plane
// routing capacity.
//
// PE and memory tables match gen_small to ensure placement compatibility.
// The only structural difference is width-based plane merging for routing.
//
//===----------------------------------------------------------------------===//

#include "cgra_pe_catalog.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

// PE instance table: same as gen_small for full placement compatibility.
static const std::map<std::string, int> SMOKE_INSTANCES = {
    {"pe_const_i32", 4}, {"pe_const_f32", 2}, {"pe_const_i64", 3},
    {"pe_const_index", 5}, {"pe_const_i1", 2}, {"pe_const_i16", 1},
    {"pe_join1", 4}, {"pe_join2", 1}, {"pe_join3", 1}, {"pe_join4", 1},
    {"pe_join_i1", 1},
    {"pe_cond_br_none", 5}, {"pe_cond_br_i32", 2}, {"pe_cond_br_f32", 1},
    {"pe_cond_br_index", 1},
    {"pe_mux_i32", 2}, {"pe_mux_f32", 1}, {"pe_mux_none", 3},
    {"pe_mux_index", 1},
    {"pe_addi", 4}, {"pe_subi", 2}, {"pe_muli", 2}, {"pe_divui", 1},
    {"pe_divsi", 1}, {"pe_remui", 1}, {"pe_remsi", 1},
    {"pe_addi_i64", 2}, {"pe_subi_i64", 1}, {"pe_muli_i64", 1},
    {"pe_cmpi_i64", 2}, {"pe_shli_i64", 1}, {"pe_remui_i64", 1},
    {"pe_addi_index", 1}, {"pe_divui_index", 1}, {"pe_divsi_index", 1},
    {"pe_remui_index", 1}, {"pe_muli_index", 1},
    {"pe_andi", 1}, {"pe_ori", 1}, {"pe_xori", 1},
    {"pe_shli", 1}, {"pe_shrui", 1}, {"pe_shrsi", 1},
    {"pe_xori_i1", 1},
    {"pe_addf", 1}, {"pe_subf", 1}, {"pe_mulf", 1}, {"pe_divf", 1},
    {"pe_fma", 1},
    {"pe_negf", 1}, {"pe_absf", 1}, {"pe_sin", 1}, {"pe_cos", 1},
    {"pe_exp", 1}, {"pe_sqrt", 1}, {"pe_log2", 1},
    {"pe_cmpi", 3}, {"pe_cmpf", 1},
    {"pe_select", 2}, {"pe_select_index", 3}, {"pe_select_f32", 1},
    {"pe_index_cast_i32", 5}, {"pe_index_cast_i64", 3},
    {"pe_index_cast_to_i32", 1}, {"pe_index_cast_to_i64", 2},
    {"pe_index_castui", 1},
    {"pe_extui", 4}, {"pe_trunci", 2},
    {"pe_extui_i1", 3}, {"pe_trunci_to_i1", 2},
    {"pe_extui_i16", 1}, {"pe_trunci_to_i16", 1}, {"pe_remui_i16", 1},
    {"pe_uitofp", 1}, {"pe_uitofp_i16", 1}, {"pe_sitofp", 1}, {"pe_fptoui", 1},
    {"pe_stream", 2}, {"pe_gate", 3}, {"pe_gate_f32", 1},
    {"pe_gate_index", 3},
    {"pe_carry", 4}, {"pe_carry_f32", 1}, {"pe_carry_none", 4},
    {"pe_carry_index", 1},
    {"pe_invariant", 4}, {"pe_invariant_i1", 2}, {"pe_invariant_none", 2},
    {"pe_invariant_f32", 1}, {"pe_invariant_index", 1},
    {"pe_load", 3}, {"pe_load_f32", 2}, {"pe_store", 2}, {"pe_store_f32", 1},
    {"pe_sink_i1", 3}, {"pe_sink_none", 2}, {"pe_sink_i32", 1},
    {"pe_sink_index", 1}, {"pe_sink_f32", 1},
};

struct MemConfig {
  std::string elemType;
  int ldCount;
  int stCount;
};

// Same memory config as gen_small for full compatibility.
static const MemConfig EXTMEM_SMOKE[] = {
    {"i32", 1, 0}, {"i32", 1, 0}, {"i32", 0, 1}, {"i32", 0, 1},
    {"i32", 1, 1},
    {"f32", 1, 0}, {"f32", 1, 0}, {"f32", 0, 1},
};

static const MemConfig PRIVMEM_SMOKE[] = {
    {"i32", 1, 1},
    {"f32", 1, 1},
};

int main(int argc, char *argv[]) {
  std::string outputPath = "loom_cgra_smoke.fabric.mlir";
  if (argc > 1)
    outputPath = argv[1];

  ADGBuilder builder("loom_cgra_smoke");
  auto catalog = registerAllPEs(builder);

  // Count connections per width plane.
  int wpConns[WP_COUNT] = {};
  for (auto &kv : SMOKE_INSTANCES) {
    auto it = catalog.find(kv.first);
    if (it == catalog.end()) {
      std::cerr << "PE not found: " << kv.first << "\n";
      return 1;
    }
    for (int i = 0; i < kv.second; i++) {
      for (auto &t : it->second.def.inputTypes)
        wpConns[typeNameToWidthPlane(t)]++;
      for (auto &t : it->second.def.outputTypes)
        wpConns[typeNameToWidthPlane(t)]++;
    }
  }

  // Module I/O: 4x i32 in, 1x index in, 1x none in; 1x i32 out, 1x none out.
  wpConns[WP_32] += 4 + 1;
  wpConns[WP_INDEX] += 1;
  wpConns[WP_NONE] += 1 + 1;

  // Memory connections through lattice (native ports only, tagged bypass).
  for (auto &em : EXTMEM_SMOKE) {
    bool isTagged = (em.ldCount > 1 || em.stCount > 1);
    if (isTagged) continue;
    WidthPlane dataWp = typeNameToWidthPlane(em.elemType);
    if (em.ldCount > 0) {
      wpConns[WP_INDEX]++;  // ld_addr
      wpConns[dataWp]++;    // ld_data
      wpConns[WP_NONE]++;   // ld_done
    }
    if (em.stCount > 0) {
      wpConns[WP_INDEX]++;  // st_addr
      wpConns[dataWp]++;    // st_data
      wpConns[WP_NONE]++;   // st_done
    }
  }
  for (auto &pm : PRIVMEM_SMOKE) {
    bool isTagged = (pm.ldCount > 1 || pm.stCount > 1);
    if (isTagged) continue;
    WidthPlane dataWp = typeNameToWidthPlane(pm.elemType);
    if (pm.ldCount > 0) {
      wpConns[WP_INDEX]++;
      wpConns[dataWp]++;
      wpConns[WP_NONE]++;
    }
    if (pm.stCount > 0) {
      wpConns[WP_INDEX]++;
      wpConns[dataWp]++;
      wpConns[WP_NONE]++;
    }
  }

  const int PORTS_PER_SW = 32;

  // Create lattice meshes per width plane (skip unused planes).
  struct PlaneInfo {
    WidthPlane wp;
    SwitchHandle sw;
    LatticeMeshResult lattice;
    LatticeConnector *connector;
  };

  std::vector<PlaneInfo> activePlanes;
  std::vector<std::unique_ptr<LatticeConnector>> connectors;

  static const char *wpNames[] = {"1", "none", "8", "16", "32", "64", "idx"};
  for (int i = 0; i < WP_COUNT; i++) {
    auto wp = static_cast<WidthPlane>(i);
    if (wpConns[i] <= 0)
      continue;
    // Add routing overhead margin: each routed connection uses ~2x ports
    // (ingress + egress at minimum, more for multi-hop routes).
    int inflatedConns = wpConns[i] * 2;
    auto dims = computeLatticeDims(inflatedConns, PORTS_PER_SW);
    std::cerr << "WP_" << wpNames[i] << ": conns=" << wpConns[i]
              << " inflated=" << inflatedConns
              << " lattice=" << dims.first << "x" << dims.second
              << " switches=" << (dims.first+1) << "x" << (dims.second+1)
              << "=" << (dims.first+1)*(dims.second+1) << "\n";
    std::string swName = std::string("sw_wp") + wpNames[i];
    auto sw = builder.newSwitch(swName)
                  .setPortCount(PORTS_PER_SW, PORTS_PER_SW)
                  .setType(widthPlaneToType(wp));
    auto lattice = builder.latticeMesh(dims.first, dims.second, sw);
    auto conn =
        std::make_unique<LatticeConnector>(builder, lattice, PORTS_PER_SW);
    activePlanes.push_back(PlaneInfo{wp, sw, lattice, conn.get()});
    connectors.push_back(std::move(conn));
  }

  // Rebuild connectors after planes vector is stable.
  for (size_t i = 0; i < activePlanes.size(); i++) {
    connectors[i] = std::make_unique<LatticeConnector>(
        builder, activePlanes[i].lattice, PORTS_PER_SW);
    activePlanes[i].connector = connectors[i].get();
  }

  // Build width-plane connector lookup.
  std::vector<LatticeConnector *> wpConn(WP_COUNT, nullptr);
  for (size_t i = 0; i < activePlanes.size(); i++)
    wpConn[activePlanes[i].wp] = activePlanes[i].connector;

  // Create extmemory instances with memref module inputs.
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

    std::string mrefName = "mem_" + em.elemType + "_" + std::to_string(idx);
    auto mref =
        builder.addModuleInput(mrefName, MemrefType::dynamic1D(dataType));
    builder.connectToModuleInput(mref, inst, 0);

    // Connect memory data ports through width-plane lattices.
    auto layout = extMemPortLayout(em.ldCount, em.stCount);
    WidthPlane dataWp = typeNameToWidthPlane(em.elemType);
    if (!layout.isTagged) {
      if (layout.ldAddrPort >= 0)
        wpConn[WP_INDEX]->feedPEInput(inst, layout.ldAddrPort);
      if (layout.stAddrPort >= 0)
        wpConn[WP_INDEX]->feedPEInput(inst, layout.stAddrPort);
      if (layout.stDataPort >= 0)
        wpConn[dataWp]->feedPEInput(inst, layout.stDataPort);
      if (layout.ldDataPort >= 0)
        wpConn[dataWp]->drainPEOutput(inst, layout.ldDataPort);
      if (layout.ldDonePort >= 0)
        wpConn[WP_NONE]->drainPEOutput(inst, layout.ldDonePort);
      if (layout.stDonePort >= 0)
        wpConn[WP_NONE]->drainPEOutput(inst, layout.stDonePort);
    }
  }

  // Module inputs.
  auto ctrl_in = builder.addModuleInput("ctrl_in", Type::none());
  wpConn[WP_NONE]->feedModuleInput(ctrl_in);

  for (int i = 0; i < 4; i++) {
    auto p = builder.addModuleInput("in" + std::to_string(i), Type::i32());
    wpConn[WP_32]->feedModuleInput(p);
  }
  {
    auto p = builder.addModuleInput("addr0", Type::index());
    wpConn[WP_INDEX]->feedModuleInput(p);
  }

  // Instantiate PEs and connect to width-plane lattices.
  for (auto &kv : SMOKE_INSTANCES) {
    auto it = catalog.find(kv.first);
    if (it == catalog.end()) {
      std::cerr << "PE not found: " << kv.first << "\n";
      return 1;
    }
    auto &entry = it->second;
    for (int i = 0; i < kv.second; i++) {
      std::string instName = kv.first.substr(3) + "_" + std::to_string(i);
      auto pe = builder.clone(entry.tmpl.handle, instName);

      for (size_t p = 0; p < entry.def.inputTypes.size(); p++) {
        auto wp = typeNameToWidthPlane(entry.def.inputTypes[p]);
        wpConn[wp]->feedPEInput(pe, p);
      }
      for (size_t p = 0; p < entry.def.outputTypes.size(); p++) {
        auto wp = typeNameToWidthPlane(entry.def.outputTypes[p]);
        wpConn[wp]->drainPEOutput(pe, p);
      }
    }
  }

  // Private memory.
  std::map<std::string, int> privMemTypeIdx;
  for (auto &pm : PRIVMEM_SMOKE) {
    int idx = privMemTypeIdx[pm.elemType]++;
    std::string defName =
        "privmem_" + pm.elemType + "_def_" + std::to_string(idx);
    std::string instName =
        "privmem_" + pm.elemType + "_" + std::to_string(idx);
    Type dataType = (pm.elemType == "f32") ? Type::f32() : Type::i32();
    auto memBuilder = builder.newMemory(defName)
                          .setLoadPorts(pm.ldCount)
                          .setStorePorts(pm.stCount)
                          .setPrivate(true)
                          .setShape(MemrefType::static1D(1024, dataType));
    if (pm.stCount > 0)
      memBuilder.setQueueDepth(4);
    auto inst = builder.clone(memBuilder, instName);

    // Connect private memory data ports through width-plane lattices.
    auto layout = privMemPortLayout(pm.ldCount, pm.stCount);
    WidthPlane dataWp = typeNameToWidthPlane(pm.elemType);
    if (!layout.isTagged) {
      if (layout.ldAddrPort >= 0)
        wpConn[WP_INDEX]->feedPEInput(inst, layout.ldAddrPort);
      if (layout.stAddrPort >= 0)
        wpConn[WP_INDEX]->feedPEInput(inst, layout.stAddrPort);
      if (layout.stDataPort >= 0)
        wpConn[dataWp]->feedPEInput(inst, layout.stDataPort);
      if (layout.ldDataPort >= 0)
        wpConn[dataWp]->drainPEOutput(inst, layout.ldDataPort);
      if (layout.ldDonePort >= 0)
        wpConn[WP_NONE]->drainPEOutput(inst, layout.ldDonePort);
      if (layout.stDonePort >= 0)
        wpConn[WP_NONE]->drainPEOutput(inst, layout.stDonePort);
    }
  }

  // Module outputs.
  auto out_i32 = builder.addModuleOutput("out_i32", Type::i32());
  auto out_none = builder.addModuleOutput("out_none", Type::none());
  wpConn[WP_32]->drainModuleOutput(out_i32);
  wpConn[WP_NONE]->drainModuleOutput(out_none);

  // Finalize all lattices.
  for (auto &p : activePlanes)
    builder.finalizeLattice(p.lattice);

  builder.exportMLIR(outputPath);
  std::cout << "Generated " << outputPath << "\n";
  return 0;
}
