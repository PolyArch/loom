//===-- gen_medium.cpp - Medium CGRA template generator --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Generates loom_cgra_medium.fabric.mlir using the ADGBuilder C++ API with
// latticeMesh topology. Each type plane gets its own lattice mesh with
// 16-port switches to stay within the 32-port limit.
//
//===----------------------------------------------------------------------===//

#include "cgra_pe_catalog.h"

#include <iostream>
#include <map>
#include <string>

static const std::map<std::string, int> MEDIUM_INSTANCES = {
    {"pe_const_i32", 7}, {"pe_const_f32", 6}, {"pe_const_i64", 2},
    {"pe_const_index", 18}, {"pe_const_i1", 2}, {"pe_const_i16", 1},
    {"pe_join1", 5}, {"pe_join2", 3}, {"pe_join3", 2}, {"pe_join4", 1},
    {"pe_join5", 1}, {"pe_join6", 1},
    {"pe_join_i1", 1},
    {"pe_cond_br_none", 17}, {"pe_cond_br_i32", 3}, {"pe_cond_br_f32", 4},
    {"pe_cond_br_index", 1},
    {"pe_mux_i32", 4}, {"pe_mux_f32", 3}, {"pe_mux_none", 9},
    {"pe_mux_index", 5}, {"pe_mux_i64", 1},
    {"pe_addi", 12}, {"pe_subi", 2}, {"pe_muli", 9}, {"pe_divui", 2},
    {"pe_divsi", 1}, {"pe_remui", 2}, {"pe_remsi", 1},
    {"pe_addi_i64", 2}, {"pe_subi_i64", 1}, {"pe_muli_i64", 1},
    {"pe_cmpi_i64", 2}, {"pe_shli_i64", 1}, {"pe_remui_i64", 1},
    {"pe_addi_index", 2}, {"pe_subi_index", 1},
    {"pe_divui_index", 1}, {"pe_divsi_index", 1},
    {"pe_remui_index", 1}, {"pe_muli_index", 1},
    {"pe_andi", 5}, {"pe_ori", 4}, {"pe_xori", 4},
    {"pe_shli", 4}, {"pe_shrui", 4}, {"pe_shrsi", 1},
    {"pe_xori_i1", 1},
    {"pe_addf", 4}, {"pe_subf", 3}, {"pe_mulf", 4}, {"pe_divf", 3},
    {"pe_fma", 7},
    {"pe_negf", 1}, {"pe_absf", 1}, {"pe_sin", 1}, {"pe_cos", 5},
    {"pe_exp", 1}, {"pe_sqrt", 1}, {"pe_log2", 1},
    {"pe_cmpi", 10}, {"pe_cmpf", 2},
    {"pe_select", 3}, {"pe_select_index", 8}, {"pe_select_f32", 2},
    {"pe_index_cast_i32", 15}, {"pe_index_cast_i64", 8},
    {"pe_index_cast_to_i32", 2}, {"pe_index_cast_to_i64", 3},
    {"pe_index_castui", 2},
    {"pe_extui", 9}, {"pe_trunci", 6},
    {"pe_extui_i1", 4}, {"pe_trunci_to_i1", 2},
    {"pe_extui_i16", 1}, {"pe_trunci_to_i16", 1}, {"pe_remui_i16", 1},
    {"pe_uitofp", 4}, {"pe_uitofp_i16", 1}, {"pe_sitofp", 1}, {"pe_fptoui", 1},
    {"pe_stream", 4}, {"pe_gate", 5}, {"pe_gate_f32", 4},
    {"pe_gate_index", 4},
    {"pe_carry", 3}, {"pe_carry_f32", 4}, {"pe_carry_none", 8},
    {"pe_carry_index", 2},
    {"pe_invariant", 4}, {"pe_invariant_i1", 3}, {"pe_invariant_none", 3},
    {"pe_invariant_f32", 2}, {"pe_invariant_index", 2}, {"pe_invariant_i64", 1},
    {"pe_load", 5}, {"pe_load_f32", 6}, {"pe_store", 3}, {"pe_store_f32", 2},
    {"pe_sink_i1", 4}, {"pe_sink_none", 4}, {"pe_sink_i32", 2},
    {"pe_sink_index", 2}, {"pe_sink_f32", 2},
};

struct MemConfig {
  std::string elemType;
  int ldCount;
  int stCount;
};

static const MemConfig EXTMEM_MEDIUM[] = {
    {"i32", 1, 0}, {"i32", 1, 0}, {"i32", 0, 1}, {"i32", 0, 1},
    {"i32", 1, 1}, {"i32", 2, 0}, {"i32", 1, 2}, {"i32", 0, 2},
    {"i32", 3, 0},
    {"f32", 1, 0}, {"f32", 1, 0}, {"f32", 1, 0}, {"f32", 1, 0},
    {"f32", 1, 0}, {"f32", 0, 1}, {"f32", 0, 1}, {"f32", 1, 1},
    {"f32", 2, 0}, {"f32", 2, 0}, {"f32", 0, 2}, {"f32", 3, 0},
};

static const MemConfig PRIVMEM_MEDIUM[] = {
    {"i32", 1, 1}, {"i32", 1, 0}, {"f32", 1, 1}, {"f32", 1, 1},
};

int main(int argc, char *argv[]) {
  std::string outputPath = "loom_cgra_medium.fabric.mlir";
  if (argc > 1) outputPath = argv[1];

  ADGBuilder builder("loom_cgra_medium");

  auto catalog = registerAllPEs(builder);

  // Count connections per type plane from PE instances
  int planeConns[TP_COUNT];
  countPlaneConnections(MEDIUM_INSTANCES, catalog, planeConns);

  // Module I/O: 4 i32, 1 i64, 1 index, 1 f32 inputs, 1 ctrl none
  //             1 i32 output, 1 f32 output, 1 none output
  planeConns[TP_I32] += 4 + 1;
  planeConns[TP_I64] += 1;
  planeConns[TP_INDEX] += 1;
  planeConns[TP_F32] += 1 + 1;
  planeConns[TP_NONE] += 1 + 1;

  // Memory connections (only native ports go through lattice)
  for (auto &em : EXTMEM_MEDIUM)
    countMemPlaneConns(em.elemType, em.ldCount, em.stCount, true, planeConns);
  for (auto &pm : PRIVMEM_MEDIUM)
    countMemPlaneConns(pm.elemType, pm.ldCount, pm.stCount, false, planeConns);

  const int PORTS_PER_SW = 32;

  struct PlaneInfo {
    TypePlane tp;
    SwitchHandle sw;
    LatticeMeshResult lattice;
    LatticeConnector *connector;
  };

  std::vector<PlaneInfo> planes;
  std::vector<std::unique_ptr<LatticeConnector>> connectors;

  for (int i = 0; i < TP_COUNT; ++i) {
    auto tp = static_cast<TypePlane>(i);
    if (planeConns[i] <= 0) planeConns[i] = 1;
    auto dims = computeLatticeDims(planeConns[i], PORTS_PER_SW);
    std::string swName = "sw_" + std::to_string(i);
    auto sw = builder.newSwitch(swName)
        .setPortCount(PORTS_PER_SW, PORTS_PER_SW)
        .setType(typePlaneToType(tp));
    auto lattice = builder.latticeMesh(dims.first, dims.second, sw);
    auto conn = std::make_unique<LatticeConnector>(builder, lattice,
                                                   PORTS_PER_SW);
    planes.push_back(PlaneInfo{tp, sw, lattice, conn.get()});
    connectors.push_back(std::move(conn));
  }

  for (size_t i = 0; i < planes.size(); ++i) {
    connectors[i] = std::make_unique<LatticeConnector>(
        builder, planes[i].lattice, PORTS_PER_SW);
    planes[i].connector = connectors[i].get();
  }

  std::vector<LatticeConnector *> planeConn(TP_COUNT);
  for (int i = 0; i < TP_COUNT; ++i)
    planeConn[i] = planes[i].connector;

  // Create extmemory instances with memref module inputs
  std::map<std::string, int> extMemTypeIdx;
  for (auto &em : EXTMEM_MEDIUM) {
    int idx = extMemTypeIdx[em.elemType]++;
    std::string defName = "extmem_" + em.elemType + "_def_" +
                          std::to_string(idx);
    std::string instName = "extmem_" + em.elemType + "_" +
                           std::to_string(idx);
    Type dataType = (em.elemType == "f32") ? Type::f32() : Type::i32();
    auto emBuilder = builder.newExtMemory(defName)
        .setLoadPorts(em.ldCount)
        .setStorePorts(em.stCount)
        .setShape(MemrefType::dynamic1D(dataType));
    if (em.stCount > 0)
      emBuilder.setQueueDepth(4);
    auto def = emBuilder;
    auto inst = builder.clone(def, instName);

    // Connect memref module input to extmem port 0
    std::string mrefName = "mem_" + em.elemType + "_" + std::to_string(idx);
    auto mref = builder.addModuleInput(mrefName,
                                       MemrefType::dynamic1D(dataType));
    builder.connectToModuleInput(mref, inst, 0);

    // Connect remaining data ports
    connectExtMem(builder, inst, instName, em.elemType,
                  em.ldCount, em.stCount, planeConn);
  }

  // Streaming module inputs
  auto ctrl_in = builder.addModuleInput("ctrl_in", Type::none());
  planeConn[TP_NONE]->feedModuleInput(ctrl_in);

  for (int i = 0; i < 4; ++i) {
    auto p = builder.addModuleInput("in" + std::to_string(i), Type::i32());
    planeConn[TP_I32]->feedModuleInput(p);
  }
  {
    auto p = builder.addModuleInput("in_i64_0", Type::i64());
    planeConn[TP_I64]->feedModuleInput(p);
  }
  {
    auto p = builder.addModuleInput("addr0", Type::index());
    planeConn[TP_INDEX]->feedModuleInput(p);
  }
  {
    auto p = builder.addModuleInput("in_f32_0", Type::f32());
    planeConn[TP_F32]->feedModuleInput(p);
  }

  // Instantiate PEs
  for (auto &kv : MEDIUM_INSTANCES) {
    auto it = catalog.find(kv.first);
    if (it == catalog.end()) {
      std::cerr << "PE not found in catalog: " << kv.first << "\n";
      return 1;
    }
    auto &entry = it->second;
    for (int i = 0; i < kv.second; ++i) {
      std::string instName = kv.first.substr(3) + "_" + std::to_string(i);
      auto pe = builder.clone(entry.tmpl.handle, instName);

      for (size_t p = 0; p < entry.def.inputTypes.size(); ++p) {
        auto tp = typeNameToPlane(entry.def.inputTypes[p]);
        planeConn[tp]->feedPEInput(pe, p);
      }
      for (size_t p = 0; p < entry.def.outputTypes.size(); ++p) {
        auto tp = typeNameToPlane(entry.def.outputTypes[p]);
        planeConn[tp]->drainPEOutput(pe, p);
      }
    }
  }

  // Private memory
  std::map<std::string, int> privMemTypeIdx;
  for (auto &pm : PRIVMEM_MEDIUM) {
    int idx = privMemTypeIdx[pm.elemType]++;
    std::string defName = "privmem_" + pm.elemType + "_def_" +
                          std::to_string(idx);
    std::string instName = "privmem_" + pm.elemType + "_" +
                           std::to_string(idx);
    Type dataType = (pm.elemType == "f32") ? Type::f32() : Type::i32();
    auto memBuilder = builder.newMemory(defName)
        .setLoadPorts(pm.ldCount)
        .setStorePorts(pm.stCount)
        .setPrivate(true)
        .setShape(MemrefType::static1D(1024, dataType));
    if (pm.stCount > 0)
      memBuilder.setQueueDepth(4);
    auto def = memBuilder;
    auto inst = builder.clone(def, instName);
    connectPrivMem(builder, inst, instName, pm.elemType,
                   pm.ldCount, pm.stCount, planeConn);
  }

  // Module outputs
  auto out_i32 = builder.addModuleOutput("out_i32", Type::i32());
  auto out_f32 = builder.addModuleOutput("out_f32", Type::f32());
  auto out_none = builder.addModuleOutput("out_none", Type::none());

  planeConn[TP_I32]->drainModuleOutput(out_i32);
  planeConn[TP_F32]->drainModuleOutput(out_f32);
  planeConn[TP_NONE]->drainModuleOutput(out_none);

  // Finalize all lattices
  for (auto &p : planes)
    builder.finalizeLattice(p.lattice);

  builder.exportMLIR(outputPath);
  std::cout << "Generated " << outputPath << "\n";
  return 0;
}
