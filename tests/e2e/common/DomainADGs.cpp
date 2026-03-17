//===-- DomainADGs.cpp - Reusable e2e ADG builders -------------*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "DomainADGs.h"

#include "fcc/ADG/ADGBuilder.h"

#include <string>
#include <vector>

using namespace fcc::adg;

namespace fcc {
namespace e2e {

void buildSpatialVectorDomain(const std::string &outputPath,
                              const SpatialVectorDomainOptions &opts) {
  ADGBuilder builder(opts.moduleName);

  auto fuAddi =
      builder.defineFU("fu_addi", {"i32", "i32"}, {"i32"}, {"arith.addi"});
  auto fuAddiIndex = builder.defineFU("fu_addi_index", {"index", "index"},
                                      {"index"}, {"arith.addi"});
  auto fuSubi =
      builder.defineFU("fu_subi", {"i32", "i32"}, {"i32"}, {"arith.subi"});
  auto fuMuli =
      builder.defineFU("fu_muli", {"i32", "i32"}, {"i32"}, {"arith.muli"});
  auto fuAndi =
      builder.defineFU("fu_andi", {"i32", "i32"}, {"i32"}, {"arith.andi"});
  auto fuOri =
      builder.defineFU("fu_ori", {"i32", "i32"}, {"i32"}, {"arith.ori"});
  auto fuXori =
      builder.defineFU("fu_xori", {"i32", "i32"}, {"i32"}, {"arith.xori"});
  auto fuShli =
      builder.defineFU("fu_shli", {"i32", "i32"}, {"i32"}, {"arith.shli"});
  auto fuShrui =
      builder.defineFU("fu_shrui", {"i32", "i32"}, {"i32"}, {"arith.shrui"});
  auto fuMuliIndex = builder.defineFU("fu_muli_index", {"index", "index"},
                                      {"index"}, {"arith.muli"});
  auto fuDivuiIndex = builder.defineFU("fu_divui_index", {"index", "index"},
                                       {"index"}, {"arith.divui"});
  auto fuCmpi =
      builder.defineFU("fu_cmpi", {"i32", "i32"}, {"i1"}, {"arith.cmpi"});
  auto fuSelect = builder.defineFU("fu_select", {"i1", "i32", "i32"}, {"i32"},
                                   {"arith.select"});
  auto fuSelectIndex =
      builder.defineFU("fu_select_index", {"i1", "index", "index"}, {"index"},
                       {"arith.select"});
  auto fuIndexCast = builder.defineFU("fu_index_cast", {"i32"}, {"index"},
                                      {"arith.index_cast"});
  auto fuTrunci = builder.defineFU("fu_trunci", {"i64"}, {"i32"},
                                   {"arith.trunci"});
  auto fuExtui = builder.defineFU("fu_extui", {"i32"}, {"i64"},
                                  {"arith.extui"});

  auto fuStream =
      builder.defineFU("fu_stream", {"index", "index", "index"},
                       {"index", "i1"}, {"dataflow.stream"});
  auto fuGateIndex = builder.defineFU("fu_gate_index", {"index", "i1"},
                                      {"index", "i1"}, {"dataflow.gate"});
  auto fuGateI32 = builder.defineFU("fu_gate_i32", {"i32", "i1"},
                                    {"i32", "i1"}, {"dataflow.gate"});
  auto fuGateNone = builder.defineFU("fu_gate_none", {"none", "i1"},
                                     {"none", "i1"}, {"dataflow.gate"});
  auto fuInvariantIndex =
      builder.defineFU("fu_invariant_index", {"i1", "index"}, {"index"},
                       {"dataflow.invariant"});
  auto fuInvariantI32 =
      builder.defineFU("fu_invariant_i32", {"i1", "i32"}, {"i32"},
                       {"dataflow.invariant"});
  auto fuInvariantNone =
      builder.defineFU("fu_invariant_none", {"i1", "none"}, {"none"},
                       {"dataflow.invariant"});
  auto fuCarryIndex = builder.defineFU("fu_carry_index",
                                       {"i1", "index", "index"}, {"index"},
                                       {"dataflow.carry"});
  auto fuCarryI32 = builder.defineFU("fu_carry_i32", {"i1", "i32", "i32"},
                                     {"i32"}, {"dataflow.carry"});
  auto fuCarryNone = builder.defineFU("fu_carry_none", {"i1", "none", "none"},
                                      {"none"}, {"dataflow.carry"});

  auto fuLoad =
      builder.defineFU("fu_load", {"index", "i32", "none"}, {"i32", "index"},
                       {"handshake.load"});
  auto fuStore =
      builder.defineFU("fu_store", {"index", "i32", "none"}, {"i32", "index"},
                       {"handshake.store"});
  auto fuConstant = builder.defineFU("fu_constant", {"none"}, {"index"},
                                     {"handshake.constant"});
  auto fuCondBrI32 = builder.defineFU("fu_cond_br_i32", {"i1", "i32"},
                                      {"i32", "i32"}, {"handshake.cond_br"});
  auto fuCondBrNone = builder.defineFU("fu_cond_br_none", {"i1", "none"},
                                       {"none", "none"},
                                       {"handshake.cond_br"});
  auto fuMuxI32 = builder.defineFU("fu_mux_i32", {"index", "i32", "i32"},
                                   {"i32"}, {"handshake.mux"});
  auto fuMuxNone = builder.defineFU("fu_mux_none",
                                    {"index", "none", "none"}, {"none"},
                                    {"handshake.mux"});
  auto fuJoin = builder.defineFU("fu_join", {"none", "none", "none", "none"},
                                 {"none"}, {"handshake.join"});

  std::vector<FUHandle> allFUs = {
      fuAddi,          fuAddiIndex,    fuSubi,         fuMuli,
      fuAndi,          fuOri,          fuXori,         fuShli,
      fuShrui,
      fuMuliIndex,     fuDivuiIndex,   fuCmpi,         fuSelect,
      fuSelectIndex,   fuIndexCast,    fuTrunci,       fuExtui,        fuStream,
      fuGateIndex,
      fuGateI32,       fuGateNone,     fuInvariantIndex,
      fuInvariantI32,  fuInvariantNone, fuCarryIndex,  fuCarryI32,
      fuCarryNone,     fuLoad,         fuStore,        fuConstant,
      fuCondBrI32,     fuCondBrNone,   fuMuxI32,       fuMuxNone,
      fuJoin,
  };

  constexpr unsigned kPEInputs = 4;
  constexpr unsigned kPEOutputs = 4;
  auto pe = builder.defineSpatialPE("vector_pe", kPEInputs, kPEOutputs,
                                    opts.dataWidth, allFUs);

  constexpr unsigned kExtMemInputPorts = 4;
  constexpr unsigned kExtMemOutputPorts = 5;
  const unsigned numSwInputs =
      opts.numPEs * kPEOutputs +
      opts.numExtMems * kExtMemOutputPorts + opts.numScalarInputs;
  const unsigned numSwOutputs =
      opts.numPEs * kPEInputs +
      opts.numExtMems * kExtMemInputPorts + opts.numScalarOutputs;

  std::vector<unsigned> swInputWidths(numSwInputs, opts.dataWidth);
  std::vector<unsigned> swOutputWidths(numSwOutputs, opts.dataWidth);
  std::vector<std::vector<bool>> fullCrossbar(
      numSwOutputs, std::vector<bool>(numSwInputs, true));
  auto sw = builder.defineSpatialSW("vector_sw", swInputWidths, swOutputWidths,
                                    fullCrossbar);

  auto rwMem = builder.defineExtMemory("vector_rwmem", 2, 1);

  auto swInst = builder.instantiateSW(sw, "sw_0");
  std::vector<InstanceHandle> peInsts;
  peInsts.reserve(opts.numPEs);
  for (unsigned i = 0; i < opts.numPEs; ++i)
    peInsts.push_back(builder.instantiatePE(
        pe, "pe_" + std::to_string(i)));

  unsigned swInputCursor = 0;
  unsigned swOutputCursor = 0;
  for (InstanceHandle peInst : peInsts) {
    for (unsigned p = 0; p < kPEOutputs; ++p)
      builder.connect(peInst, p, swInst, swInputCursor++);
    for (unsigned p = 0; p < kPEInputs; ++p)
      builder.connect(swInst, swOutputCursor++, peInst, p);
  }

  std::vector<InstanceHandle> memInsts;
  memInsts.reserve(opts.numExtMems);
  for (unsigned i = 0; i < opts.numExtMems; ++i) {
    auto inst = builder.instantiateExtMem(
        rwMem, "extmem_" + std::to_string(i));
    memInsts.push_back(inst);

    auto memref = builder.addMemrefInput("mem_" + std::to_string(i),
                                         "memref<?xi32>");
    builder.connectMemrefToExtMem(memref, inst);
  }

  std::vector<unsigned> scalarInputs;
  scalarInputs.reserve(opts.numScalarInputs);
  for (unsigned i = 0; i < opts.numScalarInputs; ++i)
    scalarInputs.push_back(builder.addScalarInput("scalar_" + std::to_string(i),
                                                  opts.dataWidth));

  std::vector<unsigned> scalarOutputs;
  scalarOutputs.reserve(opts.numScalarOutputs);
  for (unsigned i = 0; i < opts.numScalarOutputs; ++i)
    scalarOutputs.push_back(
        builder.addScalarOutput("scalar_out_" + std::to_string(i),
                                opts.dataWidth));

  for (InstanceHandle memInst : memInsts) {
    builder.associateExtMemWithSW(memInst, swInst, swInputCursor,
                                  swOutputCursor);
    swInputCursor += kExtMemOutputPorts;
    swOutputCursor += kExtMemInputPorts;
  }

  for (unsigned i = 0; i < scalarInputs.size(); ++i)
    builder.connectScalarInputToInstance(scalarInputs[i], swInst,
                                         swInputCursor++);
  for (unsigned i = 0; i < scalarOutputs.size(); ++i)
    builder.connectInstanceToScalarOutput(swInst, swOutputCursor++,
                                          scalarOutputs[i]);

  builder.exportMLIR(outputPath);
}

} // namespace e2e
} // namespace fcc
