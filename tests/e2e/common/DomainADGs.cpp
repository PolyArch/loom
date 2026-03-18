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

static unsigned getExtMemorySwInputPortCount(unsigned ldCount,
                                             unsigned stCount) {
  return ldCount + stCount + ldCount;
}

static unsigned getExtMemorySwOutputPortCount(unsigned ldCount,
                                              unsigned stCount) {
  return ldCount + stCount * 2;
}

void buildSpatialVectorDomain(const std::string &outputPath,
                              const SpatialVectorDomainOptions &opts) {
  ADGBuilder builder(opts.moduleName);

  auto fuAddi = builder.defineBinaryFU("fu_addi", "arith.addi", "i32", "i32");
  auto fuAddiIndex =
      builder.defineBinaryFU("fu_addi_index", "arith.addi", "index", "index");
  auto fuSubi = builder.defineBinaryFU("fu_subi", "arith.subi", "i32", "i32");
  auto fuMuli = builder.defineBinaryFU("fu_muli", "arith.muli", "i32", "i32");
  auto fuDivui =
      builder.defineBinaryFU("fu_divui", "arith.divui", "i32", "i32");
  auto fuRemui =
      builder.defineBinaryFU("fu_remui", "arith.remui", "i32", "i32");
  auto fuAndi = builder.defineBinaryFU("fu_andi", "arith.andi", "i32", "i32");
  auto fuOri = builder.defineBinaryFU("fu_ori", "arith.ori", "i32", "i32");
  auto fuXori = builder.defineBinaryFU("fu_xori", "arith.xori", "i32", "i32");
  auto fuShli = builder.defineBinaryFU("fu_shli", "arith.shli", "i32", "i32");
  auto fuShrui =
      builder.defineBinaryFU("fu_shrui", "arith.shrui", "i32", "i32");
  auto fuMuliIndex =
      builder.defineBinaryFU("fu_muli_index", "arith.muli", "index", "index");
  auto fuDivuiIndex = builder.defineBinaryFU("fu_divui_index", "arith.divui",
                                             "index", "index");
  auto fuCmpi =
      builder.defineFU("fu_cmpi", {"i32", "i32"}, {"i1"}, {"arith.cmpi"});
  auto fuSelect = builder.defineFU("fu_select", {"i1", "i32", "i32"}, {"i32"},
                                   {"arith.select"});
  auto fuSelectIndex = builder.defineSelectFU("fu_select_index", "index");
  auto fuIndexCast = builder.defineIndexCastFU("fu_index_cast", "i32",
                                               "index");
  auto fuTrunci =
      builder.defineUnaryFU("fu_trunci", "arith.trunci", "i64", "i32");
  auto fuExtui =
      builder.defineUnaryFU("fu_extui", "arith.extui", "i32", "i64");

  auto fuStream = builder.defineStreamFU("fu_stream");
  auto fuGateIndex = builder.defineGateFU("fu_gate_index", "index");
  auto fuGateI32 = builder.defineGateFU("fu_gate_i32", "i32");
  auto fuGateNone = builder.defineGateFU("fu_gate_none", "none");
  auto fuInvariantIndex =
      builder.defineInvariantFU("fu_invariant_index", "index");
  auto fuInvariantI32 =
      builder.defineInvariantFU("fu_invariant_i32", "i32");
  auto fuInvariantNone =
      builder.defineInvariantFU("fu_invariant_none", "none");
  auto fuCarryIndex = builder.defineCarryFU("fu_carry_index", "index");
  auto fuCarryI32 = builder.defineCarryFU("fu_carry_i32", "i32");
  auto fuCarryNone = builder.defineCarryFU("fu_carry_none", "none");

  auto fuLoad = builder.defineLoadFU("fu_load", "index", "i32");
  auto fuStore = builder.defineStoreFU("fu_store", "index", "i32");
  auto fuConstant =
      builder.defineConstantFU("fu_constant", "index", "0 : index");
  auto fuCondBrI32 = builder.defineCondBrFU("fu_cond_br_i32", "i32");
  auto fuCondBrNone = builder.defineCondBrFU("fu_cond_br_none", "none");
  auto fuMuxI32 = builder.defineMuxFU("fu_mux_i32", "i32");
  auto fuMuxNone = builder.defineMuxFU("fu_mux_none", "none");
  auto fuJoin = builder.defineJoinFU("fu_join", 4);

  std::vector<FUHandle> allFUs = {
      fuAddi,          fuAddiIndex,    fuSubi,         fuMuli,
      fuDivui,         fuRemui,        fuAndi,         fuOri,
      fuXori,          fuShli,
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

  const unsigned kExtMemInputPorts =
      getExtMemorySwOutputPortCount(opts.maxLdCount, opts.maxStCount);
  const unsigned kExtMemOutputPorts =
      getExtMemorySwInputPortCount(opts.maxLdCount, opts.maxStCount);
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

  auto rwMem = builder.defineExtMemory("vector_rwmem", opts.maxLdCount,
                                       opts.maxStCount);

  SwitchBankDomainSpec domain;
  domain.sw = sw;
  domain.pe = pe;
  domain.numPEs = opts.numPEs;
  domain.peInputCount = kPEInputs;
  domain.peOutputCount = kPEOutputs;
  domain.extMem = rwMem;
  domain.numExtMems = opts.numExtMems;
  domain.swInputPortsPerExtMem = kExtMemOutputPorts;
  domain.swOutputPortsPerExtMem = kExtMemInputPorts;
  domain.extMemPrefix = "extmem";
  domain.extMemrefType = "memref<?xi32>";
  domain.scalarInputTypes.assign(opts.numScalarInputs,
                                 "!fabric.bits<" +
                                     std::to_string(opts.dataWidth) + ">");
  domain.scalarOutputTypes.assign(opts.numScalarOutputs,
                                  "!fabric.bits<" +
                                      std::to_string(opts.dataWidth) + ">");
  (void)builder.buildSwitchBankDomain(domain);

  builder.exportMLIR(outputPath);
}

} // namespace e2e
} // namespace fcc
