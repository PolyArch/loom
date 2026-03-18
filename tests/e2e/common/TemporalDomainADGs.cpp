//===-- TemporalDomainADGs.cpp - Temporal e2e ADG builders -----*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#include "TemporalDomainADGs.h"
#include "fcc/ADG/ADGBuilder.h"

namespace fcc {
namespace e2e {

static std::string bitsType(unsigned width) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

static std::string taggedBitsType(unsigned width) {
  return "!fabric.tagged<" + bitsType(width) + ", i1>";
}

void buildTemporalReductionDomain(const std::string &outputPath,
                                  const TemporalReductionDomainOptions &opts) {
  const std::string bitsTy = bitsType(opts.dataWidth);
  const std::string taggedTy = taggedBitsType(opts.dataWidth);
  fcc::adg::ADGBuilder builder(opts.moduleName);

  std::vector<fcc::adg::FUHandle> fus;
  fus.push_back(builder.defineJoinFU("fu_join", 1));
  fus.push_back(builder.defineJoinFU("fu_join_1", 2));
  fus.push_back(
      builder.defineConstantFU("fu_const_index", "index", "0 : index"));
  fus.push_back(
      builder.defineConstantFU("fu_const_index_1", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU("fu_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineIndexCastFU("fu_index_cast", "i32", "index"));
  fus.push_back(builder.defineStreamFU("fu_stream"));
  fus.push_back(builder.defineGateFU("fu_gate_index", "index"));
  fus.push_back(builder.defineGateFU("fu_gate_i32", "i32"));
  fus.push_back(builder.defineCarryFU("fu_carry_i32", "i32"));
  fus.push_back(builder.defineCarryFU("fu_carry_none", "none"));
  fus.push_back(builder.defineCarryFU("fu_carry_none_1", "none"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_i32", "i32"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_none", "none"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_none_1", "none"));
  fus.push_back(builder.defineLoadFU("fu_load", "index", "i32"));
  fus.push_back(builder.defineBinaryFU("fu_addi", "arith.addi", "i32", "i32"));
  fus.push_back(builder.defineCmpiFU("fu_cmpi", "i32", "eq"));
  fus.push_back(builder.defineSelectFU("fu_select_index", "index"));
  fus.push_back(builder.defineMuxFU("fu_mux_i32", "i32"));
  fus.push_back(builder.defineMuxFU("fu_mux_none", "none"));
  fus.push_back(builder.defineMuxFU("fu_mux_none_1", "none"));

  fcc::adg::TemporalPESpec peSpec;
  peSpec.name = "tpe_reduce";
  peSpec.inputTypes = {taggedTy, taggedTy, taggedTy, taggedTy, taggedTy};
  peSpec.outputTypes = {taggedTy, taggedTy, taggedTy};
  peSpec.functionUnits = fus;
  peSpec.numRegister = opts.numRegister;
  peSpec.numInstruction = opts.numInstruction;
  peSpec.regFifoDepth = opts.regFifoDepth;
  auto tpe = builder.defineTemporalPE(peSpec);

  fcc::adg::ExtMemorySpec extSpec;
  extSpec.name = "extmem_0";
  extSpec.ldPorts = 1;
  extSpec.stPorts = 0;
  extSpec.lsqDepth = 0;
  extSpec.memrefType = opts.memrefType;
  auto ext = builder.defineExtMemory(extSpec);

  auto memrefs = builder.addMemrefInputs("mem", 1, opts.memrefType);
  auto mem0 = memrefs[0];
  auto inputs = builder.addInputs("arg", {bitsTy, bitsTy, bitsTy});
  auto outputs = builder.addOutputs("out", {bitsTy, bitsTy});
  auto n = inputs[0];
  auto init = inputs[1];
  auto ctrl = inputs[2];
  auto sumOut = outputs[0];
  auto doneOut = outputs[1];

  auto extInsts = builder.instantiateExtMemArray(1, ext, "ext");
  auto extInst = extInsts[0];
  builder.connectMemrefToExtMem(mem0, extInst);

  auto tags =
      builder.createAddTagBank(bitsTy, taggedTy, {0, 0, 0, 0, 0});
  auto tagN = tags[0];
  auto tagInit = tags[1];
  auto tagCtrl = tags[2];
  auto tagLdData = tags[3];
  auto tagLdDone = tags[4];

  builder.connectInputToInstance(n, tagN, 0);
  builder.connectInputToInstance(init, tagInit, 0);
  builder.connectInputToInstance(ctrl, tagCtrl, 0);
  builder.connect(extInst, 0, tagLdData, 0);
  builder.connect(extInst, 1, tagLdDone, 0);

  auto tpeInst = builder.instantiatePE(tpe, "tpe_0");
  builder.connect(tagN, 0, tpeInst, 0);
  builder.connect(tagInit, 0, tpeInst, 1);
  builder.connect(tagCtrl, 0, tpeInst, 2);
  builder.connect(tagLdData, 0, tpeInst, 3);
  builder.connect(tagLdDone, 0, tpeInst, 4);

  auto dels = builder.createDelTagBank(taggedTy, bitsTy, 3);
  auto delAddr = dels[0];
  auto delSum = dels[1];
  auto delDone = dels[2];
  builder.connect(tpeInst, 0, delAddr, 0);
  builder.connect(tpeInst, 1, delSum, 0);
  builder.connect(tpeInst, 2, delDone, 0);
  builder.connect(delAddr, 0, extInst, 1);
  builder.connectInstanceToOutput(delSum, 0, sumOut);
  builder.connectInstanceToOutput(delDone, 0, doneOut);

  builder.exportMLIR(outputPath);
}

void buildTemporalScanDomain(const std::string &outputPath,
                             const TemporalScanDomainOptions &opts) {
  const std::string bitsTy = bitsType(opts.dataWidth);
  const std::string taggedTy = taggedBitsType(opts.dataWidth);
  fcc::adg::ADGBuilder builder(opts.moduleName);

  std::vector<fcc::adg::FUHandle> fus;
  fus.push_back(builder.defineJoinFU("fu_join", 1));
  fus.push_back(builder.defineJoinFU("fu_join_1", 2));
  fus.push_back(
      builder.defineConstantFU("fu_const_index", "index", "0 : index"));
  fus.push_back(
      builder.defineConstantFU("fu_const_index_1", "index", "1 : index"));
  fus.push_back(builder.defineConstantFU("fu_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineIndexCastFU("fu_index_cast", "i32", "index"));
  fus.push_back(builder.defineStreamFU("fu_stream"));
  fus.push_back(builder.defineGateFU("fu_gate_index", "index"));
  fus.push_back(builder.defineGateFU("fu_gate_i32", "i32"));
  fus.push_back(builder.defineCarryFU("fu_carry_i32", "i32"));
  fus.push_back(builder.defineCarryFU("fu_carry_none", "none"));
  fus.push_back(builder.defineCarryFU("fu_carry_none_1", "none"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_none", "none"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_none_1", "none"));
  fus.push_back(builder.defineCondBrFU("fu_cond_br_none_2", "none"));
  fus.push_back(builder.defineLoadFU("fu_load", "index", "i32"));
  fus.push_back(builder.defineStoreFU("fu_store", "index", "i32"));
  fus.push_back(builder.defineBinaryFU("fu_addi", "arith.addi", "i32", "i32"));
  fus.push_back(builder.defineCmpiFU("fu_cmpi", "i32", "eq"));
  fus.push_back(builder.defineSelectFU("fu_select_index", "index"));
  fus.push_back(builder.defineMuxFU("fu_mux_none", "none"));
  fus.push_back(builder.defineMuxFU("fu_mux_none_1", "none"));

  fcc::adg::TemporalPESpec peSpec;
  peSpec.name = "tpe_scan";
  peSpec.inputTypes = {taggedTy, taggedTy, taggedTy, taggedTy, taggedTy};
  peSpec.outputTypes = {taggedTy, taggedTy, taggedTy, taggedTy};
  peSpec.functionUnits = fus;
  peSpec.numRegister = opts.numRegister;
  peSpec.numInstruction = opts.numInstruction;
  peSpec.regFifoDepth = opts.regFifoDepth;
  auto tpe = builder.defineTemporalPE(peSpec);

  fcc::adg::ExtMemorySpec memInSpec;
  memInSpec.name = "extmem_in";
  memInSpec.ldPorts = 1;
  memInSpec.stPorts = 0;
  memInSpec.lsqDepth = 0;
  memInSpec.memrefType = opts.memrefType;
  auto extIn = builder.defineExtMemory(memInSpec);

  fcc::adg::ExtMemorySpec memOutSpec;
  memOutSpec.name = "extmem_out";
  memOutSpec.ldPorts = 0;
  memOutSpec.stPorts = 1;
  memOutSpec.lsqDepth = 0;
  memOutSpec.memrefType = opts.memrefType;
  auto extOut = builder.defineExtMemory(memOutSpec);

  auto memrefs = builder.addMemrefInputs("mem", 2, opts.memrefType);
  auto memIn = memrefs[0];
  auto memOut = memrefs[1];
  auto inputs = builder.addInputs("arg", {bitsTy, bitsTy});
  auto outputs = builder.addOutputs("out", {bitsTy});
  auto n = inputs[0];
  auto ctrl = inputs[1];
  auto doneOut = outputs[0];

  auto ldInsts = builder.instantiateExtMemArray(1, extIn, "ld");
  auto stInsts = builder.instantiateExtMemArray(1, extOut, "st");
  auto ldInst = ldInsts[0];
  auto stInst = stInsts[0];
  builder.connectMemrefToExtMem(memIn, ldInst);
  builder.connectMemrefToExtMem(memOut, stInst);

  auto tags = builder.createAddTagBank(bitsTy, taggedTy, {0, 0, 0, 0, 0});
  auto tagN = tags[0];
  auto tagCtrl = tags[1];
  auto tagLdData = tags[2];
  auto tagLdDone = tags[3];
  auto tagStDone = tags[4];
  builder.connectInputToInstance(n, tagN, 0);
  builder.connectInputToInstance(ctrl, tagCtrl, 0);
  builder.connect(ldInst, 0, tagLdData, 0);
  builder.connect(ldInst, 1, tagLdDone, 0);
  builder.connect(stInst, 0, tagStDone, 0);

  auto tpeInst = builder.instantiatePE(tpe, "tpe_0");
  builder.connect(tagN, 0, tpeInst, 0);
  builder.connect(tagCtrl, 0, tpeInst, 1);
  builder.connect(tagLdData, 0, tpeInst, 2);
  builder.connect(tagLdDone, 0, tpeInst, 3);
  builder.connect(tagStDone, 0, tpeInst, 4);

  auto dels = builder.createDelTagBank(taggedTy, bitsTy, 4);
  auto delLdAddr = dels[0];
  auto delStData = dels[1];
  auto delStAddr = dels[2];
  auto delDone = dels[3];
  builder.connect(tpeInst, 0, delLdAddr, 0);
  builder.connect(tpeInst, 1, delStData, 0);
  builder.connect(tpeInst, 2, delStAddr, 0);
  builder.connect(tpeInst, 3, delDone, 0);
  builder.connect(delLdAddr, 0, ldInst, 1);
  builder.connect(delStAddr, 0, stInst, 1);
  builder.connect(delStData, 0, stInst, 2);
  builder.connectInstanceToOutput(delDone, 0, doneOut);

  builder.exportMLIR(outputPath);
}

} // namespace e2e
} // namespace fcc
