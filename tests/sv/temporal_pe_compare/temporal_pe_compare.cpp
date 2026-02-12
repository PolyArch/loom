//===-- temporal_pe_compare.cpp - SV test: temporal PE compare FU --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests temporal PE with a compare FU (arith.cmpi) alongside an arithmetic FU
// (arith.addi).  The temporal PE uses interface tagged(i16, i2) with 2 FUs,
// 2 instructions, and 0 registers.
//
// FU 0: arith.cmpi, (i16, i16) -> i1, compare predicate = 0 (eq)
// FU 1: arith.addi, (i16, i16) -> i16
//
// Config layout (16 bits):
//   [3:0]   FU0 cmpi predicate (4 bits)
//   [9:4]   insn0 (6 bits)
//   [15:10] insn1 (6 bits)
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_compare");

  // FU 0: arith.cmpi, (i16, i16) -> i1, latency=0, predicate=0 (eq)
  auto cmpi_fu = builder.newPE("fu_cmpi")
      .addOp("arith.cmpi")
      .setInputPorts({Type::i16(), Type::i16()})
      .setOutputPorts({Type::i1()})
      .setLatency(0, 0, 0)
      .setComparePredicate(0);

  // FU 1: arith.addi, (i16, i16) -> i16, latency=0
  auto addi_fu = builder.newPE("fu_addi")
      .addOp("arith.addi")
      .setInputPorts({Type::i16(), Type::i16()})
      .setOutputPorts({Type::i16()})
      .setLatency(0, 0, 0);

  // Temporal PE: interface tagged(i16, i2), 2 FUs, 2 instructions, 0 registers
  auto tpe = builder.newTemporalPE("tpe0")
      .setInterface(Type::tagged(Type::i16(), Type::iN(2)))
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .addFU(cmpi_fu)
      .addFU(addi_fu);

  auto t0 = builder.clone(tpe, "t0");

  // Module ports: tagged(i16, i2) = 18-bit
  auto in0 = builder.addModuleInput("in0",
      Type::tagged(Type::i16(), Type::iN(2)));
  auto in1 = builder.addModuleInput("in1",
      Type::tagged(Type::i16(), Type::iN(2)));
  auto out = builder.addModuleOutput("out",
      Type::tagged(Type::i16(), Type::iN(2)));

  builder.connectToModuleInput(in0, t0, 0);
  builder.connectToModuleInput(in1, t0, 1);
  builder.connectToModuleOutput(t0, 0, out);

  builder.exportMLIR("Output/temporal_pe_compare.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
