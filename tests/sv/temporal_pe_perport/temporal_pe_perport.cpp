//===-- temporal_pe_perport.cpp - SV test: per-port temporal PE --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests temporal PE with per-port widths: FU port widths narrower than the
// interface type T. The temporal PE interface uses tagged(i32, i4) but the
// FU PEs use i16 (adder) and i8 (multiplier) ports.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("temporal_pe_perport");

  // FU 0: adder with i16 ports (narrower than interface i32)
  auto adder = builder.newPE("fu_add16")
      .addOp("arith.addi")
      .setInputPorts({Type::i16(), Type::i16()})
      .setOutputPorts({Type::i16()})
      .setLatency(1, 1, 1);

  // FU 1: multiplier with i8 ports (even narrower)
  auto multiplier = builder.newPE("fu_mul8")
      .addOp("arith.muli")
      .setInputPorts({Type::i8(), Type::i8()})
      .setOutputPorts({Type::i8()})
      .setLatency(1, 1, 1);

  // Temporal PE: interface tagged(i32, i4), contains both FUs
  // Per-port widths: IN_i_DW = max(16,8) = 16, OUT_0_DW = max(16,8) = 16
  auto tpe = builder.newTemporalPE("tpe0")
      .setInterface(Type::tagged(Type::i32(), Type::iN(4)))
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .addFU(adder)
      .addFU(multiplier);

  auto t0 = builder.clone(tpe, "t0");

  // Module ports: match temporal PE interface type (tagged(i32, i4) = 36-bit)
  // The SV generator adapts between 36-bit module ports and 20-bit per-port
  // temporal PE instance ports via emitDataAssign.
  auto in0 = builder.addModuleInput("in0",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto in1 = builder.addModuleInput("in1",
      Type::tagged(Type::i32(), Type::iN(4)));
  auto out = builder.addModuleOutput("out",
      Type::tagged(Type::i32(), Type::iN(4)));

  builder.connectToModuleInput(in0, t0, 0);
  builder.connectToModuleInput(in1, t0, 1);
  builder.connectToModuleOutput(t0, 0, out);

  builder.exportMLIR("Output/temporal_pe_perport.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
