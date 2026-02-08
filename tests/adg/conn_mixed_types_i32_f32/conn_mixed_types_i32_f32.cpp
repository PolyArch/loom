//===-- conn_mixed_types_i32_f32.cpp - ADG test: separate i32/f32 paths -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_mixed_types_i32_f32");

  auto adder_i32 = builder.newPE("adder_i32")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto adder_f32 = builder.newPE("adder_f32")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f32(), Type::f32()})
      .setOutputPorts({Type::f32()})
      .addOp("arith.addf");

  auto iadd0 = builder.clone(adder_i32, "iadd0");
  auto fadd0 = builder.clone(adder_f32, "fadd0");

  // Integer datapath
  auto ia = builder.addModuleInput("ia", Type::i32());
  auto ib = builder.addModuleInput("ib", Type::i32());
  auto iout = builder.addModuleOutput("iresult", Type::i32());

  // Float datapath
  auto fa = builder.addModuleInput("fa", Type::f32());
  auto fb = builder.addModuleInput("fb", Type::f32());
  auto fout = builder.addModuleOutput("fresult", Type::f32());

  // i32 chain
  builder.connectToModuleInput(ia, iadd0, 0);
  builder.connectToModuleInput(ib, iadd0, 1);
  builder.connectToModuleOutput(iadd0, 0, iout);

  // f32 chain
  builder.connectToModuleInput(fa, fadd0, 0);
  builder.connectToModuleInput(fb, fadd0, 1);
  builder.connectToModuleOutput(fadd0, 0, fout);

  builder.exportMLIR("Output/conn_mixed_types_i32_f32.fabric.mlir");
  return 0;
}
