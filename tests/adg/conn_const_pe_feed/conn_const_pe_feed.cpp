//===-- conn_const_pe_feed.cpp - ADG test: const PE -> compute PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_const_pe_feed");

  auto cpe = builder.newConstantPE("const_i32")
      .setOutputType(Type::i32());

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto c0 = builder.clone(cpe, "c0");
  auto add0 = builder.clone(adder, "add0");

  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto x = builder.addModuleInput("x", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(ctrl, c0, 0);
  builder.connectPorts(c0, 0, add0, 0);
  builder.connectToModuleInput(x, add0, 1);
  builder.connectToModuleOutput(add0, 0, out);

  builder.exportMLIR("Output/conn_const_pe_feed.fabric.mlir");
  return 0;
}
