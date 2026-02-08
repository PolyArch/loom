//===-- mixed_all_pe_types.cpp - ADG test: compute + const + load + store PE -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_all_pe_types");

  auto compute = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto cpe = builder.newConstantPE("const_gen")
      .setOutputType(Type::i32());

  auto lpe = builder.newLoadPE("loader")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto spe = builder.newStorePE("storer")
      .setDataType(Type::i32())
      .setInterfaceCategory(InterfaceCategory::Native);

  auto comp0 = builder.clone(compute, "comp0");
  auto c0 = builder.clone(cpe, "c0");
  auto ld0 = builder.clone(lpe, "ld0");
  auto st0 = builder.clone(spe, "st0");

  // Module I/O
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto x = builder.addModuleInput("x", Type::i32());
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto ld_data = builder.addModuleInput("ld_data", Type::i32());
  auto ld_ctrl = builder.addModuleInput("ld_ctrl", Type::none());
  auto st_addr = builder.addModuleInput("st_addr", Type::index());
  auto st_ctrl = builder.addModuleInput("st_ctrl", Type::none());
  auto comp_result = builder.addModuleOutput("comp_result", Type::i32());
  auto ld_out = builder.addModuleOutput("ld_out", Type::i32());
  auto ld_addr_out = builder.addModuleOutput("ld_addr_out", Type::index());
  auto st_addr_out = builder.addModuleOutput("st_addr_out", Type::index());
  auto st_done = builder.addModuleOutput("st_done", Type::none());

  // Constant PE: ctrl -> const -> compute input 0
  builder.connectToModuleInput(ctrl, c0, 0);
  builder.connectPorts(c0, 0, comp0, 0);
  builder.connectToModuleInput(x, comp0, 1);
  builder.connectToModuleOutput(comp0, 0, comp_result);

  // Load PE: addr, data_in, ctrl -> data_out, addr_out
  builder.connectToModuleInput(ld_addr, ld0, 0);
  builder.connectToModuleInput(ld_data, ld0, 1);
  builder.connectToModuleInput(ld_ctrl, ld0, 2);
  builder.connectToModuleOutput(ld0, 0, ld_out);
  builder.connectToModuleOutput(ld0, 1, ld_addr_out);

  // Store PE: addr, data, ctrl -> addr_out, done
  builder.connectToModuleInput(st_addr, st0, 0);
  builder.connectToModuleInput(x, st0, 1);
  builder.connectToModuleInput(st_ctrl, st0, 2);
  builder.connectToModuleOutput(st0, 0, st_addr_out);
  builder.connectToModuleOutput(st0, 1, st_done);

  builder.exportMLIR("Output/mixed_all_pe_types.fabric.mlir");
  return 0;
}
