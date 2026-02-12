//===-- pe_compare.cpp - SV test: runtime-configurable compare --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Tests runtime-configurable compare predicates for arith.cmpi and arith.cmpf.
// Exercises:
//   p0 - singleOp arith.cmpi (native, i32 inputs, i1 output)
//   p1 - singleOp arith.cmpf (native, f32 inputs, i1 output)
//   p2 - bodyMLIR PE with both arith.cmpi and arith.cmpf
//
// Each compare op contributes 4 config bits; cmpi generates error detection
// for predicate values >= 10.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("pe_compare");

  // p0: singleOp arith.cmpi, (i32, i32) -> i1, native
  auto pe0 = builder.newPE("cmpi_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i1()})
      .addOp("arith.cmpi")
      .setComparePredicate(2); // slt (initial config value)

  auto p0 = builder.clone(pe0, "p0");

  // p1: singleOp arith.cmpf, (f32, f32) -> i1, native
  auto pe1 = builder.newPE("cmpf_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f32(), Type::f32()})
      .setOutputPorts({Type::i1()})
      .addOp("arith.cmpf")
      .setComparePredicate(4); // olt (initial config value)

  auto p1 = builder.clone(pe1, "p1");

  // p2: bodyMLIR with both arith.cmpi and arith.cmpf,
  // (i32, i32, f32, f32) -> (i1, i1)
  // Uses separate inputs per op to avoid fork deadlock in simulation.
  auto pe2 = builder.newPE("multi_cmp_pe")
      .setLatency(0, 0, 0)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32(), Type::f32(), Type::f32()})
      .setOutputPorts({Type::i1(), Type::i1()})
      .setBodyMLIR(
          "^bb0(%a: i32, %b: i32, %c: f32, %d: f32):\n"
          "  %cmp_i = arith.cmpi slt, %a, %b : i32\n"
          "  %cmp_f = arith.cmpf olt, %c, %d : f32\n"
          "  fabric.yield %cmp_i, %cmp_f : i1, i1\n");

  auto p2 = builder.clone(pe2, "p2");

  // Module ports for p0 (cmpi)
  auto a0 = builder.addModuleInput("a0", Type::i32());
  auto b0 = builder.addModuleInput("b0", Type::i32());
  auto r0 = builder.addModuleOutput("r0", Type::i1());

  builder.connectToModuleInput(a0, p0, 0);
  builder.connectToModuleInput(b0, p0, 1);
  builder.connectToModuleOutput(p0, 0, r0);

  // Module ports for p1 (cmpf)
  auto a1 = builder.addModuleInput("a1", Type::f32());
  auto b1 = builder.addModuleInput("b1", Type::f32());
  auto r1 = builder.addModuleOutput("r1", Type::i1());

  builder.connectToModuleInput(a1, p1, 0);
  builder.connectToModuleInput(b1, p1, 1);
  builder.connectToModuleOutput(p1, 0, r1);

  // Module ports for p2 (multi-cmp): i32 pair for cmpi, f32 pair for cmpf
  auto a2 = builder.addModuleInput("a2", Type::i32());
  auto b2 = builder.addModuleInput("b2", Type::i32());
  auto c2 = builder.addModuleInput("c2", Type::f32());
  auto d2 = builder.addModuleInput("d2", Type::f32());
  auto r2_i = builder.addModuleOutput("r2_i", Type::i1());
  auto r2_f = builder.addModuleOutput("r2_f", Type::i1());

  builder.connectToModuleInput(a2, p2, 0);
  builder.connectToModuleInput(b2, p2, 1);
  builder.connectToModuleInput(c2, p2, 2);
  builder.connectToModuleInput(d2, p2, 3);
  builder.connectToModuleOutput(p2, 0, r2_i);
  builder.connectToModuleOutput(p2, 1, r2_f);

  builder.exportMLIR("Output/pe_compare.fabric.mlir");
  builder.exportSV("Output/sv");
  return 0;
}
