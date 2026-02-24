//===-- topo_lattice_dup_place.cpp - ADG test: duplicate placement --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
// Negative test: placing two PEs in the same lattice cell should trigger error.

#include <loom/adg.h>

#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

using namespace loom::adg;

static bool expectError(void (*fn)()) {
  pid_t pid = fork();
  if (pid == 0) {
    fn();
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void duplicatePlacement() {
  ADGBuilder builder("test");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(2, 2, sw);

  builder.placePEInLattice(lattice, 0, 0, pe, "pe_first");
  // Second placement in the same cell should fail.
  builder.placePEInLattice(lattice, 0, 0, pe, "pe_second");
}

int main() {
  assert(expectError(duplicatePlacement) && "duplicate cell placement should fail");

  // Build a minimal valid ADG so the test harness MLIR check passes.
  {
    ADGBuilder b("topo_lattice_dup_place");
    auto pe = b.newPE("p")
        .setLatency(1, 1, 1).setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi");
    auto inst = b.clone(pe, "i0");
    auto a = b.addModuleInput("a", Type::i32());
    auto c = b.addModuleInput("c", Type::i32());
    auto out = b.addModuleOutput("out", Type::i32());
    b.connectToModuleInput(a, inst, 0);
    b.connectToModuleInput(c, inst, 1);
    b.connectToModuleOutput(inst, 0, out);
    b.exportMLIR("Output/topo_lattice_dup_place.fabric.mlir");
  }
  return 0;
}
