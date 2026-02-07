//===-- fuzzer.cpp - ADG fuzzer: generates 200 random ADGs -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Deterministic fuzzer that generates random ADG builder programs.
// Uses seeded mt19937 for reproducibility. Each generated ADG is acyclic
// with all ports connected.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace loom::adg;

static constexpr unsigned FUZZER_SEED = 42;
static constexpr unsigned FUZZER_COUNT = 200;

// Allowed single-op names for random PE bodies.
static const char *const kOps[] = {
    "arith.addi", "arith.subi", "arith.muli", "arith.andi",
    "arith.ori",  "arith.xori", "arith.addf", "arith.subf",
    "arith.mulf",
};
static constexpr unsigned kNumOps = sizeof(kOps) / sizeof(kOps[0]);

// Value types available for random selection.
struct TypeInfo {
  const char *name;
  std::function<Type()> make;
  bool isFloat;
};

static const TypeInfo kTypes[] = {
    {"i8", []() { return Type::i8(); }, false},
    {"i16", []() { return Type::i16(); }, false},
    {"i32", []() { return Type::i32(); }, false},
    {"i64", []() { return Type::i64(); }, false},
    {"f32", []() { return Type::f32(); }, true},
    {"f64", []() { return Type::f64(); }, true},
};
static constexpr unsigned kNumTypes = sizeof(kTypes) / sizeof(kTypes[0]);

int main() {
  unsigned seed = FUZZER_SEED;
  unsigned count = FUZZER_COUNT;

  // Allow env override.
  if (const char *s = std::getenv("LOOM_FUZZER_SEED"))
    seed = std::atoi(s);
  if (const char *c = std::getenv("LOOM_FUZZER_COUNT"))
    count = std::atoi(c);

  unsigned passed = 0, failed = 0;

  for (unsigned idx = 0; idx < count; ++idx) {
    std::mt19937 rng(seed + idx);

    std::string name = "fuzz_" + std::to_string(seed) + "_" + std::to_string(idx);
    ADGBuilder builder(name);

    // Pick a random value type.
    unsigned typeIdx = rng() % kNumTypes;
    const auto &ti = kTypes[typeIdx];
    Type valType = ti.make();

    // Pick a compatible op.
    unsigned opIdx;
    if (ti.isFloat) {
      // Only float ops (indices 6, 7, 8).
      opIdx = 6 + rng() % 3;
    } else {
      // Only int ops (indices 0-5).
      opIdx = rng() % 6;
    }

    // All PEs use 2 inputs for simplicity (all arith ops are binary).
    unsigned numPEDefs = 1 + rng() % 3;     // 1-3 PE definitions
    unsigned numInstances = 2 + rng() % 8;   // 2-9 instances

    // Create PE definitions (all with exactly 2 inputs).
    struct PEDefInfo {
      PEHandle handle;
      unsigned numIn;
    };
    std::vector<PEDefInfo> peDefs;
    for (unsigned p = 0; p < numPEDefs; ++p) {
      unsigned nIn = 2; // Binary ops need exactly 2 inputs.
      std::vector<Type> inPorts(nIn, valType);
      auto pe = builder.newPE("pe_def_" + std::to_string(p))
          .setLatency(1, 1, 1 + rng() % 3)
          .setInterval(1, 1, 1)
          .setInputPorts(inPorts)
          .setOutputPorts({valType})
          .addOp(kOps[opIdx]);
      peDefs.push_back({pe, nIn});
    }

    // Create instances tracking actual port counts.
    struct InstInfo {
      InstanceHandle handle;
      unsigned numIn;
      unsigned numOut;
    };
    std::vector<InstInfo> insts;

    for (unsigned i = 0; i < numInstances; ++i) {
      unsigned pi = rng() % peDefs.size();
      auto h = builder.clone(peDefs[pi].handle, "inst_" + std::to_string(i));
      insts.push_back({h, peDefs[pi].numIn, 1});
    }

    // Create random acyclic connections.
    // Only connect from lower-index to higher-index instances (guarantees acyclic).
    std::vector<std::vector<bool>> inputConnected(numInstances);
    for (auto &ic : inputConnected)
      ic.resize(2, false); // max 2 inputs tracked

    unsigned numConns = rng() % (numInstances - 1);
    for (unsigned c = 0; c < numConns; ++c) {
      if (numInstances < 2) break;
      unsigned src = rng() % (numInstances - 1);
      unsigned dst = src + 1 + rng() % (numInstances - src - 1);
      unsigned dstPort = rng() % std::min(insts[dst].numIn, 2u);
      if (!inputConnected[dst][dstPort]) {
        builder.connectPorts(insts[src].handle, 0, insts[dst].handle, dstPort);
        inputConnected[dst][dstPort] = true;
      }
    }

    // Connect unconnected input ports to module inputs.
    unsigned moduleInputIdx = 0;
    for (unsigned i = 0; i < numInstances; ++i) {
      for (unsigned p = 0; p < insts[i].numIn; ++p) {
        if (!inputConnected[i][p]) {
          auto port = builder.addModuleInput(
              "in_" + std::to_string(moduleInputIdx++), valType);
          builder.connectToModuleInput(port, insts[i].handle, p);
        }
      }
    }

    // Connect at least one instance output to a module output.
    unsigned numOutputs = 1 + rng() % 3;
    for (unsigned o = 0; o < numOutputs; ++o) {
      unsigned srcInst = rng() % numInstances;
      auto port = builder.addModuleOutput(
          "out_" + std::to_string(o), valType);
      builder.connectToModuleOutput(insts[srcInst].handle, 0, port);
    }

    // Export all MLIR files flat in Output/ so the test harness glob finds them.
    std::string outPath = "Output/" + name + ".fabric.mlir";
    builder.exportMLIR(outPath);
    passed++;
  }

  // The test harness expects exit code 0 if things ran.
  // The exportMLIR calls exit(1) on failure, so if we get here, all passed.
  return 0;
}
