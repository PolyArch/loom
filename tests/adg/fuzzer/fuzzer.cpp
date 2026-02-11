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
// Two-pass architecture:
//   Pass 1 (--gen-cpp): Generate standalone C++ source files in Output/
//   Pass 2 (default): Build and validate ADGs directly via ADGBuilder API
//
// The test harness compiles this file, runs it (pass 2 by default), and
// validates generated MLIR. Pass 1 is available for manual inspection.
//
//===----------------------------------------------------------------------===//

#include "fuzzer_config.h"

#include <loom/adg.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace loom::adg;

// Allowed single-op names for random PE bodies.
static const char *const kOps[] = {
    "arith.addi", "arith.subi", "arith.muli", "arith.andi",
    "arith.ori",  "arith.xori", "arith.addf", "arith.subf",
    "arith.mulf",
};
static constexpr unsigned kNumOps = sizeof(kOps) / sizeof(kOps[0]);

// Value types available for random selection.
struct TypeInfo {
  const char *mlirName;  // e.g. "i32"
  const char *apiCall;   // e.g. "Type::i32()"
  std::function<Type()> make;
  bool isFloat;
};

static const TypeInfo kTypes[] = {
    {"i8", "Type::i8()", []() { return Type::i8(); }, false},
    {"i16", "Type::i16()", []() { return Type::i16(); }, false},
    {"i32", "Type::i32()", []() { return Type::i32(); }, false},
    {"i64", "Type::i64()", []() { return Type::i64(); }, false},
    {"f32", "Type::f32()", []() { return Type::f32(); }, true},
    {"f64", "Type::f64()", []() { return Type::f64(); }, true},
};
static constexpr unsigned kNumTypes = sizeof(kTypes) / sizeof(kTypes[0]);

/// Parameters for a single generated ADG.
struct FuzzParams {
  std::string name;
  unsigned typeIdx;
  unsigned opIdx;
  unsigned numPEDefs;
  unsigned numInstances;
  std::vector<unsigned> instPEDef;   // which PE def each instance uses
  std::vector<unsigned> latMax;      // per PE def
  // Connections: (src, dstPort) pairs indexed by dst instance.
  // -1 means unconnected (use module input).
  std::vector<std::vector<int>> inputSrc; // [inst][port] = src inst or -1
  unsigned numOutputs;
  std::vector<unsigned> outputSrcInst;
};

/// Derive random parameters for a single ADG.
static FuzzParams generateParams(unsigned seed, unsigned idx) {
  std::mt19937 rng(seed + idx);
  FuzzParams p;
  p.name = "fuzz_" + std::to_string(seed) + "_" + std::to_string(idx);

  p.typeIdx = rng() % kNumTypes;
  if (kTypes[p.typeIdx].isFloat) {
    p.opIdx = 6 + rng() % 3;
  } else {
    p.opIdx = rng() % 6;
  }

  p.numPEDefs = 1 + rng() % 3;
  p.numInstances = 2 + rng() % 8;

  p.latMax.resize(p.numPEDefs);
  for (unsigned i = 0; i < p.numPEDefs; ++i)
    p.latMax[i] = 1 + rng() % 3;

  p.instPEDef.resize(p.numInstances);
  for (unsigned i = 0; i < p.numInstances; ++i)
    p.instPEDef[i] = rng() % p.numPEDefs;

  // Random acyclic connections (single-fanout: each output port used once).
  std::vector<bool> srcUsed(p.numInstances, false);
  p.inputSrc.resize(p.numInstances, std::vector<int>(2, -1));
  unsigned numConns = rng() % (p.numInstances - 1);
  for (unsigned c = 0; c < numConns; ++c) {
    if (p.numInstances < 2) break;
    unsigned src = rng() % (p.numInstances - 1);
    unsigned dst = src + 1 + rng() % (p.numInstances - src - 1);
    unsigned dstPort = rng() % 2;
    if (p.inputSrc[dst][dstPort] == -1 && !srcUsed[src]) {
      p.inputSrc[dst][dstPort] = (int)src;
      srcUsed[src] = true;
    }
  }

  p.numOutputs = 1 + rng() % 3;
  p.outputSrcInst.resize(p.numOutputs);
  for (unsigned o = 0; o < p.numOutputs; ++o) {
    unsigned base = rng() % p.numInstances;
    bool found = false;
    for (unsigned tries = 0; tries < p.numInstances; ++tries) {
      unsigned candidate = (base + tries) % p.numInstances;
      if (!srcUsed[candidate]) {
        p.outputSrcInst[o] = candidate;
        srcUsed[candidate] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      p.numOutputs = o;
      p.outputSrcInst.resize(o);
      break;
    }
  }

  // Ensure at least one module output exists.
  if (p.numOutputs == 0) {
    p.numOutputs = 1;
    // Reclaim instance 0 from any internal connection to use as output.
    for (unsigned i = 0; i < p.numInstances; ++i) {
      for (unsigned port = 0; port < 2; ++port) {
        if (p.inputSrc[i][port] == 0)
          p.inputSrc[i][port] = -1;
      }
    }
    p.outputSrcInst = {0};
  }

  return p;
}

/// Build an ADG directly from parameters and export MLIR.
static void buildAndExport(const FuzzParams &p) {
  ADGBuilder builder(p.name);
  Type valType = kTypes[p.typeIdx].make();

  struct PEDefInfo { PEHandle handle; };
  std::vector<PEDefInfo> peDefs;
  for (unsigned i = 0; i < p.numPEDefs; ++i) {
    auto pe = builder.newPE("pe_def_" + std::to_string(i))
        .setLatency(1, 1, p.latMax[i])
        .setInterval(1, 1, 1)
        .setInputPorts({valType, valType})
        .setOutputPorts({valType})
        .addOp(kOps[p.opIdx]);
    peDefs.push_back({pe});
  }

  std::vector<InstanceHandle> insts;
  for (unsigned i = 0; i < p.numInstances; ++i) {
    auto h = builder.clone(peDefs[p.instPEDef[i]].handle,
                           "inst_" + std::to_string(i));
    insts.push_back(h);
  }

  unsigned moduleInputIdx = 0;
  for (unsigned i = 0; i < p.numInstances; ++i) {
    for (unsigned port = 0; port < 2; ++port) {
      if (p.inputSrc[i][port] >= 0) {
        builder.connectPorts(insts[p.inputSrc[i][port]], 0,
                             insts[i], port);
      } else {
        auto mPort = builder.addModuleInput(
            "in_" + std::to_string(moduleInputIdx++), valType);
        builder.connectToModuleInput(mPort, insts[i], port);
      }
    }
  }

  // Ensure all PE output ports are consumed. Track which instances have their
  // output connected as a source in internal connections or module outputs.
  std::vector<bool> outputUsed(p.numInstances, false);
  for (unsigned i = 0; i < p.numInstances; ++i) {
    for (unsigned port = 0; port < 2; ++port) {
      if (p.inputSrc[i][port] >= 0)
        outputUsed[p.inputSrc[i][port]] = true;
    }
  }

  for (unsigned o = 0; o < p.numOutputs; ++o) {
    auto port = builder.addModuleOutput(
        "out_" + std::to_string(o), valType);
    builder.connectToModuleOutput(insts[p.outputSrcInst[o]], 0, port);
    outputUsed[p.outputSrcInst[o]] = true;
  }

  // Create module outputs for any instance whose output is still dangling.
  for (unsigned i = 0; i < p.numInstances; ++i) {
    if (!outputUsed[i]) {
      auto port = builder.addModuleOutput(
          "sink_" + std::to_string(i), valType);
      builder.connectToModuleOutput(insts[i], 0, port);
    }
  }

  builder.exportMLIR("Output/" + p.name + ".fabric.mlir");
}

/// Generate a standalone C++ source file for an ADG in its own subdirectory.
static void writeCppSource(const FuzzParams &p, const std::string &outDir) {
  // Create per-case subdirectory: Output/<name>/<name>.cpp
  std::string caseDir = outDir + "/" + p.name;
  // Use system() for mkdir since we're just in a test tool.
  std::string mkdirCmd = "mkdir -p " + caseDir;
  (void)std::system(mkdirCmd.c_str());
  std::string path = caseDir + "/" + p.name + ".cpp";
  std::ofstream os(path);

  os << "#include <loom/adg.h>\n";
  os << "using namespace loom::adg;\n\n";
  os << "int main() {\n";
  os << "  ADGBuilder builder(\"" << p.name << "\");\n";
  os << "  Type valType = " << kTypes[p.typeIdx].apiCall << ";\n\n";

  // PE definitions.
  for (unsigned i = 0; i < p.numPEDefs; ++i) {
    os << "  auto pe_def_" << i << " = builder.newPE(\"pe_def_"
       << i << "\")\n";
    os << "      .setLatency(1, 1, " << p.latMax[i] << ")\n";
    os << "      .setInterval(1, 1, 1)\n";
    os << "      .setInputPorts({valType, valType})\n";
    os << "      .setOutputPorts({valType})\n";
    os << "      .addOp(\"" << kOps[p.opIdx] << "\");\n\n";
  }

  // Instances.
  for (unsigned i = 0; i < p.numInstances; ++i) {
    os << "  auto inst_" << i << " = builder.clone(pe_def_"
       << p.instPEDef[i] << ", \"inst_" << i << "\");\n";
  }
  os << "\n";

  // Connections.
  unsigned moduleInputIdx = 0;
  for (unsigned i = 0; i < p.numInstances; ++i) {
    for (unsigned port = 0; port < 2; ++port) {
      if (p.inputSrc[i][port] >= 0) {
        os << "  builder.connectPorts(inst_" << p.inputSrc[i][port]
           << ", 0, inst_" << i << ", " << port << ");\n";
      } else {
        os << "  auto in_" << moduleInputIdx
           << " = builder.addModuleInput(\"in_" << moduleInputIdx
           << "\", valType);\n";
        os << "  builder.connectToModuleInput(in_" << moduleInputIdx
           << ", inst_" << i << ", " << port << ");\n";
        moduleInputIdx++;
      }
    }
  }
  os << "\n";

  for (unsigned o = 0; o < p.numOutputs; ++o) {
    os << "  auto out_" << o << " = builder.addModuleOutput(\"out_" << o
       << "\", valType);\n";
    os << "  builder.connectToModuleOutput(inst_" << p.outputSrcInst[o]
       << ", 0, out_" << o << ");\n";
  }
  os << "\n";

  // Compute which outputs are used and emit sinks for dangling ones.
  std::vector<bool> genOutputUsed(p.numInstances, false);
  for (unsigned i = 0; i < p.numInstances; ++i) {
    for (unsigned port = 0; port < 2; ++port) {
      if (p.inputSrc[i][port] >= 0)
        genOutputUsed[p.inputSrc[i][port]] = true;
    }
  }
  for (unsigned o = 0; o < p.numOutputs; ++o)
    genOutputUsed[p.outputSrcInst[o]] = true;

  unsigned sinkIdx = 0;
  for (unsigned i = 0; i < p.numInstances; ++i) {
    if (!genOutputUsed[i]) {
      os << "  auto sink_" << sinkIdx
         << " = builder.addModuleOutput(\"sink_" << sinkIdx
         << "\", valType);\n";
      os << "  builder.connectToModuleOutput(inst_" << i
         << ", 0, sink_" << sinkIdx << ");\n";
      sinkIdx++;
    }
  }
  os << "\n";

  os << "  builder.exportMLIR(\"Output/" << p.name << ".fabric.mlir\");\n";
  // Note: the generated binary runs from within Output/<name>/, so its
  // Output/ subdirectory is at Output/<name>/Output/.
  os << "  return 0;\n";
  os << "}\n";
}

int main(int argc, char **argv) {
  unsigned seed = FUZZER_SEED;
  unsigned count = FUZZER_COUNT;

  // Allow env override.
  if (const char *s = std::getenv("LOOM_FUZZER_SEED"))
    seed = std::atoi(s);
  if (const char *c = std::getenv("LOOM_FUZZER_COUNT"))
    count = std::atoi(c);

  // Check for --gen-cpp mode.
  bool genCpp = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--gen-cpp")
      genCpp = true;
  }

  for (unsigned idx = 0; idx < count; ++idx) {
    FuzzParams p = generateParams(seed, idx);

    if (genCpp) {
      writeCppSource(p, "Output");
    } else {
      buildAndExport(p);
    }
  }

  return 0;
}
