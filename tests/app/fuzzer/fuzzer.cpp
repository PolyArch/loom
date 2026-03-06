//===-- fuzzer.cpp - APP fuzzer: generates random loom apps -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Deterministic fuzzer that generates random C++ applications annotated with
// loom pragmas.  Uses seeded mt19937 for reproducibility.
//
// Two-pass architecture (mirrors the ADG fuzzer):
//   Pass 1 (--gen-cpp): Generate standalone test directories in Output/
//                        Each contains <name>.h, <name>.cpp, and main.cpp
//   Pass 2 (default):   Compile and run each generated app via loom --as-clang,
//                        then verify CPU vs DSA results match.
//
// Randomised parameters per test case:
//   - Data type (uint32_t, int32_t, float)
//   - Loop nesting depth (1-3)
//   - Loop body operation (+, -, *, &, |, ^  for integer;  +, -, * for float)
//   - Pragma combinations on LOOM_ACCEL, LOOM_TARGET, LOOM_PARALLEL,
//     LOOM_UNROLL, LOOM_TRIPCOUNT, LOOM_REDUCE, LOOM_STREAM, LOOM_MEMORY_BANK
//   - Number of input arrays (1-3) and whether they are LOOM_STREAM
//   - Whether the kernel is a reduction (scalar accumulator)
//   - Array sizes and initialisation seeds
//
//===----------------------------------------------------------------------===//

#include "fuzzer_config.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Data-type descriptors
// ---------------------------------------------------------------------------

struct TypeInfo {
  const char *cType;     // e.g. "uint32_t"
  const char *fmtSpec;   // printf format specifier
  bool isFloat;
  bool isSigned;
};

static const TypeInfo kTypes[] = {
    {"uint32_t", "%u", false, false},
    {"int32_t", "%d", false, true},
    {"float", "%f", true, false},
};
static constexpr unsigned kNumTypes = sizeof(kTypes) / sizeof(kTypes[0]);

// ---------------------------------------------------------------------------
// Binary operators
// ---------------------------------------------------------------------------

struct OpInfo {
  const char *symbol;       // e.g. "+"
  const char *reduceArg;    // LOOM_REDUCE argument, or nullptr
  bool intOnly;
};

// Index 0-5: available for integers.  Index 0-2: available for floats.
static const OpInfo kOps[] = {
    {"+", "+", false},
    {"-", nullptr, false},
    {"*", "*", false},
    {"&", "&", true},
    {"|", "|", true},
    {"^", "^", true},
};
static constexpr unsigned kNumOps = sizeof(kOps) / sizeof(kOps[0]);
static constexpr unsigned kNumFloatOps = 3; // +, -, *

// ---------------------------------------------------------------------------
// Pragma helpers – schedule names for LOOM_PARALLEL
// ---------------------------------------------------------------------------

static const char *const kSchedules[] = {"contiguous", "interleaved"};

// ---------------------------------------------------------------------------
// Parameters for a single generated app
// ---------------------------------------------------------------------------

struct FuzzParams {
  std::string name;

  // Type and operation
  unsigned typeIdx;
  unsigned opIdx;

  // Loop structure
  unsigned numLoops; // 1, 2, or 3 nesting levels

  // Array dimensions – outer/mid/inner trip counts
  unsigned dimOuter;
  unsigned dimMid;   // only used when numLoops >= 2
  unsigned dimInner; // only used when numLoops >= 3

  // Input arrays
  unsigned numInputs; // 1, 2, or 3

  // Reduction: if true the innermost loop accumulates a scalar
  bool isReduction;

  // Pragma toggles & values
  bool hasAccelName;
  bool hasTarget;
  unsigned targetKind; // 0=spatial, 1=temporal

  // LOOM_STREAM on each input (indexed 0..numInputs-1)
  std::vector<bool> inputStream;

  // LOOM_MEMORY_BANK on first input
  bool hasMemoryBank;
  unsigned memoryBankCount; // 2, 4, or 8
  bool memoryBankBlock;     // true=block, false=cyclic (default)

  // Per-loop pragmas (indexed 0..numLoops-1, 0=outermost)
  struct LoopPragmas {
    bool hasParallel;
    bool parallelAuto;
    unsigned parallelDegree; // 2, 4, or 8
    bool hasSchedule;
    unsigned scheduleIdx; // index into kSchedules

    bool hasUnroll;
    bool unrollAuto;
    unsigned unrollFactor; // 2, 4, or 8

    bool hasTripcount;
    unsigned tripcountKind; // 0=single, 1=range, 2=full
    unsigned tripcountVal;  // used for single
    unsigned tripcountMin;
    unsigned tripcountMax;

    bool hasNoParallel;
    bool hasNoUnroll;
  };
  std::vector<LoopPragmas> loopPragmas;

  // Reduction operator (only when isReduction)
  unsigned reduceOpIdx; // index into kOps (must have reduceArg != nullptr)
};

// ---------------------------------------------------------------------------
// Pick a random element from a small set of allowed values
// ---------------------------------------------------------------------------

static unsigned pickFrom(std::mt19937 &rng,
                         const std::vector<unsigned> &choices) {
  return choices[rng() % choices.size()];
}

// ---------------------------------------------------------------------------
// Generate deterministic random parameters for one app
// ---------------------------------------------------------------------------

static FuzzParams generateParams(unsigned seed, unsigned idx) {
  std::mt19937 rng(seed + idx);
  FuzzParams p;
  p.name = "fuzz_" + std::to_string(seed) + "_" + std::to_string(idx);

  // Type
  p.typeIdx = rng() % kNumTypes;
  bool isFloat = kTypes[p.typeIdx].isFloat;

  // Operation – constrain to float-safe ops when using float
  p.opIdx = isFloat ? (rng() % kNumFloatOps) : (rng() % kNumOps);

  // Loop nesting
  p.numLoops = 1 + rng() % 3;

  // Dimensions (small for fast execution)
  std::vector<unsigned> dimChoices = {4, 8, 16, 32};
  p.dimOuter = pickFrom(rng, dimChoices);
  p.dimMid = (p.numLoops >= 2) ? pickFrom(rng, dimChoices) : 1;
  p.dimInner = (p.numLoops >= 3) ? pickFrom(rng, dimChoices) : 1;

  // Input arrays: 2-3 for element-wise (need >=2 to exercise the operator),
  // 1-2 for reductions (single input is fine since the accumulator is the
  // second operand).  The isReduction flag is set below, so we pick numInputs
  // first and then clamp if needed after deciding isReduction.
  if (p.numLoops == 1) {
    p.numInputs = 1 + rng() % 3; // 1, 2, or 3
  } else {
    p.numInputs = 1 + rng() % 2; // 1 or 2
  }

  // Reduction: more likely on single-loop, possible on innermost of nested
  if (p.numLoops == 1) {
    p.isReduction = (rng() % 3) == 0; // 33% chance
  } else {
    p.isReduction = (rng() % 4) == 0; // 25% chance
  }

  // Pick reduction operator (only from ops that have reduceArg)
  if (p.isReduction) {
    // Valid reduce ops: + (0), * (2), & (3), | (4), ^ (5)
    if (isFloat) {
      // Float reductions: + or *
      std::vector<unsigned> floatReduce = {0, 2};
      p.reduceOpIdx = pickFrom(rng, floatReduce);
    } else {
      std::vector<unsigned> intReduce = {0, 2, 3, 4, 5};
      p.reduceOpIdx = pickFrom(rng, intReduce);
    }
  } else {
    p.reduceOpIdx = 0; // unused
    // Non-reduction element-wise: need at least 2 inputs to exercise the op
    if (p.numInputs < 2) p.numInputs = 2;
  }

  // LOOM_ACCEL name
  p.hasAccelName = (rng() % 2) == 0;

  // LOOM_TARGET
  p.hasTarget = (rng() % 3) == 0; // 33% chance
  p.targetKind = rng() % 2;

  // LOOM_STREAM per input
  p.inputStream.resize(p.numInputs);
  for (unsigned i = 0; i < p.numInputs; ++i) {
    p.inputStream[i] = (rng() % 3) == 0; // 33% chance
  }

  // LOOM_MEMORY_BANK on first input
  p.hasMemoryBank = (rng() % 4) == 0; // 25% chance
  std::vector<unsigned> bankChoices = {2, 4, 8};
  p.memoryBankCount = pickFrom(rng, bankChoices);
  p.memoryBankBlock = (rng() % 2) == 0;

  // Per-loop pragmas
  p.loopPragmas.resize(p.numLoops);
  for (unsigned L = 0; L < p.numLoops; ++L) {
    auto &lp = p.loopPragmas[L];

    // LOOM_PARALLEL vs LOOM_NO_PARALLEL (mutually exclusive)
    unsigned parallelChoice = rng() % 5;
    // 0=none, 1=auto, 2=degree-only, 3=degree+schedule, 4=no_parallel
    lp.hasNoParallel = (parallelChoice == 4);
    lp.hasParallel = (parallelChoice >= 1 && parallelChoice <= 3);
    lp.parallelAuto = (parallelChoice == 1);
    std::vector<unsigned> degreeChoices = {2, 4, 8};
    lp.parallelDegree = pickFrom(rng, degreeChoices);
    lp.hasSchedule = (parallelChoice == 3);
    lp.scheduleIdx = rng() % 2;

    // LOOM_UNROLL vs LOOM_NO_UNROLL (mutually exclusive)
    unsigned unrollChoice = rng() % 4;
    // 0=none, 1=auto, 2=factor, 3=no_unroll
    lp.hasNoUnroll = (unrollChoice == 3);
    lp.hasUnroll = (unrollChoice >= 1 && unrollChoice <= 2);
    lp.unrollAuto = (unrollChoice == 1);
    std::vector<unsigned> unrollChoices = {2, 4, 8};
    lp.unrollFactor = pickFrom(rng, unrollChoices);

    // LOOM_TRIPCOUNT
    lp.hasTripcount = (rng() % 2) == 0; // 50% chance
    lp.tripcountKind = rng() % 3;       // 0=single, 1=range, 2=full

    // Determine the actual trip count for this loop level
    unsigned actualTrip;
    if (L == 0)
      actualTrip = p.dimOuter;
    else if (L == 1)
      actualTrip = p.dimMid;
    else
      actualTrip = p.dimInner;

    lp.tripcountVal = actualTrip;
    // Ensure min <= actualTrip <= max
    lp.tripcountMin = 1;
    lp.tripcountMax = actualTrip * 4;
  }

  return p;
}

// ---------------------------------------------------------------------------
// Code generation helpers
// ---------------------------------------------------------------------------

/// Emit the include block for generated .cpp files.
static void emitIncludes(std::ostream &os, const std::string &headerName) {
  os << "#include \"" << headerName << "\"\n";
  os << "#include <loom/loom.h>\n\n";
}

/// Total number of elements in the output array.
static std::string totalElements(const FuzzParams &p) {
  if (p.numLoops == 1)
    return "N";
  if (p.numLoops == 2)
    return "N * M";
  return "N * M * P";
}

/// Emit the dimension parameter list (N, and optionally M, P).
static std::string dimParams(const FuzzParams &p, const char *type) {
  std::string s;
  s += std::string("const ") + type + " N";
  if (p.numLoops >= 2)
    s += std::string(", const ") + type + " M";
  if (p.numLoops >= 3)
    s += std::string(", const ") + type + " P";
  return s;
}

/// Emit the dimension argument list (N, and optionally M, P).
static std::string dimArgs(const FuzzParams &p) {
  std::string s = "N";
  if (p.numLoops >= 2) s += ", M";
  if (p.numLoops >= 3) s += ", N, P";
  return s;
}

/// Dimension argument list for calling from main (uses actual dim values).
static std::string dimArgsMain(const FuzzParams &p) {
  std::string s = std::to_string(p.dimOuter);
  if (p.numLoops >= 2)
    s += ", " + std::to_string(p.dimMid);
  if (p.numLoops >= 3)
    s += ", " + std::to_string(p.dimOuter) + ", " +
         std::to_string(p.dimInner);
  return s;
}

/// Build the function signature parameter list (shared by CPU and DSA).
/// Returns e.g.: "const uint32_t* __restrict__ a, const uint32_t* __restrict__
/// b, uint32_t* __restrict__ out, const uint32_t N"
static std::string funcParams(const FuzzParams &p, bool isDsa) {
  const char *cType = kTypes[p.typeIdx].cType;
  std::ostringstream os;

  // Input arrays
  for (unsigned i = 0; i < p.numInputs; ++i) {
    if (i > 0) os << ",\n                ";
    if (isDsa && p.inputStream[i])
      os << "LOOM_STREAM ";
    if (isDsa && i == 0 && p.hasMemoryBank) {
      os << "LOOM_MEMORY_BANK(" << p.memoryBankCount;
      if (p.memoryBankBlock) os << ", block";
      os << ") ";
    }
    os << "const " << cType << "* __restrict__ "
       << (char)('a' + i);
  }

  if (p.isReduction) {
    // Reduction: takes init_value, returns scalar
    os << ",\n                const " << cType << " init_value";
  } else {
    // Element-wise: output array
    os << ",\n                " << cType << "* __restrict__ out";
  }

  os << ",\n                " << dimParams(p, "uint32_t");

  return os.str();
}

/// Emit the loop pragmas for a given nesting level.
static void emitLoopPragmas(std::ostream &os, const FuzzParams &p,
                            unsigned level, const std::string &indent) {
  const auto &lp = p.loopPragmas[level];

  // LOOM_PARALLEL / LOOM_NO_PARALLEL
  if (lp.hasNoParallel) {
    os << indent << "LOOM_NO_PARALLEL\n";
  } else if (lp.hasParallel) {
    if (lp.parallelAuto) {
      os << indent << "LOOM_PARALLEL()\n";
    } else if (lp.hasSchedule) {
      os << indent << "LOOM_PARALLEL(" << lp.parallelDegree << ", "
         << kSchedules[lp.scheduleIdx] << ")\n";
    } else {
      os << indent << "LOOM_PARALLEL(" << lp.parallelDegree << ")\n";
    }
  }

  // LOOM_UNROLL / LOOM_NO_UNROLL
  if (lp.hasNoUnroll) {
    os << indent << "LOOM_NO_UNROLL\n";
  } else if (lp.hasUnroll) {
    if (lp.unrollAuto) {
      os << indent << "LOOM_UNROLL()\n";
    } else {
      os << indent << "LOOM_UNROLL(" << lp.unrollFactor << ")\n";
    }
  }

  // LOOM_TRIPCOUNT
  if (lp.hasTripcount) {
    switch (lp.tripcountKind) {
    case 0: // single
      os << indent << "LOOM_TRIPCOUNT(" << lp.tripcountVal << ")\n";
      break;
    case 1: // range
      os << indent << "LOOM_TRIPCOUNT_RANGE(" << lp.tripcountMin << ", "
         << lp.tripcountMax << ")\n";
      break;
    case 2: // full
      os << indent << "LOOM_TRIPCOUNT_FULL(" << lp.tripcountVal << ", "
         << lp.tripcountVal << ", " << lp.tripcountMin << ", "
         << lp.tripcountMax << ")\n";
      break;
    }
  }
}

/// Emit the loop body expression (the actual computation).
/// For element-wise: out[idx] = a[idx] <op> b[idx] (or just a[idx] <op> const)
/// For reduction: acc = acc <op> a[idx] (or acc <op> a[idx] <op> b[idx])
static void emitBody(std::ostream &os, const FuzzParams &p,
                     const std::string &indent, const std::string &idx) {
  const char *op = kOps[p.opIdx].symbol;

  if (p.isReduction) {
    // Accumulate into 'acc' – use the reduction operator, not the general op
    const char *reduceOp = kOps[p.reduceOpIdx].symbol;
    os << indent << "acc = acc " << reduceOp << " " << "a[" << idx << "]";
    if (p.numInputs >= 2)
      os << " " << reduceOp << " b[" << idx << "]";
    os << ";\n";
  } else {
    // Element-wise
    os << indent << "out[" << idx << "] = a[" << idx << "]";
    for (unsigned i = 1; i < p.numInputs; ++i)
      os << " " << op << " " << (char)('a' + i) << "[" << idx << "]";
    os << ";\n";
  }
}

/// Build the linear index expression for a given nesting depth.
/// 1 loop:  "i"
/// 2 loops: "i * M + j"
/// 3 loops: "i * M * P + j * P + k"
static std::string linearIdx(const FuzzParams &p) {
  if (p.numLoops == 1) return "i";
  if (p.numLoops == 2) return "i * M + j";
  return "i * M * P + j * P + k";
}

/// Emit the full function body (loop nest + body).
static void emitFuncBody(std::ostream &os, const FuzzParams &p, bool isDsa) {
  const char *cType = kTypes[p.typeIdx].cType;
  bool isFloat = kTypes[p.typeIdx].isFloat;
  std::string idx = linearIdx(p);

  // Reduction initialisation
  if (p.isReduction) {
    if (isDsa) {
      const char *reduceArg = kOps[p.reduceOpIdx].reduceArg;
      os << "  LOOM_REDUCE(" << reduceArg << ")\n";
    }
    os << "  " << cType << " acc = init_value;\n";
  }

  // Loop nest
  const char *loopVars[] = {"i", "j", "k"};
  const char *loopBounds[] = {"N", "M", "P"};
  unsigned depth = p.numLoops;

  for (unsigned L = 0; L < depth; ++L) {
    std::string indent(2 + L * 2, ' ');

    // Emit pragmas only for DSA version
    if (isDsa) {
      emitLoopPragmas(os, p, L, indent);
    }

    os << indent << "for (uint32_t " << loopVars[L] << " = 0; "
       << loopVars[L] << " < " << loopBounds[L] << "; ++"
       << loopVars[L] << ") {\n";
  }

  // Body
  {
    std::string indent(2 + depth * 2, ' ');
    emitBody(os, p, indent, idx);
  }

  // Close loops
  for (unsigned L = depth; L > 0; --L) {
    std::string indent(2 + (L - 1) * 2, ' ');
    os << indent << "}\n";
  }

  // Return for reduction
  if (p.isReduction) {
    os << "  return acc;\n";
  }
}

// ---------------------------------------------------------------------------
// Write the three source files for a generated test case
// ---------------------------------------------------------------------------

/// Write <name>.h
static void writeHeader(const FuzzParams &p, const std::string &dir) {
  std::string path = dir + "/" + p.name + ".h";
  std::ofstream os(path);
  const char *cType = kTypes[p.typeIdx].cType;
  std::string guard = p.name + "_H";
  for (auto &c : guard) c = (char)toupper((unsigned char)c);

  os << "// Generated by loom app fuzzer\n";
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#include <cstdint>\n\n";

  // CPU declaration
  if (p.isReduction) {
    os << cType << " " << p.name << "_cpu(" << funcParams(p, false)
       << ");\n\n";
    os << cType << " " << p.name << "_dsa(" << funcParams(p, false)
       << ");\n\n";
  } else {
    os << "void " << p.name << "_cpu(" << funcParams(p, false) << ");\n\n";
    os << "void " << p.name << "_dsa(" << funcParams(p, false) << ");\n\n";
  }

  os << "#endif // " << guard << "\n";
}

/// Write <name>.cpp
static void writeImpl(const FuzzParams &p, const std::string &dir) {
  std::string path = dir + "/" + p.name + ".cpp";
  std::ofstream os(path);
  const char *cType = kTypes[p.typeIdx].cType;

  os << "// Generated by loom app fuzzer\n";
  emitIncludes(os, p.name + ".h");

  // CPU version
  os << "// CPU reference implementation\n";
  if (p.isReduction) {
    os << cType << " " << p.name << "_cpu(" << funcParams(p, false) << ") {\n";
  } else {
    os << "void " << p.name << "_cpu(" << funcParams(p, false) << ") {\n";
  }
  emitFuncBody(os, p, /*isDsa=*/false);
  os << "}\n\n";

  // DSA version
  os << "// DSA accelerated implementation\n";
  if (p.hasTarget) {
    os << "LOOM_TARGET(\""
       << (p.targetKind == 0 ? "spatial" : "temporal") << "\")\n";
  }
  if (p.hasAccelName) {
    os << "LOOM_ACCEL(\"" << p.name << "\")\n";
  } else {
    os << "LOOM_ACCEL()\n";
  }

  if (p.isReduction) {
    os << cType << " " << p.name << "_dsa(" << funcParams(p, true) << ") {\n";
  } else {
    os << "void " << p.name << "_dsa(" << funcParams(p, true) << ") {\n";
  }
  emitFuncBody(os, p, /*isDsa=*/true);
  os << "}\n";
}

/// Write main.cpp
static void writeMain(const FuzzParams &p, const std::string &dir) {
  std::string path = dir + "/main.cpp";
  std::ofstream os(path);
  const char *cType = kTypes[p.typeIdx].cType;
  bool isFloat = kTypes[p.typeIdx].isFloat;

  os << "// Generated by loom app fuzzer\n";
  os << "#include \"" << p.name << ".h\"\n";
  os << "#include <cstdio>\n";
  if (isFloat) os << "#include <cmath>\n";
  os << "\n";

  os << "int main() {\n";

  // Dimension constants
  os << "  const uint32_t N = " << p.dimOuter << ";\n";
  if (p.numLoops >= 2)
    os << "  const uint32_t M = " << p.dimMid << ";\n";
  if (p.numLoops >= 3)
    os << "  const uint32_t P = " << p.dimInner << ";\n";

  // Total element count
  std::string total = totalElements(p);
  os << "  const uint32_t total = " << total << ";\n\n";

  // Input arrays
  for (unsigned i = 0; i < p.numInputs; ++i) {
    os << "  " << cType << " " << (char)('a' + i) << "[total];\n";
  }

  // Initialise inputs with deterministic values
  os << "\n  // Initialise inputs\n";
  for (unsigned i = 0; i < p.numInputs; ++i) {
    char varName = (char)('a' + i);
    os << "  for (uint32_t idx = 0; idx < total; ++idx) {\n";
    if (isFloat) {
      // Small float values to avoid overflow
      os << "    " << varName << "[idx] = (float)((idx * "
         << (i * 7 + 3) << " + " << (i + 1) << ") % 10) / 10.0f + 0.1f;\n";
    } else {
      // Small integer values to avoid overflow (especially for multiply)
      os << "    " << varName << "[idx] = (" << cType << ")((idx * "
         << (i * 7 + 3) << " + " << (i + 1) << ") % 10);\n";
    }
    os << "  }\n";
  }
  os << "\n";

  if (p.isReduction) {
    // Reduction test
    if (isFloat) {
      os << "  " << cType << " init_value = 0.0f;\n";
    } else {
      // For multiply reduction, init must be 1; for others, 0 is fine
      // But to keep values small, use 0 for all (multiply with 0 tests edge)
      // Actually use a small nonzero value to make it interesting
      if (kOps[p.reduceOpIdx].symbol[0] == '*') {
        os << "  " << cType << " init_value = 1;\n";
      } else {
        os << "  " << cType << " init_value = 0;\n";
      }
    }

    // Dimension args for function call
    std::string dArgs;
    dArgs = "N";
    if (p.numLoops >= 2) dArgs += ", M";
    if (p.numLoops >= 3) dArgs += ", P";

    // Build input arg list
    std::string inputArgs;
    for (unsigned i = 0; i < p.numInputs; ++i) {
      if (i > 0) inputArgs += ", ";
      inputArgs += (char)('a' + i);
    }

    os << "  " << cType << " cpu_result = " << p.name << "_cpu("
       << inputArgs << ", init_value, " << dArgs << ");\n";
    os << "  " << cType << " dsa_result = " << p.name << "_dsa("
       << inputArgs << ", init_value, " << dArgs << ");\n\n";

    // Compare
    if (isFloat) {
      os << "  if (std::fabs(cpu_result - dsa_result) > 1e-2f) {\n";
    } else {
      os << "  if (cpu_result != dsa_result) {\n";
    }
    os << "    printf(\"" << p.name << ": FAILED\\n\");\n";
    os << "    return 1;\n";
    os << "  }\n";
  } else {
    // Element-wise test
    os << "  " << cType << " cpu_out[total];\n";
    os << "  " << cType << " dsa_out[total];\n\n";

    // Initialise output arrays to 0
    os << "  for (uint32_t idx = 0; idx < total; ++idx) {\n";
    if (isFloat) {
      os << "    cpu_out[idx] = 0.0f;\n";
      os << "    dsa_out[idx] = 0.0f;\n";
    } else {
      os << "    cpu_out[idx] = 0;\n";
      os << "    dsa_out[idx] = 0;\n";
    }
    os << "  }\n\n";

    // Dimension args for function call
    std::string dArgs;
    dArgs = "N";
    if (p.numLoops >= 2) dArgs += ", M";
    if (p.numLoops >= 3) dArgs += ", P";

    // Build input arg list
    std::string inputArgs;
    for (unsigned i = 0; i < p.numInputs; ++i) {
      if (i > 0) inputArgs += ", ";
      inputArgs += (char)('a' + i);
    }

    os << "  " << p.name << "_cpu(" << inputArgs << ", cpu_out, "
       << dArgs << ");\n";
    os << "  " << p.name << "_dsa(" << inputArgs << ", dsa_out, "
       << dArgs << ");\n\n";

    // Compare
    os << "  for (uint32_t idx = 0; idx < total; ++idx) {\n";
    if (isFloat) {
      os << "    if (std::fabs(cpu_out[idx] - dsa_out[idx]) > 1e-4f) {\n";
    } else {
      os << "    if (cpu_out[idx] != dsa_out[idx]) {\n";
    }
    os << "      printf(\"" << p.name << ": FAILED\\n\");\n";
    os << "      return 1;\n";
    os << "    }\n";
    os << "  }\n";
  }

  os << "\n  printf(\"" << p.name << ": PASSED\\n\");\n";
  os << "  return 0;\n";
  os << "}\n";
}

/// Write all three files for a test case into Output/<name>/
static void writeCppSource(const FuzzParams &p, const std::string &outDir) {
  std::string caseDir = outDir + "/" + p.name;
  std::string mkdirCmd = "mkdir -p " + caseDir;
  (void)std::system(mkdirCmd.c_str());

  writeHeader(p, caseDir);
  writeImpl(p, caseDir);
  writeMain(p, caseDir);
}

// ---------------------------------------------------------------------------
// Direct compile-and-run (pass 2)
// ---------------------------------------------------------------------------

/// Compile and run a generated test case using loom --as-clang.
/// Returns 0 on success, 1 on failure.
static int compileAndRun(const FuzzParams &p, const std::string &outDir) {
  std::string caseDir = outDir + "/" + p.name;

  // Ensure source files exist (generate them if needed)
  {
    std::ifstream check(caseDir + "/" + p.name + ".cpp");
    if (!check.good()) {
      writeCppSource(p, outDir);
    }
  }

  std::string outputBin = caseDir + "/Output/" + p.name;
  std::string mkdirCmd = "mkdir -p " + caseDir + "/Output";
  (void)std::system(mkdirCmd.c_str());

  // Compile
  std::string compileCmd =
      "loom --as-clang " + caseDir + "/" + p.name + ".cpp " + caseDir +
      "/main.cpp -o " + outputBin + " 2>&1";
  int rc = std::system(compileCmd.c_str());
  if (rc != 0) {
    fprintf(stderr, "%s: compilation failed (exit %d)\n", p.name.c_str(), rc);
    return 1;
  }

  // Run
  std::string runCmd = "cd " + caseDir + " && Output/" + p.name + " 2>&1";
  rc = std::system(runCmd.c_str());
  if (rc != 0) {
    fprintf(stderr, "%s: execution failed (exit %d)\n", p.name.c_str(), rc);
    return 1;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  unsigned seed = FUZZER_SEED;
  unsigned count = FUZZER_COUNT;

  // Allow env override.
  if (const char *s = std::getenv("LOOM_FUZZER_SEED"))
    seed = (unsigned)std::atoi(s);
  if (const char *c = std::getenv("LOOM_FUZZER_COUNT"))
    count = (unsigned)std::atoi(c);

  // Check for --gen-cpp mode.
  bool genCpp = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--gen-cpp")
      genCpp = true;
  }

  int failures = 0;
  for (unsigned idx = 0; idx < count; ++idx) {
    FuzzParams p = generateParams(seed, idx);

    if (genCpp) {
      writeCppSource(p, "Output");
    } else {
      // Generate sources and attempt compile+run
      writeCppSource(p, "Output");
      failures += compileAndRun(p, "Output");
    }
  }

  if (!genCpp && failures > 0) {
    fprintf(stderr, "\n%d / %d test cases FAILED\n", failures, count);
    return 1;
  }

  return 0;
}
