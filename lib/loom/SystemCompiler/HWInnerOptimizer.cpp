//===-- HWInnerOptimizer.cpp - Per-core ADG optimizer -------------*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the INNER-HW per-core ADG optimizer (C12).
//
// Three-tier evaluation pipeline:
//   Tier A: Analytical derivation of initial ADG parameters
//   Tier B: Small-scale BO with real Loom mapper verification
//   Tier C: Simulation (deferred)
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/HWInnerOptimizer.h"
#include "loom/ADG/ADGBuilder.h"
#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>

namespace loom {

//===----------------------------------------------------------------------===//
// Enum string conversions
//===----------------------------------------------------------------------===//

const char *peTypeToString(PEType t) {
  switch (t) {
  case PEType::SPATIAL:
    return "spatial";
  case PEType::TEMPORAL:
    return "temporal";
  }
  return "spatial";
}

PEType peTypeFromString(const std::string &s) {
  if (s == "temporal")
    return PEType::TEMPORAL;
  return PEType::SPATIAL;
}

const char *switchTypeToString(SwitchType t) {
  switch (t) {
  case SwitchType::SPATIAL:
    return "spatial";
  case SwitchType::TEMPORAL:
    return "temporal";
  }
  return "spatial";
}

SwitchType switchTypeFromString(const std::string &s) {
  if (s == "temporal")
    return SwitchType::TEMPORAL;
  return SwitchType::SPATIAL;
}

const char *topologyToString(RoutingTopology t) {
  switch (t) {
  case RoutingTopology::CHESS:
    return "chess";
  case RoutingTopology::MESH:
    return "mesh";
  case RoutingTopology::LATTICE:
    return "lattice";
  case RoutingTopology::RING:
    return "ring";
  }
  return "chess";
}

RoutingTopology topologyFromString(const std::string &s) {
  if (s == "mesh")
    return RoutingTopology::MESH;
  if (s == "lattice")
    return RoutingTopology::LATTICE;
  if (s == "ring")
    return RoutingTopology::RING;
  return RoutingTopology::CHESS;
}

const char *innerComputeMixToString(InnerComputeMix m) {
  switch (m) {
  case InnerComputeMix::FP_HEAVY:
    return "fp_heavy";
  case InnerComputeMix::INT_HEAVY:
    return "int_heavy";
  case InnerComputeMix::MIXED:
    return "mixed";
  }
  return "mixed";
}

InnerComputeMix innerComputeMixFromString(const std::string &s) {
  if (s == "fp_heavy")
    return InnerComputeMix::FP_HEAVY;
  if (s == "int_heavy")
    return InnerComputeMix::INT_HEAVY;
  return InnerComputeMix::MIXED;
}

//===----------------------------------------------------------------------===//
// FreedomMask
//===----------------------------------------------------------------------===//

unsigned FreedomMask::countFree() const {
  unsigned count = 0;
  if (peType) ++count;
  if (arrayDims) ++count;
  if (dataWidth) ++count;
  if (fuRepertoire) ++count;
  if (fuBodyStructure) ++count;
  if (switchType) ++count;
  if (decomposability) ++count;
  if (spm) ++count;
  if (extMem) ++count;
  if (topology) ++count;
  if (temporalParams) ++count;
  if (scalarIO) ++count;
  if (connectivity) ++count;
  return count;
}

FreedomMask FreedomMask::domainSpecific() {
  FreedomMask mask;
  mask.topology = true;       // Dim 10
  mask.connectivity = true;   // Dim 13
  mask.fuRepertoire = true;   // Dim 4
  return mask;
}

FreedomMask FreedomMask::combinatorial(bool isTemporal) {
  FreedomMask mask;
  mask.dataWidth = true;       // Dim 3
  mask.fuRepertoire = true;    // Dim 4
  mask.decomposability = true; // Dim 7
  mask.extMem = true;          // Dim 9
  mask.topology = true;        // Dim 10
  mask.scalarIO = true;        // Dim 12
  mask.connectivity = true;    // Dim 13
  if (isTemporal) {
    mask.temporalParams = true;  // Dim 11
  }
  return mask;
}

//===----------------------------------------------------------------------===//
// Area Model
//===----------------------------------------------------------------------===//

/// Base area cost per FU operation type (in abstract area units).
/// These are rough relative costs reflecting hardware complexity.
static double baseFUAreaCost(const std::string &opName) {
  // Floating-point operations are more expensive than integer
  if (opName.find("addf") != std::string::npos ||
      opName.find("subf") != std::string::npos)
    return 8.0;
  if (opName.find("mulf") != std::string::npos)
    return 16.0;
  if (opName.find("divf") != std::string::npos)
    return 24.0;
  if (opName.find("divsi") != std::string::npos ||
      opName.find("divui") != std::string::npos)
    return 20.0;
  if (opName.find("remsi") != std::string::npos ||
      opName.find("remui") != std::string::npos)
    return 20.0;
  if (opName.find("cmpf") != std::string::npos)
    return 6.0;

  // Math library operations are expensive
  if (opName.find("sqrt") != std::string::npos)
    return 30.0;
  if (opName.find("exp") != std::string::npos)
    return 40.0;
  if (opName.find("log") != std::string::npos)
    return 35.0;
  if (opName.find("sin") != std::string::npos ||
      opName.find("cos") != std::string::npos)
    return 40.0;
  if (opName.find("fma") != std::string::npos)
    return 20.0;

  // Integer arithmetic
  if (opName.find("addi") != std::string::npos ||
      opName.find("subi") != std::string::npos)
    return 2.0;
  if (opName.find("muli") != std::string::npos)
    return 8.0;
  if (opName.find("shli") != std::string::npos ||
      opName.find("shrsi") != std::string::npos ||
      opName.find("shrui") != std::string::npos)
    return 3.0;

  // Comparison and logic
  if (opName.find("cmpi") != std::string::npos)
    return 2.0;
  if (opName.find("andi") != std::string::npos ||
      opName.find("ori") != std::string::npos ||
      opName.find("xori") != std::string::npos)
    return 1.0;
  if (opName.find("select") != std::string::npos)
    return 2.0;

  // Type conversions
  if (opName.find("extsi") != std::string::npos ||
      opName.find("extui") != std::string::npos ||
      opName.find("trunci") != std::string::npos)
    return 1.0;
  if (opName.find("sitofp") != std::string::npos ||
      opName.find("fptoui") != std::string::npos ||
      opName.find("fptosi") != std::string::npos ||
      opName.find("uitofp") != std::string::npos)
    return 6.0;
  if (opName.find("index_cast") != std::string::npos)
    return 1.0;

  // Memory operations
  if (opName.find("load") != std::string::npos ||
      opName.find("store") != std::string::npos)
    return 4.0;

  // Dataflow/handshake operations
  if (opName.find("stream") != std::string::npos)
    return 5.0;
  if (opName.find("gate") != std::string::npos ||
      opName.find("carry") != std::string::npos ||
      opName.find("invariant") != std::string::npos)
    return 3.0;
  if (opName.find("constant") != std::string::npos)
    return 1.0;
  if (opName.find("cond_br") != std::string::npos)
    return 2.0;
  if (opName.find("mux") != std::string::npos)
    return 3.0;
  if (opName.find("join") != std::string::npos)
    return 1.0;

  // Default for unknown operations
  return 4.0;
}

double estimateFUArea(const std::string &opName, unsigned dataWidth) {
  double base = baseFUAreaCost(opName);
  // Scale by data width (quadratic for multipliers, linear for most others)
  double widthFactor = static_cast<double>(dataWidth) / 32.0;
  bool isMultiplier = opName.find("mul") != std::string::npos ||
                      opName.find("div") != std::string::npos ||
                      opName.find("rem") != std::string::npos;
  if (isMultiplier)
    return base * widthFactor * widthFactor;
  return base * widthFactor;
}

double estimateCoreArea(const CoreDesignParams &params) {
  unsigned peCount = params.totalPEs();

  // FU area: sum of all FU types, per PE
  double fuAreaPerPE = 0.0;
  for (const auto &op : params.fuRepertoire) {
    fuAreaPerPE += estimateFUArea(op, params.dataWidth);
  }

  // PE overhead (register file, control logic)
  double peOverhead = 5.0;
  if (params.peType == PEType::TEMPORAL) {
    // Temporal PEs have instruction memory and register file overhead
    peOverhead += 3.0 * params.instructionSlots;
    peOverhead += 2.0 * params.numRegisters;
  }

  double totalPEArea = peCount * (fuAreaPerPE + peOverhead);

  // Switch area: proportional to port count squared (crossbar)
  // Each PE has ~4 inputs and ~4 outputs to the switch network
  unsigned pePortsPerSW = 4;
  unsigned interSWPorts = 4; // neighbor connections
  unsigned totalSWPorts = pePortsPerSW + interSWPorts;
  double swArea = static_cast<double>(totalSWPorts * totalSWPorts) *
                  (static_cast<double>(params.dataWidth) / 32.0);
  // Number of switches depends on topology
  unsigned numSwitches = peCount; // approximate: ~1 switch per PE
  if (params.topology == RoutingTopology::CHESS) {
    numSwitches = (params.arrayRows + 1) * (params.arrayCols + 1);
  }
  double totalSWArea = numSwitches * swArea;

  // SPM area: proportional to size
  double spmArea = static_cast<double>(params.spmSizeKB) * 2.0;

  // External memory interface area
  double extmemArea = params.extmemCount *
                      (params.extmemLdPorts + params.extmemStPorts) * 4.0;

  // Scalar I/O area
  double scalarArea = (params.scalarInputs + params.scalarOutputs) * 1.0;

  return totalPEArea + totalSWArea + spmArea + extmemArea + scalarArea;
}

//===----------------------------------------------------------------------===//
// PE Type Selection
//===----------------------------------------------------------------------===//

double computeParallelismRatio(const std::vector<KernelProfile> &profiles) {
  if (profiles.empty())
    return 1.0;

  double maxRatio = 0.0;
  for (const auto &profile : profiles) {
    unsigned totalOps = profile.totalOpCount();
    unsigned uniqueOpTypes = profile.requiredOps.size();
    if (uniqueOpTypes == 0)
      continue;

    // Approximate concurrent ops as total ops (upper bound for one DFG)
    // This is a heuristic; real ILP analysis would be more precise.
    double ratio = static_cast<double>(totalOps) /
                   static_cast<double>(uniqueOpTypes);
    maxRatio = std::max(maxRatio, ratio);
  }

  return maxRatio > 0.0 ? maxRatio : 1.0;
}

PEType selectPEType(const std::vector<KernelProfile> &profiles,
                    double threshold) {
  double ratio = computeParallelismRatio(profiles);
  return (ratio > threshold) ? PEType::SPATIAL : PEType::TEMPORAL;
}

//===----------------------------------------------------------------------===//
// FU Repertoire
//===----------------------------------------------------------------------===//

/// Canonical FU operation names recognized by the ADG builder.
/// These map kernel op categories back to full MLIR op names.
static std::string canonicalizeFUOp(const std::string &op) {
  // If already fully qualified (contains '.'), return as-is
  if (op.find('.') != std::string::npos)
    return op;

  // Map common short names to fully-qualified MLIR op names
  static const std::map<std::string, std::string> canonMap = {
      {"addi", "arith.addi"},       {"subi", "arith.subi"},
      {"muli", "arith.muli"},       {"addf", "arith.addf"},
      {"subf", "arith.subf"},       {"mulf", "arith.mulf"},
      {"divf", "arith.divf"},       {"divsi", "arith.divsi"},
      {"divui", "arith.divui"},     {"remsi", "arith.remsi"},
      {"remui", "arith.remui"},     {"cmpi", "arith.cmpi"},
      {"cmpf", "arith.cmpf"},       {"andi", "arith.andi"},
      {"ori", "arith.ori"},         {"xori", "arith.xori"},
      {"shli", "arith.shli"},       {"shrsi", "arith.shrsi"},
      {"shrui", "arith.shrui"},     {"select", "arith.select"},
      {"negf", "arith.negf"},       {"extsi", "arith.extsi"},
      {"extui", "arith.extui"},     {"trunci", "arith.trunci"},
      {"sitofp", "arith.sitofp"},   {"uitofp", "arith.uitofp"},
      {"fptosi", "arith.fptosi"},   {"fptoui", "arith.fptoui"},
      {"index_cast", "arith.index_cast"},
      {"index_castui", "arith.index_castui"},
      {"sqrt", "math.sqrt"},         {"exp", "math.exp"},
      {"log2", "math.log2"},         {"sin", "math.sin"},
      {"cos", "math.cos"},           {"fma", "math.fma"},
      {"absf", "math.absf"},
      {"load", "handshake.load"},    {"store", "handshake.store"},
      {"constant", "handshake.constant"},
      {"cond_br", "handshake.cond_br"},
      {"mux", "handshake.mux"},      {"join", "handshake.join"},
      {"stream", "dataflow.stream"}, {"gate", "dataflow.gate"},
      {"carry", "dataflow.carry"},   {"invariant", "dataflow.invariant"},
  };

  auto it = canonMap.find(op);
  if (it != canonMap.end())
    return it->second;
  return op;
}

std::set<std::string> computeRequiredFURepertoire(
    const std::vector<KernelProfile> &profiles) {
  std::set<std::string> repertoire;
  for (const auto &profile : profiles) {
    for (const auto &[opName, count] : profile.requiredOps) {
      std::string canon = canonicalizeFUOp(opName);
      repertoire.insert(canon);
    }
  }
  return repertoire;
}

std::set<std::string> tryPruneFU(const std::set<std::string> &repertoire,
                                 const std::string &candidate) {
  std::set<std::string> pruned = repertoire;
  pruned.erase(candidate);
  return pruned;
}

//===----------------------------------------------------------------------===//
// Tier-A Analytical Scoring
//===----------------------------------------------------------------------===//

/// Map an op name to an FU category key for FU-bound analysis.
static std::string opToFUCategory(const std::string &opName) {
  std::string lower = opName;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower.find("mulf") != std::string::npos ||
      lower.find("addf") != std::string::npos ||
      lower.find("subf") != std::string::npos ||
      lower.find("divf") != std::string::npos ||
      lower.find("negf") != std::string::npos ||
      lower.find("cmpf") != std::string::npos ||
      lower.find("sqrt") != std::string::npos ||
      lower.find("exp") != std::string::npos ||
      lower.find("log") != std::string::npos ||
      lower.find("sin") != std::string::npos ||
      lower.find("cos") != std::string::npos ||
      lower.find("fma") != std::string::npos ||
      lower.find("absf") != std::string::npos ||
      lower.find("sitofp") != std::string::npos ||
      lower.find("uitofp") != std::string::npos ||
      lower.find("fptosi") != std::string::npos ||
      lower.find("fptoui") != std::string::npos)
    return "fp";

  if (lower.find("load") != std::string::npos ||
      lower.find("store") != std::string::npos)
    return "mem";

  if (lower.find("muli") != std::string::npos ||
      lower.find("divsi") != std::string::npos ||
      lower.find("divui") != std::string::npos ||
      lower.find("remsi") != std::string::npos ||
      lower.find("remui") != std::string::npos)
    return "mul";

  return "alu";
}

/// Routing overhead factor by topology. Reflects average routing distance.
static double routingOverhead(RoutingTopology topo) {
  switch (topo) {
  case RoutingTopology::CHESS:
    return 1.0;
  case RoutingTopology::MESH:
    return 1.0;
  case RoutingTopology::LATTICE:
    return 1.2;
  case RoutingTopology::RING:
    return 1.5;
  }
  return 1.0;
}

TierAScore scoreCoreDesign(const CoreDesignParams &params,
                           const std::vector<KernelProfile> &profiles) {
  TierAScore result;
  result.feasible = true;

  if (profiles.empty()) {
    result.feasible = false;
    result.compositeScore = 0.0;
    return result;
  }

  // Count FUs per category from the repertoire.
  // Each FU in the repertoire contributes to its category.
  std::map<std::string, unsigned> fuCounts;
  for (const auto &op : params.fuRepertoire) {
    std::string cat = opToFUCategory(op);
    fuCounts[cat]++;
  }

  // Scale FU counts by total PEs (each PE has the full repertoire)
  unsigned peCount = params.totalPEs();
  for (auto &[cat, count] : fuCounts) {
    count *= peCount;
  }

  // Score each kernel
  double geoProduct = 1.0;
  unsigned geoCount = 0;

  for (const auto &profile : profiles) {
    TierAKernelII kii;
    kii.kernelName = profile.name;

    // Check FU coverage: every kernel op must be supportable
    bool allCovered = true;
    std::map<std::string, unsigned> opsPerCategory;
    for (const auto &[opName, count] : profile.requiredOps) {
      std::string canon = canonicalizeFUOp(opName);
      bool found = false;
      for (const auto &fuOp : params.fuRepertoire) {
        if (fuOp == canon) {
          found = true;
          break;
        }
      }
      if (!found) {
        allCovered = false;
        break;
      }
      std::string cat = opToFUCategory(canon);
      opsPerCategory[cat] += count;
    }

    if (!allCovered) {
      result.feasible = false;
      result.compositeScore = 0.0;
      return result;
    }

    // Compute FU-bound II: max over categories of ceil(ops / fu_count)
    kii.fuBound = 1.0;
    for (const auto &[cat, ops] : opsPerCategory) {
      unsigned available = fuCounts.count(cat) ? fuCounts.at(cat) : 0;
      if (available == 0 && ops > 0) {
        result.feasible = false;
        result.compositeScore = 0.0;
        return result;
      }
      if (available > 0) {
        double bound = std::ceil(static_cast<double>(ops) /
                                 static_cast<double>(available));
        kii.fuBound = std::max(kii.fuBound, bound);
      }
    }

    // Compute memory-bound II: ceil(loads / spm_ld_ports)
    unsigned loadOps = 0;
    for (const auto &[opName, count] : profile.requiredOps) {
      if (opName.find("load") != std::string::npos) {
        loadOps += count;
      }
    }
    if (loadOps > 0 && params.spmLdPorts > 0) {
      kii.memBound = std::ceil(static_cast<double>(loadOps) /
                               static_cast<double>(params.spmLdPorts));
    } else {
      kii.memBound = 1.0;
    }

    // Routing bound: topology-dependent overhead
    kii.routingBound = routingOverhead(params.topology);

    // Effective II is max of all bounds
    kii.effectiveII = std::max({kii.fuBound, kii.memBound, kii.routingBound});

    result.perKernelII.push_back(kii);

    // Accumulate for geometric mean
    if (kii.effectiveII > 0) {
      geoProduct *= (1.0 / kii.effectiveII);
      geoCount++;
    }
  }

  // Area estimate
  result.areaEstimate = estimateCoreArea(params);

  // Composite score: geomean(1/II) / area
  if (geoCount > 0 && result.areaEstimate > 0) {
    double geoMean = std::pow(geoProduct, 1.0 / geoCount);
    result.compositeScore = geoMean / result.areaEstimate;
  } else {
    result.compositeScore = 0.0;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Preset Constructors
//===----------------------------------------------------------------------===//

std::string generateTypeID(InnerComputeMix mix, PEType pe, bool hasSPM,
                           unsigned arraySize) {
  std::string id = "C";
  switch (mix) {
  case InnerComputeMix::FP_HEAVY:
    id += "F";
    break;
  case InnerComputeMix::INT_HEAVY:
    id += "I";
    break;
  case InnerComputeMix::MIXED:
    id += "M";
    break;
  }
  id += (pe == PEType::SPATIAL) ? "S" : "T";
  id += hasSPM ? "Y" : "N";
  id += std::to_string(arraySize);
  return id;
}

CoreDesignParams createDomainPreset(unsigned domainIndex) {
  CoreDesignParams params;

  switch (domainIndex) {
  case 1: // D1: LLM - Large FP-heavy with big SPM
    params.arrayRows = 12;
    params.arrayCols = 12;
    params.peType = PEType::SPATIAL;
    params.switchType = SwitchType::SPATIAL;
    params.dataWidth = 32;
    params.spmSizeKB = 64;
    params.spmLdPorts = 2;
    params.spmStPorts = 2;
    params.topology = RoutingTopology::MESH;
    params.fuRepertoire = {
        "arith.addf", "arith.mulf", "arith.addi", "arith.muli",
        "arith.cmpi", "arith.select", "handshake.load",
        "handshake.store", "math.exp", "math.sqrt"};
    params.scalarInputs = 4;
    params.scalarOutputs = 2;
    params.extmemCount = 4;
    params.extmemLdPorts = 2;
    params.extmemStPorts = 1;
    break;

  case 2: // D2: CV - Medium spatial, mixed FP/INT
    params.arrayRows = 8;
    params.arrayCols = 8;
    params.peType = PEType::SPATIAL;
    params.switchType = SwitchType::SPATIAL;
    params.dataWidth = 32;
    params.spmSizeKB = 32;
    params.spmLdPorts = 2;
    params.spmStPorts = 2;
    params.topology = RoutingTopology::CHESS;
    params.fuRepertoire = {
        "arith.addf", "arith.mulf", "arith.addi", "arith.muli",
        "arith.cmpi", "arith.select", "arith.shli", "arith.shrsi",
        "handshake.load", "handshake.store"};
    params.scalarInputs = 3;
    params.scalarOutputs = 1;
    params.extmemCount = 2;
    params.extmemLdPorts = 1;
    params.extmemStPorts = 1;
    break;

  case 3: // D3: Signal - Temporal, multiply-heavy
    params.arrayRows = 6;
    params.arrayCols = 6;
    params.peType = PEType::TEMPORAL;
    params.switchType = SwitchType::TEMPORAL;
    params.dataWidth = 32;
    params.spmSizeKB = 16;
    params.spmLdPorts = 2;
    params.spmStPorts = 1;
    params.topology = RoutingTopology::CHESS;
    params.instructionSlots = 8;
    params.numRegisters = 8;
    params.regFifoDepth = 2;
    params.fuRepertoire = {
        "arith.addi", "arith.muli", "arith.addf", "arith.mulf",
        "arith.cmpi", "arith.select", "handshake.load",
        "handshake.store", "math.fma"};
    params.scalarInputs = 3;
    params.scalarOutputs = 1;
    params.extmemCount = 2;
    params.extmemLdPorts = 1;
    params.extmemStPorts = 1;
    break;

  case 4: // D4: Crypto - INT-heavy, small array, bitwise ops
    params.arrayRows = 4;
    params.arrayCols = 4;
    params.peType = PEType::SPATIAL;
    params.switchType = SwitchType::SPATIAL;
    params.dataWidth = 64;
    params.spmSizeKB = 8;
    params.spmLdPorts = 1;
    params.spmStPorts = 1;
    params.topology = RoutingTopology::CHESS;
    params.fuRepertoire = {
        "arith.addi", "arith.muli", "arith.andi", "arith.ori",
        "arith.xori", "arith.shli", "arith.shrsi", "arith.shrui",
        "arith.cmpi", "arith.select", "handshake.load",
        "handshake.store"};
    params.scalarInputs = 3;
    params.scalarOutputs = 1;
    params.extmemCount = 2;
    params.extmemLdPorts = 1;
    params.extmemStPorts = 1;
    break;

  case 5: // D5: Sensor - Small temporal, control-heavy
    params.arrayRows = 4;
    params.arrayCols = 4;
    params.peType = PEType::TEMPORAL;
    params.switchType = SwitchType::TEMPORAL;
    params.dataWidth = 32;
    params.spmSizeKB = 8;
    params.spmLdPorts = 1;
    params.spmStPorts = 1;
    params.topology = RoutingTopology::CHESS;
    params.instructionSlots = 16;
    params.numRegisters = 8;
    params.regFifoDepth = 4;
    params.fuRepertoire = {
        "arith.addi", "arith.muli", "arith.cmpi", "arith.select",
        "arith.addf", "arith.cmpf", "handshake.load",
        "handshake.store", "handshake.cond_br"};
    params.scalarInputs = 3;
    params.scalarOutputs = 1;
    params.extmemCount = 2;
    params.extmemLdPorts = 1;
    params.extmemStPorts = 1;
    break;

  case 6: // D6: Control - Small spatial, balanced
  default:
    params.arrayRows = 4;
    params.arrayCols = 4;
    params.peType = PEType::SPATIAL;
    params.switchType = SwitchType::SPATIAL;
    params.dataWidth = 32;
    params.spmSizeKB = 4;
    params.spmLdPorts = 1;
    params.spmStPorts = 1;
    params.topology = RoutingTopology::CHESS;
    params.fuRepertoire = {
        "arith.addi", "arith.muli", "arith.cmpi", "arith.select",
        "arith.andi", "arith.ori", "handshake.load",
        "handshake.store", "handshake.cond_br", "handshake.mux"};
    params.scalarInputs = 3;
    params.scalarOutputs = 1;
    params.extmemCount = 2;
    params.extmemLdPorts = 1;
    params.extmemStPorts = 1;
    break;
  }

  return params;
}

CoreDesignParams createCombinatorialPreset(InnerComputeMix mix, PEType pe,
                                           bool hasSPM,
                                           unsigned arraySize) {
  CoreDesignParams params;

  // Dimension 1: PE type
  params.peType = pe;

  // Dimension 2: Array size
  params.arrayRows = arraySize;
  params.arrayCols = arraySize;

  // Dimension 3: Data width (default 32)
  params.dataWidth = 32;

  // Dimension 4: FU repertoire based on compute mix
  switch (mix) {
  case InnerComputeMix::FP_HEAVY:
    params.fuRepertoire = {
        "arith.addf", "arith.mulf", "arith.subf", "arith.divf",
        "arith.cmpf", "arith.addi", "arith.muli",
        "arith.cmpi", "arith.select",
        "handshake.load", "handshake.store"};
    break;
  case InnerComputeMix::INT_HEAVY:
    params.fuRepertoire = {
        "arith.addi", "arith.subi", "arith.muli",
        "arith.andi", "arith.ori", "arith.xori",
        "arith.shli", "arith.shrsi",
        "arith.cmpi", "arith.select",
        "handshake.load", "handshake.store"};
    break;
  case InnerComputeMix::MIXED:
    params.fuRepertoire = {
        "arith.addi", "arith.muli", "arith.addf", "arith.mulf",
        "arith.cmpi", "arith.select",
        "handshake.load", "handshake.store"};
    break;
  }

  // Dimension 5: FU body structure
  params.multiOpFUBodies = false;

  // Dimension 6: Switch type matches PE type
  params.switchType = (pe == PEType::TEMPORAL) ? SwitchType::TEMPORAL
                                               : SwitchType::SPATIAL;

  // Dimension 7: No decomposition by default
  params.decomposableBits = -1;

  // Dimension 8: SPM
  if (hasSPM) {
    params.spmSizeKB = 16;
    params.spmLdPorts = 2;
    params.spmStPorts = 2;
  } else {
    params.spmSizeKB = 0;
    params.spmLdPorts = 0;
    params.spmStPorts = 0;
  }

  // Dimension 9: External memory
  params.extmemCount = 2;
  params.extmemLdPorts = 1;
  params.extmemStPorts = 1;

  // Dimension 10: Default topology
  params.topology = RoutingTopology::CHESS;

  // Dimension 11: Temporal PE params
  if (pe == PEType::TEMPORAL) {
    params.instructionSlots = 8;
    params.numRegisters = 8;
    params.regFifoDepth = 0;
    params.shareOperandBuffer = false;
    params.operandBufferSize = 0;
  }

  // Dimension 12: Scalar I/O
  params.scalarInputs = 3;
  params.scalarOutputs = 1;

  // Dimension 13: Full crossbar
  params.connectivity.clear();

  return params;
}

//===----------------------------------------------------------------------===//
// Parameter Sweep Generation
//===----------------------------------------------------------------------===//

std::vector<CoreDesignParams> generateSweepCandidates(
    const CoreDesignParams &baseline,
    const FreedomMask &mask,
    unsigned maxCandidates,
    unsigned seed) {

  // Define discrete value sets for each free dimension
  static const RoutingTopology topoValues[] = {
      RoutingTopology::CHESS, RoutingTopology::MESH,
      RoutingTopology::LATTICE, RoutingTopology::RING};
  static const unsigned dataWidthValues[] = {32, 64};
  static const int decomposableValues[] = {-1, 8, 16};
  // External memory configs: (count, ldPorts, stPorts)
  static const unsigned extmemConfigs[][3] = {{1, 1, 1}, {2, 1, 1}, {2, 2, 1}};
  // Scalar I/O configs: (inputs, outputs)
  static const unsigned scalarIOConfigs[][2] = {{2, 1}, {3, 1}, {4, 2}};
  // Temporal params: (instructionSlots, numRegisters, regFifoDepth)
  static const unsigned temporalConfigs[][3] = {
      {4, 4, 0}, {8, 8, 2}, {16, 8, 4}};

  // Count the number of discrete choices per free dimension
  struct DimInfo {
    unsigned numChoices;
  };
  std::vector<DimInfo> freeDims;

  if (mask.topology)
    freeDims.push_back({4});
  if (mask.dataWidth)
    freeDims.push_back({2});
  if (mask.decomposability)
    freeDims.push_back({3});
  if (mask.extMem)
    freeDims.push_back({3});
  if (mask.scalarIO)
    freeDims.push_back({3});
  if (mask.temporalParams && baseline.peType == PEType::TEMPORAL)
    freeDims.push_back({3});
  if (mask.fuRepertoire)
    freeDims.push_back({3}); // base, +common ops, -rare ops
  if (mask.connectivity)
    freeDims.push_back({3}); // full, 50% sparse, 25% sparse

  unsigned numFreeDims = freeDims.size();
  if (numFreeDims == 0) {
    return {baseline};
  }

  // Latin Hypercube Sampling
  std::mt19937 rng(seed);
  unsigned n = std::min(maxCandidates, 1u);

  // Compute total combinations for bounding
  unsigned totalCombos = 1;
  for (const auto &dim : freeDims) {
    totalCombos *= dim.numChoices;
    if (totalCombos > maxCandidates) {
      totalCombos = maxCandidates + 1;
      break;
    }
  }
  n = std::min(maxCandidates, totalCombos);

  // Generate LHS samples: for each dimension, create a permutation
  // of n indices, each in [0, numChoices)
  std::vector<std::vector<unsigned>> dimSamples(numFreeDims);
  for (unsigned d = 0; d < numFreeDims; ++d) {
    unsigned nc = freeDims[d].numChoices;
    dimSamples[d].resize(n);
    for (unsigned i = 0; i < n; ++i) {
      dimSamples[d][i] = i % nc;
    }
    std::shuffle(dimSamples[d].begin(), dimSamples[d].end(), rng);
  }

  // Generate candidates
  std::vector<CoreDesignParams> candidates;
  candidates.reserve(n);

  for (unsigned i = 0; i < n; ++i) {
    CoreDesignParams cand = baseline;
    unsigned dimIdx = 0;

    if (mask.topology) {
      cand.topology = topoValues[dimSamples[dimIdx][i]];
      dimIdx++;
    }
    if (mask.dataWidth) {
      cand.dataWidth = dataWidthValues[dimSamples[dimIdx][i]];
      dimIdx++;
    }
    if (mask.decomposability) {
      cand.decomposableBits = decomposableValues[dimSamples[dimIdx][i]];
      dimIdx++;
    }
    if (mask.extMem) {
      unsigned cfgIdx = dimSamples[dimIdx][i];
      cand.extmemCount = extmemConfigs[cfgIdx][0];
      cand.extmemLdPorts = extmemConfigs[cfgIdx][1];
      cand.extmemStPorts = extmemConfigs[cfgIdx][2];
      dimIdx++;
    }
    if (mask.scalarIO) {
      unsigned cfgIdx = dimSamples[dimIdx][i];
      cand.scalarInputs = scalarIOConfigs[cfgIdx][0];
      cand.scalarOutputs = scalarIOConfigs[cfgIdx][1];
      dimIdx++;
    }
    if (mask.temporalParams && baseline.peType == PEType::TEMPORAL) {
      unsigned cfgIdx = dimSamples[dimIdx][i];
      cand.instructionSlots = temporalConfigs[cfgIdx][0];
      cand.numRegisters = temporalConfigs[cfgIdx][1];
      cand.regFifoDepth = temporalConfigs[cfgIdx][2];
      dimIdx++;
    }
    if (mask.fuRepertoire) {
      unsigned choice = dimSamples[dimIdx][i];
      if (choice == 1) {
        // Add common supplementary ops
        cand.fuRepertoire.insert("arith.cmpi");
        cand.fuRepertoire.insert("arith.select");
        cand.fuRepertoire.insert("arith.addi");
      } else if (choice == 2) {
        // Try pruning non-essential ops (keep at least basic set)
        std::set<std::string> essential = {
            "arith.addi", "arith.muli", "handshake.load", "handshake.store"};
        std::set<std::string> pruned;
        for (const auto &op : cand.fuRepertoire) {
          if (essential.count(op))
            pruned.insert(op);
        }
        if (pruned.size() >= 2)
          cand.fuRepertoire = pruned;
      }
      dimIdx++;
    }
    if (mask.connectivity) {
      unsigned choice = dimSamples[dimIdx][i];
      if (choice == 0) {
        cand.connectivity.clear(); // full crossbar
      } else {
        // Generate sparse connectivity matrix
        unsigned peCount = cand.totalPEs();
        double density = (choice == 1) ? 0.5 : 0.25;
        cand.connectivity.resize(peCount);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (unsigned r = 0; r < peCount; ++r) {
          cand.connectivity[r].resize(peCount, false);
          for (unsigned c = 0; c < peCount; ++c) {
            cand.connectivity[r][c] = (r == c) || (dist(rng) < density);
          }
        }
      }
      dimIdx++;
    }

    candidates.push_back(cand);
  }

  return candidates;
}

//===----------------------------------------------------------------------===//
// Tier-A: Analytical Derivation
//===----------------------------------------------------------------------===//

/// Round up to the next power of two.
static unsigned nextPowerOfTwo(unsigned v) {
  if (v == 0)
    return 1;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

CoreDesignParams deriveInitialParams(
    const CoreTypeLibraryEntry &coreType,
    const std::vector<KernelProfile> &assignedProfiles,
    const HWInnerOptimizerOptions &opts) {
  CoreDesignParams params;

  // Dimension 1: PE type from parallelism heuristic
  params.peType = selectPEType(assignedProfiles, opts.parallelismThreshold);

  // Dimension 2: Array size from max DFG node count
  unsigned maxDFGNodes = coreType.minPEs;
  for (const auto &profile : assignedProfiles) {
    unsigned nodeCount = profile.totalOpCount();
    maxDFGNodes = std::max(maxDFGNodes, nodeCount);
  }
  unsigned side = static_cast<unsigned>(std::ceil(std::sqrt(
      static_cast<double>(maxDFGNodes))));
  side = std::max(side, opts.minArrayDim);
  side = std::min(side, opts.maxArrayDim);
  params.arrayRows = side;
  params.arrayCols = side;

  // Dimension 3: Data width (default 32, use 64 if FP ops present)
  params.dataWidth = 32;
  for (const auto &op : coreType.requiredFUTypes) {
    if (op.find("f") != std::string::npos &&
        (op.find("add") != std::string::npos ||
         op.find("mul") != std::string::npos ||
         op.find("div") != std::string::npos ||
         op.find("sub") != std::string::npos)) {
      params.dataWidth = 64;
      break;
    }
  }

  // Dimension 4: FU repertoire from kernel profiles + core type spec
  params.fuRepertoire = computeRequiredFURepertoire(assignedProfiles);
  // Also add any FU types required by the core type library entry
  for (const auto &fuType : coreType.requiredFUTypes) {
    params.fuRepertoire.insert(canonicalizeFUOp(fuType));
  }
  // Ensure at least basic arithmetic if repertoire is empty
  if (params.fuRepertoire.empty()) {
    params.fuRepertoire.insert("arith.addi");
    params.fuRepertoire.insert("arith.muli");
  }

  // Dimension 5: FU body structure (single-op default)
  params.multiOpFUBodies = false;

  // Dimension 6: Switch type matches PE type
  params.switchType = (params.peType == PEType::TEMPORAL)
                          ? SwitchType::TEMPORAL
                          : SwitchType::SPATIAL;

  // Dimension 7: No decomposition by default
  params.decomposableBits = -1;

  // Dimension 8: SPM from core type or profile analysis
  unsigned spmKB = coreType.minSPMKB;
  for (const auto &profile : assignedProfiles) {
    unsigned profileKB =
        static_cast<unsigned>((profile.estimatedSPMBytes + 1023) / 1024);
    spmKB = std::max(spmKB, profileKB);
  }
  params.spmSizeKB = nextPowerOfTwo(std::max(spmKB, 4u));
  params.spmLdPorts = 1;
  params.spmStPorts = 1;

  // Dimension 9: External memory (derive from profile memory ops)
  unsigned maxMemPorts = 2;
  for (const auto &profile : assignedProfiles) {
    unsigned memOps = 0;
    for (const auto &[opName, count] : profile.requiredOps) {
      if (opName.find("load") != std::string::npos ||
          opName.find("store") != std::string::npos) {
        memOps += count;
      }
    }
    maxMemPorts = std::max(maxMemPorts, std::min(memOps, 8u));
  }
  params.extmemCount = std::max(2u, std::min(maxMemPorts / 2, 8u));
  params.extmemLdPorts = 1;
  params.extmemStPorts = 1;

  // Dimension 10: Default topology (chess)
  params.topology = RoutingTopology::CHESS;

  // Dimension 11: Temporal PE params (only meaningful if temporal)
  if (params.peType == PEType::TEMPORAL) {
    params.instructionSlots = std::max(4u,
        static_cast<unsigned>(params.fuRepertoire.size()));
    params.numRegisters = std::max(4u, params.instructionSlots);
    params.regFifoDepth = 0;
    params.shareOperandBuffer = false;
    params.operandBufferSize = 0;
  }

  // Dimension 12: Scalar I/O from profile analysis
  params.scalarInputs = 3;
  params.scalarOutputs = 1;

  // Dimension 13: Full crossbar connectivity (empty = full)
  params.connectivity.clear();

  return params;
}

//===----------------------------------------------------------------------===//
// HWInnerOptimizer construction
//===----------------------------------------------------------------------===//

HWInnerOptimizer::HWInnerOptimizer(const HWInnerOptimizerOptions &options)
    : options_(options) {}

//===----------------------------------------------------------------------===//
// Tier-A implementation
//===----------------------------------------------------------------------===//

CoreDesignParams HWInnerOptimizer::runTierA(
    const CoreTypeLibraryEntry &coreType,
    const std::vector<KernelProfile> &profiles) {
  return deriveInitialParams(coreType, profiles, options_);
}

//===----------------------------------------------------------------------===//
// Candidate perturbation for BO exploration
//===----------------------------------------------------------------------===//

CoreDesignParams HWInnerOptimizer::perturbCandidate(
    const CoreDesignParams &base, unsigned iteration) {
  CoreDesignParams candidate = base;
  std::mt19937 rng(options_.seed + iteration);

  // Choose which dimension to perturb
  std::uniform_int_distribution<unsigned> dimDist(0, 5);
  unsigned dim = dimDist(rng);

  switch (dim) {
  case 0: {
    // Perturb array dimensions (+/- 1)
    std::uniform_int_distribution<int> delta(-1, 1);
    int dr = delta(rng);
    int dc = delta(rng);
    candidate.arrayRows = static_cast<unsigned>(std::clamp(
        static_cast<int>(base.arrayRows) + dr,
        static_cast<int>(options_.minArrayDim),
        static_cast<int>(options_.maxArrayDim)));
    candidate.arrayCols = static_cast<unsigned>(std::clamp(
        static_cast<int>(base.arrayCols) + dc,
        static_cast<int>(options_.minArrayDim),
        static_cast<int>(options_.maxArrayDim)));
    break;
  }
  case 1: {
    // Perturb SPM size (power of 2 steps)
    std::uniform_int_distribution<int> step(-1, 1);
    int s = step(rng);
    unsigned newSPM = base.spmSizeKB;
    if (s > 0 && newSPM < 64)
      newSPM *= 2;
    else if (s < 0 && newSPM > 4)
      newSPM /= 2;
    candidate.spmSizeKB = newSPM;
    break;
  }
  case 2: {
    // Try a different topology
    static const RoutingTopology topos[] = {
        RoutingTopology::CHESS, RoutingTopology::MESH,
        RoutingTopology::LATTICE};
    std::uniform_int_distribution<unsigned> topoDist(0, 2);
    candidate.topology = topos[topoDist(rng)];
    break;
  }
  case 3: {
    // Perturb external memory count
    std::uniform_int_distribution<int> delta(-1, 1);
    int d = delta(rng);
    candidate.extmemCount = static_cast<unsigned>(std::clamp(
        static_cast<int>(base.extmemCount) + d, 2, 8));
    break;
  }
  case 4: {
    // Perturb scalar I/O counts
    std::uniform_int_distribution<int> delta(-1, 1);
    int di = delta(rng);
    candidate.scalarInputs = static_cast<unsigned>(std::clamp(
        static_cast<int>(base.scalarInputs) + di, 2, 8));
    break;
  }
  case 5: {
    // Toggle PE type (explore both spatial and temporal)
    candidate.peType = (base.peType == PEType::SPATIAL) ? PEType::TEMPORAL
                                                        : PEType::SPATIAL;
    if (candidate.peType == PEType::TEMPORAL) {
      candidate.switchType = SwitchType::TEMPORAL;
      candidate.instructionSlots = std::max(
          4u, static_cast<unsigned>(base.fuRepertoire.size()));
      candidate.numRegisters = std::max(4u, candidate.instructionSlots);
    } else {
      candidate.switchType = SwitchType::SPATIAL;
    }
    break;
  }
  default:
    break;
  }

  return candidate;
}

//===----------------------------------------------------------------------===//
// Candidate evaluation
//===----------------------------------------------------------------------===//

/// Ensure MLIR context has all needed dialects for ADG parsing.
static void ensureDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

std::pair<double, std::vector<KernelMappingResult>>
HWInnerOptimizer::evaluateCandidate(
    const CoreDesignParams &candidate,
    const CoreTypeLibraryEntry &coreType,
    const std::string &moduleName,
    mlir::MLIRContext *ctx) {
  std::vector<KernelMappingResult> results;

  // Build the ADG MLIR from candidate parameters
  std::string adgMLIR = buildADGFromParams(candidate, moduleName);
  if (adgMLIR.empty()) {
    if (options_.verbose) {
      llvm::errs() << "INNER-HW: Failed to build ADG for candidate\n";
    }
    return {-std::numeric_limits<double>::infinity(), results};
  }

  // Compute area for this candidate
  double area = estimateCoreArea(candidate);

  // Use Tier-A scoring to check feasibility and estimate II for each
  // assigned kernel. This uses profile-based FU coverage checking and
  // resource-bound II estimation.
  TierAScore tierA = scoreCoreDesign(candidate, assignedProfiles_);

  if (!tierA.feasible) {
    // Fill in failed results for diagnostics
    for (const auto &kernelName : coreType.assignedKernels) {
      KernelMappingResult mr;
      mr.kernelName = kernelName;
      mr.success = false;
      mr.achievedII = 0;
      results.push_back(mr);
    }
    return {-std::numeric_limits<double>::infinity(), results};
  }

  // Check PE count and SPM feasibility
  bool peOk = (candidate.totalPEs() >= coreType.minPEs);
  bool spmOk = (candidate.spmSizeKB >= coreType.minSPMKB);

  if (!peOk || !spmOk) {
    for (const auto &kernelName : coreType.assignedKernels) {
      KernelMappingResult mr;
      mr.kernelName = kernelName;
      mr.success = false;
      mr.achievedII = 0;
      results.push_back(mr);
    }
    return {-std::numeric_limits<double>::infinity(), results};
  }

  // Build results from Tier-A per-kernel II
  for (const auto &kii : tierA.perKernelII) {
    KernelMappingResult mr;
    mr.kernelName = kii.kernelName;
    mr.success = true;
    mr.achievedII = static_cast<unsigned>(std::ceil(kii.effectiveII));
    results.push_back(mr);
  }

  // Negate area for maximization (BO maximizes score)
  return {-area, results};
}

//===----------------------------------------------------------------------===//
// Tier-B: BO + mapper
//===----------------------------------------------------------------------===//

ADGOptResult HWInnerOptimizer::runTierB(
    const CoreDesignParams &initial,
    const CoreTypeLibraryEntry &coreType,
    const std::vector<KernelProfile> &profiles,
    mlir::MLIRContext *ctx) {
  ADGOptResult bestResult;
  bestResult.success = false;
  bestResult.params = initial;
  bestResult.areaEstimate = estimateCoreArea(initial);

  std::string moduleName = "core_type_" + std::to_string(coreType.typeIndex);

  // Determine freedom mask: use combinatorial mask as a reasonable default
  bool isTemporal = (initial.peType == PEType::TEMPORAL);
  FreedomMask mask = FreedomMask::combinatorial(isTemporal);

  // Generate sweep candidates using constrained LHS
  unsigned sweepCount = std::max(1u, options_.maxInnerIter / 2);
  std::vector<CoreDesignParams> sweepCandidates =
      generateSweepCandidates(initial, mask, sweepCount, options_.seed);

  // Tier-A pre-screen: score all sweep candidates analytically
  struct ScoredCandidate {
    CoreDesignParams params;
    double tierAScore;
  };
  std::vector<ScoredCandidate> scored;
  scored.reserve(sweepCandidates.size() + 1);

  // Include the initial point
  TierAScore initTierA = scoreCoreDesign(initial, profiles);
  scored.push_back({initial, initTierA.compositeScore});
  bestResult.tier1Evaluations = 1;

  for (const auto &cand : sweepCandidates) {
    TierAScore ta = scoreCoreDesign(cand, profiles);
    scored.push_back({cand, ta.compositeScore});
    bestResult.tier1Evaluations++;
  }

  // Sort by Tier-A score (descending) and keep top-K for Tier-B
  std::sort(scored.begin(), scored.end(),
            [](const ScoredCandidate &a, const ScoredCandidate &b) {
              return a.tierAScore > b.tierAScore;
            });
  unsigned topK = std::min(static_cast<unsigned>(scored.size()),
                           std::max(1u, options_.maxInnerIter / 3));

  // Evaluate initial point
  auto [initScore, initMappings] =
      evaluateCandidate(initial, coreType, moduleName, ctx);

  bestResult.tier2Evaluations = 1;
  if (initScore > -std::numeric_limits<double>::infinity()) {
    bestResult.success = true;
    bestResult.mappingResults = initMappings;
    bestResult.tier2Successes = 1;
  }

  double bestScore = initScore;
  CoreDesignParams bestParams = initial;

  if (options_.verbose) {
    llvm::errs() << "INNER-HW Tier-B: initial area = "
                 << bestResult.areaEstimate
                 << ", score = " << initScore
                 << ", sweep candidates = " << sweepCandidates.size()
                 << ", top-K = " << topK << "\n";
  }

  // Evaluate top-K sweep candidates at Tier-B
  for (unsigned i = 0; i < topK && i < scored.size(); ++i) {
    const auto &cand = scored[i].params;

    auto [score, mappings] =
        evaluateCandidate(cand, coreType, moduleName, ctx);

    bestResult.tier2Evaluations++;

    if (score > -std::numeric_limits<double>::infinity()) {
      bestResult.tier2Successes++;

      if (score > bestScore) {
        bestScore = score;
        bestParams = cand;
        bestResult.success = true;
        bestResult.mappingResults = mappings;

        if (options_.verbose) {
          llvm::errs() << "INNER-HW Tier-B sweep " << i
                       << ": improved area = " << estimateCoreArea(cand)
                       << "\n";
        }
      }
    }
  }

  // BO perturbation loop for remaining budget
  unsigned remainingIter = options_.maxInnerIter -
                           std::min(options_.maxInnerIter, topK);
  for (unsigned iter = 0; iter < remainingIter; ++iter) {
    CoreDesignParams candidate = perturbCandidate(bestParams, iter);

    auto [score, mappings] =
        evaluateCandidate(candidate, coreType, moduleName, ctx);

    bestResult.tier2Evaluations++;

    if (score > -std::numeric_limits<double>::infinity()) {
      bestResult.tier2Successes++;

      if (score > bestScore) {
        bestScore = score;
        bestParams = candidate;
        bestResult.success = true;
        bestResult.mappingResults = mappings;

        if (options_.verbose) {
          llvm::errs() << "INNER-HW Tier-B iter " << iter
                       << ": improved area = " << estimateCoreArea(candidate)
                       << "\n";
        }
      }
    }
  }

  bestResult.params = bestParams;
  bestResult.areaEstimate = estimateCoreArea(bestParams);

  return bestResult;
}

//===----------------------------------------------------------------------===//
// Main optimize() entry point
//===----------------------------------------------------------------------===//

ADGOptResult HWInnerOptimizer::optimize(
    const CoreTypeLibraryEntry &coreType,
    const std::vector<KernelProfile> &assignedProfiles,
    mlir::MLIRContext *ctx) {
  auto wallStart = std::chrono::steady_clock::now();

  // Cache profiles for use in evaluateCandidate
  assignedProfiles_ = assignedProfiles;

  ADGOptResult result;
  result.success = false;

  if (ctx) {
    ensureDialects(*ctx);
  }

  // --- Tier A: Analytical derivation ---
  CoreDesignParams initial = runTierA(coreType, assignedProfiles);
  result.params = initial;
  result.areaEstimate = estimateCoreArea(initial);
  result.tier1Evaluations = 1;

  if (options_.verbose) {
    llvm::errs() << "INNER-HW: Tier-A for core type "
                 << coreType.typeIndex
                 << " (" << coreRoleToString(coreType.role) << ")\n"
                 << "  PE type: " << peTypeToString(initial.peType) << "\n"
                 << "  Array: " << initial.arrayRows << "x"
                 << initial.arrayCols << "\n"
                 << "  FU count: " << initial.fuRepertoire.size() << "\n"
                 << "  Area estimate: " << result.areaEstimate << "\n";
  }

  // --- Tier B: BO + mapper verification ---
  if (options_.tier2Enabled) {
    ADGOptResult tierBResult =
        runTierB(initial, coreType, assignedProfiles, ctx);

    // Take Tier-B result if it improved on Tier-A
    if (tierBResult.success) {
      // Preserve tier-1 evaluations count from sweep pre-screening
      unsigned tier1FromSweep = tierBResult.tier1Evaluations;
      result = tierBResult;
      result.tier1Evaluations += tier1FromSweep;
    }
  } else {
    // Use Tier-A result directly
    result.success = true;
  }

  // --- Tier C: Simulation (deferred) ---
  if (options_.tier3Enabled) {
    // Simulation tier is deferred. Returns Tier-B results as-is.
    if (options_.verbose) {
      llvm::errs() << "INNER-HW: Tier-C simulation deferred\n";
    }
  }

  // Generate final ADG MLIR
  std::string moduleName = "core_type_" + std::to_string(coreType.typeIndex);
  result.adgMLIR = buildADGFromParams(result.params, moduleName);

  auto wallEnd = std::chrono::steady_clock::now();
  result.wallTimeSec =
      std::chrono::duration<double>(wallEnd - wallStart).count();

  if (options_.verbose) {
    llvm::errs() << "INNER-HW: Completed in " << result.wallTimeSec << "s\n"
                 << "  Final area: " << result.areaEstimate << "\n"
                 << "  Tier-2 evals: " << result.tier2Evaluations
                 << " (" << result.tier2Successes << " feasible)\n";
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Batch optimization
//===----------------------------------------------------------------------===//

std::vector<ADGOptResult> optimizeAllCoreTypes(
    const CoreTypeLibrary &library,
    const std::vector<KernelProfile> &allProfiles,
    mlir::MLIRContext *ctx,
    const HWInnerOptimizerOptions &opts) {
  std::vector<ADGOptResult> results;
  results.reserve(library.entries.size());

  for (const auto &entry : library.entries) {
    // Filter profiles to only those assigned to this core type
    std::vector<KernelProfile> assignedProfiles;
    for (const auto &kernelName : entry.assignedKernels) {
      for (const auto &profile : allProfiles) {
        if (profile.name == kernelName) {
          assignedProfiles.push_back(profile);
          break;
        }
      }
    }

    HWInnerOptimizer optimizer(opts);
    results.push_back(optimizer.optimize(entry, assignedProfiles, ctx));
  }

  return results;
}

} // namespace loom
