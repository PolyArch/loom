//===-- tapestry_adg_gen.cpp - Generate diverse CGRA core type ADGs -*- C++ -*-===//
//
// Generates multiple ADG MLIR files representing different CGRA core types
// for heterogeneous multi-core experiments. Each core type has different PE
// array sizes, arithmetic operation support, and memory bandwidth.
//
// Core types:
//   gp       : General Purpose  -- 6x6, integer arithmetic + bit ops
//   dsp      : DSP/Math         -- 6x6, integer + floating-point ops
//   ai       : AI/Matrix        -- 8x8 (larger), integer + float, more memory
//   ctrl     : Lightweight/Ctrl -- 4x4 (smaller), integer-only, minimal memory
//
//===----------------------------------------------------------------------===//

#include "loom/ADG/ADGBuilder.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>
#include <vector>

using namespace loom::adg;

static llvm::cl::opt<std::string>
    outputDir("output-dir",
              llvm::cl::desc("Directory for generated ADG files"),
              llvm::cl::init("out/adg_library"));

//===----------------------------------------------------------------------===//
// Common infrastructure
//===----------------------------------------------------------------------===//

static constexpr unsigned kDataWidth = 64;

static std::string bitsType(unsigned width = kDataWidth) {
  return "!fabric.bits<" + std::to_string(width) + ">";
}

/// Define the baseline dataflow FUs that every core type needs for the mapper
/// to handle control flow, memory access, and data routing. Without these, no
/// real kernel can be mapped.
///
/// Includes i64 variants of index_cast, trunci, extui, extsi that the clang
/// front-end produces on 64-bit targets. Also includes divui on index which
/// many loop-based DFGs require.
static std::vector<FUHandle>
defineBaselineFUs(ADGBuilder &builder, const std::string &prefix) {
  std::vector<FUHandle> fus;

  // Constants (all types observed in benchmark DFGs)
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_i32", "i32", "0 : i32"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_i64", "i64", "0 : i64"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_index", "index", "0 : index"));
  fus.push_back(builder.defineConstantFU(
      prefix + "_const_f32", "f32", "0.000000e+00 : f32"));

  // Index casts -- all type combinations from benchmark DFGs
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_index_to_i32", "index", "i32"));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_i32_to_index", "i32", "index"));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_index_to_i64", "index", "i64"));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_i64_to_index", "i64", "index"));
  fus.push_back(builder.defineIndexCastFU(
      prefix + "_i8_to_index", "i8", "index"));

  // Integer truncation and extension (clang x86_64 target)
  fus.push_back(builder.defineUnaryFU(
      prefix + "_trunci_i64_i32", "arith.trunci", "i64", "i32"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_extsi_i32_i64", "arith.extsi", "i32", "i64"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_extui_i32_i64", "arith.extui", "i32", "i64"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_extui_i8_i32", "arith.extui", "i8", "i32"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_extui_i8_i64", "arith.extui", "i8", "i64"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_extui_i1_i32", "arith.extui", "i1", "i32"));

  // Float/int conversion (needed by many benchmark domains)
  fus.push_back(builder.defineUnaryFU(
      prefix + "_uitofp_i32_f32", "arith.uitofp", "i32", "f32"));
  fus.push_back(builder.defineUnaryFU(
      prefix + "_sitofp_i32_f32", "arith.sitofp", "i32", "f32"));

  // Dataflow control FUs (all value types observed in benchmark DFGs)
  fus.push_back(builder.defineStreamFU(prefix + "_stream"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_i32", "i32"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_none", "none"));
  fus.push_back(builder.defineMuxFU(prefix + "_mux_index", "index"));
  fus.push_back(builder.defineJoinFU(prefix + "_join", 4));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i32", "i32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_index", "index"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_f32", "f32"));
  fus.push_back(builder.defineGateFU(prefix + "_gate_i1", "i1"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_i32", "i32"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_none", "none"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_f32", "f32"));
  fus.push_back(builder.defineCarryFU(prefix + "_carry_i64", "i64"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_i32", "i32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_none", "none"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_f32", "f32"));
  fus.push_back(builder.defineCondBrFU(prefix + "_cond_br_i64", "i64"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i32", "i32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_index", "index"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_f32", "f32"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i64", "i64"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_none", "none"));
  fus.push_back(builder.defineInvariantFU(prefix + "_invariant_i1", "i1"));

  // Memory access FUs
  fus.push_back(builder.defineLoadFU(prefix + "_load", "index", "i32"));
  fus.push_back(builder.defineStoreFU(prefix + "_store", "index", "i32"));

  // Comparison and selection (all types observed in DFGs)
  fus.push_back(builder.defineSelectFU(prefix + "_select_i32", "i32"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_index", "index"));
  fus.push_back(builder.defineSelectFU(prefix + "_select_i1", "i1"));
  fus.push_back(builder.defineCmpiFU(prefix + "_cmpi_i32", "i32", "slt"));
  fus.push_back(builder.defineCmpiFU(prefix + "_cmpi_i64", "i64", "slt"));

  // Index-typed arithmetic (loop index computations)
  fus.push_back(builder.defineBinaryFU(
      prefix + "_addi_index", "arith.addi", "index", "index"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_muli_index", "arith.muli", "index", "index"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_divui_index", "arith.divui", "index", "index"));

  // i64-typed arithmetic (produced by clang on 64-bit targets)
  fus.push_back(builder.defineBinaryFU(
      prefix + "_addi_i64", "arith.addi", "i64", "i64"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_muli_i64", "arith.muli", "i64", "i64"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_shli_i64", "arith.shli", "i64", "i64"));
  fus.push_back(builder.defineBinaryFU(
      prefix + "_shrui_i64", "arith.shrui", "i64", "i64"));

  return fus;
}

/// Build one core type ADG and export it.
///
/// Uses chess mesh topology with 4 PE ports per PE and distributed boundary
/// ports across left and right sides for memory and scalar I/O.
static void buildCoreTypeADG(
    const std::string &coreName,
    const std::string &outputPath,
    unsigned rows, unsigned cols,
    const std::function<void(ADGBuilder &, const std::string &,
                             std::vector<FUHandle> &)> &addArithFUs,
    unsigned numExtMems,
    unsigned numScalarInputs,
    unsigned numScalarOutputs) {

  constexpr unsigned kPEInputs = 4;
  constexpr unsigned kPEOutputs = 4;

  const std::string moduleName = coreName + "_core";
  ADGBuilder builder(moduleName);

  // Build FU list: baseline + core-specific arithmetic
  std::vector<FUHandle> fus = defineBaselineFUs(builder, moduleName);
  addArithFUs(builder, moduleName, fus);

  // Define PE with all FUs
  auto pe = builder.defineSpatialPE(
      moduleName + "_pe",
      std::vector<std::string>(kPEInputs, bitsType()),
      std::vector<std::string>(kPEOutputs, bitsType()),
      fus);

  // External memory port counts (ld=1, st=1):
  //   output ports: ld_data, ld_done, st_done = 3
  //   input ports: memref + ld_addr + st_addr + st_data = 4 (memref at 0)
  constexpr unsigned kExtMemOutputs = 3;
  constexpr unsigned kExtMemDataInputs = 3;

  // Distribute extmem ports across left/right sides
  unsigned leftIngressMems = (numExtMems + 1) / 2;
  unsigned rightIngressMems = numExtMems / 2;
  unsigned leftEgressMems = (numExtMems + 1) / 2;
  unsigned rightEgressMems = numExtMems / 2;

  ChessMeshOptions meshOpts;
  meshOpts.topLeftExtraInputs =
      leftIngressMems * kExtMemOutputs + numScalarInputs;
  meshOpts.topRightExtraInputs = rightIngressMems * kExtMemOutputs;
  meshOpts.bottomLeftExtraOutputs = leftEgressMems * kExtMemDataInputs;
  meshOpts.bottomRightExtraOutputs =
      rightEgressMems * kExtMemDataInputs + numScalarOutputs;

  auto mesh = builder.buildChessMesh(
      rows, cols,
      [&](unsigned, unsigned) { return pe; },
      meshOpts);

  // Define and instantiate external memory
  ExtMemorySpec extMemSpec;
  extMemSpec.name = moduleName + "_extmem";
  extMemSpec.ldPorts = 1;
  extMemSpec.stPorts = 1;
  extMemSpec.memrefType = "memref<?xi32>";
  extMemSpec.numRegion = 1;
  auto extMem = builder.defineExtMemory(extMemSpec);
  auto extMems = builder.instantiateExtMemArray(numExtMems, extMem, "extmem");
  auto memrefs = builder.addMemrefInputs("buffer", numExtMems, "memref<?xi32>");
  for (unsigned idx = 0; idx < extMems.size(); ++idx)
    builder.connectMemrefToExtMem(memrefs[idx], extMems[idx]);

  // Wire ext memory to boundary ports (round-robin left/right)
  unsigned leftIngressIdx = 0;
  unsigned rightIngressIdx = meshOpts.topLeftExtraInputs;
  unsigned leftEgressIdx = 0;
  unsigned rightEgressIdx = meshOpts.bottomLeftExtraOutputs;

  for (unsigned memIdx = 0; memIdx < extMems.size(); ++memIdx) {
    InstanceHandle mem = extMems[memIdx];
    unsigned &ingressIdx = (memIdx % 2 == 0) ? leftIngressIdx : rightIngressIdx;
    unsigned &egressIdx = (memIdx % 2 == 0) ? leftEgressIdx : rightEgressIdx;

    for (unsigned outPort = 0; outPort < kExtMemOutputs; ++outPort) {
      builder.connect(mem, outPort,
                      mesh.ingressPorts[ingressIdx].instance,
                      mesh.ingressPorts[ingressIdx].port);
      ++ingressIdx;
    }
    for (unsigned inPort = 0; inPort < kExtMemDataInputs; ++inPort) {
      builder.connect(mesh.egressPorts[egressIdx].instance,
                      mesh.egressPorts[egressIdx].port,
                      mem, 1 + inPort);
      ++egressIdx;
    }
  }

  // Wire scalar I/O through remaining boundary ports
  std::vector<unsigned> scalarIns = builder.addInputs(
      "scalar", std::vector<std::string>(numScalarInputs, bitsType()));
  std::vector<unsigned> scalarOuts = builder.addOutputs(
      "scalar_out", std::vector<std::string>(numScalarOutputs, bitsType()));

  unsigned scalarIngressIdx = leftIngressMems * kExtMemOutputs;
  for (unsigned idx = 0; idx < scalarIns.size(); ++idx, ++scalarIngressIdx)
    builder.connectInputToPort(scalarIns[idx],
                               mesh.ingressPorts[scalarIngressIdx]);

  unsigned scalarEgressIdx = numExtMems * kExtMemDataInputs;
  for (unsigned idx = 0; idx < scalarOuts.size(); ++idx, ++scalarEgressIdx)
    builder.connectPortToOutput(mesh.egressPorts[scalarEgressIdx],
                                scalarOuts[idx]);

  builder.exportMLIR(outputPath);
}

//===----------------------------------------------------------------------===//
// Core type definitions
//===----------------------------------------------------------------------===//

/// General Purpose core: integer arithmetic + bitwise ops + division.
static void buildGPCore(const std::string &outputPath) {
  buildCoreTypeADG(
      "gp", outputPath,
      /*rows=*/6, /*cols=*/6,
      [](ADGBuilder &builder, const std::string &prefix,
         std::vector<FUHandle> &fus) {
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addi", "arith.addi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subi", "arith.subi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_muli", "arith.muli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_andi", "arith.andi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_ori", "arith.ori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_xori", "arith.xori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_xori_i8", "arith.xori", "i8", "i8"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shli", "arith.shli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrsi", "arith.shrsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrui", "arith.shrui", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_divsi", "arith.divsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_remsi", "arith.remsi", "i32", "i32"));
      },
      /*numExtMems=*/4,
      /*numScalarInputs=*/6,
      /*numScalarOutputs=*/4);
}

/// DSP/Math core: integer + floating-point + math intrinsics.
static void buildDSPCore(const std::string &outputPath) {
  buildCoreTypeADG(
      "dsp", outputPath,
      /*rows=*/6, /*cols=*/6,
      [](ADGBuilder &builder, const std::string &prefix,
         std::vector<FUHandle> &fus) {
        // Integer ops (full set including bitwise for broad kernel coverage)
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addi", "arith.addi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subi", "arith.subi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_muli", "arith.muli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_andi", "arith.andi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_ori", "arith.ori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_xori", "arith.xori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shli", "arith.shli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrsi", "arith.shrsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrui", "arith.shrui", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_divsi", "arith.divsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_remsi", "arith.remsi", "i32", "i32"));
        // Floating-point ops
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addf", "arith.addf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subf", "arith.subf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_mulf", "arith.mulf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_divf", "arith.divf", "f32", "f32"));
        fus.push_back(builder.defineCmpfFU(
            prefix + "_cmpf", "f32", "olt"));
        fus.push_back(builder.defineSelectFU(
            prefix + "_select_f32", "f32"));
        // Float/int conversion
        fus.push_back(builder.defineUnaryFU(
            prefix + "_sitofp", "arith.sitofp", "i32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_uitofp", "arith.uitofp", "i32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_fptosi", "arith.fptosi", "f32", "i32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_negf", "arith.negf", "f32", "f32"));
        // Math intrinsics
        fus.push_back(builder.defineFUWithBody(
            prefix + "_fma", {"f32", "f32", "f32"}, {"f32"},
            "%0 = math.fma %arg0, %arg1, %arg2 : f32\n"
            "fabric.yield %0 : f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_absf", "math.absf", "f32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_sqrt", "math.sqrt", "f32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_sin", "math.sin", "f32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_cos", "math.cos", "f32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_exp", "math.exp", "f32", "f32"));
        fus.push_back(builder.defineFUWithBody(
            prefix + "_floor", {"f32"}, {"f32"},
            "%0 = math.floor %arg0 : f32\n"
            "fabric.yield %0 : f32"));
      },
      /*numExtMems=*/4,
      /*numScalarInputs=*/6,
      /*numScalarOutputs=*/4);
}

/// AI/Matrix core: larger array for dense matrix computation.
static void buildAICore(const std::string &outputPath) {
  buildCoreTypeADG(
      "ai", outputPath,
      /*rows=*/8, /*cols=*/8,
      [](ADGBuilder &builder, const std::string &prefix,
         std::vector<FUHandle> &fus) {
        // Integer ops (full set for broad kernel coverage)
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addi", "arith.addi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subi", "arith.subi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_muli", "arith.muli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_andi", "arith.andi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_ori", "arith.ori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_xori", "arith.xori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shli", "arith.shli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrsi", "arith.shrsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrui", "arith.shrui", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_divsi", "arith.divsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_remsi", "arith.remsi", "i32", "i32"));
        // Floating-point ops for matrix math
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addf", "arith.addf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subf", "arith.subf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_mulf", "arith.mulf", "f32", "f32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_divf", "arith.divf", "f32", "f32"));
        fus.push_back(builder.defineCmpfFU(
            prefix + "_cmpf", "f32", "olt"));
        fus.push_back(builder.defineSelectFU(
            prefix + "_select_f32", "f32"));
        // Float/int conversion
        fus.push_back(builder.defineUnaryFU(
            prefix + "_sitofp", "arith.sitofp", "i32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_uitofp", "arith.uitofp", "i32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_fptosi", "arith.fptosi", "f32", "i32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_negf", "arith.negf", "f32", "f32"));
        // Math intrinsics
        fus.push_back(builder.defineFUWithBody(
            prefix + "_fma", {"f32", "f32", "f32"}, {"f32"},
            "%0 = math.fma %arg0, %arg1, %arg2 : f32\n"
            "fabric.yield %0 : f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_absf", "math.absf", "f32", "f32"));
        fus.push_back(builder.defineUnaryFU(
            prefix + "_sqrt", "math.sqrt", "f32", "f32"));
      },
      /*numExtMems=*/4,
      /*numScalarInputs=*/6,
      /*numScalarOutputs=*/4);
}

/// Lightweight/Control core: small array, integer-only.
static void buildCtrlCore(const std::string &outputPath) {
  buildCoreTypeADG(
      "ctrl", outputPath,
      /*rows=*/4, /*cols=*/4,
      [](ADGBuilder &builder, const std::string &prefix,
         std::vector<FUHandle> &fus) {
        fus.push_back(builder.defineBinaryFU(
            prefix + "_addi", "arith.addi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_subi", "arith.subi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_muli", "arith.muli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_andi", "arith.andi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_ori", "arith.ori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_xori", "arith.xori", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shli", "arith.shli", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrsi", "arith.shrsi", "i32", "i32"));
        fus.push_back(builder.defineBinaryFU(
            prefix + "_shrui", "arith.shrui", "i32", "i32"));
      },
      /*numExtMems=*/2,
      /*numScalarInputs=*/4,
      /*numScalarOutputs=*/2);
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Generate diverse CGRA core type ADGs for heterogeneous multi-core\n");

  // Ensure output directory exists
  if (auto ec = llvm::sys::fs::create_directories(outputDir.getValue())) {
    llvm::errs() << "error: cannot create output directory '"
                 << outputDir << "': " << ec.message() << "\n";
    return 1;
  }

  llvm::SmallString<256> path;
  auto makePath = [&](const std::string &name) -> std::string {
    path = outputDir.getValue();
    llvm::sys::path::append(path, name);
    return std::string(path);
  };

  llvm::outs() << "Generating core type ADGs to " << outputDir << "/\n";

  buildGPCore(makePath("gp_core.fabric.mlir"));
  llvm::outs() << "  [1/4] gp_core.fabric.mlir    (6x6 chess, integer+bitwise)\n";

  buildDSPCore(makePath("dsp_core.fabric.mlir"));
  llvm::outs() << "  [2/4] dsp_core.fabric.mlir   (6x6 chess, integer+float+math)\n";

  buildAICore(makePath("ai_core.fabric.mlir"));
  llvm::outs() << "  [3/4] ai_core.fabric.mlir    (8x8 chess, int+float, 4 extmem)\n";

  buildCtrlCore(makePath("ctrl_core.fabric.mlir"));
  llvm::outs() << "  [4/4] ctrl_core.fabric.mlir  (4x4 chess, integer-only)\n";

  llvm::outs() << "Done. Generated 4 core type ADGs.\n";
  return 0;
}
