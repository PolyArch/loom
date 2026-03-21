#include "SVGenInternal.h"

#include "fcc/SVGen/SVEmitter.h"
#include "fcc/SVGen/SVModuleRegistry.h"

#include "fcc/Dialect/Fabric/FabricOps.h"
#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

namespace fcc {
namespace svgen {

// Compute config bit count for a spatial switch.
static unsigned computeSpatialSwConfigBits(fcc::fabric::SpatialSwOp op) {
  auto fnType = op.getFunctionType();
  unsigned numIn = fnType.getNumInputs();
  unsigned numOut = fnType.getNumResults();

  // Connectivity-constrained bit count.
  unsigned routeBits = 0;
  if (auto connTable = op.getConnectivityTable()) {
    for (auto rowAttr : *connTable) {
      auto row = mlir::cast<mlir::StringAttr>(rowAttr).getValue();
      for (char ch : row) {
        if (ch == '1')
          ++routeBits;
      }
    }
  } else {
    routeBits = numIn * numOut;
  }

  // Per-input discard bit.
  return routeBits + numIn;
}

// Compute config bit count for a temporal switch.
static unsigned computeTemporalSwConfigBits(fcc::fabric::TemporalSwOp op) {
  auto fnType = op.getFunctionType();
  unsigned numIn = fnType.getNumInputs();
  unsigned numOut = fnType.getNumResults();
  unsigned slotCount = static_cast<unsigned>(op.getNumRouteTable());

  // Each slot: valid(1) + tag(tagWidth) + route bits.
  unsigned tagWidth = 0;
  if (numIn > 0) {
    mlir::Type firstIn = fnType.getInput(0);
    tagWidth = SVEmitter::getTagWidth(firstIn);
  }
  if (tagWidth == 0)
    tagWidth = 1;

  unsigned routeBitsPerSlot = 0;
  if (auto connTable = op.getConnectivityTable()) {
    for (auto rowAttr : *connTable) {
      auto row = mlir::cast<mlir::StringAttr>(rowAttr).getValue();
      for (char ch : row) {
        if (ch == '1')
          ++routeBitsPerSlot;
      }
    }
  } else {
    routeBitsPerSlot = numIn * numOut;
  }

  return slotCount * (1 + tagWidth + routeBitsPerSlot);
}

// Compute config bit count for an add_tag.
static unsigned computeAddTagConfigBits(fcc::fabric::AddTagOp op) {
  mlir::Type resultType = op.getResult().getType();
  return SVEmitter::getTagWidth(resultType);
}

// Compute config bit count for a map_tag.
static unsigned computeMapTagConfigBits(fcc::fabric::MapTagOp op) {
  unsigned tableSize = static_cast<unsigned>(op.getTableSize());
  if (tableSize == 0)
    return 0;
  mlir::Type inputType = op.getTagged().getType();
  mlir::Type outputType = op.getResult().getType();
  unsigned inTagW = SVEmitter::getTagWidth(inputType);
  unsigned outTagW = SVEmitter::getTagWidth(outputType);
  if (inTagW == 0)
    inTagW = 1;
  if (outTagW == 0)
    outTagW = 1;
  // Per-entry: valid(1) + srcTag + dstTag.
  return tableSize * (1 + inTagW + outTagW);
}

// Compute config bit count for a FIFO (only bypassable FIFOs have config).
static unsigned computeFifoConfigBits(fcc::fabric::FifoOp op) {
  if (op.getBypassable())
    return 1;
  return 0;
}

// Compute config bit count for a spatial PE.
static unsigned computeSpatialPEConfigBits(fcc::fabric::SpatialPEOp op) {
  auto fnType = op.getFunctionType();
  unsigned numPEInputs = fnType.getNumInputs();
  unsigned numPEOutputs = fnType.getNumResults();

  // Count FUs and gather max values.
  unsigned numFU = 0;
  unsigned maxFUInputs = 0;
  unsigned maxFUOutputs = 0;
  unsigned maxFUConfigBits = 0;

  op.getBody().front().walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    ++numFU;
    auto fuFnType = fuOp.getFunctionType();
    unsigned fuIn = fuFnType.getNumInputs();
    unsigned fuOut = fuFnType.getNumResults();
    maxFUInputs = std::max(maxFUInputs, fuIn);
    maxFUOutputs = std::max(maxFUOutputs, fuOut);

    // Count mux ops for FU internal config bits.
    unsigned fuCfgBits = 0;
    fuOp.getBody().front().walk([&](fcc::fabric::MuxOp muxOp) {
      unsigned numMuxIn = muxOp.getInputs().size();
      unsigned selBits = numMuxIn > 1 ? llvm::Log2_32_Ceil(numMuxIn) : 0;
      fuCfgBits += selBits + 2; // sel + discard + disconnect
    });
    maxFUConfigBits = std::max(maxFUConfigBits, fuCfgBits);
  });

  if (numFU == 0)
    return 0;

  // enable(1) + opcode(clog2(numFU))
  unsigned opcodeBits = numFU > 1 ? llvm::Log2_32_Ceil(numFU) : 0;
  unsigned enableBit = 1;

  // Per-FU-input mux: clog2(numPEInputs) + discard + disconnect
  unsigned perInputMuxBits = 0;
  if (numPEInputs > 0) {
    unsigned selBits =
        numPEInputs > 1 ? llvm::Log2_32_Ceil(numPEInputs) : 0;
    perInputMuxBits = selBits + 2; // + discard + disconnect
  }
  unsigned inputMuxBits = maxFUInputs * perInputMuxBits;

  // Per-FU-output demux: clog2(numPEOutputs) + discard + disconnect
  unsigned perOutputDemuxBits = 0;
  if (numPEOutputs > 0) {
    unsigned selBits =
        numPEOutputs > 1 ? llvm::Log2_32_Ceil(numPEOutputs) : 0;
    perOutputDemuxBits = selBits + 2;
  }
  unsigned outputDemuxBits = maxFUOutputs * perOutputDemuxBits;

  return enableBit + opcodeBits + inputMuxBits + outputDemuxBits +
         maxFUConfigBits;
}

// Compute config bit count for a temporal PE.
static unsigned computeTemporalPEConfigBits(fcc::fabric::TemporalPEOp op) {
  auto fnType = op.getFunctionType();
  unsigned numPEInputs = fnType.getNumInputs();
  unsigned numPEOutputs = fnType.getNumResults();
  unsigned numInstruction = static_cast<unsigned>(op.getNumInstruction());
  unsigned numRegister = static_cast<unsigned>(op.getNumRegister());

  unsigned numFU = 0;
  unsigned maxFUInputs = 0;
  unsigned maxFUOutputs = 0;
  unsigned maxFUConfigBits = 0;

  op.getBody().front().walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    ++numFU;
    auto fuFnType = fuOp.getFunctionType();
    unsigned fuIn = fuFnType.getNumInputs();
    unsigned fuOut = fuFnType.getNumResults();
    maxFUInputs = std::max(maxFUInputs, fuIn);
    maxFUOutputs = std::max(maxFUOutputs, fuOut);

    unsigned fuCfgBits = 0;
    fuOp.getBody().front().walk([&](fcc::fabric::MuxOp muxOp) {
      unsigned numMuxIn = muxOp.getInputs().size();
      unsigned selBits = numMuxIn > 1 ? llvm::Log2_32_Ceil(numMuxIn) : 0;
      fuCfgBits += selBits + 2;
    });
    maxFUConfigBits = std::max(maxFUConfigBits, fuCfgBits);
  });

  if (numFU == 0 || numInstruction == 0)
    return 0;

  unsigned tagWidth = 0;
  if (numPEInputs > 0)
    tagWidth = SVEmitter::getTagWidth(fnType.getInput(0));
  if (tagWidth == 0)
    tagWidth = 1;

  // Per-instruction config layout:
  // valid(1) + tag(tagWidth) + opcode(clog2(numFU))
  unsigned opcodeBits = numFU > 1 ? llvm::Log2_32_Ceil(numFU) : 0;

  // Per-operand source: from PE input mux or register
  // sourceIsReg(1) + muxSel(clog2(numPEInputs)) or regIdx(clog2(numRegister))
  unsigned maxSourceBits = 0;
  unsigned inputMuxBits =
      numPEInputs > 1 ? llvm::Log2_32_Ceil(numPEInputs) : 0;
  unsigned regIdxBits = numRegister > 1 ? llvm::Log2_32_Ceil(numRegister) : 0;
  maxSourceBits = 1 + std::max(inputMuxBits, regIdxBits);
  unsigned operandBits = maxFUInputs * maxSourceBits;

  // Per-result dest: to PE output or register
  unsigned destIsReg = 1;
  unsigned outputDemuxBits =
      numPEOutputs > 1 ? llvm::Log2_32_Ceil(numPEOutputs) : 0;
  unsigned maxDestBits = destIsReg + std::max(outputDemuxBits, regIdxBits);
  unsigned resultBits = maxFUOutputs * maxDestBits;

  unsigned perInstructionBits =
      1 + tagWidth + opcodeBits + operandBits + resultBits + maxFUConfigBits;

  return numInstruction * perInstructionBits;
}

// Compute config bits for memory module.
static unsigned computeMemoryConfigBits(fcc::fabric::MemoryOp op) {
  unsigned numRegion = static_cast<unsigned>(std::max<int64_t>(1, op.getNumRegion()));
  // Per-region: address offset word (32 bits).
  return numRegion * 32;
}

// Compute config bits for extmemory module.
static unsigned computeExtMemoryConfigBits(fcc::fabric::ExtMemoryOp op) {
  unsigned numRegion = static_cast<unsigned>(std::max<int64_t>(1, op.getNumRegion()));
  return numRegion * 32;
}

std::vector<ModuleConfigSlice>
computeConfigLayout(fcc::fabric::ModuleOp fabricMod) {
  std::vector<ModuleConfigSlice> slices;
  unsigned totalWordOffset = 0;

  auto addSlice = [&](llvm::StringRef name, unsigned bits) {
    ModuleConfigSlice slice;
    slice.moduleName = name.str();
    slice.bitCount = bits;
    slice.wordCount = (bits + 31) / 32;
    slice.wordOffset = totalWordOffset;
    totalWordOffset += slice.wordCount;
    slices.push_back(std::move(slice));
  };

  auto &body = fabricMod.getBody().front();

  // Walk in block order to match ConfigGen ordering.
  for (auto &op : body.getOperations()) {
    if (auto spatialSw = mlir::dyn_cast<fcc::fabric::SpatialSwOp>(op)) {
      std::string name = SVEmitter::sanitizeName(
          spatialSw.getSymName().value_or("spatial_sw"));
      addSlice(name, computeSpatialSwConfigBits(spatialSw));
    } else if (auto temporalSw =
                   mlir::dyn_cast<fcc::fabric::TemporalSwOp>(op)) {
      std::string name = SVEmitter::sanitizeName(
          temporalSw.getSymName().value_or("temporal_sw"));
      addSlice(name, computeTemporalSwConfigBits(temporalSw));
    } else if (auto addTag = mlir::dyn_cast<fcc::fabric::AddTagOp>(op)) {
      addSlice("add_tag", computeAddTagConfigBits(addTag));
    } else if (auto mapTag = mlir::dyn_cast<fcc::fabric::MapTagOp>(op)) {
      addSlice("map_tag", computeMapTagConfigBits(mapTag));
    } else if (auto fifo = mlir::dyn_cast<fcc::fabric::FifoOp>(op)) {
      std::string name =
          SVEmitter::sanitizeName(fifo.getSymName().value_or("fifo"));
      unsigned bits = computeFifoConfigBits(fifo);
      if (bits > 0)
        addSlice(name, bits);
    } else if (auto spatialPE =
                   mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
      std::string name = SVEmitter::sanitizeName(
          spatialPE.getSymName().value_or("spatial_pe"));
      addSlice(name, computeSpatialPEConfigBits(spatialPE));
    } else if (auto temporalPE =
                   mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
      std::string name = SVEmitter::sanitizeName(
          temporalPE.getSymName().value_or("temporal_pe"));
      addSlice(name, computeTemporalPEConfigBits(temporalPE));
    } else if (auto memOp = mlir::dyn_cast<fcc::fabric::MemoryOp>(op)) {
      std::string name =
          SVEmitter::sanitizeName(memOp.getSymName().value_or("memory"));
      addSlice(name, computeMemoryConfigBits(memOp));
    } else if (auto extMemOp =
                   mlir::dyn_cast<fcc::fabric::ExtMemoryOp>(op)) {
      std::string name = SVEmitter::sanitizeName(
          extMemOp.getSymName().value_or("extmemory"));
      addSlice(name, computeExtMemoryConfigBits(extMemOp));
    }
    // fabric.instance, fabric.del_tag, fabric.yield: no config bits.
  }

  return slices;
}

} // namespace svgen
} // namespace fcc
