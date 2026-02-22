//===-- config_region_pack.cpp - Multi-region config packing test --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that ConfigGen packs multiple memory region entries at distinct bit
// positions (not all at bit 0), so multi-region configs do not overwrite each
// other.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <cstdio>
#include <fstream>

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

void setIntAttr(Node *node, mlir::MLIRContext &ctx,
                llvm::StringRef name, int64_t value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value)));
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test: Memory node with 2 regions, 2 mapped operations.
  // The config packing should place region 0 and region 1 entries at
  // different bit positions. If the bug (bit position reset per region)
  // is present, both entries would start at bit 0 and region 1 would
  // overwrite region 0.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // Two DFG memory operations.
    auto sw0 = std::make_unique<Node>();
    sw0->kind = Node::OperationNode;
    setStringAttr(sw0.get(), ctx, "op_name", "handshake.load");
    setStringAttr(sw0.get(), ctx, "resource_class", "functional");
    IdIndex swId0 = dfg.addNode(std::move(sw0));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId0;
      port->direction = Port::Input;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId0)->inputPorts.push_back(pid);
    }

    auto sw1 = std::make_unique<Node>();
    sw1->kind = Node::OperationNode;
    setStringAttr(sw1.get(), ctx, "op_name", "handshake.store");
    setStringAttr(sw1.get(), ctx, "resource_class", "functional");
    IdIndex swId1 = dfg.addNode(std::move(sw1));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId1;
      port->direction = Port::Input;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId1)->inputPorts.push_back(pid);
    }

    // ADG: memory node with numRegion=2, tag_width=4.
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.memory");
    setStringAttr(hwNode.get(), ctx, "resource_class", "memory");
    setStringAttr(hwNode.get(), ctx, "sym_name", "mem0");
    setIntAttr(hwNode.get(), ctx, "numRegion", 2);
    setIntAttr(hwNode.get(), ctx, "tag_width", 4);
    IdIndex hwId = adg.addNode(std::move(hwNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    // Map both SW nodes to the memory HW node.
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(swId0, hwId, dfg, adg);
    state.mapNode(swId1, hwId, dfg, adg);

    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 2);

    TEST_ASSERT(state.swNodeToHwNode[swId0] == hwId);
    TEST_ASSERT(state.swNodeToHwNode[swId1] == hwId);

    // Run ConfigGen and verify bit-level output for multi-region memory.
    // With the fix: region 0 starts at bit 0, region 1 starts at
    // bit REGION_ENTRY_WIDTH. tag_width=4, so entry width =
    // 1(valid) + 4(start_tag) + 5(end_tag) + 16(addr_offset) = 26 bits.
    // Region 1 starts at bit 26, total 52 bits across 2 words.
    //
    // Without the fix: both regions start at bit 0, region 1
    // overwrites region 0.
    {
      // Create temp directory for output files.
      llvm::SmallString<128> tmpDir;
      llvm::sys::fs::createUniqueDirectory("loom-test", tmpDir);
      llvm::SmallString<128> basePath(tmpDir);
      llvm::sys::path::append(basePath, "test_config");

      ConfigGen gen;
      bool genOk = gen.generate(state, dfg, adg, std::string(basePath),
                                false, "balanced", 0);
      TEST_ASSERT(genOk);

      // Read the binary config and verify bit layout.
      std::string binPath = std::string(basePath) + ".config.bin";
      std::ifstream binFile(binPath, std::ios::binary);
      TEST_ASSERT(binFile.good());

      std::vector<uint8_t> binData(
          (std::istreambuf_iterator<char>(binFile)),
          std::istreambuf_iterator<char>());
      binFile.close();

      // Config should have at least 2 words (52 bits > 32 bits).
      TEST_ASSERT(binData.size() >= 8);

      // Reconstruct words (little-endian).
      uint32_t word0 = binData[0] | (binData[1] << 8) |
                        (binData[2] << 16) | (binData[3] << 24);
      uint32_t word1 = binData[4] | (binData[5] << 8) |
                        (binData[6] << 16) | (binData[7] << 24);

      // Region 0 (bits 0-25):
      //   valid(bit 0) = 1
      //   start_tag(bits 1-4) = 0
      //   end_tag(bits 5-9) = 1 (LSB first: bit 5 = 1, rest 0)
      //   addr_offset(bits 10-25) = 0
      // Region 0 contributes: bit 0 = 1, bit 5 = 1 => 0x21 in low bits.
      TEST_ASSERT((word0 & 0x1) == 1);   // Region 0 valid bit.
      TEST_ASSERT((word0 & 0x20) != 0);  // Region 0 end_tag LSB at bit 5.

      // Region 1 (bits 26-51):
      //   valid(bit 26) = 1
      //   start_tag(bits 27-30) = 1 (LSB: bit 27 = 1)
      //   end_tag(bits 31-35) = 2 (LSB first: bit 31=0, bit 32=1)
      // Region 1 valid bit at bit 26 of word0.
      TEST_ASSERT((word0 & (1U << 26)) != 0);  // Region 1 valid.
      TEST_ASSERT((word0 & (1U << 27)) != 0);  // Region 1 start_tag LSB.

      // Word1 should have end_tag bit 1 of region 1 at bit 0.
      TEST_ASSERT((word1 & 0x1) != 0);  // Region 1 end_tag=2, bit 1 set.

      // Clean up temp files.
      std::remove(binPath.c_str());
      std::string addrPath = std::string(basePath) + "_addr.h";
      std::remove(addrPath.c_str());
      llvm::sys::fs::remove(tmpDir);
    }
  }

  return 0;
}
