//===-- TDGToSSGBuilder.cpp - TDG MLIR -> SSG conversion -------------------===//
//
// Walks a TDG MLIR module to build an SSG (SystemGraph<KernelNode, DataDependency>).
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/TDGToSSGBuilder.h"
#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SystemCompiler/Contract.h"
#include "loom/SystemCompiler/KernelProfiler.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: extract data volume from contract attributes
//===----------------------------------------------------------------------===//

namespace {

/// Return the byte width of an MLIR type (0 if unknown).
uint64_t typeSizeBytes(mlir::Type ty) {
  if (ty.isF64() || ty.isInteger(64))
    return 8;
  if (ty.isF32() || ty.isInteger(32))
    return 4;
  if (ty.isF16() || ty.isBF16() || ty.isInteger(16))
    return 2;
  if (ty.isInteger(8))
    return 1;
  // Conservative fallback for unknown types.
  return 0;
}

/// Parse tile_shape from ContractOp, compute volume = product(dims) * typeSize.
/// Returns 0 when tile_shape is absent or contains symbolic (non-numeric) dims.
uint64_t extractDataVolume(loom::tdg::ContractOp contractOp) {
  auto tileShapeAttr = contractOp.getTileShapeAttr();
  if (!tileShapeAttr)
    return 0;

  auto dims = loom::parseShapeExpr(tileShapeAttr.getValue().str());
  if (dims.empty())
    return 0;

  uint64_t product = 1;
  for (const auto &dimStr : dims) {
    // Try to parse as a number; bail on symbolic dimensions.
    char *end = nullptr;
    unsigned long val = std::strtoul(dimStr.c_str(), &end, 10);
    if (end == dimStr.c_str() || *end != '\0' || val == 0)
      return 0;
    product *= val;
  }

  uint64_t elemBytes = typeSizeBytes(contractOp.getDataType());
  if (elemBytes == 0)
    return 0;

  return product * elemBytes;
}

} // namespace

//===----------------------------------------------------------------------===//
// TDGToSSGBuilder::build
//===----------------------------------------------------------------------===//

SSG TDGToSSGBuilder::build(
    mlir::ModuleOp tdgModule,
    const std::map<std::string, mlir::ModuleOp> &dfgModules,
    mlir::MLIRContext &ctx) {

  SSG ssg;
  KernelProfiler profiler;

  // Track kernel names seen for duplicate detection, and name -> NodeId.
  std::set<std::string> seenKernels;
  std::map<std::string, SSG::NodeId> nameToId;

  // Walk tdg.graph ops (there should be exactly one).
  tdgModule.walk([&](loom::tdg::GraphOp graphOp) {
    // Walk tdg.kernel ops to create KernelNode entries.
    graphOp.walk([&](loom::tdg::KernelOp kernelOp) {
      std::string kernelName = kernelOp.getSymName().str();

      // Check for duplicate kernel names.
      if (seenKernels.count(kernelName)) {
        llvm::errs() << "TDGToSSGBuilder: duplicate kernel name '"
                     << kernelName << "', skipping\n";
        return;
      }
      seenKernels.insert(kernelName);

      KernelNode node;
      node.name = kernelName;
      node.kernelId = kernelName;
      node.kernelType = kernelOp.getKernelType().str();

      // Look up the corresponding DFG module.
      auto dfgIt = dfgModules.find(kernelName);
      if (dfgIt != dfgModules.end() && dfgIt->second) {
        node.hasDFG = true;

        // Profile the DFG module and convert to lightweight ComputeProfile.
        KernelProfile kp = profiler.profile(dfgIt->second, &ctx);
        KernelNode::ComputeProfile cp;
        cp.estimatedMinII = kp.estimatedMinII;
        cp.estimatedSPMBytes = kp.estimatedSPMBytes;
        cp.estimatedComputeCycles = kp.estimatedComputeCycles;
        node.computeProfile = cp;

        // Collect variant names from the DFG modules map.
        // Variants are typically keyed as "kernelName_v0", "kernelName_v1", etc.
        for (const auto &[key, _] : dfgModules) {
          if (key == kernelName || key.find(kernelName + "_v") == 0)
            node.variantSet.insert(key);
        }
      } else {
        // Missing DFG module: emit diagnostic, use empty profile.
        llvm::errs() << "TDGToSSGBuilder: no DFG module for kernel '"
                     << kernelName << "', using empty profile\n";
        node.hasDFG = false;
      }

      SSG::NodeId nid = ssg.addNode(std::move(node));
      nameToId[kernelName] = nid;
    });

    // Walk tdg.contract ops to create DataDependency edges.
    graphOp.walk([&](loom::tdg::ContractOp contractOp) {
      std::string producerName = contractOp.getProducer().str();
      std::string consumerName = contractOp.getConsumer().str();

      auto srcIt = nameToId.find(producerName);
      auto dstIt = nameToId.find(consumerName);
      if (srcIt == nameToId.end() || dstIt == nameToId.end()) {
        llvm::errs() << "TDGToSSGBuilder: edge references unknown kernel ("
                     << producerName << " -> " << consumerName
                     << "), skipping\n";
        return;
      }

      DataDependency dep;
      dep.producerKernel = producerName;
      dep.consumerKernel = consumerName;
      dep.dataVolume = extractDataVolume(contractOp);

      ssg.addEdge(srcIt->second, dstIt->second, std::move(dep));
    });
  });

  return ssg;
}

} // namespace loom
