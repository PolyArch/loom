//===-- TDGToSSGBuilder.cpp - TDG MLIR -> SSG conversion -------------------===//
//
// Walks a TDG MLIR module to build a SystemGraph<KernelNode, SSGDataDependency>.
//
//===----------------------------------------------------------------------===//

#include "loom/SystemCompiler/TDGToSSGBuilder.h"
#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/SystemCompiler/KernelProfiler.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace loom {

//===----------------------------------------------------------------------===//
// Helper: extract data volume from contract attributes
//===----------------------------------------------------------------------===//

namespace {

/// Try to extract a numeric data volume from a contract op's attributes.
/// The TDG MLIR currently encodes data volume in tile_shape or as
/// min_buffer_elements. Returns 0 if no volume information is present.
uint64_t extractDataVolume(loom::tdg::ContractOp contractOp) {
  // Check for min_buffer_elements as a proxy for data volume.
  if (auto minBuf = contractOp.getMinBufferElements())
    return static_cast<uint64_t>(minBuf.value());
  return 0;
}

/// Convert an MLIR type to a data type name string.
std::string typeToName(mlir::Type ty) {
  if (ty.isF32())
    return "f32";
  if (ty.isF64())
    return "f64";
  if (ty.isF16())
    return "f16";
  if (ty.isInteger(64))
    return "i64";
  if (ty.isInteger(32))
    return "i32";
  if (ty.isInteger(16))
    return "i16";
  if (ty.isInteger(8))
    return "i8";
  return "unknown";
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

  // Track kernel names seen for duplicate detection.
  std::set<std::string> seenKernels;

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
      node.kernelType = kernelOp.getKernelType().str();

      // Look up the corresponding DFG module.
      auto dfgIt = dfgModules.find(kernelName);
      if (dfgIt != dfgModules.end() && dfgIt->second) {
        node.hasDFG = true;

        // Profile the DFG module.
        node.computeProfile = profiler.profile(dfgIt->second, &ctx);
        if (node.computeProfile.name.empty())
          node.computeProfile.name = kernelName;

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
        node.computeProfile.name = kernelName;
      }

      ssg.addNode(std::move(node));
    });

    // Walk tdg.contract ops to create SSGDataDependency edges.
    graphOp.walk([&](loom::tdg::ContractOp contractOp) {
      SSGDataDependency dep;
      dep.producerName = contractOp.getProducer().str();
      dep.consumerName = contractOp.getConsumer().str();
      dep.ordering = contractOp.getOrdering().str();
      dep.dataTypeName = typeToName(contractOp.getDataType());
      dep.visibility = contractOp.getVisibility().str();
      dep.dataVolume = extractDataVolume(contractOp);

      ssg.addEdge(std::move(dep));
    });
  });

  return ssg;
}

} // namespace loom
