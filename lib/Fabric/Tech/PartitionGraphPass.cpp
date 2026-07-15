#include "Fabric/Tech/Passes.h"

#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/Partitioner/Materializer.h"
#include "Fabric/Tech/Partitioner/Partitioner.h"
#include "Fabric/Tech/TemplateLibrary.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace {

struct PartitionGraphPass
    : public ::mlir::PassWrapper<PartitionGraphPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionGraphPass)

  PartitionGraphPass() = default;
  PartitionGraphPass(const PartitionGraphPass &other)
      : ::mlir::PassWrapper<PartitionGraphPass,
                            ::mlir::OperationPass<::mlir::ModuleOp>>(other) {
    configPath = other.configPath;
  }
  explicit PartitionGraphPass(std::string path) {
    configPath = std::move(path);
  }

  ::llvm::StringRef getArgument() const final {
    return "loom-partition-graph-into-subgraphs";
  }
  ::llvm::StringRef getDescription() const final {
    return "Partition each dataflow.graph body into dataflow.subgraphs "
           "according to the configured tech-mapping algorithm.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::dataflow::DataflowDialect, ::mlir::arith::ArithDialect,
                    ::mlir::math::MathDialect>();
  }

  Option<std::string> configPath{
      *this, "config",
      ::llvm::cl::desc("Path to a resolved YAML or JSON configuration file."),
      ::llvm::cl::init("")};

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();

    ::llvm::Expected<::loom::ResolvedConfig> resolved =
        configPath.empty() ? ::llvm::Expected<::loom::ResolvedConfig>(
                                 ::loom::defaultResolvedConfig())
                           : ::loom::loadResolvedConfig(configPath);
    if (!resolved) {
      ::mlir::emitError(module.getLoc())
          << "loom-partition-graph-into-subgraphs: "
          << ::llvm::toString(resolved.takeError());
      return signalPassFailure();
    }
    const ::loom::ResolvedFabricTechMapConfig &cfg = resolved->fabricTechMap;

    ::llvm::SmallVector<::fabric::FuOp> fus;
    module.walk([&](::fabric::FuOp fu) { fus.push_back(fu); });

    ::llvm::SmallVector<::dataflow::GraphOp> graphs;
    module.walk([&](::dataflow::GraphOp g) { graphs.push_back(g); });

    auto lib = ::fabric::TemplateLibrary::build(&getContext(), fus);
    auto partitioner = ::fabric::createPartitioner(cfg.algorithm);

    ::mlir::OpBuilder builder(&getContext());
    for (::dataflow::GraphOp g : graphs) {
      auto partition = partitioner->run(g, *lib, cfg);
      ::fabric::applyPartition(g, partition, builder);
    }
  }
};

} // namespace

namespace fabric {

std::unique_ptr<::mlir::Pass> createPartitionGraphPass(std::string configPath) {
  return std::make_unique<PartitionGraphPass>(std::move(configPath));
}

} // namespace fabric
