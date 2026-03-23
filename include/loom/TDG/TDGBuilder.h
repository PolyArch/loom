#ifndef LOOM_TDG_TDGBUILDER_H
#define LOOM_TDG_TDGBUILDER_H

#include "loom/SystemCompiler/Contract.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace tdg {
class GraphOp;
class KernelOp;
class ContractOp;
} // namespace tdg
} // namespace loom

namespace loom {

/// Opaque handle to a kernel in the builder.
struct KernelHandle {
  unsigned index;
};

/// Opaque handle to a contract in the builder.
struct ContractHandle {
  unsigned index;
};

/// Builder for constructing kernel computation bodies.
class KernelBodyBuilder {
public:
  explicit KernelBodyBuilder(mlir::OpBuilder &builder) : builder_(builder) {}
  mlir::OpBuilder &getBuilder() { return builder_; }

private:
  mlir::OpBuilder &builder_;
};

/// C++ DSL for programmatically constructing a Task Dataflow Graph.
///
/// Wraps TDG MLIR dialect construction with a fluent builder API.
/// The typical usage is:
///   1. Create a TaskDataflowGraph with a name
///   2. Add kernels via kernel()
///   3. Connect kernels via connect()
///   4. Optionally refine contracts (tile shape, visibility, etc.)
///   5. Export to MLIR via buildMLIR() or exportMLIR()
class TaskDataflowGraph {
public:
  explicit TaskDataflowGraph(const std::string &name);

  /// Add a kernel from a source file reference.
  KernelHandle kernel(const std::string &name,
                      const std::string &kernelType);

  /// Add a kernel with a programmatic body.
  KernelHandle kernel(const std::string &name,
                      const std::string &kernelType,
                      std::function<void(KernelBodyBuilder &)> body);

  /// Connect a producer to a consumer with a contract edge.
  ContractHandle connect(KernelHandle producer, KernelHandle consumer,
                         Ordering ordering = Ordering::FIFO);

  /// Set the tile shape for a contract.
  void setTileShape(ContractHandle c, std::vector<int64_t> shape);

  /// Set the memory visibility for a contract.
  void setVisibility(ContractHandle c, Visibility vis);

  /// Set double-buffering for a contract.
  void setDoubleBuffering(ContractHandle c, bool enable);

  /// Set the data type name for a contract.
  void setDataType(ContractHandle c, const std::string &typeName);

  /// Set backpressure mode for a contract.
  void setBackpressure(ContractHandle c, Backpressure bp);

  /// Set buffer size constraints for a contract.
  void setBufferElements(ContractHandle c, int64_t minElems, int64_t maxElems);

  /// Build the MLIR module containing the TDG graph.
  mlir::OwningOpRef<mlir::ModuleOp> buildMLIR(mlir::MLIRContext &ctx);

  /// Export the TDG as an MLIR text file.
  bool exportMLIR(mlir::MLIRContext &ctx, const std::string &path);

private:
  struct KernelDesc {
    std::string name;
    std::string kernelType;
    std::function<void(KernelBodyBuilder &)> bodyBuilder;
  };

  struct ContractDesc {
    unsigned producerIdx;
    unsigned consumerIdx;
    Ordering ordering = Ordering::FIFO;
    std::string dataTypeName = "f32";
    std::vector<int64_t> tileShape;
    Visibility visibility = Visibility::LOCAL_SPM;
    Backpressure backpressure = Backpressure::BLOCK;
    bool doubleBuffering = false;
    int64_t minBufferElements = 0;
    int64_t maxBufferElements = 0;
  };

  std::string graphName_;
  std::vector<KernelDesc> kernels_;
  std::vector<ContractDesc> contracts_;
};

} // namespace loom

#endif
