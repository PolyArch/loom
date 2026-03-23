#include "loom/TDG/TDGBuilder.h"
#include "loom/Dialect/TDG/TDGDialect.h"
#include "loom/Dialect/TDG/TDGOps.h"
#include "loom/Dialect/TDG/TDGTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <fstream>

using namespace mlir;
using namespace loom;
using namespace loom::tdg;

TaskDataflowGraph::TaskDataflowGraph(const std::string &name)
    : graphName_(name) {}

KernelHandle TaskDataflowGraph::kernel(const std::string &name,
                                       const std::string &kernelType) {
  KernelDesc desc;
  desc.name = name;
  desc.kernelType = kernelType;
  kernels_.push_back(std::move(desc));
  return KernelHandle{static_cast<unsigned>(kernels_.size() - 1)};
}

KernelHandle TaskDataflowGraph::kernel(
    const std::string &name, const std::string &kernelType,
    std::function<void(KernelBodyBuilder &)> body) {
  KernelDesc desc;
  desc.name = name;
  desc.kernelType = kernelType;
  desc.bodyBuilder = std::move(body);
  kernels_.push_back(std::move(desc));
  return KernelHandle{static_cast<unsigned>(kernels_.size() - 1)};
}

ContractHandle TaskDataflowGraph::connect(KernelHandle producer,
                                          KernelHandle consumer,
                                          Ordering ordering) {
  assert(producer.index < kernels_.size() && "invalid producer handle");
  assert(consumer.index < kernels_.size() && "invalid consumer handle");
  ContractDesc desc;
  desc.producerIdx = producer.index;
  desc.consumerIdx = consumer.index;
  desc.ordering = ordering;
  contracts_.push_back(std::move(desc));
  return ContractHandle{static_cast<unsigned>(contracts_.size() - 1)};
}

void TaskDataflowGraph::setTileShape(ContractHandle c,
                                     std::vector<int64_t> shape) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].tileShape = std::move(shape);
}

void TaskDataflowGraph::setVisibility(ContractHandle c, Visibility vis) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].visibility = vis;
}

void TaskDataflowGraph::setDoubleBuffering(ContractHandle c, bool enable) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].doubleBuffering = enable;
}

void TaskDataflowGraph::setDataType(ContractHandle c,
                                    const std::string &typeName) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].dataTypeName = typeName;
}

void TaskDataflowGraph::setBackpressure(ContractHandle c, Backpressure bp) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].backpressure = bp;
}

void TaskDataflowGraph::setBufferElements(ContractHandle c, int64_t minElems,
                                          int64_t maxElems) {
  assert(c.index < contracts_.size() && "invalid contract handle");
  contracts_[c.index].minBufferElements = minElems;
  contracts_[c.index].maxBufferElements = maxElems;
}

OwningOpRef<ModuleOp> TaskDataflowGraph::buildMLIR(MLIRContext &ctx) {
  ctx.getOrLoadDialect<TDGDialect>();

  auto loc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(module->getBody());

  // Create tdg.graph
  auto graphOp = builder.create<GraphOp>(
      loc,
      /*sym_name=*/builder.getStringAttr(graphName_),
      /*target_arch=*/StringAttr());

  // Create the single block in the graph region
  Block *graphBlock = new Block();
  graphOp.getBody().push_back(graphBlock);

  builder.setInsertionPointToEnd(graphBlock);

  // Create tdg.kernel ops
  for (const auto &kd : kernels_) {
    auto kernelOp = builder.create<KernelOp>(
        loc,
        /*sym_name=*/builder.getStringAttr(kd.name),
        /*kernel_type=*/builder.getStringAttr(kd.kernelType),
        /*tile_hint=*/DenseI64ArrayAttr(),
        /*target_core_type=*/StringAttr());

    // Create the kernel body region
    Block *kernelBlock = new Block();
    kernelOp.getBody().push_back(kernelBlock);

    if (kd.bodyBuilder) {
      OpBuilder bodyBuilder(&ctx);
      bodyBuilder.setInsertionPointToEnd(kernelBlock);
      KernelBodyBuilder kbb(bodyBuilder);
      kd.bodyBuilder(kbb);
    }
  }

  // Create tdg.contract ops
  for (const auto &cd : contracts_) {
    // Resolve the data type
    Type dataType;
    if (cd.dataTypeName == "f32")
      dataType = builder.getF32Type();
    else if (cd.dataTypeName == "f64")
      dataType = builder.getF64Type();
    else if (cd.dataTypeName == "i32")
      dataType = builder.getI32Type();
    else if (cd.dataTypeName == "i64")
      dataType = builder.getI64Type();
    else if (cd.dataTypeName == "i16")
      dataType = builder.getI16Type();
    else if (cd.dataTypeName == "i8")
      dataType = builder.getIntegerType(8);
    else if (cd.dataTypeName == "f16")
      dataType = builder.getF16Type();
    else
      dataType = builder.getF32Type(); // fallback

    DenseI64ArrayAttr tileShapeAttr;
    if (!cd.tileShape.empty())
      tileShapeAttr = builder.getDenseI64ArrayAttr(cd.tileShape);

    builder.create<ContractOp>(
        loc,
        /*producer=*/kernels_[cd.producerIdx].name,
        /*consumer=*/kernels_[cd.consumerIdx].name,
        /*ordering=*/StringRef(orderingToString(cd.ordering)),
        /*data_type=*/dataType,
        /*production_rate=*/AffineMapAttr(),
        /*consumption_rate=*/AffineMapAttr(),
        /*steady_state_ratio=*/DenseI64ArrayAttr(),
        /*tile_shape=*/tileShapeAttr,
        /*min_buffer_elements=*/
        cd.minBufferElements > 0
            ? builder.getI64IntegerAttr(cd.minBufferElements)
            : IntegerAttr(),
        /*max_buffer_elements=*/
        cd.maxBufferElements > 0
            ? builder.getI64IntegerAttr(cd.maxBufferElements)
            : IntegerAttr(),
        /*backpressure=*/StringRef(backpressureToString(cd.backpressure)),
        /*double_buffering=*/cd.doubleBuffering,
        /*visibility=*/StringRef(visibilityToString(cd.visibility)),
        /*producer_writeback=*/StringRef("EAGER"),
        /*consumer_prefetch=*/StringRef("NONE"),
        /*may_fuse=*/true,
        /*may_replicate=*/true,
        /*may_pipeline=*/true,
        /*may_reorder=*/false,
        /*may_retile=*/true);
  }

  return module;
}

bool TaskDataflowGraph::exportMLIR(MLIRContext &ctx, const std::string &path) {
  auto module = buildMLIR(ctx);
  if (!module)
    return false;

  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec);
  if (ec)
    return false;

  module->print(output);
  return true;
}
