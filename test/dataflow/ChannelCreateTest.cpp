#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <variant>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "ChannelCreateTest: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

dataflow::CanonicalDataflowArtifact finalize(llvm::StringRef program) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(program, &context());
  if (!module)
    fail("failed to parse fixture");
  return take(dataflow::finalizeCanonicalDataflow(module.get()));
}

bool rejects(llvm::StringRef program) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(program, &context());
  if (!module)
    fail("failed to parse rejection fixture");
  auto artifact = dataflow::finalizeCanonicalDataflow(module.get());
  if (artifact)
    return false;
  llvm::consumeError(artifact.takeError());
  return true;
}

const char *hostCreatedProgram(bool reverseDefinitions = false) {
  static constexpr const char *forward = R"mlir(
module {
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %message = arith.constant 11 : i32
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host() {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %producer = dataflow.thread.launch @producer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir";
  static constexpr const char *reversed = R"mlir(
module {
  func.func private @host() {
    %wire = dataflow.channel.create : !dataflow.channel<i32>
    %send = dataflow.thread.launch @producer(%wire)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %receive = dataflow.thread.launch @consumer(%wire)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %wire: !dataflow.channel<i32>) ctrl (%start: none) {
    %value = dataflow.channel.receive %wire : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %wire: !dataflow.channel<i32>) ctrl (%start: none) {
    %value = arith.constant 11 : i32
    dataflow.channel.send %wire, %value : !dataflow.channel<i32>
    dataflow.thread.yield
  }
}
)mlir";
  return reverseDefinitions ? reversed : forward;
}

const char *threadLocalProgram() {
  return R"mlir(
module {
  dataflow.thread private @relay domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %message = arith.constant 23 : i32
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    %received = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host() {
    %first = dataflow.thread.launch @relay()
        : () -> !dataflow.thread_token
    %second = dataflow.thread.launch @relay()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir";
}

dataflow::RootThreadLaunchRef
sendingRoot(const dataflow::CanonicalDataflowProgramView &view) {
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       view.rootThreadLaunches()) {
    bool sends = false;
    root.callee->walk([&](dataflow::ChannelSendOp) { sends = true; });
    if (sends)
      return root.ref;
  }
  fail("host-created fixture has no sending root");
}

void hostCreationAndCanonicalRoundTrip() {
  auto artifact = finalize(hostCreatedProgram());
  auto view = take(artifact.view());
  dataflow::ChannelProducerRef producer{
      dataflow::ThreadChannelSendSiteRef{sendingRoot(view), 0}};
  auto consumers = take(view.channelConsumers(producer));
  require(consumers.size() == 1 &&
              std::holds_alternative<dataflow::ThreadChannelReceiveSiteRef>(
                  consumers.front().consumer),
          "host-created channel did not derive one direct consumer");

  auto imported = take(dataflow::importCanonicalDataflow(
      artifact.identity(), artifact.canonicalBytes()));
  unsigned creates = 0;
  imported.module().walk([&](dataflow::ChannelCreateOp) { ++creates; });
  require(creates == 1, "canonical round-trip lost channel.create");

  auto reordered = finalize(hostCreatedProgram(true));
  require(artifact.canonicalBytes().bytes().equals(
              reordered.canonicalBytes().bytes()),
          "canonical bytes depend on symbol or SSA authoring order");
}

void threadLocalCreationIsRootContextual() {
  auto artifact = finalize(threadLocalProgram());
  auto view = take(artifact.view());
  require(view.rootThreadLaunches().size() == 2,
          "thread-local fixture lost a root launch");
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       view.rootThreadLaunches()) {
    dataflow::ChannelProducerRef producer{
        dataflow::ThreadChannelSendSiteRef{root.ref, 0}};
    auto consumers = take(view.channelConsumers(producer));
    require(consumers.size() == 1,
            "thread-local channel did not derive one consumer per root");
    const auto *receive = std::get_if<dataflow::ThreadChannelReceiveSiteRef>(
        &consumers.front().consumer);
    require(receive && receive->launch == root.ref,
            "thread-local channel relation crossed root-launch contexts");
  }
}

void invalidUseSurfacesAreRejected() {
  require(rejects(R"mlir(
module {
  func.func private @escape(!dataflow.channel<i32>)
  func.func @host() {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    func.call @escape(%channel) : (!dataflow.channel<i32>) -> ()
    return
  }
}
)mlir"),
          "a channel.create result escaped through func.call");

  require(rejects(R"mlir(
module {
  dataflow.thread private @producer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %message = arith.constant 1 : i32
    dataflow.channel.send %channel, %message : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @consumer domain(#dataflow.thread_domain<dense>)(
      %channel: !dataflow.channel<i32>) ctrl (%ctrl: none) {
    %message = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host() {
    %channel = dataflow.channel.create : !dataflow.channel<i32>
    %first = dataflow.thread.launch @producer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %second = dataflow.thread.launch @producer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %consumer = dataflow.thread.launch @consumer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir"),
          "a host-created channel admitted two producer bindings");
}

} // namespace

int main() {
  hostCreationAndCanonicalRoundTrip();
  threadLocalCreationIsRootContextual();
  invalidUseSurfacesAreRejected();
  llvm::outs() << "channel create workflow anchors passed\n";
  return EXIT_SUCCESS;
}
