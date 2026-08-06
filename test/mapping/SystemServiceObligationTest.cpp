#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System service obligation anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail("adverse input unexpectedly succeeded");
  std::string diagnostic = llvm::toString(value.takeError());
  require(llvm::StringRef(diagnostic).contains(expected),
          "adverse diagnostic did not contain '" + expected +
              "': " + diagnostic);
}

bool projectionsEqual(
    llvm::ArrayRef<loom::mapping::SystemServiceObligationProjection> lhs,
    llvm::ArrayRef<loom::mapping::SystemServiceObligationProjection> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (left.key != right.key || left.members != right.members ||
        left.sinks != right.sinks || left.exposures != right.exposures ||
        left.legs != right.legs)
      return false;
  return true;
}

unsigned schemaLegCount(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<dataflow::CanonicalProducerTerminalView> producers,
    const loom::mapping::SystemServiceObligationProjection &obligation,
    const dataflow::ServiceMemberRef &member) {
  if (std::holds_alternative<dataflow::MessageTransferMemberRef>(member)) {
    const auto *producer =
        std::get_if<dataflow::CanonicalProducerTerminalRef>(&obligation.key);
    require(producer != nullptr,
            "a message-transfer member must belong to a transfer obligation");
    auto found = llvm::find_if(producers, [&](const auto &view) {
      return view.terminal == *producer;
    });
    require(found != producers.end(),
            "a transfer obligation has no Dataflow producer projection");
    return take(dataflow::semantics::CanonicalService::messageTransfer(
                    found->payloadType))
        .legCount();
  }

  dataflow::ContextualActorRef contextual =
      std::holds_alternative<dataflow::AddressedMemoryActorMemberRef>(member)
          ? std::get<dataflow::AddressedMemoryActorMemberRef>(member).actor
          : std::get<dataflow::FenceActorMemberRef>(member).actor;
  auto actor = take(dataflow.resolve(contextual.actor));
  return take(dataflow::semantics::CanonicalService::forActor(actor.op))
      .legCount();
}

void requireLegTotality(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<loom::mapping::SystemServiceObligationProjection>
        obligations) {
  std::vector<dataflow::CanonicalProducerTerminalView> producers;
  for (const dataflow::RootThreadLaunchRef &root : roots)
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root, [&](const dataflow::CanonicalProducerTerminalView &view) {
              producers.push_back(view);
              return llvm::Error::success();
            }))
      fail(llvm::toString(std::move(error)));

  for (const auto &obligation : obligations) {
    std::vector<std::vector<std::uint8_t>> encodedLegs;
    encodedLegs.reserve(obligation.legs.size());
    for (const loom::mapping::CanonicalServiceLegKey &leg : obligation.legs) {
      require(leg.obligation == obligation.key,
              "a service leg points at a different obligation");
      require(llvm::is_contained(obligation.members, leg.member),
              "a service leg names a member outside its obligation");
      encodedLegs.push_back(take(loom::mapping::encodeCanonicalServiceLegKey(
          dataflow.identity(), leg)));
    }
    for (std::size_t index = 1; index < encodedLegs.size(); ++index)
      require(encodedLegs[index - 1] < encodedLegs[index],
              "service legs are not canonical sorted and unique");

    for (const dataflow::ServiceMemberRef &member : obligation.members) {
      std::vector<dataflow::StructuralOrdinal> ordinals;
      for (const loom::mapping::CanonicalServiceLegKey &leg : obligation.legs)
        if (leg.member == member)
          ordinals.push_back(leg.ordinal);
      const unsigned expected =
          schemaLegCount(dataflow, producers, obligation, member);
      require(ordinals.size() == expected,
              "a service member has a missing or extra schema-local leg");
      for (unsigned ordinal = 0; ordinal < expected; ++ordinal)
        require(ordinals[ordinal] == ordinal,
                "a service member does not own the exact schema ordinal set");
    }
  }
}

dataflow::CanonicalDataflowProgramView buildView(llvm::StringRef program) {
  static std::vector<std::unique_ptr<mlir::MLIRContext>> contexts;
  static std::vector<dataflow::CanonicalDataflowArtifact> artifacts;
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(program, context.get());
  require(static_cast<bool>(module), "fixture did not parse");
  dataflow::CanonicalDataflowArtifact artifact =
      take(dataflow::finalizeCanonicalDataflow(*module));
  contexts.push_back(std::move(context));
  artifacts.push_back(std::move(artifact));
  return take(artifacts.back().view());
}

dataflow::CanonicalDataflowProgramView buildServiceView() {
  return buildView(R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @g(%ctrl: none, %x: i32,
                            %mem: memref<8xi32>)
      -> (i32, memref<8xi32>)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 1>} {
    %index = arith.constant 0 : index
    %value, %loaded = dataflow.load %mem[%index] %ctrl : memref<8xi32>
    %done = dataflow.fence %loaded
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values(%value : i32) streams()
        memories(%mem : memref<8xi32>) complete(%done : none)
  }
  dataflow.thread private @t domain(#dataflow.thread_domain<dense>)(
      %x: i32, %mem: memref<8xi32>) ctrl (%ctrl: none) {
    %value, %memory, %done = dataflow.graph.launch @g deps(%ctrl)
        values(%x) stream_inputs() memories(%mem) stream_outputs()
        : (none, i32, memref<8xi32>) -> (i32, memref<8xi32>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%x: i32, %mem: memref<8xi32>) {
    %first = dataflow.thread.launch @t(%x, %mem)
        : (i32, memref<8xi32>) -> !dataflow.thread_token
    %second = dataflow.thread.launch @t(%x, %mem)
        : (i32, memref<8xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir");
}

dataflow::CanonicalDataflowProgramView buildMulticastView() {
  return buildView(R"mlir(
module {
  dataflow.graph private @consumer(%start: none, %input: i32) -> ()
      attributes {input_segments = array<i32: 0, 1, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    dataflow.graph.return %start : none
  }
  dataflow.thread private @producer
      domain(#dataflow.thread_domain<dense>)(%channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) {
    %value = arith.constant 7 : i32
    dataflow.channel.send %channel, %value : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  dataflow.thread private @stream_consumer
      domain(#dataflow.thread_domain<dense>)(%channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @consumer deps(%ctrl) values()
        stream_inputs(%channel source_map affine_map<() -> ()>) memories()
        stream_outputs() : (none, !dataflow.channel<i32>) -> none
    dataflow.thread.yield %done : none
  }
  dataflow.thread private @direct_consumer
      domain(#dataflow.thread_domain<dense>)(%channel: !dataflow.channel<i32>)
      ctrl (%ctrl: none) {
    %value = dataflow.channel.receive %channel : !dataflow.channel<i32>
    dataflow.thread.yield
  }
  func.func private @host(%channel: !dataflow.channel<i32>) {
    %producer = dataflow.thread.launch @producer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %stream = dataflow.thread.launch @stream_consumer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    %direct = dataflow.thread.launch @direct_consumer(%channel)
        : (!dataflow.channel<i32>) -> !dataflow.thread_token
    return
  }
}
)mlir");
}

void exactServiceClosure() {
  dataflow::CanonicalDataflowProgramView dataflow = buildServiceView();
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  require(roots.size() == 2, "fixture must have two root launches");

  std::vector<loom::mapping::SystemServiceObligationProjection> obligations =
      take(loom::mapping::projectSystemServiceObligations(dataflow, roots));
  require(obligations.size() == 16,
          "two roots must derive fourteen transfers, one memory obligation, "
          "and one fence obligation");

  unsigned transferCount = 0;
  unsigned memoryCount = 0;
  unsigned fenceCount = 0;
  unsigned legCount = 0;
  for (const auto &obligation : obligations) {
    legCount += obligation.legs.size();
    if (std::holds_alternative<dataflow::CanonicalProducerTerminalRef>(
            obligation.key)) {
      ++transferCount;
      require(obligation.members.size() == 1 && obligation.sinks.size() >= 1 &&
                  obligation.exposures.empty() && obligation.legs.size() == 1,
              "one transfer obligation must own one member, its sinks, and "
              "one service leg");
      continue;
    }
    const auto &operation =
        std::get<loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key);
    if (std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(
            operation)) {
      ++memoryCount;
      require(obligation.members.size() == 2 &&
                  obligation.exposures.size() == 2 &&
                  obligation.legs.size() == 4,
              "one logical memory must aggregate both rooted actors and "
              "exposures without turning exposures into service members");
    } else {
      ++fenceCount;
      require(obligation.members.size() == 2 && obligation.exposures.empty() &&
                  obligation.legs.size() == 4,
              "one fence family must aggregate both rooted contexts");
    }
  }
  require(transferCount == 14 && memoryCount == 1 && fenceCount == 1 &&
              legCount == 22,
          "derived service closure has the wrong typed family counts");
  requireLegTotality(dataflow, roots, obligations);

  std::vector<dataflow::RootThreadLaunchRef> alternateRoots{roots[1], roots[0],
                                                            roots[1]};
  require(projectionsEqual(take(loom::mapping::projectSystemServiceObligations(
                               dataflow, alternateRoots)),
                           obligations),
          "alternate root authoring order and duplicates did not converge");

  for (std::size_t index = 1; index < obligations.size(); ++index)
    require(take(loom::mapping::encodeSystemServiceObligationKey(
                dataflow.identity(), obligations[index - 1].key)) <
                take(loom::mapping::encodeSystemServiceObligationKey(
                    dataflow.identity(), obligations[index].key)),
            "obligation projection is not in canonical key order");

  for (const auto &obligation : obligations) {
    std::vector<std::uint8_t> encoded =
        take(loom::mapping::encodeSystemServiceObligationKey(
            dataflow.identity(), obligation.key));
    auto decoded = loom::mapping::decodeSystemServiceObligationKey(
        encoded, dataflow.identity());
    require(decoded && *decoded == obligation.key,
            "obligation key did not round trip byte-exactly");
    for (const loom::mapping::CanonicalServiceLegKey &leg : obligation.legs) {
      std::vector<std::uint8_t> legBytes =
          take(loom::mapping::encodeCanonicalServiceLegKey(dataflow.identity(),
                                                           leg));
      auto decodedLeg = loom::mapping::decodeCanonicalServiceLegKey(
          legBytes, dataflow.identity());
      require(decodedLeg && *decodedLeg == leg,
              "service leg key did not round trip byte-exactly");
    }
  }

  std::vector<std::uint8_t> trailing =
      take(loom::mapping::encodeSystemServiceObligationKey(dataflow.identity(),
                                                           obligations[0].key));
  trailing.push_back(0);
  requireFailureContains(loom::mapping::decodeSystemServiceObligationKey(
                             trailing, dataflow.identity()),
                         "trailing service-obligation bytes");
  requireFailureContains(loom::mapping::decodeSystemServiceObligationKey(
                             std::vector<std::uint8_t>{0xff, 0xff, 0xff, 0xff},
                             dataflow.identity()),
                         "unknown service-obligation kind");

  loom::ArtifactIdentity::Storage foreignBytes = dataflow.identity().bytes();
  foreignBytes[0] ^= 0xff;
  loom::ArtifactIdentity foreignIdentity =
      take(loom::ArtifactIdentity::fromBytes(foreignBytes));
  dataflow::RootThreadLaunchRef foreignRoot{foreignIdentity,
                                            roots.front().entity};
  requireFailureContains(
      loom::mapping::projectSystemServiceObligations(dataflow, {foreignRoot}),
      "nested Canonical Dataflow references bind different artifacts");
}

void channelMulticastClosure() {
  dataflow::CanonicalDataflowProgramView dataflow = buildMulticastView();
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  require(roots.size() == 3, "multicast fixture must have three root launches");

  auto obligations =
      take(loom::mapping::projectSystemServiceObligations(dataflow, roots));
  requireLegTotality(dataflow, roots, obligations);
  unsigned channelObligations = 0;
  for (const auto &obligation : obligations) {
    const auto *transfer =
        std::get_if<dataflow::CanonicalProducerTerminalRef>(&obligation.key);
    if (!transfer ||
        !std::holds_alternative<dataflow::ChannelProducerTerminalRef>(
            *transfer))
      continue;
    ++channelObligations;
    require(obligation.members.size() == 1 && obligation.sinks.size() == 2 &&
                obligation.legs.size() == 1,
            "one channel producer must retain its complete multicast sink set");
  }
  require(channelObligations == 1,
          "multicast fixture must derive exactly one channel obligation");

  requireFailureContains(
      loom::mapping::projectSystemServiceObligations(dataflow, {}),
      "root thread launch scope is empty");
}

} // namespace

int main() {
  exactServiceClosure();
  channelMulticastClosure();
  return EXIT_SUCCESS;
}
