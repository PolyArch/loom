#include "PnR/System/SystemPnrSearchDomain.h"

#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System PnR service-domain anchor failed: " << message
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
void requireUnsupported(
    llvm::Expected<T> value,
    loom::pnr::UnsupportedSystemPnrSearchDomainReason expectedReason,
    llvm::StringRef expectedDiagnostic, const llvm::Twine &message) {
  if (value)
    fail(message);
  bool matched = false;
  llvm::Error remaining = llvm::handleErrors(
      value.takeError(),
      [&](const loom::pnr::UnsupportedSystemPnrSearchDomain &error) {
        matched = true;
        require(error.reason() == expectedReason,
                "unsupported search-domain reason changed");
        std::string diagnostic;
        llvm::raw_string_ostream stream(diagnostic);
        error.log(stream);
        stream.flush();
        require(llvm::StringRef(diagnostic).contains(expectedDiagnostic),
                "unsupported search-domain diagnostic changed");
      });
  if (remaining)
    fail(llvm::toString(std::move(remaining)));
  require(matched, message);
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef expectedDiagnostic,
                            const llvm::Twine &message) {
  if (value)
    fail(message);
  const std::string diagnostic = llvm::toString(value.takeError());
  if (!llvm::StringRef(diagnostic).contains(expectedDiagnostic))
    fail(message + ": " + diagnostic);
}

struct FramedRange final {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct FramedServiceEntry final {
  FramedRange bytes;
  std::size_t targetCountOffset = 0;
  std::uint64_t targetCount = 0;
  std::vector<FramedRange> targets;
  std::size_t terminalCountOffset = 0;
  std::uint64_t terminalCount = 0;
  std::vector<FramedRange> terminals;
};

struct FramedServiceSection final {
  std::size_t countOffset = 0;
  std::uint64_t count = 0;
  std::vector<FramedServiceEntry> entries;
};

class FramingCursor final {
public:
  explicit FramingCursor(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  std::size_t position() const { return offset_; }
  bool atEnd() const { return offset_ == bytes_.size(); }

  std::uint32_t u32() {
    require(bytes_.size() - offset_ >= 4, "truncated u32 test framing");
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
      value = (value << 8) | bytes_[offset_ + index];
    offset_ += 4;
    return value;
  }

  std::uint64_t u64() {
    require(bytes_.size() - offset_ >= 8, "truncated u64 test framing");
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index)
      value = (value << 8) | bytes_[offset_ + index];
    offset_ += 8;
    return value;
  }

  void skip(std::uint64_t size) {
    require(size <= bytes_.size() - offset_, "truncated test framing");
    offset_ += static_cast<std::size_t>(size);
  }

  void skipSizedBytes() { skip(u64()); }

  void skipRootReference() {
    auto decoded = take(
        loom::decodeArtifactRootReferencePrefix(bytes_.drop_front(offset_)));
    skip(decoded.byteCount);
  }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

void skipFabricDomain(FramingCursor &cursor) {
  const std::uint64_t count = cursor.u64();
  for (std::uint64_t index = 0; index < count; ++index)
    cursor.skipSizedBytes();
}

void skipPresburgerCell(FramingCursor &cursor) {
  const std::uint64_t rowWidth =
      static_cast<std::uint64_t>(cursor.u32()) + cursor.u32() + 1;
  require(rowWidth <= std::numeric_limits<std::size_t>::max() / 8,
          "test Presburger row width exceeds native range");
  const std::uint64_t rowBytes = rowWidth * 8;
  const auto skipRows = [&](std::uint64_t count) {
    require(rowBytes == 0 ||
                count <= std::numeric_limits<std::uint64_t>::max() / rowBytes,
            "test Presburger table exceeds u64 range");
    cursor.skip(count * rowBytes);
  };
  skipRows(cursor.u64());
  skipRows(cursor.u64());
}

void skipAtomDomain(FramingCursor &cursor) {
  require(cursor.u32() <= 2, "test search-atom domain kind is not canonical");
  cursor.skipSizedBytes();
}

FramedServiceSection locateServiceSection(llvm::ArrayRef<std::uint8_t> bytes) {
  FramingCursor cursor(bytes);
  cursor.skipRootReference();
  cursor.skipRootReference();
  cursor.skipRootReference();

  const std::uint64_t rootCount = cursor.u64();
  for (std::uint64_t index = 0; index < rootCount; ++index)
    cursor.skipSizedBytes();

  const std::uint64_t bindingCount = cursor.u64();
  for (std::uint64_t binding = 0; binding < bindingCount; ++binding) {
    cursor.u32();
    cursor.skipSizedBytes();
    const std::uint64_t atomCount = cursor.u64();
    for (std::uint64_t atom = 0; atom < atomCount; ++atom) {
      skipPresburgerCell(cursor);
      skipAtomDomain(cursor);
    }
  }

  FramedServiceSection section;
  section.countOffset = cursor.position();
  section.count = cursor.u64();
  section.entries.reserve(section.count);
  for (std::uint64_t service = 0; service < section.count; ++service) {
    FramedServiceEntry entry;
    entry.bytes.begin = cursor.position();
    cursor.skipSizedBytes();
    entry.targetCountOffset = cursor.position();
    entry.targetCount = cursor.u64();
    entry.targets.reserve(entry.targetCount);
    for (std::uint64_t target = 0; target < entry.targetCount; ++target) {
      FramedRange range{cursor.position(), 0};
      cursor.skipSizedBytes();
      cursor.skipSizedBytes();
      require(cursor.u32() <= 1,
              "test target-compatibility kind is not canonical");
      cursor.skipSizedBytes();
      range.end = cursor.position();
      entry.targets.push_back(range);
    }
    entry.terminalCountOffset = cursor.position();
    entry.terminalCount = cursor.u64();
    entry.terminals.reserve(entry.terminalCount);
    for (std::uint64_t terminal = 0; terminal < entry.terminalCount;
         ++terminal) {
      FramedRange range{cursor.position(), 0};
      const std::uint32_t kind = cursor.u32();
      require(kind <= 1, "test terminal discriminant is not canonical");
      cursor.skipSizedBytes();
      if (kind == 1)
        cursor.u64();
      require(cursor.u32() <= 1,
              "test bound-terminal endpoint kind is not canonical");
      cursor.skipSizedBytes();
      skipFabricDomain(cursor);
      range.end = cursor.position();
      entry.terminals.push_back(range);
    }
    entry.bytes.end = cursor.position();
    section.entries.push_back(std::move(entry));
  }
  require(cursor.atEnd(), "test service framing left trailing bytes");
  return section;
}

void overwriteU64(std::vector<std::uint8_t> &bytes, std::size_t offset,
                  std::uint64_t value) {
  require(offset <= bytes.size() && bytes.size() - offset >= 8,
          "test u64 replacement is out of bounds");
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes[offset++] = static_cast<std::uint8_t>(value >> shift);
}

std::vector<std::uint8_t>
duplicateFramedEntry(llvm::ArrayRef<std::uint8_t> bytes,
                     std::size_t countOffset, std::uint64_t count,
                     FramedRange entry) {
  std::vector<std::uint8_t> copy(bytes.begin() + entry.begin,
                                 bytes.begin() + entry.end);
  std::vector<std::uint8_t> result(bytes.begin(), bytes.end());
  result.insert(result.begin() + entry.end, copy.begin(), copy.end());
  overwriteU64(result, countOffset, count + 1);
  return result;
}

std::vector<std::uint8_t> removeFramedEntry(llvm::ArrayRef<std::uint8_t> bytes,
                                            std::size_t countOffset,
                                            std::uint64_t count,
                                            FramedRange entry) {
  require(count != 0 && entry.begin <= entry.end && entry.end <= bytes.size(),
          "test entry removal is out of bounds");
  std::vector<std::uint8_t> result(bytes.begin(), bytes.end());
  result.erase(result.begin() + entry.begin, result.begin() + entry.end);
  overwriteU64(result, countOffset, count - 1);
  return result;
}

std::vector<std::uint8_t>
swapAdjacentFramedEntries(llvm::ArrayRef<std::uint8_t> bytes, FramedRange first,
                          FramedRange second) {
  require(first.end == second.begin,
          "test entries are not adjacent in canonical framing");
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  result.insert(result.end(), bytes.begin(), bytes.begin() + first.begin);
  result.insert(result.end(), bytes.begin() + second.begin,
                bytes.begin() + second.end);
  result.insert(result.end(), bytes.begin() + first.begin,
                bytes.begin() + first.end);
  result.insert(result.end(), bytes.begin() + second.end, bytes.end());
  return result;
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-pnr-service-domain", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

dataflow::CanonicalDataflowArtifact
buildMemoryDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load(%ctrl: none, %memory: memref<8xi32>)
      -> (i32, memref<8xi32>)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 0, 1>} {
    %index = arith.constant 0 : index
    %value, %done = dataflow.load %memory[%index] %ctrl : memref<8xi32>
    %fenced = dataflow.fence %done
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    dataflow.graph.return values(%value : i32) streams()
        memories(%memory : memref<8xi32>) complete(%fenced : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %memory: memref<8xi32>) ctrl (%ctrl: none) {
    %value, %result, %done = dataflow.graph.launch @load deps(%ctrl)
        values() stream_inputs() memories(%memory) stream_outputs()
        : (none, memref<8xi32>) -> (i32, memref<8xi32>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%memory: memref<8xi32>) {
    %completion = dataflow.thread.launch @worker(%memory)
        : (memref<8xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse memory Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildMessageDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  dataflow.graph private @produce(%start: none) -> i32
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %constant = dataflow.constant %start {const_value = 7 : i32} : i32
    %published:2 = dataflow.sync %start, %constant
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
  dataflow.graph private @consume(%start: none, %value: i32) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %consumed:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values() streams() memories()
        complete(%consumed#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)()
      ctrl (%ctrl: none) {
    %value, %produced = dataflow.graph.launch @produce deps(%ctrl) values()
        stream_inputs() memories() stream_outputs()
        : (none) -> (i32, none)
    %consumed = dataflow.graph.launch @consume deps(%produced) values(%value)
        stream_inputs() memories() stream_outputs()
        : (none, i32) -> none
    dataflow.thread.yield %consumed : none
  }
  func.func private @host() {
    %completion = dataflow.thread.launch @worker()
        : () -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse message Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::adg::FinalizedFabricDesign
buildTransformFabric(const loom::ArtifactStore &store,
                     const loom::fabric::FinalizedFabricRoot &base,
                     mlir::MLIRContext &context) {
  auto baseSystem = take(loom::fabric::requireSystemRoot(base.view()));
  const auto baseEndpoint =
      baseSystem.artifact().systemServiceEndpoints().front();
  const auto *baseCapabilities =
      baseSystem.serviceEndpointCapabilities(baseEndpoint);
  require(baseCapabilities, "builtin memory endpoint has no capabilities");
  const auto capability = llvm::find_if(
      baseCapabilities->capabilities(), [](const auto &candidate) {
        return candidate.kind() == dataflow::semantics::ServiceKind::MemoryRead;
      });
  require(capability != baseCapabilities->capabilities().end(),
          "builtin memory endpoint has no read capability");
  const auto *addressed =
      std::get_if<loom::fabric::AddressedMemoryCapabilityDomain>(
          &capability->domain());
  require(addressed, "builtin read capability is not addressed memory");

  auto module = take(loom::fabric::importEntireFabricRoot(
      base.directDependencies().front().root, store));
  loom::adg::DesignBuilder design(store);
  auto system = take(loom::adg::expandBuiltinSystem(
      design, loom::adg::BuiltinTargetPreset::Small, module));
  auto transform = take(system.createServiceTransform());
  auto clock = take(system.createHardwareDomain());
  auto rate = take(system.createServiceRate(
      clock, 1, 1, 4,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>)));

  auto initiateCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MemoryRead,
          loom::fabric::CanonicalServiceEndpointRole::Initiate, *addressed,
          rate));
  auto serveCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MemoryRead,
          loom::fabric::CanonicalServiceEndpointRole::Serve, *addressed, rate));
  auto initiateSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(initiateCapability)}));
  auto serveSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(serveCapability)}));
  auto initiate = take(system.addServiceEndpoint(transform, initiateSet));
  auto serve = take(system.addServiceEndpoint(transform, serveSet));
  auto initiateMemory = take(initiate.memory());
  auto serveMemory = take(serve.memory());

  auto boundary = take(system.addExternalBoundary());
  auto initiateMessage =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::IntegerType::get(&context, 32)})),
          rate));
  auto serveMessage =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Serve,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::IntegerType::get(&context, 32)})),
          rate));
  auto initiateMessageSet =
      take(loom::fabric::CanonicalServiceCapabilitySet::create(
          {std::move(initiateMessage)}));
  auto serveMessageSet =
      take(loom::fabric::CanonicalServiceCapabilitySet::create(
          {std::move(serveMessage)}));
  const auto bits32 = take(loom::adg::PortType::bits(32));
  auto messageSource =
      take(system.addServiceEndpoint(boundary, initiateMessageSet, bits32));
  auto messageSink =
      take(system.addServiceEndpoint(boundary, serveMessageSet, bits32));

  const auto baseTransport = baseSystem.transportResources().front();
  const auto *resourceContract = baseSystem.artifact().resourceContract(
      loom::fabric::FabricInventoryOwnerRef::of(baseTransport));
  require(resourceContract, "builtin transport has no resource contract");
  const auto bits128 = take(loom::adg::PortType::bits(128));
  auto transport = take(
      system.addTransportResource({{bits128}, {bits128}, *resourceContract}));
  auto input = take(transport.input(0));
  auto output = take(transport.output(0));

  if (llvm::Error error = system.attachServiceLegCarriers(
          initiateMemory, dataflow::semantics::ServiceKind::MemoryRead, 0,
          {output}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          initiateMemory, dataflow::semantics::ServiceKind::MemoryRead, 1,
          {input}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          serveMemory, dataflow::semantics::ServiceKind::MemoryRead, 0,
          {input}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.attachServiceLegCarriers(
          serveMemory, dataflow::semantics::ServiceKind::MemoryRead, 1,
          {output}))
    fail(llvm::toString(std::move(error)));

  if (llvm::Error error =
          transform.close({initiateMemory}, {serveMemory},
                          loom::fabric::AddressMaskXorTransform{64, 4095, 1}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = clock.close(
          {transform.domainMember(), initiate.domainMember(),
           serve.domainMember(), boundary.domainMember(),
           messageSource.domainMember(), messageSink.domainMember(),
           transport.domainMember()},
          take(loom::fabric::ClockDomainContractRecord::create(1'000, 0))))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(llvm::toString(std::move(error)));
  return take(std::move(design).finalize());
}

std::vector<loom::fabric::FabricTransportEndpointRef>
directMessageEndpoints(const loom::fabric::FabricSystemRootView &system,
                       bool source) {
  const auto direction =
      take(dataflow::semantics::getCanonicalServiceLegDirection(
          dataflow::semantics::ServiceKind::MessageTransfer, 0));
  const bool initiatorIsSource =
      direction == dataflow::semantics::ServiceLegDirection::InitiatorToServer;
  std::vector<loom::fabric::FabricTransportEndpointRef> endpoints;
  for (const auto endpoint : system.artifact().systemServiceEndpoints()) {
    const auto *capabilities = system.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() !=
            loom::fabric::CanonicalServiceEndpointPlane::Transport)
      continue;
    const auto capability =
        llvm::find_if(capabilities->capabilities(), [](const auto &candidate) {
          return candidate.kind() ==
                 dataflow::semantics::ServiceKind::MessageTransfer;
        });
    if (capability == capabilities->capabilities().end())
      continue;
    const bool endpointIsInitiator =
        capability->role() ==
        loom::fabric::CanonicalServiceEndpointRole::Initiate;
    if (source ? endpointIsInitiator != initiatorIsSource
               : endpointIsInitiator == initiatorIsSource)
      continue;
    endpoints.push_back(
        {loom::fabric::FabricTransportEndpointOwnerRef::of(endpoint), 0});
  }
  llvm::sort(endpoints, [](const auto &left, const auto &right) {
    return loom::fabric::canonicalFabricBytes(left) <
           loom::fabric::canonicalFabricBytes(right);
  });
  return endpoints;
}

const loom::pnr::SystemSearchServiceDomain &
memoryServiceDomain(const loom::pnr::SystemPnrSearchDomainView &view) {
  const auto found =
      llvm::find_if(view.serviceObligations(), [](const auto &domain) {
        const auto *operation =
            std::get_if<loom::mapping::OperationServiceObligationFamilyKey>(
                &domain.key);
        return operation &&
               std::holds_alternative<dataflow::LogicalMemoryRootOrViewRef>(
                   *operation);
      });
  require(found != view.serviceObligations().end(),
          "H omitted the addressed-memory service obligation");
  return *found;
}

const loom::pnr::SystemSearchServiceDomain &
fenceServiceDomain(const loom::pnr::SystemPnrSearchDomainView &view) {
  const auto found =
      llvm::find_if(view.serviceObligations(), [](const auto &domain) {
        const auto *operation =
            std::get_if<loom::mapping::OperationServiceObligationFamilyKey>(
                &domain.key);
        return operation &&
               std::holds_alternative<dataflow::FenceActorFamilyRef>(
                   *operation);
      });
  require(found != view.serviceObligations().end(),
          "H omitted the fence service obligation");
  return *found;
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  const llvm::ArrayRef<std::uint8_t> descriptor =
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes();
  require(llvm::StringRef(reinterpret_cast<const char *>(descriptor.data()),
                          descriptor.size()) ==
              "loom.system_pnr_search_domain.3.0",
          "H schema descriptor did not switch atomically to version 3.0");
  static constexpr llvm::StringLiteral priorDescriptor =
      "loom.system_pnr_search_domain.2.0";
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);

  auto dataflow = buildMemoryDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto dataflowView = take(dataflow.view());
  require(dataflowView.rootThreadLaunches().size() == 1,
          "fixture must contain one root launch");

  auto design = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto system =
      take(loom::fabric::requireSystemRoot(design.roots().front().view()));
  std::vector<dataflow::RootThreadLaunchRef> roots{
      dataflowView.rootThreadLaunches().front().ref};
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, system, roots, store));
  auto plan = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflowView, constraints.view().rootThreadLaunches()));
  const auto config = take(loom::pnr::projectResolvedSystemPnrConfigView(
      loom::defaultResolvedConfig()));
  auto domain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflowView, system, config, constraints, plan, {}, store));
  auto priorDigest = take(loom::pnr::computeSystemPnrSearchDomainDigest(
      llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(priorDescriptor.data()),
          priorDescriptor.size()),
      domain.canonicalViewBytes()));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          llvm::ArrayRef<std::uint8_t>(
              reinterpret_cast<const std::uint8_t *>(priorDescriptor.data()),
              priorDescriptor.size()),
          domain.canonicalViewBytes(), priorDigest, store),
      "schema descriptor is not exact version 3.0",
      "H 2.0 descriptor was accepted by the H 3.0 owner");

  auto obligations = take(loom::mapping::projectSystemServiceObligations(
      dataflowView, constraints.view().rootThreadLaunches()));
  require(domain.serviceObligations().size() == obligations.size(),
          "H service-domain closure differs from the Dataflow projection");
  for (auto &&[service, obligation] :
       llvm::zip_equal(domain.serviceObligations(), obligations)) {
    require(service.key == obligation.key,
            "H service obligations are not in canonical key order");
    require(service.targetCompatibility.empty() &&
                service.transferTerminalCompatibility.empty(),
            "operation service without a legal graph target acquired a row");
  }

  const auto &service = memoryServiceDomain(domain);
  require(service.targetCompatibility.empty() &&
              service.transferTerminalCompatibility.empty(),
          "memory service scanned endpoints outside its graph domain");

  const auto &fence = fenceServiceDomain(domain);
  require(fence.targetCompatibility.empty() &&
              fence.transferTerminalCompatibility.empty(),
          "fence service scanned endpoints outside its graph domain");

  constexpr llvm::StringLiteral oldDescriptor =
      "loom.system_pnr_search_domain.1.0";
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          {reinterpret_cast<const std::uint8_t *>(oldDescriptor.data()),
           oldDescriptor.size()},
          domain.canonicalViewBytes(), domain.digest(), store),
      "schema descriptor is not exact version 3.0",
      "strict H adoption accepted the retired version 1.0 descriptor");

  auto adopted = take(loom::pnr::adoptSystemPnrSearchDomain(
      loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
      domain.canonicalViewBytes(), domain.digest(), store));
  require(adopted.canonicalViewBytes() == domain.canonicalViewBytes() &&
              adopted.serviceObligations().size() ==
                  domain.serviceObligations().size(),
          "strict H adoption lost the service-domain projection");

  const FramedServiceSection framing =
      locateServiceSection(domain.canonicalViewBytes());
  require(framing.entries.size() == domain.serviceObligations().size(),
          "test service framing count differs from typed H");
  const std::size_t memoryServiceIndex =
      static_cast<std::size_t>(&service - domain.serviceObligations().data());
  const FramedServiceEntry &memoryServiceFraming =
      framing.entries[memoryServiceIndex];
  require(memoryServiceFraming.targetCount == 0 &&
              memoryServiceFraming.terminalCount == 0,
          "wire framing invented rows for an empty graph target domain");

  require(domain.serviceObligations().size() >= 2,
          "strict service-order fixture requires two obligations");
  auto duplicateServiceBytes =
      duplicateFramedEntry(domain.canonicalViewBytes(), framing.countOffset,
                           framing.count, framing.entries.front().bytes);
  auto duplicateServiceDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateServiceBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateServiceBytes, duplicateServiceDigest, store),
      "service-obligation domains are not strictly ordered",
      "strict H adoption accepted a duplicate service-obligation key");

  auto unorderedServiceBytes = swapAdjacentFramedEntries(
      domain.canonicalViewBytes(), framing.entries[0].bytes,
      framing.entries[1].bytes);
  auto unorderedServiceDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedServiceBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedServiceBytes, unorderedServiceDigest, store),
      "service-obligation domains are not strictly ordered",
      "strict H adoption accepted out-of-order service-obligation keys");

  auto missingServiceBytes =
      removeFramedEntry(domain.canonicalViewBytes(), framing.countOffset,
                        framing.count, framing.entries.front().bytes);
  auto missingServiceDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          missingServiceBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          missingServiceBytes, missingServiceDigest, store),
      "service-obligation closure differs from Dataflow",
      "strict H adoption accepted a missing service obligation");

  auto transformDesign =
      buildTransformFabric(store, design.roots().front(), context);
  auto transformSystem = take(
      loom::fabric::requireSystemRoot(transformDesign.roots().front().view()));
  auto transformConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflowView, transformSystem, roots, store));
  auto transformPlan =
      take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
          dataflowView, transformConstraints.view().rootThreadLaunches()));
  requireUnsupported(
      loom::pnr::projectSystemPnrSearchDomain(dataflowView, transformSystem,
                                              config, transformConstraints,
                                              transformPlan, {}, store),
      loom::pnr::UnsupportedSystemPnrSearchDomainReason::
          ServiceTransformProjectionUnavailable,
      "service-transform closure is not implemented",
      "H silently underapproximated a Fabric with service transforms");

  auto messageDataflow = buildMessageDataflow(context);
  take(dataflow::publishCanonicalDataflow(messageDataflow, store));
  auto messageView = take(messageDataflow.view());
  require(messageView.rootThreadLaunches().size() == 1,
          "message fixture must contain one root launch");
  std::vector<dataflow::RootThreadLaunchRef> messageRoots{
      messageView.rootThreadLaunches().front().ref};
  auto messageConstraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          messageView, transformSystem, messageRoots, store));
  auto messagePlan = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      messageView, messageConstraints.view().rootThreadLaunches()));
  auto messageDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      messageView, transformSystem, config, messageConstraints, messagePlan, {},
      store));
  require(!messageDomain.serviceObligations().empty() &&
              llvm::all_of(messageDomain.serviceObligations(),
                           [](const auto &service) {
                             return std::holds_alternative<
                                 loom::mapping::TransferObligationFamilyKey>(
                                 service.key);
                           }),
          "message-only Dataflow acquired an operation-service obligation");
  const auto expectedMessageSources =
      directMessageEndpoints(transformSystem, /*source=*/true);
  const auto expectedMessageSinks =
      directMessageEndpoints(transformSystem, /*source=*/false);
  require(!expectedMessageSources.empty() && !expectedMessageSinks.empty(),
          "message fixture has no direct transport-plane endpoints");
  std::vector<dataflow::CanonicalProducerTerminalView> messageProducers;
  for (const dataflow::RootThreadLaunchRef &root : messageRoots)
    if (llvm::Error error = messageView.forEachProducerTerminal(
            root, [&](const dataflow::CanonicalProducerTerminalView &producer) {
              messageProducers.push_back(producer);
              return llvm::Error::success();
            }))
      fail(llvm::toString(std::move(error)));
  auto messageObligations = take(loom::mapping::projectSystemServiceObligations(
      messageView, messageRoots));
  std::size_t directlyAdmittedObligations = 0;
  for (const auto &messageService : messageDomain.serviceObligations()) {
    const auto *producer =
        std::get_if<loom::mapping::TransferObligationFamilyKey>(
            &messageService.key);
    require(producer, "message-only H contains a non-transfer obligation");
    const auto producerView =
        llvm::find_if(messageProducers, [&](const auto &candidate) {
          return candidate.terminal == *producer;
        });
    require(producerView != messageProducers.end(),
            "MessageTransfer obligation has no exact producer view");
    const bool directlyAdmitted = producerView->payloadType.isInteger(32);
    directlyAdmittedObligations += directlyAdmitted;

    const auto obligation =
        llvm::find_if(messageObligations, [&](const auto &candidate) {
          return candidate.key == messageService.key;
        });
    require(obligation != messageObligations.end(),
            "H MessageTransfer obligation has no Dataflow projection");
    require(messageService.targetCompatibility.empty(),
            "MessageTransfer acquired an operation target row");
    std::size_t sourceCount = 0;
    std::size_t sinkCount = 0;
    for (const auto &terminal : messageService.transferTerminalCompatibility) {
      const auto *bound = std::get_if<loom::pnr::SystemMessageTerminalEndpoint>(
          &terminal.boundEndpoint);
      require(bound, "MessageTransfer acquired a memory endpoint row");
      const std::vector<loom::fabric::FabricTransportEndpointRef> expected =
          directlyAdmitted
              ? std::vector<
                    loom::fabric::FabricTransportEndpointRef>{bound->endpoint}
              : std::vector<loom::fabric::FabricTransportEndpointRef>{};
      require(terminal.compatibleTransportEndpoints == expected,
              "MessageTransfer row did not preserve its exact endpoint");
      if (std::holds_alternative<
              loom::mapping::SystemTransferSourceTerminalKey>(
              terminal.terminal)) {
        ++sourceCount;
        require(llvm::is_contained(expectedMessageSources, bound->endpoint),
                "MessageTransfer source row bound the wrong endpoint");
      } else {
        ++sinkCount;
        require(llvm::is_contained(expectedMessageSinks, bound->endpoint),
                "MessageTransfer sink row bound the wrong endpoint");
      }
    }
    require(sourceCount ==
                obligation->legs.size() * expectedMessageSources.size(),
            "MessageTransfer omitted or added a source terminal");
    require(sinkCount == obligation->legs.size() * obligation->sinks.size() *
                             expectedMessageSinks.size(),
            "MessageTransfer omitted or added a sink terminal");
  }
  require(directlyAdmittedObligations > 0,
          "message fixture did not exercise direct endpoint admission");

  const FramedServiceSection messageFraming =
      locateServiceSection(messageDomain.canonicalViewBytes());
  const auto framedMessage =
      llvm::find_if(messageFraming.entries, [](const auto &entry) {
        return entry.terminals.size() >= 2;
      });
  require(framedMessage != messageFraming.entries.end(),
          "terminal strict-order fixture requires two rows");
  auto duplicateTerminalBytes = duplicateFramedEntry(
      messageDomain.canonicalViewBytes(), framedMessage->terminalCountOffset,
      framedMessage->terminalCount, framedMessage->terminals.front());
  auto duplicateTerminalDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateTerminalBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          duplicateTerminalBytes, duplicateTerminalDigest, store),
      "transfer-terminal compatibility rows are not strictly ordered",
      "strict H adoption accepted a duplicate terminal row");

  auto unorderedTerminalBytes = swapAdjacentFramedEntries(
      messageDomain.canonicalViewBytes(), framedMessage->terminals[0],
      framedMessage->terminals[1]);
  auto unorderedTerminalDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedTerminalBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          unorderedTerminalBytes, unorderedTerminalDigest, store),
      "transfer-terminal compatibility rows are not strictly ordered",
      "strict H adoption accepted out-of-order terminal rows");

  auto missingTerminalBytes = removeFramedEntry(
      messageDomain.canonicalViewBytes(), framedMessage->terminalCountOffset,
      framedMessage->terminalCount, framedMessage->terminals.front());
  auto missingTerminalDigest =
      take(loom::pnr::computeSystemPnrSearchDomainDigest(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          missingTerminalBytes));
  requireFailureContains(
      loom::pnr::adoptSystemPnrSearchDomain(
          loom::pnr::systemPnrSearchDomainSchemaDescriptorBytes(),
          missingTerminalBytes, missingTerminalDigest, store),
      "transfer-terminal compatibility row closure differs",
      "strict H adoption accepted a missing terminal row");

  llvm::outs() << "System PnR service-domain anchors passed\n";
  return EXIT_SUCCESS;
}
